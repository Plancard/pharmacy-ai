"""Roboflow에서 내보낸 YOLO 라벨의 클래스 번호를 우리 ID로 되돌리기

왜 필요한가
  Roboflow는 내보낼 때 '데이터셋에 있는 클래스만' '이름 알파벳순'으로 0부터 다시 번호를 매긴다.
  → 클래스 이름 앞의 ID 접두어('0_tylenol')로 우리 ID를 되찾는다. 접두어가 없으면 --ref의 이름으로 맞춘다.
  (Roboflow는 한글 이름을 '-'로 바꾸므로 클래스 이름은 영문 + ID 접두어로 쓴다)
  파일 이름의 Roboflow 꼬리(g01_s01_jpg.rf.xxxx)도 원래 이름(g01_s01)으로 바꾼다.

사용
  # 압축 푼 Roboflow 폴더(안에 data.yaml, train/labels ...) → data/tray/labels
  python cv/remap_rf.py --rf ~/Downloads/tray-pills.v1 --ref data/tray/prelabels/data.yaml --out data/tray/labels
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

ID_PREFIX = re.compile(r"^(\d+)_")
RF_NAME = re.compile(r"^(.*)_(jpg|jpeg|png)\.rf\.[0-9a-zA-Z]+$")


def load_names(path):
    """data.yaml의 names (리스트 또는 {id: name}) → {id: name}"""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = d["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def norm(s):
    return str(s).replace(" ", "").lower()


def main():
    ap = argparse.ArgumentParser(description="Roboflow YOLO 라벨 클래스 번호 되돌리기")
    ap.add_argument("--rf", required=True, help="압축 푼 Roboflow 내보내기 폴더")
    ap.add_argument("--ref", default="data/tray/prelabels/data.yaml", help="우리 ID 기준 data.yaml")
    ap.add_argument("--out", default="data/tray/labels", help="결과 라벨 폴더")
    ap.add_argument("--force", action="store_true", help="결과 폴더의 같은 이름 덮어쓰기")
    args = ap.parse_args()

    rf = Path(args.rf).expanduser()
    rf_names = load_names(rf / "data.yaml")
    ref = {norm(v): k for k, v in load_names(args.ref).items()} if Path(args.ref).exists() else {}

    mapping, missing = {}, []
    for i, n in rf_names.items():
        m = ID_PREFIX.match(str(n))
        if m:                                   # '0_tylenol' → 0
            mapping[i] = int(m.group(1))
        elif norm(n) in ref:
            mapping[i] = ref[norm(n)]
        else:
            missing.append(n)
    if missing:
        sys.exit(f"ID를 알 수 없는 클래스 이름: {missing} → Roboflow Classes & Tags에서 '0_tylenol'처럼 'ID_이름'으로 바꾸세요")
    if len(set(mapping.values())) != len(mapping):
        sys.exit(f"서로 다른 클래스가 같은 ID로 감: {mapping}")
    print("번호 변환 (Roboflow → 우리 ID):")
    for i in sorted(mapping):
        print(f"  {i} → {mapping[i]}  {rf_names[i]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(rf.glob("*/labels/*.txt"))
    if not files:
        sys.exit(f"{rf}/*/labels/*.txt 없음")
    stems = {}
    for f in files:
        m = RF_NAME.match(f.stem)
        stem = m.group(1) if m else f.stem
        if stem in stems:
            print(f"  [주의] 같은 사진이 두 번: {stem} (뒤의 것 사용)")
        stems[stem] = f
    clash = [s for s in stems if (out / f"{s}.txt").exists()]
    if clash and not args.force:
        sys.exit(f"{out}에 이미 있음 {len(clash)}개(예: {clash[:3]}). 덮어쓰려면 --force")
    n_box = 0
    for stem, f in sorted(stems.items()):
        dst = out / f"{stem}.txt"
        lines = []
        for line in f.read_text().splitlines():
            p = line.split()
            if not p:
                continue
            if len(p) != 5:
                sys.exit(f"{f.name}: 박스가 아닌 줄(폴리곤?) → Roboflow 내보내기 형식 확인: {line[:60]}")
            lines.append(" ".join([str(mapping[int(p[0])])] + p[1:]))
        dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines)
    print(f"{len(stems)}장, 박스 {n_box}개 → {out}")


if __name__ == "__main__":
    main()
