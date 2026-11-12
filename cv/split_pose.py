"""트레이 꼭짓점(pose) 학습용 분할과 ultralytics yaml 생성.

검출 분할(split.py)과 같은 그룹 배정을 따른다. split.py를 먼저 돌려야 한다.
  - 그룹 사진·빈 트레이: data/splits/tray_{train,val,test}.txt에 들어간 곳 그대로
  - rot_ (180° 사진): test  — 11/13 rotated 판정 때 꼭짓점 성능도 같이 본다
  - pose_ (11/12 꼭짓점 추가 촬영 32장): --pose-val에 적은 촬영 조건은 val, 나머지는 train.
    pose_s01~16(약 있음)과 s17~32(빈 트레이)는 같은 조건을 두 번 찍은 것이라 짝(sNN, sNN+16)을 같이 보낸다.
    기본 val은 회전이 큰 4조건(s05 −20°, s07 −30°, s08 +30° 멀리, s12 +25°) × 2 = 8장.
    기존 val에는 큰 회전 사진이 없어서, 회전에 약한 문제(pose_v1)가 고쳐졌는지 보려면 val에 회전 사진이 필요하다
  - cv08_ (REQ-CV-08 재촬영 요청 확인용, 라벨 없음): 어디에도 넣지 않는다
  - sample_ 등 나머지: train

입력
  Roboflow tray-corners YOLOv8 내보내기 폴더 (*/images/*.jpg + */labels/*.txt)
  파일 이름은 g01_s01_jpg.rf.<hash>.jpg 형태 → 앞부분으로 원래 이름을 찾는다

출력 (data/splits/)
  pose_train.txt / pose_val.txt / pose_test.txt
  pose.yaml   kpt_shape [4, 3], flip_idx [1, 0, 3, 2] (점 순서 tl·tr·br·bl, 좌우 반전 시 tl↔tr·br↔bl)

사용 예
  python cv/split_pose.py --rf data/tray/rf/corners_v2/<내보내기 폴더>
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".png", ".jpg", ".jpeg")
RF_NAME = re.compile(r"^(.+?)_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)
POSE_PAIR = 16                       # pose_sNN(약 있음)과 pose_s(NN+16)(빈 트레이)은 같은 촬영 조건
POSE_VAL_DEFAULT = "5,7,8,12"        # 회전이 큰 조건 (촬영 배치표 1112 꼭짓점 표의 번호)
POSE_NAME = re.compile(r"^pose_s(\d+)$")


def orig_stem(p):
    """g01_s01_jpg.rf.abc123 → g01_s01 (Roboflow가 붙인 꼬리를 뗀다)"""
    m = RF_NAME.match(p.stem)
    return m.group(1) if m else p.stem


def pose_split(name, val_conditions):
    """pose_s05 → val 또는 train. 짝(s05·s21)은 같은 곳으로 간다."""
    n = int(POSE_NAME.match(name).group(1))
    cond = (n - 1) % POSE_PAIR + 1
    return "val" if cond in val_conditions else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True, help="압축 푼 Roboflow tray-corners 내보내기 폴더")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--pose-val", default=POSE_VAL_DEFAULT,
                    help="pose_ 사진 중 val로 보낼 촬영 조건 번호(1~16, 쉼표). 빈 값이면 전부 train")
    args = ap.parse_args()
    val_conditions = {int(x) for x in args.pose_val.split(",") if x.strip()}
    if not val_conditions <= set(range(1, POSE_PAIR + 1)):
        raise SystemExit(f"--pose-val은 1~{POSE_PAIR} 사이 번호: {args.pose_val}")

    data, rf = Path(args.data), Path(args.rf)
    splits = data / "splits"

    # 검출 분할에서 원래 이름별 배정 읽기
    assign = {}
    for s in ("train", "val", "test"):
        f = splits / f"tray_{s}.txt"
        if not f.exists():
            raise SystemExit(f"{f} 없음 → 먼저 python cv/split.py")
        for line in f.read_text(encoding="utf-8").split():
            assign[Path(line).stem] = s

    lists = {"train": [], "val": [], "test": []}
    no_label, skipped = [], []
    for img in sorted(rf.glob("*/images/*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        name = orig_stem(img)
        if name.startswith("cv08_"):
            skipped.append(name)
            continue
        if not (img.parent.parent / "labels" / f"{img.stem}.txt").exists():
            no_label.append(img.name)
            continue
        if name in assign:
            s = assign[name]
        elif name.startswith("rot_"):
            s = "test"
        elif POSE_NAME.match(name):
            s = pose_split(name, val_conditions)
        else:
            s = "train"
        lists[s].append(img.resolve())

    for s, items in lists.items():
        (splits / f"pose_{s}.txt").write_text("".join(f"{p}\n" for p in sorted(items)), encoding="utf-8")

    cfg = {
        "path": str(data.resolve()),
        "train": "splits/pose_train.txt",
        "val": "splits/pose_val.txt",
        "test": "splits/pose_test.txt",
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
        "names": {0: "tray"},
    }
    (splits / "pose.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    for s, items in lists.items():
        kinds = Counter(orig_stem(p).split("_")[0] if not orig_stem(p).startswith("g") else "group"
                        for p in items)
        detail = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        print(f"{s:5s} {len(items):3d}장 ({detail})")
    if skipped:
        print(f"[제외] REQ-CV-08 확인용 {len(skipped)}장: {' '.join(sorted(skipped))}")
    if no_label:
        print(f"[확인] 라벨 없는 이미지 {len(no_label)}장: {' '.join(no_label[:5])}")
    print(f"→ {splits / 'pose.yaml'}")


if __name__ == "__main__":
    main()
