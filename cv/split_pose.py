"""트레이 꼭짓점(pose) 학습용 분할과 ultralytics yaml 생성.

검출 분할(split.py)과 같은 그룹 배정을 따른다. split.py를 먼저 돌려야 한다.
  - 그룹 사진·빈 트레이: data/splits/tray_{train,val,test}.txt에 들어간 곳 그대로
  - rot_ (180° 사진): test  — 11/13 rotated 판정 때 꼭짓점 성능도 같이 본다
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


def orig_stem(p):
    """g01_s01_jpg.rf.abc123 → g01_s01 (Roboflow가 붙인 꼬리를 뗀다)"""
    m = RF_NAME.match(p.stem)
    return m.group(1) if m else p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True, help="압축 푼 Roboflow tray-corners 내보내기 폴더")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

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
    no_label = []
    for img in sorted(rf.glob("*/images/*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        if not (img.parent.parent / "labels" / f"{img.stem}.txt").exists():
            no_label.append(img.name)
            continue
        name = orig_stem(img)
        if name in assign:
            s = assign[name]
        elif name.startswith("rot_"):
            s = "test"
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
    if no_label:
        print(f"[확인] 라벨 없는 이미지 {len(no_label)}장: {' '.join(no_label[:5])}")
    print(f"→ {splits / 'pose.yaml'}")


if __name__ == "__main__":
    main()
