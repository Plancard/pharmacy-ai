"""꼭짓점(pose) 라벨 → 트레이 영역 분할(seg) 라벨과 ultralytics yaml 생성.

pose 모델이 꼭짓점을 정밀하게 못 찍는 문제(11/16 확인: 예측 사각형이 일정하게 돌아가고 커지거나 작아짐,
학습 사진에서도 칸 경계 이동 중앙 ~140px)에 대한 대안 A.
새 라벨링 없이 Roboflow 꼭짓점 4점을 그대로 다각형(트레이 윗면 테두리)으로 쓴다.

분할은 split_pose.py가 만든 pose_{train,val,test}.txt를 그대로 따른다 (먼저 split.py → split_pose.py).

출력
  data/tray/seg/images/<이름>.jpg   Roboflow 내보내기 이미지 복사본 (ultralytics가 images→labels로 라벨을 찾기 때문)
  data/tray/seg/labels/<이름>.txt   "0 x1 y1 x2 y2 x3 y3 x4 y4" (비율, tl·tr·br·bl)
  data/splits/seg_{train,val,test}.txt, data/splits/seg.yaml

사용 예
  python cv/pose2seg.py
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warp import read_kpt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    data = Path(args.data)
    splits = data / "splits"
    seg = data / "tray" / "seg"
    img_dir, lbl_dir = seg / "images", seg / "labels"
    if seg.exists():
        shutil.rmtree(seg)          # 이전 결과가 남아 섞이지 않게 매번 새로 만든다
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    counts, bad = {}, []
    for s in ("train", "val", "test"):
        src_list = splits / f"pose_{s}.txt"
        if not src_list.exists():
            raise SystemExit(f"{src_list} 없음 → 먼저 python cv/split_pose.py --rf <내보내기 폴더>")
        out = []
        for line in src_list.read_text(encoding="utf-8").split():
            ip = Path(line)
            pts, _, n = read_kpt(ip.parent.parent / "labels" / f"{ip.stem}.txt")
            if pts is None:
                bad.append(ip.name)
                continue
            dst = img_dir / ip.name
            shutil.copy2(ip, dst)
            xy = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
            (lbl_dir / f"{ip.stem}.txt").write_text(f"0 {xy}\n", encoding="utf-8")
            out.append(dst.resolve())
        (splits / f"seg_{s}.txt").write_text("".join(f"{p}\n" for p in sorted(out)), encoding="utf-8")
        counts[s] = len(out)

    cfg = {
        "path": str(data.resolve()),
        "train": "splits/seg_train.txt",
        "val": "splits/seg_val.txt",
        "test": "splits/seg_test.txt",
        "names": {0: "tray"},
    }
    (splits / "seg.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print("  ".join(f"{s} {n}장" for s, n in counts.items()))
    if bad:
        print(f"[확인] 꼭짓점 라벨 없는 이미지 {len(bad)}장: {' '.join(bad[:5])}")
    print(f"→ {splits / 'seg.yaml'}")


if __name__ == "__main__":
    main()
