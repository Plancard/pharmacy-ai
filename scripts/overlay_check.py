"""결과 오버레이 그림 눈 확인 (cv/pipeline.py, 11/23). val 트레이 사진마다 정답표의 계획으로 검수하고 그림을 남긴다.

    python scripts/overlay_check.py                       # data/splits/pose_val.txt 전부
    python scripts/overlay_check.py --limit 6 --errors    # 일부러 틀린 칸이 있는 그룹만, 6장

정답표(data/tray/cells.csv · photos.csv)의 planned를 A.4 계획으로 쓴다 (cv.tray_inspect 평가와 같은 방법).
test 그룹(g03·g06·g08·g10·g11·g20)과 rot 사진은 돌리지 않는다 (11/20 1회 원칙).
판정 수치를 재는 스크립트가 아니다 — 그림이 판정 표와 맞게 그려지는지 눈으로 보는 용도.
출력: <out>/<사진 이름>.overlay.jpg, 화면에 사진별 판정 요약
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_GROUPS = {"g03", "g06", "g08", "g10", "g11", "g20"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", default="data/splits/pose_val.txt")
    ap.add_argument("--tray", default="data/tray")
    ap.add_argument("--out", default="data/tray/overlay_check")
    ap.add_argument("--limit", type=int, default=0, help="이 장수만 (0 = 전부)")
    ap.add_argument("--errors", action="store_true", help="일부러 틀린 칸이 있는 그룹 사진만")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from cv import detect, pipeline, tray
    from cv.stub import TrayNotFound
    from cv.tray_inspect import group_of, load_truth, plan_from_counts

    truth = load_truth(Path(args.tray) / "cells.csv")
    with open(Path(args.tray) / "photos.csv", newline="", encoding="utf-8-sig") as f:
        photo_group = {Path(r["file"].strip()).stem: r["group"].strip() for r in csv.DictReader(f)}
    det_model, seg_model = YOLO(detect.DET_MODEL), YOLO(tray.SEG_MODEL)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    done = 0
    for p in map(Path, Path(args.list).read_text(encoding="utf-8").split()):
        group = group_of(p, photo_group, truth)
        if group is None or group in TEST_GROUPS or p.stem.startswith("rot_"):
            continue
        if args.errors and not any(c["error"] for c in truth[group].values()):
            continue
        plan = plan_from_counts({k: v["expected"] for k, v in truth[group].items()}, group)
        dst = out / f"{p.stem}.overlay.jpg"
        try:
            a3 = pipeline.run(p, plan, overlay_path=dst, model=det_model, seg_model=seg_model)
        except TrayNotFound as e:
            print(f"{p.name:40s} {group}  재촬영: {e}")
            continue
        s = a3["summary"]
        bad = ", ".join(f"{c['day']}일 {c['time']} {c['status']}" for c in a3["cells"] if c["status"] != "ok")
        wrong = ", ".join(f"{d}일 {t}" for (d, t), c in truth[group].items() if c["error"])
        print(f"{p.name:40s} {group}  일치 {s['ok']} 불일치 {s['mismatch']} 확인 {s['need_review']}"
              f"  | 판정: {bad or '-'}  | 정답표 틀린 칸: {wrong or '-'}"
              f"{'' if dst.exists() else '  (그림 없음)'}")
        done += 1
        if args.limit and done >= args.limit:
            break
    print(f"→ {out} ({done}장)")


if __name__ == "__main__":
    main()
