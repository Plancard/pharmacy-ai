"""트레이 라벨 자동 대조 (라벨링 기준 문서 '작업 절차 4')

수정이 끝난 라벨(remap_rf.py 결과)의 칸별 박스 수·클래스가 cells.csv의 actual과 맞는지 본다.
박스 중심점으로 칸을 정한다(prelabel·서비스와 같은 규칙). 안 맞는 사진만 Roboflow에서 다시 연다.

검사
  - g??_s??: 칸 12개마다 클래스별 개수 == actual
  - empty_*: 박스 0개
  - 라벨 파일이 없는 사진, 사진이 없는 라벨
  - 테스트 그룹(--test)은 맞더라도 '전수 확인' 표시

사용
  python cv/check_labels.py                       # data/tray/labels, images, cells.csv, photos.csv
  python cv/check_labels.py --test g01,g11,g15,g17,g18,g19
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel import TIMES, cell_of, load_cells, load_photos  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def fmt(c):
    return ";".join(f"{k}:{v}" for k, v in sorted(c.items())) or "(빈 칸)"


def main():
    ap = argparse.ArgumentParser(description="트레이 라벨 ↔ cells.csv 자동 대조")
    ap.add_argument("--tray", default="data/tray")
    ap.add_argument("--labels", default="data/tray/labels")
    ap.add_argument("--test", default="", help="테스트 그룹(쉼표로 구분) → 전수 확인 표시")
    ap.add_argument("--out", default="data/tray/check_report.csv")
    args = ap.parse_args()

    tray, lbl = Path(args.tray), Path(args.labels)
    cells = load_cells(tray / "cells.csv")
    photos = load_photos(tray / "photos.csv")
    test = {g.strip() for g in args.test.split(",") if g.strip()}

    imgs = {p.stem for p in (tray / "images").iterdir() if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem for p in lbl.glob("*.txt")}
    rows = []
    for s in sorted(labels - imgs):
        rows.append({"file": s, "cell": "-", "expected": "-", "labeled": "-", "note": "사진 없는 라벨"})

    ok = bad = 0
    for stem in sorted(imgs):
        if stem.startswith(("rot_", "sample_")):
            continue
        f = lbl / f"{stem}.txt"
        if not f.exists():
            rows.append({"file": stem, "cell": "-", "expected": "-", "labeled": "-", "note": "라벨 없음"})
            bad += 1
            continue
        boxes = [l.split() for l in f.read_text().splitlines() if l.strip()]
        if stem.startswith("empty_"):
            if boxes:
                rows.append({"file": stem, "cell": "-", "expected": "(빈 트레이)",
                             "labeled": f"박스 {len(boxes)}개", "note": "빈 트레이에 박스"})
                bad += 1
            else:
                ok += 1
            continue
        group = photos.get(f"{stem}.jpg")
        if group is None:
            rows.append({"file": stem, "cell": "-", "expected": "-", "labeled": "-", "note": "photos.csv에 없음"})
            bad += 1
            continue

        got = {}
        for b in boxes:
            day, t = cell_of(float(b[1]), float(b[2]), 1.0, 1.0)   # 정규화 좌표 그대로
            got.setdefault((day, t), Counter())[int(b[0])] += 1
        wrong = 0
        for day in (1, 2, 3):
            for t in TIMES:
                exp, have = cells[(group, day, t)], got.get((day, t), Counter())
                if +exp != +have:
                    wrong += 1
                    rows.append({"file": stem, "cell": f"{day}일 {t}", "expected": fmt(exp),
                                 "labeled": fmt(have), "note": "불일치"})
        if wrong:
            bad += 1
        else:
            ok += 1
            if group in test:
                rows.append({"file": stem, "cell": "-", "expected": "-", "labeled": "-",
                             "note": "테스트 그룹: 개수는 맞음, 박스 위치·크기 전수 확인"})

    with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["file", "cell", "expected", "labeled", "note"])
        w.writeheader()
        w.writerows(rows)

    print(f"일치 {ok}장, 다시 볼 사진 {bad}장 → {args.out}")
    for r in rows:
        if r["note"] != "테스트 그룹: 개수는 맞음, 박스 위치·크기 전수 확인":
            print(f"  {r['file']:12s} {r['cell']:12s} 기대 {r['expected']:14s} 라벨 {r['labeled']:14s} {r['note']}")


if __name__ == "__main__":
    main()
