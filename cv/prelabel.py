"""트레이 사진 사전 라벨링 (라벨링 기준 문서 '작업 절차 3')

기준 모델(AI Hub 학습)의 박스 → 박스 중심점으로 칸 배정 → 그 칸의 cells.csv actual로 클래스
→ Roboflow에 올릴 YOLO txt. 사람은 Roboflow에서 고치기만 한다.

클래스 붙이는 규칙
  - actual 1종: 칸 안 박스 모두 그 클래스
  - actual 2종 이상: 모델 점수가 높은 박스부터, 모델이 actual 안의 클래스로 본 박스는 그 클래스,
    나머지는 남은 개수가 많은 클래스로 채운다 → report.csv에 '확인' 표시
  - actual이 빈 칸의 박스: 오검출로 보고 버린다(report.csv에 남김)
  - 박스 수가 actual 개수보다 적거나 많으면 report.csv에 남긴다(Roboflow에서 추가·삭제)

파일 규칙
  - g07_s03.jpg 같은 본촬영 사진: photos.csv의 group으로 cells.csv를 찾는다
  - empty_s01.jpg: 빈 txt만 만든다
  - rot_s01.jpg: 학습에 안 쓰므로 건너뛴다

사용
  # 몇 장만 먼저 보고 conf·imgsz 정하기
  python cv/prelabel.py --model runs/detect/base_v0/weights/best.pt --limit 5 --vis data/tray/prelabel_vis
  # 전체
  python cv/prelabel.py --model runs/detect/base_v0/weights/best.pt --vis data/tray/prelabel_vis
"""
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

TIMES = ["morning", "noon", "evening", "bedtime"]  # 행 0~3 (사진 위 → 아래)
ROWS, COLS = 4, 3                                  # 4행(복용 시점) × 3열(1~3일)
WARP_W, WARP_H = 960, 1280
IMG_EXTS = {".jpg", ".jpeg", ".png"}
RF_NAMES = {0: "0_tylenol", 1: "1_geborin", 2: "2_trispen", 3: "3_pronknon",
            4: "4_arpen", 5: "5_yobisin", 6: "6_tripline", 7: "7_mucodyne"}  # cv/classes.yaml ID와 같게


# ---------------------------------------------------------------- 칸 / 배치 기록
def cell_of(cx, cy, w, h):
    """중심점(픽셀) → (day, time). 경계선 위는 오른쪽·아래 칸, 이미지 밖은 가장 가까운 칸."""
    col = min(max(math.floor(cx * COLS / w), 0), COLS - 1)
    row = min(max(math.floor(cy * ROWS / h), 0), ROWS - 1)
    return col + 1, TIMES[row]


def parse_counts(s):
    """'0:1;6:1' → Counter({0: 1, 6: 1}), '' → Counter()"""
    out = Counter()
    for part in (s or "").replace(" ", "").split(";"):
        if not part:
            continue
        c, n = part.split(":")
        out[int(c)] += int(n)
    return out


def load_cells(path):
    """cells.csv → {(group, day, time): Counter(actual)}. 형식 오류는 모아서 한 번에 알린다."""
    cells, errors = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            try:
                g, day, t = r["group"].strip(), int(r["day"]), r["time"].strip()
                if day not in (1, 2, 3) or t not in TIMES:
                    raise ValueError(f"day={day}, time={t}")
                key = (g, day, t)
                if key in cells:
                    raise ValueError("같은 칸이 두 번 나옴")
                cells[key] = parse_counts(r["actual"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"  {path.name} {i}행: {e}")
    groups = Counter(g for g, _, _ in cells)
    for g, n in sorted(groups.items()):
        if n != ROWS * COLS:
            errors.append(f"  {g}: 칸 {n}개 (12개여야 함)")
    if errors:
        sys.exit("cells.csv 형식 오류\n" + "\n".join(errors))
    return cells


def load_photos(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["file"].strip(): r["group"].strip() for r in csv.DictReader(f)}


# ---------------------------------------------------------------- 클래스 배정
def assign(boxes, actual):
    """boxes: [(conf, pred_cls)], actual: Counter → (박스별 클래스 또는 None, 메모 목록)"""
    if not actual:
        return [None] * len(boxes), ([f"빈 칸인데 박스 {len(boxes)}개 → 버림"] if boxes else [])

    left = Counter(actual)
    out = [None] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][0])

    for i in order:                      # 1) 모델 예측이 actual 안에 있으면 그대로
        c = boxes[i][1]
        if left[c] > 0:
            out[i] = c
            left[c] -= 1
    guessed = 0
    for i in order:                      # 2) 나머지는 남은 개수가 많은 클래스부터
        if out[i] is not None:
            continue
        rem = +left
        if rem:
            c = min(rem, key=lambda k: (-rem[k], k))
            left[c] -= 1
            guessed += 1
        else:                            # 초과 박스: 가장 많은 클래스로 두고 확인
            c = min(actual, key=lambda k: (-actual[k], k))
        out[i] = c

    notes = []
    need, got = sum(actual.values()), len(boxes)
    if got < need:
        notes.append(f"박스 부족 {need - got}개 → 추가")
    elif got > need:
        notes.append(f"박스 초과 {got - need}개 → 확인")
    if len(actual) > 1 and guessed:
        notes.append(f"{len(actual)}종 칸, 모델과 다르게 배정 {guessed}개 → 확인")
    return out, notes


# ---------------------------------------------------------------- 박스 정리 (모델이 트레이 사진에 서툴러서)
def _inter(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def clean_boxes(dets, iou=0.5, contain=0.85, merged_ratio=1.5):
    """dets: [(cx, cy, w, h, conf, cls)] → (남길 박스, 버린 박스)
    1) 큰 박스가 작은 박스를 거의 통째로 품고(contain) 면적이 merged_ratio배 이상 → 알약 2개를 묶은 박스로 보고 큰 쪽 버림
    2) 클래스 무시하고 IoU > iou 인 중복 → 점수 낮은 쪽 버림 (같은 알약에 0번·7번 박스가 겹치는 경우)"""
    area = [d[2] * d[3] for d in dets]
    drop = set()
    for i in range(len(dets)):
        for j in range(len(dets)):
            if i != j and area[i] >= merged_ratio * area[j] and _inter(dets[i], dets[j]) >= contain * area[j]:
                drop.add(i)
    rest = sorted((i for i in range(len(dets)) if i not in drop), key=lambda i: -dets[i][4])
    keep = []
    for i in rest:
        if all(_inter(dets[i], dets[k]) / (area[i] + area[k] - _inter(dets[i], dets[k])) <= iou for k in keep):
            keep.append(i)
        else:
            drop.add(i)
    return [dets[i] for i in sorted(keep)], [dets[i] for i in sorted(drop)]


# ---------------------------------------------------------------- 추론
def make_predictor(model_path, conf, imgsz):
    from ultralytics import YOLO

    model = YOLO(model_path)

    def predict(img_path):
        r = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        h, w = r.orig_shape
        xywh = r.boxes.xywh.cpu().numpy().tolist()
        cf = r.boxes.conf.cpu().numpy().tolist()
        cl = r.boxes.cls.cpu().numpy().astype(int).tolist()
        return (w, h), [(*b, c, k) for b, c, k in zip(xywh, cf, cl)]

    return predict, dict(model.names)


def draw(img_path, w, h, labeled, dropped, out_path):
    import cv2

    img = cv2.imread(str(img_path))
    for c in range(1, COLS):
        x = round(w * c / COLS)
        cv2.line(img, (x, 0), (x, h), (160, 160, 160), 1)
    for r in range(1, ROWS):
        y = round(h * r / ROWS)
        cv2.line(img, (0, y), (w, y), (160, 160, 160), 1)
    palette = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 200, 255),
               (255, 0, 255), (255, 255, 0), (0, 0, 0), (128, 0, 255)]
    for cx, cy, bw, bh, conf, cls in labeled:
        p1, p2 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
        col = palette[cls % len(palette)]
        cv2.rectangle(img, p1, p2, col, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for cx, cy, bw, bh, *_ in dropped:  # 버린 박스는 회색 점선 대신 얇은 회색
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)),
                      (int(cx + bw / 2), int(cy + bh / 2)), (128, 128, 128), 1)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------- 본체
def run(args, predict, names):
    tray = Path(args.tray)
    img_dir, out_dir = tray / "images", Path(args.out)
    cells = load_cells(tray / "cells.csv")
    photos = load_photos(tray / "photos.csv")

    if out_dir.exists() and any(out_dir.glob("*.txt")) and not args.force:
        sys.exit(f"{out_dir}에 txt가 이미 있음. 덮어쓰려면 --force")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis) if args.vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]

    report, stats = [], Counter()
    for p in imgs:
        name = p.name
        if name.startswith("rot_"):
            stats["회전(건너뜀)"] += 1
            continue
        if name.startswith("empty_"):
            (out_dir / f"{p.stem}.txt").write_text("")
            stats["빈 트레이"] += 1
            continue
        group = photos.get(name)
        if group is None:
            print(f"  [건너뜀] photos.csv에 없음: {name}")
            stats["photos.csv 없음"] += 1
            continue
        if (group, 1, "morning") not in cells:
            print(f"  [건너뜀] cells.csv에 그룹 없음: {name} ({group})")
            stats["cells.csv 없음"] += 1
            continue

        (w, h), dets = predict(p)
        if (w, h) != (WARP_W, WARP_H):
            print(f"  [주의] {name} 크기 {w}×{h} (보정본은 {WARP_W}×{WARP_H})")

        pre_dropped = []
        if not args.no_clean:
            dets, pre_dropped = clean_boxes(dets, iou=args.iou)
        by_cell = {}
        for d in dets:
            by_cell.setdefault(cell_of(d[0], d[1], w, h), []).append(d)

        labeled, dropped = [], [(*d[:5], None) for d in pre_dropped]
        stats["정리로 버린 박스"] += len(pre_dropped)
        for day in (1, 2, 3):
            for t in TIMES:
                actual = cells[(group, day, t)]
                ds = by_cell.get((day, t), [])
                need = sum(actual.values())
                if actual and not args.keep_extra and len(ds) > need:  # 칸 개수보다 많으면 점수 높은 것만
                    ds = sorted(ds, key=lambda d: -d[4])
                    dropped += [(*d[:5], None) for d in ds[need:]]
                    report.append({"file": name, "group": group, "day": day, "time": t,
                                   "actual": ";".join(f"{k}:{v}" for k, v in sorted(actual.items())),
                                   "boxes": len(ds), "note": f"초과 {len(ds) - need}개 버림(점수 낮은 것) → 확인"})
                    ds = ds[:need]
                classes, notes = assign([(d[4], d[5]) for d in ds], actual)
                for d, c in zip(ds, classes):
                    (labeled if c is not None else dropped).append((*d[:5], c))
                for n in notes:
                    report.append({"file": name, "group": group, "day": day, "time": t,
                                   "actual": ";".join(f"{k}:{v}" for k, v in sorted(actual.items())),
                                   "boxes": len(ds), "note": n})

        lines = [f"{c} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                 for cx, cy, bw, bh, _, c in labeled]
        (out_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if vis_dir:
            draw(p, w, h, labeled, dropped, vis_dir / f"{p.stem}.jpg")
        stats["사진"] += 1
        stats["박스"] += len(labeled)
        stats["버린 박스"] += len(dropped)

    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "group", "day", "time", "actual", "boxes", "note"])
        wr.writeheader()
        wr.writerows(report)

    # Roboflow YOLO 업로드용 클래스 목록. Roboflow는 한글 이름을 '-'로 바꿔 버리므로 영문 + ID 접두어
    # (내보낼 때 번호가 알파벳순으로 바뀌어도 remap_rf.py가 접두어로 ID를 되찾는다)
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(f"nc: {len(names)}\nnames:\n")
        for k in sorted(names):
            f.write(f"  {k}: {RF_NAMES.get(k, f'{k}_class')}\n")

    print("결과:", ", ".join(f"{k} {v}" for k, v in stats.items()))
    bad = len({r["file"] for r in report})
    print(f"확인할 칸 {len(report)}개 (사진 {bad}장) → {out_dir / 'report.csv'}")
    print(f"라벨 → {out_dir}" + (f", 시각화 → {vis_dir}" if vis_dir else ""))


def main():
    ap = argparse.ArgumentParser(description="트레이 사진 사전 라벨링")
    ap.add_argument("--model", required=True, help="기준 모델 가중치(best.pt)")
    ap.add_argument("--tray", default="data/tray", help="images/, cells.csv, photos.csv가 있는 폴더")
    ap.add_argument("--out", default="data/tray/prelabels", help="사전 라벨 txt 폴더(최종 labels/와 분리)")
    ap.add_argument("--conf", type=float, default=0.2, help="박스 신뢰도 하한(트레이는 도메인이 달라 낮게)")
    ap.add_argument("--imgsz", type=int, default=960, help="추론 크기")
    ap.add_argument("--vis", help="박스·격자를 그린 확인용 이미지 폴더")
    ap.add_argument("--limit", type=int, help="앞에서 N장만")
    ap.add_argument("--force", action="store_true", help="기존 txt 덮어쓰기")
    ap.add_argument("--iou", type=float, default=0.5, help="클래스 무시 중복 제거 IoU 기준")
    ap.add_argument("--no-clean", action="store_true", help="묶음 박스·중복 박스 정리 끄기")
    ap.add_argument("--keep-extra", action="store_true", help="칸 개수보다 많은 박스도 남기기(예전 동작)")
    args = ap.parse_args()

    predict, names = make_predictor(args.model, args.conf, args.imgsz)
    run(args, predict, names)


if __name__ == "__main__":
    main()
