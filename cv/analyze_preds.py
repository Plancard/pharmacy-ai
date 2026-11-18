"""예측 결과 분석: 검출 오탐·누락을 칸 단위로, 꼭짓점 오차를 보정 이미지 픽셀로 (11/12 v1 분석용)

det  — 검출 예측(yolo predict save_txt save_conf)을 정답 라벨과 대조한다
  예측 박스마다 종류를 붙인다 (박스 겹침 IoU 기준, 신뢰도 높은 박스부터)
    ok      정답 알약과 겹치고(IoU ≥ 0.5) 클래스도 맞음
    class   정답 알약과 겹치지만 클래스가 다름 (혼동)
    dup     이미 다른 박스가 잡은 알약에 또 겹친 박스 (같은 알약 중복)
    offset  알약과 조금만 겹침 (0.1 ≤ IoU < 0.5) — 알약 일부, 벽에 비친 상 등
    bg      알약과 거의 안 겹침 (IoU < 0.1) — 빈 곳, 벽 반사 등
  정답 중 아무 박스도 못 받은 것은 miss.
  출력
    표 1: 종류 × 신뢰도 구간
    표 2: 신뢰도 기준선별 칸 개수 일치율 (서비스 검수와 같은 기준: 박스 중심점 → 칸, 칸마다 클래스별 개수 비교)
          기본(클래스별 NMS 결과 그대로)과 '클래스 무관 중복 제거'를 나란히
    <out>/det_<name>_errors.csv  오류 하나당 한 줄 (사진, 칸, 종류, 정답·예측 클래스, 신뢰도, IoU)
    <out>/vis_<name>/<사진>.jpg   오류가 있는 사진: 칸 격자 + 정답(초록) + 오류 박스(빨강 ≥ 0.25, 주황 < 0.25) + miss(노랑)

pose — 꼭짓점 예측을 Roboflow 꼭짓점 라벨과 대조한다
  꼭짓점 오차를 '보정 후 960×1280 이미지'의 픽셀로 바꿔서 본다 (칸 하나 320px)
    corner_px  모서리 4개 오차(보정 이미지 기준)의 최대값
    grid_px    예측 꼭짓점으로 보정했을 때 칸 경계선(격자점 20개)이 정답 대비 움직인 최대 거리
  출력: 요약 + <out>/pose_<name>.csv

사용 (학원 PC, 저장소 루트)
  python cv/analyze_preds.py det --name val   --pred models/pred_val/labels   --list data/splits/tray_val.txt
  python cv/analyze_preds.py det --name train --pred models/pred_train/labels --list data/splits/tray_train.txt
  python cv/analyze_preds.py pose --name val --pred models/pose_pred_val/labels --list data/splits/pose_val.txt
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel import TIMES, cell_of  # noqa: E402
from warp import OUT_H, OUT_W, check_corners, read_kpt  # noqa: E402

SHORT = ["0_tylenol", "1_geborin", "2_trispen", "3_pronknon", "4_arpen", "5_yobisin", "6_tripline", "7_mucodyne"]
BINS = [(0.10, 0.25), (0.25, 0.50), (0.50, 0.70), (0.70, 1.01)]
KINDS = ["ok", "class", "dup", "offset", "bg"]


# ---------------------------------------------------------------- 공통
def read_list(path):
    return [Path(l.strip()) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def name(c):
    return SHORT[c] if 0 <= c < len(SHORT) else str(c)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_xyxy(cx, cy, w, h, W=OUT_W, H=OUT_H):
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def center_cell(box):
    return cell_of((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, OUT_W, OUT_H)


# ---------------------------------------------------------------- det
def read_gt(path):
    out = []
    if path.exists():
        for l in path.read_text().splitlines():
            v = l.split()
            if len(v) >= 5:
                out.append((int(v[0]), to_xyxy(*map(float, v[1:5]))))
    return out


def read_pred(path):
    out = []
    if path.exists():
        for l in path.read_text().splitlines():
            v = l.split()
            if len(v) >= 6:
                out.append((int(v[0]), to_xyxy(*map(float, v[1:5])), float(v[5])))
            elif len(v) == 5:
                raise SystemExit(f"{path.name}: 신뢰도 값이 없음 → predict에 save_conf=True를 주고 다시 실행")
    return sorted(out, key=lambda p: -p[2])


def classify(gts, preds):
    """→ (예측별 [(kind, gt_idx, iou)], 정답별 matched 여부)"""
    taken = [False] * len(gts)
    res = []
    for pc, pb, _ in preds:
        best, bi = 0.0, -1
        for i, (_, gb) in enumerate(gts):
            v = iou(pb, gb)
            if v > best:
                best, bi = v, i
        if best >= 0.5:
            if not taken[bi]:
                taken[bi] = True
                res.append(("ok" if gts[bi][0] == pc else "class", bi, best))
            else:
                res.append(("dup", bi, best))
        elif best >= 0.1:
            res.append(("offset", bi, best))
        else:
            res.append(("bg", bi, best))
    return res, taken


def agnostic_nms(preds, thr):
    keep = []
    for p in preds:   # 이미 신뢰도 내림차순
        if all(iou(p[1], k[1]) < thr for k in keep):
            keep.append(p)
    return keep


def cell_counts(items):
    """items: (cls, box, ...) → {(day, time): Counter(cls)}"""
    out = defaultdict(Counter)
    for it in items:
        out[center_cell(it[1])][it[0]] += 1
    return out


def sweep(data, nms_iou):
    """신뢰도 기준선별 칸 일치율. data: [(stem, gts, preds)]"""
    rows = []
    cells_all = [(d, t) for d in (1, 2, 3) for t in TIMES]
    for t100 in range(10, 95, 5):
        t = t100 / 100
        res = {}
        for mode in ("base", "agn"):
            ok_cells = ok_imgs = extra = missing = 0
            for _, gts, preds in data:
                ps = [p for p in preds if p[2] >= t]
                if mode == "agn":
                    ps = agnostic_nms(ps, nms_iou)
                g, p = cell_counts(gts), cell_counts(ps)
                img_ok = True
                for c in cells_all:
                    if g.get(c, Counter()) == p.get(c, Counter()):
                        ok_cells += 1
                    else:
                        img_ok = False
                        diff_extra = p.get(c, Counter()) - g.get(c, Counter())
                        diff_miss = g.get(c, Counter()) - p.get(c, Counter())
                        extra += sum(diff_extra.values())
                        missing += sum(diff_miss.values())
                ok_imgs += img_ok
            res[mode] = (ok_cells, ok_imgs, extra, missing)
        rows.append((t, res))
    return rows


def draw(img_path, out_path, gts, preds, kinds, taken):
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return
    if img.shape[1] != OUT_W or img.shape[0] != OUT_H:
        img = cv2.resize(img, (OUT_W, OUT_H))
    for x in range(0, OUT_W + 1, OUT_W // 3):
        cv2.line(img, (x, 0), (x, OUT_H), (160, 160, 160), 1)
    for y in range(0, OUT_H + 1, OUT_H // 4):
        cv2.line(img, (0, y), (OUT_W, y), (160, 160, 160), 1)
    for c, b in gts:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 1)
    for i, (c, b) in enumerate(gts):
        if not taken[i]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 230, 255), 3)
            cv2.putText(img, f"miss {name(c)}", (int(b[0]), max(int(b[1]) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
    for (c, b, conf), (k, _, _) in zip(preds, kinds):
        if k == "ok":
            continue
        col = (0, 0, 255) if conf >= 0.25 else (0, 140, 255)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        cv2.putText(img, f"{k} {name(c)} {conf:.2f}", (int(b[0]), min(int(b[3]) + 14, OUT_H - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def run_det(args):
    imgs = read_list(args.list)
    pred_dir, lbl_dir, out = Path(args.pred), Path(args.labels), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = defaultdict(Counter)        # kind → Counter(bin)
    by_cls = defaultdict(Counter)       # kind → Counter(pred cls)  (신뢰도 ≥ 0.25)
    confuse = Counter()                 # (gt, pred) 혼동 (≥ 0.25)
    n_gt = n_miss = 0
    data, err_rows = [], []
    for ip in imgs:
        stem = ip.stem
        gts = read_gt(lbl_dir / f"{stem}.txt")
        preds = read_pred(pred_dir / f"{stem}.txt")
        if not (lbl_dir / f"{stem}.txt").exists() and not stem.startswith("empty_"):
            print(f"  [확인] 정답 라벨 없음: {stem}")
        kinds, taken = classify(gts, preds)
        data.append((stem, gts, preds))
        n_gt += len(gts)
        for (pc, pb, conf), (k, gi, v) in zip(preds, kinds):
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= conf < hi) if conf >= BINS[0][0] else None
            if b is not None:
                table[k][b] += 1
            if conf >= 0.25 and k != "ok":
                by_cls[k][pc] += 1
                if k == "class":
                    confuse[(gts[gi][0], pc)] += 1
            if k != "ok":
                day, t = center_cell(pb)
                err_rows.append({"image": stem, "day": day, "time": t, "kind": k,
                                 "gt": name(gts[gi][0]) if gi >= 0 and k in ("class", "dup", "offset") else "",
                                 "pred": name(pc), "conf": f"{conf:.3f}", "iou": f"{v:.2f}"})
        for i, ok in enumerate(taken):
            if not ok:
                n_miss += 1
                day, t = center_cell(gts[i][1])
                err_rows.append({"image": stem, "day": day, "time": t, "kind": "miss", "gt": name(gts[i][0]),
                                 "pred": "", "conf": "", "iou": ""})
        if any(k != "ok" for k, _, _ in kinds) or not all(taken):
            draw(ip, out / f"vis_{args.name}" / f"{stem}.jpg", gts, preds, kinds, taken)

    print(f"\n== 검출 분석 [{args.name}] 사진 {len(imgs)}장, 정답 알약 {n_gt}개 ==")
    print("\n[표 1] 예측 박스 종류 × 신뢰도 구간")
    head = "종류      " + "".join(f"{lo:.2f}~{min(hi, 1):.2f}".rjust(12) for lo, hi in BINS) + "   합계"
    print(head)
    for k in KINDS:
        row = [table[k][i] for i in range(len(BINS))]
        print(f"{k:<10}" + "".join(f"{x:>12}" for x in row) + f"{sum(row):>7}")
    print(f"miss(신뢰도 0.1 이상 박스를 하나도 못 받은 알약): {n_miss}개")

    if confuse:
        print("\n혼동(신뢰도 ≥ 0.25, 정답 → 예측): " + ", ".join(f"{name(g)}→{name(p)} {n}" for (g, p), n in confuse.most_common()))
    for k in ("dup", "offset", "bg"):
        if by_cls[k]:
            print(f"{k}(≥ 0.25) 예측 클래스: " + ", ".join(f"{name(c)} {n}" for c, n in by_cls[k].most_common()))

    n_cells = len(imgs) * 12
    print(f"\n[표 2] 신뢰도 기준선별 칸 개수 일치 (칸 {n_cells}개, 사진 {len(imgs)}장)  · 중복 제거 IoU {args.nms_iou}")
    print("기준선 |      기본: 맞은 칸  사진  초과  부족 | 클래스 무관 중복 제거: 맞은 칸  사진  초과  부족")
    for t, res in sweep(data, args.nms_iou):
        b, a = res["base"], res["agn"]
        print(f"  {t:.2f} | {b[0]:>4}/{n_cells} {b[0] / n_cells:6.1%} {b[1]:>4} {b[2]:>5} {b[3]:>5} |"
              f"            {a[0]:>4}/{n_cells} {a[0] / n_cells:6.1%} {a[1]:>4} {a[2]:>5} {a[3]:>5}")

    csv_path = out / f"det_{args.name}_errors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "day", "time", "kind", "gt", "pred", "conf", "iou"])
        w.writeheader()
        w.writerows(err_rows)
    print(f"\n→ {csv_path} (오류 {len(err_rows)}줄), 그림 {out / f'vis_{args.name}'}/")


# ---------------------------------------------------------------- pose
def read_pose_pred(path):
    """→ (pts 비율 (4,2), 박스 신뢰도). 여러 줄이면 신뢰도가 가장 높은 줄."""
    best = None
    for l in Path(path).read_text().splitlines():
        v = [float(x) for x in l.split()[1:]]
        if len(v) == 17:        # 박스 4 + (x, y, conf)×4 + 박스 신뢰도
            pts, conf = np.array(v[4:16]).reshape(4, 3)[:, :2], v[16]
        elif len(v) == 13:      # 박스 4 + (x, y)×4 + 박스 신뢰도
            pts, conf = np.array(v[4:12]).reshape(4, 2), v[12]
        elif len(v) in (16, 12):
            pts, conf = np.array(v[4:]).reshape(4, -1)[:, :2], 1.0
        else:
            raise SystemExit(f"{Path(path).name}: 값 {len(v)}개 — 예상 밖의 pose txt 형식")
        if best is None or conf > best[1]:
            best = (pts, conf)
    return best


def run_pose(args):
    import cv2
    imgs = read_list(args.list)
    pred_dir, out = Path(args.pred), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = np.float32([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]])
    grid = np.float32([[x, y] for x in np.linspace(0, OUT_W, 4) for y in np.linspace(0, OUT_H, 5)]).reshape(-1, 1, 2)
    rows = []
    for ip in imgs:
        gt_path = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        pr_path = pred_dir / f"{ip.stem}.txt"
        img = cv2.imread(str(ip))
        if img is None or not gt_path.exists():
            print(f"  [건너뜀] 사진 또는 정답 없음: {ip.name}")
            continue
        h, w = img.shape[:2]
        g, _, _ = read_kpt(gt_path)
        if not pr_path.exists():
            rows.append({"image": ip.stem, "status": "no_pred"})
            continue
        p, conf = read_pose_pred(pr_path)
        g_px, p_px = (g * [w, h]).astype(np.float32), (p * [w, h]).astype(np.float32)
        status, _ = check_corners(p_px, w, h)
        Hg = cv2.getPerspectiveTransform(g_px, dst)
        Hp = cv2.getPerspectiveTransform(p_px, dst)
        corner = np.linalg.norm(cv2.perspectiveTransform(p_px.reshape(-1, 1, 2), Hg).reshape(-1, 2) - dst, axis=1)
        moved = cv2.perspectiveTransform(cv2.perspectiveTransform(grid, np.linalg.inv(Hg)), Hp)
        shift = np.linalg.norm((moved - grid).reshape(-1, 2), axis=1)
        orig = np.linalg.norm(p_px - g_px, axis=1)
        rows.append({"image": ip.stem, "status": status, "conf": f"{conf:.3f}",
                     "corner_px": f"{corner.max():.1f}", "grid_px": f"{shift.max():.1f}",
                     "worst": ["tl", "tr", "br", "bl"][int(corner.argmax())],
                     "orig_px": f"{orig.max():.1f}"})
    done = [r for r in rows if "grid_px" in r]
    print(f"\n== 꼭짓점 분석 [{args.name}] 사진 {len(imgs)}장 (대조 {len(done)}장) ==")
    if done:
        cp = np.array([float(r["corner_px"]) for r in done])
        gp = np.array([float(r["grid_px"]) for r in done])
        print(f"모서리 오차(보정 이미지 px, 사진별 최대): 평균 {cp.mean():.1f} · 중앙 {np.median(cp):.1f} · 최대 {cp.max():.1f}")
        print(f"칸 경계 이동(px, 칸 320):              평균 {gp.mean():.1f} · 중앙 {np.median(gp):.1f} · 최대 {gp.max():.1f}")
        for lim in (8, 16, 32, 64):
            print(f"  경계 이동 > {lim:>2}px (칸의 {lim / 320:.1%}): {int((gp > lim).sum())}장")
        print("상태: " + ", ".join(f"{k} {n}" for k, n in Counter(r["status"] for r in rows).most_common()))
        print("경계 이동이 큰 사진 5장: " + ", ".join(f"{r['image'][:20]}({r['grid_px']}, {r['worst']})"
                                                for r in sorted(done, key=lambda r: -float(r["grid_px"]))[:5]))
    csv_path = out / f"pose_{args.name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "status", "conf", "corner_px", "grid_px", "worst", "orig_px"])
        w.writeheader()
        w.writerows(rows)
    print(f"→ {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="검출·꼭짓점 예측 분석")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("det", help="검출 오탐·누락, 칸 개수 일치율")
    d.add_argument("--name", required=True)
    d.add_argument("--pred", required=True, help="predict 결과 labels 폴더 (save_txt save_conf)")
    d.add_argument("--list", required=True, help="예측한 사진 목록 (data/splits/tray_*.txt)")
    d.add_argument("--labels", default="data/tray/labels")
    d.add_argument("--nms-iou", type=float, default=0.5, help="클래스 무관 중복 제거 IoU")
    d.add_argument("--out", default="models/analysis")
    p = sub.add_parser("pose", help="꼭짓점 오차")
    p.add_argument("--name", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--list", required=True, help="data/splits/pose_*.txt (Roboflow 내보내기 images 경로)")
    p.add_argument("--out", default="models/analysis")
    args = ap.parse_args()
    run_det(args) if args.cmd == "det" else run_pose(args)


if __name__ == "__main__":
    main()
