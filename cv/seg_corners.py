"""트레이 분할(seg) 예측 → 꼭짓점 4개 (대안 A, pose2seg.py와 짝).

마스크 윤곽선에서 네 변을 직선으로 맞추고, 이웃한 두 직선의 교점을 꼭짓점으로 쓴다.
순서는 라벨과 같은 규칙: 트레이는 세로로 긴 3열×4행이므로 짧은 변 두 개 중 위쪽이 tl→tr,
거기서 시계 방향으로 tl·tr·br·bl (회전 사진도 화면 기준, 180° 사진은 화면 위쪽 짧은 변이 tl→tr).

출력은 pose 예측과 같은 txt 형식이라 analyze_preds.py pose로 그대로 비교할 수 있다.
  "0 cx cy w h x_tl y_tl x_tr y_tr x_br y_br x_bl y_bl conf" (비율)

사용 예 (학원 PC, 저장소 루트)
  python cv/seg_corners.py --model models/seg_v1/weights/best.pt --list data/splits/pose_val.txt --out models/seg_pred_val/labels
  python cv/analyze_preds.py pose --name seg_val --pred models/seg_pred_val/labels --list data/splits/pose_val.txt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def _line(pts):
    """점들 → 직선 (점 p, 방향 d)"""
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    a = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, q - p)[0]
    return p + t * d


def rough_quad(poly):
    """윤곽선 → 볼록 껍질을 단순화해 꼭짓점 4개 (못 찾으면 None)"""
    hull = cv2.convexHull(poly.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    for f in np.linspace(0.005, 0.1, 40):
        a = cv2.approxPolyDP(hull, f * peri, True)
        if len(a) == 4:
            return a.reshape(4, 2).astype(np.float64)
        if len(a) < 4:
            break
    return None


def refine(poly, quad, band=0.03, trim=0.15):
    """각 변 근처의 윤곽선 점으로 직선을 다시 맞춘 뒤 교점 → 꼭짓점. 모서리가 둥근 부분(trim)은 뺀다."""
    poly = poly.reshape(-1, 2).astype(np.float64)
    peri = sum(np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4))
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        t = (poly - a) @ ab / (L * L)
        d = poly - a
        dist = np.abs(ab[0] * d[:, 1] - ab[1] * d[:, 0]) / L
        sel = poly[(t > trim) & (t < 1 - trim) & (dist < band * peri)]
        if len(sel) < 5:
            return quad
        lines.append(_line(sel))
    out = []
    for i in range(4):
        p = _intersect(lines[i - 1], lines[i])
        if p is None:
            return quad
        out.append(p)
    return np.array(out)


def order_corners(q):
    """→ tl·tr·br·bl. 화면 좌표(y 아래)에서 시계 방향, 위쪽 짧은 변이 tl→tr"""
    c = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]   # y가 아래라 각도 오름차순 = 시계 방향
    side = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    k = 0 if side[0] + side[2] <= side[1] + side[3] else 1           # q[k]→q[k+1]이 짧은 변
    if (q[k][1] + q[k + 1][1]) > (q[(k + 2) % 4][1] + q[(k + 3) % 4][1]):
        k += 2                                                       # 위쪽 짧은 변에서 시작
    return np.roll(q, -k, axis=0)


def quad_from_polygon(poly):
    """윤곽선 → 꼭짓점 4개. 윤곽선을 따라 도는 순서이고 시작점은 임의다 (order_corners로 정리하기 전). 못 찾으면 None."""
    quad = rough_quad(poly)
    return None if quad is None else refine(poly, quad)


def long_axis(q):
    """4점(윤곽선 순서) → 긴 변 두 개의 평균 방향 벡터 (부호는 의미 없음)"""
    a = (q[1] - q[0]) + (q[2] - q[3])      # 변 q0→q1 과 변 q3→q2
    b = (q[2] - q[1]) + (q[3] - q[0])      # 변 q1→q2 와 변 q0→q3
    return a if np.linalg.norm(a) >= np.linalg.norm(b) else b


def is_sideways(q, max_deg=30.0):
    """긴 변이 화면 가로에서 max_deg 이내면 True (트레이를 옆으로 눕혀 찍음). 4점은 순서를 정리하기 전이어도 된다.

    order_corners는 짧은 변을 항상 위쪽에 놓기 때문에 순서를 정리한 뒤에는 가로세로 비율이 늘 0.75라서
    눕힘을 비율로 알 수 없다. 그래서 정리하기 전의 네 점에서 긴 변의 방향으로 판정한다.
    긴 변이 가로에 가까울수록 어느 짧은 변이 위인지가 노이즈로 갈려 방향이 불안정해진다.
    """
    vx, vy = long_axis(np.asarray(q, dtype=float))
    return bool(np.degrees(np.arctan2(abs(vy), abs(vx))) < max_deg)


def corners_from_polygon(poly):
    quad = quad_from_polygon(poly)
    return None if quad is None else order_corners(quad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", required=True, help="예측할 사진 목록 (data/splits/pose_*.txt)")
    ap.add_argument("--out", required=True, help="txt를 쓸 폴더 (매번 새로 씀)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = [Path(l) for l in Path(args.list).read_text(encoding="utf-8").split()]
    fail = []
    for ip in imgs:
        (out / f"{ip.stem}.txt").unlink(missing_ok=True)
        r = model.predict(str(ip), imgsz=args.imgsz, conf=args.conf, retina_masks=True, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            fail.append(ip.name)
            continue
        i = int(r.boxes.conf.argmax())
        h, w = r.orig_shape
        q = corners_from_polygon(r.masks.xy[i])
        if q is None:
            fail.append(ip.name)
            continue
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
        kp = (q / [w, h]).reshape(-1)
        vals = box + kp.tolist() + [float(r.boxes.conf[i])]
        (out / f"{ip.stem}.txt").write_text("0 " + " ".join(f"{v:.6f}" for v in vals) + "\n")
    print(f"꼭짓점 {len(imgs) - len(fail)}/{len(imgs)}장 → {out}")
    if fail:
        print(f"[실패] 마스크 없음 또는 사각형 못 찾음 {len(fail)}장: {' '.join(fail[:5])}")


if __name__ == "__main__":
    main()
