"""처방전 사진 펴기: 종이의 네 모서리를 찾아 정면(A4 세로)으로 원근 보정한다 (F-CV-01 전처리).

종이가 조금만 기울어도 왼쪽 약품명과 오른쪽 횟수 · 일수 칸의 높이가 한 줄 가까이 어긋나
Agent 규칙 경로가 약품 줄을 못 묶는다 (11/12 촬영본 20장: 규칙 성공 2장, LLM으로도 9/41줄).
OCR 전에 종이를 펴서 표의 행이 수평이 되게 한다. 학습 없이 OpenCV만 쓴다.

방법 (긴 변 1000px로 줄여서 찾고, 원본 해상도에서 편다)
  1. 변 찾기: Canny 경계 → 긴 직선(HoughLinesP)을 가로 · 세로로 나누고, 가장 바깥쪽 직선 묶음을 종이의 네 변으로 본다.
     네 변의 교점이 모서리다. 그림자가 진 모서리나 사진 밖으로 조금 나간 모서리도 변을 연장해 찾는다
  2. 1이 안 되면 밝은 영역(Otsu 이진화)의 가장 큰 덩어리를 4점 다각형으로 근사한다
  3. 모서리를 좌상 · 우상 · 우하 · 좌하로 정렬하고 A4 세로(2480×3508)로 원근 변환
  종이가 사진의 20% 미만이거나 사각형이 너무 찌그러지면 펴지 않고 원본을 그대로 쓴다 (status=skip).
  종이가 사진 테두리에 걸려 잘렸으면 edge=yes (펴기는 하되, 잘린 부분은 없으므로 재촬영 안내 후보).
  종이가 가로로 놓였으면(긴 변이 가로) 90° 돌려 세로로 맞춘다. 위아래 180°는 판단하지 않는다.

사용 예
  python cv/rectify_doc.py --src data/rx_photo --out data/rx_photo_rect
  python cv/ocr_test.py --images data/rx_photo_rect --out data/ocr_photo_rect --gt data/rx_photo/gt.json
  출력 폴더에는 펴진 사진(같은 이름, .jpg)과 rectify.csv(파일별 상태 · 모서리 좌표)가 생긴다.
  --debug를 주면 찾은 모서리를 그린 확인용 사진(<이름>_corners.jpg)도 남긴다.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 2480, 3508          # A4 300dpi 세로 (가상 처방전과 같은 크기)
WORK = 1000                        # 모서리 찾기용 축소 크기 (긴 변)
MIN_AREA = 0.20                    # 종이가 사진에서 차지해야 하는 최소 비율
MAX_ASPECT_ERR = 0.35              # 찾은 사각형 세로/가로 비가 A4(1.414)에서 벗어나도 되는 정도
CUT_RUN = 0.05                     # 종이(밝은 영역)가 사진 테두리에 이 비율 넘게 닿아 있으면 '종이가 잘림'
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def order_corners(pts):
    """4점 → 좌상 · 우상 · 우하 · 좌하 (x+y 최소가 좌상, 최대가 우하, y-x 최소가 우상)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def _quad_from_contour(c):
    peri = cv2.arcLength(c, True)
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32)   # 모서리가 둥글거나 잘렸을 때
    s, d = hull.sum(1), hull[:, 1] - hull[:, 0]
    return np.array([hull[s.argmin()], hull[d.argmin()], hull[s.argmax()], hull[d.argmax()]], np.float32)


def _line_through(points):
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(points, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


def _intersect(l1, l2):
    (p, r), (q, s) = l1, l2
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-6:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    return p + t * r


def _quad_from_lines(gray):
    """가장 바깥쪽 긴 직선 네 묶음(위 · 아래 · 왼 · 오른)의 교점. 못 찾으면 None."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=int(0.15 * min(h, w)), maxLineGap=15)
    if segs is None:
        return None
    horiz, vert = [], []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        (horiz if ang < 35 or ang > 145 else vert if 55 < ang < 125 else []).append((x1, y1, x2, y2))
    if len(horiz) < 2 or len(vert) < 2:
        return None

    def side(group, axis, outer_min):
        # axis=1(가로선은 y로), 0(세로선은 x로). 가장 바깥 직선과 가까운(1.5% 안) 선분을 모아 한 직선으로 맞춘다
        mids = np.array([(g[axis] + g[axis + 2]) / 2 for g in group])
        edge = mids.min() if outer_min else mids.max()
        tol = 0.015 * (h if axis == 1 else w)
        pts = [pt for g, m in zip(group, mids) if abs(m - edge) <= tol for pt in ((g[0], g[1]), (g[2], g[3]))]
        return _line_through(pts), edge

    (top, ty), (bottom, by) = side(horiz, 1, True), side(horiz, 1, False)
    (left, lx), (right, rx) = side(vert, 0, True), side(vert, 0, False)
    if by - ty < 0.4 * h or rx - lx < 0.3 * w:          # 한쪽 변만 잡힌 경우
        return None
    corners = [_intersect(top, left), _intersect(top, right), _intersect(bottom, right), _intersect(bottom, left)]
    if any(c is None for c in corners):
        return None
    q = np.array(corners, np.float32)
    if (np.abs(q[:, 0] - w / 2) > w).any() or (np.abs(q[:, 1] - h / 2) > h).any():   # 말이 안 되게 먼 교점
        return None
    return q


def _edge_support(edges, a, b, n=200):
    """선분 a-b 위 점들 중 경계(edges)에 닿는 비율. 사진 밖으로 나간 부분은 세지 않는다."""
    h, w = edges.shape
    t = np.linspace(0, 1, n)[:, None]
    pts = np.round(a + (b - a) * t).astype(int)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if inside.sum() < n * 0.3:
        return 0.0
    p = pts[inside]
    return float(edges[p[:, 1], p[:, 0]].mean() / 255)


def _best_sides(candidates, edges):
    """후보 사각형들에서 변마다(위 · 오른 · 아래 · 왼) 경계에 가장 잘 붙는 변을 골라 교점으로 모서리를 다시 만든다."""
    lines = []
    for k in range(4):                                   # 0 위(0→1), 1 오른(1→2), 2 아래(2→3), 3 왼(3→0)
        best = max(candidates, key=lambda q: _edge_support(edges, q[k], q[(k + 1) % 4]))
        a, b = best[k].astype(np.float64), best[(k + 1) % 4].astype(np.float64)
        lines.append((a, (b - a) / max(np.linalg.norm(b - a), 1e-6)))
    corners = [_intersect(lines[3], lines[0]), _intersect(lines[0], lines[1]),
               _intersect(lines[1], lines[2]), _intersect(lines[2], lines[3])]
    if any(c is None for c in corners):
        return None
    return np.array(corners, np.float32)


def find_paper(img):
    """→ (원본 좌표의 네 모서리 [좌상, 우상, 우하, 좌하] 또는 None, 면적 비율, 종이가 사진 밖으로 잘렸는지).

    두 방법(바깥 직선 · 밝은 영역)으로 사각형 후보를 만들고, 변마다 경계에 더 잘 붙는 쪽을 고른다.
    그림자가 진 쪽은 밝은 영역이 틀리고, 배경에 직선이 있으면 직선 쪽이 틀리기 쉬워 서로 보완한다.
    """
    h, w = img.shape[:2]
    scale = WORK / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    area = small.shape[0] * small.shape[1]
    candidates = []
    if gray.std() < 10:                                  # 밝기 차이가 거의 없음 (종이 · 배경 구분 불가)
        return None, 0.0, False

    quad = _quad_from_lines(gray)
    if quad is not None:
        candidates.append(order_corners(quad))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))   # 글자 구멍 메우기
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cut = max(bw[0].mean(), bw[-1].mean(), bw[:, 0].mean(), bw[:, -1].mean()) / 255 > CUT_RUN
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if MIN_AREA <= cv2.contourArea(c) / area <= 0.98:   # 사진 전체가 밝으면 종이를 찾은 게 아니다
            candidates.append(order_corners(_quad_from_contour(c)))
    if not candidates:
        return None, 0.0, cut

    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((5, 5), np.uint8))
    quad = _best_sides(candidates, edges) if len(candidates) > 1 else candidates[0]
    if quad is None:
        quad = candidates[0]
    quad = order_corners(quad)
    ratio = cv2.contourArea(quad) / area
    if ratio < MIN_AREA:
        return None, ratio, cut
    return quad / scale, ratio, cut


def side_lengths(q):
    top, right = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
    bottom, left = np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0])
    return (top + bottom) / 2, (left + right) / 2


def rectify(img):
    """→ (펴진 이미지 또는 원본, 정보 dict). info["status"]: ok · skip(종이를 못 찾음) · rotated90(가로로 놓인 종이)."""
    quad, ratio, cut = find_paper(img)
    info = {"status": "skip", "area": round(float(ratio), 3), "corners": "", "edge": "yes" if cut else ""}
    if quad is None:
        return img, info
    wid, hei = side_lengths(quad)
    if wid > hei:                                                  # 가로로 놓인 종이 → 모서리 순서를 한 칸 돌려 세로로
        quad = np.roll(quad, -1, axis=0)
        wid, hei = hei, wid
        info["status"] = "rotated90"
    aspect = hei / max(wid, 1)
    if abs(aspect - OUT_H / OUT_W) > MAX_ASPECT_ERR:
        info["aspect"] = round(float(aspect), 2)
        info["status"] = "skip"
        return img, info
    M = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32))
    out = cv2.warpPerspective(img, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if info["status"] == "skip":
        info["status"] = "ok"
    info["aspect"] = round(float(aspect), 2)
    info["corners"] = " ".join(f"{x:.0f},{y:.0f}" for x, y in quad)
    return out, info


def main():
    ap = argparse.ArgumentParser(description="처방전 사진 펴기 (원근 보정)")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--debug", action="store_true", help="찾은 모서리를 그린 사진도 남긴다")
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"{args.src}에 이미지가 없음")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{p.name}: 읽을 수 없음")
            continue
        out, info = rectify(img)
        cv2.imwrite(str(args.out / (p.stem + ".jpg")), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if args.debug and info["corners"]:
            dbg = img.copy()
            pts = np.array([[float(v) for v in xy.split(",")] for xy in info["corners"].split()], np.int32)
            cv2.polylines(dbg, [pts], True, (0, 0, 255), 8)
            cv2.imwrite(str(args.out / f"{p.stem}_corners.jpg"), cv2.resize(dbg, None, fx=0.3, fy=0.3))
        rows.append({"file": p.name, **info})
        print(f"{p.name}: {info['status']}  종이 {info['area']:.0%}  세로/가로 {info.get('aspect', '-')}"
              + ("  [종이가 사진 밖으로 잘림]" if info.get("edge") else ""))
    with open(args.out / "rectify.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "status", "edge", "area", "aspect", "corners"])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] != "skip" for r in rows)
    print(f"→ {args.out} (펴짐 {n_ok}/{len(rows)}, rectify.csv)")


if __name__ == "__main__":
    main()
