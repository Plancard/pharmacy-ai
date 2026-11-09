"""트레이 원근 보정: 꼭짓점 4점 → 960×1280 정면 이미지 (라벨링 기준 문서 '작업 절차 2')

꼭짓점은 사진 기준 좌상·우상·우하·좌하 순서이고, 칸 영역 바깥선이 만나는 안쪽 모서리에 찍는다
(트레이 바깥 모서리는 둥글어 위치가 흔들린다). 보정본은 가로 3칸 × 세로 4칸, 칸 하나 320×320.

입력
  --src  원본 사진 폴더 (g07_s03.jpg, empty_s01.jpg, rot_s01.jpg ...)
  --kpt  Roboflow에서 YOLO pose(Keypoint) 형식으로 내보낸 labels 폴더
         한 줄: class cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4  (0~1 비율, v 없는 형식도 읽음)
         Roboflow가 붙이는 이름(g07_s03_jpg.rf.abcd1234.txt)은 원래 이름으로 되돌려 맞춘다
         내보낼 때 Resize는 끄거나 Stretch만 쓴다(Fit은 비율 좌표가 어긋남)

검사 (corners.csv의 status)
  ok        보정함
  small     보정했지만 트레이가 사진에 비해 작음(확인)
  cut       꼭짓점이 안 보이거나 사진 밖 → 보정 안 함, REQ-CV-08 테스트 자료
  sideways  가로로 긴 사각형 = 트레이를 옆으로 놓고 찍음 → 보정 안 함, 다시 촬영
  order     점 순서가 꼬였거나 뒤집힘 → 보정 안 함, Roboflow에서 순서 수정
  no_kpt    꼭짓점 라벨 없음
  180°는 모양이 같아 여기서 알 수 없다 → 촬영 규칙(아침 줄 위) + 검수 단계 rotated 판정

사용
  python cv/warp.py --src data/tray/raw --kpt data/tray/raw_kpt --out data/tray/images

서비스에서는 warp()와 check_corners()를 가져다 쓰고, 꼭짓점만 pose 모델 예측으로 바꾼다.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 960, 1280            # 3열 × 4행, 칸 320×320
PORTRAIT = OUT_W / OUT_H            # 0.75, 옆으로 놓으면 1.33
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]
RF_NAME = re.compile(r"^(.*)_(jpg|jpeg|png)\.rf\.[0-9a-zA-Z]+$")


# ---------------------------------------------------------------- 서비스와 같이 쓰는 부분
def check_corners(pts, img_w, img_h, sideways_ratio=1.0, min_area=0.10, vis=None):
    """pts: (4,2) 픽셀 좌표, 좌상·우상·우하·좌하. → (status, 정보 dict)"""
    pts = np.asarray(pts, dtype=np.float64)
    info = {}
    if vis is not None and min(vis) <= 0:
        return "cut", info
    m = 0.01 * max(img_w, img_h)  # 1% 여유
    if (pts[:, 0] < -m).any() or (pts[:, 0] > img_w + m).any() \
            or (pts[:, 1] < -m).any() or (pts[:, 1] > img_h + m).any():
        return "cut", info

    # 화면 좌표(y 아래로)에서 좌상→우상→우하→좌하는 시계 방향: 모든 외적이 양수여야 볼록·순서 정상
    cross = [_cross(pts[(i + 1) % 4] - pts[i], pts[(i + 2) % 4] - pts[(i + 1) % 4]) for i in range(4)]
    if not all(c > 0 for c in cross):
        return "order", info

    top, right = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
    bottom, left = np.linalg.norm(pts[2] - pts[3]), np.linalg.norm(pts[3] - pts[0])
    ratio = (top + bottom) / (left + right)
    area = 0.5 * abs(sum(_cross(pts[i], pts[(i + 1) % 4]) for i in range(4))) / (img_w * img_h)
    info.update(ratio=round(float(ratio), 3), area=round(float(area), 3))
    # 정상 0.75, 옆으로 1.33 → 기본 기준 1.0(두 값의 기하평균)
    if ratio > sideways_ratio:
        return "sideways", info
    if area < min_area:
        return "small", info
    return "ok", info


def warp(img, pts, size=(OUT_W, OUT_H)):
    """원본 BGR 이미지 + 꼭짓점 4점 → size 크기 정면 이미지"""
    w, h = size
    src = np.asarray(pts, dtype=np.float32)
    # 휴대폰 원본은 트레이가 보정본보다 훨씬 커서, 한 번에 줄이면 각인이 계단처럼 깨진다
    # → 트레이 높이가 출력의 1.5배쯤 되게 먼저 INTER_AREA로 줄인 뒤 펴기
    tray_h = max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
    s = tray_h / (1.5 * h)
    if s > 1:
        img = cv2.resize(img, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        src = src / s
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- Roboflow 라벨 읽기
def base_stem(stem):
    m = RF_NAME.match(stem)
    return m.group(1) if m else stem


def read_kpt(path):
    """→ (pts 비율 (4,2), vis 또는 None). 줄이 여러 개면 첫 줄을 쓰고 경고."""
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    if not lines:
        return None, None, 0
    v = [float(x) for x in lines[0][5:]]
    if len(v) == 12:
        a = np.array(v).reshape(4, 3)
        return a[:, :2], a[:, 2].tolist(), len(lines)
    if len(v) == 8:
        return np.array(v).reshape(4, 2), None, len(lines)
    raise ValueError(f"{path.name}: 키포인트 값 {len(v)}개 (4점이면 8 또는 12개)")


# ---------------------------------------------------------------- 본체
def main():
    ap = argparse.ArgumentParser(description="트레이 원근 보정")
    ap.add_argument("--src", default="data/tray/raw", help="원본 사진 폴더")
    ap.add_argument("--kpt", default="data/tray/raw_kpt", help="Roboflow YOLO pose 라벨 폴더")
    ap.add_argument("--out", default="data/tray/images", help="보정본 폴더")
    ap.add_argument("--csv", default="data/tray/corners.csv", help="꼭짓점·검사 결과 기록")
    ap.add_argument("--sideways", type=float, default=1.0, help="가로/세로 비율이 이보다 크면 옆으로 놓은 것")
    ap.add_argument("--min-area", type=float, default=0.10, help="트레이 면적/사진 면적 하한(경고만)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--force", action="store_true", help="기존 보정본 덮어쓰기")
    args = ap.parse_args()

    src, kpt_dir, out = Path(args.src), Path(args.kpt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kpts = {}
    for p in kpt_dir.glob("*.txt"):
        b = base_stem(p.stem)
        if b in kpts:
            print(f"  [주의] 같은 사진의 라벨이 둘: {b}")
        kpts[b] = p

    rows, counts = [], {}
    for img_path in sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS):
        stem = img_path.stem
        dst = out / f"{stem}.jpg"
        row = {"file": img_path.name}
        if dst.exists() and not args.force:
            sys.exit(f"{dst}가 이미 있음. 덮어쓰려면 --force")

        img = cv2.imread(str(img_path))  # EXIF 회전 적용(Roboflow Auto-Orient와 같은 기준)
        if img is None:
            row["status"] = "read_error"
        elif stem not in kpts:
            row["status"] = "no_kpt"
        else:
            h, w = img.shape[:2]
            rel, vis, n = read_kpt(kpts[stem])
            if rel is None:
                row["status"] = "no_kpt"
            else:
                if n > 1:
                    print(f"  [주의] {img_path.name}: 트레이 라벨 {n}개, 첫 줄만 씀")
                pts = rel * [w, h]
                status, info = check_corners(pts, w, h, args.sideways, args.min_area, vis)
                row.update(status=status, **info)
                for i, name in enumerate(["tl", "tr", "br", "bl"]):
                    row[f"{name}_x"], row[f"{name}_y"] = round(pts[i, 0], 1), round(pts[i, 1], 1)
                if status in ("ok", "small"):
                    cv2.imwrite(str(dst), warp(img, pts), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)

    for b in sorted(set(kpts) - {r["file"].rsplit(".", 1)[0] for r in rows}):
        print(f"  [주의] 라벨은 있는데 원본 사진이 없음: {b}")

    fields = ["file", "status", "ratio", "area"] + [f"{c}_{a}" for c in ("tl", "tr", "br", "bl") for a in "xy"]
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    print("결과:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for r in rows:
        if r["status"] not in ("ok",):
            print(f"  {r['status']:9s} {r['file']}  ratio={r.get('ratio', '-')} area={r.get('area', '-')}")
    print(f"보정본 → {out}, 기록 → {args.csv}")


if __name__ == "__main__":
    main()
