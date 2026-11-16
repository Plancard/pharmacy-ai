"""트레이 원근 보정(F-CV-02)과 칸 좌표 매핑(F-CV-04) — 서비스에서 부르는 트레이 모듈.

    from cv.tray import rectify, assign_cells
    r = rectify("var/uploads/tray.jpg")      # → RectifyResult (보정 이미지 960×1280, 변환 행렬 …)
    dets = assign_cells(detections)           # F-CV-03 검출 목록 → 칸(cell)·플래그(flags) 붙임

rectify 처리 절차 (기능분석서 F-CV-02)
  1. 트레이 영역 분할 (YOLO26n-seg, imgsz 960, 분할 신뢰도 0.5 잠정)
  2. 마스크 윤곽선의 네 변을 직선으로 맞추고 이웃한 두 직선의 교점을 꼭짓점으로 (cv/seg_corners.py)
     순서는 사진 기준 좌상·우상·우하·좌하 (위쪽 짧은 변이 좌상→우상, 시계 방향)
  3. 호모그래피로 960×1280 정면 이미지 (3열 × 4행, 칸 320×320). 긴 변이 세로가 되도록 90° 회전을 바로잡는다
     (180°는 모양이 같아 여기서 알 수 없다 → 촬영 규칙 + 검수 단계 rotated 판정)
  아래 경우는 모두 TrayNotFound(E_TRAY_NOT_FOUND) → 재촬영 요청
    트레이를 못 찾음 / 분할 신뢰도 0.5 미만 / 윤곽선에서 네 변을 못 찾음 /
    꼭짓점이 사진 밖(트레이 일부만 찍힘) / 꼭짓점 순서가 꼬임 / 가로세로 비율이 3:4에서 10% 이상 벗어남

assign_cells 처리 절차 (기능분석서 F-CV-04)
  박스 중심점이 들어간 칸으로 cell {day, time}을 정한다 (중심점이 경계선 위면 오른쪽·아래쪽 칸).
  박스 면적의 20%(잠정) 이상이 자기 칸 밖으로 나가 있으면 boundary,
  같은 칸 안 두 박스의 IoU가 0.5(잠정) 이상이면 두 박스 모두 overlap.
  중심점이 트레이 바깥이면 cell을 None으로 둔다 → 판정(F-CV-05)에서 제외하고 그 사진은 확인 필요로 기록한다.

점검용 (사진 목록으로 rectify를 돌려 성공·재촬영 사유·가로세로 비율을 CSV로 남긴다, 저장소 루트에서)
  python -m cv.tray --list data/splits/pose_val.txt --model models/seg_v1/weights/best.pt --csv data/tray/rectify_check.csv

모델은 처음 부를 때 한 번만 읽는다. 환경변수 CV_SEG_MODEL로 가중치 경로를 바꾼다.
Gunicorn 스레드끼리 동시에 predict를 부르지 않게 잠근다.
"""
import argparse
import csv
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv.seg_corners import corners_from_polygon
from cv.stub import DAYS, TIMES, TrayNotFound
from cv.warp import OUT_H, OUT_W, check_corners, warp

ROOT = Path(__file__).resolve().parents[1]
SEG_MODEL = os.environ.get("CV_SEG_MODEL", str(ROOT / "models" / "seg_v1" / "weights" / "best.pt"))
SEG_IMGSZ = 960
SEG_CONF = 0.5          # 분할 신뢰도 기준 (잠정, 11/17 모델 확정 때 정함 — 기능분석서 7장)
ASPECT = OUT_W / OUT_H  # 3:4 = 0.75
ASPECT_TOL = 0.10       # 꼭짓점 사각형의 가로세로 비율이 3:4에서 이 비율 이상 벗어나면 재촬영
BOUNDARY_FRAC = 0.20    # 자기 칸 밖으로 나간 박스 면적 비율 (잠정)
OVERLAP_IOU = 0.5       # 같은 칸 안 두 박스의 IoU (잠정)

_model = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()


def get_model():
    """seg 모델 (한 번만 만든다)"""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(SEG_MODEL)
    return _model


@dataclass
class RectifyResult:
    image: np.ndarray       # 보정 이미지 (BGR, 1280×960)
    matrix: np.ndarray      # 원본 픽셀 → 보정 이미지 픽셀 3×3 호모그래피
    corners: np.ndarray     # 원본 이미지의 꼭짓점 (4,2), 좌상·우상·우하·좌하 (90° 바로잡은 뒤의 순서)
    conf: float             # 분할 신뢰도
    ratio: float            # (위+아래 변)/(왼+오른 변), 정상 0.75
    area: float             # 트레이 면적 / 사진 면적
    small: bool = False     # 트레이가 사진에 비해 작음(면적 10% 미만): 보정은 했지만 확인 대상
    corrected: bool = True


def find_corners(img, model=None):
    """BGR 이미지 → (꼭짓점 (4,2) 좌상·우상·우하·좌하, 분할 신뢰도). 못 찾으면 TrayNotFound."""
    model = model or get_model()
    with _predict_lock:
        r = model.predict(img, imgsz=SEG_IMGSZ, conf=SEG_CONF, retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        raise TrayNotFound(f"트레이를 찾지 못함 (분할 신뢰도 {SEG_CONF} 미만)")
    i = int(r.boxes.conf.argmax())
    quad = corners_from_polygon(r.masks.xy[i])
    if quad is None:
        raise TrayNotFound("마스크 윤곽선에서 네 변을 찾지 못함")
    return quad, float(r.boxes.conf[i])


def _ratio(pts):
    top, right = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
    bottom, left = np.linalg.norm(pts[2] - pts[3]), np.linalg.norm(pts[3] - pts[0])
    return (top + bottom) / (left + right)


def _upright(pts):
    """위·아래 변이 긴 변이면(가로로 놓인 트레이) 꼭짓점 순서를 한 칸 돌려 긴 변이 세로가 되게 한다."""
    return np.roll(pts, -1, axis=0) if _ratio(pts) > 1.0 else pts


def rectify_image(img, model=None):
    """BGR 이미지 → RectifyResult. 재촬영이 필요하면 TrayNotFound."""
    h, w = img.shape[:2]
    pts, conf = find_corners(img, model)
    pts = _upright(pts)
    # sideways 판정은 위에서 직접 돌렸으므로 여기서는 끈다 (cut·order·small만 본다)
    status, info = check_corners(pts, w, h, sideways_ratio=float("inf"))
    if status == "cut":
        raise TrayNotFound("트레이 일부가 사진 밖으로 나감")
    if status == "order":
        raise TrayNotFound("꼭짓점 순서가 맞지 않음")
    ratio = info["ratio"]
    if abs(ratio / ASPECT - 1) > ASPECT_TOL:
        raise TrayNotFound(f"트레이 가로세로 비율 {ratio:.2f}이 3:4({ASPECT:.2f})에서 {ASPECT_TOL:.0%} 이상 벗어남")
    dst = np.float32([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]])
    matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    return RectifyResult(image=warp(img, pts), matrix=matrix, corners=pts, conf=conf,
                         ratio=ratio, area=info["area"], small=(status == "small"))


def rectify(image_path, *, model=None):
    """F-CV-02 인터페이스: 트레이 사진 → RectifyResult (EXIF 방향은 OpenCV가 반영한다)"""
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise TrayNotFound(f"이미지를 읽을 수 없음: {path.name}")
    return rectify_image(img, model)


# ---------------------------------------------------------------- F-CV-04 칸 좌표 매핑
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assign_cells(detections, rows=4, cols=3, size=(OUT_W, OUT_H)):
    """F-CV-03 검출 목록(bbox는 보정 이미지 픽셀 [x1,y1,x2,y2]) → 칸·플래그를 붙인 새 목록 (입력은 안 바꾼다).

    각 검출에 cell {"day": 1~3, "time": morning·noon·evening·bedtime} 또는 None, flags ["boundary", "overlap"]가 붙는다.
    """
    if rows != len(TIMES) or cols != len(DAYS):
        raise ValueError(f"트레이는 {len(TIMES)}행 × {len(DAYS)}열: rows={rows}, cols={cols}")
    cw, ch = size[0] / cols, size[1] / rows
    out = []
    for det in detections:
        d = dict(det)
        x1, y1, x2, y2 = d["bbox"]
        col = int(np.floor((x1 + x2) / 2 / cw))
        row = int(np.floor((y1 + y2) / 2 / ch))
        d["flags"] = []
        if 0 <= col < cols and 0 <= row < rows:
            d["cell"] = {"day": DAYS[col], "time": TIMES[row]}
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if area > 0:
                ix = max(0.0, min(x2, (col + 1) * cw) - max(x1, col * cw))
                iy = max(0.0, min(y2, (row + 1) * ch) - max(y1, row * ch))
                if 1 - ix * iy / area >= BOUNDARY_FRAC:
                    d["flags"].append("boundary")
        else:
            d["cell"] = None
        out.append(d)

    by_cell = {}
    for d in out:
        if d["cell"] is not None:
            by_cell.setdefault((d["cell"]["day"], d["cell"]["time"]), []).append(d)
    for ds in by_cell.values():
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                if _iou(ds[i]["bbox"], ds[j]["bbox"]) >= OVERLAP_IOU:
                    for d in (ds[i], ds[j]):
                        if "overlap" not in d["flags"]:
                            d["flags"].append("overlap")
    return out


# ---------------------------------------------------------------- 점검용 실행
def main():
    ap = argparse.ArgumentParser(description="트레이 보정 점검: 사진마다 rectify를 돌려 결과를 CSV로 기록")
    ap.add_argument("--list", required=True, help="사진 경로 목록 (data/splits/pose_*.txt 형식)")
    ap.add_argument("--model", default=SEG_MODEL)
    ap.add_argument("--csv", default="data/tray/rectify_check.csv")
    ap.add_argument("--out", help="보정본을 저장할 폴더 (없으면 저장 안 함)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in Path(args.list).read_text(encoding="utf-8").split():
        p = Path(p)
        row = {"file": p.name}
        try:
            r = rectify(p, model=model)
            row.update(result="ok", conf=round(r.conf, 3), ratio=round(float(r.ratio), 3),
                       area=r.area, small="yes" if r.small else "")
            if out:
                cv2.imwrite(str(out / f"{p.stem}.jpg"), r.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except TrayNotFound as e:
            row.update(result="E_TRAY_NOT_FOUND", reason=str(e))
        rows.append(row)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "result", "conf", "ratio", "area", "small", "reason"]
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    ok = sum(r["result"] == "ok" for r in rows)
    print(f"보정 성공 {ok}/{len(rows)}장, 재촬영 요청 {len(rows) - ok}장 → {args.csv}")
    for r in rows:
        if r["result"] != "ok":
            print(f"  {r['file']}  {r['reason']}")


if __name__ == "__main__":
    main()
