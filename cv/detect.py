"""알약 검출·분류(F-CV-03) — 서비스에서 부르는 검출 모듈.

    from cv.detect import detect_pills, detect
    dets = detect_pills(rectified_image)     # 보정 이미지(960×1280) → [{class_id, item_code, bbox, conf}]
    a2 = detect("var/uploads/tray.jpg")      # 사진 → rectify → 검출 → 칸 배정 → A.2 (cv/stub.detect와 같은 형식·예외)

처리 절차 (기능분석서 F-CV-03)
  YOLO26n 추론 (imgsz 960, 출력 최소 conf 0.70, 기준선 미만은 버림)
  기준선 0.70은 11/17 확정값이다: 낮은 점수의 유령 박스가 칸에 들어가 거짓 일치가 되는 것보다
  놓쳐서 불일치(확인 요청)가 되는 쪽이 안전하다. 검증 24장에서 칸 개수 일치는 0.70에서 288/288칸·24/24장
  (0.60은 287/288·23/24, 0.75부터 부족). 그래서 검수의 저신뢰(low_conf) 사유는 따로 두지 않는다.
  박스는 보정 이미지 픽셀 [x1, y1, x2, y2], 위→아래·왼→오른쪽 순서로 돌려준다.
  item_code(품목기준코드)는 class_id에서 약품 사전(agent/resources/drug_dictionary.json)으로 구하고, 없으면 None.

detect()의 오류: 트레이를 못 찾으면 cv.stub.TrayNotFound(E_TRAY_NOT_FOUND). 실제 모듈이 나오면
cv/stub.py의 detect를 이 함수로 바꾼다 (함수 이름·인자·반환 형식은 그대로, plan은 받기만 하고 쓰지 않는다).

점검용 (보정된 사진 목록으로 검출 결과를 CSV로 남긴다, 저장소 루트에서)
  python -m cv.detect --list data/splits/pose_val.txt --csv data/tray/detect_check.csv

모델은 처음 부를 때 한 번만 읽는다. 환경변수 CV_DET_MODEL로 가중치 경로를 바꾼다.
Gunicorn 스레드끼리 동시에 predict를 부르지 않게 잠근다.
"""
import argparse
import csv
import json
import os
import threading
from pathlib import Path

import numpy as np

from cv import tray
from cv.stub import TrayNotFound
from cv.warp import OUT_W

ROOT = Path(__file__).resolve().parents[1]
DET_MODEL = os.environ.get("CV_DET_MODEL", str(ROOT / "models" / "v2b" / "weights" / "last.pt"))
DET_IMGSZ = 960
DET_CONF = 0.70         # 검출 기준선 (11/17 확정). 이 미만은 알약으로 인정하지 않는다
DRUGS = ROOT / "agent" / "resources" / "drug_dictionary.json"

_model = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()
_item_codes = None


def get_model():
    """검출 모델 (한 번만 만든다)"""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(DET_MODEL)
    return _model


def item_codes():
    """class_id → 품목기준코드 (약품 사전에서 한 번만 읽는다)"""
    global _item_codes
    if _item_codes is None:
        drugs = json.loads(DRUGS.read_text(encoding="utf-8"))["drugs"]
        _item_codes = {d["class_id"]: d["item_code"] for d in drugs if d.get("class_id") is not None}
    return _item_codes


def detect_pills(img, model=None):
    """보정 이미지(BGR, 960×1280) → 검출 목록 [{class_id, item_code, bbox, conf}].

    conf가 기준선(DET_CONF) 미만인 박스는 모델에서도, 여기서도 버린다. 칸·플래그는 assign_cells가 붙인다.
    """
    model = model or get_model()
    with _predict_lock:
        r = model.predict(img, imgsz=DET_IMGSZ, conf=DET_CONF, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = np.asarray(r.boxes.xyxy, dtype=float).reshape(-1, 4)
    conf = np.asarray(r.boxes.conf, dtype=float).reshape(-1)
    cls = np.asarray(r.boxes.cls, dtype=float).reshape(-1)
    codes = item_codes()
    dets = []
    for box, c, k in zip(xyxy, conf, cls):
        if c < DET_CONF:
            continue
        class_id = int(k)
        dets.append({
            "class_id": class_id,
            "item_code": codes.get(class_id),
            "bbox": [round(float(v), 1) for v in box],
            "conf": round(float(c), 3),
        })
    # 위→아래·왼→오른쪽 (박스 중심 기준)
    dets.sort(key=lambda d: ((d["bbox"][1] + d["bbox"][3]) / 2, (d["bbox"][0] + d["bbox"][2]) / 2))
    return dets


def detect(image_path, plan=None, *, model=None, seg_model=None):
    """F-CV-03 인터페이스 (cv/stub.detect와 같은 형식): 트레이 사진 → A.2 검출 결과.

    트레이 보정(F-CV-02) → 알약 검출 → 칸·플래그 배정(F-CV-04)까지 한다. plan은 쓰지 않는다.
    트레이를 못 찾으면 TrayNotFound.
    """
    path = Path(image_path)
    r = tray.rectify(path, model=seg_model)
    dets = tray.assign_cells(detect_pills(r.image, model))
    return {
        "image_id": path.stem,
        "tray": {"rows": 4, "cols": 3, "cell_px": OUT_W // 3, "corrected": True},
        "detections": dets,
    }


# ---------------------------------------------------------------- 점검용 실행
def main():
    ap = argparse.ArgumentParser(description="검출 점검: 사진마다 보정·검출을 돌려 결과를 CSV로 기록")
    ap.add_argument("--list", required=True, help="사진 경로 목록 (data/splits/pose_*.txt 형식)")
    ap.add_argument("--model", default=DET_MODEL)
    ap.add_argument("--seg-model", default=tray.SEG_MODEL)
    ap.add_argument("--csv", default="data/tray/detect_check.csv")
    args = ap.parse_args()

    from ultralytics import YOLO
    det_model, seg_model = YOLO(args.model), YOLO(args.seg_model)
    rows, failed = [], []
    for p in Path(args.list).read_text(encoding="utf-8").split():
        p = Path(p)
        try:
            a2 = detect(p, model=det_model, seg_model=seg_model)
        except TrayNotFound as e:
            failed.append((p.name, str(e)))
            continue
        for d in a2["detections"]:
            cell = d["cell"]
            rows.append({"file": p.name, "class_id": d["class_id"], "item_code": d["item_code"], "conf": d["conf"],
                         "day": cell["day"] if cell else "", "time": cell["time"] if cell else "",
                         "flags": " ".join(d["flags"]), "bbox": " ".join(str(v) for v in d["bbox"])})
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "class_id", "item_code", "conf", "day", "time", "flags", "bbox"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"검출 {len(rows)}개 (기준선 {DET_CONF}), 재촬영 요청 {len(failed)}장 → {args.csv}")
    for name, reason in failed:
        print(f"  {name}  {reason}")


if __name__ == "__main__":
    main()
