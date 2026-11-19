"""처방전·약봉투 텍스트 추출 (F-CV-01, F-CV-06) — 서비스에서 부르는 OCR 모듈.

    from cv.ocr import extract_text
    a1 = extract_text("var/uploads/rx.jpg", doc_type="prescription", image_id="…")   # → A.1 dict

처리 절차 (기능분석서 F-CV-01)
  1. 이미지 읽기 (EXIF 방향 반영)
  2. 처방전 펴기: 종이 네 모서리를 찾아 A4 세로로 원근 보정 (cv/rectify_doc.py, 못 찾으면 원본 그대로)
  3. 긴 변 MAX_SIDE로 줄여 EasyOCR(ko, en) — 줄 단위 텍스트·4점 좌표·신뢰도
  4. 4점 좌표 → [x1, y1, x2, y2] (OCR에 넣은 이미지, 즉 펴진 이미지의 픽셀 기준)
  5. 위→아래, 왼쪽→오른쪽 정렬, line_id 부여
  줄이 하나도 없으면 OcrEmpty(E_OCR_EMPTY) → 재촬영 요청.

A.1의 좌표는 '펴진 이미지' 기준이다. 웹이 원본 위치를 표시하려면(F-WEB-06) read_document()가
돌려주는 펴진 이미지를 저장해 그 위에 그린다. A.1 형식(image_id·doc_type·width·height·lines)은 바꾸지 않는다.

EasyOCR 모델은 처음 부를 때 한 번만 읽는다(수 초). Gunicorn 스레드끼리 동시에 readtext를 부르지 않게 잠근다.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

from cv.rectify_doc import rectify
from cv.stub import OcrEmpty

MAX_SIDE = 1600          # OCR에 넣는 긴 변 (잠정, 기능분석서 F-CV-01) — cv/ocr_eval.py --max-side로 비교
LOW_CONF = 0.5           # 이 미만 줄은 웹에서 강조 (판단은 웹·Agent, 여기서는 그대로 반환)
DOC_TYPES = ("prescription", "med_bag")

_reader = None
_reader_lock = threading.Lock()
_ocr_lock = threading.Lock()


def get_reader():
    """EasyOCR Reader (한 번만 만든다). 첫 실행은 모델 다운로드가 필요하다."""
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr
            _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _reader


def _shrink(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def to_lines(results, scale=1.0):
    """EasyOCR readtext 결과 [(4점, 글자, 신뢰도)] → A.1 lines (scale로 나눠 OCR 입력 전 좌표로 되돌림)"""
    lines = []
    for box, text, conf in results:
        text = str(text).strip()
        if not text:
            continue
        xs = [float(p[0]) / scale for p in box]
        ys = [float(p[1]) / scale for p in box]
        lines.append({"text": text,
                      "bbox": [int(min(xs)), int(min(ys)), int(round(max(xs))), int(round(max(ys)))],
                      "conf": round(float(conf), 2)})
    lines.sort(key=lambda l: ((l["bbox"][1] + l["bbox"][3]) / 2, l["bbox"][0]))
    return [{"line_id": i, **l} for i, l in enumerate(lines)]


def _edge_cut(info):
    """펴기 정보 -> A.1 paper_edge_cut. 종이를 찾았을 때만 true/false, 못 찾았으면 None(모름)"""
    if info.get("status") not in ("ok", "rotated90"):
        return None
    return info.get("edge") == "yes"


def read_document(image_path, doc_type="prescription", image_id=None, *, flatten=True,
                  max_side=MAX_SIDE, reader=None, rect_out=None):
    """→ (A.1 dict, OCR에 쓴 이미지(펴진 이미지 또는 원본), 펴기 정보 dict)

    펴기 정보: {"status": ok · rotated90 · skip, "edge": "yes"(종이가 사진 밖으로 잘림) 또는 ""}
    edge가 yes면 OCR은 하되 재촬영 안내 후보다(잘린 줄은 읽을 수 없다).
    """
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type은 {DOC_TYPES} 중 하나: {doc_type}")
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)       # EXIF 방향은 OpenCV가 반영한다
    if img is None:
        raise OcrEmpty(f"이미지를 읽을 수 없음: {path.name}")
    info = {"status": "off", "edge": ""}
    if flatten:
        img, info = rectify(img)
    small, s = _shrink(img, max_side)
    reader = reader or get_reader()
    with _ocr_lock:
        results = reader.readtext(small, detail=1, paragraph=False)
    lines = to_lines(results, s)
    if not lines:
        raise OcrEmpty(path.stem)
    h, w = img.shape[:2]
    a1 = {"image_id": image_id or path.stem, "doc_type": doc_type, "width": w, "height": h, "lines": lines}
    edge = _edge_cut(info)
    if edge is not None:
        a1["paper_edge_cut"] = edge
    if rect_out is not None:  # 웹이 OCR 박스를 그릴 펴진 이미지(a1 좌표와 같은 크기)
        Path(rect_out).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(rect_out), img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"펴진 이미지를 저장하지 못함: {rect_out}")
    return a1, img, {"status": info.get("status", "off"), "edge": info.get("edge", "")}


def extract_text(image_path, doc_type="prescription", image_id=None, **kw):
    """F-CV-01 인터페이스: 이미지 → A.1 OCR 결과 dict"""
    return read_document(image_path, doc_type, image_id, **kw)[0]
