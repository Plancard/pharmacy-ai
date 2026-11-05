"""
처방전 OCR 테스트 (EasyOCR)

사용법
  pip install easyocr
  python ocr_test.py --images ./rx_images --out ./ocr_out
  python ocr_test.py --images ./rx_images --out ./ocr_out --preprocess   # 전처리 비교
  python ocr_test.py --images ./rx_images --out ./ocr_out --gt gt.json  # 약품명 인식률 계산

출력 (ocr_out/)
  <이미지명>.json     Agent에 넘길 OCR 결과 (텍스트, 박스, 신뢰도)
  <이미지명>_box.png  박스를 그린 확인용 이미지
  summary.json        이미지별 처리 시간, 평균 신뢰도, 약품명 인식률
"""
import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from rapidfuzz import fuzz

import cv2
import easyocr
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def preprocess(img):
    """흑백 + 1.5배 확대 + 대비 보정. 사진 촬영본에서 효과가 큰 편."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def shrink(img, max_side=1800):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s) if s < 1 else (img, 1.0)

def run_ocr(reader, img, scale=1.0):
    results = reader.readtext(img, detail=1, paragraph=False)
    items = []
    for box, text, conf in results:
        pts = [[float(x) / scale, float(y) / scale] for x, y in box]  # 원본 좌표로 환산
        items.append({"text": text, "confidence": round(float(conf), 4), "box": pts})
    return items


def draw_boxes(img, items, path):
    canvas = img.copy()
    for it in items:
        pts = np.array(it["box"], dtype=np.int32)
        color = (0, 160, 0) if it["confidence"] >= 0.5 else (0, 0, 220)  # 낮은 신뢰도는 빨강
        cv2.polylines(canvas, [pts], True, color, 2)
    cv2.imwrite(str(path), canvas)


def norm(s):
    return "".join(s.split()).lower()


def drug_hit_rate(items, drugs, threshold=80):
    texts = [norm(it["text"]) for it in items]
    hits = []
    for name in drugs:
        n = norm(name)
        best = max([fuzz.partial_ratio(n, t) for t in texts] + [0.0])
        hits.append({"drug": name, "hit": best >= threshold, "best_ratio": round(best, 1)})
    rate = sum(h["hit"] for h in hits) / len(hits) if hits else None
    return rate, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="ocr_out")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--gt", help='정답 파일 예: {"rx_001.jpg": ["타이레놀정500mg", "게보린정"]}')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8")) if args.gt else {}

    t0 = time.time()
    reader = easyocr.Reader(["ko", "en"], gpu=False)  # 첫 실행 시 모델 다운로드(인터넷 필요)
    print(f"모델 로딩 {time.time() - t0:.1f}s")

    summary = []
    for path in sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS):
        img = cv2.imread(str(path))

        if img is None:
            print(f"읽기 실패: {path.name}")
            continue
        h, w = img.shape[:2]
        small, s = shrink(img)

        t = time.time()
        if args.preprocess:
            items = run_ocr(reader, preprocess(small), scale=s * 1.5)
        else:
            items = run_ocr(reader, small, scale=s)
        elapsed = time.time() - t

        result = {
            "image": path.name,
            "width": w,
            "height": h,
            "engine": "easyocr",
            "preprocess": args.preprocess,
            "items": items,
        }
        (out / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        draw_boxes(img, items, out / f"{path.stem}_box.png")

        row = {
            "image": path.name,
            "seconds": round(elapsed, 2),
            "n_items": len(items),
            "mean_conf": round(sum(i["confidence"] for i in items) / len(items), 3) if items else 0,
        }
        if path.name in gt:
            row["drug_hit_rate"], row["drug_hits"] = drug_hit_rate(items, gt[path.name])
        summary.append(row)
        print(f"{path.name}: {row['seconds']}s, {row['n_items']}개, 평균 신뢰도 {row['mean_conf']}"
              + (f", 약품명 인식률 {row['drug_hit_rate']:.0%}" if "drug_hit_rate" in row else ""))

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
