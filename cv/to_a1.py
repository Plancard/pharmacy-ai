"""
ocr_test.py 결과를 부록 A.1 형식(CV → Agent)으로 변환

사용법
  python cv/to_a1.py --src ocr_out --out a1_samples

입력: ocr_test.py가 만든 <이미지명>.json (summary.json은 건너뜀)
  {"image", "width", "height", "engine", "preprocess",
   "items": [{"text", "confidence", "box": [[x, y] * 4]}]}

출력: a1_samples/<이미지명>.json
  {"image_id": "rx_001", "width": 1240, "height": 1754,
   "lines": [{"text": "타이레놀정/500mg", "bbox": [x1, y1, x2, y2], "conf": 0.94}]}

- bbox는 EasyOCR의 4점 좌표를 감싸는 축 정렬 사각형(정수 픽셀, 원본 이미지 기준)
- lines는 박스 중심 y(위→아래), 그다음 x 순서. 같은 줄 묶기는 Agent가 한다
"""
import argparse
import json
from pathlib import Path


def to_bbox(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return [int(min(xs)), int(min(ys)), int(round(max(xs))), int(round(max(ys)))]


def convert(src):
    lines = []
    for it in src["items"]:
        text = it["text"].strip()
        if not text:
            continue
        lines.append({"text": text, "bbox": to_bbox(it["box"]), "conf": round(float(it["confidence"]), 2)})
    # 박스 중심 y, 그다음 x 순으로 정렬
    lines.sort(key=lambda l: ((l["bbox"][1] + l["bbox"][3]) / 2, l["bbox"][0]))
    return {
        "image_id": Path(src["image"]).stem,
        "width": src["width"],
        "height": src["height"],
        "lines": lines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ocr_out")
    ap.add_argument("--out", default="a1_samples")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.src).glob("*.json")):
        if path.name == "summary.json":
            continue
        a1 = convert(json.loads(path.read_text(encoding="utf-8")))
        (out / path.name).write_text(json.dumps(a1, ensure_ascii=False, indent=2), encoding="utf-8")
        low = sum(l["conf"] < 0.5 for l in a1["lines"])
        print(f"{path.name}: {len(a1['lines'])}줄, 저신뢰(<0.5) {low}줄")


if __name__ == "__main__":
    main()
