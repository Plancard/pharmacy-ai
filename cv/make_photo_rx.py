"""가상 처방전 PNG를 '폰으로 찍은 것처럼' 바꾼 시험용 사진을 만든다 (REQ-CV-01 OCR 개발 중 점검용).

rx8 가상 처방전 20장(agent/resources/samples/rx8_img)은 화면에서 바로 그린 깨끗한 이미지라 OCR이 쉽다.
실제로는 약사가 종이 처방전을 폰으로 찍으므로, 같은 글자에 촬영 조건만 입혀 정확도가 얼마나 떨어지는지 미리 본다.
글자는 그대로라 정답 파일(gt.json · ground_truth.json)은 이름만 바꿔 그대로 쓴다.
최종 평가는 인쇄해서 실제로 찍은 사진으로 따로 한다 (이 사진은 개발 중 점검용).

입히는 조건 (사진마다 seed로 고정, 난이도 3단계를 번갈아)
  - 책상 배경 위에 종이, 원근(네 모서리를 따로 밀기), 회전
  - 조명 기울기(한쪽이 어두움)와 부드러운 그림자 한 덩이
  - 흐림(초점 · 손떨림), 센서 잡음, 약간 노란 색감
  - JPEG 압축

사용 예
  python cv/make_photo_rx.py                                  # → data/rx_photo/ (20장 + gt.json · ground_truth.json · params.csv)
  python cv/ocr_test.py --images data/rx_photo --out data/ocr_photo --gt data/rx_photo/gt.json
  python cv/ocr_test.py --images agent/resources/samples/rx8_img --out data/ocr_clean --gt agent/resources/samples/rx8_img/gt.json
  → 두 summary.json의 약품명 인식률을 비교한다

출력 이름: rx8_01.png → rx8p_01.jpg (p = photo). data/는 커밋하지 않으므로 다른 PC에서는 다시 돌린다(seed가 같으면 같은 사진).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "agent" / "resources" / "samples" / "rx8_img"
OUT = ROOT / "data" / "rx_photo"
CANVAS = (3000, 4000)                 # 폰 세로 사진 (가로, 세로)

# 난이도별 범위: (원근 비율, 회전 °, 조명 최저 밝기, 그림자 진하기, 초점 흐림 σ, 손떨림 길이 px, 잡음 σ, JPEG 품질)
LEVELS = {
    "easy":   dict(persp=(0.00, 0.03), rot=(0, 3),  light=(0.85, 0.95), shadow=(0.00, 0.10), blur=(0.0, 0.6),
                   motion=(0, 0),  noise=(1, 3), jpeg=(85, 92)),
    "medium": dict(persp=(0.03, 0.06), rot=(3, 7),  light=(0.70, 0.85), shadow=(0.10, 0.25), blur=(0.6, 1.2),
                   motion=(0, 5),  noise=(3, 6), jpeg=(75, 85)),
    "hard":   dict(persp=(0.06, 0.10), rot=(7, 12), light=(0.55, 0.70), shadow=(0.25, 0.40), blur=(1.2, 2.0),
                   motion=(5, 11), noise=(6, 9), jpeg=(65, 75)),
}
ORDER = ("easy", "medium", "hard")


def u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def place_on_desk(doc, rng, p):
    """종이를 책상 위에 놓고 원근 · 회전을 준 사진 크기 이미지와 사용한 값."""
    W, H = CANVAS
    h, w = doc.shape[:2]
    # 책상: 회갈색 + 큰 얼룩
    base = np.array([rng.integers(70, 110), rng.integers(80, 115), rng.integers(90, 125)], dtype=np.float32)
    desk = np.ones((H, W, 3), np.float32) * base
    blob = cv2.GaussianBlur(rng.normal(0, 25, (H // 50, W // 50)).astype(np.float32), (0, 0), 3)
    desk += cv2.resize(blob, (W, H))[..., None]

    # 종이가 사진 세로의 약 85%를 차지하게, 가운데 근처
    scale = 0.85 * H / h
    dw, dh = w * scale, h * scale
    cx, cy = W / 2 + rng.uniform(-0.04, 0.04) * W, H / 2 + rng.uniform(-0.03, 0.03) * H
    angle = math.radians(u(rng, p["rot"]) * rng.choice([-1, 1]))
    corners = np.array([[-dw / 2, -dh / 2], [dw / 2, -dh / 2], [dw / 2, dh / 2], [-dw / 2, dh / 2]])
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    dst = corners @ rot.T + [cx, cy]
    persp = u(rng, p["persp"])
    dst += rng.uniform(-1, 1, (4, 2)) * persp * np.array([dw, dh])      # 네 모서리를 따로 밀어 원근
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    paper = cv2.warpPerspective(doc.astype(np.float32), M, (W, H), flags=cv2.INTER_AREA, borderValue=0)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    img = paper * mask + desk * (1 - mask)
    return img, {"rotate_deg": round(math.degrees(angle), 1), "perspective": round(persp, 3)}


def lighting(img, rng, p):
    """조명 · 그림자는 부드러워서 1/8 크기로 계산한 뒤 키운다 (큰 흐림을 원본 크기로 하면 느림)."""
    FH, FW = img.shape[:2]
    H, W = FH // 8, FW // 8
    low = u(rng, p["light"])
    theta = rng.uniform(0, 2 * math.pi)                       # 어두워지는 방향
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = ((xx / W - 0.5) * math.cos(theta) + (yy / H - 0.5) * math.sin(theta)) + 0.5
    gain = 1.0 - (1.0 - low) * np.clip(t, 0, 1)
    # 부드러운 그림자 (손 · 폰 그림자처럼 한쪽 가장자리에서 들어옴)
    depth = u(rng, p["shadow"])
    shadow = np.zeros((H, W), np.float32)
    if depth > 0:
        r = int(min(W, H) * rng.uniform(0.25, 0.45))
        edge = rng.integers(0, 4)
        c = [(int(rng.uniform(0.2, 0.8) * W), 0), (W, int(rng.uniform(0.2, 0.8) * H)),
             (int(rng.uniform(0.2, 0.8) * W), H), (0, int(rng.uniform(0.2, 0.8) * H))][edge]
        cv2.circle(shadow, c, r, 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), r * 0.35)
    gain = gain * (1.0 - depth * shadow)
    gain = cv2.resize(gain.astype(np.float32), (FW, FH), interpolation=cv2.INTER_LINEAR)
    warm = np.array([0.93, 0.98, 1.0], np.float32)             # BGR: 파랑을 줄여 약간 노랗게
    return img * gain[..., None] * warm, {"light_min": round(low, 2), "shadow": round(depth, 2)}


def blur_noise(img, rng, p):
    sigma = u(rng, p["blur"])
    if sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    length = int(round(u(rng, p["motion"])))
    if length >= 2:
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0 / length
        a = rng.uniform(0, 180)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), a, 1.0), (length, length))
        k /= max(k.sum(), 1e-6)
        img = cv2.filter2D(img, -1, k)
    n = u(rng, p["noise"])
    img = img + rng.normal(0, n, img.shape).astype(np.float32)
    return img, {"blur_sigma": round(sigma, 2), "motion_px": length, "noise": round(n, 1)}


def make_one(src_path, out_path, level, seed):
    rng = np.random.default_rng(seed)
    p = LEVELS[level]
    doc = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if doc is None:
        raise SystemExit(f"읽을 수 없음: {src_path}")
    img, a = place_on_desk(doc, rng, p)
    img, b = lighting(img, rng, p)
    img, c = blur_noise(img, rng, p)
    q = int(round(u(rng, p["jpeg"])))
    cv2.imwrite(str(out_path), np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return {"level": level, "seed": seed, **a, **b, **c, "jpeg": q}


def main():
    ap = argparse.ArgumentParser(description="가상 처방전 → 촬영본처럼 바꾼 시험 사진")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", choices=ORDER, help="한 난이도로만 만든다 (기본: easy·medium·hard를 번갈아)")
    args = ap.parse_args()

    srcs = sorted(args.src.glob("rx8_*.png"))
    if not srcs:
        raise SystemExit(f"{args.src}에 rx8_*.png가 없음")
    args.out.mkdir(parents=True, exist_ok=True)
    rename = {s.name: s.stem.replace("rx8_", "rx8p_") + ".jpg" for s in srcs}

    rows = []
    for i, s in enumerate(srcs):
        level = args.level or ORDER[i % len(ORDER)]
        info = make_one(s, args.out / rename[s.name], level, args.seed * 1000 + i)
        rows.append({"file": rename[s.name], "source": s.name, **info})
        print(f"{rename[s.name]}  {level:6s} 회전 {info['rotate_deg']:+5.1f}°  원근 {info['perspective']:.3f}  "
              f"흐림 {info['blur_sigma']:.2f}+{info['motion_px']}px  조명 {info['light_min']:.2f}  JPEG {info['jpeg']}")

    for name in ("gt.json", "ground_truth.json"):          # 글자는 같으므로 정답은 이름만 바꿔 쓴다
        f = args.src / name
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            moved = {rename[k]: v for k, v in data.items() if k in rename}
            (args.out / name).write_text(json.dumps(moved, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(args.out / "params.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {args.out} ({len(rows)}장, gt.json · ground_truth.json · params.csv)")


if __name__ == "__main__":
    main()
