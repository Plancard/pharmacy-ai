"""AI Hub 경구약제 라벨(COCO 형식, TL zip) → YOLO 라벨 변환.

- 라벨 zip은 풀지 않고 안에서 바로 읽는다.
- K코드 → 클래스 ID는 cv/classes.yaml을 따른다.
- 출력: data/pills/labels/<K코드>/<이미지 이름>.txt  (ultralytics 규칙: images ↔ labels 평행 경로)

사용 예:
    python cv/convert_aihub.py K-029163            # 한 약품
    python cv/convert_aihub.py --all               # classes.yaml 전체
    python cv/convert_aihub.py K-029163 --check 5  # 변환 후 bbox 그린 확인용 이미지 5장 저장
"""
import argparse
import csv
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AIHUB = DATA / "aihub"
TL_DIR = (AIHUB / "166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터"
          / "01.데이터" / "1.Training" / "라벨링데이터" / "단일경구약제_5000종")
IMG_DIR = DATA / "pills" / "images"
LBL_DIR = DATA / "pills" / "labels"
CHECK_DIR = DATA / "pills" / "check"
IMG_EXTS = (".png", ".jpg", ".jpeg")


def load_classes():
    cfg = yaml.safe_load((ROOT / "cv" / "classes.yaml").read_text(encoding="utf-8"))
    return {c["k_code"]: c for c in cfg["classes"]}


def find_tl_zip(kcode):
    with open(AIHUB / "pill_index.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["dl_mapping_code"] == kcode:
                return TL_DIR / r["zip"]
    raise SystemExit(f"{kcode}: pill_index.csv에 없음")


def find_image(kcode, file_name):
    """JSON의 파일명(png)과 확장자가 달라도(jpg로 줄인 경우) 찾는다."""
    stem = Path(file_name).stem
    for ext in IMG_EXTS:
        p = IMG_DIR / kcode / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def to_yolo(bbox, W, H):
    """COCO [x, y, w, h] (픽셀) → YOLO (cx, cy, w, h) 0~1. 이미지 밖은 잘라낸다."""
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    clipped = (x1, y1, x2, y2) != (x, y, x + w, y + h)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None, clipped
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H), clipped


def convert(kcode, cls_id):
    tl = find_tl_zip(kcode)
    if not tl.exists():
        raise SystemExit(f"{kcode}: 라벨 zip 없음 → {tl}")
    out_dir = LBL_DIR / kcode
    out_dir.mkdir(parents=True, exist_ok=True)
    stat = Counter()

    with zipfile.ZipFile(tl) as z:
        names = [n for n in z.namelist() if kcode in n and n.endswith(".json")]
        for n in names:
            stat["json"] += 1
            d = json.loads(z.read(n))
            info = d["images"][0]

            # 라벨 자체가 다른 약을 가리키는지 (K-016982 같은 원본 오류 대비)
            if info.get("dl_mapping_code", kcode) != kcode:
                stat["label_code_mismatch"] += 1
                continue

            img_path = find_image(kcode, info["file_name"])
            if img_path is None:
                stat["missing_image"] += 1
                continue

            # JSON 크기와 실제 이미지 크기가 다르면(축소 전달 등) 비율로 맞춘다
            W, H = Image.open(img_path).size
            sx, sy = W / info["width"], H / info["height"]
            if (sx, sy) != (1.0, 1.0):
                stat["resized"] += 1

            anns = d.get("annotations", [])
            if len(anns) != 1:
                stat[f"ann_{len(anns)}"] += 1

            lines = []
            for a in anns:
                x, y, w, h = a["bbox"]
                box, clipped = to_yolo((x * sx, y * sy, w * sx, h * sy), W, H)
                stat["clipped"] += clipped
                if box is None:
                    stat["bad_box"] += 1
                    continue
                lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in box))

            if lines:
                (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
                stat["written"] += 1

    n_img = sum(1 for p in (IMG_DIR / kcode).glob("*") if p.suffix.lower() in IMG_EXTS)
    stat["images_in_folder"] = n_img
    stat["images_without_label"] = n_img - stat["written"]
    return stat


def draw_check(kcode, n):
    """bbox를 그린 이미지를 data/pills/check/<K코드>/에 저장해 눈으로 확인한다."""
    labels = sorted((LBL_DIR / kcode).glob("*.txt"))
    out = CHECK_DIR / kcode
    out.mkdir(parents=True, exist_ok=True)
    for lbl in random.sample(labels, min(n, len(labels))):
        img_path = find_image(kcode, lbl.stem + ".png")
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for line in lbl.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            dr.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline="red", width=4)
            dr.text((cx - w / 2, cy - h / 2 - 14), f"class {c}", fill="red")
        im.thumbnail((640, 640))
        im.save(out / f"{lbl.stem}.jpg", quality=85)
    print(f"  확인용 이미지 → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kcodes", nargs="*", help="변환할 K코드 (예: K-029163)")
    ap.add_argument("--all", action="store_true", help="classes.yaml의 전체 약품")
    ap.add_argument("--check", type=int, default=0, help="bbox 확인 이미지 장수")
    args = ap.parse_args()

    classes = load_classes()
    targets = list(classes) if args.all else args.kcodes
    if not targets:
        ap.error("K코드를 주거나 --all을 쓰세요")

    for k in targets:
        if k not in classes:
            print(f"[건너뜀] {k}: classes.yaml에 없음")
            continue
        if not (IMG_DIR / k).exists():
            print(f"[건너뜀] {k} ({classes[k]['name']}): 이미지 폴더 없음")
            continue
        c = classes[k]
        stat = convert(k, c["id"])
        print(f"[{k}] {c['name']} (class {c['id']})")
        for key, v in stat.items():
            print(f"  {key}: {v}")
        if args.check:
            draw_check(k, args.check)


if __name__ == "__main__":
    main()
