"""RunPod 업로드용 데이터 묶음 만들기: data.yaml이 가리키는 이미지를 긴 변 기준으로 줄이고 JPEG로 저장

- 라벨(YOLO txt)은 0~1 비율 좌표라 그대로 복사한다
- 원본(data/pills, data/tray)은 건드리지 않고, 별도 폴더에 묶음을 만든다
- 묶음 안의 data.yaml·train.txt·val.txt는 './images/...' 상대 경로라 RunPod 어디에 풀어도 그대로 쓴다
- train 목록에 같은 사진이 반복돼 있으면(11/12 트레이 비중 올리기) 목록의 반복은 유지하고 변환은 한 번만 한다

사용
  python cv/shrink.py                                   # data_base.yaml → data/upload/base, 긴 변 1024
  python cv/shrink.py --max 640                         # 더 작게
  python cv/shrink.py --tar                             # data/upload/base.tar까지 (파일 하나로 올리기)
  python cv/shrink.py --yaml data/splits/data.yaml --out data/upload/v1 --tar   # 11/12 트레이 포함본

RunPod에서
  tar -xf base.tar && yolo detect train data=base/data.yaml ...
"""
import argparse
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- data.yaml 읽기
def resolve_root(yaml_path, data):
    """ultralytics와 같은 규칙: path 없으면 yaml 폴더, 상대 경로면 현재 폴더 → yaml 폴더 순"""
    p = data.get("path")
    if not p:
        return yaml_path.parent.resolve()
    p = Path(p)
    if p.is_absolute():
        return p
    return p.resolve() if p.exists() else (yaml_path.parent / p).resolve()


def list_images(entry, root):
    """train/val 값 하나(폴더, txt 목록, 또는 그 리스트) → 이미지 절대 경로 목록(반복 유지)"""
    if isinstance(entry, (list, tuple)):
        return [x for e in entry for x in list_images(e, root)]
    p = Path(entry)
    p = p if p.is_absolute() else root / p
    if p.is_dir():
        return sorted(x.resolve() for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
    if p.suffix == ".txt":
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("./"):
                q = p.parent / s[2:]
            else:
                q = Path(s)
                if not q.is_absolute():
                    q = q if q.exists() else root / q
            out.append(q.resolve())
        return out
    sys.exit(f"train/val 값을 읽을 수 없음: {entry}")


def label_of(img):
    """ultralytics 규칙: 마지막 /images/ → /labels/, 확장자 → .txt"""
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def bundle_rel(img):
    """data/pills/images/K-000573/a.png → pills/K-000573/a  (images 위 폴더 이름으로 출처 구분)"""
    parts = img.parts
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(parts[i - 1], *parts[i + 1:]).with_suffix("")


# ---------------------------------------------------------------- 변환
def shrink_one(src, dst, max_side, quality):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return src, "read_error", 0, 0
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return src, "ok", src.stat().st_size, dst.stat().st_size


def main():
    ap = argparse.ArgumentParser(description="RunPod 업로드용 축소 묶음")
    ap.add_argument("--yaml", default="data/splits/data_base.yaml")
    ap.add_argument("--out", default="data/upload/base", help="묶음 폴더")
    ap.add_argument("--max", type=int, default=1024, help="긴 변 최대 픽셀(작은 사진은 키우지 않음)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 품질")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tar", action="store_true", help="묶음을 <out>.tar 하나로도 만든다")
    ap.add_argument("--force", action="store_true", help="기존 묶음 지우고 다시 만들기")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = resolve_root(yaml_path, data)
    out = Path(args.out)
    if out.exists():
        if not args.force:
            sys.exit(f"{out}이 이미 있음. 다시 만들려면 --force")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    lists, todo = {}, {}
    for sp in SPLITS:
        if not data.get(sp):
            continue
        rels = []
        for img in list_images(data[sp], root):
            rel = bundle_rel(img)
            prev = todo.setdefault(rel, img)
            if prev != img:
                sys.exit(f"묶음 안 이름이 겹침: {prev} / {img}")
            rels.append(rel)
        lists[sp] = rels
    if not todo:
        sys.exit("이미지가 하나도 없음")

    print(f"{yaml_path} → {out}  (이미지 {len(todo)}장, 긴 변 {args.max}, JPEG {args.quality})")
    errors, before, after, no_label = [], 0, 0, 0
    with ThreadPoolExecutor(args.workers) as ex:
        jobs = [ex.submit(shrink_one, src, out / "images" / rel.with_suffix(".jpg"), args.max, args.quality)
                for rel, src in todo.items()]
        for n, j in enumerate(jobs, 1):
            src, status, b, a = j.result()
            if status != "ok":
                errors.append(f"  {status}: {src}")
            before, after = before + b, after + a
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}")

    for rel, src in todo.items():
        lab = label_of(src)
        dst = out / "labels" / rel.with_suffix(".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if lab.exists():
            shutil.copy2(lab, dst)
        else:
            no_label += 1  # 라벨 없는 사진 = 배경 이미지로 학습됨

    for sp, rels in lists.items():
        (out / f"{sp}.txt").write_text(
            "".join(f"./images/{r.with_suffix('.jpg').as_posix()}\n" for r in rels), encoding="utf-8")
    new = {k: v for k, v in data.items() if k not in ("path", *SPLITS, "download")}
    new = {**{sp: f"{sp}.txt" for sp in lists}, **new}
    (out / "data.yaml").write_text(yaml.safe_dump(new, allow_unicode=True, sort_keys=False), encoding="utf-8")

    mb = 1024 * 1024
    print("목록:", ", ".join(f"{sp} {len(r)}줄" for sp, r in lists.items()))
    print(f"용량: {before / mb:,.0f}MB → {after / mb:,.0f}MB")
    if no_label:
        print(f"  [주의] 라벨 없는 사진 {no_label}장 (빈 트레이가 아니면 확인)")
    if errors:
        print("읽기 실패\n" + "\n".join(errors))

    if args.tar:
        tar_path = out.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as t:
            t.add(out, arcname=out.name)
        print(f"tar → {tar_path} ({tar_path.stat().st_size / mb:,.0f}MB)")


if __name__ == "__main__":
    main()
