"""트레이 원본 사진 방향 고정 + EXIF 제거 + 이름 변경 (Roboflow 업로드 전)

왜 필요한가
  갤럭시 S23 FE를 바닥과 나란히 들고 찍으면 EXIF Orientation이 6/8로 뒤바뀌어 기록된다(픽셀 방향은 늘 같음).
  Roboflow(Auto-Orient)와 cv2.imread는 EXIF대로 돌리므로, 그대로 두면 사진마다 180°가 뒤집힐 수 있다.
  → EXIF는 무시하고 원본 픽셀(4000×3000 가로)을 시계 방향 90° 고정 회전 → 3000×4000 세로, EXIF 없이 저장.

이름 변경
  --names 에 촬영 순서대로 된 이름 목록을 주면(photos.csv의 file 열, 또는 한 줄에 하나인 txt)
  원본을 촬영 시각(EXIF DateTimeOriginal+SubSec, 없으면 파일 이름) 순으로 정렬해 차례로 붙인다.
  개수가 다르면 멈춘다. --names 를 안 주면 원래 이름을 그대로 쓴다.

사용
  # 1) 미리보기: 대응표만 출력, 아무것도 안 씀
  python cv/fix_orient.py --src ~/tray_orig --names data/tray/photos.csv
  # 2) 실제 저장 + 확인용 썸네일 모음(contact.jpg)
  python cv/fix_orient.py --src ~/tray_orig --names data/tray/photos.csv --apply
  # photos.csv에 아직 안 찍은 사진까지 적혀 있으면 앞에서부터 원본 수만큼만
  python cv/fix_orient.py --src data/tray/raw --names data/tray/photos.csv --partial
  # 이미 이름을 g01_s01.jpg처럼 바꿔 둔 경우
  python cv/fix_orient.py --src ~/tray_orig --apply

  결과 폴더(기본 data/tray/fixed)를 Roboflow에 올리고, warp.py도 --src data/tray/fixed 로 돌린다.
  contact.jpg를 열어 모든 트레이가 세로·아침 줄 위인지 눈으로 확인한다.
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png"}
TAG_DATETIME_ORIGINAL, TAG_SUBSEC_ORIGINAL = 36867, 37521
EXIF_IFD = 0x8769


def shot_key(path):
    """촬영 시각 정렬 키. EXIF가 없으면 파일 이름."""
    try:
        with Image.open(path) as im:
            ifd = im.getexif().get_ifd(EXIF_IFD)
            dt = ifd.get(TAG_DATETIME_ORIGINAL)
            sub = str(ifd.get(TAG_SUBSEC_ORIGINAL, "0")).strip().ljust(3, "0")
            if dt:
                return (0, f"{dt}.{sub}", path.name)
    except Exception:
        pass
    return (1, "", path.name)


def read_names(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            names = [r["file"].strip() for r in csv.DictReader(f) if r.get("file", "").strip()]
    else:
        names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    names = [n if Path(n).suffix else n + ".jpg" for n in names]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        sys.exit(f"이름 목록에 중복: {sorted(dup)}")
    return names


def fix(src_path, dst_path, quality):
    with Image.open(src_path) as im:
        w, h = im.size
        if w > h:
            out = im.transpose(Image.Transpose.ROTATE_270)  # 시계 방향 90°
            note = ""
        else:
            out = im.copy()  # 이미 세로 픽셀 → 회전 안 함
            note = "이미 세로라 회전 안 함(확인)"
        out = out.convert("RGB")
        out.save(dst_path, "JPEG", quality=quality, subsampling=0)  # exif 인자 없음 → EXIF 제거
    return out.size, note


def contact_sheet(paths, out_path, thumb=240, cols=10):
    rows = (len(paths) + cols - 1) // cols
    tw, th = thumb, thumb * 4 // 3
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = ImageOps.contain(im, (tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 18)
        sheet.paste(im, (x + (tw - im.width) // 2, y))
        d.text((x + 4, y + th + 2), p.stem, fill="black")
    sheet.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser(description="트레이 원본 방향 고정·EXIF 제거·이름 변경")
    ap.add_argument("--src", required=True, help="폰에서 옮긴 원본 폴더")
    ap.add_argument("--out", default="data/tray/fixed", help="결과 폴더")
    ap.add_argument("--names", help="촬영 순서 이름 목록(photos.csv 또는 txt). 없으면 원래 이름 유지")
    ap.add_argument("--partial", action="store_true",
                    help="이름 목록이 원본보다 길면(아직 안 찍은 사진 포함) 앞에서부터 원본 수만큼만 쓴다")
    ap.add_argument("--thumb", type=int, default=240, help="contact.jpg 썸네일 가로 폭(px)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--apply", action="store_true", help="실제로 저장(없으면 대응표만 출력)")
    ap.add_argument("--force", action="store_true", help="결과 폴더에 같은 이름이 있으면 덮어쓰기")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    files = sorted((p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS), key=shot_key)
    if not files:
        sys.exit(f"{src}에 사진 없음")
    no_exif = [p.name for p in files if shot_key(p)[0] == 1]

    if args.names:
        names = read_names(args.names)
        if len(names) > len(files) and args.partial:
            print(f"  [partial] 이름 {len(names)}개 중 앞 {len(files)}개만 씀 "
                  f"(마지막: {names[len(files) - 1]}, 다음 촬영: {names[len(files)]})")
            names = names[:len(files)]
        if len(names) != len(files):
            hint = " → 아직 안 찍은 사진이 목록에 있으면 --partial" if len(names) > len(files) else ""
            sys.exit(f"개수가 다름: 원본 {len(files)}장, 이름 {len(names)}개{hint}")
        if no_exif:
            print(f"  [주의] 촬영 시각 없는 사진 {len(no_exif)}장은 파일 이름 순으로 맨 뒤: {no_exif[:5]}")
    else:
        names = [p.stem + ".jpg" for p in files]

    out = Path(args.out)
    pairs = list(zip(files, names))
    print(f"{'원본':40s} → 새 이름   (촬영 시각)")
    for p, n in pairs:
        k = shot_key(p)
        print(f"{p.name:40s} → {n:14s} {k[1] or '-'}")

    if not args.apply:
        print(f"\n미리보기만 했습니다. 순서가 맞으면 --apply 를 붙여 다시 실행하세요. ({len(pairs)}장)")
        return

    out.mkdir(parents=True, exist_ok=True)
    clash = [n for _, n in pairs if (out / n).exists()]
    if clash and not args.force:
        sys.exit(f"{out}에 이미 있음 {len(clash)}개(예: {clash[:3]}). 덮어쓰려면 --force")

    done, notes = [], []
    for p, n in pairs:
        size, note = fix(p, out / n, args.quality)
        done.append(out / n)
        if note:
            notes.append(f"{n}: {note} {size}")
    sheet = out.parent / "contact.jpg"
    contact_sheet(done, sheet, thumb=args.thumb, cols=6 if args.thumb >= 400 else 10)
    print(f"\n{len(done)}장 → {out}")
    for s in notes:
        print("  [주의]", s)
    print(f"확인용 썸네일 → {sheet}  (모든 트레이가 세로이고 아침 줄이 위인지 보기)")


if __name__ == "__main__":
    main()
