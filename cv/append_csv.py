"""추가 촬영 기록을 기존 photos.csv · cells.csv 끝에 덧붙인다 (열 이름 기준, 중복은 거절).

추가분 CSV는 열이 기존보다 적을 수 있다 (예: photos_add는 file,group만, cells_add는 group,day,time,actual만).
기존 파일의 열 순서대로 맞추고, 추가분에 없는 열은 --default 값 또는 --copy로 다른 열 값을 채우고, 그 밖에는 빈 칸으로 둔다.
기존 파일에 없는 열이 추가분에 있으면 멈춘다 (오타일 수 있으므로). 같은 키가 이미 있으면 아무것도 쓰지 않고 멈춘다.

사용 예 (먼저 --dry-run으로 확인)
  python cv/append_csv.py data/tray/photos.csv photos_add_1112.csv --key file --dry-run
  python cv/append_csv.py data/tray/photos.csv photos_add_1112.csv --key file
  python cv/append_csv.py data/tray/cells.csv cells_add_1112.csv --key group,day,time \\
      --copy planned=actual --default error=none
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def pairs(items, what):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what}는 열=값 형식: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v
    return out


def merge(base_cols, base_rows, add_cols, add_rows, key, defaults, copies):
    """→ 덧붙일 행 목록. 문제가 있으면 SystemExit."""
    unknown = [c for c in add_cols if c not in base_cols]
    if unknown:
        raise SystemExit(f"기존 파일에 없는 열: {', '.join(unknown)} (기존 열: {', '.join(base_cols)})")
    for k in key:
        if k not in base_cols or k not in add_cols:
            raise SystemExit(f"키 열 {k}가 두 파일 모두에 있어야 함")
    for c in list(defaults) + list(copies):
        if c not in base_cols:
            raise SystemExit(f"--default/--copy의 열 {c}가 기존 파일에 없음")
    for src in copies.values():
        if src not in add_cols:
            raise SystemExit(f"--copy 원본 열 {src}가 추가분에 없음")

    seen = {tuple(r[k] for k in key) for r in base_rows}
    dup, new_keys, out = [], set(), []
    for r in add_rows:
        kv = tuple((r.get(k) or "").strip() for k in key)
        if not any(kv):
            continue                                    # 빈 줄
        if kv in seen or kv in new_keys:
            dup.append(",".join(kv))
            continue
        new_keys.add(kv)
        row = {}
        for c in base_cols:
            if c in add_cols and (r.get(c) or "") != "":
                row[c] = r[c]
            elif c in copies:
                row[c] = r.get(copies[c]) or ""
            else:
                row[c] = defaults.get(c, r.get(c) or "")
        out.append(row)
    if dup:
        raise SystemExit(f"이미 있는 키 {len(dup)}개 → 아무것도 쓰지 않음: {' '.join(dup[:5])}")
    return out


def main():
    ap = argparse.ArgumentParser(description="CSV 덧붙이기 (열 이름 기준)")
    ap.add_argument("base", help="기존 CSV (덧붙여질 파일)")
    ap.add_argument("add", help="추가분 CSV")
    ap.add_argument("--key", required=True, help="중복 판단 열 (쉼표로 여러 개)")
    ap.add_argument("--default", action="append", default=[], help="추가분에 없는 열의 기본값, 열=값")
    ap.add_argument("--copy", action="append", default=[], help="추가분에 없는 열을 다른 열 값으로, 대상=원본")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base, add = Path(args.base), Path(args.add)
    base_cols, base_rows = read(base)
    add_cols, add_rows = read(add)
    key = [k.strip() for k in args.key.split(",") if k.strip()]
    rows = merge(base_cols, base_rows, add_cols, add_rows, key, pairs(args.default, "--default"),
                 pairs(args.copy, "--copy"))

    print(f"{base}: 기존 {len(base_rows)}줄 + 추가 {len(rows)}줄 = {len(base_rows) + len(rows)}줄")
    for r in rows[:3]:
        print("  " + ",".join(r[c] for c in base_cols))
    if len(rows) > 3:
        print(f"  … (마지막) " + ",".join(rows[-1][c] for c in base_cols))
    if args.dry_run:
        print("--dry-run: 쓰지 않음")
        return

    backup = base.with_name(base.name + ".bak")
    shutil.copy2(base, backup)
    text = base.read_bytes()
    with open(base, "a", encoding="utf-8", newline="") as f:
        if text and not text.endswith(b"\n"):
            f.write("\n")
        w = csv.DictWriter(f, fieldnames=base_cols, lineterminator="\n")
        w.writerows(rows)
    print(f"→ 썼음 (백업 {backup.name})")


if __name__ == "__main__":
    sys.exit(main())
