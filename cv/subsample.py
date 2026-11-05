"""약품별 AI Hub 이미지를 촬영 조건이 고르게 남도록 N장으로 줄인다.

라벨 JSON의 촬영 조건(앞뒷면, 배경색, 조명색, 카메라 위도)으로 그룹을 나누고,
그룹마다 비슷한 장수를 배정한 뒤 그룹 안에서는 카메라 경도가 고르게 퍼지도록 고른다.
고르지 않은 이미지는 지우지 않고 data/pills/unused/<K코드>/로 옮긴다 (되돌릴 수 있음).

사용 예:
    python cv/subsample.py K-004378 --n 324            # 분포와 선택 계획만 출력 (기본, 파일 안 건드림)
    python cv/subsample.py K-004378 --n 324 --apply    # 실제로 옮김
    python cv/subsample.py K-004378 --n 324 --min-la 70 --apply   # 카메라 위도 70 이상만 사용
"""
import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict

from convert_aihub import DATA, IMG_DIR, LBL_DIR, find_image, find_tl_zip

UNUSED = DATA / "pills" / "unused"
GROUP_KEYS = ("drug_dir", "back_color", "light_color", "camera_la")


def load_meta(kcode):
    items = []
    with zipfile.ZipFile(find_tl_zip(kcode)) as z:
        for n in z.namelist():
            if kcode not in n or not n.endswith(".json"):
                continue
            info = json.loads(z.read(n))["images"][0]
            img = find_image(kcode, info["file_name"])
            if img is None:
                continue
            items.append({"img": img, **{k: info.get(k) for k in GROUP_KEYS},
                          "camera_lo": int(info.get("camera_lo") or 0)})
    return items


def allocate(groups, n):
    """작은 그룹부터 남은 장수를 그룹 수로 나눠 배정 (그룹이 작으면 전부, 남는 몫은 다음 그룹으로)."""
    quota, remaining = {}, n
    order = sorted(groups, key=lambda g: len(groups[g]))
    for i, g in enumerate(order):
        q = min(len(groups[g]), remaining // (len(order) - i))
        quota[g] = q
        remaining -= q
    return quota


def pick_spread(items, q):
    """카메라 경도 순으로 정렬해 q장을 같은 간격으로 고른다."""
    items = sorted(items, key=lambda r: r["camera_lo"])
    return [items[int((j + 0.5) * len(items) / q)] for j in range(q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kcode")
    ap.add_argument("--n", type=int, default=324, help="남길 장수")
    ap.add_argument("--min-la", type=int, default=0, help="이 값 미만의 카메라 위도는 제외")
    ap.add_argument("--apply", action="store_true", help="실제로 옮김 (없으면 계획만 출력)")
    args = ap.parse_args()

    items = load_meta(args.kcode)
    print(f"[{args.kcode}] 라벨과 짝이 맞는 이미지 {len(items)}장")
    for k in GROUP_KEYS + ("camera_lo",):
        dist = Counter(r[k] for r in items)
        print(f"  {k}: " + ", ".join(f"{v}={c}" for v, c in sorted(dist.items(), key=lambda x: str(x[0]))))

    pool = [r for r in items if int(r["camera_la"] or 0) >= args.min_la]
    if args.min_la:
        print(f"  위도 {args.min_la} 이상: {len(pool)}장")
    if len(pool) <= args.n:
        print(f"  남길 장수({args.n}) 이상이 아니므로 줄일 필요 없음")
        return

    groups = defaultdict(list)
    for r in pool:
        groups[tuple(r[k] for k in GROUP_KEYS)].append(r)
    quota = allocate(groups, args.n)
    keep = [r for g, q in quota.items() if q for r in pick_spread(groups[g], q)]
    keep_set = {r["img"] for r in keep}

    print(f"  그룹 {len(groups)}개, 선택 {len(keep)}장 (그룹당 {min(quota.values())}~{max(quota.values())}장)")
    for k in GROUP_KEYS:
        print(f"  선택 후 {k}: " + ", ".join(
            f"{v}={c}" for v, c in sorted(Counter(r[k] for r in keep).items(), key=lambda x: str(x[0]))))

    all_imgs = [p for p in (IMG_DIR / args.kcode).iterdir() if p.is_file()]
    move = [p for p in all_imgs if p not in keep_set]
    print(f"  옮길 이미지 {len(move)}장 → {UNUSED / args.kcode}")
    if not args.apply:
        print("  (계획만 출력함. 실제로 옮기려면 --apply)")
        return

    for sub in ("images", "labels"):
        (UNUSED / args.kcode / sub).mkdir(parents=True, exist_ok=True)
    for p in move:
        shutil.move(str(p), UNUSED / args.kcode / "images" / p.name)
        lbl = LBL_DIR / args.kcode / f"{p.stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), UNUSED / args.kcode / "labels" / lbl.name)
    print(f"  완료: 남은 이미지 {len(list((IMG_DIR / args.kcode).iterdir()))}장")


if __name__ == "__main__":
    main()
