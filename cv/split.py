"""학습·검증·테스트 분할과 ultralytics data yaml 생성.

AI Hub 이미지는 약품별로 무작위 분할하고, 실제 트레이 사진은 배치 그룹 단위로 나눈다
(같은 배치를 찍은 사진이 학습과 테스트에 나뉘어 들어가지 않게).
경로가 PC마다 달라서 yaml과 목록 파일은 저장소가 아니라 data/splits/에 만든다.
RunPod 등 다른 PC에서는 이 스크립트를 다시 돌리면 된다(시드가 같으면 결과도 같다).

입력
  data/pills/images/<K코드>/*.png       AI Hub 이미지 (labels/<K코드>/*.txt와 짝)
  data/tray/images/*.jpg                보정된 트레이 사진 (g07_s03.jpg, empty_s01.jpg)
  data/tray/photos.csv                  file,group,tilt,light,note
  data/tray/cells.csv                   group,day,time,planned,actual,error,flags

그룹 배정은 한 번 정하면 고정한다 (11/12)
  data/splits/tray_groups.csv (group,split)에 그룹별 배정을 적어 두고, 다음 실행부터는 이 파일을 그대로 따른다.
  - 파일이 없고 예전 tray_*.txt가 있으면 그 목록에서 배정을 읽어 파일을 만든다 (11/12 이전 분할을 그대로 이어 가기)
  - 둘 다 없으면 조건 탐색으로 새로 나누고 파일을 만든다
  - 파일에 없는 새 그룹(추가 촬영)은 --new-split(기본 train)으로 보내고 파일 끝에 덧붙인다
  - 분할을 처음부터 다시 하려면 --reshuffle (test가 바뀌므로 최종 평가 전에는 쓰지 않는다)
  AI Hub 분할은 같은 rng를 먼저 쓰므로 AI Hub 장수가 바뀌면 탐색 결과도 바뀐다. 그래서 탐색 결과를 파일로 고정한다.

출력 (data/splits/)
  aihub_train.txt / aihub_val.txt
  tray_groups.csv  그룹 → train · val · test (고정 기록)
  tray_train.txt / tray_val.txt / tray_test.txt
  data_base.yaml   기준 모델: AI Hub만 (11/9)
  data.yaml        본 학습: AI Hub 학습분 + 트레이 학습분, 검증·테스트는 트레이 사진만
  split_report.txt 그룹 배정 결과

사용 예
  python cv/split.py              # AI Hub만 있으면 data_base.yaml까지만 만든다
  python cv/split.py --seed 1 --reshuffle   # 그룹 배정을 새로 탐색 (tray_groups.csv를 덮어씀)
"""
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".png", ".jpg", ".jpeg")
TARGET = {"train": 50, "val": 20, "test": 30}   # 트레이 사진 장수 목표
N_GROUPS = {"train": 10, "val": 4, "test": 6}   # 20그룹 기준
EMPTY_VAL = 4                                   # 빈 트레이 20장 중 검증으로 보낼 장수(나머지는 학습)
SPLITS = ("train", "val", "test")
GROUP_OF = re.compile(r"^(g\d+)_s\d+$")         # g07_s03 → g07


def load_classes():
    cfg = yaml.safe_load((ROOT / "cv" / "classes.yaml").read_text(encoding="utf-8"))
    return sorted(cfg["classes"], key=lambda c: c["id"])


def parse_counts(s):
    """'0:1;6:1' → {0: 1, 6: 1}"""
    out = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if part:
            k, v = part.split(":")
            out[int(k)] = int(v)
    return out


# ---------- AI Hub ----------

def split_aihub(data, classes, val_ratio, rng):
    train, val = [], []
    for c in classes:
        img_dir = data / "pills" / "images" / c["k_code"]
        lbl_dir = data / "pills" / "labels" / c["k_code"]
        imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS
                      and (lbl_dir / f"{p.stem}.txt").exists()) if img_dir.exists() else []
        if not imgs:
            print(f"  [AI Hub] {c['id']} {c['name']}: 라벨 있는 이미지 없음 (건너뜀)")
            continue
        rng.shuffle(imgs)
        n_val = max(1, round(len(imgs) * val_ratio))
        val += imgs[:n_val]
        train += imgs[n_val:]
        print(f"  [AI Hub] {c['id']} {c['name']}: 학습 {len(imgs) - n_val} / 검증 {n_val}")
    return sorted(train), sorted(val)


# ---------- 트레이 사진 ----------

def load_tray(data):
    tray = data / "tray"
    photos_csv, cells_csv = tray / "photos.csv", tray / "cells.csv"
    if not photos_csv.exists() or not cells_csv.exists():
        return None

    with open(photos_csv, encoding="utf-8-sig") as f:
        photos = list(csv.DictReader(f))
    with open(cells_csv, encoding="utf-8-sig") as f:
        cells = list(csv.DictReader(f))

    groups = defaultdict(lambda: {"photos": [], "cells": []})
    problems = []
    for r in photos:
        # 빈 트레이는 아래 empties로 따로 나누고, 180° 사진(rot_)은 검수 rotated 평가용이라 검출 분할에서 뺀다
        if not r["group"] or r["file"].startswith("rot_"):
            continue
        p = tray / "images" / r["file"]
        if not p.exists():
            problems.append(f"사진 없음: {r['file']}")
            continue
        groups[r["group"]]["photos"].append(p)
    for r in cells:
        groups[r["group"]]["cells"].append(r)

    info = {}
    for g, d in sorted(groups.items()):
        if not d["photos"]:
            problems.append(f"{g}: cells.csv에는 있는데 사진이 없음")
            continue
        if len(d["cells"]) != 12:
            problems.append(f"{g}: 칸 기록 {len(d['cells'])}줄 (12줄이어야 함)")
        classes = set()
        for c in d["cells"]:
            classes |= set(parse_counts(c["actual"]))
        flags = {x for c in d["cells"] for x in (c.get("flags") or "").split(";") if x}
        info[g] = {
            "photos": sorted(d["photos"]),
            "errors": sum((c.get("error") or "none") != "none" for c in d["cells"]),
            "flags": flags,
            "mixed_0_6": {0, 6} <= classes,   # 타이레놀-트립라인 혼동 평가용
        }

    empties = sorted(p for p in (tray / "images").glob("empty_*") if p.suffix.lower() in IMG_EXTS)
    return info, empties, problems


def score(assign, info):
    """(조건 위반 수, 장수 목표와의 차이). 작을수록 좋다."""
    by = defaultdict(list)
    for g, s in assign.items():
        by[s].append(info[g])
    bad = 0
    test, val = by["test"], by["val"]
    bad += max(0, 3 - sum(x["errors"] > 0 for x in test))   # 테스트에 잘못 담은 칸 있는 그룹 3개 이상
    bad += max(0, 1 - sum(x["errors"] > 0 for x in val))    # 검증에도 1개 이상
    if any(x["mixed_0_6"] for x in info.values()):
        bad += 0 if any(x["mixed_0_6"] for x in test) else 1
    all_flags = set().union(*(x["flags"] for x in info.values()))
    test_flags = set().union(*(x["flags"] for x in test)) if test else set()
    bad += len(all_flags - test_flags)                       # 겹침·경계·뒤집힘이 테스트에도 있게
    dev = sum(abs(sum(len(x["photos"]) for x in by[s]) - TARGET[s]) for s in TARGET)
    return bad, dev


def split_tray(info, rng, tries=20000):
    gs = sorted(info)
    total = sum(N_GROUPS.values())
    # 그룹 수가 20이 아니면 비율대로 나눈다
    n_test = round(len(gs) * N_GROUPS["test"] / total)
    n_val = round(len(gs) * N_GROUPS["val"] / total)
    best, best_s = None, None
    for _ in range(tries):
        rng.shuffle(gs)
        assign = {g: "test" for g in gs[:n_test]}
        assign |= {g: "val" for g in gs[n_test:n_test + n_val]}
        assign |= {g: "train" for g in gs[n_test + n_val:]}
        s = score(assign, info)
        if best_s is None or s < best_s:
            best, best_s = dict(assign), s
            if s == (0, 0):
                break
    return best, best_s


# ---------- 그룹 배정 고정 ----------

def read_groups_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r.get("split") not in SPLITS]
    if bad:
        raise SystemExit(f"{path}: split 값이 train·val·test가 아님: {bad[0]}")
    return {r["group"]: r["split"] for r in rows}


def groups_from_lists(out):
    """예전 tray_{train,val,test}.txt에서 그룹 배정을 읽는다. 한 그룹이 두 곳에 있으면 멈춘다."""
    assign = {}
    for s in SPLITS:
        f = out / f"tray_{s}.txt"
        if not f.exists():
            return None
        for line in f.read_text(encoding="utf-8").split():
            m = GROUP_OF.match(Path(line).stem)
            if not m:
                continue                     # empty_ 등 그룹 없는 사진
            g = m.group(1)
            if assign.setdefault(g, s) != s:
                raise SystemExit(f"{g}가 tray_{assign[g]}.txt와 tray_{s}.txt에 함께 있음 → 목록을 확인")
    return assign or None


def write_groups_csv(path, assign):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "split"])
        for g in sorted(assign):
            w.writerow([g, assign[g]])


def fixed_assign(out, info, new_split):
    """고정 배정을 읽고(없으면 예전 목록에서 만들고) 새 그룹을 new_split으로 보낸다.
    → (배정, 출처 설명, 새로 넣은 그룹) 또는 고정 배정이 없으면 None"""
    path = out / "tray_groups.csv"
    if path.exists():
        assign, src = read_groups_csv(path), "tray_groups.csv"
    else:
        assign, src = groups_from_lists(out), "예전 tray_*.txt"
        if assign is None:
            return None
    gone = sorted(set(assign) - set(info))
    if gone:
        print(f"  [확인] 배정에는 있는데 사진이 없는 그룹: {' '.join(gone)} (배정은 그대로 둔다)")
    added = sorted(set(info) - set(assign))
    for g in added:
        assign[g] = new_split
    write_groups_csv(path, assign)
    return {g: s for g, s in assign.items() if g in info}, src, added


# ---------- 출력 ----------

def write_list(path, items):
    path.write_text("".join(f"{p.resolve()}\n" for p in items), encoding="utf-8")


def write_yaml(path, data, train, val, test, classes):
    cfg = {"path": str(data.resolve()), "train": train, "val": val}
    if test:
        cfg["test"] = test
    cfg["names"] = {c["id"]: c["name"] for c in classes}
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aihub-val", type=float, default=0.1, help="AI Hub 검증 비율")
    ap.add_argument("--tray-repeat", type=int, default=1, help="data.yaml에 트레이 학습 목록을 넣는 횟수")
    ap.add_argument("--new-split", choices=SPLITS, default="train", help="고정 배정에 없는 새 그룹을 보낼 곳")
    ap.add_argument("--reshuffle", action="store_true", help="고정 배정을 무시하고 그룹 배정을 새로 탐색")
    args = ap.parse_args()

    data = Path(args.data)
    out = data / "splits"
    out.mkdir(parents=True, exist_ok=True)
    classes = load_classes()
    rng = random.Random(args.seed)

    print("AI Hub")
    a_train, a_val = split_aihub(data, classes, args.aihub_val, rng)
    write_list(out / "aihub_train.txt", a_train)
    write_list(out / "aihub_val.txt", a_val)
    write_yaml(out / "data_base.yaml", data, "splits/aihub_train.txt", "splits/aihub_val.txt", None, classes)
    print(f"  → data_base.yaml (학습 {len(a_train)} / 검증 {len(a_val)})")

    loaded = load_tray(data)
    if loaded is None:
        print("트레이: data/tray/photos.csv·cells.csv 없음 → data.yaml은 촬영 후에 만든다")
        return
    info, empties, problems = loaded
    for p in problems:
        print(f"  [확인] {p}")
    if not info:
        print("트레이: 사용할 그룹 없음")
        return

    fixed = None if args.reshuffle else fixed_assign(out, info, args.new_split)
    if fixed is None:
        assign, (bad, dev) = split_tray(info, rng)
        write_groups_csv(out / "tray_groups.csv", assign)
        origin = f"조건 탐색 (seed {args.seed}, 조건 위반 {bad}, 장수 목표 차이 {dev})"
    else:
        assign, src, added = fixed
        bad = 0
        origin = f"고정 배정 ({src})" + (f", 새 그룹 {' '.join(added)} → {args.new_split}" if added else "")
    lists = {"train": [], "val": [], "test": []}
    for g in sorted(assign):
        lists[assign[g]] += info[g]["photos"]
    n_group_photos = {s: len(v) for s, v in lists.items()}
    lists["val"] += empties[:EMPTY_VAL]
    lists["train"] += empties[EMPTY_VAL:]
    for s, items in lists.items():
        write_list(out / f"tray_{s}.txt", sorted(items))

    write_yaml(out / "data.yaml", data,
               ["splits/aihub_train.txt"] + ["splits/tray_train.txt"] * args.tray_repeat,
               "splits/tray_val.txt", "splits/tray_test.txt", classes)

    lines = [origin]
    for s in ("train", "val", "test"):
        gs = sorted(g for g in assign if assign[g] == s)
        n_err = sum(info[g]["errors"] > 0 for g in gs)
        flags = sorted(set().union(*(info[g]["flags"] for g in gs))) if gs else []
        lines.append(f"{s:5s} 그룹 {len(gs):2d}개, 사진 {n_group_photos[s]:3d}장 "
                     f"(오류 칸 있는 그룹 {n_err}, flags {','.join(flags) or '-'}): {' '.join(gs)}")
    lines.append(f"빈 트레이: 학습 {len(empties[EMPTY_VAL:])} / 검증 {len(empties[:EMPTY_VAL])}")
    report = "\n".join(lines)
    (out / "split_report.txt").write_text(report + "\n", encoding="utf-8")
    print("트레이\n  " + report.replace("\n", "\n  "))
    if bad:
        print("  [경고] 조건을 다 만족하는 분할이 없음. 오류 칸·flags가 있는 그룹이 너무 적은지 확인")
    print(f"  → data.yaml, tray_groups.csv")


if __name__ == "__main__":
    main()
