"""OCR 단계 측정: 처방 약 줄마다 품목코드·1회량·횟수·일수를 OCR이 제대로 읽었는지 (REQ-CV-01 기준 변경안, 11/13 합의).

약품명 CER 대신, 시스템이 실제로 쓰는 값(품목코드로 약 식별, 숫자 셋으로 조제 계획)을 OCR 출력에서 바로 센다.
Agent 구조화(F-AG-02)를 거치지 않으므로 OCR만의 성능이다. 약품명 CER은 참고 지표로 함께 낸다.

판정 (정답 ground_truth.json의 약 줄마다)
  품목코드  OCR 줄 안의 숫자 덩어리 중 하나가 정답 9자리와 같으면 맞음 (OCR 숫자 보정 없이 그대로)
  같은 행   품목코드가 든 줄(없으면 약품명이 가장 비슷한 줄)을 기준으로, 세로로 겹치는 오른쪽 줄들을 그 약의 행으로 본다
  1회량     그 행에서 '숫자 [단위]'만 있는 줄 (예: "1 TAB", "2", "0.5 정")의 숫자 = 정답 dose
  횟수      그 행에서 '숫자회' = 정답 times
  일수      그 행에서 '숫자일' = 정답 days
  약품명 CER  기준 줄에서 코드·번호를 뗀 글자와 정답 약품명의 편집 거리 / 정답 길이 (공백 무시, 참고)

사용 예 (저장소 루트)
  python cv/ocr_eval.py --images data/rx_photo --gt data/rx_photo/ground_truth.json --out data/ocr_eval_photo
  python cv/ocr_eval.py --images agent/resources/samples/rx8_img --gt agent/resources/samples/rx8_img/ground_truth.json --out data/ocr_eval_clean
  python cv/ocr_eval.py --a1 data/ocr_eval_photo/a1 --gt data/rx_photo/ground_truth.json      # 저장한 A.1로 다시 채점만
  옵션: --no-flatten (펴기 끔), --max-side 1800 (OCR 입력 크기 비교)

출력 (--out)
  a1/<이름>.json   A.1 결과 (다시 채점할 때 --a1로 씀)
  eval.csv         약 줄 하나당 한 줄 (사진, 약, 코드·1회량·횟수·일수 맞음 여부, 약품명 CER, 펴기 상태)
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTS = {".jpg", ".jpeg", ".png"}
TARGET = {"code": 0.95, "numbers": 0.90}          # 잠정 수용 기준 (깨끗한 원본). 촬영본은 둘 다 0.85 잠정
DOSE_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(TAB|CAP|T|C|정|캡슐|캅셀|ml|mL|g)?\s*$", re.I)
TIMES_RE = re.compile(r"(\d+)\s*회")
DAYS_RE = re.compile(r"(\d+)\s*일")
CODE_PREFIX_RE = re.compile(r"^\s*(\(\d+\))?\s*\d{6,}\s*")


# ---------------------------------------------------------------- 채점
def squash(s):
    return "".join(str(s).split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def digit_runs(text):
    return re.findall(r"\d+", text)


def _yc(l):
    return (l["bbox"][1] + l["bbox"][3]) / 2


def _same_row(anchor, line):
    """line의 세로 중심이 anchor 줄 높이(위아래 60% 여유) 안에 있고 anchor보다 오른쪽에서 시작"""
    y1, y2 = anchor["bbox"][1], anchor["bbox"][3]
    h = y2 - y1
    return y1 - 0.6 * h <= _yc(line) <= y2 + 0.6 * h and line["bbox"][0] > anchor["bbox"][0] and line is not anchor


def find_anchor(lines, drug):
    """→ (기준 줄, 코드 맞음 여부). 코드가 든 줄이 우선, 없으면 약품명이 가장 비슷한 줄(CER 0.5 이하)"""
    for l in lines:
        if drug["code"] in digit_runs(l["text"]):
            return l, True
    name = squash(drug["name"])
    best, best_cer = None, 1.0
    for l in lines:
        t = squash(CODE_PREFIX_RE.sub("", l["text"]))
        if not t:
            continue
        # 줄에 약품명이 섞여 있을 수 있어 정답 길이만큼 잘라 비교
        cer = min(edit_distance(name, t[i:i + len(name)]) for i in range(max(1, len(t) - len(name) + 1))) / len(name)
        if cer < best_cer:
            best, best_cer = l, cer
    return (best if best_cer <= 0.5 else None), False


def read_numbers(lines, anchor):
    """기준 줄과 같은 행의 줄들에서 (1회량, 횟수, 일수) — 못 읽으면 None"""
    row = [l for l in lines if _same_row(anchor, l)]
    dose = times = days = None
    for l in sorted(row, key=lambda l: l["bbox"][0]):
        t = l["text"]
        if times is None and (m := TIMES_RE.search(t)):
            times = int(m.group(1))
            continue
        if days is None and (m := DAYS_RE.search(t)):
            days = int(m.group(1))
            continue
        if dose is None and (m := DOSE_RE.match(t)):
            dose = float(m.group(1).replace(",", "."))
    return dose, times, days


def name_cer(anchor, drug):
    if anchor is None:
        return 1.0
    name = squash(drug["name"])
    t = squash(CODE_PREFIX_RE.sub("", anchor["text"]))
    return round(min(edit_distance(name, t), len(name)) / len(name), 3)


def score_image(a1, gt_drugs):
    rows = []
    for d in gt_drugs:
        anchor, code_ok = find_anchor(a1["lines"], d)
        dose, times, days = read_numbers(a1["lines"], anchor) if anchor else (None, None, None)
        rows.append({
            "drug": d["name"], "code": d["code"],
            "code_ok": code_ok,
            "dose_ok": dose is not None and abs(dose - float(d["dose"])) < 1e-6,
            "times_ok": times == int(d["times"]),
            "days_ok": days == int(d["days"]),
            "read": f"{dose if dose is not None else '-'}/{times or '-'}/{days or '-'}",
            "name_cer": name_cer(anchor, d),
        })
    for r in rows:
        r["all_ok"] = r["code_ok"] and r["dose_ok"] and r["times_ok"] and r["days_ok"]
    return rows


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    code = sum(r["code_ok"] for r in rows)
    nums = sum(r["dose_ok"] + r["times_ok"] + r["days_ok"] for r in rows)
    return {
        "lines": n,
        "code": code / n, "code_n": code,
        "numbers": nums / (3 * n), "numbers_n": nums,
        "all": sum(r["all_ok"] for r in rows) / n, "all_n": sum(r["all_ok"] for r in rows),
        "name_cer": sum(r["name_cer"] for r in rows) / n,
    }


# ---------------------------------------------------------------- 실행
def gt_for(gt, path):
    """정답 키는 파일 이름(rx8_01.png) — 확장자·촬영본 접두어(rx8p_)가 달라도 찾는다"""
    stem = Path(path).stem
    by_stem = {Path(k).stem: v for k, v in gt.items()}
    return by_stem.get(stem) or by_stem.get(stem.replace("rx8p_", "rx8_"))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, help="처방전 이미지 폴더 (OCR을 돌린다)")
    src.add_argument("--a1", type=Path, help="저장해 둔 A.1 json 폴더 (채점만)")
    ap.add_argument("--gt", type=Path, required=True, help="ground_truth.json (약 줄마다 code·name·dose·times·days)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ocr_eval")
    ap.add_argument("--no-flatten", action="store_true", help="처방전 펴기를 끄고 원본으로 OCR")
    ap.add_argument("--max-side", type=int, default=None, help="OCR 입력 긴 변 (기본: cv/ocr.py MAX_SIDE)")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, secs = [], []
    if args.images:
        from cv.ocr import MAX_SIDE, get_reader, read_document
        from cv.stub import OcrEmpty
        (args.out / "a1").mkdir(exist_ok=True)
        t0 = time.time()
        reader = get_reader()
        print(f"EasyOCR 로딩 {time.time() - t0:.1f}s")
        files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in EXTS)
        items = []
        for p in files:
            g = gt_for(gt, p)
            if g is None:
                continue
            t = time.time()
            try:
                a1, _, info = read_document(p, flatten=not args.no_flatten,
                                            max_side=args.max_side or MAX_SIDE, reader=reader)
            except OcrEmpty:
                a1, info = {"image_id": p.stem, "lines": []}, {"status": "empty", "edge": ""}
            secs.append(time.time() - t)
            (args.out / "a1" / f"{p.stem}.json").write_text(json.dumps(a1, ensure_ascii=False, indent=1), encoding="utf-8")
            items.append((p, a1, g, info, secs[-1]))
    else:
        items = []
        for f in sorted(args.a1.glob("*.json")):
            g = gt_for(gt, f)
            if g is not None:
                items.append((f, json.loads(f.read_text(encoding="utf-8")), g, {"status": "-", "edge": ""}, None))

    if not items:
        raise SystemExit("정답과 이름이 맞는 이미지가 없음 (정답 키는 파일 이름, 예: rx8_01.png)")
    for p, a1, g, info, sec in items:
        r = score_image(a1, g["drugs"])
        for x in r:
            x.update(image=p.stem, flatten=info["status"], paper_cut=info.get("edge", ""))
        rows += r
        s = summarize(r)
        print(f"{p.stem}: 코드 {s['code_n']}/{s['lines']}  숫자 {s['numbers_n']}/{3 * s['lines']}  "
              f"줄 전체 {s['all_n']}/{s['lines']}  펴기 {info['status']}"
              + ("  [종이 잘림]" if info.get("edge") else "") + (f"  {sec:.1f}s" if sec else ""))

    with open(args.out / "eval.csv", "w", encoding="utf-8-sig", newline="") as fh:
        cols = ["image", "drug", "code", "code_ok", "dose_ok", "times_ok", "days_ok", "all_ok", "read",
                "name_cer", "flatten", "paper_cut"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    s = summarize(rows)
    print(f"\n== OCR 단계 측정: 사진 {len(items)}장, 약 줄 {s['lines']}개 ==")
    print(f"품목코드     {s['code_n']:3d}/{s['lines']}  {s['code']:.1%}   (잠정 기준 {TARGET['code']:.0%}, 촬영본 85%)")
    print(f"숫자 항목    {s['numbers_n']:3d}/{3 * s['lines']}  {s['numbers']:.1%}   (1회량·횟수·일수, 잠정 기준 {TARGET['numbers']:.0%}, 촬영본 85%)")
    for k, label in (("dose_ok", "1회량"), ("times_ok", "횟수"), ("days_ok", "일수")):
        print(f"  {label:5s} {sum(r[k] for r in rows):3d}/{s['lines']}")
    print(f"줄 전체 정답 {s['all_n']:3d}/{s['lines']}  {s['all']:.1%}   (코드 + 숫자 셋 모두)")
    print(f"약품명 CER   평균 {s['name_cer']:.3f}   (참고)")
    if secs:
        print(f"처리 시간    사진당 평균 {sum(secs) / len(secs):.1f}s · 최대 {max(secs):.1f}s (CPU)")
    print(f"→ {args.out / 'eval.csv'}")


if __name__ == "__main__":
    main()
