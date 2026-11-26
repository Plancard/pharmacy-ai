"""rx8 가상 처방전 사진 20장의 구조화·약품 매칭 정확도 측정 (REQ-AG-02 · 03 수용 기준) — 손으로 돌려 보는 스크립트.

    python scripts/agent/measure_rx8_photos.py 사진폴더                 # 규칙 경로만 (규칙이 못 읽으면 실패로 센다)
    python scripts/agent/measure_rx8_photos.py 사진폴더 --llm           # 서비스와 같이 규칙 → 실패하면 LLM 경로 (OpenAI를 부를 수 있다)
    python scripts/agent/measure_rx8_photos.py 사진폴더 --llm --out var/accuracy/rx8_photos_1126.json
    python scripts/agent/measure_rx8_photos.py --a1-dir agent/resources/samples/rx8_a1   # OCR 없이 A.1 JSON으로 (스크립트 점검용, 100%가 나와야 한다)

pytest가 아니다. rx8 처방전(agent/resources/samples/rx8_img)을 인쇄해 찍은 사진을 실제 OCR(cv/ocr.py, EasyOCR)로 읽고,
서비스와 같은 구조화 → 약품 매칭 → 위험 신호 대조(DUR API 없이)를 거쳐 rx8_expected 정답과 견준다.
measure_accuracy.py는 CV가 OCR해 둔 cv_a1 5장을 재고, 이 스크립트는 명세서 수용 기준의 20장을 사진부터 잰다.

사진 이름의 마지막 숫자로 정답을 고른다: prescription_00101.jpg → rx8_01, …, 00120 → rx8_20 (rx8_07.png 같은 이름도 된다).
사진은 개인 촬영본이라 저장소에 넣지 않는다 (var/rx8_photos 등에 두고 경로를 준다). var/ 는 git에 올라가지 않는다.

채점
- 약품명은 매칭을 거친 조제지시서의 이름으로 잰다 (대상 약은 표준 이름, 대상 외 약은 OCR이 읽은 글자).
  OCR이 읽은 글자 그대로의 일치율은 참고 줄로 따로 보인다 (OCR 품질, REQ-CV-01).
- 1회량 · 1일 횟수(복용 시점 수) · 투약 일수, 대상 약 매칭(품목기준코드로 확정), A.5 형식 검증, 조제 계획 판정을 본다.
- 위험 단계는 DUR API 없이 성분 중복만 대조한 결과라 참고로만 보인다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/measure_rx8_photos.py` 로 돌려도 agent 패키지를 찾게 한다

from agent.config import RESOURCES_DIR  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.drug_match import match_order  # noqa: E402
from agent.errors import ConfigError, StructureError  # noqa: E402
from agent.pipeline import structure_prescription  # noqa: E402
from agent.plan import PlanError, build_plan  # noqa: E402
from agent.risk import check_order  # noqa: E402
from agent.schema import DispensingOrder  # noqa: E402

EXPECTED_DIR = RESOURCES_DIR / "samples" / "rx8_expected"
FIELDS = ("name", "count_per_dose", "times_per_day", "days")
LABELS = {"name": "약품명", "count_per_dose": "1회량", "times_per_day": "1일 횟수", "days": "투약 일수"}
TARGET_RATE = 0.9
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")

OcrFn = Callable[[Path, str], dict[str, Any]]
LlmFn = Callable[..., dict[str, Any]]


def rx_id_of(path: Path) -> str | None:
    """사진 이름의 마지막 숫자 → rx8_NN (끝 두 자리, 1~20). 고를 수 없으면 None."""
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    n = int(nums[-1]) % 100
    return f"rx8_{n:02d}" if 1 <= n <= 20 else None


def squash(text: str | None) -> str:
    return "".join((text or "").split())


def no_llm(text: str, feedback: str | None = None) -> dict[str, Any]:
    raise StructureError("규칙 경로만 재는 중이라 LLM을 부르지 않았습니다.")


def real_ocr(path: Path, rx_id: str) -> dict[str, Any]:
    from cv.ocr import read_document     # EasyOCR · torch는 사진을 잴 때만 읽는다
    from cv.stub import OcrEmpty
    try:
        a1, _img, _rect = read_document(path, image_id=rx_id)
    except OcrEmpty as e:                # 글자를 하나도 못 찾은 사진은 그 장만 실패로 센다
        raise StructureError(f"OCR이 글자를 찾지 못했습니다: {e}", code="E_OCR_EMPTY") from e
    return a1


def item_view(d) -> dict[str, Any]:
    return {"name": squash(d.name), "ocr_name": squash(d.printed_name or d.name), "count_per_dose": d.count_per_dose,
            "times_per_day": len(d.times), "days": d.days, "item_code": d.item_code, "match_method": d.match_method}


def expected_view(item: dict[str, Any]) -> dict[str, Any]:
    return {"name": squash(item["name"]), "count_per_dose": item["count_per_dose"], "times_per_day": len(item["times"]),
            "days": item["days"], "item_code": item["item_code"], "target": item["target"]}


def plan_result(order: DispensingOrder) -> str:
    try:
        build_plan(order)
        return "ok"
    except PlanError as e:
        return "E_PLAN_RANGE" if e.code else "unmatched"


def score(rx_id: str, a1: dict[str, Any], dictionary, llm_fn: LlmFn) -> dict[str, Any]:
    """처방전 한 장을 서비스와 같이 처리해 정답과 견준다. 구조화에 실패하면 error만 채운다."""
    expected = json.loads((EXPECTED_DIR / f"{rx_id}.json").read_text(encoding="utf-8"))
    exp = [expected_view(i) for i in expected["items"]]
    row: dict[str, Any] = {"rx_id": rx_id, "expected": exp, "wrong": []}
    try:
        order = structure_prescription(f"P-{rx_id}", a1, llm_fn=llm_fn)
    except StructureError as e:
        row["error"] = f"{e.code}: {e}"
        return row
    order = check_order(match_order(order, dictionary, api=None), dictionary)
    try:
        DispensingOrder.model_validate(order.model_dump())
        row["schema_ok"] = True
    except ValueError as e:
        row["schema_ok"] = False
        row["wrong"].append(f"A.5 형식 검증 실패: {e}")
    got = [item_view(d) for d in order.items]
    row.update(got=got, source=order.source, warnings=list(order.warnings),
               risk=order.risk_check.level if order.risk_check else None, plan=plan_result(order))
    row["hits"] = {f: 0 for f in FIELDS}
    row["ocr_name_hits"] = row["target_ok"] = 0
    for i, e in enumerate(exp, start=1):
        g = got[i - 1] if i <= len(got) else None
        if g is not None and g["ocr_name"] == e["name"]:
            row["ocr_name_hits"] += 1
        for f in FIELDS:
            if g is not None and g[f] == e[f]:
                row["hits"][f] += 1
            else:
                row["wrong"].append(f"{i}번 {LABELS[f]} 정답 {e[f]!r} 결과 {None if g is None else g[f]!r}")
        if not e["target"]:
            continue
        if g is not None and g["match_method"] == "code" and g["item_code"] == e["item_code"]:
            row["target_ok"] += 1
        else:
            row["wrong"].append(f"{i}번 매칭 정답 {e['item_code']} 결과 {None if g is None else (g['item_code'], g['match_method'])}")
    if len(got) != len(exp):
        row["wrong"].append(f"약품 줄 수 정답 {len(exp)} 결과 {len(got)}")
    if row["plan"] != expected["plan"]:
        row["wrong"].append(f"조제 계획 정답 {expected['plan']} 결과 {row['plan']}")
    row["risk_ok"] = row["risk"] == expected["risk_level"]
    return row


def summarize(rows: list[dict[str, Any]], dictionary) -> dict[str, Any]:
    done = [r for r in rows if "error" not in r]
    lines = sum(len(r["expected"]) for r in rows)
    hits = {f: sum(r["hits"][f] for r in done) for f in FIELDS}
    codes_ok = {e["item_code"] for r in done for e, g in zip(r["expected"], r["got"])
                if e["target"] and g["match_method"] == "code" and g["item_code"] == e["item_code"]}
    all_codes = {e.item_code for e in dictionary.entries}
    rates = {f: (hits[f] / lines if lines else 0.0) for f in FIELDS}
    return {
        "sheets": len(rows), "lines": lines, "hits": hits, "rates": rates,
        "total_rate": sum(hits.values()) / (lines * len(FIELDS)) if lines else 0.0,
        "passed": bool(lines) and min(rates.values()) >= TARGET_RATE,
        "ocr_name_hits": sum(r["ocr_name_hits"] for r in done),
        "target_lines": sum(e["target"] for r in rows for e in r["expected"]),
        "target_ok": sum(r["target_ok"] for r in done),
        "codes_ok": sorted(codes_ok), "codes_missing": sorted(all_codes - codes_ok),
        "schema_ok": sum(bool(r.get("schema_ok")) for r in done), "failed": len(rows) - len(done),
        "rule": sum(r["source"] == "rule" for r in done), "llm": sum(r["source"] == "llm" for r in done),
        "exact": sum(not r["wrong"] for r in done),
        "plan_ok": sum(not any(w.startswith("조제 계획") for w in r["wrong"]) for r in done),
        "risk_ok": sum(r["risk_ok"] for r in done),
    }


def report(s: dict[str, Any], seconds: list[float], *, photos: bool, llm: bool, out: Callable[[str], None]) -> None:
    lines, n = s["lines"], s["sheets"]
    done = n - s["failed"]
    out("")
    out(f"== rx8 처방전 {n}장 · 약품 {lines}줄 ({'사진 → 실제 OCR' if photos else 'A.1 JSON'}, {'규칙→LLM' if llm else '규칙 경로만'}) ==")
    out("  " + " · ".join(f"{LABELS[f]} {s['rates'][f]:.1%} ({s['hits'][f]}/{lines})" for f in FIELDS))
    if lines:
        out(f"  (참고) OCR이 읽은 약품명 글자 그대로 일치 {s['ocr_name_hits'] / lines:.1%} ({s['ocr_name_hits']}/{lines})"
            " — 약품명 칸은 매칭 후 조제지시서 이름으로 잰다")
    out(f"  4개 칸 합계 {s['total_rate']:.1%} ({sum(s['hits'].values())}/{lines * len(FIELDS)})  "
        f"목표 칸마다 {TARGET_RATE:.0%} {'달성' if s['passed'] else '미달'}")
    missing = f" (빠진 코드: {', '.join(s['codes_missing'])})" if s["codes_missing"] else ""
    out(f"  대상 약 매칭 {s['target_ok']}/{s['target_lines']}줄 · 8종 중 {len(s['codes_ok'])}종 확정{missing}")
    out(f"  출력 형식 검증 {s['schema_ok']}/{done} · 구조화 실패 {s['failed']}장 · 경로 규칙 {s['rule']} / LLM {s['llm']}")
    out(f"  처방전 완전 일치 {s['exact']}/{n} · 조제 계획 판정 일치 {s['plan_ok']}/{done} · 위험 단계 일치 {s['risk_ok']}/{done} (DUR API 없이)")
    if seconds:
        out(f"  처리 시간 평균 {sum(seconds) / len(seconds):.1f}초, 최대 {max(seconds):.1f}초 ({'OCR 포함' if photos else 'OCR 제외'})")


def main(
    argv: list[str] | None = None,
    *,
    ocr_fn: OcrFn = real_ocr,
    llm_fn: LlmFn | None = None,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(description="rx8 가상 처방전 사진 20장 구조화·약품 매칭 정확도 측정")
    parser.add_argument("photos", nargs="?", help="처방전 사진 폴더 (jpg · png)")
    parser.add_argument("--a1-dir", help="OCR 대신 이 폴더의 A.1 JSON(rx8_NN.json)을 쓴다 (점검용)")
    parser.add_argument("--llm", action="store_true", help="규칙 경로가 실패하면 서비스처럼 LLM 경로로 넘어간다 (OpenAI 호출)")
    parser.add_argument("--out", help="결과를 JSON 파일로도 저장한다")
    args = parser.parse_args(argv)
    if bool(args.photos) == bool(args.a1_dir):
        parser.error("사진 폴더와 --a1-dir 중 하나만 주세요")

    if llm_fn is None:
        if args.llm:
            from agent.llm import extract_llm
            llm_fn = extract_llm
        else:
            llm_fn = no_llm
    try:
        dictionary = load_dictionary()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    folder = Path(args.a1_dir or args.photos)
    if not folder.is_dir():
        out(f"폴더가 없습니다: {folder}")
        return 1
    if args.a1_dir:
        sources = sorted(folder.glob("rx8_*.json"))
    else:
        sources = sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)

    rows: list[dict[str, Any]] = []
    seconds: list[float] = []
    seen: set[str] = set()
    for src in sources:
        rx_id = src.stem if args.a1_dir else rx_id_of(src)
        if rx_id is None or rx_id in seen or not (EXPECTED_DIR / f"{rx_id}.json").exists():
            out(f"  건너뜀: {src.name} (rx8_01~20 중 하나로 정할 수 없거나 이미 쟀음)")
            continue
        seen.add(rx_id)
        started = time.perf_counter()
        try:
            a1 = json.loads(src.read_text(encoding="utf-8")) if args.a1_dir else ocr_fn(src, rx_id)
            row = score(rx_id, a1, dictionary, llm_fn)
        except ConfigError as e:          # --llm인데 키가 없는 등
            out(f"설정 문제: {e}")
            return 1
        except StructureError as e:       # OCR이 글자를 하나도 못 찾음
            expected = json.loads((EXPECTED_DIR / f"{rx_id}.json").read_text(encoding="utf-8"))
            row = {"rx_id": rx_id, "expected": [expected_view(i) for i in expected["items"]], "wrong": [],
                   "error": f"{e.code}: {e}"}
        sec = time.perf_counter() - started
        seconds.append(sec)
        row.update(file=src.name, seconds=round(sec, 1))
        rows.append(row)
        if "error" in row:
            out(f"  {rx_id}  구조화 실패 ({row['error']})  {sec:.1f}초")
            continue
        mark = "일치" if not row["wrong"] else f"틀림 {len(row['wrong'])}"
        out(f"  {rx_id}  {row['source']:4}  약품 {len(row['got'])}/{len(row['expected'])}줄  {mark}  {sec:.1f}초")
        for w in row["wrong"]:
            out(f"      - {w}")

    if not rows:
        out(f"잴 처방전이 없습니다: {folder}")
        return 1
    summary = summarize(rows, dictionary)
    report(summary, seconds, photos=not args.a1_dir, llm=args.llm, out=out)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        out(f"  결과 파일: {args.out}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
