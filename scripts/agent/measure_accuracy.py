"""정답 대비 구조화·약품 매칭 정확도 측정 (6단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/measure_accuracy.py            # 규칙 경로만 (키 없이 돈다)
    python scripts/agent/measure_accuracy.py --llm      # 규칙 경로 + LLM 경로 (OpenAI를 실제로 부른다)
    python scripts/agent/measure_accuracy.py --llm --out var/accuracy/1110.json   # 결과를 파일로도 남긴다

pytest가 아니다. agent/resources/samples/cv_a1 의 가상 처방전 5장을 경로마다 따로 돌려
조제지시서(검증 + 약품 매칭까지)를 만들고, cv_expected 의 정답과 견준다.
LLM 경로는 규칙 경로가 성공해도 일부러 따로 돌린다 (두 경로를 각각 재기 위해). LLM은 한 장에 한 번만 부른다(재시도 없음).
LLM에는 교부번호와 약품 표만 보낸다 (환자 정보는 보내지 않는다).
약품 매칭은 약품 사전만 쓴다. 식약처 API는 부르지 않는다 (네트워크 상태에 따라 결과가 달라지지 않게).

끝에 나오는 "일정표 정확도 기록" 줄을 일정표에 붙여 넣는다. --out 으로 저장하는 JSON은 칸마다 정답·결과가 들어 있다.
var/ 는 git에 올라가지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/measure_accuracy.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.accuracy import (  # noqa: E402
    CANDIDATE,
    CELL_FIELDS,
    CONFIRMED,
    FIELD_LABELS,
    HEADLINE_FIELDS,
    MATCH_LABELS,
    MATCH_OUTCOMES,
    NOT_TARGET,
    TARGET_RATE,
    SampleResult,
    Summary,
    name_misread_lines,
    score_sample,
    summarize,
)
from agent.drug_dictionary import DrugDictionary, load_dictionary  # noqa: E402
from agent.drug_match import match_order  # noqa: E402
from agent.errors import ConfigError, StructureError  # noqa: E402
from agent.llm import extract_llm  # noqa: E402
from agent.structure import extract_rule, ocr_to_llm_text  # noqa: E402
from agent.validate import order_from_extracted  # noqa: E402

SAMPLES = config.RESOURCES_DIR / "samples"
PATH_LABELS = {"rule": "규칙 경로", "llm": "LLM 경로"}
# 줄마다 풀어서 보여 줄 매칭 결과 (정답을 정했거나 대상 약이 아니라 정하지 않은 줄은 문제가 아니라서 개수만 센다)
DETAILED_OUTCOMES = tuple(o for o in MATCH_OUTCOMES if o not in (CONFIRMED, NOT_TARGET))

LlmFn = Callable[..., dict[str, Any]]


def sample_ids() -> list[str]:
    return sorted(p.stem for p in (SAMPLES / "cv_expected").glob("rx_*.json"))


def load(kind: str, sample_id: str) -> dict[str, Any]:
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def run_path(path: str, sample_id: str, dictionary: DrugDictionary, *, llm_fn: LlmFn) -> SampleResult:
    """처방전 한 장을 한 경로로 돌려 정답과 견준다. 구조화에 실패해도 멈추지 않고 실패로 센다."""
    ocr, expected = load("cv_a1", sample_id), load("cv_expected", sample_id)
    started = time.perf_counter()
    order, error = None, None
    try:
        extracted = extract_rule(ocr) if path == "rule" else llm_fn(ocr_to_llm_text(ocr), None)
        order = match_order(order_from_extracted(f"ACC-{sample_id}", extracted, source=path), dictionary)
    except StructureError as e:
        error = str(e)
    seconds = time.perf_counter() - started
    return score_sample(sample_id, path, order, expected, seconds=seconds, error=error)


# ------------------------------------------------------------ 출력


def pct(correct: int, total: int) -> str:
    return f"{Summary.rate(correct, total):.1%} ({correct}/{total})"


def field_parts(summary: Summary, names: tuple[str, ...]) -> str:
    return " · ".join(f"{FIELD_LABELS[n]} {pct(summary.field_correct.get(n, 0), summary.field_total.get(n, 0))}" for n in names)


def match_parts(summary: Summary) -> str:
    return " · ".join(f"{MATCH_LABELS[o]} {summary.matches[o]}" for o in MATCH_OUTCOMES)


def verdict(summary: Summary) -> str:
    return f"목표 {TARGET_RATE:.0%} " + ("달성" if summary.meets_target else "미달")


def report(summary: Summary, results: list[SampleResult], out: Callable[[str], None]) -> None:
    out(f"== {PATH_LABELS[summary.path]} ==")
    for r in results:
        ok_cells = sum(c.ok for c in r.ocr_checks)
        line = (
            f"  {r.sample_id}  칸 {ok_cells}/{len(r.ocr_checks)}  약품 줄 {r.ocr_lines_correct}/{r.line_count}"
            f"  약사 확인 {r.flagged}줄  {r.seconds:.1f}초"
        )
        out(line + (f"  [실패] {r.error}" if r.error else ""))
    out(f"  칸 정확도 (OCR 기준) {pct(summary.ocr_correct, summary.ocr_total)}  ({verdict(summary)})")
    out(f"    {field_parts(summary, HEADLINE_FIELDS)}")
    out(f"    {field_parts(summary, ('issue_no', 'printed_code'))}")
    out(
        f"  약품 줄 완전 일치 {pct(summary.lines_correct, summary.lines)}"
        f" | 처방전 완전 일치 {summary.all_correct}/{summary.samples} | 구조화 실패 {summary.failed}장"
    )
    out(f"  진짜 값 기준 칸 정확도 {pct(summary.true_correct, summary.true_total)}  (약품명 제외)")
    out(f"  약품 매칭 ({summary.lines}줄): {match_parts(summary)}")
    out(f"  약사 확인 표시 {summary.flagged}줄 | 시간 평균 {sum(summary.seconds) / max(1, len(summary.seconds)):.1f}초, 최대 {max(summary.seconds, default=0):.1f}초")

    for outcome in DETAILED_OUTCOMES:
        rows = [(r.sample_id, m) for r in results for m in r.match_checks if m.outcome == outcome]
        if rows:
            out(f"  약품 매칭 - {MATCH_LABELS[outcome]}:")
            for sample_id, m in rows:
                out(f"    {sample_id} {m.describe()}")
    wrong = [(r.sample_id, c) for r in results for c in r.ocr_checks if not c.ok]
    if wrong:
        out("  OCR 기준으로 틀린 칸:")
        for sample_id, c in wrong:
            out(f"    {sample_id} {c.describe()}")
    true_wrong = [(r.sample_id, c) for r in results for c in r.true_checks if not c.ok]
    if true_wrong:
        out("  진짜 값 기준으로 틀린 칸 (OCR 오류 포함):")
        for sample_id, c in true_wrong:
            out(f"    {sample_id} {c.describe()}")


def record_line(summary: Summary, today: date, *, model: str | None = None) -> str:
    """일정표 정확도 기록에 붙여 넣을 한 줄."""
    head = f"{today:%Y-%m-%d} | {PATH_LABELS[summary.path]}" + (f" ({model})" if model else "")
    return (
        f"{head} | 처방전 {summary.samples}장 · 약품 {summary.lines}줄"
        f" | {field_parts(summary, HEADLINE_FIELDS)}"
        f" | 칸 정확도 {pct(summary.ocr_correct, summary.ocr_total)} · {verdict(summary)}"
        f" | 약품 매칭: 정답 코드 {summary.matches[CONFIRMED]}줄 · 후보로 정답 {summary.matches[CANDIDATE]}줄"
        f" · 잘못 정함 {summary.wrong_matches}줄"
    )


def as_json(summaries: dict[str, Summary], results: dict[str, list[SampleResult]], today: date) -> dict[str, Any]:
    def checks(cs):
        return [{"line": c.line, "field": c.field, "want": c.want, "got": c.got, "ok": c.ok} for c in cs]

    def matches(ms):
        return [{"line": m.line, "name": m.name, "want": m.want, "got": m.got, "outcome": m.outcome} for m in ms]

    return {
        "date": f"{today:%Y-%m-%d}",
        "target_rate": TARGET_RATE,
        "paths": {
            path: {
                "model": config.OPENAI_MODEL if path == "llm" else None,
                "ocr_rate": s.ocr_rate, "ocr_correct": s.ocr_correct, "ocr_total": s.ocr_total,
                "true_rate": s.true_rate, "true_correct": s.true_correct, "true_total": s.true_total,
                "line_rate": s.line_rate, "all_correct": s.all_correct, "samples": s.samples,
                "failed": s.failed, "flagged": s.flagged, "seconds": [round(x, 2) for x in s.seconds],
                "meets_target": s.meets_target,
                "fields": {n: {"correct": s.field_correct.get(n, 0), "total": s.field_total.get(n, 0)}
                           for n in ("issue_no", *CELL_FIELDS)},
                "matches": dict(s.matches),
                "results": [
                    {"sample_id": r.sample_id, "error": r.error, "seconds": round(r.seconds, 2), "flagged": r.flagged,
                     "ocr_checks": checks(r.ocr_checks), "true_checks": checks(r.true_checks),
                     "match_checks": matches(r.match_checks)}
                    for r in results[path]
                ],
            }
            for path, s in summaries.items()
        },
    }


def main(
    argv: list[str] | None = None,
    *,
    llm_fn: LlmFn = extract_llm,
    out: Callable[[str], None] = print,
    today: Callable[[], date] = date.today,
) -> int:
    parser = argparse.ArgumentParser(description="정답 대비 구조화·약품 매칭 정확도 측정")
    parser.add_argument("--llm", action="store_true", help="LLM 경로도 잰다 (OpenAI를 실제로 부른다)")
    parser.add_argument("--out", help="결과를 JSON 파일로도 저장한다")
    args = parser.parse_args(argv)

    try:
        dictionary = load_dictionary()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    ids = sample_ids()
    misread = sum(name_misread_lines(load("cv_expected", s)) for s in ids)
    total_lines = sum(len(load("cv_expected", s)["items"]) for s in ids)
    out(f"처방전 {len(ids)}장, 약품 {total_lines}줄 (그중 OCR이 약품명을 틀리게 읽은 줄 {misread}줄)")

    paths = ["rule", "llm"] if args.llm else ["rule"]
    summaries: dict[str, Summary] = {}
    results: dict[str, list[SampleResult]] = {}
    for path in paths:
        try:
            results[path] = [run_path(path, s, dictionary, llm_fn=llm_fn) for s in ids]
        except ConfigError as e:          # 키가 없으면 이 경로는 잴 수 없다
            out(f"== {PATH_LABELS[path]} == 설정 문제로 재지 못했습니다: {e}")
            continue
        summaries[path] = summarize(path, results[path])
        report(summaries[path], results[path], out)

    day = today()
    if summaries:
        out("== 일정표 정확도 기록 (그대로 붙여 넣으세요) ==")
        for path, summary in summaries.items():
            out(record_line(summary, day, model=config.OPENAI_MODEL if path == "llm" else None))

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(as_json(summaries, results, day), ensure_ascii=False, indent=2), encoding="utf-8")
        out(f"결과 파일: {args.out}")
    return 0 if len(summaries) == len(paths) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
