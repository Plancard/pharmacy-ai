"""정답 대비 구조화·약품 매칭 정확도 측정 (11/6).

    python measure_accuracy.py            # 규칙 경로만 (키 없이 돈다)
    python measure_accuracy.py --llm      # 규칙 경로 + LLM 경로 (OpenAI를 실제로 부른다)
    python measure_accuracy.py --llm --out data/agent/accuracy_1106.json   # 결과를 파일로도 남긴다

pytest가 아니다. data/agent/cv_samples의 가상 처방전 5장을 경로마다 따로 돌려
조제지시서(검증 + 약품 매칭까지)를 만들고, data/agent/cv_expected의 정답과 칸 단위로 견준다.
LLM 경로는 규칙 경로가 성공해도 일부러 따로 돌린다 (두 경로를 각각 재기 위해).
LLM에는 교부번호와 약품 표만 보낸다 (환자 정보는 보내지 않는다). 약품 API는 부르지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from agent.accuracy import SampleResult, Summary, score_sample, summarize, name_misread_lines
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import match_order
from agent.errors import ConfigError, StructureError
from agent.llm_extract import extract_llm
from agent.structure import extract_rule, ocr_to_llm_text
from agent.validate import order_from_extracted

DATA = Path(__file__).resolve().parent / "data" / "agent"
PATH_LABELS = {"rule": "규칙 경로", "llm": "LLM 경로"}

LlmFn = Callable[..., dict[str, Any]]


def sample_ids() -> list[str]:
    return sorted(p.stem for p in (DATA / "cv_expected").glob("rx_*.json"))


def load(kind: str, sample_id: str) -> dict[str, Any]:
    return json.loads((DATA / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def run_path(path: str, sample_id: str, dictionary: DrugDictionary, *, llm_fn: LlmFn) -> SampleResult:
    """처방전 한 장을 한 경로로 돌려 정답과 견준다. 구조화에 실패해도 멈추지 않고 실패로 센다."""
    ocr, expected = load("cv_samples", sample_id), load("cv_expected", sample_id)
    started = time.perf_counter()
    order, error = None, None
    try:
        extracted = extract_rule(ocr) if path == "rule" else llm_fn(ocr_to_llm_text(ocr), None)
        order = match_order(order_from_extracted(f"ACC-{sample_id}", extracted, source=path), dictionary)
    except StructureError as e:
        error = str(e)
    seconds = time.perf_counter() - started
    return score_sample(sample_id, path, order, expected, seconds=seconds, error=error)


def report(summary: Summary, results: list[SampleResult], out: Callable[[str], None]) -> None:
    out(f"== {PATH_LABELS[summary.path]} ==")
    for r in results:
        ok_cells = sum(c.ok for c in r.ocr_checks)
        line = (
            f"  {r.sample_id}  칸 {ok_cells}/{len(r.ocr_checks)}  약품 줄 {r.ocr_lines_correct}/{r.line_count}"
            f"  약사 확인 {r.flagged}줄  {r.seconds:.1f}초"
        )
        out(line + (f"  [실패] {r.error}" if r.error else ""))
    out(
        f"  OCR 기준 칸 정확도 {summary.ocr_rate:.1%} ({summary.ocr_correct}/{summary.ocr_total})"
        f" | 약품 줄 완전 일치 {summary.line_rate:.1%} ({summary.lines_correct}/{summary.lines})"
        f" | 처방전 완전 일치 {summary.all_correct}/{summary.samples} | 구조화 실패 {summary.failed}장"
    )
    out(f"  진짜 값 기준 칸 정확도 {summary.true_rate:.1%} ({summary.true_correct}/{summary.true_total})  (약품명 제외)")
    out(f"  약사 확인 표시 {summary.flagged}줄 | 시간 평균 {sum(summary.seconds) / max(1, len(summary.seconds)):.1f}초, 최대 {max(summary.seconds, default=0):.1f}초")
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


def as_json(summaries: dict[str, Summary], results: dict[str, list[SampleResult]]) -> dict[str, Any]:
    def checks(cs):
        return [{"line": c.line, "field": c.field, "want": c.want, "got": c.got, "ok": c.ok} for c in cs]

    return {
        path: {
            "ocr_rate": s.ocr_rate, "ocr_correct": s.ocr_correct, "ocr_total": s.ocr_total,
            "true_rate": s.true_rate, "true_correct": s.true_correct, "true_total": s.true_total,
            "line_rate": s.line_rate, "all_correct": s.all_correct, "samples": s.samples,
            "failed": s.failed, "flagged": s.flagged, "seconds": [round(x, 2) for x in s.seconds],
            "results": [
                {"sample_id": r.sample_id, "error": r.error, "seconds": round(r.seconds, 2), "flagged": r.flagged,
                 "ocr_checks": checks(r.ocr_checks), "true_checks": checks(r.true_checks)}
                for r in results[path]
            ],
        }
        for path, s in summaries.items()
    }


def main(argv: list[str] | None = None, *, llm_fn: LlmFn = extract_llm, out: Callable[[str], None] = print) -> int:
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

    if args.out:
        Path(args.out).write_text(json.dumps(as_json(summaries, results), ensure_ascii=False, indent=2), encoding="utf-8")
        out(f"결과 파일: {args.out}")
    return 0 if len(summaries) == len(paths) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
