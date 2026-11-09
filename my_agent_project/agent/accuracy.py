"""구조화·약품 매칭 정확도 계산 (11/6).

조제지시서(검증·매칭을 거친 DispensingOrder)를 정답 파일(data/agent/cv_expected)과 칸 단위로 견준다.
정답은 두 가지로 본다.
- OCR 기준: OCR이 읽은 글자를 정답으로 본다. Agent가 OCR 결과를 제대로 옮겼는지 (구조화 정확도).
  OCR이 틀리게 읽은 이름·교부번호도 그대로 옮겼으면 맞은 것이다.
- 진짜 값 기준: 처방전에 실제로 적힌 값(true_issue_no, ground_truth)을 정답으로 본다. OCR 오류까지 포함한
  시스템 전체의 정확도. 이름은 Agent가 고치지 않으므로 이 기준에서 빼고, 대신 품목기준코드로 본다.

여기에는 계산만 있다. 실행·출력은 measure_accuracy.py가 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.drug_names import match_key, normalize_name

# OCR 기준으로 견주는 약품 칸
OCR_FIELDS = ("printed_code", "name", "count_per_dose", "times_per_day", "days", "item_code")
# 진짜 값 기준으로 견주는 약품 칸 (이름은 뺀다: Agent는 OCR이 읽은 이름을 고치지 않는다)
TRUE_FIELDS = ("printed_code", "count_per_dose", "times_per_day", "days", "item_code")

FIELD_LABELS = {
    "issue_no": "교부번호",
    "printed_code": "코드",
    "name": "약품명",
    "count_per_dose": "1회 투약량",
    "times_per_day": "1일 횟수",
    "days": "투약 일수",
    "item_code": "품목기준코드",
    "extra": "정답에 없는 약품",
}


@dataclass(frozen=True)
class Check:
    """칸 하나의 비교 결과. line은 약품 줄 번호(0부터), 교부번호는 None."""

    line: int | None
    field: str
    want: Any
    got: Any
    ok: bool

    def describe(self) -> str:
        where = "" if self.line is None else f"{self.line + 1}번째 약품 "
        return f"{where}{FIELD_LABELS[self.field]}: 정답 {self.want!r}, 결과 {self.got!r}"


def _same(field_name: str, want: Any, got: Any) -> bool:
    if field_name == "name":
        # 띄어쓰기·'/'만 다른 것은 같은 이름이다
        return isinstance(got, str) and normalize_name(want) == normalize_name(got)
    return want == got


def compare(order, expected: dict[str, Any], *, truth: bool = False) -> list[Check]:
    """조제지시서와 정답의 칸별 비교. order가 None이면(구조화 실패) 모든 칸이 틀린 것이다."""
    fields = TRUE_FIELDS if truth else OCR_FIELDS
    issue_want = expected["true_issue_no"] if truth else expected["issue_no"]
    issue_got = order.issue_no if order is not None else None
    checks = [Check(None, "issue_no", issue_want, issue_got, order is not None and issue_got == issue_want)]

    got_lines = list(order.items) if order is not None else []
    for i, want_line in enumerate(expected["items"]):
        got_line = got_lines[i].model_dump() if i < len(got_lines) else None
        for name in fields:
            got = got_line[name] if got_line is not None else None
            checks.append(Check(i, name, want_line[name], got, got_line is not None and _same(name, want_line[name], got)))
    for i in range(len(expected["items"]), len(got_lines)):
        checks.append(Check(i, "extra", None, got_lines[i].name, False))
    return checks


def lines_all_correct(checks: list[Check], line_count: int) -> int:
    """정답의 약품 줄 중 모든 칸이 맞은 줄의 수."""
    return sum(
        1 for i in range(line_count)
        if all(c.ok for c in checks if c.line == i and c.field != "extra")
    )


def name_misread_lines(expected: dict[str, Any]) -> int:
    """OCR이 약품명을 틀리게 읽은 줄의 수 (함량의 o/O -> 0 보정 뒤에도 진짜 이름과 다른 줄)."""
    return sum(1 for d in expected["items"] if match_key(d["name"]) != match_key(d["true_name"]))


@dataclass
class SampleResult:
    """처방전 한 장을 한 경로로 돌린 결과."""

    sample_id: str
    path: str                       # "rule" | "llm"
    seconds: float
    error: str | None               # 구조화에 실패했으면 그 이유 (처방전 내용은 없다)
    ocr_checks: list[Check]
    true_checks: list[Check]
    line_count: int                 # 정답의 약품 줄 수
    flagged: int                    # 약사 확인 표시가 붙은 줄 수

    @property
    def ocr_lines_correct(self) -> int:
        return lines_all_correct(self.ocr_checks, self.line_count)

    @property
    def all_correct(self) -> bool:
        return self.error is None and all(c.ok for c in self.ocr_checks)


def score_sample(sample_id: str, path: str, order, expected: dict[str, Any], *, seconds: float, error: str | None = None) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        path=path,
        seconds=seconds,
        error=error,
        ocr_checks=compare(order, expected),
        true_checks=compare(order, expected, truth=True),
        line_count=len(expected["items"]),
        flagged=len(order.flagged_indices) if order is not None else 0,
    )


@dataclass
class Summary:
    """한 경로의 전체 결과."""

    path: str
    samples: int = 0
    failed: int = 0                 # 구조화 자체에 실패한 장 수
    all_correct: int = 0            # OCR 기준으로 모든 칸이 맞은 장 수
    ocr_correct: int = 0
    ocr_total: int = 0
    true_correct: int = 0
    true_total: int = 0
    lines: int = 0
    lines_correct: int = 0
    flagged: int = 0
    seconds: list[float] = field(default_factory=list)

    @staticmethod
    def rate(correct: int, total: int) -> float:
        return correct / total if total else 0.0

    @property
    def ocr_rate(self) -> float:
        return self.rate(self.ocr_correct, self.ocr_total)

    @property
    def true_rate(self) -> float:
        return self.rate(self.true_correct, self.true_total)

    @property
    def line_rate(self) -> float:
        return self.rate(self.lines_correct, self.lines)


def summarize(path: str, results: list[SampleResult]) -> Summary:
    s = Summary(path=path)
    for r in results:
        s.samples += 1
        s.failed += r.error is not None
        s.all_correct += r.all_correct
        s.ocr_correct += sum(c.ok for c in r.ocr_checks)
        s.ocr_total += len(r.ocr_checks)
        s.true_correct += sum(c.ok for c in r.true_checks)
        s.true_total += len(r.true_checks)
        s.lines += r.line_count
        s.lines_correct += r.ocr_lines_correct
        s.flagged += r.flagged
        s.seconds.append(r.seconds)
    return s
