"""구조화·약품 매칭 정확도 계산 (6단계, REQ-AG-02).

조제지시서(구조화와 약품 매칭을 거친 DispensingOrder)를 정답 파일(agent/resources/samples/cv_expected)과 견준다.
여기에는 계산만 있다. 샘플을 읽고 돌려 보고 출력하는 일은 scripts/agent/measure_accuracy.py가 한다.

두 가지를 잰다.

1. 구조화 정확도 (칸 단위). 일정표의 "약품명·1회량·횟수·일수 정확도"이고 목표는 90%(TARGET_RATE)다.
   정답은 두 가지로 본다.
   - OCR 기준: OCR이 읽은 글자를 정답으로 본다. Agent가 OCR 결과를 제대로 옮겼는지 (구조화 정확도).
     OCR이 틀리게 읽은 이름·교부번호도 그대로 옮겼으면 맞은 것이다.
   - 진짜 값 기준: 처방전에 실제로 적힌 값(true_issue_no)을 정답으로 본다. OCR 오류까지 포함한 시스템 전체의 정확도.
     Agent는 OCR이 읽은 이름을 고치지 않으므로 이 기준에서 약품명은 뺀다.

2. 약품 매칭 결과 (줄 단위). 품목기준코드는 칸 정확도에 섞지 않고 따로 센다.
   Agent는 확신이 없으면 약을 정하지 않고 후보만 보인다(REQ-AG-03). 그래서 "코드가 정답과 같은가"만 보면
   조심스럽게 후보로만 보인 줄이 틀린 것으로 잡힌다. 줄마다 아래 여섯 가지 중 하나로 센다.
   가장 나쁜 것은 "잘못 정함"이다. 약사가 후보를 고르는 수고는 있어도 안전하지만, 틀린 약을 정해 놓으면 위험하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.drug_names import fix_ocr_digits, normalize_name
from agent.schema import DispensingOrder, DrugLine

TARGET_RATE = 0.9   # 칸 정확도 목표 (일정표 19단계: 가상 처방전의 정확도 90% 이상)

# OCR 기준으로 견주는 약품 칸
CELL_FIELDS = ("printed_code", "name", "count_per_dose", "times_per_day", "days")
# 진짜 값 기준으로 견주는 약품 칸 (이름은 뺀다: Agent는 OCR이 읽은 이름을 고치지 않는다)
TRUE_CELL_FIELDS = ("printed_code", "count_per_dose", "times_per_day", "days")
# 일정표 정확도 기록에 적는 칸
HEADLINE_FIELDS = ("name", "count_per_dose", "times_per_day", "days")

FIELD_LABELS = {
    "issue_no": "교부번호",
    "printed_code": "인쇄 코드",
    "name": "약품명",
    "count_per_dose": "1회 투약량",
    "times_per_day": "1일 횟수",
    "days": "투약 일수",
    "extra": "정답에 없는 약품",
}

# 약품 매칭 결과 (줄마다 하나)
CONFIRMED = "confirmed"      # 정답 품목기준코드를 정했다
CANDIDATE = "candidate"      # 정하지 않았지만 정답이 후보에 있다 (약사가 고르면 된다)
NOT_TARGET = "not_target"    # 대상 약이 아니고, 정하지 않았다 (정답)
NOT_FOUND = "not_found"      # 대상 약인데 정하지 못했고 후보에도 정답이 없다
WRONG = "wrong"              # 틀린 약을 정했다 (대상 약을 다른 약으로, 또는 대상이 아닌 약을 대상 약으로)
MISSING = "missing"          # 이 줄을 읽지 못했다 (구조화 단계의 문제)
MATCH_OUTCOMES = (CONFIRMED, CANDIDATE, NOT_TARGET, NOT_FOUND, WRONG, MISSING)

MATCH_LABELS = {
    CONFIRMED: "정답 코드를 정함",
    CANDIDATE: "정답을 후보로 보임",
    NOT_TARGET: "대상 약 아님",
    NOT_FOUND: "못 찾음",
    WRONG: "잘못 정함",
    MISSING: "줄 못 읽음",
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


def _value(line: DrugLine, field_name: str) -> Any:
    if field_name == "times_per_day":
        return len(line.times)   # 1일 투여 횟수는 복용 시점의 수다
    return getattr(line, field_name)


def compare(order: DispensingOrder | None, expected: dict[str, Any], *, truth: bool = False) -> list[Check]:
    """조제지시서와 정답의 칸별 비교. order가 None이면(구조화 실패) 모든 칸이 틀린 것이다."""
    fields = TRUE_CELL_FIELDS if truth else CELL_FIELDS
    issue_want = expected["true_issue_no"] if truth else expected["issue_no"]
    issue_got = order.issue_no if order is not None else None
    checks = [Check(None, "issue_no", issue_want, issue_got, order is not None and issue_got == issue_want)]

    got_lines = list(order.items) if order is not None else []
    for i, want_line in enumerate(expected["items"]):
        got_line = got_lines[i] if i < len(got_lines) else None
        for name in fields:
            got = _value(got_line, name) if got_line is not None else None
            checks.append(Check(i, name, want_line[name], got, got_line is not None and _same(name, want_line[name], got)))
    for i in range(len(expected["items"]), len(got_lines)):
        checks.append(Check(i, "extra", None, got_lines[i].name, False))
    return checks


def lines_all_correct(checks: list[Check], line_count: int) -> int:
    """정답의 약품 줄 중 모든 칸이 맞은 줄의 수. (정답에 없는 약품의 칸은 줄 번호가 line_count 이상이라 세지 않는다)"""
    return sum(1 for i in range(line_count) if all(c.ok for c in checks if c.line == i))


def name_misread_lines(expected: dict[str, Any]) -> int:
    """OCR이 약품명을 틀리게 읽은 줄의 수 (함량의 o/O -> 0 보정 뒤에도 진짜 이름과 다른 줄).

    OCR이 얼마나 틀렸는지 보는 숫자라서 한글 보정(캠술 -> 캡슐 등, drug_names.fix_ocr_letters)은 하지 않고 센다.
    매칭이 그 오인식을 얼마나 견뎠는지는 약품 매칭 결과(정답 코드 · 후보)로 본다.
    """
    def key(name: str) -> str:
        return normalize_name(fix_ocr_digits(name))

    return sum(1 for d in expected["items"] if key(d["name"]) != key(d["true_name"]))


# ------------------------------------------------------------------ 약품 매칭


@dataclass(frozen=True)
class MatchCheck:
    """약품 한 줄의 매칭 결과. want가 None이면 대상 약이 아닌 줄이다."""

    line: int
    name: str | None            # 조제지시서에 적힌 이름 (줄을 읽지 못했으면 정답 쪽 이름)
    want: str | None
    got: str | None
    outcome: str

    def describe(self) -> str:
        want = self.want or "없음(대상 약 아님)"
        got = self.got or "정하지 않음"
        return f"{self.line + 1}번째 약품 {self.name!r}: 정답 {want}, 결과 {got} ({MATCH_LABELS[self.outcome]})"


def classify_match(want: str | None, line: DrugLine | None) -> str:
    """정답 코드(대상 약이 아니면 None)와 약품 한 줄의 매칭 결과를 견주어 여섯 가지 중 하나를 돌려준다."""
    if line is None:
        return MISSING
    got = line.item_code
    if want is None:
        return NOT_TARGET if got is None else WRONG
    if got == want:
        return CONFIRMED
    if got is not None:
        return WRONG
    if any(c.item_code == want for c in line.match_candidates):
        return CANDIDATE
    return NOT_FOUND


def match_checks(order: DispensingOrder | None, expected: dict[str, Any]) -> list[MatchCheck]:
    """정답의 약품 줄마다 매칭 결과를 센다. order가 None이면 모든 줄이 MISSING이다."""
    got_lines = list(order.items) if order is not None else []
    checks = []
    for i, want_line in enumerate(expected["items"]):
        line = got_lines[i] if i < len(got_lines) else None
        checks.append(MatchCheck(
            line=i,
            name=line.name if line is not None else want_line["name"],
            want=want_line["item_code"],
            got=line.item_code if line is not None else None,
            outcome=classify_match(want_line["item_code"], line),
        ))
    return checks


# ------------------------------------------------------------------ 처방전 한 장 · 한 경로 전체


@dataclass
class SampleResult:
    """처방전 한 장을 한 경로로 돌린 결과."""

    sample_id: str
    path: str                       # "rule" | "llm"
    seconds: float
    error: str | None               # 구조화에 실패했으면 그 이유 (처방전 내용은 없다)
    ocr_checks: list[Check]
    true_checks: list[Check]
    match_checks: list[MatchCheck]
    line_count: int                 # 정답의 약품 줄 수
    flagged: int                    # 약사 확인 표시가 붙은 줄 수

    @property
    def ocr_lines_correct(self) -> int:
        return lines_all_correct(self.ocr_checks, self.line_count)

    @property
    def all_correct(self) -> bool:
        return self.error is None and all(c.ok for c in self.ocr_checks)


def score_sample(
    sample_id: str,
    path: str,
    order: DispensingOrder | None,
    expected: dict[str, Any],
    *,
    seconds: float,
    error: str | None = None,
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        path=path,
        seconds=seconds,
        error=error,
        ocr_checks=compare(order, expected),
        true_checks=compare(order, expected, truth=True),
        match_checks=match_checks(order, expected),
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
    field_correct: dict[str, int] = field(default_factory=dict)   # OCR 기준, 칸 이름별
    field_total: dict[str, int] = field(default_factory=dict)
    matches: dict[str, int] = field(default_factory=lambda: dict.fromkeys(MATCH_OUTCOMES, 0))

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

    def field_rate(self, name: str) -> float:
        return self.rate(self.field_correct.get(name, 0), self.field_total.get(name, 0))

    @property
    def meets_target(self) -> bool:
        """OCR 기준 칸 정확도가 목표(90%) 이상인가. 측정한 장이 없으면 아니다."""
        return self.ocr_total > 0 and self.ocr_rate >= TARGET_RATE

    @property
    def wrong_matches(self) -> int:
        """틀린 약을 정한 줄의 수. 0이어야 한다."""
        return self.matches[WRONG]


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
        for c in r.ocr_checks:
            s.field_total[c.field] = s.field_total.get(c.field, 0) + 1
            s.field_correct[c.field] = s.field_correct.get(c.field, 0) + c.ok
        for m in r.match_checks:
            s.matches[m.outcome] += 1
    return s
