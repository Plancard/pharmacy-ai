"""데이터 형식 (명세서 v0.5 부록 A). CV·웹과 주고받는 값의 모양을 이 파일에만 적는다.

    A.1 OcrResult         CV → Agent     처방전·약봉투 OCR 결과
    A.3 Inspection        CV → Agent·웹   트레이 검수 결과 (12칸)
    A.4 Plan              Agent → CV     조제 계획
    A.5 DispensingOrder   Agent → 웹      조제지시서 (+ risk_check: 처방약끼리 위험 신호 대조 결과)
    A.6 ConsultResult     Agent → 웹      상담 결과 (+ 복용 약까지 본 위험 신호 재대조)
    (A.2 알약 검출 결과는 CV 안에서만 쓰는 형식이라 Agent는 다루지 않는다.)

규칙
  - 약은 품목기준코드 item_code(숫자 9자리)로 가리킨다. 이름은 화면에 보이는 용도다.
  - 시점 값은 morning · noon · evening · bedtime, 일차는 1~3 (트레이 한 장 = 3일 x 4시점).
  - 모양이 어긋난 값은 pydantic ValidationError를 낸다. 알 수 없는 키도 거부한다 (extra="forbid").
    키가 하나 늘거나 이름이 달라졌다는 것은 팀이 합의한 형식이 어긋났다는 뜻이므로, 조용히 넘기지 않고 알아챈다.
    형식을 바꿀 때는 명세서 부록 A와 이 파일을 함께 고친다.
  - 이 파일은 맨 아래층이다. 같은 층의 config · errors 말고는 부르지 않는다.
  - 조제 계획이 트레이 범위를 넘을 때의 오류(E_PLAN_RANGE)는 조제 계획을 만드는 쪽(plan.py)이 낸다.
    여기 Plan은 범위를 넘은 값을 만들 수 없게 막을 뿐이다.
"""
from collections.abc import Iterable
from typing import Annotated, Literal, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

# ------------------------------------------------------------------ 공통 값

TimeSlot = Literal["morning", "noon", "evening", "bedtime"]   # 트레이의 행 순서 (위 → 아래)
TIME_SLOTS: tuple[str, ...] = get_args(TimeSlot)
TRAY_DAYS = 3                                                 # 트레이 한 장에 담는 일수 (열)
TRAY_CELLS = TRAY_DAYS * len(TIME_SLOTS)                      # 12칸

DocType = Literal["prescription", "med_bag"]                  # A.1: 처방전 · 약봉투

# 위험 신호 수준. 낮은 쪽에서 높은 쪽 순서다. danger는 진행을 막고, caution은 화면 표시만 한다 (명세서 2.2·2.8).
RiskLevel = Literal["none", "caution", "warning", "danger"]
RISK_LEVELS: tuple[str, ...] = get_args(RiskLevel)
FindingLevel = Literal["caution", "warning", "danger"]        # 항목 하나의 수준. 항목이 있다는 것은 none이 아니라는 뜻이다
RiskScope = Literal["rx_only", "rx_with_current_meds"]        # 대조한 범위: 처방약끼리 / 복용 중인 약까지

# 약품 매칭이 약을 정한 방법 (Agent 내부, 4~5단계). code=인쇄된 코드가 사전의 품목기준코드와 같다 · name=사전에서 이름으로 · api=식약처 API에서 이름으로 · none=정하지 못함.
# 자동 확정은 code뿐이다. name · api · none 줄은 언제나 약사 확인 표시가 붙는다 (drug_match.py가 지킨다).
MatchMethod = Literal["code", "name", "api", "none"]

CellStatus = Literal["ok", "mismatch", "need_review"]         # A.3 칸 판정
CellReason = Literal["low_conf", "boundary", "overlap", "rotated"]   # A.3 확인이 필요한 이유


def risk_rank(level: str) -> int:
    """수준의 높낮이 (none 0 · caution 1 · warning 2 · danger 3)."""
    return RISK_LEVELS.index(level)


def max_risk(levels: Iterable[str]) -> RiskLevel:
    """가장 높은 수준. 하나도 없으면 none."""
    return max(levels, key=risk_rank, default="none")   # type: ignore[return-value]


def _check_bbox(v: list[int]) -> list[int]:
    x1, y1, x2, y2 = v
    if min(v) < 0 or x1 >= x2 or y1 >= y2:
        raise ValueError("bbox는 [x1, y1, x2, y2] 이고 0 이상이며 x1 < x2, y1 < y2 여야 합니다")
    return v


ItemCode = Annotated[str, StringConstraints(pattern=r"^[0-9]{9}$")]       # 품목기준코드. \d는 다른 나라 숫자도 받으므로 [0-9]
Bbox = Annotated[list[int], Field(min_length=4, max_length=4), AfterValidator(_check_bbox)]
Day = Annotated[int, Field(ge=1, le=TRAY_DAYS)]
Count = Annotated[int, Field(ge=0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ------------------------------------------------------------------ A.1 OCR 결과 (CV → Agent)

class OcrLine(_Model):
    line_id: int = Field(ge=0)               # 정렬 뒤 0부터 매긴 번호. A.5의 source_line_ids가 이 값을 가리킨다
    text: str = Field(min_length=1)
    bbox: Bbox                               # 원본 이미지 기준 픽셀 [x1, y1, x2, y2]
    conf: float = Field(ge=0.0, le=1.0)      # 글자 인식 신뢰도


class OcrResult(_Model):
    """처방전·약봉투 한 장의 OCR 결과. 같은 줄 묶기는 CV가 아니라 Agent(structure.py)가 한다."""

    image_id: str = Field(min_length=1)
    doc_type: DocType
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    lines: list[OcrLine]                     # 비어 있어도 형식은 맞다. 글자가 없는 사진(E_OCR_EMPTY)은 구조화 단계가 가려낸다

    @model_validator(mode="after")
    def _lines_sorted_and_inside(self):
        prev = None
        for i, ln in enumerate(self.lines):
            if ln.line_id != i:
                raise ValueError(f"line_id는 0부터 차례로 매겨야 합니다 (lines[{i}].line_id={ln.line_id})")
            x1, y1, x2, y2 = ln.bbox
            if x2 > self.width or y2 > self.height:
                raise ValueError(f"line_id {ln.line_id}의 bbox가 이미지({self.width}x{self.height}) 밖입니다")
            key = ((y1 + y2) / 2, x1)        # 박스 중심 y(위 → 아래), 그다음 x(왼 → 오)
            if prev is not None and key < prev:
                raise ValueError(f"lines는 박스 중심 y, 그다음 x 순서여야 합니다 (line_id {ln.line_id})")
            prev = key
        return self


# ------------------------------------------------------------------ A.3 검수 결과 (CV → Agent·웹)

class InspectionCell(_Model):
    day: Day
    time: TimeSlot
    expected: dict[ItemCode, Count]          # 조제 계획이 정한 것 {품목기준코드: 알 수}
    actual: dict[ItemCode, Count]            # 사진에서 센 것
    status: CellStatus
    reasons: list[CellReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_agrees_with_counts(self):
        same = {k: v for k, v in self.expected.items() if v} == {k: v for k, v in self.actual.items() if v}
        if self.status == "ok" and not same:
            raise ValueError(f"{self.day}일 {self.time}: ok인데 expected와 actual이 다릅니다")
        if self.status == "mismatch" and same:
            raise ValueError(f"{self.day}일 {self.time}: mismatch인데 expected와 actual이 같습니다")
        return self


class InspectionSummary(_Model):
    ok: int = Field(ge=0)
    mismatch: int = Field(ge=0)
    need_review: int = Field(ge=0)
    hitl_required: bool                      # 약사 확인(HITL ②)이 필요한가


class Inspection(_Model):
    prescription_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    summary: InspectionSummary
    cells: list[InspectionCell]              # 12칸 전부 (빈 칸도 expected·actual이 {} 로 들어 있다)

    @model_validator(mode="after")
    def _cells_complete_and_summary_agrees(self):
        keys = [(c.day, c.time) for c in self.cells]
        if len(keys) != TRAY_CELLS or len(set(keys)) != TRAY_CELLS:
            raise ValueError(f"cells는 {TRAY_DAYS}일 x {len(TIME_SLOTS)}시점 = {TRAY_CELLS}칸이 하나씩 있어야 합니다")
        for status in get_args(CellStatus):
            n = sum(c.status == status for c in self.cells)
            if getattr(self.summary, status) != n:
                raise ValueError(f"summary.{status}={getattr(self.summary, status)}인데 cells에는 {n}칸입니다")
        return self


# ------------------------------------------------------------------ A.4 조제 계획 (Agent → CV)

class PlanItem(_Model):
    item_code: ItemCode
    count: int = Field(ge=1)


class PlanCell(_Model):
    day: Day
    time: TimeSlot
    items: list[PlanItem] = Field(min_length=1)   # 약이 든 칸만 적는다. 계획에 없는 칸은 빈 칸이다

    @model_validator(mode="after")
    def _items_unique(self):
        codes = [it.item_code for it in self.items]
        if len(set(codes)) != len(codes):
            raise ValueError(f"{self.day}일 {self.time}: 같은 약이 두 번 적혔습니다")
        return self


class Plan(_Model):
    prescription_id: str = Field(min_length=1)
    days: int = Field(ge=1, le=TRAY_DAYS)    # 계획이 다루는 일수
    plan: list[PlanCell]

    @model_validator(mode="after")
    def _cells_unique_and_within_days(self):
        keys = [(c.day, c.time) for c in self.plan]
        if len(set(keys)) != len(keys):
            raise ValueError("plan에 같은 칸(day, time)이 두 번 나옵니다")
        for c in self.plan:
            if c.day > self.days:
                raise ValueError(f"{c.day}일 칸이 days={self.days}를 넘습니다")
        return self


# ------------------------------------------------------------------ 위험 신호 (A.5 risk_check · A.6 new_findings)

class RiskFinding(_Model):
    """위험 신호 항목 하나."""

    type: str = Field(min_length=1)          # DUR 유형 키 (ingredient_dup 등). 영어 키 목록은 13단계에서 DUR API 응답을 보고 확정한다
    item_codes: list[ItemCode] = Field(min_length=1)   # 관련된 약
    level: FindingLevel
    message: str = Field(min_length=1)       # 약사에게 그대로 보이는 한국어 문장

    @model_validator(mode="after")
    def _codes_unique(self):
        if len(set(self.item_codes)) != len(self.item_codes):
            raise ValueError("item_codes에 같은 약이 두 번 나옵니다")
        return self


class RiskCheck(_Model):
    """A.5의 risk_check. level은 항목 중 가장 높은 수준이고, 항목이 없으면 none이다."""

    scope: RiskScope
    level: RiskLevel
    findings: list[RiskFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _level_is_highest_finding(self):
        want = max_risk(f.level for f in self.findings)
        if self.level != want:
            raise ValueError(f"level={self.level}인데 항목 중 가장 높은 수준은 {want}입니다")
        return self


# ------------------------------------------------------------------ A.5 조제지시서 (Agent → 웹)

class MatchCandidate(_Model):
    """약품 매칭이 확정하지 못하고 약사에게 보여 주는 후보 하나. 후보는 고른 것이 아니다."""

    item_code: ItemCode
    name: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)     # 이름이 비슷한 정도 (자모 단위, 소수 셋째 자리까지)
    reason: str = Field(min_length=1)        # 후보로 올린 이유 ("처방전 코드" · "이름이 비슷함" · "의약품 API 검색")


MAX_CANDIDATES = 3                           # 후보는 많아야 3개 (REQ-AG-03)


class DrugLine(_Model):
    """조제지시서의 약품 한 줄."""

    item_code: ItemCode | None               # 약을 정하지 못했으면 None. 약품 매칭 전에도 None이다 (웹에 넘기기 전 검사: check_ready_for_web)
    name: str = Field(min_length=1)
    count_per_dose: int = Field(ge=1)        # 1회 투약량 (알 수)
    times: list[TimeSlot] = Field(min_length=1)   # 복용 시점. 개수가 1일 투여 횟수다
    days: int = Field(ge=1)                  # 투약 일수. 3일을 넘으면 조제 계획 단계에서 E_PLAN_RANGE로 약사에게 돌아간다
    source_line_ids: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)   # 이 줄을 읽은 A.1 line_id들. LLM 경로는 모른다(빈 목록)

    # A.5 밖의 확장. 웹이 이미 승인 화면에서 쓴다 (web/agent_gateway.py)
    needs_pharmacist_check: bool = False
    check_reasons: list[str] = Field(default_factory=list)   # 확인이 필요한 이유 (약사에게 그대로 보인다)

    # A.5 밖의 Agent 내부 필드. 웹이 몰라도 되고, 함께 넘어가면 승인 카드가 보여 줄 수 있다 (2단계)
    printed_code: str | None = None          # 처방전에 인쇄된 코드 (보험코드 자리). 약품 매칭이 사전에서 약을 찾을 때 쓴다
    ocr_conf: float | None = Field(default=None, ge=0.0, le=1.0)   # 이 줄을 읽은 OCR 칸들의 최저 신뢰도. LLM 경로는 모른다(None)

    # 약품 매칭의 결과 (4단계). 매칭 전이거나 약사가 직접 정한 줄은 match_method가 None이다
    match_method: MatchMethod | None = None
    match_candidates: list[MatchCandidate] = Field(default_factory=list, max_length=MAX_CANDIDATES)   # 정하지 못했거나 코드와 이름이 어긋날 때 보여 주는 후보

    @model_validator(mode="after")
    def _consistent(self):
        if len(set(self.times)) != len(self.times):
            raise ValueError(f"{self.name}: times에 같은 시점이 두 번 나옵니다")
        if len(set(self.source_line_ids)) != len(self.source_line_ids):
            raise ValueError(f"{self.name}: source_line_ids에 같은 번호가 두 번 나옵니다")
        if self.check_reasons and not self.needs_pharmacist_check:
            raise ValueError(f"{self.name}: check_reasons가 있으면 needs_pharmacist_check가 True여야 합니다")
        if self.match_method in ("code", "name", "api") and self.item_code is None:
            raise ValueError(f"{self.name}: match_method={self.match_method}이면 item_code가 있어야 합니다")
        if self.match_method == "none" and self.item_code is not None:
            raise ValueError(f"{self.name}: match_method=none인데 item_code가 있습니다")
        if self.match_candidates and not self.needs_pharmacist_check:
            raise ValueError(f"{self.name}: 후보를 보여 줄 때는 needs_pharmacist_check가 True여야 합니다")
        codes = [c.item_code for c in self.match_candidates]
        if len(set(codes)) != len(codes):
            raise ValueError(f"{self.name}: match_candidates에 같은 약이 두 번 나옵니다")
        return self


class DispensingOrder(_Model):
    """조제지시서: 처방전 한 장을 구조화한 결과."""

    prescription_id: str = Field(min_length=1)
    items: list[DrugLine] = Field(min_length=1)

    # A.5 밖의 Agent 내부 필드 (2단계). 웹은 몰라도 된다
    issue_no: str | None = None              # 처방전 교부번호 ("발급 연월일 및 번호" 칸의 글자)
    source: Literal["rule", "llm"] | None = None   # 어느 경로로 구조화했나. 약사가 직접 고친 뒤에는 None일 수 있다
    warnings: list[str] = Field(default_factory=list)   # 구조화 중 생긴, 약사가 알아야 할 경고 (약사에게 그대로 보인다)

    # None = 아직 위험 신호 대조 전 (구조화 직후). 웹에 넘기는 조제지시서에는 반드시 들어 있다.
    # 이 대조는 처방약끼리만 본다(scope=rx_only). 복용 중인 약까지 본 결과는 A.6에 담긴다.
    risk_check: RiskCheck | None = None

    @model_validator(mode="after")
    def _risk_check_refers_to_items(self):
        if self.risk_check is None:
            return self
        if self.risk_check.scope != "rx_only":
            raise ValueError("조제지시서의 risk_check는 처방약끼리만 본다 (scope=rx_only)")
        codes = {d.item_code for d in self.items if d.item_code}
        for f in self.risk_check.findings:
            unknown = [c for c in f.item_codes if c not in codes]
            if unknown:
                raise ValueError(f"risk_check 항목({f.type})이 처방에 없는 약을 가리킵니다: {unknown}")
        return self

    def check_ready_for_web(self) -> None:
        """웹(승인 화면)에 넘겨도 되는 조제지시서인가. 아니면 ValueError.

        - 위험 신호 대조가 끝났다 (risk_check가 있다)
        - 약을 정하지 못한 줄(item_code가 없는 줄)은 약사 확인 표시가 켜져 있다 — 그래야 승인 화면이 강조해 보여 준다
        구조화 직후나 약품 매칭 전의 조제지시서는 이 검사를 통과하지 못하는 것이 정상이다.
        """
        problems = []
        if self.risk_check is None:
            problems.append("위험 신호 대조(risk_check)가 아직 없습니다")
        for i, d in enumerate(self.items, start=1):
            if d.item_code is None and not d.needs_pharmacist_check:
                problems.append(f"{i}번째 약품({d.name})은 약을 정하지 못했는데 약사 확인 표시가 없습니다")
        if problems:
            raise ValueError("웹에 넘길 수 없는 조제지시서입니다: " + "; ".join(problems))

    @property
    def all_matched(self) -> bool:
        """모든 약품에 품목기준코드가 채워졌는가."""
        return all(d.item_code for d in self.items)

    @property
    def flagged_indices(self) -> list[int]:
        """약사 확인 표시가 붙은 약품 줄의 번호(0부터). 승인 화면이 강조해 보여 준다."""
        return [i for i, d in enumerate(self.items) if d.needs_pharmacist_check]


# ------------------------------------------------------------------ A.6 상담 결과 (Agent → 웹)

class Evidence(_Model):
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class ConsultResult(_Model):
    """상담 한 번의 결과. risk_level은 A.5 risk_check와 이번 재대조 중 높은 쪽이다."""

    prescription_id: str = Field(min_length=1)
    risk_scope: RiskScope
    new_findings: list[RiskFinding] = Field(default_factory=list)   # 승인 화면(A.5)에서 이미 보인 것을 뺀, 이번에 새로 나온 항목
    recheck_after_dispense: bool             # 조제가 끝난 뒤 다시 대조해야 하는가
    risk_level: RiskLevel
    answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        if self.risk_scope == "rx_only" and self.new_findings:
            raise ValueError("처방약끼리만 본 상담(rx_only)에는 새 항목이 없다 (이미 A.5 risk_check로 보였다)")
        if risk_rank(self.risk_level) < risk_rank(max_risk(f.level for f in self.new_findings)):
            raise ValueError("risk_level이 new_findings의 가장 높은 수준보다 낮습니다")
        return self
