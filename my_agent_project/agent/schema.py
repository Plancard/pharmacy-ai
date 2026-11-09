"""조제지시서 스키마 (11/2).

처방전 한 장을 구조화한 결과(DispensingOrder)와 그 안의 약품 한 줄(DrugLine)을 정의한다.
규칙 경로와 LLM 경로가 똑같이 이 스키마로 결과를 내고, 검증을 통과해야 다음 노드로 넘긴다.
"""
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 트레이의 행 순서: 아침, 점심, 저녁, 취침 전 (4행 x 3일). 복용 시점 값은 여기 한 곳에서만 정한다.
TimeSlot = Literal["morning", "noon", "evening", "bedtime"]
SLOT_ORDER: tuple[str, ...] = get_args(TimeSlot)
TRAY_DAYS = 3   # 트레이 한 장에 담을 수 있는 일수. 투약 일수가 이보다 길면 약사가 정해야 한다.


def _is_9_digits(value: str) -> bool:
    return len(value) == 9 and value.isascii() and value.isdigit()


class MatchCandidate(BaseModel):
    """약품 매칭에서 약사에게 보여 줄 후보 하나."""

    item_code: str
    name: str
    score: float = Field(ge=0.0, le=1.0)   # 이름이 비슷한 정도
    reason: str

    @field_validator("item_code")
    @classmethod
    def _item_code_is_9_digits(cls, v):
        if not _is_9_digits(v):
            raise ValueError("item_code는 숫자 9자리여야 합니다")
        return v


class DrugLine(BaseModel):
    """조제지시서의 약품 한 줄."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # --- 명세서 부록 A.5 (조제지시서, Agent -> 웹)의 필드. 이름과 순서를 A.5와 같게 둔다
    item_code: str | None = None      # 품목기준코드 9자리. 약품 매칭 전, 또는 약을 정하지 못했을 때는 None
    name: str = Field(min_length=1)
    count_per_dose: int = Field(ge=1)                      # 1회 투약량 (알약 수)
    times: list[TimeSlot]                                   # 복용 시점 (morning·noon·evening·bedtime)
    days: int = Field(ge=1)                                # 투약 일수
    source_line_ids: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)   # 이 줄을 읽은 A.1 line_id들. LLM 경로는 모른다(빈 목록)

    # --- A.5 밖의 Agent 내부 필드 (웹에도 함께 넘겨 승인 카드에서 쓸 수 있다)
    printed_code: str | None = None   # 처방전에 인쇄된 코드 (보험코드일 수 있다)
    times_per_day: int = Field(ge=1, le=len(SLOT_ORDER))   # 1일 투여 횟수. 하루 최대 4회 (복용 시점이 4개)
    ocr_conf: float | None = Field(default=None, ge=0.0, le=1.0)   # 이 줄을 읽은 OCR 칸들의 최저 신뢰도. LLM 경로는 모른다(None)

    # --- 약품 매칭 결과 (agent/drug_match.py가 채운다)
    # item_code를 어떻게 정했나: 사전의 코드 / 사전의 이름 / 의약품 API의 이름 / 못 정함
    match_method: Literal["code", "name", "api", "none"] = "none"
    match_candidates: list[MatchCandidate] = Field(default_factory=list)   # 약을 정하지 못했을 때 약사에게 보여 줄 후보
    needs_pharmacist_check: bool = False     # True면 HITL ①에서 강조해 약사가 확인한다
    check_reasons: list[str] = Field(default_factory=list)    # 확인이 필요한 이유 (약사에게 그대로 보인다)

    @field_validator("item_code")
    @classmethod
    def _item_code_is_9_digits(cls, v):
        if v is not None and not _is_9_digits(v):
            raise ValueError("item_code는 숫자 9자리여야 합니다")
        return v

    @model_validator(mode="after")
    def _slots_match_times(self):
        if len(self.times) != self.times_per_day or len(set(self.times)) != len(self.times):
            raise ValueError("times(복용 시점)는 times_per_day와 개수가 같고 중복이 없어야 합니다")
        return self

    @model_validator(mode="after")
    def _match_fields_agree(self):
        if self.match_method != "none" and self.item_code is None:
            raise ValueError("match_method가 none이 아니면 item_code가 있어야 합니다")
        if self.check_reasons and not self.needs_pharmacist_check:
            raise ValueError("확인이 필요한 이유(check_reasons)가 있으면 needs_pharmacist_check가 True여야 합니다")
        return self


class DispensingOrder(BaseModel):
    """조제지시서: 처방전 한 장을 구조화한 결과."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prescription_id: str = Field(min_length=1)
    source: Literal["rule", "llm"]                      # 어느 경로로 구조화했는지
    issue_no: str | None = None                         # 처방전 교부번호
    items: list[DrugLine] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)   # 구조화 중 생긴, 약사가 알아야 할 경고

    @property
    def all_matched(self) -> bool:
        """모든 약품에 품목기준코드가 채워졌는가 (약품 매칭 뒤에 True가 된다)."""
        return all(d.item_code for d in self.items)

    @property
    def flagged_indices(self) -> list[int]:
        """약사 확인 표시가 붙은 약품 줄의 번호(0부터). 승인 화면이 강조해 보여 준다."""
        return [i for i, d in enumerate(self.items) if d.needs_pharmacist_check]
