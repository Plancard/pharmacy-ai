"""위험 신호 대조 결과(A.5 risk_check)를 만든다 (8단계, 일정표 11/10, REQ-AG-05 · 07).

dur.py가 찾은 사실(DurHit)에 단계를 붙이고, 조제지시서의 risk_check와 경고로 옮긴다.

- 단계는 DUR 유형에 따른 규칙이 정한다 (명세서 2.8절). LLM이 정하지 않는다 (REQ-AG-07).
  LLM은 항목의 설명 문장만 쓴다. 18단계에서 붙기 전까지는 여기 있는 고정 문장이 그 자리를 채운다.
- 이 대조는 처방약끼리만 본다 (scope=rx_only). 복용 중인 약까지 본 결과는 A.6에 담긴다 (15단계).
- 대조하지 못한 약(약을 정하지 못했거나 주성분을 모르는 약)은 조제지시서의 경고로 남긴다.
  '해당 없음'이 '안전함'으로 읽히지 않게 하기 위해서다.

단계 규칙(2.8절)은 병용금기 · 연령금기 · 임부금기가 danger, 성분 중복 · 효능군 중복이 warning,
용량 · 투여기간 · 노인 · 분할 주의가 caution이다. DUR 유형 키는 의약품 API 응답을 보고 확정하므로(13단계),
지금 표에는 8단계에서 만드는 성분 중복만 있다. 표에 없는 유형이 오면 단계를 짐작하지 않고 멈춘다.

판단 층이다. 파일도 네트워크도 쓰지 않는다.
"""
from __future__ import annotations

from typing import Iterable

from agent.drug_dictionary import DrugDictionary
from agent.dur import INGREDIENT_DUP, NO_INGREDIENTS, NO_ITEM_CODE, DurHit, DurReport, Unchecked, find_ingredient_duplicates
from agent.schema import DispensingOrder, FindingLevel, RiskCheck, RiskFinding, max_risk

# DUR 유형 -> 단계 (명세서 2.8절). 13단계에서 DUR API 응답의 유형 키를 보고 더한다.
LEVEL_BY_TYPE: dict[str, FindingLevel] = {
    INGREDIENT_DUP: "warning",
}

LEVEL_LABELS = {"none": "해당 없음", "caution": "주의", "warning": "경고", "danger": "위험"}

# 약사가 확인하기 전에는 진행하지 않는 단계 (일정표 8단계 · 명세서 2.2절). caution은 화면에 표시만 한다.
CONFIRM_LEVELS = ("warning", "danger")

# 대조하지 못했다는 경고는 모두 이 말로 끝난다. 다시 대조할 때 지난 경고를 가려 지우는 데 쓴다.
UNCHECKED_TAIL = "성분 중복을 대조하지 못했습니다."
_UNCHECKED_REASONS = {
    NO_ITEM_CODE: "품목기준코드를 정하지 못해",
    NO_INGREDIENTS: "주성분을 알 수 없어(약품 사전에 없거나 주성분 칸이 비어 있어)",
}


def finding_level(dur_type: str) -> FindingLevel:
    """DUR 유형의 단계. 표에 없는 유형은 단계를 짐작하지 않고 ValueError."""
    try:
        return LEVEL_BY_TYPE[dur_type]
    except KeyError:
        raise ValueError(f"단계 규칙이 없는 DUR 유형입니다: {dur_type!r}") from None


def requires_confirmation(level: str) -> bool:
    """이 단계는 약사가 확인한 뒤에 진행하는가 (warning · danger)."""
    return level in CONFIRM_LEVELS


def ingredient_dup_message(hit: DurHit) -> str:
    """성분 중복 항목의 설명 문장 (고정 문장. 18단계에서 LLM 설명이 이 자리를 대신할 수 있다)."""
    return f"{' · '.join(hit.names)}에 {', '.join(hit.ingredients)} 성분이 함께 들어 있습니다. 하루 총량을 확인하세요."


_MESSAGES = {INGREDIENT_DUP: ingredient_dup_message}


def finding_from_hit(hit: DurHit) -> RiskFinding:
    return RiskFinding(
        type=hit.type,
        item_codes=list(hit.item_codes),
        level=finding_level(hit.type),
        message=_MESSAGES[hit.type](hit),
    )


def risk_check_from_hits(hits: Iterable[DurHit]) -> RiskCheck:
    """찾은 항목들을 A.5의 risk_check로 만든다. 항목이 없으면 level은 none이다."""
    findings = [finding_from_hit(hit) for hit in hits]
    return RiskCheck(scope="rx_only", level=max_risk(f.level for f in findings), findings=findings)


def unchecked_warning(item: Unchecked) -> str:
    return f"{item.name}: {_UNCHECKED_REASONS[item.reason]} {UNCHECKED_TAIL}"


def is_unchecked_warning(text: str) -> bool:
    return text.endswith(UNCHECKED_TAIL)


def check_order(order: DispensingOrder, dictionary: DrugDictionary) -> DispensingOrder:
    """처방약끼리 위험 신호를 대조해 risk_check를 채운 새 조제지시서를 돌려준다. 받은 조제지시서는 바꾸지 않는다.

    다시 대조해도 결과가 같다: 지난번의 '대조하지 못함' 경고는 지우고 이번 것으로 바꾼다.
    """
    report: DurReport = find_ingredient_duplicates(order.items, dictionary)
    kept = [w for w in order.warnings if not is_unchecked_warning(w)]
    data = order.model_dump(mode="json")
    data["risk_check"] = risk_check_from_hits(report.hits).model_dump(mode="json")
    data["warnings"] = [*kept, *(unchecked_warning(u) for u in report.unchecked)]
    return DispensingOrder.model_validate(data)
