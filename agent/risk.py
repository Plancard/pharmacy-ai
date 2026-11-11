"""위험 신호 대조 결과(A.5 risk_check)를 만든다 (8단계, 일정표 11/10, REQ-AG-05 · 07).

dur.py가 찾은 사실(DurHit)에 단계를 붙이고, 조제지시서의 risk_check와 경고로 옮긴다.

- 단계는 DUR 유형에 따른 규칙이 정한다 (명세서 2.8절). LLM이 정하지 않는다 (REQ-AG-07).
  LLM은 항목의 설명 문장만 쓴다. 18단계에서 붙기 전까지는 여기 있는 고정 문장이 그 자리를 채운다.
- 이 대조는 처방약끼리만 본다 (scope=rx_only). 복용 중인 약까지 본 결과는 A.6에 담긴다 (15단계).
- 대조하지 못한 약(약을 정하지 못했거나 주성분을 모르는 약)은 조제지시서의 경고로 남긴다.
  '해당 없음'이 '안전함'으로 읽히지 않게 하기 위해서다.

단계 규칙(2.8절)은 병용금기 · 연령금기 · 임부금기가 danger, 성분 중복 · 효능군 중복이 warning,
용량 · 투여기간 · 노인 · 분할 주의가 caution이다. 1차는 병용금기와 중복만 판정하므로 표에는 이 셋만 있다 (13단계).
표에 없는 유형이 오면 단계를 짐작하지 않고 멈춘다.
DUR 조회가 없었으면(키가 없거나 --no-api) 병용금기 · 효능군 중복은 대조하지 않았다고 경고로 알린다.

판단 층이다. 파일도 네트워크도 쓰지 않는다.
"""
from __future__ import annotations

from typing import Iterable

from agent.drug_dictionary import DrugDictionary
from agent.dur import (
    COMBINATION_TABOO,
    DUR_FAILED,
    EFFICACY_DUP,
    INGREDIENT_DUP,
    NO_INGREDIENTS,
    NO_ITEM_CODE,
    TABOO_PARTIAL,
    DurHit,
    DurLookup,
    DurReport,
    Unchecked,
    find_all,
)
from agent.schema import DispensingOrder, FindingLevel, RiskCheck, RiskFinding, max_risk

# DUR 유형 -> 단계 (명세서 2.8절). 1차는 병용금기와 중복만 판정한다.
LEVEL_BY_TYPE: dict[str, FindingLevel] = {
    COMBINATION_TABOO: "danger",
    INGREDIENT_DUP: "warning",
    EFFICACY_DUP: "warning",
}
TYPE_LABELS = {COMBINATION_TABOO: "병용금기", INGREDIENT_DUP: "성분 중복", EFFICACY_DUP: "효능군 중복"}

LEVEL_LABELS = {"none": "해당 없음", "caution": "주의", "warning": "경고", "danger": "위험"}

# 약사가 확인하기 전에는 진행하지 않는 단계 (일정표 8단계 · 명세서 2.2절). caution은 화면에 표시만 한다.
CONFIRM_LEVELS = ("warning", "danger")

# 대조하지 못했다는 경고는 모두 이 말로 끝난다. 다시 대조할 때 지난 경고를 가려 지우는 데 쓴다.
UNCHECKED_TAIL = "성분 중복을 대조하지 못했습니다."
DUR_UNCHECKED_TAIL = "병용금기 · 효능군 중복을 대조하지 못했습니다."
NO_DUR_WARNING = "DUR API를 쓰지 않아 병용금기 · 효능군 중복은 대조하지 않았습니다. 약사가 직접 확인하세요."
TABOO_PARTIAL_TAIL = "병용금기 상대 약이 많아 앞부분만 대조했습니다."
_UNCHECKED_REASONS = {
    NO_ITEM_CODE: "품목기준코드를 정하지 못해",
    NO_INGREDIENTS: "주성분을 알 수 없어(약품 사전에 없거나 주성분 칸이 비어 있어)",
}
_RISK_WARNING_ENDS = (UNCHECKED_TAIL, DUR_UNCHECKED_TAIL, NO_DUR_WARNING, TABOO_PARTIAL_TAIL)


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


def combination_taboo_message(hit: DurHit) -> str:
    """병용금기 항목의 설명 문장. 금기 사유(DUR)가 있으면 덧붙인다."""
    text = f"{' · '.join(hit.names)}는 함께 쓰면 안 되는 조합입니다(병용금기)."
    return f"{text} 사유: {hit.detail}" if hit.detail else text


def efficacy_dup_message(hit: DurHit) -> str:
    return f"{' · '.join(hit.names)}는 같은 효능군({', '.join(hit.ingredients)})입니다. 함께 쓸 필요가 있는지 확인하세요."


_MESSAGES = {
    COMBINATION_TABOO: combination_taboo_message,
    INGREDIENT_DUP: ingredient_dup_message,
    EFFICACY_DUP: efficacy_dup_message,
}


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
    if item.reason == DUR_FAILED:
        why = f"DUR API를 쓰지 못해({item.detail}) " if item.detail else "DUR API를 쓰지 못해 "
        return f"{item.name}부터 {why}{DUR_UNCHECKED_TAIL}"
    if item.reason == TABOO_PARTIAL:
        return f"{item.name}: {TABOO_PARTIAL_TAIL}"
    return f"{item.name}: {_UNCHECKED_REASONS[item.reason]} {UNCHECKED_TAIL}"


def is_unchecked_warning(text: str) -> bool:
    """위험 신호 대조가 남긴 경고인가 (다시 대조할 때 지우고 새로 쓴다)."""
    return text.endswith(_RISK_WARNING_ENDS)


def report_warnings(report: DurReport) -> list[str]:
    warnings = [unchecked_warning(u) for u in report.unchecked]
    if not report.dur_used:
        warnings.append(NO_DUR_WARNING)
    return warnings


def apply_report(order: DispensingOrder, report: DurReport) -> DispensingOrder:
    """대조 결과를 risk_check와 경고로 옮긴 새 조제지시서. 지난번 대조의 경고는 지우고 이번 것으로 바꾼다."""
    kept = [w for w in order.warnings if not is_unchecked_warning(w)]
    data = order.model_dump(mode="json")
    data["risk_check"] = risk_check_from_hits(report.hits).model_dump(mode="json")
    data["warnings"] = [*kept, *report_warnings(report)]
    return DispensingOrder.model_validate(data)


def check_order(order: DispensingOrder, dictionary: DrugDictionary, dur: DurLookup | None = None) -> DispensingOrder:
    """처방약끼리 위험 신호를 대조해 risk_check를 채운 새 조제지시서를 돌려준다. 받은 조제지시서는 바꾸지 않는다.

    dur(DUR 조회)가 있으면 병용금기 · 효능군 중복도 본다. 없으면 성분 중복만 보고, 나머지는 대조하지 않았다고 알린다.
    다시 대조해도 결과가 같다: 지난번의 '대조하지 못함' 경고는 지우고 이번 것으로 바꾼다.
    """
    return apply_report(order, find_all(order.items, dictionary, dur))
