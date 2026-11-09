"""그래프의 노드 (11/6 뼈대).

노드는 상태를 받아 바꿀 칸만 dict로 돌려준다. LangGraph 없이도 함수로 불러 시험할 수 있다.
실제 일은 agent/ 패키지가 하고, 노드는 상태에서 꺼내 넘기고 결과를 다시 담기만 한다.

    structure  OCR -> 조제지시서 (규칙 경로 -> 실패하면 LLM 경로)
    match      약품 -> 품목기준코드 (약품 사전 -> 사전에 없으면 식약처 의약품 API), 약사 확인 표시
    approve    약사 승인 대기 (자리. 다음 주에 interrupt로 멈추고 약사 응답을 받는다)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.drug_dictionary import DrugDictionary
from agent.drug_match import ApiSearch, match_order
from agent.errors import StructureError
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder

StructureFn = Callable[[str, dict[str, Any]], DispensingOrder]

# 의약품 API를 쓰지 못할 때 조제지시서에 남기는 경고. 사전에서 못 찾은 약이 있을 때만 남긴다.
NO_API_WARNING = "의약품 API를 쓰지 않아(DRUG_API_KEY 없음) 약품 사전(대상 8종)에서만 품목기준코드를 찾았습니다."


@dataclass(frozen=True)
class Deps:
    """노드가 쓰는 도구. 테스트에서는 가짜로 바꿔 끼운다."""

    dictionary: DrugDictionary
    structure: StructureFn = structure_prescription
    api: ApiSearch | None = None          # 의약품 API 조회 (agent.drug_api.default_api). None이면 사전만 쓴다


def _order_dict(order: DispensingOrder) -> dict[str, Any]:
    return order.model_dump(mode="json")


def structure_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """OCR 결과를 조제지시서로 만든다. 못 만들면 실패로 표시하고 이유를 남긴다.

    키가 없는 등의 개발 환경 문제(ConfigError)는 잡지 않는다. 처방전마다 실패로 넘길 일이 아니다.
    """
    try:
        order = deps.structure(state["prescription_id"], state["ocr"])
    except StructureError as e:
        return {
            "order": None, "status": "failed", "notice": str(e), "error_code": e.code,
            "errors": [f"[structure] {e.code or '-'} {e}"],
        }
    return {"order": _order_dict(order), "status": "structured", "notice": None}


def match_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """약품마다 품목기준코드를 정하고 약사 확인 표시를 붙인다.

    사전에 없는 약은 의약품 API에서 이름으로 찾는다 (deps.api가 있을 때). API가 실패해도 멈추지 않고
    그 줄에 이유를 남겨 약사 확인으로 넘긴다 (REQ-NF-06).
    """
    order = match_order(DispensingOrder.model_validate(state["order"]), deps.dictionary, api=deps.api)
    if deps.api is None and not order.all_matched and NO_API_WARNING not in order.warnings:
        order.warnings = [*order.warnings, NO_API_WARNING]
    return {"order": _order_dict(order), "status": "matched"}


def approve_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """약사 승인 대기 (자리).

    지금은 상태만 '승인 대기'로 바꾸고 끝난다. 다음 주에 여기서 interrupt로 그래프를 멈추고,
    약사의 승인·수정·반려를 받아 다음 노드로 보낸다.
    """
    order = DispensingOrder.model_validate(state["order"])
    flagged = len(order.flagged_indices)
    notice = "약사 승인을 기다립니다." + (f" 확인이 필요한 약품이 {flagged}줄 있습니다." if flagged else "")
    return {"status": "awaiting_approval", "notice": notice}
