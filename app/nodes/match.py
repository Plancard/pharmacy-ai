"""match 노드: 조제지시서의 약품 → 품목기준코드.

약사가 수정한 뒤에도 같은 노드를 다시 쓴다 (다시 매칭이 곧 다시 검증).
"""
from __future__ import annotations

from pydantic import ValidationError

from app.domain.drug_catalog import Catalog, get_default_catalog
from app.domain.drug_match import match_order, unconfirmed_indices
from app.schemas import DispensingOrder


def match_node(state: dict, catalog: Catalog | None = None) -> dict:
    order_data = state.get("order")
    if not order_data:
        return {"status": "failed", "notice": "조제지시서가 없어 약품 매칭을 할 수 없습니다.", "errors": ["match: order 없음"]}
    try:
        order = DispensingOrder.model_validate(order_data)
    except ValidationError:
        return {"status": "failed", "notice": "조제지시서 형식이 올바르지 않아 약품 매칭을 할 수 없습니다.", "errors": ["match: order 형식 오류"]}

    matched = match_order(order, catalog or get_default_catalog())
    pending = unconfirmed_indices(matched)
    notice = f"약사 확인이 필요한 약품이 {len(pending)}개 있습니다." if pending else ""
    return {"order": matched.model_dump(mode="json"), "status": "matched", "notice": notice}
