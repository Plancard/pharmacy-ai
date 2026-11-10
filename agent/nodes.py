"""그래프의 노드 (7단계, 일정표 11/9, REQ-AG-01).

노드는 상태를 받아 바꿀 칸만 dict로 돌려준다. LangGraph 없이도 함수로 불러 시험할 수 있다.
실제 일은 agent/ 의 다른 파일이 하고, 노드는 상태에서 꺼내 넘기고 결과를 다시 담기만 한다.

    supervisor  사진 종류를 보고 흐름을 정한다 (규칙). 메시지 하나에 한 번만 돈다
    structure   OCR -> 조제지시서 (규칙 경로 -> 실패하면 LLM 경로)
    match       약품 -> 품목기준코드 (약품 사전 -> 사전에 없으면 식약처 의약품 API), 약사 확인 표시
    approve     약사 승인 대기 (자리. 10단계에서 interrupt로 멈추고 약사 응답을 받는다)
    ask_type    사진 종류를 되묻는다 (자리. 10단계에서 HITL type_select interrupt가 된다)
    recheck     복용 약 재대조 (자리. 15단계에서 채운다)
    tray        트레이 검수 (자리. 12단계에서 채운다)

슈퍼바이저의 규칙은 여기서는 사진 종류만 가린다. 글의 의도까지 가르는 intent.py · supervisor.py는 11단계에서 만들고,
그때 decide_flow가 그쪽으로 옮겨 간다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.drug_dictionary import DrugDictionary
from agent.drug_match import ApiSearch, match_order
from agent.errors import StructureError
from agent.schema import DispensingOrder
from agent.state import PHOTO_KINDS, Flow

StructureFn = Callable[[str, dict[str, Any] | None], DispensingOrder]   # OCR가 없으면(None) 구조화는 E_OCR_EMPTY로 실패한다

# 의약품 API를 쓰지 못할 때 조제지시서에 남기는 경고. 사전에서 못 찾은 약이 있을 때만 남긴다.
NO_API_WARNING = "의약품 API를 쓰지 않아(DRUG_API_KEY 없음) 약품 사전(대상 8종)에서만 품목기준코드를 찾았습니다."

# 사진 종류 -> 흐름. 처방전만 지금 끝까지 간다.
KIND_FLOW: dict[str, Flow] = {"prescription": "prescription", "med_bag": "recheck", "tray": "tray"}
KIND_LABELS = {"prescription": "처방전", "med_bag": "약 포장", "tray": "트레이"}
DOC_KINDS = ("prescription", "med_bag")     # OCR(A.1)의 doc_type이 가질 수 있는 값

NOT_READY_NOTICES = {
    "recheck": "복용 약 재대조 흐름은 아직 준비 중입니다.",
    "tray": "트레이 검수 흐름은 아직 준비 중입니다.",
}
ASK_TYPE_NOTICE = "사진 종류를 정하지 못했습니다. 처방전 · 약 포장 · 트레이 중에서 골라 주세요."


@dataclass(frozen=True)
class Deps:
    """노드가 쓰는 도구. 테스트에서는 가짜로 바꿔 끼운다."""

    dictionary: DrugDictionary
    structure: StructureFn                # (처방 ID, OCR) -> 조제지시서. 실제로는 graph.default_structure
    api: ApiSearch | None = None          # 의약품 API 조회 (agent.drug_api.default_api). None이면 사전만 쓴다


def _order_dict(order: DispensingOrder) -> dict[str, Any]:
    return order.model_dump(mode="json")


# ------------------------------------------------------------ 슈퍼바이저 (규칙)


def _ask(reason: str) -> tuple[Flow, str]:
    return "ask_type", f"사진 종류를 정하지 못했습니다. {reason} 처방전 · 약 포장 · 트레이 중에서 골라 주세요."


def decide_flow(photo_kind: Any, ocr: Any) -> tuple[Flow, str | None]:
    """사진 종류로 흐름을 정한다. (흐름, 사진 종류를 되물을 때의 안내) 를 돌려준다.

    두 단서를 본다: 들어온 사진 종류(photo_kind)와 OCR이 읽은 문서 종류(ocr의 doc_type).
    - 한쪽만 있으면 그것을 따른다. 둘 다 있고 같으면 그것을 따른다.
    - 둘 다 없거나, 알 수 없는 값이거나, 서로 다르면 맞혀 보지 않고 되묻는다 (ask_type).
      처방전을 약 포장으로 잘못 다루면 약사가 모르고 지나가기 때문이다.
    """
    if photo_kind is not None and photo_kind not in PHOTO_KINDS:
        return _ask("알 수 없는 사진 종류입니다.")
    doc_type = ocr.get("doc_type") if isinstance(ocr, dict) else None
    if doc_type is not None and doc_type not in DOC_KINDS:
        return _ask("OCR이 읽은 문서 종류를 알 수 없습니다.")
    if photo_kind is not None and doc_type is not None and photo_kind != doc_type:
        return _ask(f"지정한 종류({KIND_LABELS[photo_kind]})와 OCR이 읽은 종류({KIND_LABELS[doc_type]})가 다릅니다.")
    kind = photo_kind or doc_type
    if kind is None:
        return _ask("사진 종류 정보가 없습니다.")
    return KIND_FLOW[kind], None


def supervisor_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """메시지 하나의 흐름을 정한다. 그래프에서 이 노드로 돌아오는 화살표는 없으므로 메시지 하나에 한 번만 돈다."""
    flow, notice = decide_flow(state.get("photo_kind"), state.get("ocr"))
    update: dict[str, Any] = {"flow": flow}
    if notice is not None:
        update["notice"] = notice
    return update


# ------------------------------------------------------------ 처방 흐름


def structure_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """OCR 결과를 조제지시서로 만든다. 못 만들면 실패로 표시하고 이유를 남긴다.

    키가 없는 등의 개발 환경 문제(ConfigError)는 잡지 않는다. 처방전마다 실패로 넘길 일이 아니다.
    """
    try:
        order = deps.structure(state["prescription_id"], state.get("ocr"))
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

    지금은 상태만 '승인 대기'로 바꾸고 끝난다. 10단계에서 여기서 interrupt(HITL ①)로 그래프를 멈추고,
    약사의 승인·수정·반려를 받아 다음 노드로 보낸다.
    """
    order = DispensingOrder.model_validate(state["order"])
    flagged = len(order.flagged_indices)
    notice = "약사 승인을 기다립니다." + (f" 확인이 필요한 약품이 {flagged}줄 있습니다." if flagged else "")
    return {"status": "awaiting_approval", "notice": notice}


# ------------------------------------------------------------ 다른 흐름의 자리


def ask_type_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """사진 종류를 되묻는 자리 (HITL type_select). 지금은 상태만 바꾸고 끝난다.

    슈퍼바이저가 남긴 이유가 있으면 그 안내를 그대로 쓴다.
    """
    return {"status": "awaiting_type", "notice": state.get("notice") or ASK_TYPE_NOTICE}


def recheck_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """복용 약 재대조 (자리). 15단계에서 채운다."""
    return {"status": "not_ready", "notice": NOT_READY_NOTICES["recheck"]}


def tray_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """트레이 검수 (자리). 12단계에서 채운다."""
    return {"status": "not_ready", "notice": NOT_READY_NOTICES["tray"]}
