"""그래프의 노드 (7 ~ 11단계, 일정표 11/9 ~ 11/13, REQ-AG-01 · 04 · 05, REQ-WEB-02 · 03).

노드는 상태를 받아 바꿀 칸만 dict로 돌려준다. LangGraph 없이도 함수로 불러 시험할 수 있다.
실제 일은 agent/ 의 다른 파일이 하고, 노드는 상태에서 꺼내 넘기고 결과를 다시 담기만 한다.

    supervisor     메시지 하나의 흐름과 순서를 정한다 (supervisor.py의 규칙). 메시지 하나에 한 번만 돈다
    ask_type       사진 종류나 글의 뜻을 되묻는다 (HITL type_select). 약사가 고르면 그것으로 흐름을 다시 정한다
    structure      OCR -> 조제지시서 (규칙 경로 -> 실패하면 LLM 경로)
    match          약품 -> 품목기준코드 (약품 사전 -> 사전에 없으면 식약처 의약품 API), 약사 확인 표시.
                   deps.tools(MCP 클라이언트)가 있으면 MCP 도구 match_drug로 부른다 (12단계)
    risk           처방약끼리 위험 신호 대조 (지금은 성분 중복). 결과를 조제지시서의 risk_check에 담는다
    approve        약사 승인 대기 (HITL ①). interrupt로 멈추고, 약사 응답(승인 · 고치기)을 받아 이어 간다.
                   글로 받은 제안과 메모를 승인 화면에 함께 보인다
    make_plan      승인된 조제지시서로 칸별 조제 계획(A.4)을 만든다. 승인 전에는 오지 않는다
    register_meds  복용 약 등록 (자리. 15단계)
    recheck        복용 약 재대조 (자리. 15단계)
    tray           트레이 검수 (자리. 14단계)
    consult        상담 답변 (자리. 16단계)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.drug_dictionary import DrugDictionary
from agent.drug_match import ApiSearch, match_order
from agent.errors import StructureError
from agent.mcp_client import DrugTools, McpToolError
from agent.plan import PlanError, build_plan
from agent.risk import LEVEL_LABELS, check_order, requires_confirmation
from agent.schema import DispensingOrder
from agent.supervisor import (  # noqa: F401  decide_flow · KIND_FLOW 등은 7단계부터 이 파일에서도 불러 썼다
    DOC_KINDS,
    KIND_FLOW,
    KIND_LABELS,
    ask_choices,
    choice_labels,
    decide_flow,
    has_photo,
    plan_after_choice,
    plan_message,
)

StructureFn = Callable[[str, dict[str, Any] | None], DispensingOrder]   # OCR가 없으면(None) 구조화는 E_OCR_EMPTY로 실패한다
AskFn = Callable[[dict[str, Any]], Any]     # 멈추고 약사에게 묻는다: 보여 줄 내용 -> 약사 응답 (실제로는 LangGraph의 interrupt)

# HITL ① (명세서 2.2절). 이름은 웹의 agent_gateway.HITL_POINTS와 같다
HITL_APPROVAL = "rx_approval"
APPROVE, EDIT = "approve", "edit"           # 약사 응답의 action. 웹의 승인 호출은 {"action": "approve", "order": 조제지시서}
APPROVED_NOTICE = "약사가 승인했습니다."
EDITED_NOTICE = "약사가 고친 내용으로 약품 매칭과 위험 신호 대조를 다시 했습니다."

# 의약품 API를 쓰지 못할 때 조제지시서에 남기는 경고. 사전에서 못 찾은 약이 있을 때만 남긴다.
# 키가 없을 때와 --no-api로 끈 때 모두 남으므로 이유는 적지 않는다.
NO_API_WARNING = "의약품 API를 쓰지 않아 약품 사전(대상 8종)에서만 품목기준코드를 찾았습니다."

NOT_READY_NOTICES = {
    "register_meds": "복용 약 등록은 아직 준비 중입니다.",
    "recheck": "복용 약 재대조 흐름은 아직 준비 중입니다.",
    "tray": "트레이 검수 흐름은 아직 준비 중입니다.",
    "consult": "상담 답변은 아직 준비 중입니다.",
}
ASK_TYPE_NOTICE = "사진 종류를 정하지 못했습니다. 처방전 · 약 포장 · 트레이 중에서 골라 주세요."
HITL_TYPE = "type_select"                   # 되묻기 (웹의 agent_gateway.HITL_POINTS와 같은 이름)


@dataclass(frozen=True)
class Deps:
    """노드가 쓰는 도구. 테스트에서는 가짜로 바꿔 끼운다."""

    dictionary: DrugDictionary
    structure: StructureFn                # (처방 ID, OCR) -> 조제지시서. 실제로는 graph.default_structure
    api: ApiSearch | None = None          # 의약품 API 조회 (agent.drug_api.default_api). None이면 사전만 쓴다
    ask: AskFn | None = None              # 약사에게 묻기. None이면 LangGraph의 interrupt (그래프 안에서만 쓸 수 있다)
    tools: DrugTools | None = None        # MCP 약품 도구 (agent.mcp_client). 있으면 매칭을 MCP로 부른다 (api는 서버 쪽 것을 쓴다)


def _order_dict(order: DispensingOrder) -> dict[str, Any]:
    return order.model_dump(mode="json")


# ------------------------------------------------------------ 슈퍼바이저 (규칙)


def supervisor_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """메시지 하나의 흐름과 순서를 정한다. 그래프에서 이 노드로 돌아오는 화살표는 없으므로 메시지 하나에 한 번만 돈다.

    실행할 흐름이 없으면(글을 메모로만 남김) noted로 끝난다.
    """
    plan = plan_message(state.get("photo_kind"), state.get("ocr"), state.get("text"), deps.dictionary)
    update = plan.as_update()
    if not plan.route:
        update["status"] = "noted"
    return update


def type_question(state: dict[str, Any]) -> dict[str, Any]:
    """되묻는 화면 (HITL type_select): 무엇을 묻는지, 고를 수 있는 값과 이름."""
    about = "photo" if has_photo(state.get("photo_kind"), state.get("ocr"), state.get("text")) else "text"
    return {
        "hitl_point": HITL_TYPE,
        "about": about,
        "choices": list(ask_choices(about)),
        "labels": choice_labels(about),
        "notice": state.get("notice") or ASK_TYPE_NOTICE,
    }


def read_choice(reply: Any, about: str) -> str:
    """되묻기의 응답 {"choice": 값}을 읽는다. 고를 수 없는 값이면 ValueError."""
    choice = reply.get("choice") if isinstance(reply, dict) else None
    if choice not in ask_choices(about):
        raise ValueError(f"고를 수 있는 값이 아닙니다: {choice!r} (가능: {', '.join(ask_choices(about))})")
    return choice


def ask_type_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """사진 종류나 글의 뜻을 되묻는다 (HITL type_select). 여기서 멈추고, 약사가 고르면 그것으로 흐름을 다시 정한다.

    사진 종류를 고르면 약사의 선택을 따른다 (OCR이 읽은 종류와 달라도 다시 묻지 않는다).
    """
    question = type_question(state)
    choice = read_choice(_ask_pharmacist(deps, question), question["about"])
    plan = plan_after_choice(question["about"], choice, state.get("text"), deps.dictionary)
    update = plan.as_update()
    update["notice"] = plan.notice
    if question["about"] == "photo":
        update["photo_kind"] = choice
    if not plan.route:
        update["status"] = "noted"
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


def _keep_pharmacist_choice(line, matched):
    """약사가 고친 줄을 다시 매칭할 때: 약사가 정한 약이 매칭 결과와 다르면 약사의 선택을 따른다 (매칭 방법은 비운다)."""
    if line.item_code is None or line.item_code == matched.item_code:
        return matched
    return line.model_copy(update={
        "match_method": None, "match_candidates": [], "needs_pharmacist_check": False, "check_reasons": [],
    })


MCP_FAILED_WARNING = "MCP 약품 도구를 쓰지 못해 Agent가 직접 약품을 찾았습니다"


def match_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """약품마다 품목기준코드를 정하고 약사 확인 표시를 붙인다.

    사전에 없는 약은 의약품 API에서 이름으로 찾는다 (deps.api가 있을 때). API가 실패해도 멈추지 않고
    그 줄에 이유를 남겨 약사 확인으로 넘긴다 (REQ-NF-06).
    약사가 고친 뒤(status=edited)에는 약사가 직접 고른 약(매칭 결과와 다른 품목기준코드)을 바꾸지 않는다.
    """
    order = DispensingOrder.model_validate(state["order"])
    mcp_failure = None
    if deps.tools is not None:
        try:
            matched, api_used = deps.tools.match_drug(order)  # 같은 match_order를 MCP 서버가 부른다
        except McpToolError as e:                             # 서버가 죽어도 조제를 멈추지 않고 직접 부른다
            mcp_failure = e
    if deps.tools is None or mcp_failure is not None:
        matched, api_used = match_order(order, deps.dictionary, api=deps.api), deps.api is not None
    if mcp_failure is not None:
        matched.warnings = [*matched.warnings, f"{MCP_FAILED_WARNING}: {mcp_failure}"]
    if state.get("status") == "edited":
        matched.items = [_keep_pharmacist_choice(line, new) for line, new in zip(order.items, matched.items)]
        matched = DispensingOrder.model_validate(matched.model_dump(mode="json"))
    if not api_used and not matched.all_matched and NO_API_WARNING not in matched.warnings:
        matched.warnings = [*matched.warnings, NO_API_WARNING]
    return {"order": _order_dict(matched), "status": "matched"}


def risk_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """처방약끼리 위험 신호를 대조해 조제지시서의 risk_check를 채운다 (REQ-AG-05).

    지금은 약품 사전의 주성분으로 성분 중복만 본다. 병용금기와 효능군 중복은 13단계에서 더한다.
    대조하지 못한 약(약을 정하지 못했거나 주성분을 모르는 약)은 조제지시서의 경고에 남는다.
    """
    order = check_order(DispensingOrder.model_validate(state["order"]), deps.dictionary)
    return {"order": _order_dict(order), "status": "checked"}


def approval_notice(order: DispensingOrder, state: dict[str, Any]) -> str:
    """승인 화면에 보일 안내. 조제 계획을 만들지 못해 돌아왔으면 그 이유를 앞에 붙인다."""
    notice = "약사 승인을 기다립니다."
    flagged = len(order.flagged_indices)
    if flagged:
        notice += f" 확인이 필요한 약품이 {flagged}줄 있습니다."
    risk = order.risk_check
    if risk is not None and risk.findings:
        notice += f" 위험 신호가 {len(risk.findings)}건 있습니다({LEVEL_LABELS[risk.level]})."
        if requires_confirmation(risk.level):
            notice += " 약사가 확인한 뒤에 승인할 수 있습니다."
    if state.get("status") == "failed":
        notice = f"{state.get('notice')} 고친 뒤 다시 승인해 주세요. " + notice
    return notice


def ocr_lines(order: DispensingOrder, ocr: Any) -> dict[str, dict[str, Any]]:
    """승인 화면에서 원문 위치를 보여 줄 OCR 줄 (조제지시서가 가리키는 줄만): {line_id: {text, conf}}."""
    used = {i for d in order.items for i in d.source_line_ids}
    lines = ocr.get("lines", []) if isinstance(ocr, dict) else []
    return {
        str(line["line_id"]): {"text": line.get("text"), "conf": line.get("conf")}
        for line in lines if isinstance(line, dict) and line.get("line_id") in used
    }


def approval_payload(state: dict[str, Any]) -> dict[str, Any]:
    """HITL ①에서 웹에 보일 내용 (web/agent_gateway.Pause의 payload와 같은 모양에 안내 · 제안 · 메모를 더했다)."""
    order = DispensingOrder.model_validate(state["order"])
    order.check_ready_for_web()
    return {
        "hitl_point": HITL_APPROVAL,
        "order": _order_dict(order),
        "ocr_lines": ocr_lines(order, state.get("ocr")),
        "notice": approval_notice(order, state),
        "proposals": list(state.get("proposals") or []),   # 글로 받은 수정 요청 · 복용 약 후보. 약사가 확인한 것만 고쳐서 반영한다
        "memo": state.get("memo"),
    }


def _pharmacist_line(item: Any) -> Any:
    """약사가 정하지 못한 줄(match_method=none)에 약을 골라 넣었으면, 그 줄은 약사가 정한 줄이다 (매칭 방법을 비운다)."""
    if isinstance(item, dict) and item.get("item_code") is not None and item.get("match_method") == "none":
        return {**item, "match_method": None, "match_candidates": []}
    return item


def read_reply(reply: Any, current: DispensingOrder) -> tuple[str, DispensingOrder]:
    """약사 응답을 읽는다. 잘못된 응답이면 ValueError (그래프를 이어 가기 전에 api.py가 먼저 이것으로 거른다).

    응답 모양: {"action": "approve" | "edit", "order": 조제지시서}. 조제지시서의 risk_check는 읽지 않는다
    (고친 뒤에는 다시 대조하므로, 약을 지우면 남은 risk_check가 없는 약을 가리킬 수 있다).
    """
    if not isinstance(reply, dict) or reply.get("action") not in (APPROVE, EDIT):
        raise ValueError("약사 응답의 action은 approve 또는 edit이어야 합니다.")
    data = reply.get("order")
    if not isinstance(data, dict):
        raise ValueError("약사 응답에 조제지시서(order)가 없습니다.")
    data = {k: v for k, v in data.items() if k != "risk_check"}
    if isinstance(data.get("items"), list):
        data["items"] = [_pharmacist_line(item) for item in data["items"]]
    order = DispensingOrder.model_validate(data)
    if order.prescription_id != current.prescription_id:
        raise ValueError(f"다른 처방의 조제지시서입니다 ({order.prescription_id} != {current.prescription_id}).")
    return reply["action"], order


def _ask_pharmacist(deps: Deps, payload: dict[str, Any]) -> Any:
    if deps.ask is not None:
        return deps.ask(payload)
    from langgraph.types import interrupt   # langgraph는 그래프 안에서 멈출 때만 필요하다

    return interrupt(payload)


def approve_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """약사 승인 대기 (HITL ①). 여기서 그래프가 멈추고, 약사가 응답하면 이 노드가 처음부터 다시 돌며 응답을 받는다.

    - 승인하고 약품 줄이 그대로면: approved → 조제 계획으로 간다.
    - 고치기(edit), 또는 약품 줄을 바꾼 채 승인하면: 고친 줄을 넣고 edited → 약품 매칭 · 위험 신호 대조를 다시 하고
      다시 여기서 멈춘다. LLM 구조화는 다시 부르지 않는다.
    위험 신호 대조를 건너뛴 조제지시서면 ValueError (그래프를 잘못 이은 것이다).
    """
    payload = approval_payload(state)
    order = DispensingOrder.model_validate(state["order"])
    action, answered = read_reply(_ask_pharmacist(deps, payload), order)
    if action == APPROVE and answered.items == order.items:
        return {"status": "approved", "notice": APPROVED_NOTICE, "error_code": None}
    data = _order_dict(order)
    data["items"] = [line.model_dump(mode="json") for line in answered.items]
    data["risk_check"] = None
    return {"order": data, "status": "edited", "notice": EDITED_NOTICE, "error_code": None}


def plan_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """승인된 조제지시서로 칸별 조제 계획(A.4)을 만든다 (REQ-AG-04).

    승인되지 않은 상태로 불리면 ValueError (그래프를 잘못 이은 것이다). 계획을 만들 수 없으면(3일 초과 · 칸당 2알 초과 등)
    실패로 표시하고 이유를 남긴다. 그래프는 승인 대기로 돌아가 약사가 고쳐 다시 승인한다 (graph.after_plan).
    """
    if state.get("status") != "approved":
        raise ValueError(f"승인되지 않은 조제지시서로는 조제 계획을 만들지 않습니다 (status={state.get('status')})")
    order = DispensingOrder.model_validate(state["order"])
    try:
        plan = build_plan(order)
    except PlanError as e:
        return {
            "plan": None, "status": "failed", "notice": str(e), "error_code": e.code,
            "errors": [f"[plan] {e.code or '-'} {e}"],
        }
    notice = f"조제 계획을 만들었습니다 ({plan.days}일, 약이 든 칸 {len(plan.plan)}개)."
    return {"plan": plan.model_dump(mode="json"), "status": "planned", "notice": notice}


# ------------------------------------------------------------ 다른 흐름의 자리


def _not_ready(flow: str, state: dict[str, Any]) -> dict[str, Any]:
    """아직 만들지 않은 흐름. 안내를 쌓고, 다른 흐름이 아무것도 하지 않았을 때만 상태를 not_ready로 바꾼다."""
    update: dict[str, Any] = {"notes": [NOT_READY_NOTICES[flow]]}
    if state.get("status") == "received":
        update.update(status="not_ready", notice=NOT_READY_NOTICES[flow])
    return update


def register_meds_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """복용 약 등록 (자리). 글에서 찾은 복용 약 후보는 제안으로만 남아 있다. 15단계에서 채운다."""
    return _not_ready("register_meds", state)


def recheck_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """복용 약 재대조 (자리). 15단계에서 채운다."""
    return _not_ready("recheck", state)


def tray_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """트레이 검수 (자리). 14단계에서 채운다."""
    return _not_ready("tray", state)


def consult_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """상담 답변 (자리). 16단계에서 채운다."""
    return _not_ready("consult", state)
