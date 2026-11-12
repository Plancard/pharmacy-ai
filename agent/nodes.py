"""그래프의 노드 (7 ~ 11단계, 일정표 11/9 ~ 11/13, REQ-AG-01 · 04 · 05, REQ-WEB-02 · 03).

노드는 상태를 받아 바꿀 칸만 dict로 돌려준다. LangGraph 없이도 함수로 불러 시험할 수 있다.
실제 일은 agent/ 의 다른 파일이 하고, 노드는 상태에서 꺼내 넘기고 결과를 다시 담기만 한다.

    supervisor     메시지 하나의 흐름과 순서를 정한다 (supervisor.py의 규칙). 메시지 하나에 한 번만 돈다
    ask_type       사진 종류나 글의 뜻을 되묻는다 (HITL type_select). 약사가 고르면 그것으로 흐름을 다시 정한다
    structure      OCR -> 조제지시서 (규칙 경로 -> 실패하면 LLM 경로)
    match          약품 -> 품목기준코드 (약품 사전 -> 사전에 없으면 식약처 의약품 API), 약사 확인 표시.
                   deps.tools(MCP 클라이언트)가 있으면 MCP 도구 match_drug로 부른다 (12단계)
    risk           처방약끼리 위험 신호 대조 (성분 중복 + DUR 조회가 있으면 병용금기 · 효능군 중복). risk_check에 담는다.
                   deps.tools(MCP)가 있으면 DUR 도구 check_interaction · check_duplicate로 부른다 (13단계)
    approve        약사 승인 대기 (HITL ①). interrupt로 멈추고, 약사 응답(승인 · 고치기)을 받아 이어 간다.
                   글로 받은 제안과 메모를 승인 화면에 함께 보인다
    make_plan      승인된 조제지시서로 칸별 조제 계획(A.4)을 만든다. 승인 전에는 오지 않는다.
                   deps.tools(MCP)가 있으면 조제 도구 build_plan으로 부른다 (14단계)
    register_meds  복용 약 등록 (자리. 15단계)
    recheck        복용 약 재대조 (자리. 15단계)
    tray           트레이 검수: CV(deps.inspect)로 트레이 사진을 조제 계획과 대조한 A.3을 받아 확인한다 (14단계).
                   전 칸 일치면 inspected로 끝나고, 아니면 inspecting으로 tray_review에 넘긴다
    tray_review    검수 확인 대기 (HITL ② inspection). 약사가 확인 완료(confirm)하면 inspected, 재조제(redo)면 planned로 돌아간다.
                   CV는 tray에서 한 번만 부른다 (interrupt에서 이어 갈 때 노드가 처음부터 다시 돌기 때문)
    consult        상담 답변 (자리. 16단계)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.drug_dictionary import DrugDictionary
from agent.drug_match import ApiSearch, match_order
from agent.errors import E_CV_INFER, StructureError
from agent.inspection import (
    CONFIRM,
    HITL_INSPECTION,
    InspectionError,
    cell_lines,
    inspection_notice,
    inspection_payload,
    needs_pharmacist,
    read_inspection,
    read_inspection_reply,
)
from agent.mcp_client import DrugTools, McpToolError
from agent.plan import PlanError, build_plan
from agent.dur import DurLookup, find_all
from agent.risk import LEVEL_LABELS, apply_report, requires_confirmation
from agent.schema import DispensingOrder, Inspection, Plan
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
# 트레이 검수 (CV, 기능분석서 F-CV-05): (사진 ID, 조제 계획 A.4) -> 검수 결과 A.3. 실패하면 예외 (.code에 A.7 코드가 있으면 그것을 쓴다).
# 사진 ID로 파일을 찾는 것은 부르는 쪽(웹 · 스크립트)이 정한다 (agent/cv_adapter.py).
InspectFn = Callable[[str, dict[str, Any]], dict[str, Any]]

# HITL ① (명세서 2.2절). 이름은 웹의 agent_gateway.HITL_POINTS와 같다
HITL_APPROVAL = "rx_approval"
APPROVE, EDIT = "approve", "edit"           # 약사 응답의 action. 웹의 승인 호출은 {"action": "approve", "order": 조제지시서}
MANUAL = "manual"                            # 사유를 적고 진행 {"action": "manual", "order", "reason"} (기능분석서 F-WEB-07 ③)
APPROVED_NOTICE = "약사가 승인했습니다."
MANUAL_NOTICE = "약사가 사유를 적고 진행했습니다."
DANGER_NEEDS_REASON = ("위험 단계 항목이 있어 그대로 승인할 수 없습니다. "
                       "처방을 고치거나(edit), 사유를 적어 진행(manual, reason)해 주세요.")
EDITED_NOTICE = "약사가 고친 내용으로 약품 매칭과 위험 신호 대조를 다시 했습니다."

# 의약품 API를 쓰지 못할 때 조제지시서에 남기는 경고. 사전에서 못 찾은 약이 있을 때만 남긴다.
# 키가 없을 때와 --no-api로 끈 때 모두 남으므로 이유는 적지 않는다.
NO_API_WARNING = "의약품 API를 쓰지 않아 약품 사전(대상 8종)에서만 품목기준코드를 찾았습니다."

NOT_READY_NOTICES = {
    "register_meds": "복용 약 등록은 아직 준비 중입니다.",
    "recheck": "복용 약 재대조 흐름은 아직 준비 중입니다.",
    "consult": "상담 답변은 아직 준비 중입니다.",
}
ASK_TYPE_NOTICE = "사진 종류를 정하지 못했습니다. 처방전 · 트레이 · 약봉투 · 포장 중에서 골라 주세요."
HITL_TYPE = "type_select"                   # 되묻기 (웹의 agent_gateway.HITL_POINTS와 같은 이름)


@dataclass(frozen=True)
class Deps:
    """노드가 쓰는 도구. 테스트에서는 가짜로 바꿔 끼운다."""

    dictionary: DrugDictionary
    structure: StructureFn                # (처방 ID, OCR) -> 조제지시서. 실제로는 graph.default_structure
    api: ApiSearch | None = None          # 의약품 API 조회 (agent.drug_api.default_api). None이면 사전만 쓴다
    ask: AskFn | None = None              # 약사에게 묻기. None이면 LangGraph의 interrupt (그래프 안에서만 쓸 수 있다)
    tools: DrugTools | None = None        # MCP 약품 도구 (agent.mcp_client). 있으면 매칭 · 대조를 MCP로 부른다 (API는 서버 쪽 것을 쓴다)
    dur: DurLookup | None = None          # DUR 조회 (agent.dur_api.DurApi). None이면 병용금기 · 효능군 중복은 대조하지 않는다
    inspect: InspectFn | None = None      # 트레이 검수 (CV). None이면 트레이 사진을 검수하지 못했다고 알린다 (가짜 '전 칸 일치'를 만들지 않는다)


def _order_dict(order: DispensingOrder) -> dict[str, Any]:
    return order.model_dump(mode="json")


# ------------------------------------------------------------ 슈퍼바이저 (규칙)


def supervisor_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """메시지 하나의 흐름과 순서를 정한다. 그래프에서 이 노드로 돌아오는 화살표는 없으므로 메시지 하나에 한 번만 돈다.

    실행할 흐름이 없으면(글을 메모로만 남김) noted로 끝난다.
    """
    plan = plan_message(state.get("photo_kind"), state.get("ocr"), state.get("text"), deps.dictionary,
                        has_prescription=has_prescription(state))
    update = plan.as_update()
    update.update(_open_prescription(state, plan.route))
    if not plan.route:
        update["status"] = "noted"
    return update


def has_prescription(state: dict[str, Any]) -> bool:
    """이 대화에 처방이 있는가 (처방 대화). 상담 대화는 prescription_id가 None이다."""
    return state.get("prescription_id") is not None


def _open_prescription(state: dict[str, Any], route) -> dict[str, Any]:
    """처방 흐름을 도는데 처방 ID가 없으면 대화 ID로 연다 (상담 대화 → 처방 대화, 11/12 결정: 같은 ID).

    웹이 처방을 만들며 prescription_id를 넣어 주는 것이 원칙이고, 빠졌을 때만 여기서 채운다.
    """
    if "prescription" in route and not has_prescription(state):
        return {"prescription_id": state["conversation_id"]}
    return {}


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
    """되묻기의 응답 {"choice": 값}을 읽는다. 사진 종류는 {"doc_type": 값}도 받는다 (기능분석서 F-WEB-05).

    고를 수 없는 값이면 ValueError.
    """
    choice = None
    if isinstance(reply, dict):
        choice = reply.get("choice", reply.get("doc_type") if about == "photo" else None)
    if choice not in ask_choices(about):
        raise ValueError(f"고를 수 있는 값이 아닙니다: {choice!r} (가능: {', '.join(ask_choices(about))})")
    return choice


def ask_type_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """사진 종류나 글의 뜻을 되묻는다 (HITL type_select). 여기서 멈추고, 약사가 고르면 그것으로 흐름을 다시 정한다.

    사진 종류를 고르면 약사의 선택을 따른다 (OCR이 읽은 종류와 달라도 다시 묻지 않는다).
    """
    question = type_question(state)
    choice = read_choice(_ask_pharmacist(deps, question), question["about"])
    plan = plan_after_choice(question["about"], choice, state.get("text"), deps.dictionary,
                             has_prescription=has_prescription(state))
    update = plan.as_update()
    update.update(_open_prescription(state, plan.route))
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
MCP_DUR_FAILED_WARNING = "MCP DUR 도구를 쓰지 못해 Agent가 직접 위험 신호를 대조했습니다"
MCP_PLAN_FAILED_WARNING = "MCP 조제 도구를 쓰지 못해 Agent가 직접 조제 계획을 만들었습니다"


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

    성분 중복은 약품 사전의 주성분으로, 병용금기 · 효능군 중복은 DUR 조회(deps.dur)로 본다.
    대조하지 못한 것(약을 정하지 못한 줄, 주성분을 모르는 약, DUR API 실패 · 미사용)은 조제지시서의 경고에 남는다.
    MCP 도구가 실패하면 직접 대조하고 경고를 남긴다 (매칭 노드와 같다).
    """
    order = DispensingOrder.model_validate(state["order"])
    mcp_failure = None
    report = None
    if deps.tools is not None:
        try:
            report = deps.tools.check_dur(order.items)
        except McpToolError as e:
            mcp_failure = e
    if report is None:
        report = find_all(order.items, deps.dictionary, deps.dur)
    order = apply_report(order, report)
    if mcp_failure is not None:
        order.warnings = [*order.warnings, f"{MCP_DUR_FAILED_WARNING}: {mcp_failure}"]
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

    응답 모양: {"action": "approve" | "edit" | "manual", "order": 조제지시서, "reason": 사유(manual일 때)}.
    조제지시서의 risk_check는 읽지 않는다 (고친 뒤에는 다시 대조하므로, 약을 지우면 남은 risk_check가 없는 약을 가리킬 수 있다).
    위험(danger) 항목이 있는 조제지시서는 그대로 승인(approve)할 수 없다: 고치거나(edit) 사유를 적어 진행(manual)한다.
    """
    if not isinstance(reply, dict) or reply.get("action") not in (APPROVE, EDIT, MANUAL):
        raise ValueError("약사 응답의 action은 approve · edit · manual 중 하나여야 합니다.")
    if reply["action"] == MANUAL and not str(reply.get("reason") or "").strip():
        raise ValueError("manual(사유를 적고 진행)에는 사유(reason)가 있어야 합니다.")
    data = reply.get("order")
    if not isinstance(data, dict):
        raise ValueError("약사 응답에 조제지시서(order)가 없습니다.")
    data = {k: v for k, v in data.items() if k != "risk_check"}
    if isinstance(data.get("items"), list):
        data["items"] = [_pharmacist_line(item) for item in data["items"]]
    order = DispensingOrder.model_validate(data)
    if order.prescription_id != current.prescription_id:
        raise ValueError(f"다른 처방의 조제지시서입니다 ({order.prescription_id} != {current.prescription_id}).")
    danger = current.risk_check is not None and current.risk_check.level == "danger"
    if reply["action"] == APPROVE and danger and order.items == current.items:
        raise ValueError(DANGER_NEEDS_REASON)          # 명세서 2.8절: 위험은 진행을 막고 약사 확인 필수
    return reply["action"], order


def _ask_pharmacist(deps: Deps, payload: dict[str, Any]) -> Any:
    if deps.ask is not None:
        return deps.ask(payload)
    from langgraph.types import interrupt   # langgraph는 그래프 안에서 멈출 때만 필요하다

    return interrupt(payload)


def approve_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """약사 승인 대기 (HITL ①). 여기서 그래프가 멈추고, 약사가 응답하면 이 노드가 처음부터 다시 돌며 응답을 받는다.

    - 승인하고 약품 줄이 그대로면: approved → 조제 계획으로 간다. 위험(danger) 항목이 있으면 그대로 승인할 수 없고
      사유를 적어 진행(manual)해야 한다. 사유는 notes에 남는다 (조제 기록의 승인 이력에 쓴다).
    - 고치기(edit), 또는 약품 줄을 바꾼 채 승인하면: 고친 줄을 넣고 edited → 약품 매칭 · 위험 신호 대조를 다시 하고
      다시 여기서 멈춘다. LLM 구조화는 다시 부르지 않는다.
    위험 신호 대조를 건너뛴 조제지시서면 ValueError (그래프를 잘못 이은 것이다).
    """
    payload = approval_payload(state)
    order = DispensingOrder.model_validate(state["order"])
    reply = _ask_pharmacist(deps, payload)
    action, answered = read_reply(reply, order)
    if action in (APPROVE, MANUAL) and answered.items == order.items:
        if action == MANUAL:
            reason = " ".join(str(reply["reason"]).split())
            return {"status": "approved", "notice": f"{MANUAL_NOTICE} 사유: {reason}", "error_code": None,
                    "notes": [f"위험 신호 확인 후 진행 사유: {reason}"]}
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
    mcp_failure = None
    try:
        plan = None
        if deps.tools is not None:
            try:
                plan = deps.tools.build_plan(order)        # 같은 build_plan을 MCP 서버가 부른다
            except McpToolError as e:                       # 서버가 죽어도 멈추지 않고 직접 만든다
                mcp_failure = e
        if plan is None:
            plan = build_plan(order)
    except PlanError as e:
        return {
            "plan": None, "status": "failed", "notice": str(e), "error_code": e.code,
            "errors": [f"[plan] {e.code or '-'} {e}"],
        }
    update: dict[str, Any] = {
        "plan": plan.model_dump(mode="json"), "status": "planned",
        "notice": f"조제 계획을 만들었습니다 ({plan.days}일, 약이 든 칸 {len(plan.plan)}개).",
    }
    if mcp_failure is not None:
        update["notes"] = [f"{MCP_PLAN_FAILED_WARNING}: {mcp_failure}"]
    return update


# ------------------------------------------------------------ 트레이 검수 (HITL ②)

TRAY_READY = ("planned", "inspected")          # 트레이 사진을 검수할 수 있는 상태: 조제 계획이 있다 (다시 찍은 사진도 받는다)
NO_PLAN_NOTICE = "조제 계획이 아직 없습니다. 조제지시서를 승인한 뒤에 트레이 사진을 올려 주세요."
NO_IMAGE_NOTICE = "트레이 사진(image_id)이 없어 검수하지 못했습니다. 사진을 다시 올려 주세요."
NO_CV_NOTICE = "트레이 검수(CV)가 연결되지 않아 검수하지 못했습니다."
CONFIRMED_NOTICE = "약사가 검수 결과를 확인했습니다."
REDO_NOTICE = "재조제: 트레이를 다시 채워 사진을 올려 주세요."


def _names(state: dict[str, Any]) -> dict[str, str]:
    """화면에 보일 약 이름: 조제지시서의 품목기준코드 -> 이름."""
    items = (state.get("order") or {}).get("items") or []
    return {i["item_code"]: i["name"] for i in items if isinstance(i, dict) and i.get("item_code")}


def _tray_failed(notice: str, code: str | None, detail: str) -> dict[str, Any]:
    """검수하지 못함. 상태(planned)는 그대로 두어 웹이 다시 찍은 사진을 기다린다 (waiting_tray). 이유와 코드만 남긴다."""
    return {"notice": notice, "error_code": code, "errors": [f"[tray] {code or '-'} {detail}"]}


def tray_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """트레이 사진을 지금 조제 계획과 대조한다 (CV deps.inspect, REQ-CV-04 · 05). 판정은 CV가 하고 Agent는 바꾸지 않는다.

    - 조제 계획이 없으면(승인 전) 안내만 남긴다.
    - CV가 실패하거나(트레이 못 찾음 등) 결과가 지금 계획과 맞지 않으면 실패 코드와 이유를 남기고 planned에 머문다.
    - 전 칸 일치면 inspected. ok가 아닌 칸이 있으면 inspecting → tray_review에서 약사 확인 (HITL ②).
    """
    if state.get("status") not in TRAY_READY or not state.get("plan"):
        update: dict[str, Any] = {"notice": NO_PLAN_NOTICE, "notes": [NO_PLAN_NOTICE]}
        if state.get("status") == "received":           # 이 메시지로 아무것도 하지 않았다: 안내만 남기고 끝남
            update["status"] = "noted"
        return update
    image_id = state.get("image_id")
    if not image_id:
        return _tray_failed(NO_IMAGE_NOTICE, E_CV_INFER, "image_id 없음")
    if deps.inspect is None:
        return _tray_failed(NO_CV_NOTICE, E_CV_INFER, "deps.inspect 없음")
    plan = Plan.model_validate(state["plan"])
    try:
        found = deps.inspect(image_id, plan.model_dump(mode="json"))
    except Exception as e:                       # CV 오류(E_TRAY_NOT_FOUND 등)도 모델 오류도 조제를 멈추지 않는다 (REQ-NF-06)
        code = getattr(e, "code", None)
        code = code if isinstance(code, str) and code.startswith("E_") else E_CV_INFER
        return _tray_failed(f"트레이 검수에 실패했습니다: {e}", code, f"{type(e).__name__}: {e}")
    try:
        inspection = read_inspection(found, plan)
    except InspectionError as e:
        return _tray_failed(str(e), e.code, str(e))
    update: dict[str, Any] = {"inspection": inspection.model_dump(mode="json"), "error_code": None,
                              "notice": inspection_notice(inspection)}
    update["status"] = "inspecting" if needs_pharmacist(inspection) else "inspected"
    return update


def tray_review_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """검수 확인 대기 (HITL ② inspection). 여기서 멈추고, 약사가 확인 완료하면 inspected, 재조제면 planned로 돌아간다.

    확인한 칸(불일치 · 확인 필요)은 notes에 남긴다 (조제 기록의 승인 이력에 쓴다).
    """
    inspection = Inspection.model_validate(state["inspection"])
    plan = Plan.model_validate(state["plan"])
    names = _names(state)
    action = read_inspection_reply(_ask_pharmacist(deps, inspection_payload(inspection, plan, names)))
    if action == CONFIRM:
        checked = "; ".join(cell_lines(inspection, names))
        return {"status": "inspected", "notice": CONFIRMED_NOTICE, "notes": [f"검수 확인(약사): {checked}"]}
    return {"status": "planned", "notice": REDO_NOTICE, "notes": ["재조제 요청(약사): 트레이를 다시 채웁니다"]}


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


def consult_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """상담 답변 (자리). 16단계에서 채운다."""
    return _not_ready("consult", state)
