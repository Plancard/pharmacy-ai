"""그래프의 노드 (7 ~ 11단계, 명세서 2.2절, REQ-AG-01 · 04 · 05, REQ-WEB-02 · 03).

노드는 상태를 받아 바꿀 칸만 dict로 돌려준다. LangGraph 없이도 함수로 불러 시험할 수 있다.
실제 일은 agent/ 의 다른 파일이 하고, 노드는 상태에서 꺼내 넘기고 결과를 다시 담기만 한다.

    router     사진 한 장의 흐름을 정한다 (router.py의 규칙: 처방전 / 트레이). 메시지 하나에 한 번만 돈다
    structure      OCR -> 조제지시서 (규칙 경로 -> 실패하면 LLM 경로)
    match          약품 -> 품목기준코드 (약품 사전 -> 사전에 없으면 식약처 의약품 API), 약사 확인 표시.
                   deps.tools(MCP 클라이언트)가 있으면 MCP 도구 match_drug로 부른다 (12단계)
    risk           처방약끼리 위험 신호 대조 (성분 중복 + DUR 조회가 있으면 병용금기 · 효능군 중복). risk_check에 담는다.
                   deps.tools(MCP)가 있으면 DUR 도구 check_interaction · check_duplicate로 부른다 (13단계)
    approve        약사 승인 대기 (HITL ①). interrupt로 멈추고, 약사 응답(승인 · 고치기)을 받아 이어 간다.
                   승인하면 승인 카드의 조제 계획을 그대로 확정해 planned(트레이 대기)로 끝난다
    make_plan      조제지시서로 칸별 조제 계획(A.4)을 만든다. 승인 전에 만들어 승인 카드에 함께 보인다 (명세서 2.2절 5단계).
                   만들 수 없으면(3일 초과 E_PLAN_RANGE · 약을 정하지 못한 줄) 계획 없이 승인 대기로 가서 이유를 보인다
                   deps.tools(MCP)가 있으면 조제 도구 build_plan으로 부른다 (14단계)
    tray           트레이 검수: CV(deps.inspect)로 트레이 사진을 조제 계획과 대조한 A.3을 받아 확인한다 (14단계).
                   전 칸 일치면 inspected로 끝나고, 아니면 inspecting으로 tray_review에 넘긴다
    tray_review    검수 확인 대기 (HITL ② inspection). 약사가 확인 완료(confirm)하면 inspected, 재조제(redo)면 planned로 돌아간다.
                   CV는 tray에서 한 번만 부른다 (interrupt에서 이어 갈 때 노드가 처음부터 다시 돌기 때문)
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
from agent.plan import NOT_TARGET_PLAN, PlanError, build_plan, not_target_lines
from agent.explain import ExplainFn, attach_evidence, explain_findings, passages_for, recheck
from agent.investigate import investigate_risks, local_evidence
from agent.llm import load_prompt
from agent.react import ToolChatFn, ToolFailure
from agent.errors import ConfigError
from agent.rag import RagIndex
from agent.dur import DurLookup, find_all
from agent.risk import LEVEL_LABELS, apply_report, requires_confirmation
from agent.schema import DispensingOrder, Inspection, Plan
from agent.router import plan_message

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
UNMATCHED_CANNOT_APPROVE = ("약을 정하지 못한 줄(품목기준코드 없음)이 있어 승인할 수 없습니다. "
                            "약을 고르거나 줄을 지운 뒤(edit) 다시 승인해 주세요.")      # 기능분석서 v1.2 5.1절
NO_PLAN_CANNOT_APPROVE = "조제 계획을 만들지 못해 승인할 수 없습니다. 조제지시서를 고친 뒤(edit) 다시 승인해 주세요."
EDITED_NOTICE = "약사가 고친 줄을 다시 검수하고 위험 신호를 다시 대조했습니다."

# 의약품 API를 쓰지 못할 때 조제지시서에 남기는 경고. 사전에서 못 찾은 약이 있을 때만 남긴다.
# 키가 없을 때와 --no-api로 끈 때 모두 남으므로 이유는 적지 않는다.
NO_API_WARNING = "의약품 API를 쓰지 않아 약품 사전(대상 8종)에서만 품목기준코드를 찾았습니다."


@dataclass(frozen=True)
class Deps:
    """노드가 쓰는 도구. 테스트에서는 가짜로 바꿔 끼운다."""

    dictionary: DrugDictionary
    structure: StructureFn                # (처방 ID, OCR) -> 조제지시서. 실제로는 graph.default_structure
    api: ApiSearch | None = None          # 의약품 API 조회 (agent.drug_api.default_api). None이면 사전만 쓴다
    ask: AskFn | None = None              # 약사에게 묻기. None이면 LangGraph의 interrupt (그래프 안에서만 쓸 수 있다)
    tools: DrugTools | None = None        # MCP 약품 도구 (agent.mcp_client). 있으면 매칭 · 대조를 MCP로 부른다 (API는 서버 쪽 것을 쓴다)
    dur: DurLookup | None = None          # DUR 조회 (agent.dur_api.DurApi). None이면 병용금기 · 효능군 중복은 대조하지 않는다
    rag: RagIndex | None = None           # 위험 신호 설명의 근거 색인 (rag.load_index). None이면 근거 없이 설명한다
    embed: Any = None                     # 검색어 임베딩 (embed.embed_texts). 색인에 임베딩이 있을 때만 쓴다. None이면 BM25만
    explain: ExplainFn | None = None      # 위험 신호 AI 설명 (llm.explain_llm). None이면 규칙 문장만 (18단계)
    react: ToolChatFn | None = None       # AI 조사의 도구 호출 대화 (llm.tool_chat, 11/16). 위험 신호 근거 조사. None이면 끈다
    inspect: InspectFn | None = None      # 트레이 검수 (CV). None이면 트레이 사진을 검수하지 못했다고 알린다 (가짜 '전 칸 일치'를 만들지 않는다)


def _order_dict(order: DispensingOrder) -> dict[str, Any]:
    return order.model_dump(mode="json")


# ------------------------------------------------------------ 라우터 (규칙)


def router_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """사진 한 장의 흐름을 정한다. 그래프에서 이 노드로 돌아오는 화살표는 없으므로 메시지 하나에 한 번만 돈다.

    실행할 흐름이 없으면(사진 종류 없음 · 이미 처방전이 있는 처방에 또 온 처방전 등) 안내만 남기고 noted로 끝난다.
    """
    plan = plan_message(state.get("photo_kind"), state.get("ocr"), has_order=state.get("order") is not None)
    update = plan.as_update()
    update.update(_open_prescription(state, plan.route))
    if not plan.route:
        update["status"] = "noted" if state.get("status") == "received" else state.get("status")
        update["notes"] = [plan.notice]
    return update


def _open_prescription(state: dict[str, Any], route) -> dict[str, Any]:
    """처방 흐름을 도는데 처방 ID가 없으면 대화 ID로 채운다 (처방 1건 = 대화 1개, 같은 ID).

    웹이 처방을 만들며 prescription_id를 넣어 주는 것이 원칙이고, 빠졌을 때만 여기서 채운다.
    """
    if "prescription" in route and state.get("prescription_id") is None:
        return {"prescription_id": state["conversation_id"]}
    return {}


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
            "errors": [f"[structure] {e.code or '-'} {e}"], "failed_line_ids": failed_line_ids(state.get("ocr"), e),
        }
    return {"order": _order_dict(order), "status": "structured", "notice": None, "failed_line_ids": []}


def failed_line_ids(ocr: Any, error: StructureError | None = None) -> list[int]:
    """구조화 실패 때 웹이 처방전 사진에서 강조할 OCR 줄 번호(A.1 line_id) (11/17 회의 Q6, 11/18 결정).

    1. 규칙 경로가 특정 약 줄 · 칸에서 실패했으면(예: '2번째 약품 줄: 1회량 칸을 찾지 못했습니다') 그 줄의 칸들만.
    2. 아니면(LLM까지 실패 등) 약품 표 본문 칸 전체 (머리글 아래 ~ '주사제' 위).
    3. 줄을 특정할 수 없는 실패(글자 없음 · 머리글을 못 찾음 등)는 빈 목록. 원문 크기는 AgentApi.result의 ocr_size.
    """
    if error is not None and error.line_ids:
        return list(error.line_ids)
    from agent.structure import table_lines

    return [line["line_id"] for line in table_lines(ocr)]


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
    edited = state.get("status") == "edited"
    previous = state.get("previous_order") if edited else None
    kept = _unchanged_lines(order, previous)                  # {줄 번호: 고치기 전의 매칭된 줄} (Q7: 바뀐 줄만 다시)
    redo = [i for i in range(len(order.items)) if i not in kept]
    notes: list[str] = []
    if edited:
        changed = ", ".join(f"{i + 1}번째({order.items[i].name})" for i in redo)
        notes.append(f"약사 수정 후 다시 검수한 줄: {changed}" if redo else "약사 수정 후 다시 검수한 줄: 없음 (지운 줄만 있음)")
    items = [kept.get(i) for i in range(len(order.items))]
    warnings = list(order.warnings)
    api_used = deps.api is not None
    if redo:
        redo_lines = [order.items[i].model_copy(update={"times_note": None, "value_conf": None, "printed_name": None}) if edited else order.items[i] for i in redo]   # 약사가 고친 줄은 시점과 값을 이미 봤다 (옛 신뢰도·처방전 이름은 고친 값에 맞지 않는다)
        sub = order.model_copy(update={"items": redo_lines, "warnings": []})
        matched_sub, api_used = _match(sub, deps)
        new_lines = matched_sub.items
        if edited:
            new_lines = [_keep_pharmacist_choice(line, new) for line, new in zip(sub.items, new_lines)]
        for i, line in zip(redo, new_lines):
            items[i] = line
        warnings += [w for w in matched_sub.warnings if w not in warnings]
    matched = DispensingOrder.model_validate({**order.model_dump(mode="json"), "warnings": warnings,
                                              "items": [d.model_dump(mode="json") for d in items]})
    if not api_used and not matched.all_matched and NO_API_WARNING not in matched.warnings:
        matched.warnings = [*matched.warnings, NO_API_WARNING]
    update: dict[str, Any] = {"status": "matched", "order": _order_dict(_mark_not_target(matched, deps.dictionary)),
                              "changed_items": redo if edited else None}
    if notes:
        update["notes"] = notes
    return update


def _match(order: DispensingOrder, deps: Deps) -> tuple[DispensingOrder, bool]:
    """약 줄들의 품목기준코드를 정한다: MCP 도구(match_drug)가 있으면 서버에서, 없거나 실패하면 직접 (같은 match_order)."""
    if deps.tools is not None:
        try:
            return deps.tools.match_drug(order)
        except McpToolError as e:                             # 서버가 죽어도 조제를 멈추지 않고 직접 부른다
            matched = match_order(order, deps.dictionary, api=deps.api)
            matched.warnings = [*matched.warnings, f"{MCP_FAILED_WARNING}: {e}"]
            return matched, deps.api is not None
    return match_order(order, deps.dictionary, api=deps.api), deps.api is not None


def _line_key(line) -> tuple:
    """줄이 바뀌었는지 보는 값: 약사가 고칠 수 있는 칸 (이름 · 품목기준코드 · 1회 알 수 · 복용 시점 · 일수)."""
    return (line.name, line.item_code, line.count_per_dose, tuple(line.times), line.days)


def _unchanged_lines(order: DispensingOrder, previous: dict[str, Any] | None) -> dict[int, Any]:
    """약사가 고친 조제지시서에서 바뀌지 않은 줄 → 고치기 전의 매칭된 줄 (매칭 결과 · 확인 표시를 그대로 쓴다, Q7).

    같은 줄이 여러 번 있으면 앞에서부터 하나씩 짝짓는다. 고치기 전이 없으면(처음 매칭) 빈 dict.
    """
    if not previous:
        return {}
    before = DispensingOrder.model_validate(previous).items
    pool: dict[tuple, list] = {}
    for line in before:
        pool.setdefault(_line_key(line), []).append(line)
    kept = {}
    for i, line in enumerate(order.items):
        same = pool.get(_line_key(line))
        if same:
            kept[i] = same.pop(0).model_copy(update={"source_line_ids": line.source_line_ids})
    return kept


NOT_TARGET_REASON = "대상 8종이 아닌 약품입니다. 8종 중 하나로 고쳐야 승인할 수 있습니다."


def _mark_not_target(order: DispensingOrder, dictionary) -> DispensingOrder:
    """대상 8종이 아닌 약 줄에 약사 확인 표시와 이유를 붙인다 (명세서 REQ-AG-03: 대상 외 약품은 승인 불가)."""
    items = []
    for d in order.items:
        if d.item_code is not None and dictionary.by_item_code(d.item_code) is None and NOT_TARGET_REASON not in d.check_reasons:
            d = d.model_copy(update={"needs_pharmacist_check": True, "check_reasons": [*d.check_reasons, NOT_TARGET_REASON]})
        items.append(d)
    return order.model_copy(update={"items": items})


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
    update: dict[str, Any] = {"status": "checked"}
    if order.risk_check is not None and order.risk_check.findings:
        names = {d.item_code: d.name for d in order.items if d.item_code}
        findings = _reuse_explanations(order.risk_check.findings, state.get("previous_order"))
        todo = [i for i, f in enumerate(findings) if f.explanation is None and not f.evidence]
        if todo:                                   # 고치기 전에 설명한 항목은 다시 부르지 않는다 (Q7)
            explained, note = _explain(deps, [findings[i] for i in todo], names)
            for i, f in zip(todo, explained):
                findings[i] = f
            if note:
                update["notes"] = [note]
        order = order.model_copy(update={"risk_check": order.risk_check.model_copy(update={"findings": findings})})
    update["order"] = _order_dict(order)
    return update


def _finding_key(f) -> tuple:
    return (f.type, tuple(sorted(f.item_codes)), f.level, f.message)


def _reuse_explanations(findings, previous: dict[str, Any] | None) -> list:
    """고치기 전에도 똑같이 있던 위험 항목(유형 · 약 · 단계 · 문장이 같음)은 그때의 AI 설명과 근거를 그대로 쓴다 (Q7)."""
    findings = list(findings)
    old = ((previous or {}).get("risk_check") or {}).get("findings") or []
    if not old:
        return findings
    from agent.schema import RiskFinding

    done = {_finding_key(f): f for f in (RiskFinding.model_validate(x) for x in old) if f.explanation or f.evidence}
    return [f.model_copy(update={"explanation": done[_finding_key(f)].explanation,
                                 "evidence": done[_finding_key(f)].evidence})
            if _finding_key(f) in done and f.explanation is None and not f.evidence else f for f in findings]


def _evidence_search(deps: Deps):
    """AI 조사의 근거 검색: MCP 서버에 search_evidence가 있으면 서버에서, 없으면 근거 색인(deps.rag)을 직접. 둘 다 없으면 None."""
    if deps.tools is not None and "search_evidence" in deps.tools.tool_names():
        def search(query: str, codes, k: int) -> list[dict[str, Any]]:
            try:
                return deps.tools.call("search_evidence", query=query, item_codes=list(codes), k=k)["results"]
            except McpToolError as e:
                raise ToolFailure(str(e)) from None
        return search
    if deps.rag is not None and len(deps.rag):
        return local_evidence(deps.rag, deps.embed)
    return None


def _explain(deps: Deps, findings, names: dict[str, str]):
    """위험 신호 항목에 AI 설명과 근거(evidence, A.5)를 붙인다 (18단계). 단계 · 유형 · 규칙 문장은 그대로.
    키가 없거나 실패하면 설명 없이 + 안내. 근거는 색인(deps.rag)이 있으면 설명과 상관없이 붙인다 (명세서 REQ-AG-06).

    AI 조사(deps.react)와 근거 색인이 있으면 LLM이 근거 검색 도구로 찾아 가며 설명한다 (11/16 ReAct).
    조사를 끝내지 못하면 한 번 부르는 설명(deps.explain)으로 대신한다.
    """
    passages = passages_for(deps.rag, findings, names)       # 근거 색인이 없으면 빈 목록
    if deps.react is not None:
        got = investigate_risks(findings, names, _evidence_search(deps), deps.react, load_prompt("react_risk"))
        if got is not None:
            explained, note = got
            # AI가 인용하지 않은 항목도 근거를 받고, 받은 근거로 설명을 한 번 더 거른다 (11/27, 한 번 부르는 설명과 같은 기준)
            return recheck(attach_evidence(explained, passages), names), note
    if deps.explain is None:
        return attach_evidence(findings, passages), None

    def call(text: str) -> dict[str, Any]:
        try:
            return deps.explain(text)
        except ConfigError as e:                 # 키가 없으면 설명 없이 (explain_findings가 StructureError를 받는다)
            raise StructureError(str(e)) from None
    return explain_findings(findings, names, call, passages)


def approval_notice(order: DispensingOrder, state: dict[str, Any]) -> str:
    """승인 화면에 보일 안내. 조제 계획을 만들지 못했으면 그 이유를 앞에 붙인다 (격자 대신 오류와 수정 안내, F-WEB-07 ③)."""
    notice = "약사 승인을 기다립니다."
    flagged = len(order.flagged_indices)
    if flagged:
        notice += f" 확인이 필요한 약품이 {flagged}줄 있습니다."
    risk = order.risk_check
    if risk is not None and risk.findings:
        notice += f" 위험 신호가 {len(risk.findings)}건 있습니다({LEVEL_LABELS[risk.level]})."
        if requires_confirmation(risk.level):
            notice += " 약사가 확인한 뒤에 승인할 수 있습니다."
    if state.get("plan") is None and state.get("notice"):
        notice = f"{state.get('notice')} 고친 뒤 다시 승인해 주세요. " + notice
    return notice


def ocr_lines(order: DispensingOrder, ocr: Any) -> dict[str, dict[str, Any]]:
    """승인 화면에서 원문 위치를 보여 줄 OCR 줄 (조제지시서가 가리키는 줄만): {line_id: {text, conf, bbox}}.

    bbox는 A.1 이미지의 픽셀 좌표다 (실제 CV OCR은 펴진 처방전 이미지 기준. 웹 F-WEB-06이 그 이미지 위에 위치를 그린다. 크기는 ocr_size).
    """
    used = {i for d in order.items for i in d.source_line_ids}
    lines = ocr.get("lines", []) if isinstance(ocr, dict) else []
    return {
        str(line["line_id"]): {"text": line.get("text"), "conf": line.get("conf"), "bbox": line.get("bbox")}
        for line in lines if isinstance(line, dict) and line.get("line_id") in used
    }


def ocr_size(ocr: Any) -> dict[str, int] | None:
    """A.1 이미지 크기 {width, height} (실제 CV OCR은 펴진 처방전 이미지). 웹이 bbox를 화면 크기에 맞춰 줄일 때 쓴다."""
    if isinstance(ocr, dict) and ocr.get("width") and ocr.get("height"):
        return {"width": ocr["width"], "height": ocr["height"]}
    return None


def approval_payload(state: dict[str, Any]) -> dict[str, Any]:
    """HITL ①에서 웹에 보일 내용 (web/agent_gateway.Pause의 payload와 같은 모양에 계획 · 안내를 더했다).

    plan은 승인 카드의 칸 배치(A.4, 명세서 REQ-AG-04)다. 만들지 못했으면 None이고 error_code(E_PLAN_RANGE 등)와
    notice에 이유가 있다. 그때는 승인할 수 없고 고치기(edit)만 받는다.
    """
    order = DispensingOrder.model_validate(state["order"])
    order.check_ready_for_web()
    return {
        "hitl_point": HITL_APPROVAL,
        "order": _order_dict(order),
        "plan": state.get("plan"),
        "error_code": state.get("error_code") if state.get("plan") is None else None,
        "ocr_lines": ocr_lines(order, state.get("ocr")),
        "ocr_size": ocr_size(state.get("ocr")),
        "notice": approval_notice(order, state),
        "changed_items": state.get("changed_items") or [],     # 약사가 고친 뒤 다시 검수한 줄 번호(0부터, Q7)
    }


def _pharmacist_line(item: Any) -> Any:
    """약사가 정하지 못한 줄(match_method=none)에 약을 골라 넣었으면, 그 줄은 약사가 정한 줄이다 (매칭 방법을 비운다)."""
    if isinstance(item, dict) and item.get("item_code") is not None and item.get("match_method") == "none":
        return {**item, "match_method": None, "match_candidates": []}
    return item


def read_reply(reply: Any, current: DispensingOrder, plan_ready: bool = True) -> tuple[str, DispensingOrder]:
    """약사 응답을 읽는다. 잘못된 응답이면 ValueError (그래프를 이어 가기 전에 api.py가 먼저 이것으로 거른다).

    응답 모양: {"action": "approve" | "edit" | "manual", "order": 조제지시서, "reason": 사유(manual일 때)}.
    조제지시서의 risk_check는 읽지 않는다 (고친 뒤에는 다시 대조하므로, 약을 지우면 남은 risk_check가 없는 약을 가리킬 수 있다).
    약품 줄을 그대로 둔 채 진행(approve · manual)할 수 없는 경우 (그때는 고치기 edit만 받는다):
    - 약을 정하지 못한 줄(item_code 없음)이 있다 (기능분석서 v1.2 5.1절, 화면설계서 7장)
    - 조제 계획을 만들지 못했다 (plan_ready=False: 3일 초과 등)
    - 위험(danger) 항목이 있는데 approve다: 고치거나(edit) 사유를 적어 진행(manual)한다
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
    if reply["action"] in (APPROVE, MANUAL) and order.items == current.items:
        if not order.all_matched:
            raise ValueError(UNMATCHED_CANNOT_APPROVE)
        if not plan_ready:
            raise ValueError(NO_PLAN_CANNOT_APPROVE)
        danger = current.risk_check is not None and current.risk_check.level == "danger"
        if reply["action"] == APPROVE and danger:
            raise ValueError(DANGER_NEEDS_REASON)      # 명세서 2.8절: 위험은 진행을 막고 약사 확인 필수
    return reply["action"], order


def _ask_pharmacist(deps: Deps, payload: dict[str, Any]) -> Any:
    if deps.ask is not None:
        return deps.ask(payload)
    from langgraph.types import interrupt   # langgraph는 그래프 안에서 멈출 때만 필요하다

    return interrupt(payload)


def approve_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """약사 승인 대기 (HITL ①). 여기서 그래프가 멈추고, 약사가 응답하면 이 노드가 처음부터 다시 돌며 응답을 받는다.

    - 승인하고 약품 줄이 그대로면: 승인 카드의 조제 계획을 그대로 확정하고 planned(트레이 대기)로 끝난다.
      위험(danger) 항목이 있으면 그대로 승인할 수 없고 사유를 적어 진행(manual)해야 한다. 사유는 notes에 남는다
      (조제 기록의 승인 이력에 쓴다). 약을 정하지 못한 줄이 있거나 계획이 없으면 진행할 수 없다 (read_reply).
    - 고치기(edit), 또는 약품 줄을 바꾼 채 승인하면: 고친 줄을 넣고 edited → 약품 매칭 · 위험 신호 대조 · 조제 계획을
      다시 하고 다시 여기서 멈춘다. 바뀐 줄만 다시 매칭하고, 고치기 전에도 있던 위험 항목은 AI 설명 · 근거를 다시 쓴다
      (11/17 회의 Q7). 중복 · 금기 대조와 조제 계획은 전체로 다시 한다. LLM 구조화는 다시 부르지 않는다.
    위험 신호 대조를 건너뛴 조제지시서면 ValueError (그래프를 잘못 이은 것이다).
    """
    payload = approval_payload(state)
    order = DispensingOrder.model_validate(state["order"])
    reply = _ask_pharmacist(deps, payload)
    action, answered = read_reply(reply, order, plan_ready=state.get("plan") is not None)
    if action in (APPROVE, MANUAL) and answered.items == order.items:
        if action == MANUAL:
            reason = " ".join(str(reply["reason"]).split())
            return {"status": "planned", "notice": f"{MANUAL_NOTICE} 사유: {reason}", "error_code": None, "previous_order": None,
                    "changed_items": None,
                    "notes": [f"위험 신호 확인 후 진행 사유: {reason}"]}
        return {"status": "planned", "notice": APPROVED_NOTICE, "error_code": None, "previous_order": None,
                "changed_items": None}
    data = _order_dict(order)
    data["items"] = [line.model_dump(mode="json") for line in answered.items]
    data["risk_check"] = None
    return {"order": data, "plan": None, "status": "edited", "notice": EDITED_NOTICE, "error_code": None,
            "previous_order": _order_dict(order)}      # 고치기 직전 (매칭 결과 · AI 설명 · 근거를 다시 쓰려고, Q7)


def plan_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """조제지시서로 칸별 조제 계획(A.4)을 만든다 (REQ-AG-04). 승인 전에 만들어 승인 카드에 함께 보인다 (명세서 2.2절).

    위험 신호 대조를 거치지 않은 조제지시서로 불리면 ValueError (그래프를 잘못 이은 것이다).
    계획을 만들 수 없으면(3일 초과 E_PLAN_RANGE · 약을 정하지 못한 줄) 계획 없이 이유와 코드를 남긴다. 그래프는 그대로
    승인 대기로 가고, 약사는 고치기(edit)만 할 수 있다 (read_reply). 상태는 바꾸지 않는다 (승인 전 단계).
    """
    order = DispensingOrder.model_validate(state["order"])
    if order.risk_check is None:
        raise ValueError("위험 신호 대조를 거치지 않은 조제지시서로는 조제 계획을 만들지 않습니다")
    not_target = not_target_lines(order, deps.dictionary)
    if not_target:                                   # CV가 모르는 약이라 계획을 만들어도 검수할 수 없다 (REQ-AG-03)
        notice = f"{NOT_TARGET_PLAN}: {', '.join(not_target)}"
        return {"plan": None, "notice": notice, "error_code": None, "errors": [f"[plan] - {notice}"]}
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
        return {"plan": None, "notice": str(e), "error_code": e.code, "errors": [f"[plan] {e.code or '-'} {e}"]}
    update: dict[str, Any] = {"plan": plan.model_dump(mode="json"), "notice": None, "error_code": None}
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
