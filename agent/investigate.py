"""AI 조사 (판단, 11/16) — ReAct 루프(react.py)를 쓰는 두 곳. 둘 다 규칙의 결과를 바꾸지 않고 덧붙이기만 한다.

도구는 MCP 서버의 것을 쓴다 (nodes가 mcp_client.react_tools · search_evidence로 넘긴다). MCP 서버를 쓰지 않으면
같은 판단 함수(drug_match.search_drug_result · check_code_result, rag.evidence_search)를 직접 부르는 도구로 대신한다.

1) 약품 매칭 실패 → 후보 조사
    suggest_candidates(order, tools, chat, prompt)  ->  (새 조제지시서, 안내 목록)
    약을 정하지 못한 줄(item_code None)마다 LLM이 search_drug · check_item_code 도구로 찾아 후보를 낸다.
    - item_code는 여전히 None이다. 후보(match_candidates)에 'AI 조사: …' 이유로 더할 뿐, 고르는 것은 약사다 (REQ-AG-03).
    - LLM이 낸 코드는 이번 조사에서 도구가 돌려준 코드만 받는다 (지어낸 코드는 버린다). 이름은 도구 결과의 이름을 쓴다.
    - 함량이 처방전과 다른 제품은 버린다 (다른 약이다). 이미 있는 후보와 같은 약은 더하지 않는다. 후보는 많아야 3개.

2) 위험 신호 → 근거 조사
    investigate_risks(findings, names, search, chat, prompt)  ->  (설명을 붙인 항목들, 안내) | None
    LLM이 search_evidence 도구로 관련 약의 허가정보(RAG 색인)를 찾아 항목마다 설명을 쓴다.
    - 단계 · 유형 · 관련 약 · 규칙 문장은 그대로. 설명은 explain.acceptable로 거른다 (위험을 낮추는 말 등).
    - 근거 번호(E1 …)는 도구가 준 것만 받고, 인용한 조각을 항목의 evidence(A.5: 제목 · 출처 · 발췌)에 담는다.
    - 검색은 위험 신호에 걸린 약의 문서 안에서만 한다.
    - 조사를 끝내지 못하면 None (부른 쪽이 한 번 부르는 설명 explain.explain_findings로 대신한다).

판단 층이다. LLM(chat) · 의약품 API(api) · 임베딩(embed)은 함수로 받는다.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from agent.drug_dictionary import DrugDictionary
from agent.drug_match import ApiSearch, check_code_result, search_drug_result, strengths_conflict
from agent.drug_names import match_key, product_key, similarity
from agent.explain import acceptable, evidence_item, explain_input
from agent.rag import EVIDENCE_TEXT, RagIndex, evidence_search
from agent.react import ReactResult, Tool, ToolChatFn, ToolFailure, run_react, trace
from agent.schema import MAX_CANDIDATES, DispensingOrder, DrugLine, Evidence, MatchCandidate, RiskFinding

EmbedFn = Callable[[list[str]], list[list[float]]]
EvidenceSearch = Callable[[str, Sequence[str], int], list[dict[str, Any]]]   # (질문, 품목기준코드들, 개수) -> 근거 줄들

MAX_LINES = 3             # 처방전 한 장에서 후보를 조사할 줄 수 (시간 · 비용)
TOOL_RESULTS = 5          # 근거 검색 한 번에 받는 조각 수
REASON_CHARS = 80
AI_REASON = "AI 조사"
AI_CHECK = "AI가 도구로 찾은 후보가 있습니다. 약사가 골라야 정해집니다."
NO_SUGGESTION = "AI 후보 조사를 끝내지 못했습니다"


# ------------------------------------------------------------ 1) 약품 매칭 실패 → 후보 조사


DRUG_TOOL_NAMES = ["search_drug", "check_item_code"]


def drug_tools(dictionary: DrugDictionary, api: ApiSearch | None) -> list[Tool]:
    """MCP 서버를 쓰지 않을 때의 search_drug · check_item_code (서버와 같은 함수 · 같은 이름). API가 없으면 check_item_code는 없다.

    MCP 서버가 있으면 nodes가 mcp_client.react_tools(DRUG_TOOL_NAMES)로 서버의 도구를 대신 넘긴다.
    """
    def search(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["name"]).strip()
        if not name:
            raise ToolFailure("name이 비었습니다")
        return search_drug_result(name, dictionary, api)

    def check(args: dict[str, Any]) -> dict[str, Any]:
        code, name = str(args["item_code"]).strip(), str(args["name"]).strip()
        if not (code.isdigit() and len(code) == 9):
            raise ToolFailure("item_code는 9자리 숫자입니다")
        return check_code_result(code, name, dictionary, api)

    tools = [Tool("search_drug", "약 이름으로 찾는다. 약품 사전에서 이름이 비슷한 약과 (있으면) 의약품 API의 제품을 돌려준다.",
                  {"type": "object", "properties": {"name": {"type": "string", "description": "찾을 약 이름"}},
                   "required": ["name"], "additionalProperties": False}, search)]
    if api is not None:
        tools.append(Tool("check_item_code", "품목기준코드(9자리)가 이 이름의 제품 코드가 맞는지 의약품 API로 확인한다. 판정: 일치 · 불일치 · 없음.",
                          {"type": "object", "properties": {"item_code": {"type": "string"}, "name": {"type": "string"}},
                           "required": ["item_code", "name"], "additionalProperties": False}, check))
    return tools


def _seen_drugs(result: ReactResult) -> dict[str, str]:
    """이번 조사에서 도구가 돌려준 약 {품목기준코드: 이름}. LLM의 답은 이 안의 코드만 받는다."""
    seen: dict[str, str] = {}
    for out in result.outputs("search_drug"):
        for row in [*(out.get("dictionary") or []), *(out.get("api") or [])]:
            seen.setdefault(str(row["item_code"]), str(row["name"]))
    for out in result.outputs("check_item_code"):
        for row in out.get("similar") or []:
            seen.setdefault(str(row["item_code"]), str(row["name"]))
    return seen


def line_question(line: DrugLine) -> str:
    """LLM에 보내는 줄 정보: 처방전에 적힌 약 이름 · 이미 있는 후보 · 정하지 못한 이유. 환자 정보는 없다."""
    parts = [f"처방전에 적힌 약 이름: {line.name}"]
    if line.match_candidates:
        parts.append("이미 있는 후보: " + ", ".join(f"{c.name}({c.item_code})" for c in line.match_candidates))
    if line.check_reasons:
        parts.append("정하지 못한 이유: " + " / ".join(line.check_reasons))
    return "\n".join(parts)


def candidates_from(result: ReactResult, line: DrugLine) -> list[MatchCandidate]:
    """LLM의 답 {"candidates": [{"item_code", "reason"}]}에서 쓸 수 있는 후보만 (도구가 준 코드 · 같은 함량 · 새 약)."""
    seen = _seen_drugs(result)
    have = {c.item_code for c in line.match_candidates}
    key = match_key(line.name)
    out: list[MatchCandidate] = []
    for row in (result.final or {}).get("candidates") or []:
        if not isinstance(row, dict):
            continue
        code = str(row.get("item_code") or "").strip()
        name = seen.get(code)
        if name is None or code in have or strengths_conflict(key, product_key(name)):
            continue
        reason = " ".join(str(row.get("reason") or "").split())[:REASON_CHARS] or "도구로 찾음"
        out.append(MatchCandidate(item_code=code, name=name, score=round(similarity(key, product_key(name)), 3),
                                  reason=f"{AI_REASON}: {reason}"))
        have.add(code)
    return out


def suggest_candidates(order: DispensingOrder, tools: Sequence[Tool], chat: ToolChatFn, prompt: str, *,
                       max_lines: int = MAX_LINES) -> tuple[DispensingOrder, list[str]]:
    """약을 정하지 못한 줄에 AI 후보를 더한 새 조제지시서와 안내 (조사 기록 · 실패). 원래 조제지시서는 바꾸지 않는다.

    tools: search_drug · check_item_code (MCP 서버의 도구 또는 drug_tools). 결과 모양이 같아야 후보를 견줄 수 있다.
    """
    items, notes, asked = list(order.items), [], 0
    for i, line in enumerate(order.items):
        if line.item_code is not None or len(line.match_candidates) >= MAX_CANDIDATES or asked >= max_lines:
            continue
        asked += 1
        result = run_react(prompt, line_question(line), tools, chat)
        if result.final is None:
            notes.append(f"{NO_SUGGESTION} ({line.name}: {result.stopped}).")
            continue
        found = candidates_from(result, line)[: MAX_CANDIDATES - len(line.match_candidates)]
        notes.append(f"AI 후보 조사 ({line.name}): 후보 {len(found)}개 | {trace(result)}")
        if found:
            reasons = line.check_reasons if AI_CHECK in line.check_reasons else [*line.check_reasons, AI_CHECK]
            items[i] = line.model_copy(update={"match_candidates": [*line.match_candidates, *found],
                                               "needs_pharmacist_check": True, "check_reasons": reasons})
    if not asked:
        return order, []
    return DispensingOrder.model_validate({**order.model_dump(mode="json"),
                                          "items": [d.model_dump(mode="json") for d in items]}), notes


# ------------------------------------------------------------ 2) 위험 신호 → 근거 조사


def local_evidence(rag: RagIndex, embed: EmbedFn | None = None) -> EvidenceSearch:
    """MCP 서버를 쓰지 않을 때의 근거 검색 (서버의 search_evidence와 같은 함수)."""
    return lambda query, codes, k: evidence_search(rag, query, codes, k, embed)


def evidence_tool(search: EvidenceSearch, allowed: Sequence[str], ids: dict[str, str],
                  rows: dict[str, dict[str, Any]] | None = None) -> Tool:
    """LLM에 주는 근거 검색 도구. 위험 신호에 걸린 약(allowed)의 문서에서만 찾고, 찾은 조각에 E1, E2 … 번호를 준다(ids에 적는다).

    실제 검색은 search가 한다 (MCP 서버의 search_evidence 또는 local_evidence). 번호와 약 제한은 여기서 지킨다.
    rows를 주면 번호마다 찾은 줄 전체(출처 · 주소 · 글)를 적어 둔다 (evidence를 만들 때 쓴다).
    """
    allowed_set = list(dict.fromkeys(allowed))

    def run(args: dict[str, Any]) -> dict[str, Any]:
        query = " ".join(str(args["query"]).split())
        if not query:
            raise ToolFailure("query가 비었습니다")
        asked = [str(c) for c in args.get("item_codes") or [] if str(c) in allowed_set]
        found = []
        for row in search(query, asked or allowed_set, TOOL_RESULTS):
            if not row.get("focused", True) or row.get("item_code") not in allowed_set:
                continue
            key = f"{row.get('drug') or row.get('item_code')} · {row.get('section')}\x00{row.get('chunk_id')}"
            number = next((n for n, t in ids.items() if t == key), None)
            if number is None:
                number = f"E{len(ids) + 1}"
                ids[number] = key
            if rows is not None:
                rows[number] = row
            found.append({"id": number, "drug": row.get("drug"), "section": row.get("section"),
                          "text": str(row.get("text") or "")[:EVIDENCE_TEXT]})
        return {"query": query, "results": found}

    return Tool("search_evidence", "위험 신호에 걸린 약의 허가정보(효능 · 주의사항 · 상호작용 등)를 찾는다. 결과의 id(E1 …)를 근거로 적는다.",
                {"type": "object", "properties": {
                    "query": {"type": "string", "description": "찾을 내용 (예: '아세트아미노펜 중복 간 손상')"},
                    "item_codes": {"type": "array", "items": {"type": "string"}, "description": "찾을 약의 품목기준코드 (비우면 전부)"}},
                 "required": ["query"], "additionalProperties": False}, run)


def row_evidence(number: str, ids: dict[str, str], rows: dict[str, dict[str, Any]]) -> Evidence:
    """근거 번호 하나를 A.5 evidence로. 찾은 줄이 없으면(rows를 주지 않음) 번호의 제목만으로."""
    row = rows.get(number) or {}
    title = ids[number].split("\x00")[0]
    return evidence_item(title, str(row.get("source") or "허가정보"), row.get("url") or None, str(row.get("text") or title))


def explanations_from(result: ReactResult, findings: Sequence[RiskFinding], names: dict[str, str],
                      ids: dict[str, str], rows: dict[str, dict[str, Any]] | None = None) -> list[RiskFinding]:
    """LLM의 답 {"items": [{"index", "text", "evidence"}]}을 걸러 explanation에, 인용한 근거를 evidence에 담는다.

    인용한 근거는 설명이 걸러져도 담는다 (도구가 찾아 준 허가정보 조각이라 그 자체로 근거다).
    """
    rows = rows or {}
    texts: dict[int, tuple[str, list[str]]] = {}
    for row in (result.final or {}).get("items") or []:
        try:
            index, text = int(row["index"]), " ".join(str(row["text"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        cited = [str(e) for e in row.get("evidence") or [] if str(e) in ids]
        if 1 <= index <= len(findings) and index not in texts:
            texts[index] = (text, cited)
    out = []
    for n, f in enumerate(findings, start=1):
        text, cited = texts.get(n, (None, []))
        keep = text if text and acceptable(text, f, names) else None
        evidence = [row_evidence(e, ids, rows) for e in dict.fromkeys(cited)]
        out.append(f.model_copy(update={"explanation": keep, "evidence": evidence}))
    return out


def investigate_risks(findings: Sequence[RiskFinding], names: dict[str, str], search: EvidenceSearch | None,
                      chat: ToolChatFn, prompt: str) -> tuple[list[RiskFinding], str | None] | None:
    """(설명을 붙인 항목들, 조사 기록). 검색(search)이 없거나 항목이 없으면 None. 조사를 끝내지 못해도 None.

    search: (질문, 품목기준코드들, 개수) -> 근거 줄들. MCP 서버의 search_evidence 또는 local_evidence.
    """
    findings = list(findings)
    if not findings or search is None:
        return None
    ids: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    codes = [c for f in findings for c in f.item_codes]
    result = run_react(prompt, explain_input(findings, names), [evidence_tool(search, codes, ids, rows)], chat)
    if result.final is None:
        return None
    return explanations_from(result, findings, names, ids, rows), f"AI 위험 신호 조사: {trace(result)}"
