"""AI 조사 (판단, 11/16) — ReAct 루프(react.py)로 위험 신호의 근거를 찾는다. 규칙의 결과를 바꾸지 않고 덧붙이기만 한다.

도구는 MCP 서버의 search_evidence를 쓴다 (nodes가 넘긴다). MCP 서버를 쓰지 않으면 같은 판단 함수(rag.evidence_search)를 직접 부른다.
(매칭 실패 줄의 AI 후보 조사는 11/18에 뺐다: 1차 대상 8종 밖의 약은 승인할 수 없어 후보를 고를 일이 없다. REQ-AG-03)

위험 신호 → 근거 조사
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

from agent.explain import acceptable, evidence_item, explain_input, finding_terms
from agent.rag import EVIDENCE_TEXT, RagIndex, evidence_search
from agent.react import ReactResult, Tool, ToolChatFn, ToolFailure, run_react, trace
from agent.schema import Evidence, RiskFinding

EmbedFn = Callable[[list[str]], list[list[float]]]
EvidenceSearch = Callable[[str, Sequence[str], int], list[dict[str, Any]]]   # (질문, 품목기준코드들, 개수) -> 근거 줄들

TOOL_RESULTS = 5          # 근거 검색 한 번에 받는 조각 수


# ------------------------------------------------------------ 위험 신호 → 근거 조사


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


def row_evidence(number: str, ids: dict[str, str], rows: dict[str, dict[str, Any]], terms: Sequence[tuple[str, int]] = ()) -> Evidence:
    """근거 번호 하나를 A.5 evidence로. 찾은 줄이 없으면(rows를 주지 않음) 번호의 제목만으로."""
    row = rows.get(number) or {}
    title = ids[number].split("\x00")[0]
    return evidence_item(title, str(row.get("source") or "허가정보"), row.get("url") or None, str(row.get("text") or title), terms)


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
        evidence = [row_evidence(e, ids, rows, finding_terms(f)) for e in dict.fromkeys(cited)]
        cited_f = f.model_copy(update={"evidence": evidence})       # 인용한 근거를 붙인 뒤 거른다 (uses_evidence가 근거를 보게)
        keep = text if text and acceptable(text, cited_f, names) else None
        out.append(cited_f.model_copy(update={"explanation": keep}))
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
