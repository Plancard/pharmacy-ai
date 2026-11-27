"""약품 도구 MCP 서버 (12단계, 일정표 11/16). 판단 층의 함수를 MCP 도구로 내놓기만 하고 새 로직은 없다.

    python -m agent.mcp_server            # 표준 입출력(stdio)으로 띄운다. .env에 DRUG_API_KEY가 있으면 의약품 API도 쓴다
    python -m agent.mcp_server --no-api   # 약품 사전만 쓴다
    python -m agent.mcp_server --http     # HTTP로 띄운다: http://127.0.0.1:8765/mcp (따로 띄운 서버에 클라이언트가 주소로 붙는다)
    python -m agent.mcp_server --http --host 0.0.0.0 --port 9000   # 다른 컨테이너 · PC에서 붙을 때

도구 7개 (명세서 3.3 '의약품 정보 · DUR API (MCP 경유)', 기능분석서 F-AG-08)
    의약품 정보 (REQ-AG-03)
    match_drug         조제지시서(A.5)의 약품마다 품목기준코드를 정한다 (그래프의 매칭 노드)  <- drug_match.match_order
    search_drug        이름으로 약 찾기: 약품 사전에서 비슷한 약 + (API가 있으면) 의약품 API의 제품  <- drug_match.search_drug_result
    check_item_code    품목기준코드가 이 이름의 제품 코드가 맞는지 의약품 API로 확인  <- drug_match.check_code_result
                       (search_drug · check_item_code는 1차 그래프가 부르지 않는다. 약 조회용이고 2차 실제 처방전 매칭의 기반이다)
    DUR (REQ-AG-05)
    check_duplicate    처방약끼리 성분 중복(사전) + 효능군 중복(DUR API) (그래프의 위험 노드)  <- dur.check_duplicate
    check_interaction  처방약끼리 병용금기 (DUR API) (그래프의 위험 노드)                <- dur.check_interaction
    조제 계획 · 근거 (REQ-AG-04 · 06)
    build_plan         승인 전 조제지시서(A.5)로 칸별 조제 계획(A.4). 3일 초과 · 약을 정하지 못한 줄 · 대상 8종 밖 약이면
                       계획 없이 이유 (그래프의 계획 노드)                                 <- plan.build_plan
    search_evidence    약 허가정보(e약은요 · 허가정보 상세)에서 근거 조각 찾기 (위험 신호 AI 근거 조사)  <- rag.evidence_search
    (11/18 삭제: verify_dictionary는 scripts/agent/verify_dictionary.py가 같은 일을 한다. get_plan은 받은 계획을 다시 보여 줄 뿐이라 뺐다)

리소스 1개
    pharmacy://dictionary   대상 약 8종 (품목기준코드 · 이름 · 주성분 · CV 클래스). LLM · 클라이언트가 읽는 고정 자료

규칙
- 서버는 표준 출력(stdout)에 아무것도 쓰지 않는다. stdout이 MCP 통신선이다(stdio일 때). 알림은 표준 오류(stderr)로만 간다.
- HTTP(streamable-http)로 띄우면 주소는 http://<host>:<port>/mcp 이다. 클라이언트는 mcp_client.http(주소) 또는 .env의 AGENT_MCP_URL로 붙는다.
  기본 host는 127.0.0.1(내 PC에서만)이다. 인증이 없으므로 0.0.0.0은 믿을 수 있는 내부망(docker compose 등)에서만 쓴다.
- 실패는 ToolError로 알린다. 부른 쪽(mcp_client)은 is_error 결과를 McpToolError로 바꾼다.
- 이 파일은 따로 도는 서버 프로세스의 입구라서 의약품 API 연결(drug_api.default_api)을 스스로 만든다.
  판단 함수에는 그 연결을 인자로 넘긴다 (판단 층은 여전히 가장자리를 직접 부르지 않는다).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from agent import config
from agent.drug_api import default_api
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import ApiSearch, check_code_result, match_order, search_drug_result
from agent.dur import DurLookup, Ref, check_duplicate, check_interaction, report_to_json
from agent.dur_api import default_dur_api
from agent.errors import ConfigError, DrugApiError
from agent.plan import NOT_TARGET_PLAN, PlanError, build_plan, not_target_lines, plan_lines
from agent.rag import RagIndex, evidence_search, load_index
from agent.schema import DispensingOrder

SERVER_NAME = "pharmacy-agent"
TOOL_NAMES = ("search_drug", "check_item_code", "match_drug", "check_interaction", "check_duplicate",
              "build_plan", "search_evidence")
DICTIONARY_URI = "pharmacy://dictionary"
MAX_EVIDENCE = 10        # search_evidence가 한 번에 돌려주는 조각 수의 최대
MAX_RESULTS = 5          # 사전 · API에서 보여 줄 약의 수
NO_API = "의약품 API를 쓰지 않는 서버입니다 (.env에 DRUG_API_KEY가 없거나 --no-api로 띄웠습니다)."
NO_DUR = "DUR API를 쓰지 않는 서버입니다 (.env에 DRUG_API_KEY가 없거나 --no-api로 띄웠습니다)."
NO_RAG = "근거 색인이 없는 서버입니다 (scripts/agent/build_rag_index.py로 agent/resources/rag_corpus를 만드세요)."


def _refs(items: list[dict[str, Any]]) -> list[Ref]:
    try:
        return [Ref(item_code=i.get("item_code") or None, name=str(i["name"])) for i in items]
    except (KeyError, TypeError, AttributeError):
        raise ToolError("items는 [{item_code, name}, ...] 모양이어야 합니다.") from None


def build_server(dictionary: DrugDictionary, api: ApiSearch | None = None, dur: DurLookup | None = None,
                 rag: RagIndex | None = None, embed: Any = None) -> MCPServer:
    """도구 7개 · 리소스 1개가 달린 서버를 만든다. 테스트는 이 서버를 프로세스 없이 바로 붙여 부른다 (mcp_client.in_process).

    rag(근거 색인)가 없으면 search_evidence는 NO_RAG로 실패한다. embed(질문 임베딩)가 없으면 BM25만으로 찾는다.
    """
    server = MCPServer(
        SERVER_NAME,
        instructions="약국 조제 보조 Agent의 약품 도구. 약품 사전(대상 8종)과 식약처 의약품 제품 허가정보 API를 쓴다.",
    )

    @server.tool(description="약 이름으로 찾는다. 약품 사전에서 이름이 비슷한 약과, 의약품 API가 있으면 API의 제품을 함께 돌려준다.")
    def search_drug(name: str) -> dict[str, Any]:
        try:
            return search_drug_result(name, dictionary, api, MAX_RESULTS)
        except (DrugApiError, ConfigError) as e:
            raise ToolError(str(e)) from None

    @server.tool(name="check_item_code",      # 파이썬 이름은 판단 함수 check_item_code와 겹치지 않게 따로 짓는다
                 description="품목기준코드(9자리)가 이 이름의 제품 코드가 맞는지 의약품 API로 확인한다. 판정: 일치 · 불일치 · 없음.")
    def check_code(item_code: str, name: str) -> dict[str, Any]:
        if api is None:
            raise ToolError(NO_API)
        try:
            return check_code_result(item_code, name, dictionary, api, MAX_RESULTS)
        except (DrugApiError, ConfigError) as e:
            raise ToolError(str(e)) from None

    @server.tool(description="조제지시서(A.5 JSON)의 약품마다 품목기준코드를 정하고 약사 확인 표시를 붙인 조제지시서를 돌려준다.")
    def match_drug(order: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = DispensingOrder.model_validate(order)
        except ValidationError as e:
            raise ToolError(f"조제지시서(A.5) 형식이 아닙니다: {e.error_count()}곳이 틀립니다.") from None
        matched = match_order(parsed, dictionary, api=api)
        return {"order": matched.model_dump(mode="json"), "api_used": api is not None}

    @server.tool(name="check_interaction",
                 description="처방약끼리 병용금기를 DUR API로 찾는다. items: [{item_code, name}]. 결과는 hits · unchecked · dur_used.")
    def check_interaction_tool(items: list[dict[str, Any]]) -> dict[str, Any]:
        if dur is None:
            raise ToolError(NO_DUR)
        return report_to_json(check_interaction(_refs(items), dur, dictionary))

    @server.tool(name="check_duplicate",
                 description="처방약끼리 성분 중복(약품 사전)과 효능군 중복(DUR API, 있을 때)을 찾는다. items: [{item_code, name}].")
    def check_duplicate_tool(items: list[dict[str, Any]]) -> dict[str, Any]:
        return report_to_json(check_duplicate(_refs(items), dictionary, dur))

    @server.tool(name="build_plan",
                 description="조제지시서(A.5)로 트레이 칸별 조제 계획(A.4)을 만든다 (승인 카드에 보이고 승인되면 확정). 만들 수 없으면(3일 초과 · "
                             "약을 정하지 못한 줄 · 대상 8종이 아닌 약) plan은 null이고 error에 코드와 이유가 있다.")
    def build_plan_tool(order: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = DispensingOrder.model_validate(order)
        except ValidationError as e:
            raise ToolError(f"조제지시서(A.5) 형식이 아닙니다: {e.error_count()}곳이 틀립니다.") from None
        not_target = not_target_lines(parsed, dictionary)
        if not_target:                    # CV가 모르는 약 (REQ-AG-03). 그래프의 계획 노드와 같은 결과
            return {"plan": None, "error": {"code": None, "message": f"{NOT_TARGET_PLAN}: {', '.join(not_target)}"}, "lines": []}
        try:
            plan = build_plan(parsed)
        except PlanError as e:            # 계획을 못 만드는 것은 도구 실패가 아니라 결과다 (약사에게 돌아갈 이유)
            return {"plan": None, "error": {"code": e.code, "message": str(e)}, "lines": []}
        names = {i.item_code: i.name for i in parsed.items if i.item_code}
        return {"plan": plan.model_dump(mode="json"), "error": None, "lines": plan_lines(plan, names)}

    @server.tool(name="search_evidence",
                 description="약 허가정보(식약처 e약은요 · 허가정보 상세)에서 근거 조각을 찾는다. query: 찾을 내용, item_codes: 찾을 약의 "
                             "품목기준코드(비우면 전부), k: 조각 수(1~10). focused=false인 조각은 물은 약이 아닌 다른 약의 문서다.")
    def search_evidence_tool(query: str, item_codes: list[str] | None = None, k: int = 5) -> dict[str, Any]:
        if rag is None or not len(rag):
            raise ToolError(NO_RAG)
        query = " ".join(str(query).split())
        if not query:
            raise ToolError("query가 비었습니다.")
        codes = [str(c) for c in (item_codes or [])]
        rows = evidence_search(rag, query, codes, max(1, min(int(k), MAX_EVIDENCE)), embed)
        return {"query": query, "item_codes": codes, "results": rows}

    @server.resource(DICTIONARY_URI, name="dictionary", mime_type="application/json",
                     description="대상 약 8종: 품목기준코드 · 이름 · 주성분 · CV 클래스 ID")
    def dictionary_resource() -> str:
        return json.dumps([{"item_code": e.item_code, "name": e.name, "ingredients": list(e.ingredients),
                            "class_id": e.class_id} for e in dictionary.entries], ensure_ascii=False)

    return server


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HTTP_PATH = "/mcp"


def _vector_search():
    from agent.vector_store import default_store   # 가장자리. 서버 프로세스의 입구라 직접 만든다

    store = default_store()
    return store.searcher(config.EMBED_MODEL) if store is not None else None


def _embed_or_none():
    """OPENAI_API_KEY가 있으면 질문 임베딩 함수, 없으면 None (BM25만)."""
    try:
        config.require_openai_key()
    except ConfigError:
        return None
    from agent.embed import embed_texts

    return embed_texts


def http_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """HTTP로 띄운 서버의 주소. 0.0.0.0으로 띄웠으면 같은 PC에서는 127.0.0.1로 붙는다."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    return f"http://{shown}:{port}{HTTP_PATH}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="약품 도구 MCP 서버 (기본 stdio, --http면 HTTP)")
    parser.add_argument("--no-api", action="store_true", help="의약품 API를 쓰지 않는다 (약품 사전만)")
    parser.add_argument("--http", action="store_true", help=f"HTTP(streamable-http)로 띄운다. 주소 http://HOST:PORT{HTTP_PATH}")
    parser.add_argument("--host", default=DEFAULT_HOST, help="--http일 때 받을 주소 (기본 %(default)s: 내 PC에서만)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="--http일 때 포트 (기본 %(default)s)")
    args = parser.parse_args(argv)
    try:
        dictionary = load_dictionary()
    except ConfigError as e:
        print(f"약품 사전을 읽지 못했습니다: {e}", file=sys.stderr)   # stdout은 통신선이라 stderr로
        return 1
    rag = load_index(config.RAG_DIR, _vector_search())           # 근거 색인은 파일이라 --no-api여도 읽는다
    embed = None if args.no_api else _embed_or_none()
    if args.no_api:
        server = build_server(dictionary, None, None, rag, None)
    else:
        server = build_server(dictionary, default_api(), default_dur_api(), rag, embed)
    if args.http:
        print(f"MCP 서버(HTTP): {http_url(args.host, args.port)} | 의약품 API: {'사용 안 함' if args.no_api else '설정대로'}"
              f" | 근거 색인: {f'조각 {len(rag)}개' if rag else '없음'} | 끝내려면 Ctrl+C", file=sys.stderr)
        server.run("streamable-http", host=args.host, port=args.port, streamable_http_path=HTTP_PATH)
    else:
        server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
