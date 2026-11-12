"""약품 도구 MCP 서버 (12단계, 일정표 11/16). 판단 층의 함수를 MCP 도구로 내놓기만 하고 새 로직은 없다.

    python -m agent.mcp_server            # 표준 입출력(stdio)으로 띄운다. .env에 DRUG_API_KEY가 있으면 의약품 API도 쓴다
    python -m agent.mcp_server --no-api   # 약품 사전만 쓴다
    python -m agent.mcp_server --http     # HTTP로 띄운다: http://127.0.0.1:8765/mcp (따로 띄운 서버에 클라이언트가 주소로 붙는다)
    python -m agent.mcp_server --http --host 0.0.0.0 --port 9000   # 다른 컨테이너 · PC에서 붙을 때

도구 (지금 6개. 조제 2개는 14단계에서 더한다)
    search_drug        이름으로 약 찾기: 약품 사전에서 비슷한 약 + (API가 있으면) 의약품 API의 제품
                       <- drug_match.rank_by_name, ApiSearch
    check_item_code    품목기준코드가 이 이름의 제품 코드가 맞는지 의약품 API로 확인  <- drug_match.check_item_code
    match_drug         조제지시서(A.5)의 약품마다 품목기준코드를 정한다                 <- drug_match.match_order
    verify_dictionary  약품 사전 8종의 코드를 의약품 API로 한 번에 확인                 <- drug_match.check_item_code
    check_interaction  처방약끼리 병용금기 (DUR API)                                   <- dur.check_interaction
    check_duplicate    처방약끼리 성분 중복(사전) + 효능군 중복(DUR API)               <- dur.check_duplicate

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
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from agent.drug_api import default_api
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import CANDIDATE_MIN, ApiSearch, check_item_code, match_order, rank_by_name
from agent.drug_names import match_key, search_term
from agent.dur import DurLookup, Ref, check_duplicate, check_interaction, report_to_json
from agent.dur_api import default_dur_api
from agent.errors import ConfigError, DrugApiError
from agent.schema import DispensingOrder

SERVER_NAME = "pharmacy-agent"
TOOL_NAMES = ("search_drug", "check_item_code", "match_drug", "verify_dictionary", "check_interaction", "check_duplicate")
MAX_RESULTS = 5          # 사전 · API에서 보여 줄 약의 수
NO_API = "의약품 API를 쓰지 않는 서버입니다 (.env에 DRUG_API_KEY가 없거나 --no-api로 띄웠습니다)."
NO_DUR = "DUR API를 쓰지 않는 서버입니다 (.env에 DRUG_API_KEY가 없거나 --no-api로 띄웠습니다)."


def _product(product: Any, score: float | None = None) -> dict[str, Any]:
    found = {"item_code": product.item_code, "name": product.name, "company": getattr(product, "company", None)}
    if score is not None:
        found["score"] = round(score, 3)
    return found


def _search_api(api: ApiSearch, name: str) -> list[Any]:
    try:
        return list(api(search_term(name)))
    except (DrugApiError, ConfigError) as e:
        raise ToolError(str(e)) from None


def _refs(items: list[dict[str, Any]]) -> list[Ref]:
    try:
        return [Ref(item_code=i.get("item_code") or None, name=str(i["name"])) for i in items]
    except (KeyError, TypeError, AttributeError):
        raise ToolError("items는 [{item_code, name}, ...] 모양이어야 합니다.") from None


def build_server(dictionary: DrugDictionary, api: ApiSearch | None = None, dur: DurLookup | None = None) -> MCPServer:
    """도구 6개가 달린 서버를 만든다. 테스트는 이 서버를 프로세스 없이 바로 붙여 부른다 (mcp_client.in_process)."""
    server = MCPServer(
        SERVER_NAME,
        instructions="약국 조제 보조 Agent의 약품 도구. 약품 사전(대상 8종)과 식약처 의약품 제품 허가정보 API를 쓴다.",
    )

    @server.tool(description="약 이름으로 찾는다. 약품 사전에서 이름이 비슷한 약과, 의약품 API가 있으면 API의 제품을 함께 돌려준다.")
    def search_drug(name: str) -> dict[str, Any]:
        ranked = [(score, e) for score, e in rank_by_name(match_key(name), dictionary) if score >= CANDIDATE_MIN]
        found: dict[str, Any] = {
            "query": name,
            "dictionary": [
                {"item_code": e.item_code, "name": e.name, "score": round(score, 3)} for score, e in ranked[:MAX_RESULTS]
            ],
            "api_used": api is not None,
            "api": None,
        }
        if api is not None:
            found["api"] = [_product(p) for p in _search_api(api, name)[:MAX_RESULTS]]
        return found

    @server.tool(name="check_item_code",      # 파이썬 이름은 판단 함수 check_item_code와 겹치지 않게 따로 짓는다
                 description="품목기준코드(9자리)가 이 이름의 제품 코드가 맞는지 의약품 API로 확인한다. 판정: 일치 · 불일치 · 없음.")
    def check_code(item_code: str, name: str) -> dict[str, Any]:
        if api is None:
            raise ToolError(NO_API)
        verdict, similar = check_item_code(item_code, name, _search_api(api, name))
        entry = dictionary.by_item_code(item_code)
        return {
            "item_code": item_code,
            "name": name,
            "verdict": verdict,
            "in_dictionary": entry is not None,
            "dictionary_name": entry.name if entry else None,
            "similar": [_product(p, score) for score, p in similar[:MAX_RESULTS]],
        }

    @server.tool(description="조제지시서(A.5 JSON)의 약품마다 품목기준코드를 정하고 약사 확인 표시를 붙인 조제지시서를 돌려준다.")
    def match_drug(order: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = DispensingOrder.model_validate(order)
        except ValidationError as e:
            raise ToolError(f"조제지시서(A.5) 형식이 아닙니다: {e.error_count()}곳이 틀립니다.") from None
        matched = match_order(parsed, dictionary, api=api)
        return {"order": matched.model_dump(mode="json"), "api_used": api is not None}

    @server.tool(description="약품 사전 8종의 품목기준코드를 의약품 API로 한 번에 확인한다 (약마다 일치 · 불일치 · 없음 · 실패).")
    def verify_dictionary() -> dict[str, Any]:
        if api is None:
            raise ToolError(NO_API)
        results = []
        for entry in dictionary.entries:
            try:
                products = list(api(search_term(entry.name)))
            except (DrugApiError, ConfigError) as e:
                results.append({"item_code": entry.item_code, "name": entry.name, "verdict": "실패", "error": str(e)})
                continue
            verdict, similar = check_item_code(entry.item_code, entry.name, products)
            results.append({
                "item_code": entry.item_code,
                "name": entry.name,
                "verdict": verdict,
                "similar": [_product(p, score) for score, p in similar[:3]],
            })
        matched = sum(r["verdict"] == "일치" for r in results)
        return {"total": len(results), "matched": matched, "results": results}

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

    return server


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HTTP_PATH = "/mcp"


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
    server = build_server(dictionary) if args.no_api else build_server(dictionary, default_api(), default_dur_api())
    if args.http:
        print(f"MCP 서버(HTTP): {http_url(args.host, args.port)} | 의약품 API: {'사용 안 함' if args.no_api else '설정대로'}"
              " | 끝내려면 Ctrl+C", file=sys.stderr)
        server.run("streamable-http", host=args.host, port=args.port, streamable_http_path=HTTP_PATH)
    else:
        server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
