"""약품 도구 MCP 클라이언트 (12단계). 노드는 이 파일로 mcp_server의 도구를 부른다.

그래프의 노드는 보통 함수(동기)라서, 비동기인 MCP 연결을 백그라운드 스레드 하나에 열어 두고
call()이 그 스레드에 요청을 넘긴 뒤 결과를 기다린다. 연결은 한 번 열면 닫을 때까지 쓴다
(stdio 서버를 부를 때마다 띄우면 Windows에서 한 번에 1초 가까이 걸린다).

    with stdio() as tools:                          # python -m agent.mcp_server 를 하위 프로세스로 띄운다
        tools.tool_names()                          # ['search_drug', 'check_item_code', 'match_drug', 'verify_dictionary']
        matched, api_used = tools.match_drug(order)

    tools = in_process(dictionary, api=None)        # 프로세스 없이 서버 객체에 바로 붙는다 (테스트 · 측정용)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.drug_dictionary import DrugDictionary
from agent.drug_match import ApiSearch
from agent.schema import DispensingOrder

if TYPE_CHECKING:
    from mcp.client.stdio import StdioServerParameters

# mcp 패키지는 연결할 때만 불러온다. 그래서 MCP를 쓰지 않는 실행(웹 · 테스트)은 mcp가 없어도 이 파일을 불러올 수 있다.
ROOT = Path(__file__).resolve().parents[1]
CONNECT_TIMEOUT_SECONDS = 30.0     # 서버를 띄우고 연결하기까지 (첫 실행은 import가 길다)
CALL_TIMEOUT_SECONDS = 60.0        # 도구 하나 (의약품 API를 여러 번 부르는 verify_dictionary가 가장 길다)


class McpToolError(Exception):
    """도구가 실패를 돌려주었거나(서버의 ToolError) 서버와 연결하지 못했다."""


class DrugTools:
    """MCP 서버 하나에 연결을 열어 두고 도구를 부르는 동기 클라이언트."""

    def __init__(self, target: Any):
        self._target = target                       # MCPServer 객체(같은 프로세스) 또는 StdioServerParameters
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._thread: threading.Thread | None = None
        self._ready = concurrent.futures.Future()   # 연결되면 도구 목록, 실패하면 예외
        self._tools: list[Any] = []
        self._broken: str | None = None             # 연결이 끊겼으면 그 이유

    # ------------------------------------------------------------ 열고 닫기
    def open(self) -> "DrugTools":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="mcp-client", daemon=True)
        self._thread.start()
        try:
            self._tools = self._ready.result(timeout=CONNECT_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            self.close()
            raise McpToolError(f"MCP 서버에 {CONNECT_TIMEOUT_SECONDS:.0f}초 안에 연결하지 못했습니다.") from None
        except McpToolError:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._loop is not None and self._queue is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = self._loop = self._queue = None

    def __enter__(self) -> "DrugTools":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        """연결을 열고, 닫으라는 신호(None)가 올 때까지 요청을 하나씩 처리한다. 연결은 이 작업 하나가 연다."""
        self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
        try:
            from mcp import Client

            async with Client(self._target) as client:
                listed = await client.list_tools()
                self._ready.set_result(list(listed.tools))
                while (request := await self._queue.get()) is not None:
                    name, arguments, answer = request
                    try:
                        answer.set_result(await client.call_tool(name, arguments))
                    except Exception as e:            # 연결 문제 등. 부른 쪽 스레드에서 다시 올린다
                        answer.set_exception(e)
        except Exception as e:
            if not self._ready.done():
                self._ready.set_exception(McpToolError(f"MCP 서버에 연결하지 못했습니다 ({type(e).__name__}: {e})."))
            else:
                self._broken = f"MCP 서버와 연결이 끊겼습니다 ({type(e).__name__}: {e})."
        finally:
            self._broken = self._broken or "MCP 연결이 닫혔습니다."
            while not self._queue.empty():                     # 끊긴 뒤 남은 요청은 바로 실패로 돌려준다
                request = self._queue.get_nowait()
                if request is not None:
                    request[2].set_exception(McpToolError(self._broken))

    # ------------------------------------------------------------ 도구 부르기
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    def tools(self) -> list[tuple[str, str]]:
        """(이름, 설명) 목록."""
        return [(t.name, t.description or "") for t in self._tools]

    def call(self, tool: str, /, **arguments: Any) -> dict[str, Any]:
        """도구 하나를 부르고 (도구 인자는 키워드로: call("search_drug", name="타이레놀")) 결과(JSON 객체)를 돌려준다. 도구가 실패하면 McpToolError."""
        if self._loop is None or self._queue is None:
            raise McpToolError("MCP 연결이 열려 있지 않습니다 (open()을 먼저 부르세요).")
        if self._broken is not None:
            raise McpToolError(self._broken)
        answer: concurrent.futures.Future = concurrent.futures.Future()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (tool, arguments, answer))
        except RuntimeError:                                   # 연결 스레드의 이벤트 루프가 이미 끝났다
            raise McpToolError(self._broken or "MCP 연결이 닫혔습니다.") from None
        try:
            result = answer.result(timeout=CALL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise McpToolError(f"도구 {tool}이(가) {CALL_TIMEOUT_SECONDS:.0f}초 안에 답하지 않았습니다.") from None
        except Exception as e:
            raise McpToolError(f"도구 {tool}을(를) 부르지 못했습니다 ({type(e).__name__}: {e}).") from None
        text = " ".join(getattr(c, "text", "") for c in result.content).strip()
        if result.is_error:
            raise McpToolError(text.removeprefix(f"Error executing tool {tool}: ") or f"도구 {tool}이(가) 실패했습니다.")
        if result.structured_content is not None:
            return dict(result.structured_content)
        return json.loads(text)

    def match_drug(self, order: DispensingOrder) -> tuple[DispensingOrder, bool]:
        """drug_match.match_order와 같은 일을 서버가 한다. (매칭한 조제지시서, 서버가 의약품 API를 썼는지)."""
        found = self.call("match_drug", order=order.model_dump(mode="json"))
        return DispensingOrder.model_validate(found["order"]), bool(found["api_used"])


def stdio_params(*, no_api: bool = False) -> "StdioServerParameters":
    """python -m agent.mcp_server 를 저장소 루트에서 띄우는 설정. 지금 쓰는 파이썬(.venv)으로 띄운다."""
    from mcp.client.stdio import StdioServerParameters

    args = ["-m", "agent.mcp_server"] + (["--no-api"] if no_api else [])
    return StdioServerParameters(command=sys.executable, args=args, cwd=str(ROOT),
                                 env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})


def stdio(*, no_api: bool = False) -> DrugTools:
    """서버를 하위 프로세스로 띄워 연결한 클라이언트 (열린 상태). 다 쓰면 close()."""
    return DrugTools(stdio_params(no_api=no_api)).open()


def in_process(dictionary: DrugDictionary, api: ApiSearch | None = None) -> DrugTools:
    """같은 프로세스의 서버 객체에 바로 붙은 클라이언트 (열린 상태). JSON으로 주고받는 것은 stdio와 같다."""
    from agent.mcp_server import build_server   # 같은 층. 쓸 때만 서버를 만든다

    return DrugTools(build_server(dictionary, api)).open()
