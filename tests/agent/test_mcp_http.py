"""MCP 서버를 HTTP로 따로 띄우고 주소로 붙기 (agent/mcp_server.py --http, mcp_client.http · from_settings).

    python -m agent.mcp_server --http                     # http://127.0.0.1:8765/mcp
    python scripts/agent/try_mcp.py --url http://127.0.0.1:8765/mcp
    .env: AGENT_USE_MCP=1, AGENT_MCP_URL=http://127.0.0.1:8765/mcp  → AgentApi.from_settings()가 그 서버에 붙는다

진짜 HTTP 서버를 띄워 보는 시험은 mcp와 uvicorn이 있을 때만 돈다 (빈 포트를 골라 하위 프로세스로 띄운다).
"""
import importlib.util
import os
import socket
import subprocess
import sys
import time

import pytest

from agent import config, mcp_client
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.mcp_client import ROOT, McpToolError, http

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
HAS_HTTP = importlib.util.find_spec("mcp") is not None and importlib.util.find_spec("uvicorn") is not None


@pytest.fixture(autouse=True)
def no_mcp_url(monkeypatch):
    monkeypatch.delenv("AGENT_MCP_URL", raising=False)


# ------------------------------------------------------------ 설정


def test_the_mcp_url_comes_from_the_environment(monkeypatch):
    assert config.mcp_url() is None
    monkeypatch.setenv("AGENT_MCP_URL", "  http://127.0.0.1:8765/mcp ")
    assert config.mcp_url() == "http://127.0.0.1:8765/mcp"
    monkeypatch.setenv("AGENT_MCP_URL", "")
    assert config.mcp_url() is None


def test_from_settings_uses_the_url_when_it_is_set_and_stdio_otherwise(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_client, "http", lambda url: calls.append(("http", url)) or "H")
    monkeypatch.setattr(mcp_client, "stdio", lambda *, no_api: calls.append(("stdio", no_api)) or "S")
    assert mcp_client.from_settings(no_api=True) == "S"
    monkeypatch.setenv("AGENT_MCP_URL", "http://127.0.0.1:9000/mcp")
    assert mcp_client.from_settings(no_api=True) == "H"
    assert calls == [("stdio", True), ("http", "http://127.0.0.1:9000/mcp")]


@pytest.mark.parametrize("url", ["127.0.0.1:8765/mcp", "ftp://x/mcp", ""])
def test_an_address_that_is_not_http_is_refused_before_connecting(url):
    with pytest.raises(McpToolError, match="http:// 또는 https://"):
        http(url)


# ------------------------------------------------------------ 서버의 실행 옵션


def test_the_server_runs_over_http_with_the_host_and_port_it_is_given(monkeypatch):
    from agent import mcp_server

    runs = []

    class Server:
        def run(self, transport, **kwargs):
            runs.append((transport, kwargs))

    monkeypatch.setattr(mcp_server, "build_server", lambda *args: Server())
    assert mcp_server.main(["--no-api", "--http"]) == 0
    assert mcp_server.main(["--no-api", "--http", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert mcp_server.main(["--no-api"]) == 0
    assert runs == [
        ("streamable-http", {"host": "127.0.0.1", "port": 8765, "streamable_http_path": "/mcp"}),
        ("streamable-http", {"host": "0.0.0.0", "port": 9000, "streamable_http_path": "/mcp"}),
        ("stdio", {}),
    ]


def test_the_address_to_connect_to():
    from agent.mcp_server import http_url

    assert http_url() == "http://127.0.0.1:8765/mcp"
    assert http_url("0.0.0.0", 9000) == "http://127.0.0.1:9000/mcp"      # 모든 주소로 받는 서버도 같은 PC에서는 127.0.0.1로 붙는다
    assert http_url("10.0.0.5", 9000) == "http://10.0.0.5:9000/mcp"


# ------------------------------------------------------------ 진짜 HTTP 서버


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(port: int, process, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"서버가 바로 끝났습니다: {process.stderr.read()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError("서버가 제시간에 뜨지 않았습니다")


@pytest.fixture
def http_server():
    port = free_port()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    env.pop("DRUG_API_KEY", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "agent.mcp_server", "--http", "--no-api", "--port", str(port)],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    try:
        wait_for(port, process)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.skipif(not HAS_HTTP, reason="mcp · uvicorn이 없으면 HTTP 서버를 띄울 수 없다")
def test_a_separately_started_server_is_reached_by_its_address(http_server):
    from agent.mcp_server import TOOL_NAMES

    with http(http_server) as tools:
        assert tuple(tools.tool_names()) == TOOL_NAMES
        found = tools.call("search_drug", name="타이레놀")
        assert found["dictionary"][0]["item_code"] == "202106092" and found["api_used"] is False
        report = tools.call("check_duplicate", items=[{"item_code": "202106092", "name": "타이레놀정500mg"},
                                                      {"item_code": "197900277", "name": "게보린정"}])
        assert [h["type"] for h in report["hits"]] == ["ingredient_dup"]


@pytest.mark.skipif(not HAS_HTTP, reason="mcp · uvicorn이 없으면 HTTP 서버를 띄울 수 없다")
def test_a_server_that_is_not_running_is_reported():
    with pytest.raises(McpToolError):
        http(f"http://127.0.0.1:{free_port()}/mcp")
