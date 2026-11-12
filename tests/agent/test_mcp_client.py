"""MCP 클라이언트 (agent/mcp_client.py) — 열고 닫기, 실패를 McpToolError로 알리기, mcp 없이도 불러올 수 있는지."""
import importlib.util
import os
import subprocess
import sys
import threading

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.mcp_client import ROOT, DrugTools, McpToolError, in_process, stdio_params

ROOT_DIR = RESOURCES_DIR.parents[1]


def test_root_is_the_repository_root():
    assert ROOT == ROOT_DIR and (ROOT / "agent" / "mcp_server.py").exists()


def test_importing_the_agent_does_not_need_the_mcp_package():
    """웹 · 테스트처럼 MCP를 쓰지 않는 실행은 mcp가 설치되어 있지 않아도 돌아야 한다."""
    code = "import sys, agent.mcp_client, agent.nodes, agent.api; print('mcp' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_calling_before_opening_is_an_error():
    with pytest.raises(McpToolError, match="열려 있지 않습니다"):
        DrugTools(object()).call("search_drug", name="타이레놀")


# ------------------------------------------------------------ 아래는 mcp가 있어야 한다
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="mcp 패키지가 없습니다")


@needs_mcp
def test_the_stdio_settings_start_the_server_module_with_this_python():
    params = stdio_params()
    assert params.command == sys.executable and params.args == ["-m", "agent.mcp_server"]
    assert params.cwd == str(ROOT_DIR) and params.env["PYTHONIOENCODING"] == "utf-8"
    assert stdio_params(no_api=True).args == ["-m", "agent.mcp_server", "--no-api"]


@needs_mcp
def test_a_server_that_cannot_start_is_reported_on_open():
    from mcp.client.stdio import StdioServerParameters

    broken = DrugTools(StdioServerParameters(command=sys.executable, args=["-c", "import sys; sys.exit(3)"]))
    with pytest.raises(McpToolError, match="MCP 서버에 연결하지 못했습니다"):
        broken.open()


@needs_mcp
def test_a_tool_error_keeps_only_the_reason():
    with in_process(DICT) as tools:
        with pytest.raises(McpToolError) as error:
            tools.call("verify_dictionary")
    assert not str(error.value).startswith("Error executing tool")


@needs_mcp
def test_an_unknown_tool_is_an_error():
    with in_process(DICT) as tools:
        with pytest.raises(McpToolError):
            tools.call("no_such_tool")


@needs_mcp
def test_closing_stops_the_thread_and_later_calls_fail():
    before = threading.active_count()
    tools = in_process(DICT)
    assert threading.active_count() == before + 1
    tools.close()
    assert threading.active_count() == before
    with pytest.raises(McpToolError):
        tools.call("search_drug", name="타이레놀")
    tools.close()                                       # 두 번 닫아도 괜찮다


@needs_mcp
def test_one_connection_serves_many_calls_from_other_threads():
    with in_process(DICT) as tools:
        found = []

        def ask():
            found.append(tools.call("search_drug", name="게보린")["dictionary"][0]["item_code"])

        workers = [threading.Thread(target=ask) for _ in range(5)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    assert found == ["197900277"] * 5


@needs_mcp
def test_the_server_does_not_need_the_working_directory_of_the_caller(tmp_path):
    """웹이 다른 폴더에서 돌아도 서버는 저장소 루트에서 뜬다 (stdio_params의 cwd)."""
    code = ("import sys; sys.path.insert(0, r'%s'); from agent.mcp_client import stdio\n"
            "t = stdio(no_api=True); print(len(t.tool_names())); t.close()") % ROOT_DIR
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "8"
