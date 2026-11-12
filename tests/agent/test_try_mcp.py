"""MCP 시험 스크립트 (scripts/agent/try_mcp.py) — 도구 4개가 보이고, 그래프 결과가 함수 호출 때와 같은지 (12단계 확인 기준).

'fake'(시험용 가짜 LangGraph)와 'real'(진짜 LangGraph, 설치되어 있을 때만)로 돈다. 서버는 같은 프로세스에 붙는다
(--in-process). 진짜 하위 프로세스(stdio)로 띄우는 시험은 test_mcp_server.py · test_mcp_client.py에 있다.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import fake_langgraph  # noqa: E402
from agent import api as agent_api  # noqa: E402
from agent.api import AgentApi  # noqa: E402
from agent.checkpoint import memory_checkpointer  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.mcp_client import DrugTools, in_process  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("try_mcp", ROOT / "scripts" / "agent" / "try_mcp.py")
try_mcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(try_mcp)
DICT = load_dictionary()


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


def run(argv):
    lines = []
    code = try_mcp.main(argv, out=lines.append)
    return code, "\n".join(lines)


def test_the_script_shows_four_tools_and_the_same_graph_results(backend):
    code, text = run(["--in-process", "--no-api"])
    assert code == 0, text
    assert "도구 8개" in text
    for name in ("search_drug", "check_item_code", "match_drug", "verify_dictionary", "check_interaction", "check_duplicate"):
        assert f"    {name}: " in text
    assert "사전  202106092  타이레놀정500mg" in text
    assert "전체 25장 중 25장이 같습니다." in text
    assert "[같음] rx8_01: 승인 → 조제 계획" in text
    assert "[같음] rx8_07: 승인 → 다시 승인 대기 (조제 계획을 만들 수 없음)" in text
    assert "[다름]" not in text


def test_the_comparison_notices_when_the_tool_answers_differently(backend):
    class Lying(DrugTools):
        def match_drug(self, order):
            matched, used = super().match_drug(order)
            matched.items[0].item_code = "199999999" if matched.items[0].item_code != "199999999" else None
            return matched, used

    from agent.mcp_server import build_server

    with Lying(build_server(DICT)) as tools:
        results = try_mcp.compare({"rx8_01": try_mcp.load_samples()["rx8_01"]}, tools)
    (name, same, detail), = results
    assert (name, same) == ("rx8_01", False)
    assert detail.startswith("승인 대기에서 다름: ") and "pause" in detail      # 약사 화면의 조제지시서가 달라졌다


def test_the_comparison_uses_the_rule_path_only(backend):
    with in_process(DICT) as tools:
        results = try_mcp.compare({"empty": {"image_id": "e", "doc_type": "prescription", "width": 1, "height": 1, "lines": []}}, tools)
    assert results[0][1] is True                                 # 둘 다 같은 실패 (LLM은 부르지 않는다)


def test_the_samples_are_the_cv_and_demo_prescriptions():
    assert list(try_mcp.load_samples()) == [f"rx_00{i}" for i in range(1, 6)] + [f"rx8_{i:02d}" for i in range(1, 21)]


def test_the_script_output_can_be_written_in_cp949(backend):
    _, text = run(["--in-process", "--no-api"])
    text.encode("cp949")


# ------------------------------------------------------------ 웹의 입구: AGENT_USE_MCP
class OpenedTools:
    def __init__(self, no_api):
        self.no_api, self.closed = no_api, False

    def close(self):
        self.closed = True


def test_the_web_agent_uses_mcp_only_when_the_setting_says_so(backend, monkeypatch):
    opened = []
    monkeypatch.setattr(agent_api, "mcp_from_settings", lambda *, no_api: opened.append(OpenedTools(no_api)) or opened[-1])
    monkeypatch.delenv("AGENT_USE_MCP", raising=False)
    plain = AgentApi.from_settings(dictionary=DICT, api=None, checkpointer=memory_checkpointer())
    assert plain.tools is None and opened == []
    plain.close()

    monkeypatch.setenv("AGENT_USE_MCP", "1")
    with_mcp = AgentApi.from_settings(dictionary=DICT, api=None, checkpointer=memory_checkpointer())
    assert with_mcp.tools is opened[0] and opened[0].no_api is True        # api=None이면 서버도 API 없이
    with_mcp.close()
    assert opened[0].closed


def test_the_web_agent_with_mcp_reaches_the_same_approval(backend, monkeypatch):
    import json

    from agent.config import RESOURCES_DIR

    ocr = json.loads((RESOURCES_DIR / "samples" / "rx8_a1" / "rx8_02.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(agent_api, "mcp_from_settings", lambda *, no_api: in_process(DICT))
    monkeypatch.setenv("AGENT_USE_MCP", "1")
    by_mcp = AgentApi.from_settings(dictionary=DICT, api=None, structure=try_mcp.rule_structure, checkpointer=memory_checkpointer())
    monkeypatch.delenv("AGENT_USE_MCP")
    plain = AgentApi.from_settings(dictionary=DICT, api=None, structure=try_mcp.rule_structure, checkpointer=memory_checkpointer())
    try:
        assert by_mcp.start("P1", ocr) == plain.start("P1", ocr)
    finally:
        by_mcp.close()


# ------------------------------------------------------------ run_graph.py --mcp
_spec2 = importlib.util.spec_from_file_location("run_graph_for_mcp", ROOT / "scripts" / "agent" / "run_graph.py")
run_graph = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(run_graph)


def test_run_graph_can_match_through_mcp_and_closes_the_connection(backend, monkeypatch):
    import threading

    opened = []

    def connect(*, no_api):
        opened.append(no_api)
        return in_process(DICT)

    monkeypatch.setattr(run_graph, "stdio", connect)
    before = threading.active_count()
    lines = []
    code = run_graph.main(["rx8_01", "--mcp", "--no-api", "--approve"], out=lines.append)
    text = "\n".join(lines)
    assert code == 0, text
    assert opened == [True]
    assert "약품 매칭: MCP 도구로 부름 (python -m agent.mcp_server, 도구 8개: search_drug, check_item_code, match_drug, verify_dictionary, check_interaction, check_duplicate, build_plan, get_plan)" in text
    assert "(code)" in text and "조제 계획:" in text
    assert threading.active_count() == before                      # 끝나면 MCP 연결을 닫는다


def test_run_graph_reports_a_server_that_cannot_start(monkeypatch):
    from agent.mcp_client import McpToolError

    def broken(*, no_api):
        raise McpToolError("MCP 서버에 연결하지 못했습니다 (ModuleNotFoundError: No module named 'mcp').")

    monkeypatch.setattr(run_graph, "stdio", broken)
    lines = []
    assert run_graph.main(["rx8_01", "--mcp"], out=lines.append) == 1
    assert lines[0].startswith("MCP 서버 문제: ") and "pip install -r agent/requirements.txt" in lines[1]


def test_a_difference_is_shown_down_to_the_field():
    a = {"pause": {"order": {"items": [{"item_code": "1", "name": "가"}], "warnings": []}}, "status": "x"}
    b = {"pause": {"order": {"items": [{"item_code": "2", "name": "가"}], "warnings": ["API 실패"]}}, "status": "x"}
    assert try_mcp.differences(a, b) == [
        'pause.order.items[0].item_code: "1" → "2"',
        'pause.order.warnings: [] → ["API 실패"]',
    ]
    assert try_mcp.differences({"k": "가" * 100}, {"k": None})[0].endswith("… → null")


def test_the_comparison_with_dur_on_both_sides_is_the_same(backend):
    from fake_dur import FakeDur

    with in_process(DICT, dur=FakeDur()) as tools:
        results = try_mcp.compare(try_mcp.load_samples(), tools, dur=FakeDur())
    assert all(same for _, same, _ in results), results


def test_the_comparison_notices_dur_only_on_one_side(backend):
    from fake_dur import FakeDur

    with in_process(DICT) as tools:                                   # 서버는 DUR 없이
        results = dict((n, s) for n, s, _ in try_mcp.compare({"rx8_01": try_mcp.load_samples()["rx8_01"]}, tools, dur=FakeDur()))
    assert results == {"rx8_01": False}


def test_the_script_uses_dur_on_both_sides_when_the_key_is_set(backend, monkeypatch):
    from fake_dur import FakeDur

    monkeypatch.setattr(try_mcp, "default_api", lambda: None)
    monkeypatch.setattr(try_mcp, "default_dur_api", lambda: FakeDur())
    code, text = run(["--in-process"])
    assert code == 0, text
    assert "| DUR API: 사용" in text and "전체 25장 중 25장이 같습니다." in text
    assert "check_interaction(타이레놀정500mg + 게보린정)\n    찾은 것 없음" in text


def test_try_mcp_can_reach_a_separately_started_server_by_its_address(backend, monkeypatch):
    urls = []
    monkeypatch.setattr(try_mcp, "http", lambda url: urls.append(url) or in_process(DICT))
    monkeypatch.setattr(try_mcp, "stdio", lambda **kw: pytest.fail("주소를 주면 하위 프로세스를 띄우지 않는다"))
    lines = []
    assert try_mcp.main(["--no-api", "--url", "http://127.0.0.1:8765/mcp"], out=lines.append) == 0
    assert urls == ["http://127.0.0.1:8765/mcp"] and "장이 같습니다" in lines[-1]
    assert lines[0].startswith("서버: HTTP: http://127.0.0.1:8765/mcp") and lines[1] == "도구 8개"


def test_try_mcp_says_how_to_start_the_server_when_the_address_does_not_answer(monkeypatch):
    from agent.mcp_client import McpToolError

    def down(url):
        raise McpToolError("MCP 서버에 연결하지 못했습니다 (ConnectError).")

    monkeypatch.setattr(try_mcp, "http", down)
    lines = []
    assert try_mcp.main(["--url", "http://127.0.0.1:8765/mcp"], out=lines.append) == 1
    assert "python -m agent.mcp_server --http" in lines[1]


def test_run_graph_can_use_a_separately_started_server(backend, monkeypatch):
    urls = []
    monkeypatch.setattr(run_graph, "http", lambda url: urls.append(url) or in_process(DICT))
    monkeypatch.setattr(run_graph, "stdio", lambda **kw: pytest.fail("주소를 주면 하위 프로세스를 띄우지 않는다"))
    lines = []
    assert run_graph.main(["rx8_01", "--no-api", "--mcp-url", "http://127.0.0.1:8765/mcp"], out=lines.append) == 0
    assert urls == ["http://127.0.0.1:8765/mcp"]
    assert any(l.startswith("약품 매칭: MCP 도구로 부름 (http://127.0.0.1:8765/mcp, 도구 8개") for l in lines)
