"""웹 ↔ 진짜 Agent 연결부 (web/langgraph_gateway.py, 11/16 초안).

Agent는 진짜 build_graph · AgentApi를 쓰되, LangGraph는 가짜(tests/agent/fake_langgraph.py)와 진짜 둘 다로 돌린다
(진짜 langgraph가 없으면 'real' 쪽은 건너뛴다). 의약품 API · OpenAI · DB는 부르지 않는다: 규칙 구조화 + 메모리 체크포인터.
OCR은 cv/stub.py의 고정 결과다. 웹 DB는 test_prescriptions.py의 가짜(FakeRx)를 그대로 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import fake_langgraph  # noqa: E402

from test_prescriptions import approve, fake_rx, make_client, upload  # noqa: E402,F401  (fake_rx는 fixture)
from web.agent_gateway import AgentError, Pause, Wait  # noqa: E402
from web.langgraph_gateway import LangGraphGateway, stub_ocr  # noqa: E402


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


def rule_only(pid, ocr):
    from agent.pipeline import structure_prescription

    return structure_prescription(pid, ocr, llm_fn=lambda *a: pytest.fail("LLM을 부르면 안 됩니다"))


def make_gateway(ocr=stub_ocr):
    from agent.api import AgentApi
    from agent.checkpoint import memory_checkpointer
    from agent.cv_adapter import stub_inspect
    from agent.drug_dictionary import load_dictionary
    from agent.graph import build_graph

    dictionary = load_dictionary()
    graph = build_graph(dictionary=dictionary, structure=rule_only, api=None, checkpointer=memory_checkpointer(),
                        inspect=stub_inspect("ok"))
    return LangGraphGateway(AgentApi(graph, dictionary), ocr)


IMAGE = Path("var/uploads/upload.jpg")      # 파일은 없어도 된다: stub OCR은 사진을 읽지 않는다


@pytest.fixture(autouse=True)
def demo_sample(monkeypatch):
    """가상 처방전 rx8_01 (타이레놀정500mg · 게보린정, 대상 8종만이라 바로 승인할 수 있다)."""
    monkeypatch.setenv("CV_STUB_SAMPLE", "rx8_01")
    monkeypatch.delenv("CV_STUB_SCENARIO", raising=False)


# ------------------------------------------------------------ 연결부만


def test_start_stops_at_the_approval_with_what_the_card_needs(backend):
    gw = make_gateway()
    pause = gw.start("P0001", IMAGE)
    assert isinstance(pause, Pause) and pause.hitl_point == "rx_approval"
    assert {"order", "ocr_lines", "ocr_size"} <= set(pause.payload)           # FakeGateway와 같은 칸 (+ notice · 제안 · 메모)
    assert pause.payload["order"]["prescription_id"] == "P0001" and pause.payload["order"]["items"]
    some = next(iter(pause.payload["ocr_lines"].values()))
    assert set(some) == {"text", "conf", "bbox"}
    assert gw.pending("P0001") == pause and gw.approved_order("P0001") is None


@pytest.mark.parametrize("sample, names", [("rx8_03", None), ("rx_004", "알모란정"), ("", "알모란정")])
def test_the_stub_sample_setting_picks_the_ocr(monkeypatch, sample, names):
    monkeypatch.setenv("CV_STUB_SAMPLE", sample)
    a1 = stub_ocr(IMAGE, "img1")
    text = " ".join(l["text"] for l in a1["lines"])
    assert a1["image_id"] == "img1" and a1["lines"]
    assert (names in text) if names else ("알모란정" not in text)


def test_approve_ends_with_waiting_tray_and_keeps_the_order(backend):
    gw = make_gateway()
    pause = gw.start("P0001", IMAGE)
    got = gw.resume("P0001", "rx_approval", {"action": "approve", "order": pause.payload["order"]})
    assert got == Wait("waiting_tray")
    assert gw.pending("P0001") is None
    assert [d["name"] for d in gw.approved_order("P0001")["items"]] == [d["name"] for d in pause.payload["order"]["items"]]


def test_a_wrong_reply_is_caught_before_the_graph_runs(backend):
    gw = make_gateway()
    gw.start("P0001", IMAGE)
    with pytest.raises(ValueError):
        gw.check_reply("P0001", "inspection", {"action": "confirm"})              # 지점이 다르다
    with pytest.raises(ValueError):
        gw.check_reply("P0001", "rx_approval", {"action": "fly"})
    assert gw.pending("P0001").hitl_point == "rx_approval"                        # 그대로 멈춰 있다


def test_an_empty_ocr_becomes_the_same_error_code(backend):
    def empty(path, image_id):
        return {"image_id": image_id, "doc_type": "prescription", "width": 10, "height": 10, "lines": []}

    with pytest.raises(AgentError) as e:
        make_gateway(empty).start("P0001", IMAGE)
    assert e.value.code == "E_OCR_EMPTY"


def test_cv_errors_keep_their_code(backend, monkeypatch):
    monkeypatch.setenv("CV_STUB_SCENARIO", "ocr_empty")
    with pytest.raises(Exception) as e:
        make_gateway().start("P0001", IMAGE)
    assert getattr(e.value, "code", None) == "E_OCR_EMPTY"


# ------------------------------------------------------------ 웹 전체 (업로드 → 승인 대기 → 승인 → 트레이 대기)


def test_the_web_flow_runs_on_the_real_agent(backend, monkeypatch, tmp_path, fake_rx):
    client = make_client(monkeypatch, tmp_path)
    client.application.extensions["agent"] = make_gateway()

    rx_id = upload(client).get_json()["id"]
    d = client.get(f"/api/prescriptions/{rx_id}").get_json()
    assert (d["status"], d["hitl_point"]) == ("waiting_approval", "rx_approval")
    card = d["card"]
    assert card["order"]["prescription_id"] == rx_id and "notice" in card and "blockers" in card
    assert card["blockers"] == [] and [i["name"] for i in card["order"]["items"]] == ["타이레놀정 500mg", "게보린정"]   # 처방전에 적힌 그대로 (코드로 정해졌다)

    r = approve(client, rx_id, confirmed=True)
    assert r.status_code == 202
    d = client.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "waiting_tray" and d["card"]["order"]["prescription_id"] == rx_id
    assert fake_rx.approvals[0]["action"] == "approve"


def test_the_backend_setting_picks_the_gateway(monkeypatch, tmp_path):
    from web import agent_gateway

    made = []
    monkeypatch.setattr("web.langgraph_gateway.from_settings", lambda: made.append(1) or "LG")

    class App:
        config = {"AGENT_BACKEND": "langgraph", "VAR_DIR": tmp_path, "FAKE_AGENT_SCENARIO": "warning"}
        extensions: dict = {}

    agent_gateway.init_app(App)
    assert App.extensions["agent"] == "LG" and made == [1]
    App.config["AGENT_BACKEND"] = "other"
    with pytest.raises(RuntimeError, match="fake · langgraph"):
        agent_gateway.init_app(App)


def test_a_reply_the_agent_would_refuse_is_rejected_with_400_and_not_recorded(backend, monkeypatch, tmp_path, fake_rx):
    """라우트가 응답을 기록하기 전에 check_reply로 거른다. 잘못된 응답이 백그라운드에서 처방을 실패로 만들지 않는다."""
    client = make_client(monkeypatch, tmp_path)
    gw = make_gateway()
    client.application.extensions["agent"] = gw
    rx_id = upload(client).get_json()["id"]
    pause = gw.pending(rx_id)

    wrong = dict(pause.payload["order"], prescription_id="다른-처방")
    monkeypatch.setattr("web.workflow.build_edited_order", lambda order, edits: wrong)
    r = approve(client, rx_id, action="edit", items=[{"name": "x"}])
    assert r.status_code == 400, r.get_json()
    assert "다른-처방" in r.get_json()["error"] or "처방" in r.get_json()["error"]   # 연결부가 되돌려 준 문장
    assert fake_rx.approvals == []                                   # 기록하지 않았다
    d = client.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "waiting_approval"                         # 처방은 그대로 승인 대기다
