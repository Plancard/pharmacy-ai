"""웹이 부르는 Agent 입구 (agent/api.py) — start · pending · resume · stage · result.

웹의 web/agent_gateway.py(FakeGateway)와 같은 모양으로 쓰이는지 본다. 'fake'(시험용 가짜 LangGraph)와
'real'(진짜 LangGraph, 설치되어 있을 때만)로 같은 시험을 돌린다. 체크포인터는 메모리다.
"""
import copy
import json

import pytest

import fake_langgraph
from agent.api import STAGE_AFTER_PLAN, AgentApi, AgentRunError, Pause
from agent.checkpoint import memory_checkpointer
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import E_OCR_EMPTY
from agent.graph import build_graph
from agent.pipeline import structure_prescription
from agent.plan import build_plan
from agent.schema import DispensingOrder, Plan

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
SAMPLES = RESOURCES_DIR / "samples"
EMPTY_OCR = {"image_id": "x", "doc_type": "prescription", "width": 10, "height": 10, "lines": []}


def load(kind, name):
    return json.loads((SAMPLES / kind / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


class Counting:
    """구조화(LLM 경로까지 포함한 단계)가 몇 번 불렸는지 센다."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prescription_id, ocr):
        self.calls += 1
        return structure_prescription(prescription_id, ocr, llm_fn=lambda *a: pytest.fail("LLM을 부르면 안 됩니다"))


def make_api(structure=None, checkpointer=None):
    graph = build_graph(dictionary=DICT, structure=structure or Counting(), api=None, checkpointer=checkpointer or memory_checkpointer())
    return AgentApi(graph)


def approve_reply(pause):
    return {"action": "approve", "order": pause.payload["order"]}


# ------------------------------------------------------------ start · pending


def test_start_stops_at_the_approval_with_what_the_web_shows(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    assert isinstance(pause, Pause) and pause.hitl_point == "rx_approval"
    assert set(pause.payload) == {"order", "ocr_lines", "notice", "proposals", "memo"}   # 웹 FakeGateway의 payload(order · ocr_lines)에 더했다
    order = DispensingOrder.model_validate(pause.payload["order"])
    assert [d.item_code for d in order.items] == ["202106092", "197900277"]
    assert order.risk_check.level == "warning"
    used = {i for d in order.items for i in d.source_line_ids}
    assert set(pause.payload["ocr_lines"]) == {str(i) for i in used}
    assert pause.payload["notice"].endswith("약사가 확인한 뒤에 승인할 수 있습니다.")
    json.dumps(pause.payload)


def test_pending_gives_the_same_pause_until_the_pharmacist_answers(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    assert api.pending("P0001") == pause
    assert api.pending("P0001") == pause                                 # 몇 번 물어도 같다 (새로고침)
    assert api.stage("P0001") == "rx_approval"


def test_nothing_is_pending_for_an_unknown_prescription(backend):
    api = make_api()
    assert api.pending("P-none") is None and api.stage("P-none") is None


def test_the_same_prescription_cannot_be_started_twice(backend):
    api = make_api()
    api.start("P0001", load("rx8_a1", "rx8_01"))
    with pytest.raises(ValueError, match="이미 시작한 대화"):
        api.start("P0001", load("rx8_a1", "rx8_02"))
    assert api.pending("P0001").payload["order"]["items"][0]["name"] == "타이레놀정 500mg"   # 첫 처방은 그대로 멈춰 있다


def test_a_structure_failure_is_an_error_with_the_a7_code(backend):
    api = make_api()
    with pytest.raises(AgentRunError) as error:
        api.start("P0001", EMPTY_OCR)
    assert error.value.code == E_OCR_EMPTY
    assert api.pending("P0001") is None and api.stage("P0001") == "failed"


def test_a_photo_that_is_not_a_prescription_does_not_stop_at_the_approval(backend):
    api = make_api()
    assert api.start("P0001", None, "tray", prescription_id="P0001") is None
    assert api.stage("P0001") == "not_ready"


def test_a_tray_photo_that_starts_a_conversation_without_a_prescription_is_not_run(backend):
    api = make_api()
    assert api.start("C0001", None, "tray") is None
    result = api.result("C0001")
    assert (result["status"], result["route"], result["prescription_id"]) == ("noted", [], None)
    assert "처방 대화에서 올려 주세요" in result["notice"]


# ------------------------------------------------------------ resume: 승인


def test_approving_makes_the_plan_and_the_stage_waits_for_the_tray(backend):
    structure = Counting()
    api = make_api(structure)
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    assert api.resume("P0001", "rx_approval", approve_reply(pause)) is None
    assert api.pending("P0001") is None
    assert api.stage("P0001") == STAGE_AFTER_PLAN == "await_tray"
    result = api.result("P0001")
    assert result["status"] == "planned" and result["error_code"] is None
    assert result["plan"] == build_plan(DispensingOrder.model_validate(result["order"])).model_dump(mode="json")
    Plan.model_validate(result["plan"])
    assert structure.calls == 1                                          # 이어 갈 때 구조화를 다시 부르지 않는다


def test_resuming_with_the_wrong_point_or_without_a_stop_is_refused(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    with pytest.raises(ValueError, match="멈춰 있는 지점"):
        api.resume("P0001", "inspection", approve_reply(pause))
    with pytest.raises(ValueError, match="멈춰 있는 지점"):
        api.resume("P-none", "rx_approval", approve_reply(pause))
    api.resume("P0001", "rx_approval", approve_reply(pause))
    with pytest.raises(ValueError, match="멈춰 있는 지점"):               # 이미 승인했다
        api.resume("P0001", "rx_approval", approve_reply(pause))


@pytest.mark.parametrize("reply", [
    None, {"action": "reject"}, {"action": "approve"}, {"action": "approve", "order": {"prescription_id": "P0001", "items": []}},
])
def test_a_bad_reply_is_refused_and_the_prescription_stays_where_it_was(backend, reply):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    with pytest.raises(ValueError):
        api.resume("P0001", "rx_approval", reply)
    assert api.pending("P0001") == pause                                # 그대로 멈춰 있어 다시 응답할 수 있다
    assert api.resume("P0001", "rx_approval", approve_reply(pause)) is None


def test_a_reply_for_another_prescription_is_refused(backend):
    api = make_api()
    api.start("P0001", load("rx8_a1", "rx8_01"))
    other = api.start("P0002", load("rx8_a1", "rx8_02"))
    with pytest.raises(ValueError, match="다른 처방"):
        api.resume("P0001", "rx_approval", approve_reply(other))


# ------------------------------------------------------------ resume: 고치기


def test_editing_checks_again_and_stops_again(backend):
    structure = Counting()
    api = make_api(structure)
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    order = copy.deepcopy(pause.payload["order"])
    order["items"] = order["items"][:1]                                 # 게보린을 지운다
    again = api.resume("P0001", "rx_approval", {"action": "edit", "order": order})
    assert again.hitl_point == "rx_approval"
    assert [d["name"] for d in again.payload["order"]["items"]] == ["타이레놀정 500mg"]
    assert again.payload["order"]["risk_check"]["level"] == "none"       # 다시 대조했다
    assert structure.calls == 1                                          # LLM 구조화는 다시 부르지 않는다
    assert api.resume("P0001", "rx_approval", approve_reply(again)) is None
    assert api.result("P0001")["status"] == "planned"


def test_approving_with_changed_lines_is_checked_again_before_planning(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_03"))
    order = copy.deepcopy(pause.payload["order"])
    order["items"][0]["days"] = 2
    again = api.resume("P0001", "rx_approval", {"action": "approve", "order": order})
    assert again is not None and again.payload["order"]["items"][0]["days"] == 2   # 바뀐 지시서로는 한 번 더 승인받는다
    assert api.resume("P0001", "rx_approval", approve_reply(again)) is None
    assert api.result("P0001")["plan"]["days"] == 3


def test_a_drug_the_pharmacist_chose_for_an_unknown_line_stays_after_an_edit(backend):
    api = make_api()
    pause = api.start("P0001", load("cv_a1", "rx_001"))                   # 알모란정은 대상 8종이 아니라 약을 정하지 못했다
    order = copy.deepcopy(pause.payload["order"])
    assert order["items"][1]["match_method"] == "none"
    order["items"][1]["item_code"] = "200308774"                        # 약사가 후보 대신 직접 약을 골랐다 (매칭 방법은 손대지 않음)
    again = api.resume("P0001", "rx_approval", {"action": "edit", "order": order})
    chosen = again.payload["order"]["items"][1]
    assert (chosen["item_code"], chosen["match_method"], chosen["needs_pharmacist_check"]) == ("200308774", None, False)


# ------------------------------------------------------------ 조제 계획을 못 만들 때


def test_a_plan_that_cannot_be_made_comes_back_to_the_approval(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_07"))                  # 한 칸에 3알
    back = api.resume("P0001", "rx_approval", approve_reply(pause))
    assert back.hitl_point == "rx_approval"
    assert back.payload["notice"].startswith("조제 계획을 만들 수 없습니다. 트레이 칸당 최대 2알을 넘습니다")
    assert api.stage("P0001") == "rx_approval"
    order = copy.deepcopy(back.payload["order"])
    order["items"] = order["items"][:2]
    again = api.resume("P0001", "rx_approval", {"action": "edit", "order": order})
    assert api.resume("P0001", "rx_approval", approve_reply(again)) is None
    result = api.result("P0001")
    assert result["status"] == "planned" and result["error_code"] is None


# ------------------------------------------------------------ 다시 켜도 이어 간다


def test_a_new_api_with_the_same_checkpointer_continues_where_it_stopped(backend):
    saver = memory_checkpointer()
    first = make_api(checkpointer=saver)
    pause = first.start("P0001", load("rx8_a1", "rx8_02"))
    second = make_api(structure=lambda pid, ocr: pytest.fail("다시 켠 뒤 구조화를 다시 부르면 안 됩니다"), checkpointer=saver)
    assert second.pending("P0001") == pause
    assert second.resume("P0001", "rx_approval", approve_reply(pause)) is None
    assert second.stage("P0001") == "await_tray"


def test_from_settings_builds_the_graph_with_the_given_tools(backend, monkeypatch):
    from agent import graph

    seen = {}
    monkeypatch.setattr(graph, "default_checkpointer", lambda: seen.setdefault("saver", memory_checkpointer()))
    api = AgentApi.from_settings(dictionary=DICT, structure=Counting(), api=None)
    assert api.graph.checkpointer is seen["saver"]
    assert api.start("P0001", load("rx8_a1", "rx8_04")).hitl_point == "rx_approval"


# ------------------------------------------------------------ 사진과 글 (11단계)


def test_a_prescription_photo_with_text_shows_the_proposal_on_the_approval(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"), "prescription", "게보린 빼 주세요")
    assert [p["kind"] for p in pause.payload["proposals"]] == ["edit_request"]
    assert [d["name"] for d in pause.payload["order"]["items"]] == ["타이레놀정 500mg", "게보린정"]   # 반영하지 않았다


def test_an_unclear_photo_stops_at_type_select_and_goes_on_after_the_choice(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_02"), "med_bag")      # 처방전인데 약봉투이라고 했다
    assert (pause.hitl_point, pause.payload["about"]) == ("type_select", "photo")
    assert api.stage("P0001") == "type_select"
    with pytest.raises(ValueError, match="고를 수 있는 값이 아닙니다"):
        api.resume("P0001", "type_select", {"choice": "question"})
    again = api.resume("P0001", "type_select", {"choice": "prescription"})
    assert again.hitl_point == "rx_approval"
    assert api.resume("P0001", "rx_approval", approve_reply(again)) is None and api.stage("P0001") == "await_tray"


def test_text_alone_starts_a_consultation_without_a_prescription(backend):
    # 11/12 결정: 처방 없는 대화의 글은 상담 (대조 없이 근거 기반 답변. 답변은 16단계)
    api = make_api()
    assert api.start("C0001", None, None, "게보린 먹고 있는데 괜찮나요?", message_id="m1") is None
    result = api.result("C0001")
    assert (result["route"], result["status"], result["proposals"]) == (["consult"], "not_ready", [])
    assert (result["conversation_id"], result["prescription_id"], result["message_id"]) == ("C0001", None, "m1")
    assert result["intents"] == ["patient_info", "question"]


def test_text_that_cannot_be_read_in_a_prescription_conversation_asks_its_meaning(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    api.resume("P0001", "rx_approval", approve_reply(pause))
    pause = api.send("P0001", None, None, "오후 3시")
    assert (pause.hitl_point, pause.payload["choices"]) == ("type_select", ["patient_info", "question", "note"])
    assert api.resume("P0001", "type_select", {"choice": "note"}) is None
    assert api.result("P0001")["memo"] == "오후 3시" and api.result("P0001")["plan"] is not None


# ------------------------------------------------------------ send: 같은 대화의 다음 메시지 (11/12 요청서)


def test_the_next_message_keeps_the_approved_order_and_the_plan(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    api.resume("P0001", "rx_approval", approve_reply(pause))
    before = api.result("P0001")
    assert api.send("P0001", None, "tray", message_id="m2", image_id="tray-1") is None
    after = api.result("P0001")
    assert (after["order"], after["plan"]) == (before["order"], before["plan"])
    assert (after["route"], after["message_id"], api.stage("P0001")) == (["tray"], "m2", "await_tray")


def test_send_also_starts_a_new_conversation(backend):
    api = make_api()
    assert api.send("P0001", load("rx8_a1", "rx8_01"), "prescription", prescription_id="P0001").hitl_point == "rx_approval"


def test_send_is_refused_while_the_conversation_waits_for_the_pharmacist(backend):
    api = make_api()
    api.start("P0001", load("rx8_a1", "rx8_01"))
    with pytest.raises(ValueError, match="약사 응답을 기다리는 대화"):
        api.send("P0001", None, "tray")
    assert api.pending("P0001").hitl_point == "rx_approval"


def test_a_consultation_becomes_a_prescription_conversation(backend):
    api = make_api()
    api.start("C0001", None, None, "같이 먹어도 되나요?")
    pause = api.send("C0001", load("rx8_a1", "rx8_02"), "prescription", prescription_id="C0001")
    assert pause.hitl_point == "rx_approval" and pause.payload["order"]["prescription_id"] == "C0001"
    assert api.result("C0001")["prescription_id"] == "C0001"


def test_text_for_a_consultation_is_not_handled_as_an_open_prescription(backend):
    api = make_api()
    api.start("C0001", None, None, "같이 먹어도 되나요?")
    with pytest.raises(ValueError, match="처방이 없는 대화"):
        api.handle_text("C0001", "게보린 빼 주세요")


# ------------------------------------------------------------ 열린 처방에 글만 (handle_text)


def test_text_for_a_prescription_waiting_for_approval_becomes_a_proposal(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    got = api.handle_text("P0001", "게보린 빼 주세요")
    assert [p["kind"] for p in got["proposals"]] == ["edit_request"] and got["proposals"][0]["drugs"][0]["name"] == "게보린정"
    assert (got["review_needed"], got["consult"], got["ask"], got["memo"]) == (False, False, False, None)
    assert api.pending("P0001") == pause                                # 그래프는 그대로 멈춰 있다 (자동 반영 없음)


def test_an_edit_request_after_the_plan_marks_the_order_for_review(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    api.resume("P0001", "rx_approval", approve_reply(pause))
    got = api.handle_text("P0001", "게보린 빼 주세요")
    assert got["review_needed"] and got["notice"].startswith("조제 내용 재검토 필요") and got["proposals"] == []


def test_a_current_med_after_the_plan_is_a_candidate_for_the_recheck(backend):
    api = make_api()
    pause = api.start("P0001", load("rx8_a1", "rx8_03"))
    api.resume("P0001", "rx_approval", approve_reply(pause))
    got = api.handle_text("P0001", "게보린 먹고 있어요")
    assert [p["kind"] for p in got["proposals"]] == ["patient_info"] and not got["review_needed"]


@pytest.mark.parametrize("text, consult, memo, ask", [
    ("같이 먹어도 되나요?", True, None, False),
    ("메모: 오후 수령", False, "메모: 오후 수령", False),
    ("오후 3시", False, None, True),
])
def test_questions_memos_and_unclear_text(backend, text, consult, memo, ask):
    api = make_api()
    api.start("P0001", load("rx8_a1", "rx8_01"))
    got = api.handle_text("P0001", text)
    assert (got["consult"], got["memo"], got["ask"]) == (consult, memo, ask)
    assert got["choices"] == (["patient_info", "question", "note"] if ask else [])


def test_text_for_an_unknown_prescription_is_refused(backend):
    with pytest.raises(ValueError, match="열린 처방이 없습니다"):
        make_api().handle_text("P-none", "게보린 빼 주세요")


def test_the_dictionary_of_the_api_is_read_from_the_settings_when_not_given(backend, monkeypatch):
    from agent import api as api_module

    monkeypatch.setattr(api_module, "load_dictionary", lambda: DICT)
    api = AgentApi(build_graph(dictionary=DICT, structure=Counting(), api=None, checkpointer=memory_checkpointer()))
    api.start("P0001", load("rx8_a1", "rx8_01"))
    assert api.handle_text("P0001", "게보린 빼 주세요")["proposals"][0]["drugs"] == [{"item_code": "197900277", "name": "게보린정"}]
