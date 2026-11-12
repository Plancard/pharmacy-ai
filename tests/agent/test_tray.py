"""트레이 검수 (14단계): A.3 확인(agent/inspection.py), tray · tray_review 노드, HITL ②, MCP 조제 도구 build_plan · get_plan.

판정은 CV가 한다. 시험에서는 CV 가짜 모듈(cv/stub.py)과 손으로 만든 A.3을 쓴다.
"""
import copy
import json

import pytest

import fake_langgraph
from agent.api import AgentApi, AgentRunError
from agent.checkpoint import memory_checkpointer
from agent.config import RESOURCES_DIR
from agent.cv_adapter import file_inspect, stub_inspect
from agent.drug_dictionary import load_dictionary
from agent.errors import E_CV_INFER, E_PLAN_RANGE, E_TRAY_NOT_FOUND
from agent.graph import build_graph
from agent.inspection import (
    HITL_INSPECTION,
    InspectionError,
    cell_lines,
    inspection_notice,
    read_inspection,
    read_inspection_reply,
)
from agent.mcp_client import McpToolError, in_process
from agent.nodes import (
    CONFIRMED_NOTICE,
    MCP_PLAN_FAILED_WARNING,
    NO_CV_NOTICE,
    NO_IMAGE_NOTICE,
    NO_PLAN_NOTICE,
    REDO_NOTICE,
    Deps,
    plan_node,
    tray_node,
    tray_review_node,
)
from agent.pipeline import structure_prescription
from agent.plan import PlanError, build_plan
from agent.schema import DispensingOrder, Plan

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
SAMPLES = RESOURCES_DIR / "samples"
TYLENOL, GEBORIN = "202106092", "197900277"
PLAN = Plan.model_validate({
    "prescription_id": "P0001", "days": 2,
    "plan": [{"day": d, "time": "morning", "items": [{"item_code": TYLENOL, "count": 1}]} for d in (1, 2)],
})
ORDER_ITEMS = [{"item_code": TYLENOL, "name": "타이레놀정500mg"}]


def load(kind, name):
    return json.loads((SAMPLES / kind / f"{name}.json").read_text(encoding="utf-8"))


def stub(scenario, plan=PLAN):
    return stub_inspect(scenario)("tray-1", plan.model_dump(mode="json"))


def planned_state(**over):
    state = {"conversation_id": "P0001", "prescription_id": "P0001", "status": "planned", "image_id": "tray-1",
             "plan": PLAN.model_dump(mode="json"), "order": {"items": ORDER_ITEMS}, "inspection": None}
    return {**state, **over}


def deps(inspect=None, ask=None, tools=None):
    return Deps(dictionary=DICT, structure=lambda *a: pytest.fail("구조화를 부르면 안 됩니다"), inspect=inspect, ask=ask, tools=tools)


# ------------------------------------------------------------ A.3 확인


@pytest.mark.parametrize("scenario, ok, mismatch, review", [
    ("ok", 12, 0, 0), ("mismatch", 11, 1, 0), ("review", 11, 0, 1), ("rotated", 0, 0, 12),
])
def test_the_cv_stub_results_are_read_as_they_are(scenario, ok, mismatch, review):
    inspection = read_inspection(stub(scenario), PLAN)
    s = inspection.summary
    assert (s.ok, s.mismatch, s.need_review, s.hitl_required) == (ok, mismatch, review, scenario != "ok")


def test_a_result_for_another_prescription_is_refused():
    other = {**stub("ok"), "prescription_id": "P0002"}
    with pytest.raises(InspectionError, match="다른 처방의 검수 결과") as e:
        read_inspection(other, PLAN)
    assert e.value.code == E_CV_INFER


def test_a_result_judged_against_an_old_plan_is_refused():
    old = Plan.model_validate({**PLAN.model_dump(mode="json"), "days": 1, "plan": PLAN.model_dump(mode="json")["plan"][:1]})
    with pytest.raises(InspectionError, match="다른 계획으로 판정함.*2일 아침"):
        read_inspection(stub("ok", old), PLAN)


def test_a_result_that_is_not_a3_is_refused():
    broken = stub("ok")
    broken["cells"] = broken["cells"][:11]
    with pytest.raises(InspectionError, match="A.3"):
        read_inspection(broken, PLAN)


def test_a_result_that_says_no_check_is_needed_while_a_cell_is_not_ok_still_asks_the_pharmacist():
    lying = stub("mismatch")
    lying["summary"]["hitl_required"] = False
    assert read_inspection(lying, PLAN).summary.hitl_required is True           # 약사 확인을 건너뛰지 않는다 (REQ-NF-03)


def test_the_cells_that_are_not_ok_are_described_with_names():
    inspection = read_inspection(stub("review"), PLAN)
    assert cell_lines(inspection, {TYLENOL: "타이레놀정500mg"}) == [
        "1일 아침 확인 필요: 예상 타이레놀정500mg 1 / 실제 타이레놀정500mg 1 (신뢰도 낮음)"]
    missing = read_inspection(stub("mismatch"), PLAN)
    assert cell_lines(missing) == [f"1일 아침 불일치: 예상 {TYLENOL} 1 / 실제 없음"]


def test_a_rotated_tray_asks_for_a_new_photo():
    assert "180°" in inspection_notice(read_inspection(stub("rotated"), PLAN))


@pytest.mark.parametrize("reply", [None, {}, {"action": "approve"}, "confirm"])
def test_only_confirm_or_redo_is_an_answer(reply):
    with pytest.raises(ValueError, match="confirm"):
        read_inspection_reply(reply)


# ------------------------------------------------------------ tray 노드


def test_all_cells_ok_ends_the_inspection_without_asking():
    update = tray_node(planned_state(), deps(stub_inspect("ok")))
    assert update["status"] == "inspected" and update["error_code"] is None
    assert update["inspection"]["summary"]["ok"] == 12 and "전 칸 일치" in update["notice"]


def test_a_cell_that_is_not_ok_goes_to_the_pharmacist():
    update = tray_node(planned_state(), deps(stub_inspect("mismatch")))
    assert update["status"] == "inspecting" and update["inspection"]["summary"]["mismatch"] == 1


@pytest.mark.parametrize("status", ["received", "checked", "approved", "failed"])
def test_without_a_plan_the_tray_is_not_inspected(status):
    called = []
    update = tray_node(planned_state(status=status, plan=None), deps(lambda *a: called.append(a)))
    assert called == [] and update["notice"] == NO_PLAN_NOTICE and "inspection" not in update
    assert update.get("status") == ("noted" if status == "received" else None)


def test_a_photo_after_a_finished_inspection_is_inspected_again():
    assert tray_node(planned_state(status="inspected"), deps(stub_inspect("ok")))["status"] == "inspected"


@pytest.mark.parametrize("state, inspect, notice", [
    (planned_state(image_id=None), stub_inspect("ok"), NO_IMAGE_NOTICE),
    (planned_state(), None, NO_CV_NOTICE),                                        # 가짜 '전 칸 일치'를 만들지 않는다
])
def test_without_a_photo_or_cv_nothing_is_judged(state, inspect, notice):
    update = tray_node(state, deps(inspect))
    assert update["notice"] == notice and update["error_code"] == E_CV_INFER
    assert "status" not in update and "inspection" not in update                  # planned 그대로: 다시 찍은 사진을 기다린다


def test_a_cv_error_keeps_its_code_and_the_plan_waits_for_a_new_photo():
    update = tray_node(planned_state(), deps(stub_inspect("no_tray")))
    assert update["error_code"] == E_TRAY_NOT_FOUND and "status" not in update
    assert update["errors"][0].startswith("[tray] E_TRAY_NOT_FOUND TrayNotFound")


def test_any_other_failure_of_the_cv_is_a_model_error():
    def broken(image_id, plan):
        raise RuntimeError("CUDA 메모리 부족")

    update = tray_node(planned_state(), deps(broken))
    assert update["error_code"] == E_CV_INFER and "CUDA 메모리 부족" in update["notice"]


def test_the_cv_gets_the_photo_id_and_the_current_plan():
    seen = []
    tray_node(planned_state(image_id="img-7"), deps(lambda image_id, plan: seen.append((image_id, plan)) or stub("ok")))
    assert seen == [("img-7", PLAN.model_dump(mode="json"))]


def test_a_result_that_does_not_fit_the_plan_is_not_used():
    update = tray_node(planned_state(), deps(lambda image_id, plan: {**stub("ok"), "prescription_id": "P9"}))
    assert update["error_code"] == E_CV_INFER and "inspection" not in update


# ------------------------------------------------------------ tray_review 노드 (HITL ②)


def inspecting(scenario="mismatch"):
    return planned_state(status="inspecting", inspection=read_inspection(stub(scenario), PLAN).model_dump(mode="json"))


def test_the_pharmacist_sees_the_cells_the_plan_and_the_names():
    shown = []
    tray_review_node(inspecting(), deps(ask=lambda payload: shown.append(payload) or {"action": "confirm"}))
    payload = shown[0]
    assert payload["hitl_point"] == HITL_INSPECTION == "inspection" and payload["choices"] == ["confirm", "redo"]
    assert payload["cells"] == ["1일 아침 불일치: 예상 타이레놀정500mg 1 / 실제 없음"]
    assert payload["names"] == {TYLENOL: "타이레놀정500mg"} and payload["plan"] == PLAN.model_dump(mode="json")
    assert payload["inspection"]["summary"]["mismatch"] == 1


def test_confirming_finishes_the_inspection_and_records_what_was_checked():
    update = tray_review_node(inspecting(), deps(ask=lambda payload: {"action": "confirm"}))
    assert update["status"] == "inspected" and update["notice"] == CONFIRMED_NOTICE
    assert update["notes"] == ["검수 확인(약사): 1일 아침 불일치: 예상 타이레놀정500mg 1 / 실제 없음"]


def test_redo_goes_back_to_waiting_for_the_tray():
    update = tray_review_node(inspecting(), deps(ask=lambda payload: {"action": "redo"}))
    assert update["status"] == "planned" and update["notice"] == REDO_NOTICE


# ------------------------------------------------------------ 그래프와 입구 (AgentApi)


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


class CountingInspect:
    """CV가 몇 번 불렸는지 센다. 시나리오는 차례로 쓴다."""

    def __init__(self, *scenarios):
        self.scenarios = list(scenarios)
        self.calls = 0

    def __call__(self, image_id, plan):
        self.calls += 1
        return stub_inspect(self.scenarios.pop(0))(image_id, plan)


def planned_api(inspect):
    graph = build_graph(dictionary=DICT, api=None, checkpointer=memory_checkpointer(), inspect=inspect,
                        structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=lambda *a: pytest.fail("LLM")))
    api = AgentApi(graph, DICT)
    pause = api.start("P0001", load("rx8_a1", "rx8_01"))
    assert api.resume("P0001", "rx_approval", {"action": "approve", "order": pause.payload["order"]}) is None
    return api


def test_a_tray_with_all_cells_ok_finishes_without_stopping(backend):
    api = planned_api(stub_inspect("ok"))
    assert api.send("P0001", None, "tray", message_id="m2", image_id="tray-1") is None
    result = api.result("P0001")
    assert api.stage("P0001") == "inspected" and result["inspection"]["summary"]["ok"] == 12
    assert result["plan"]["prescription_id"] == result["inspection"]["prescription_id"] == "P0001"


def test_a_mismatch_stops_at_hitl_2_and_the_cv_runs_only_once(backend):
    inspect = CountingInspect("mismatch")
    api = planned_api(inspect)
    pause = api.send("P0001", None, "tray", image_id="tray-1")
    assert pause.hitl_point == "inspection" and api.stage("P0001") == "inspection"
    assert api.resume("P0001", "inspection", {"action": "confirm"}) is None
    assert inspect.calls == 1                                       # 이어 갈 때 CV 추론을 다시 하지 않는다
    result = api.result("P0001")
    assert result["status"] == "inspected" and result["notes"][-1].startswith("검수 확인(약사): 1일 아침 불일치")


def test_redo_waits_for_a_new_photo_which_is_inspected_again(backend):
    inspect = CountingInspect("review", "ok")
    api = planned_api(inspect)
    api.send("P0001", None, "tray", image_id="tray-1")
    assert api.resume("P0001", "inspection", {"action": "redo"}) is None and api.stage("P0001") == "await_tray"
    assert api.send("P0001", None, "tray", image_id="tray-2") is None
    assert (api.stage("P0001"), inspect.calls) == ("inspected", 2)


def test_a_wrong_answer_to_hitl_2_leaves_the_graph_waiting(backend):
    api = planned_api(stub_inspect("mismatch"))
    api.send("P0001", None, "tray", image_id="tray-1")
    with pytest.raises(ValueError, match="confirm"):
        api.resume("P0001", "inspection", {"action": "approve"})
    assert api.pending("P0001").hitl_point == "inspection"


def test_a_failed_inspection_is_an_error_with_its_code_and_the_plan_stays(backend):
    api = planned_api(CountingInspect("no_tray", "ok"))
    before = api.result("P0001")
    with pytest.raises(AgentRunError) as e:
        api.send("P0001", None, "tray", image_id="tray-1")
    assert e.value.code == E_TRAY_NOT_FOUND and api.stage("P0001") == "await_tray"
    assert api.result("P0001")["plan"] == before["plan"]
    assert api.send("P0001", None, "tray", image_id="tray-2") is None and api.stage("P0001") == "inspected"


def test_a_tray_photo_before_the_approval_is_not_sent_while_waiting(backend):
    graph = build_graph(dictionary=DICT, api=None, checkpointer=memory_checkpointer(), inspect=stub_inspect("ok"),
                        structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=lambda *a: pytest.fail("LLM")))
    api = AgentApi(graph, DICT)
    api.start("P0001", load("rx8_a1", "rx8_01"))
    with pytest.raises(ValueError, match="약사 응답을 기다리는 대화"):
        api.send("P0001", None, "tray", image_id="tray-1")


# ------------------------------------------------------------ 조제 계획: MCP 조제 도구 (build_plan · get_plan)


def approved_order():
    order = DispensingOrder.model_validate(load("a5_order", "s01_combo_a"))
    return {"status": "approved", "order": order.model_dump(mode="json")}


class FakeTools:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def build_plan(self, order):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result if self.result is not None else build_plan(order)


def test_the_plan_is_made_by_the_mcp_tool_when_there_is_one():
    tools = FakeTools()
    update = plan_node(approved_order(), deps(tools=tools))
    assert tools.calls == 1 and update["status"] == "planned" and "notes" not in update


def test_a_broken_mcp_server_does_not_stop_the_plan():
    update = plan_node(approved_order(), deps(tools=FakeTools(error=McpToolError("연결이 끊겼습니다"))))
    assert update["status"] == "planned" and update["plan"]["days"] == 3
    assert update["notes"] == [f"{MCP_PLAN_FAILED_WARNING}: 연결이 끊겼습니다"]


def test_a_plan_the_mcp_tool_cannot_make_goes_back_with_its_code():
    update = plan_node(approved_order(), deps(tools=FakeTools(error=PlanError("칸당 2알을 넘습니다", code=E_PLAN_RANGE))))
    assert (update["status"], update["error_code"], update["plan"]) == ("failed", E_PLAN_RANGE, None)


@pytest.fixture
def server_tools():
    with in_process(DICT) as tools:
        yield tools


def order_without_risk(**item):
    data = load("a5_order", "s01_combo_a")
    data.pop("risk_check", None)
    data["items"][0].update(item)
    return DispensingOrder.model_validate(data)


def test_build_plan_over_mcp_is_the_same_as_the_function(server_tools):
    order = order_without_risk()
    assert server_tools.build_plan(order) == build_plan(order)


def test_build_plan_over_mcp_gives_the_reason_and_code_when_it_cannot(server_tools):
    with pytest.raises(PlanError, match="트레이\\(3일\\)를 넘습니다") as e:
        server_tools.build_plan(order_without_risk(days=5))
    assert e.value.code == E_PLAN_RANGE
    answer = server_tools.call("build_plan", order=order_without_risk(days=5).model_dump(mode="json"))
    assert answer["plan"] is None and answer["error"]["code"] == E_PLAN_RANGE and answer["lines"] == []


def test_build_plan_refuses_something_that_is_not_an_order(server_tools):
    with pytest.raises(McpToolError, match="A.5"):
        server_tools.call("build_plan", order={"items": []})


def test_get_plan_shows_the_cells_with_names_and_can_pick_one_cell(server_tools):
    plan = build_plan(order_without_risk()).model_dump(mode="json")
    everything = server_tools.call("get_plan", plan=plan)
    assert everything["days"] == 3 and len(everything["cells"]) == len(plan["plan"])
    one = server_tools.call("get_plan", plan=plan, day=2, time="noon")
    assert [c["label"] for c in one["cells"]] == ["2일 점심"]
    assert one["cells"][0]["items"] == [{"item_code": TYLENOL, "name": "타이레놀정500mg", "count": 1},
                                        {"item_code": GEBORIN, "name": "게보린정", "count": 1}]
    assert one["pills"] == 2 and everything["lines"][0].startswith("아침: 1일 타이레놀정500mg 1 · 게보린정 1")


@pytest.mark.parametrize("arguments, words", [({"time": "lunch"}, "time은"), ({"plan_broken": True}, "A.4")])
def test_get_plan_refuses_a_wrong_time_or_plan(server_tools, arguments, words):
    plan = build_plan(order_without_risk()).model_dump(mode="json")
    if arguments.pop("plan_broken", False):
        plan = {**plan, "days": 9}
    with pytest.raises(McpToolError, match=words):
        server_tools.call("get_plan", plan=plan, **arguments)


# ------------------------------------------------------------ CV 연결 (agent/cv_adapter.py)


def test_file_inspect_turns_the_photo_id_into_a_path_for_the_cv():
    seen = []
    inspect = file_inspect(lambda path, plan: seen.append((path, plan)) or "A.3", lambda image_id: f"/uploads/{image_id}.jpg")
    assert inspect("img-3", {"p": 1}) == "A.3" and seen == [("/uploads/img-3.jpg", {"p": 1})]


def test_the_stub_needs_a_known_scenario():
    with pytest.raises(ValueError, match="알 수 없는 시나리오"):
        stub_inspect("broken")("tray-1", copy.deepcopy(PLAN.model_dump(mode="json")))
