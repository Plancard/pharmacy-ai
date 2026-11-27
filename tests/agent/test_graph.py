"""그래프 두 장 (agent/graph.py) — 상위 그래프(라우터의 규칙 분기)와 처방 서브그래프(구조화 → 매칭 → 위험 신호 대조 → 승인 대기 → 조제 계획).

시험은 세 겹이다.
1. 설계표(PRESCRIPTION_CHART · TOP_CHART)만 보는 시험. LangGraph가 필요 없다.
2. 'fake' 쪽: 시험용 가짜 LangGraph(fake_langgraph.py)에 그래프를 끼워 끝까지 돌린다. LangGraph가 없어도 돈다.
3. 'real' 쪽: 같은 시험을 진짜 LangGraph로 돌린다. langgraph가 설치되어 있을 때만 돌고, 없으면 건너뛴다.

승인 대기(HITL ①)에서 그래프가 멈추므로 메모리 체크포인터를 붙여 돌리고, 약사 응답은 resume_run으로 넣는다.
멈춘 동안 그래프가 돌려준 값은 멈춘 노드에 들어가기 전의 상태다(진짜 LangGraph의 규칙). 승인 화면의 조제지시서는
멈춘 내용(__interrupt__)에 있다.
"""
import json
import subprocess
import sys
import typing
from pathlib import Path

import pytest

import fake_langgraph
from agent import graph
from agent.config import RESOURCES_DIR
from agent.drug_api import ApiDrug
from agent.drug_dictionary import dictionary_from_data, load_dictionary
from agent.errors import E_LLM, E_OCR_EMPTY, E_PLAN_RANGE, ConfigError, StructureError
from agent.graph import (
    END,
    PRESCRIPTION_CHART,
    STEPS,
    SUBGRAPH_NODE,
    TOP_CHART,
    _subgraph_step,
    after_approve,
    after_structure,
    after_router,
    build_graph,
    build_prescription_graph,
    resume_run,
    run_message,
    run_prescription,
    thread_config,
)
from agent.checkpoint import memory_checkpointer
from agent.nodes import NO_API_WARNING
from agent.pipeline import structure_prescription
from agent.risk import UNCHECKED_TAIL
from agent.schema import DispensingOrder, DrugLine, Plan
from agent.state import FLOWS, GraphState
from graph_calls import first_state, send

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = RESOURCES_DIR / "samples"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
STATE_KEYS = set(typing.get_type_hints(GraphState))
EMPTY_OCR = {"image_id": "x", "doc_type": "prescription", "width": 10, "height": 10, "lines": []}
CHARTS = {"top": TOP_CHART, "prescription": PRESCRIPTION_CHART}
RESULT_KEYS = ("order", "status", "notice", "error_code", "errors")     # 서브그래프 혼자 돌린 결과와 상위 그래프의 결과가 같아야 하는 칸


def load(kind, sample_id):
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback):
    raise AssertionError("이 시험에서는 LLM을 부르면 안 됩니다")


def real_structure(prescription_id, ocr):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


# ------------------------------------------------------------ 1. 설계표


@pytest.fixture(params=sorted(CHARTS))
def chart(request):
    return CHARTS[request.param]


def test_node_names_do_not_collide_with_state_fields(chart):
    assert set(chart.nodes).isdisjoint(STATE_KEYS)       # LangGraph는 상태 칸과 같은 이름의 노드를 거절한다


def test_every_arrow_starts_and_ends_at_a_known_place(chart):
    known = set(chart.nodes)
    assert chart.entry in known
    for source, target in chart.edges:
        assert source in known and target in known | {END}
    for source, (choose, targets) in chart.branches.items():
        assert source in known and callable(choose)
        assert set(targets.values()) <= known | {END}
    assert not [s for s, _ in chart.edges if s in chart.branches], "고정 화살표와 갈림길이 한 노드에 함께 있으면 안 된다"
    assert len({s for s, _ in chart.edges}) == len(chart.edges), "한 노드에서 나가는 고정 화살표는 하나"
    assert len(set(chart.nodes)) == len(chart.nodes)


def test_every_node_has_a_way_out(chart):
    assert all(chart.successors()[name] for name in chart.nodes)


def test_every_node_can_be_reached_from_the_entry(chart):
    reached, todo = set(), [chart.entry]
    while todo:
        node = todo.pop()
        if node in reached or node == END:
            continue
        reached.add(node)
        todo.extend(chart.successors()[node])
    assert reached == set(chart.nodes)


def test_every_node_leads_to_the_end(chart):
    nxt = chart.successors()
    leads_to_end = {name for name, targets in nxt.items() if END in targets}
    changed = True
    while changed:
        changed = False
        for name, targets in nxt.items():
            if name not in leads_to_end and targets & leads_to_end:
                leads_to_end.add(name)
                changed = True
    assert leads_to_end == set(chart.nodes)


def without_the_ways_back_through_the_approval(chart):
    """승인 대기에서 돌아가는 한 갈래(고치기: approve -> match)를 뺀 다음 노드 표."""
    nxt = chart.successors()
    if "approve" in nxt:
        nxt["approve"] = nxt["approve"] - {"match"}
    return nxt


def test_the_only_ways_back_go_through_the_approval_so_the_router_runs_once_per_message(chart):
    nxt = without_the_ways_back_through_the_approval(chart)
    assert not any(chart.entry in targets for targets in chart.successors().values())   # 시작 노드로 돌아오는 화살표가 없다

    state: dict[str, int] = {}                                         # 0 = 지나는 중, 1 = 끝남

    def visit(node):
        if node == END or state.get(node) == 1:
            return
        assert state.get(node) != 0, f"{node}로 돌아오는 길이 있습니다"
        state[node] = 0
        for target in nxt[node]:
            visit(target)
        state[node] = 1

    visit(chart.entry)


def test_the_only_way_back_is_the_edit():
    nxt = PRESCRIPTION_CHART.successors()
    assert nxt["approve"] == {"match", END} and nxt["make_plan"] == {"approve"}   # 계획 실패도 그대로 승인 대기로 간다


def test_every_node_function_belongs_to_one_graph_and_the_subgraph_node_has_none():
    functions = [name for c in CHARTS.values() for name in c.nodes if name != SUBGRAPH_NODE]
    assert sorted(functions) == sorted(STEPS)                          # 함수가 없는 노드도, 쓰이지 않는 함수도 없다
    assert SUBGRAPH_NODE in TOP_CHART.nodes and SUBGRAPH_NODE not in STEPS


def test_the_top_graph_starts_at_the_router_and_the_subgraph_at_structure():
    assert (TOP_CHART.entry, PRESCRIPTION_CHART.entry) == ("router", "structure")


def test_the_router_can_start_any_flow_and_the_flows_only_go_forward():
    from agent.router import FLOW_ORDER

    _, targets = TOP_CHART.branches[TOP_CHART.entry]
    assert set(targets) - {"end"} == set(FLOWS) == set(FLOW_ORDER)
    for i, flow in enumerate(FLOW_ORDER):                             # 앞에서 뒤로만: 되돌아가는 길이 없다
        _, after = TOP_CHART.branches[flow]
        extra = {"tray": {"tray_review"}}.get(flow, set())             # 약사 확인 노드를 하나 더 거칠 수 있다
        assert set(after) == {*FLOW_ORDER[i + 1:], "end", *extra}
        assert after["end"] == END and all(after[f] == f for f in FLOW_ORDER[i + 1:])
    _, after_review = TOP_CHART.branches["tray_review"]               # 검수 확인 뒤에는 트레이 다음 흐름으로만
    assert set(after_review) == {*FLOW_ORDER[FLOW_ORDER.index("tray") + 1:], "end"}
    assert [n for n, targets in TOP_CHART.successors().items() if "tray_review" in targets] == ["tray"]
    assert TOP_CHART.nodes == ("router", "prescription", "tray", "tray_review")   # 처방 · 트레이 흐름만


def test_the_prescription_flow_is_a_subgraph_that_goes_structure_match_risk_plan_approve():
    assert "prescription" in TOP_CHART.nodes and TOP_CHART.edges == ()
    assert PRESCRIPTION_CHART.branches["structure"][1] == {"match": "match", "end": END}
    assert PRESCRIPTION_CHART.branches["approve"][1] == {"match": "match", "end": END}
    assert PRESCRIPTION_CHART.edges == (("match", "risk"), ("risk", "make_plan"), ("make_plan", "approve"))
    assert PRESCRIPTION_CHART.nodes == ("structure", "match", "risk", "make_plan", "approve")


def test_the_plan_is_made_right_before_the_approval():
    nxt = PRESCRIPTION_CHART.successors()                              # 명세서 2.2절: 계획은 승인 카드에 함께 보인다
    assert [name for name, targets in nxt.items() if "make_plan" in targets] == ["risk"]
    assert [name for name, targets in nxt.items() if "approve" in targets] == ["make_plan"]


@pytest.mark.parametrize("state, step", [
    ({"status": "edited", "order": {"items": []}}, "match"),            # 약사가 고쳤으면 다시 매칭 · 대조 · 계획
    ({"status": "planned", "order": {"items": []}}, "end"),             # 승인: 승인 카드의 계획을 확정하고 끝
    ({"status": "checked", "order": {"items": []}}, "end"),
    ({"status": "edited", "order": None}, "end"),
    ({"status": "planned"}, "end"),
    ({}, "end"),
])
def test_after_approve_ends_when_approved_and_goes_back_to_matching_when_edited(state, step):
    assert after_approve(state) == step
    assert step in PRESCRIPTION_CHART.branches["approve"][1]


def test_the_risk_check_sits_between_matching_and_approval():
    nxt = PRESCRIPTION_CHART.successors()
    assert nxt["match"] == {"risk"} and nxt["risk"] == {"make_plan"}    # 매칭을 거친 약만 대조하고, 대조 · 계획을 거친 것만 승인 자리로 간다


@pytest.mark.parametrize("route, step", [
    (["prescription"], "prescription"), (["tray"], "tray"),
    (["tray", "prescription"], "prescription"),                        # 적힌 순서가 아니라 정해진 순서(FLOW_ORDER)로
    ([], "end"), (None, "end"), ("prescription", "end"), (["unknown"], "end"),   # 모르는 값은 건너뛴다
    (["consult"], "end"), (["recheck"], "end"), (["ask_type"], "end"),  # v0.8에서 뺀 흐름은 없다
])
def test_after_the_router_goes_to_the_first_flow_of_the_route(route, step):
    choose, targets = TOP_CHART.branches["router"]
    assert after_router({"route": route}) == choose({"route": route}) == step
    assert step in targets


@pytest.mark.parametrize("after, route, step", [
    ("prescription", ["prescription", "tray"], "tray"),
    ("prescription", ["prescription"], "end"),
    ("tray", ["prescription", "tray"], "end"),                          # 이미 지난 흐름으로 돌아가지 않는다
])
def test_each_flow_goes_to_the_next_flow_of_the_route(after, route, step):
    choose, targets = TOP_CHART.branches[after]
    assert choose({"route": route}) == step and step in targets


def test_after_the_router_without_a_route_ends():
    assert after_router({}) == "end"


@pytest.mark.parametrize("state, step", [
    ({"status": "structured", "order": {"items": []}}, "match"),
    ({"status": "structured", "order": None}, "end"),
    ({"status": "structured"}, "end"),
    ({"status": "failed", "order": None}, "end"),
    ({"status": "failed", "order": {"items": []}}, "end"),
    ({"status": "received", "order": {"items": []}}, "end"),
    ({"status": "matched", "order": {"items": []}}, "end"),
    ({}, "end"),
])
def test_after_structure_goes_on_only_when_the_order_was_made(state, step):
    assert after_structure(state) == step
    assert step in PRESCRIPTION_CHART.branches["structure"][1]


def test_importing_the_graph_does_not_import_langgraph():
    code = "import sys, agent.graph; sys.exit(1 if 'langgraph' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")


# ------------------------------------------------------------ 2·3. 돌리기 (fake · real)


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


@pytest.fixture
def calls(monkeypatch):
    """노드가 불린 순서를 기록한다. 그래프를 만들기 전에 끼워야 하므로 시험 안에서 make()는 이 뒤에 부른다."""
    log: list[str] = []

    def recording(name, fn):
        def wrapper(state, deps):
            log.append(name)
            return fn(state, deps)
        return wrapper

    for name, fn in list(STEPS.items()):
        monkeypatch.setitem(STEPS, name, recording(name, fn))
    return log


def make(*, structure=real_structure, api=None, checkpointer=None):
    return build_graph(dictionary=DICT, structure=structure, api=api, checkpointer=checkpointer or memory_checkpointer())


def make_sub(*, structure=real_structure, api=None, checkpointer=None):
    return build_prescription_graph(dictionary=DICT, structure=structure, api=api, checkpointer=checkpointer or memory_checkpointer())


def line(name, times=("morning",), days=3):
    return DrugLine(item_code=None, name=name, printed_code=None, count_per_dose=1, times=list(times), days=days)


def structure_of(*names, times=("morning",), days=3):
    """OCR을 보지 않고 약품 이름 줄만 가진 조제지시서를 돌려주는 구조화 (시연 조합을 만들어 넣는 데 쓴다)."""
    return lambda prescription_id, ocr: DispensingOrder(prescription_id=prescription_id, items=[line(n, times, days) for n in names])


def paused(state):
    """그래프가 멈췄으면 멈춘 내용(승인 화면), 아니면 None."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value if interrupts else None


def approve(compiled, prescription_id, state):
    """화면의 조제지시서를 그대로 승인한다."""
    return resume_run(compiled, prescription_id, {"action": "approve", "order": paused(state)["order"]})


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_a_sample_prescription_stops_at_the_approval(backend, calls, sample_id):
    compiled = make()
    state = send(compiled, f"P-{sample_id}", load("cv_a1", sample_id))
    assert calls == ["router", "structure", "match", "risk", "make_plan", "approve"]
    shown = paused(state)
    assert shown["hitl_point"] == "rx_approval"
    order = DispensingOrder.model_validate(shown["order"])
    assert order.prescription_id == f"P-{sample_id}"
    assert len(order.items) == len(load("cv_expected", sample_id)["items"])
    # 가상 처방전의 인쇄 코드(9000000xx)는 품목기준코드가 아니라서, 모든 줄이 이름 단계로 가고 약사 확인이 붙는다
    assert len(order.flagged_indices) == len(order.items)
    assert order.risk_check is not None and order.risk_check.scope == "rx_only"
    assert f"약사 승인을 기다립니다. 확인이 필요한 약품이 {len(order.items)}줄 있습니다." in shown["notice"]
    assert (shown["plan"] is not None) is order.all_matched              # 약을 정하지 못한 줄이 있으면 계획 없이 이유만
    order.check_ready_for_web()                                         # 승인 화면에 넘겨도 되는 조제지시서다
    # 멈춘 동안의 상위 상태는 처방 서브그래프에 들어가기 전이다 (진짜 LangGraph의 규칙). 계획은 승인 카드(shown)에 있다
    assert (state["flow"], state["status"], state["order"], state["plan"]) == ("prescription", "received", None, None)
    snapshot = compiled.get_state(thread_config(f"P-{sample_id}"))
    assert [i.value for i in snapshot.interrupts] == [shown]


def test_the_state_while_stopped_has_every_field(backend):
    state = send(make(), "P0001", load("cv_a1", "rx_001"))
    assert set(state) - {"__interrupt__"} == STATE_KEYS


def test_the_order_shown_for_approval_is_the_order_of_the_steps_by_hand(backend):
    from agent.drug_match import match_order
    from agent.risk import check_order

    ocr = load("cv_a1", "rx_004")
    shown = paused(send(make(), "P0001", ocr))
    by_hand = match_order(real_structure("P0001", ocr), DICT, api=None)
    by_hand.warnings = [*by_hand.warnings, NO_API_WARNING] if not by_hand.all_matched else by_hand.warnings
    assert DispensingOrder.model_validate(shown["order"]) == check_order(by_hand, DICT)


def test_without_an_api_the_order_says_which_drugs_the_dictionary_could_not_find(backend):
    shown = paused(send(make(api=None), "P0001", load("cv_a1", "rx_005")))
    assert NO_API_WARNING in shown["order"]["warnings"]


def test_the_api_is_used_when_one_is_given(backend):
    asked = []

    def api(term):
        asked.append(term)
        return [ApiDrug(item_code="199900001", name="전혀다른약정", company="가상제약")]

    shown = paused(send(make(api=api), "P0001", load("cv_a1", "rx_005")))
    assert asked and NO_API_WARNING not in shown["order"]["warnings"]


@pytest.mark.parametrize("code", [E_OCR_EMPTY, E_LLM])
def test_a_structure_failure_ends_the_run_before_matching(backend, calls, code):
    def fails(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다.", code=code)

    final = send(make(structure=fails), "P0001", EMPTY_OCR)
    assert calls == ["router", "structure"] and paused(final) is None
    assert (final["status"], final["order"], final["error_code"], final["notice"]) == ("failed", None, code, "자동 구조화에 실패했습니다.")
    assert final["errors"] == [f"[structure] {code} 자동 구조화에 실패했습니다."]


@pytest.mark.parametrize("builder, run", [(build_graph, send), (build_prescription_graph, run_prescription)], ids=["top", "subgraph"])
def test_the_dictionary_that_is_given_is_the_one_used(backend, builder, run):
    only_gevorin = dictionary_from_data({"drugs": [{"item_code": "197900277", "name": "게보린정", "ingredients": ["아세트아미노펜"]}]})
    compiled = builder(dictionary=only_gevorin, structure=real_structure, api=None, checkpointer=memory_checkpointer())
    order = DispensingOrder.model_validate(paused(run(compiled, "P0001", load("cv_a1", "rx_004")))["order"])
    assert [d.item_code for d in order.items] == [None, "197900277", None, None]      # 타이레놀도 사전에 없으니 못 찾는다
    assert order.risk_check.level == "none"                                          # 같은 성분을 가진 짝이 사전에 없으니 중복도 못 본다


def test_an_ocr_without_text_fails_with_the_empty_ocr_code(backend, calls):
    final = send(make(), "P0001", EMPTY_OCR)
    assert calls == ["router", "structure"]
    assert final["status"] == "failed" and final["error_code"] == E_OCR_EMPTY and final["flow"] == "prescription"


def test_a_config_problem_comes_out_of_the_graph_as_it_is(backend):
    def broken(prescription_id, ocr):
        raise ConfigError("OPENAI_API_KEY가 없습니다.")

    with pytest.raises(ConfigError):
        send(make(structure=broken), "P0001", load("cv_a1", "rx_001"))


# ------------------------------------------------------------ 위험 신호 대조 (시연 조합 A · B)


def test_demo_combination_a_stops_at_the_approval_with_the_ingredient_duplicate(backend):
    shown = paused(send(make(), "P0001", load("cv_a1", "rx_004")))   # rx_004: 타이레놀 · 게보린 · 프론크논 · 알모란
    risk = DispensingOrder.model_validate(shown["order"]).risk_check
    [finding] = risk.findings
    assert (risk.scope, risk.level) == ("rx_only", "warning")
    assert (finding.type, finding.level, finding.item_codes) == ("ingredient_dup", "warning", ["202106092", "197900277"])
    assert "아세트아미노펜" in finding.message
    assert shown["notice"].endswith("위험 신호가 1건 있습니다(경고). 약사가 확인한 뒤에 승인할 수 있습니다.")


def test_demo_combination_b_stops_at_the_approval_with_the_ingredient_duplicate(backend):
    shown = paused(send(make(structure=structure_of("트리스펜연질캡슐", "아르펜정")), "P0001", load("cv_a1", "rx_001")))
    risk = DispensingOrder.model_validate(shown["order"]).risk_check
    [finding] = risk.findings
    assert (risk.level, finding.item_codes) == ("warning", ["201110120", "200201239"])
    assert "이부프로펜" in finding.message


def test_a_prescription_without_a_shared_ingredient_says_none(backend):
    shown = paused(send(make(structure=structure_of("프론크논캡슐", "뮤코다인캡슐200mg")), "P0001", load("cv_a1", "rx_001")))
    risk = DispensingOrder.model_validate(shown["order"]).risk_check
    assert (risk.scope, risk.level, risk.findings) == ("rx_only", "none", [])
    assert shown["notice"] == "약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다."


def test_a_drug_that_could_not_be_checked_is_told_to_the_pharmacist(backend):
    shown = paused(send(make(), "P0001", load("cv_a1", "rx_001")))   # rx_001: 타이레놀 · 알모란(대상 아님)
    assert f"알모란정: 품목기준코드를 정하지 못해 {UNCHECKED_TAIL}" in shown["order"]["warnings"]


# ------------------------------------------------------------ 승인 · 고치기 · 조제 계획 (HITL ①)


def test_approving_confirms_the_plan_shown_on_the_approval(backend, calls):
    compiled = make(structure=structure_of("타이레놀정500mg", "게보린정"))
    state = send(compiled, "P0001", load("cv_a1", "rx_001"))
    shown = paused(state)
    calls.clear()
    final = approve(compiled, "P0001", state)
    assert calls == ["approve"]                                          # 구조화 · 매칭 · 대조 · 계획은 다시 돌지 않는다
    assert paused(final) is None
    assert (final["status"], final["error_code"], final["errors"]) == ("planned", None, [])
    plan = Plan.model_validate(final["plan"])
    assert (plan.prescription_id, plan.days, len(plan.plan)) == ("P0001", 3, 3)
    assert final["plan"] == shown["plan"]                                # 승인 카드에 보인 계획을 그대로 확정
    assert final["notice"] == "약사가 승인했습니다."
    assert final["order"]["risk_check"]["level"] == "warning"           # 승인한 조제지시서가 상태에 남는다
    snapshot = compiled.get_state(thread_config("P0001"))
    assert snapshot.interrupts == () and snapshot.values["status"] == "planned"
    json.dumps(final)


def test_the_plan_is_the_plan_of_the_approved_order(backend):
    from agent.plan import build_plan

    compiled = make(structure=structure_of("트리스펜연질캡슐", "아르펜정"))
    final = approve(compiled, "P0001", send(compiled, "P0001", load("cv_a1", "rx_001")))
    assert final["plan"] == build_plan(DispensingOrder.model_validate(final["order"])).model_dump(mode="json")


def test_editing_runs_matching_and_checking_again_and_stops_again(backend, calls):
    compiled = make(structure=structure_of("타이레놀정500mg", "게보린정"))
    shown = paused(send(compiled, "P0001", load("cv_a1", "rx_001")))
    order = json.loads(json.dumps(shown["order"]))
    order["items"] = order["items"][:1]                                # 게보린을 지운다
    calls.clear()
    state = resume_run(compiled, "P0001", {"action": "edit", "order": order})
    assert calls == ["approve", "match", "risk", "make_plan", "approve"]   # 구조화(LLM)는 다시 부르지 않는다
    again = paused(state)
    assert again["hitl_point"] == "rx_approval"
    assert [i["name"] for i in again["order"]["items"]] == ["타이레놀정500mg"]
    assert again["order"]["risk_check"]["level"] == "none"              # 다시 대조했다: 중복이 사라졌다
    assert [c["items"] for c in again["plan"]["plan"]][0] == [{"item_code": "202106092", "count": 1}]   # 계획도 다시 만들었다
    final = approve(compiled, "P0001", state)
    assert final["status"] == "planned" and [c["items"] for c in final["plan"]["plan"]][0] == [{"item_code": "202106092", "count": 1}]


def test_a_plan_that_cannot_be_made_shows_the_reason_on_the_approval_and_only_edits_go_on(backend, calls):
    compiled = make(structure=structure_of("타이레놀정500mg", "게보린정", days=5))          # 트레이(3일)를 넘는다
    state = send(compiled, "P0001", load("cv_a1", "rx_001"))
    shown = paused(state)
    assert shown["plan"] is None and shown["error_code"] == E_PLAN_RANGE   # 승인 전에 알린다 (명세서 A.7)
    assert shown["notice"].startswith("조제 계획을 만들 수 없습니다. 투약 일수가 트레이(3일)를 넘습니다")
    assert "고친 뒤 다시 승인해 주세요. 약사 승인을 기다립니다." in shown["notice"]
      # 계획 없이는 승인할 수 없다. 잘못된 응답은 그래프에 넣기 전에 거른다 (api.check_reply → read_reply).
    # 그래프 안에서 예외가 나면 진짜 LangGraph는 그 응답을 멈춘 자리에 남겨, 다음 resume에도 같은 응답이 다시 쓰인다.
    from agent.nodes import read_reply
    with pytest.raises(ValueError, match="조제 계획을 만들지 못해 승인할 수 없습니다"):
        read_reply({"action": "approve", "order": shown["order"]}, DispensingOrder.model_validate(shown["order"]),
                   plan_ready=False)                                # 계획 없이는 승인할 수 없다
    order = json.loads(json.dumps(shown["order"]))
    for item in order["items"]:
        item["days"] = 3
    calls.clear()
    again = paused(resume_run(compiled, "P0001", {"action": "edit", "order": order}))
    assert calls == ["approve", "match", "risk", "make_plan", "approve"]
    assert again["plan"] is not None and again["error_code"] is None
    final = resume_run(compiled, "P0001", {"action": "approve", "order": again["order"]})
    assert (final["status"], final["error_code"]) == ("planned", None)
    assert final["errors"] == [f"[plan] {E_PLAN_RANGE} {shown['notice'].split(' 고친 뒤')[0]}"]   # 실패 기록은 남는다


def test_a_new_graph_with_the_same_checkpointer_resumes_where_it_stopped(backend, calls):
    """프로세스를 다시 켠 것처럼 그래프를 새로 만들어도, 같은 체크포인터면 같은 곳에서 이어 간다."""
    saver = memory_checkpointer()
    first = make(structure=structure_of("타이레놀정500mg", "게보린정"), checkpointer=saver)
    shown = paused(send(first, "P0001", load("cv_a1", "rx_001")))
    second = make(structure=lambda pid, ocr: pytest.fail("이어 갈 때 구조화를 다시 부르면 안 됩니다"), checkpointer=saver)
    assert [i.value for i in second.get_state(thread_config("P0001")).interrupts] == [shown]
    final = resume_run(second, "P0001", {"action": "approve", "order": shown["order"]})
    assert final["status"] == "planned"


def test_prescriptions_are_kept_apart_by_their_id(backend):
    compiled = make(structure=structure_of("타이레놀정500mg", "게보린정"))
    a = paused(send(compiled, "P-A", load("cv_a1", "rx_001")))
    paused(send(compiled, "P-B", load("cv_a1", "rx_001")))
    final = resume_run(compiled, "P-A", {"action": "approve", "order": a["order"]})
    assert final["prescription_id"] == "P-A" and final["status"] == "planned"
    assert compiled.get_state(thread_config("P-B")).interrupts                   # B는 그대로 멈춰 있다


def test_the_plan_is_made_before_the_approval(backend, calls):
    state = send(make(structure=structure_of("타이레놀정500mg", "게보린정")), "P0001", load("cv_a1", "rx_001"))
    assert calls[-2:] == ["make_plan", "approve"] and paused(state)["plan"] is not None


# ------------------------------------------------------------ 서브그래프만 단독으로


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_the_subgraph_alone_shows_the_same_approval_as_the_top_graph(backend, sample_id):
    ocr = load("cv_a1", sample_id)
    top = paused(send(make(), f"P-{sample_id}", ocr))
    alone = paused(run_prescription(make_sub(), f"P-{sample_id}", ocr))
    assert alone == top


@pytest.mark.parametrize("names", [("타이레놀정500mg", "게보린정"), ("트리스펜연질캡슐", "아르펜정")], ids=["A", "B"])
def test_the_subgraph_alone_gives_the_same_plan_as_the_top_graph(backend, names):
    results = []
    for build, run in ((make, send), (make_sub, run_prescription)):
        compiled = build(structure=structure_of(*names))
        final = approve(compiled, "P0001", run(compiled, "P0001", load("cv_a1", "rx_001")))
        results.append({k: final[k] for k in ("order", "plan", "status", "notice", "error_code", "errors")})
    assert results[0] == results[1]
    [finding] = results[0]["order"]["risk_check"]["findings"]
    assert finding["type"] == "ingredient_dup" and finding["level"] == "warning"


def test_the_subgraph_alone_does_not_pass_the_router(backend, calls):
    state = run_prescription(make_sub(), "P0001", load("cv_a1", "rx_004"))
    assert calls == ["structure", "match", "risk", "make_plan", "approve"]
    assert paused(state)["hitl_point"] == "rx_approval"
    assert state["flow"] is None                                        # 흐름을 정하는 일은 라우터의 몫이다


def test_the_subgraph_alone_stops_at_a_structure_failure_too(backend, calls):
    final = run_prescription(make_sub(), "P0001", EMPTY_OCR)
    assert calls == ["structure"]
    assert final["status"] == "failed" and final["error_code"] == E_OCR_EMPTY and len(final["errors"]) == 1


def test_a_failure_is_not_written_twice_when_the_subgraph_runs_inside_the_top_graph(backend):
    final = send(make(), "P0001", EMPTY_OCR)
    assert len(final["errors"]) == 1                                    # 서브그래프의 기록이 상위의 합치기 규칙에 두 번 쌓이지 않는다


def test_errors_that_were_already_in_the_state_are_kept_and_not_repeated(backend):
    start = {**first_state("P0001", EMPTY_OCR), "errors": ["먼저 쌓인 기록"]}
    final = make().invoke(start, thread_config("P0001"))
    assert final["errors"][0] == "먼저 쌓인 기록" and len(final["errors"]) == 2
    assert final["errors"][1].startswith("[structure] ")


ROUTES = [
    # photo_kind, ocr, 지난 노드, 흐름, 상태
    ("tray", None, ["router", "tray"], "tray", "noted"),         # 조제 계획이 없으면 검수하지 않고 안내만 (14단계)
]
NOT_RUN = [                                                          # 흐름을 돌리지 않고 안내만
    (None, None, "사진 종류 정보가 없습니다."),
    ("med_bag", None, "받을 수 없는 사진 종류입니다: med_bag"),
    ("receipt", None, "받을 수 없는 사진 종류입니다: receipt"),
    (None, {"doc_type": "med_bag", "lines": []}, "처방전이 아닙니다: med_bag"),
]


@pytest.mark.parametrize("photo_kind, ocr, path, flow, status", ROUTES)
def test_the_router_sends_each_kind_of_photo_to_its_flow(backend, calls, photo_kind, ocr, path, flow, status):
    final = send(make(structure=lambda pid, o: pytest.fail("처방 구조화가 불리면 안 됩니다")), "P0001", ocr, photo_kind)
    assert calls == path
    assert (final["flow"], final["status"], final["route"]) == (flow, status, [flow])
    assert final["order"] is None and final["errors"] == [] and paused(final) is None


@pytest.mark.parametrize("photo_kind, ocr, notice", NOT_RUN)
def test_photos_out_of_scope_only_leave_a_notice(backend, calls, photo_kind, ocr, notice):
    final = send(make(structure=lambda pid, o: pytest.fail("처방 구조화가 불리면 안 됩니다")), "P0001", ocr, photo_kind)
    assert calls == ["router"]
    assert (final["flow"], final["status"], final["route"]) == (None, "noted", [])
    assert notice in final["notice"] and final["notes"] == [final["notice"]] and paused(final) is None


@pytest.mark.parametrize("photo_kind, ocr", [("prescription", load("cv_a1", "rx_001")), ("med_bag", None), ("tray", None), (None, None)])
def test_the_router_runs_once_per_message(backend, calls, photo_kind, ocr):
    send(make(), "P0001", ocr, photo_kind)
    assert calls.count("router") == 1 and calls[0] == "router"


def test_the_router_does_not_run_again_when_the_approval_resumes(backend, calls):
    compiled = make(structure=structure_of("타이레놀정500mg", "게보린정"))
    state = send(compiled, "P0001", load("cv_a1", "rx_001"))
    approve(compiled, "P0001", state)
    assert calls.count("router") == 1


def test_run_message_is_the_same_as_invoking_with_the_first_state_and_the_thread(backend):
    ocr = load("cv_a1", "rx_003")
    by_helper = send(make(), "P0001", ocr)
    by_hand = make().invoke(first_state("P0001", ocr), thread_config("P0001"))
    assert paused(by_helper) == paused(by_hand)
    assert {k: v for k, v in by_helper.items() if k != "__interrupt__"} == {k: v for k, v in by_hand.items() if k != "__interrupt__"}


def test_run_prescription_starts_with_the_prescription_kind(backend):
    state = run_prescription(make_sub(), "P0001", load("cv_a1", "rx_003"))
    assert state["photo_kind"] == "prescription" and state["prescription_id"] == "P0001"


def test_the_thread_is_the_prescription_id():
    assert thread_config("P0042") == {"configurable": {"thread_id": "P0042"}}


# ------------------------------------------------------------ 서브그래프를 부르는 노드


class StubSubgraph:
    """invoke만 흉내 내는 서브그래프. 받은 상태를 기록하고 정해 둔 결과를 돌려준다."""

    def __init__(self, result):
        self.result = result
        self.seen = []

    def invoke(self, state):
        self.seen.append(state)
        return self.result


def test_the_subgraph_node_returns_only_the_fields_the_subgraph_changed():
    state = {"prescription_id": "P0001", "flow": "prescription", "status": "received", "notice": None, "order": None, "errors": []}
    sub = StubSubgraph({**state, "status": "awaiting_approval", "notice": "안내", "order": {"items": []}})
    update = _subgraph_step(sub)(state)
    assert update == {"status": "awaiting_approval", "notice": "안내", "order": {"items": []}}
    assert sub.seen == [state]


def test_the_subgraph_node_returns_only_the_new_errors():
    state = {"status": "received", "errors": ["먼저"]}
    update = _subgraph_step(StubSubgraph({"status": "failed", "errors": ["먼저", "새것 1", "새것 2"]}))(state)
    assert update == {"status": "failed", "errors": ["새것 1", "새것 2"]}


def test_the_subgraph_node_leaves_errors_out_when_there_is_nothing_new():
    state = {"status": "received", "errors": ["먼저"]}
    assert "errors" not in _subgraph_step(StubSubgraph({"status": "matched", "errors": ["먼저"]}))(state)
    assert _subgraph_step(StubSubgraph({"status": "matched"}))({"status": "received"}) == {"status": "matched"}


def test_the_subgraph_node_writes_a_field_the_state_did_not_have_yet():
    assert _subgraph_step(StubSubgraph({"notice": None}))({}) == {"notice": None}


def test_the_subgraph_node_writes_a_field_that_became_empty():
    update = _subgraph_step(StubSubgraph({"notice": None, "order": None}))({"notice": "지난 안내", "order": {"items": []}})
    assert update == {"notice": None, "order": None}


# ------------------------------------------------------------ build_graph가 도구를 고르는 방법


def test_the_real_structure_is_used_when_none_is_given(monkeypatch):
    fake_langgraph.install(monkeypatch)
    seen = []
    monkeypatch.setattr(graph, "default_structure", lambda pid, ocr: seen.append((pid, ocr)) or real_structure(pid, ocr))
    ocr = load("cv_a1", "rx_001")
    from agent.structure import agent_view

    send(build_graph(dictionary=DICT, api=None, checkpointer=memory_checkpointer()), "P0009", ocr)
    assert seen == [("P0009", agent_view(ocr))]                    # 그래프에는 표 밖 글자를 비운 OCR이 들어간다 (11/13)
    seen.clear()
    run_prescription(build_prescription_graph(dictionary=DICT, api=None, checkpointer=memory_checkpointer()), "P0010", ocr)
    assert seen == [("P0010", agent_view(ocr))]


def test_the_default_structure_uses_the_openai_path_as_its_fallback(monkeypatch):
    seen = {}

    def fake_structure(prescription_id, ocr, *, llm_fn):
        seen.update(id=prescription_id, ocr=ocr, llm_fn=llm_fn)
        return "order"

    monkeypatch.setattr(graph, "structure_prescription", fake_structure)
    assert graph.default_structure("P0001", {"a": 1}) == "order"
    assert seen == {"id": "P0001", "ocr": {"a": 1}, "llm_fn": graph.extract_llm}


@pytest.mark.parametrize("builder", [build_graph, build_prescription_graph], ids=["top", "subgraph"])
def test_the_api_comes_from_the_settings_unless_one_is_given(monkeypatch, builder):
    fake_langgraph.install(monkeypatch)
    run = send if builder is build_graph else run_prescription
    asked = []

    def from_settings(term):
        asked.append(term)
        return []

    monkeypatch.setattr(graph, "default_api", lambda: from_settings)
    run(builder(dictionary=DICT, structure=real_structure, checkpointer=memory_checkpointer()), "P0001", load("cv_a1", "rx_005"))
    assert asked

    asked.clear()                                                       # api=None이면 설정에 키가 있어도 부르지 않는다
    monkeypatch.setattr(graph, "default_api", lambda: pytest.fail("api=None이면 default_api를 부르면 안 됩니다"))
    run(builder(dictionary=DICT, structure=real_structure, api=None, checkpointer=memory_checkpointer()), "P0001", load("cv_a1", "rx_005"))
    assert not asked


@pytest.mark.parametrize("builder", [build_graph, build_prescription_graph], ids=["top", "subgraph"])
def test_a_broken_dictionary_stops_the_graph_before_any_prescription_arrives(monkeypatch, tmp_path, builder):
    fake_langgraph.install(monkeypatch)
    monkeypatch.setattr("agent.config.DRUG_DICTIONARY_PATH", tmp_path / "missing.json")
    with pytest.raises(ConfigError):
        builder(structure=real_structure, api=None, checkpointer=None)


def test_the_top_graph_reads_the_dictionary_and_the_api_settings_only_once(monkeypatch):
    """상위 그래프와 그 안의 서브그래프가 같은 도구를 나눠 쓴다 (사전 파일을 두 번 읽지 않는다)."""
    fake_langgraph.install(monkeypatch)
    counts = {"dictionary": 0, "api": 0}

    def load_once():
        counts["dictionary"] += 1
        return DICT

    def api_once():
        counts["api"] += 1
        return None

    def checkpointer_once():
        counts["checkpointer"] = counts.get("checkpointer", 0) + 1
        return memory_checkpointer()

    monkeypatch.setattr(graph, "load_dictionary", load_once)
    monkeypatch.setattr(graph, "default_api", api_once)
    monkeypatch.setattr(graph, "default_checkpointer", checkpointer_once)
    build_graph(structure=real_structure)
    assert counts == {"dictionary": 1, "api": 1, "checkpointer": 1}       # 서브그래프는 상위의 체크포인터를 이어받는다


def test_the_subgraph_inside_the_top_graph_shares_its_dictionary(monkeypatch):
    fake_langgraph.install(monkeypatch)
    only_gevorin = dictionary_from_data({"drugs": [{"item_code": "197900277", "name": "게보린정"}]})
    monkeypatch.setattr(graph, "load_dictionary", lambda: only_gevorin)
    state = send(build_graph(structure=real_structure, api=None, checkpointer=memory_checkpointer()), "P0001", load("cv_a1", "rx_004"))
    assert [d["item_code"] for d in paused(state)["order"]["items"]] == [None, "197900277", None, None]


@pytest.mark.parametrize("builder", [build_graph, build_prescription_graph], ids=["top", "subgraph"])
def test_the_checkpointer_comes_from_the_settings_unless_one_is_given(monkeypatch, builder):
    fake_langgraph.install(monkeypatch)
    made = []
    monkeypatch.setattr(graph, "default_checkpointer", lambda: made.append("default") or memory_checkpointer())
    builder(dictionary=DICT, structure=real_structure, api=None)
    assert made == ["default"]
    given = memory_checkpointer()
    assert builder(dictionary=DICT, structure=real_structure, api=None, checkpointer=given).checkpointer is given
    assert made == ["default"]


def test_the_subgraph_inside_the_top_graph_has_no_checkpointer_of_its_own(monkeypatch):
    fake_langgraph.install(monkeypatch)
    seen = []
    real_compile = graph._compile

    def spy(chart, make_node, checkpointer=None):
        seen.append((chart.entry, checkpointer))
        return real_compile(chart, make_node, checkpointer)

    monkeypatch.setattr(graph, "_compile", spy)
    saver = memory_checkpointer()
    build_graph(dictionary=DICT, structure=real_structure, api=None, checkpointer=saver)
    assert seen == [("structure", None), ("router", saver)]


def lg(target):
    return fake_langgraph.END if target == END else target


def test_build_graph_puts_the_design_tables_into_langgraph_as_they_are(monkeypatch):
    fake_langgraph.install(monkeypatch)
    builder = make().graph
    assert set(builder.nodes) == set(TOP_CHART.nodes)
    assert set(builder.edges) == {(fake_langgraph.START, TOP_CHART.entry), *((s, lg(t)) for s, t in TOP_CHART.edges)}
    assert set(builder.branches) == set(TOP_CHART.branches)
    for source, (choose, targets) in TOP_CHART.branches.items():
        got_choose, got_map = builder.branches[source]
        assert got_choose is choose
        assert got_map == {name: lg(t) for name, t in targets.items()}


def test_build_prescription_graph_puts_the_design_table_into_langgraph_as_it_is(monkeypatch):
    fake_langgraph.install(monkeypatch)
    builder = make_sub().graph
    assert set(builder.nodes) == set(PRESCRIPTION_CHART.nodes)
    assert set(builder.edges) == {(fake_langgraph.START, PRESCRIPTION_CHART.entry), *((s, lg(t)) for s, t in PRESCRIPTION_CHART.edges)}
    assert set(builder.branches) == set(PRESCRIPTION_CHART.branches)
    for source, (choose, targets) in PRESCRIPTION_CHART.branches.items():
        got_choose, got_map = builder.branches[source]
        assert got_choose is choose
        assert got_map == {name: lg(t) for name, t in targets.items()}


# ------------------------------------------------------------ DUR 조회를 붙이는 규칙 (13단계)
def test_dur_follows_the_api_setting(monkeypatch):
    from agent.dur_api import DurApi
    from agent.graph import _make_deps

    structure = lambda pid, ocr: None  # noqa: E731
    assert _make_deps(DICT, structure, None).dur is None                       # api=None (시험 · --no-api): DUR도 쓰지 않는다
    monkeypatch.delenv("DRUG_API_KEY", raising=False)
    from agent.graph import _FROM_SETTINGS
    assert _make_deps(DICT, structure, _FROM_SETTINGS).dur is None            # 키가 없으면 없다
    monkeypatch.setenv("DRUG_API_KEY", "k")
    assert isinstance(_make_deps(DICT, structure, _FROM_SETTINGS).dur, DurApi)
    assert _make_deps(DICT, structure, None).dur is None                       # 키가 있어도 api=None이면 쓰지 않는다
    given = object()
    assert _make_deps(DICT, structure, None, None, given).dur is given          # 직접 주면 그것을 쓴다


def test_notes_from_the_prescription_subgraph_are_not_doubled(monkeypatch):
    """승인 노드가 남긴 진행 사유(notes)가 상위 그래프에 한 번만 쌓인다."""
    import fake_langgraph
    from agent.checkpoint import memory_checkpointer
    from agent.graph import build_graph, resume_run
    from agent.state import initial_state

    fake_langgraph.install(monkeypatch)
    graph = build_graph(dictionary=DICT, structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=None), api=None,
                        checkpointer=memory_checkpointer())
    shown = send(graph, "P1", load("rx8_a1", "rx8_01"))["__interrupt__"][0].value
    final = resume_run(graph, "P1", {"action": "manual", "order": shown["order"], "reason": "확인함"})
    assert final["notes"].count("위험 신호 확인 후 진행 사유: 확인함") == 1
    assert final["status"] == "planned"


def test_the_subgraph_step_returns_only_the_new_notes_and_errors():
    from agent.graph import _subgraph_step

    class Sub:
        def invoke(self, state):
            return {**state, "notes": state["notes"] + ["새 안내"], "errors": state["errors"] + ["새 오류"], "status": "approved"}

    state = {"notes": ["예전 안내"], "errors": ["예전 오류"], "status": "checked"}
    assert _subgraph_step(Sub())(state) == {"notes": ["새 안내"], "errors": ["새 오류"], "status": "approved"}


# ------------------------------------------------------------ 같은 대화의 다음 메시지 · 상담 대화 (11/12 요청서)


def test_the_next_message_in_the_same_conversation_keeps_the_approved_order_and_the_plan(backend):
    compiled = make(structure=structure_of("타이레놀정500mg", "게보린정"))
    planned = approve(compiled, "P0001", send(compiled, "P0001", load("cv_a1", "rx_001")))
    assert planned["status"] == "planned" and planned["plan"] is not None
    tray = run_message(compiled, "P0001", {"kind": "tray", "image_id": "img2"})
    assert (tray["order"], tray["plan"], tray["status"]) == (planned["order"], planned["plan"], "planned")
    assert (tray["route"], tray["image_id"], tray["prescription_id"]) == (["tray"], "img2", "P0001")
    assert compiled.get_state(thread_config("P0001")).values["order"] == planned["order"]


def test_the_first_message_in_a_conversation_starts_from_the_first_state(backend):
    state = run_message(make(), "C0001", {"image_id": "img1"})
    assert (state["conversation_id"], state["prescription_id"], state["image_id"]) == ("C0001", None, "img1")
    assert (state["route"], state["flow"]) == ([], None)
    assert state["order"] is None and state["status"] == "noted"              # 처방전 사진이 와야 시작한다
    assert "처방전 사진" in state["notice"]


def test_a_prescription_photo_opens_the_prescription_even_when_the_web_did_not_give_its_id(backend):
    compiled = make(structure=structure_of("타이레놀정500mg"))
    state = run_message(compiled, "C0002", {"kind": "prescription"}, load("cv_a1", "rx_001"))
    assert paused(state)["order"]["prescription_id"] == "C0002"


