"""상위 그래프 (agent/graph.py) — 슈퍼바이저의 규칙 분기와 처방 흐름의 뼈대.

시험은 세 겹이다.
1. 화살표 표(NODES · EDGES · BRANCHES)만 보는 시험. LangGraph가 필요 없다.
2. 'fake' 쪽: 시험용 가짜 LangGraph(fake_langgraph.py)에 build_graph를 끼워 끝까지 돌린다. LangGraph가 없어도 돈다.
3. 'real' 쪽: 같은 시험을 진짜 LangGraph로 돌린다. langgraph가 설치되어 있을 때만 돌고, 없으면 건너뛴다.
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
from agent.errors import E_LLM, E_OCR_EMPTY, ConfigError, StructureError
from agent.graph import (
    BRANCHES,
    EDGES,
    END,
    ENTRY,
    NODES,
    after_structure,
    after_supervisor,
    build_graph,
    run_message,
    successors,
)
from agent.nodes import NO_API_WARNING
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder
from agent.state import FLOWS, GraphState, initial_state

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = RESOURCES_DIR / "samples"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
STATE_KEYS = set(typing.get_type_hints(GraphState))
EMPTY_OCR = {"image_id": "x", "doc_type": "prescription", "width": 10, "height": 10, "lines": []}


def load(kind, sample_id):
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback):
    raise AssertionError("이 시험에서는 LLM을 부르면 안 됩니다")


def real_structure(prescription_id, ocr):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


# ------------------------------------------------------------ 1. 화살표 표


def test_node_names_do_not_collide_with_state_fields():
    assert set(NODES).isdisjoint(STATE_KEYS)           # LangGraph는 상태 칸과 같은 이름의 노드를 거절한다


def test_every_arrow_starts_and_ends_at_a_known_place():
    known = set(NODES)
    assert ENTRY in known
    for source, target in EDGES:
        assert source in known and target in known | {END}
    for source, (choose, targets) in BRANCHES.items():
        assert source in known and callable(choose)
        assert set(targets.values()) <= known | {END}
    assert not [s for s, _ in EDGES if s in BRANCHES], "고정 화살표와 갈림길이 한 노드에 함께 있으면 안 된다"
    assert len({s for s, _ in EDGES}) == len(EDGES), "한 노드에서 나가는 고정 화살표는 하나"


def test_every_node_has_a_way_out():
    assert all(successors()[name] for name in NODES)


def test_every_node_can_be_reached_from_the_entry():
    reached, todo = set(), [ENTRY]
    while todo:
        node = todo.pop()
        if node in reached or node == END:
            continue
        reached.add(node)
        todo.extend(successors()[node])
    assert reached == set(NODES)


def test_every_node_leads_to_the_end():
    leads_to_end = {name for name, nxt in successors().items() if END in nxt}
    changed = True
    while changed:
        changed = False
        for name, nxt in successors().items():
            if name not in leads_to_end and nxt & leads_to_end:
                leads_to_end.add(name)
                changed = True
    assert leads_to_end == set(NODES)


def test_the_graph_has_no_way_back_so_the_supervisor_runs_once_per_message():
    nxt = successors()
    assert not any(ENTRY in targets for targets in nxt.values())      # 슈퍼바이저로 돌아오는 화살표가 없다

    state: dict[str, int] = {}                                         # 0 = 지나는 중, 1 = 끝남

    def visit(node):
        if node == END or state.get(node) == 1:
            return
        assert state.get(node) != 0, f"{node}로 돌아오는 길이 있습니다"
        state[node] = 0
        for target in nxt[node]:
            visit(target)
        state[node] = 1

    visit(ENTRY)


def test_every_flow_has_a_first_node():
    choose, targets = BRANCHES[ENTRY]
    assert set(targets) == set(FLOWS)
    assert targets == {"prescription": "structure", "recheck": "recheck", "tray": "tray", "ask_type": "ask_type"}


def test_the_prescription_flow_goes_structure_match_approve():
    assert BRANCHES["structure"][1] == {"match": "match", "end": END}
    assert ("match", "approve") in EDGES and ("approve", END) in EDGES


@pytest.mark.parametrize("flow, branch, node", [
    ("prescription", "prescription", "structure"), ("recheck", "recheck", "recheck"),
    ("tray", "tray", "tray"), ("ask_type", "ask_type", "ask_type"),
    (None, "ask_type", "ask_type"), ("", "ask_type", "ask_type"), ("unknown", "ask_type", "ask_type"),
    (7, "ask_type", "ask_type"), (["tray"], "ask_type", "ask_type"),     # 알 수 없는 흐름은 사진 종류를 되묻는다
])
def test_after_the_supervisor(flow, branch, node):
    choose, targets = BRANCHES[ENTRY]
    assert after_supervisor({"flow": flow}) == choose({"flow": flow}) == branch
    assert targets[branch] == node


def test_after_the_supervisor_without_a_flow_asks_the_kind():
    assert after_supervisor({}) == "ask_type"


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
    assert step in BRANCHES["structure"][1]


def test_importing_the_graph_does_not_import_langgraph():
    code = "import sys, agent.graph; sys.exit(1 if 'langgraph' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")


# ------------------------------------------------------------ 2·3. 끝까지 돌리기 (fake · real)


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


@pytest.fixture
def calls(monkeypatch):
    """노드가 불린 순서를 기록한다."""
    log: list[str] = []

    def recording(name, fn):
        def wrapper(state, deps):
            log.append(name)
            return fn(state, deps)
        return wrapper

    for name, fn in list(NODES.items()):
        monkeypatch.setitem(NODES, name, recording(name, fn))
    return log


def make(*, structure=real_structure, api=None):
    return build_graph(dictionary=DICT, structure=structure, api=api)


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_a_sample_prescription_goes_all_the_way_to_the_approval_place(backend, calls, sample_id):
    final = run_message(make(), f"P-{sample_id}", load("cv_a1", sample_id))
    assert calls == ["supervisor", "structure", "match", "approve"]
    assert (final["flow"], final["status"], final["error_code"], final["errors"]) == ("prescription", "awaiting_approval", None, [])
    order = DispensingOrder.model_validate(final["order"])
    assert order.prescription_id == f"P-{sample_id}"
    assert len(order.items) == len(load("cv_expected", sample_id)["items"])
    # 가상 처방전의 인쇄 코드(9000000xx)는 품목기준코드가 아니라서, 모든 줄이 이름 단계로 가고 약사 확인이 붙는다
    assert len(order.flagged_indices) == len(order.items)
    assert final["notice"] == f"약사 승인을 기다립니다. 확인이 필요한 약품이 {len(order.items)}줄 있습니다."
    json.dumps(final)                                                   # 체크포인터가 저장할 수 있는 값만 있다


def test_the_final_state_has_every_field(backend):
    final = run_message(make(), "P0001", load("cv_a1", "rx_001"))
    assert set(final) == STATE_KEYS


def test_the_graph_gives_the_same_order_as_running_the_nodes_by_hand(backend):
    from agent.drug_match import match_order

    ocr = load("cv_a1", "rx_004")
    final = run_message(make(), "P0001", ocr)
    by_hand = match_order(real_structure("P0001", ocr), DICT, api=None)
    by_hand.warnings = [*by_hand.warnings, NO_API_WARNING] if not by_hand.all_matched else by_hand.warnings
    assert DispensingOrder.model_validate(final["order"]) == by_hand


def test_without_an_api_the_order_says_which_drugs_the_dictionary_could_not_find(backend):
    final = run_message(make(api=None), "P0001", load("cv_a1", "rx_005"))
    assert NO_API_WARNING in final["order"]["warnings"]


def test_the_api_is_used_when_one_is_given(backend):
    asked = []

    def api(term):
        asked.append(term)
        return [ApiDrug(item_code="199900001", name="전혀다른약정", company="가상제약")]

    final = run_message(make(api=api), "P0001", load("cv_a1", "rx_005"))
    assert asked and NO_API_WARNING not in final["order"]["warnings"]


@pytest.mark.parametrize("code", [E_OCR_EMPTY, E_LLM])
def test_a_structure_failure_ends_the_run_before_matching(backend, calls, code):
    def fails(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다.", code=code)

    final = run_message(make(structure=fails), "P0001", EMPTY_OCR)
    assert calls == ["supervisor", "structure"]
    assert (final["status"], final["order"], final["error_code"], final["notice"]) == ("failed", None, code, "자동 구조화에 실패했습니다.")
    assert final["errors"] == [f"[structure] {code} 자동 구조화에 실패했습니다."]


def test_the_dictionary_that_is_given_is_the_one_used(backend):
    only_gevorin = dictionary_from_data({"drugs": [{"item_code": "197900277", "name": "게보린정"}]})
    final = run_message(build_graph(dictionary=only_gevorin, structure=real_structure, api=None), "P0001", load("cv_a1", "rx_004"))
    order = DispensingOrder.model_validate(final["order"])
    assert [d.item_code for d in order.items] == [None, "197900277", None, None]      # 타이레놀도 사전에 없으니 못 찾는다


def test_an_ocr_without_text_fails_with_the_empty_ocr_code(backend, calls):
    final = run_message(make(), "P0001", EMPTY_OCR)
    assert calls == ["supervisor", "structure"]
    assert final["status"] == "failed" and final["error_code"] == E_OCR_EMPTY and final["flow"] == "prescription"


def test_a_config_problem_comes_out_of_the_graph_as_it_is(backend):
    def broken(prescription_id, ocr):
        raise ConfigError("OPENAI_API_KEY가 없습니다.")

    with pytest.raises(ConfigError):
        run_message(make(structure=broken), "P0001", load("cv_a1", "rx_001"))


ROUTES = [
    # photo_kind, ocr, 지난 노드, 흐름, 상태
    ("med_bag", None, ["supervisor", "recheck"], "recheck", "not_ready"),
    (None, {"doc_type": "med_bag", "lines": []}, ["supervisor", "recheck"], "recheck", "not_ready"),
    ("tray", None, ["supervisor", "tray"], "tray", "not_ready"),
    (None, None, ["supervisor", "ask_type"], "ask_type", "awaiting_type"),
    ("receipt", None, ["supervisor", "ask_type"], "ask_type", "awaiting_type"),
    ("prescription", {"doc_type": "med_bag", "lines": []}, ["supervisor", "ask_type"], "ask_type", "awaiting_type"),
]


@pytest.mark.parametrize("photo_kind, ocr, path, flow, status", ROUTES)
def test_the_supervisor_sends_each_kind_of_photo_to_its_flow(backend, calls, photo_kind, ocr, path, flow, status):
    final = run_message(make(structure=lambda pid, o: pytest.fail("처방 구조화가 불리면 안 됩니다")), "P0001", ocr, photo_kind)
    assert calls == path
    assert (final["flow"], final["status"]) == (flow, status)
    assert final["order"] is None and final["errors"] == []


def test_a_photo_kind_that_is_given_wins_over_nothing_but_not_over_a_disagreeing_ocr(backend, calls):
    ocr = load("cv_a1", "rx_001")                                       # 이 OCR은 doc_type이 처방전이다
    agree = run_message(make(), "P0001", ocr, "prescription")
    assert agree["status"] == "awaiting_approval"
    calls.clear()
    clash = run_message(make(), "P0001", ocr, "med_bag")
    assert calls == ["supervisor", "ask_type"] and clash["status"] == "awaiting_type"
    assert "지정한 종류(약 포장)와 OCR이 읽은 종류(처방전)가 다릅니다." in clash["notice"]


@pytest.mark.parametrize("photo_kind, ocr", [("prescription", load("cv_a1", "rx_001")), ("med_bag", None), ("tray", None), (None, None)])
def test_the_supervisor_runs_once_per_message(backend, calls, photo_kind, ocr):
    run_message(make(), "P0001", ocr, photo_kind)
    assert calls.count("supervisor") == 1 and calls[0] == "supervisor"


def test_run_message_is_the_same_as_invoking_with_the_first_state(backend):
    compiled = make()
    ocr = load("cv_a1", "rx_003")
    assert run_message(compiled, "P0001", ocr) == compiled.invoke(initial_state("P0001", ocr))


def test_two_messages_one_after_another_do_not_share_state(backend):
    compiled = make()
    first = run_message(compiled, "P0001", load("cv_a1", "rx_001"))
    second = run_message(compiled, "P0002", EMPTY_OCR)
    assert first["status"] == "awaiting_approval" and first["errors"] == []
    assert second["status"] == "failed" and len(second["errors"]) == 1
    assert second["order"] is None and second["prescription_id"] == "P0002"


# ------------------------------------------------------------ build_graph가 도구를 고르는 방법


def test_the_real_structure_is_used_when_none_is_given(monkeypatch):
    fake_langgraph.install(monkeypatch)
    seen = []
    monkeypatch.setattr(graph, "default_structure", lambda pid, ocr: seen.append((pid, ocr)) or real_structure(pid, ocr))
    ocr = load("cv_a1", "rx_001")
    run_message(build_graph(dictionary=DICT, api=None), "P0009", ocr)
    assert seen == [("P0009", ocr)]


def test_the_default_structure_uses_the_openai_path_as_its_fallback(monkeypatch):
    seen = {}

    def fake_structure(prescription_id, ocr, *, llm_fn):
        seen.update(id=prescription_id, ocr=ocr, llm_fn=llm_fn)
        return "order"

    monkeypatch.setattr(graph, "structure_prescription", fake_structure)
    assert graph.default_structure("P0001", {"a": 1}) == "order"
    assert seen == {"id": "P0001", "ocr": {"a": 1}, "llm_fn": graph.extract_llm}


def test_the_api_comes_from_the_settings_unless_one_is_given(monkeypatch):
    fake_langgraph.install(monkeypatch)
    asked = []

    def from_settings(term):
        asked.append(term)
        return []

    monkeypatch.setattr(graph, "default_api", lambda: from_settings)
    run_message(build_graph(dictionary=DICT, structure=real_structure), "P0001", load("cv_a1", "rx_005"))
    assert asked

    asked.clear()                                                       # api=None이면 설정에 키가 있어도 부르지 않는다
    monkeypatch.setattr(graph, "default_api", lambda: pytest.fail("api=None이면 default_api를 부르면 안 됩니다"))
    run_message(build_graph(dictionary=DICT, structure=real_structure, api=None), "P0001", load("cv_a1", "rx_005"))
    assert not asked


def test_a_broken_dictionary_stops_the_graph_before_any_prescription_arrives(monkeypatch, tmp_path):
    fake_langgraph.install(monkeypatch)
    monkeypatch.setattr("agent.config.DRUG_DICTIONARY_PATH", tmp_path / "missing.json")
    with pytest.raises(ConfigError):
        build_graph(structure=real_structure, api=None)


def test_build_graph_puts_the_arrow_table_into_langgraph_as_it_is(monkeypatch):
    fake_langgraph.install(monkeypatch)
    compiled = make()
    builder = compiled.graph
    assert set(builder.nodes) == set(NODES)

    def lg(target):
        return fake_langgraph.END if target == END else target

    assert set(builder.edges) == {(fake_langgraph.START, ENTRY), *((s, lg(t)) for s, t in EDGES)}
    assert set(builder.branches) == set(BRANCHES)
    for source, (choose, targets) in BRANCHES.items():
        got_choose, got_map = builder.branches[source]
        assert got_choose is choose
        assert got_map == {name: lg(t) for name, t in targets.items()}
