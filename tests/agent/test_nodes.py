"""그래프의 노드 (agent/nodes.py). LangGraph 없이 함수로 불러 시험한다."""
import copy
import dataclasses
import json
import typing

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_api import ApiDrug
from agent.drug_dictionary import load_dictionary
from agent.errors import E_LLM, E_OCR_EMPTY, ConfigError, StructureError
from agent.nodes import (
    ASK_TYPE_NOTICE,
    NO_API_WARNING,
    NOT_READY_NOTICES,
    Deps,
    approve_node,
    ask_type_node,
    decide_flow,
    match_node,
    recheck_node,
    structure_node,
    supervisor_node,
    tray_node,
)
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder, DrugLine
from agent.state import GraphState, initial_state

SAMPLES = RESOURCES_DIR / "samples"
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL = "202106092"


def load(kind, sample_id):
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback):
    raise AssertionError("이 시험에서는 LLM을 부르면 안 됩니다")


def real_structure(prescription_id, ocr):
    """실제 구조화에서 LLM만 막아 둔 것. 샘플은 규칙 경로로 읽히므로 LLM은 불리지 않는다."""
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


def deps(*, structure=real_structure, api=None):
    return Deps(dictionary=DICT, structure=structure, api=api)


def merged(state, update):
    """LangGraph가 하는 대로 노드가 돌려준 칸을 상태에 합친다 (errors는 쌓고, 나머지는 덮어쓴다)."""
    hints = typing.get_type_hints(GraphState, include_extras=True)
    out = dict(state)
    for key, value in update.items():
        assert key in hints, f"상태에 없는 칸: {key}"
        reducer = getattr(hints[key], "__metadata__", (None,))[0]
        out[key] = reducer(out[key], value) if reducer else value
    return out


def line(name, printed_code=None):
    return DrugLine(item_code=None, name=name, printed_code=printed_code, count_per_dose=1, times=["morning"], days=3)


def order_of(*lines):
    return DispensingOrder(prescription_id="P0001", items=list(lines)).model_dump(mode="json")


def state_with(order=None, **extra):
    return {**initial_state("P0001"), "order": order, **extra}


# ------------------------------------------------------------ 슈퍼바이저: 사진 종류로 흐름 정하기


@pytest.mark.parametrize("photo_kind, ocr, flow", [
    ("prescription", None, "prescription"),
    ("med_bag", None, "recheck"),
    ("tray", None, "tray"),
    (None, {"doc_type": "prescription"}, "prescription"),
    (None, {"doc_type": "med_bag"}, "recheck"),
    ("prescription", {"doc_type": "prescription"}, "prescription"),
    ("med_bag", {"doc_type": "med_bag"}, "recheck"),
    ("tray", {}, "tray"),                                       # 트레이에는 문서 종류가 없다
    ("prescription", {"lines": []}, "prescription"),            # doc_type이 없어도 사진 종류가 있으면 따른다
])
def test_the_flow_follows_the_photo_kind(photo_kind, ocr, flow):
    assert decide_flow(photo_kind, ocr) == (flow, None)


@pytest.mark.parametrize("photo_kind, ocr, reason", [
    (None, None, "사진 종류 정보가 없습니다."),
    (None, {}, "사진 종류 정보가 없습니다."),
    (None, "글자", "사진 종류 정보가 없습니다."),                  # OCR 형식이 아니면 단서로 보지 않는다
    ("receipt", None, "알 수 없는 사진 종류입니다."),
    ("", None, "알 수 없는 사진 종류입니다."),
    (["prescription"], None, "알 수 없는 사진 종류입니다."),
    ("receipt", {"doc_type": "prescription"}, "알 수 없는 사진 종류입니다."),
    (None, {"doc_type": "receipt"}, "OCR이 읽은 문서 종류를 알 수 없습니다."),
    ("prescription", {"doc_type": "receipt"}, "OCR이 읽은 문서 종류를 알 수 없습니다."),
    ("prescription", {"doc_type": "med_bag"}, "지정한 종류(처방전)와 OCR이 읽은 종류(약 포장)가 다릅니다."),
    ("med_bag", {"doc_type": "prescription"}, "지정한 종류(약 포장)와 OCR이 읽은 종류(처방전)가 다릅니다."),
    ("tray", {"doc_type": "prescription"}, "지정한 종류(트레이)와 OCR이 읽은 종류(처방전)가 다릅니다."),
])
def test_an_unclear_photo_kind_is_asked_about_not_guessed(photo_kind, ocr, reason):
    flow, notice = decide_flow(photo_kind, ocr)
    assert flow == "ask_type"
    assert notice == f"사진 종류를 정하지 못했습니다. {reason} 처방전 · 약 포장 · 트레이 중에서 골라 주세요."


def test_supervisor_sets_only_the_flow_when_the_kind_is_clear():
    state = initial_state("P0001", load("cv_a1", "rx_001"))
    assert supervisor_node(state, deps()) == {"flow": "prescription"}


def test_supervisor_also_leaves_the_question_when_it_asks():
    update = supervisor_node(initial_state("P0001"), deps())
    assert update["flow"] == "ask_type"
    assert update["notice"].startswith("사진 종류를 정하지 못했습니다.")
    assert set(update) == {"flow", "notice"}


def test_supervisor_reads_the_kind_from_the_ocr_of_a_state_without_photo_kind():
    state = {"prescription_id": "P0001", "ocr": {"doc_type": "med_bag"}}      # 칸이 몇 개 없어도 읽는다
    assert supervisor_node(state, deps()) == {"flow": "recheck"}


# ------------------------------------------------------------ 구조화


def test_structure_makes_the_order_from_a_sample():
    state = initial_state("P0001", load("cv_a1", "rx_001"))
    update = structure_node(state, deps())
    assert set(update) == {"order", "status", "notice"}
    assert update["status"] == "structured" and update["notice"] is None
    order = DispensingOrder.model_validate(update["order"])
    assert order.prescription_id == "P0001" and len(order.items) == 2
    json.dumps(update)                                              # 체크포인터가 저장할 수 있는 값이다


def test_structure_hands_the_id_and_the_ocr_to_the_structure_function():
    seen = []

    def spy(prescription_id, ocr):
        seen.append((prescription_id, ocr))
        return DispensingOrder.model_validate(order_of(line("게보린정")))

    ocr = load("cv_a1", "rx_002")
    structure_node(initial_state("P0007", ocr), deps(structure=spy))
    assert seen == [("P0007", ocr)]


@pytest.mark.parametrize("code", [E_OCR_EMPTY, E_LLM, None])
def test_a_structure_failure_is_recorded_in_the_state(code):
    def fails(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다.", code=code)

    update = structure_node(initial_state("P0001", {}), deps(structure=fails))
    assert update == {
        "order": None, "status": "failed", "notice": "자동 구조화에 실패했습니다.", "error_code": code,
        "errors": [f"[structure] {code or '-'} 자동 구조화에 실패했습니다."],
    }


def test_a_state_without_ocr_fails_with_the_empty_ocr_code():
    update = structure_node(initial_state("P0001", None), deps())
    assert update["status"] == "failed" and update["error_code"] == E_OCR_EMPTY and update["order"] is None


def test_a_config_problem_is_not_turned_into_a_failure_of_one_prescription():
    def broken(prescription_id, ocr):
        raise ConfigError("OPENAI_API_KEY가 없습니다.")

    with pytest.raises(ConfigError):
        structure_node(initial_state("P0001", {}), deps(structure=broken))


# ------------------------------------------------------------ 약품 매칭


def structured(sample_id):
    return merged(initial_state("P0001", load("cv_a1", sample_id)), structure_node(initial_state("P0001", load("cv_a1", sample_id)), deps()))


def test_match_finds_the_item_codes_and_marks_the_lines_to_check():
    update = match_node(structured("rx_001"), deps())
    assert set(update) == {"order", "status"} and update["status"] == "matched"
    order = DispensingOrder.model_validate(update["order"])
    assert order.items[0].item_code == TYLENOL                       # 타이레놀은 대상 8종이라 이름으로 찾는다
    assert order.items[0].needs_pharmacist_check                      # 이름으로 정한 줄은 언제나 약사 확인
    json.dumps(update)


def test_without_the_api_the_order_says_so_when_a_drug_is_not_in_the_dictionary():
    update = match_node(structured("rx_005"), deps(api=None))        # rx_005는 대상 8종이 아닌 약뿐이다
    order = DispensingOrder.model_validate(update["order"])
    assert not order.all_matched
    assert order.warnings.count(NO_API_WARNING) == 1


def test_the_no_api_warning_is_not_repeated_when_match_runs_again():
    first = match_node(structured("rx_005"), deps())
    second = match_node({**structured("rx_005"), "order": first["order"]}, deps())
    assert DispensingOrder.model_validate(second["order"]).warnings.count(NO_API_WARNING) == 1


def test_no_warning_when_every_drug_was_found_in_the_dictionary():
    state = state_with(order_of(line("게보린정"), line("타이레놀정500mg")))
    order = DispensingOrder.model_validate(match_node(state, deps(api=None))["order"])
    assert order.all_matched and NO_API_WARNING not in order.warnings


def test_no_warning_when_an_api_was_given():
    asked = []

    def api(term):
        asked.append(term)
        return [ApiDrug(item_code="199900001", name="전혀다른약정", company="가상제약")]

    state = state_with(order_of(line("가나다정")))
    order = DispensingOrder.model_validate(match_node(state, deps(api=api))["order"])
    assert asked and NO_API_WARNING not in order.warnings


def test_a_failing_api_does_not_stop_the_match():
    from agent.errors import DrugApiError

    def api(term):
        raise DrugApiError("연결할 수 없습니다.")

    state = state_with(order_of(line("가나다정")))
    update = match_node(state, deps(api=api))
    order = DispensingOrder.model_validate(update["order"])
    assert update["status"] == "matched" and order.items[0].needs_pharmacist_check


# ------------------------------------------------------------ 승인 자리


def test_approve_says_how_many_lines_need_a_check():
    state = merged(structured("rx_001"), match_node(structured("rx_001"), deps()))
    update = approve_node(state, deps())
    assert update == {"status": "awaiting_approval", "notice": "약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다."}


def test_approve_has_nothing_to_add_when_no_line_needs_a_check():
    # 인쇄된 코드가 사전의 품목기준코드와 같고 이름도 맞으면 자동으로 확정된다 (약사 확인 없음)
    state = state_with(order_of(line("타이레놀정500mg", printed_code=TYLENOL)))
    state = merged(state, match_node(state, deps()))
    assert approve_node(state, deps()) == {"status": "awaiting_approval", "notice": "약사 승인을 기다립니다."}


# ------------------------------------------------------------ 다른 흐름의 자리


def test_ask_type_stops_with_the_question_the_supervisor_left():
    state = merged(initial_state("P0001"), supervisor_node(initial_state("P0001"), deps()))
    update = ask_type_node(state, deps())
    assert update == {"status": "awaiting_type", "notice": state["notice"]}
    assert "사진 종류 정보가 없습니다." in update["notice"]


def test_ask_type_has_a_question_of_its_own_when_none_was_left():
    assert ask_type_node(initial_state("P0001"), deps()) == {"status": "awaiting_type", "notice": ASK_TYPE_NOTICE}


def test_the_flows_that_are_not_built_yet_say_so():
    assert recheck_node(initial_state("P0001"), deps()) == {"status": "not_ready", "notice": NOT_READY_NOTICES["recheck"]}
    assert tray_node(initial_state("P0001"), deps()) == {"status": "not_ready", "notice": NOT_READY_NOTICES["tray"]}
    assert NOT_READY_NOTICES["recheck"] != NOT_READY_NOTICES["tray"]


# ------------------------------------------------------------ 공통


NODE_FUNCTIONS = [supervisor_node, structure_node, match_node, approve_node, ask_type_node, recheck_node, tray_node]


@pytest.mark.parametrize("node", NODE_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_node_does_not_change_the_state_it_was_given(node):
    state = merged(structured("rx_001"), {})
    before = copy.deepcopy(state)
    node(state, deps())
    assert state == before


@pytest.mark.parametrize("node", NODE_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_node_only_writes_fields_the_state_has(node):
    state = merged(structured("rx_001"), {})
    assert set(node(state, deps())) <= set(typing.get_type_hints(GraphState))


def test_the_tools_cannot_be_swapped_while_the_graph_runs():
    with pytest.raises(dataclasses.FrozenInstanceError):
        deps().api = lambda term: []


def test_a_sample_goes_through_the_nodes_in_order_to_the_approval_place():
    state = initial_state("P0001", load("cv_a1", "rx_002"))
    for node in (supervisor_node, structure_node, match_node, approve_node):
        state = merged(state, node(state, deps()))
    assert (state["flow"], state["status"], state["error_code"], state["errors"]) == ("prescription", "awaiting_approval", None, [])
    assert state["notice"].startswith("약사 승인을 기다립니다.")
    json.dumps(state)
