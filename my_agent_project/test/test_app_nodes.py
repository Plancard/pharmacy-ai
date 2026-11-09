"""LangGraph 노드·분기 (app/nodes.py, app/edges.py, app/state.py). LangGraph 없이 함수로 시험한다."""
import json

import pytest

from agent.config import DATA_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import ConfigError, StructureError
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder
from app.edges import after_structure
from app.nodes import NO_API_WARNING, Deps, approve_node, match_node, structure_node
from app.state import GraphState, initial_state

DICT = load_dictionary()
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]


def load(kind, sample_id):
    return json.loads((DATA_DIR / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback=None):
    raise AssertionError("규칙 경로가 성공하면 LLM을 부르지 않아야 한다")


def rule_only(prescription_id, ocr):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


DEPS = Deps(dictionary=DICT, structure=rule_only)


def state_of(sample_id):
    return initial_state(f"P-{sample_id}", load("cv_samples", sample_id))


def merged(state, update):
    """노드가 돌려준 칸을 상태에 합친다 (errors는 이어 붙인다)."""
    out = {**state, **update}
    if "errors" in update:
        out["errors"] = state.get("errors", []) + update["errors"]
    return out


# ------------------------------------------------------------ state


def test_initial_state():
    ocr = {"lines": []}
    assert initial_state("P1", ocr) == {
        "prescription_id": "P1", "ocr": ocr, "order": None, "status": "received", "notice": None,
        "error_code": None, "errors": [],
    }


def test_state_fields_and_the_errors_reducer():
    import operator
    import typing

    hints = typing.get_type_hints(GraphState, include_extras=True)
    assert set(hints) == {"prescription_id", "ocr", "order", "status", "notice", "error_code", "errors"}
    assert hints["errors"].__metadata__ == (operator.add,)     # 덮어쓰지 않고 쌓는다
    assert set(typing.get_args(hints["status"])) == {"received", "structured", "matched", "awaiting_approval", "failed"}


# ------------------------------------------------------------ structure


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_structure_makes_an_order_as_json(sample_id):
    update = structure_node(state_of(sample_id), DEPS)
    assert set(update) == {"order", "status", "notice"}
    assert update["status"] == "structured" and update["notice"] is None
    json.dumps(update["order"], ensure_ascii=False)            # JSON으로 바꿀 수 있어야 한다 (체크포인터)
    order = DispensingOrder.model_validate(update["order"])
    assert order.prescription_id == f"P-{sample_id}" and order.source == "rule"
    assert [d.printed_code for d in order.items] == [d["printed_code"] for d in load("cv_expected", sample_id)["items"]]


def test_structure_failure_is_recorded_not_raised():
    def fail(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code="E_LLM")

    update = structure_node(state_of("rx_001"), Deps(dictionary=DICT, structure=fail))
    assert update == {
        "order": None, "status": "failed", "notice": "자동 구조화에 실패했습니다. 직접 입력해 주세요.",
        "error_code": "E_LLM", "errors": ["[structure] E_LLM 자동 구조화에 실패했습니다. 직접 입력해 주세요."],
    }


def test_a_config_problem_is_not_hidden_as_a_failed_prescription():
    def no_key(prescription_id, ocr):
        raise ConfigError("OPENAI_API_KEY가 설정되지 않았습니다.")

    with pytest.raises(ConfigError):
        structure_node(state_of("rx_001"), Deps(dictionary=DICT, structure=no_key))


def test_structure_passes_the_prescription_id_and_ocr():
    seen = []

    def spy(prescription_id, ocr):
        seen.append((prescription_id, ocr))
        return rule_only(prescription_id, ocr)

    state = state_of("rx_002")
    structure_node(state, Deps(dictionary=DICT, structure=spy))
    assert seen == [("P-rx_002", state["ocr"])]


# ------------------------------------------------------------ match


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_match_fills_item_codes_and_flags(sample_id):
    state = merged(state_of(sample_id), structure_node(state_of(sample_id), DEPS))
    update = match_node(state, DEPS)
    assert set(update) == {"order", "status"} and update["status"] == "matched"
    order = DispensingOrder.model_validate(update["order"])
    assert [d.item_code for d in order.items] == [d["item_code"] for d in load("cv_expected", sample_id)["items"]]
    json.dumps(update["order"], ensure_ascii=False)


def test_match_does_not_change_the_input_state():
    state = merged(state_of("rx_001"), structure_node(state_of("rx_001"), DEPS))
    before = json.dumps(state, ensure_ascii=False, sort_keys=True)
    match_node(state, DEPS)
    assert json.dumps(state, ensure_ascii=False, sort_keys=True) == before


def test_match_refuses_a_broken_order_in_the_state():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        match_node({"order": {"prescription_id": "P1"}}, DEPS)


# ------------------------------------------------------------ approve (자리)


@pytest.mark.parametrize(
    "sample_id, notice",
    [
        ("rx_001", "약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다."),
        ("rx_002", "약사 승인을 기다립니다. 확인이 필요한 약품이 1줄 있습니다."),
        ("rx_005", "약사 승인을 기다립니다. 확인이 필요한 약품이 4줄 있습니다."),
    ],
)
def test_approve_waits_and_says_how_many_lines_need_checking(sample_id, notice):
    state = merged(state_of(sample_id), structure_node(state_of(sample_id), DEPS))
    state = merged(state, match_node(state, DEPS))
    assert approve_node(state, DEPS) == {"status": "awaiting_approval", "notice": notice}


def test_approve_without_flags_says_only_that_it_waits():
    state = merged(state_of("rx_002"), structure_node(state_of("rx_002"), DEPS))
    order = DispensingOrder.model_validate(state["order"])       # 매칭 전이라 확인 표시가 없다
    assert order.flagged_indices == []
    assert approve_node(state, DEPS) == {"status": "awaiting_approval", "notice": "약사 승인을 기다립니다."}


# ------------------------------------------------------------ edges


def test_after_structure_goes_to_match_only_when_structured_with_an_order():
    assert after_structure({"status": "structured", "order": {"x": 1}}) == "match"
    assert after_structure({"status": "failed", "order": None}) == "end"
    assert after_structure({"status": "structured", "order": None}) == "end"   # 상태가 어긋나면 멈춘다
    assert after_structure({"status": "failed", "order": {"x": 1}}) == "end"
    assert after_structure({}) == "end"
    assert after_structure({"status": "received", "order": {"x": 1}}) == "end"


# ------------------------------------------------------------ main.py: langgraph가 없을 때


def test_main_says_how_to_install_langgraph_when_it_is_missing(monkeypatch):
    import sys

    import main

    monkeypatch.setitem(sys.modules, "langgraph", None)          # import langgraph가 실패하게 만든다
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    lines = []
    assert main.main([], out=lines.append) == 1
    assert lines == [
        "약품 API: 사용 안 함 (.env에 DRUG_API_KEY가 없어 약품 사전 8종만 씁니다)",
        "설정 문제: langgraph를 불러오지 못했습니다 (langgraph.graph). `pip install langgraph`로 설치하세요.",
    ]


def test_a_failure_without_a_code_is_still_recorded():
    def fail(prescription_id, ocr):
        raise StructureError("처방 ID가 없습니다.")

    update = structure_node(state_of("rx_001"), Deps(dictionary=DICT, structure=fail))
    assert update["error_code"] is None and update["errors"] == ["[structure] - 처방 ID가 없습니다."]


def test_an_empty_ocr_gives_e_ocr_empty_through_the_real_structure_function():
    update = structure_node(initial_state("P1", {"lines": []}), DEPS)
    assert (update["status"], update["error_code"]) == ("failed", "E_OCR_EMPTY")


# ------------------------------------------------------------ match: 의약품 API


class FakeApi:
    def __init__(self, products=(), error=None):
        self.products, self.error, self.calls = list(products), error, []

    def __call__(self, name):
        self.calls.append(name)
        if self.error:
            raise self.error
        return self.products


def structured(sample_id):
    return merged(state_of(sample_id), structure_node(state_of(sample_id), DEPS))


def test_match_asks_the_api_only_for_drugs_missing_from_the_dictionary():
    from agent.drug_api import ApiDrug

    api = FakeApi([ApiDrug("199900031", "알모탄정", "가상제약")])
    update = match_node(structured("rx_002"), Deps(dictionary=DICT, structure=rule_only, api=api))
    order = DispensingOrder.model_validate(update["order"])
    assert api.calls == ["알모란정"]                      # 게보린정·프론크논캠술은 사전에서 정해진다
    assert [(d.item_code, d.match_method) for d in order.items] == [
        ("197900277", "code"), ("200200099", "code"), ("199900031", "api"),
    ]
    assert order.items[2].needs_pharmacist_check            # API로 정한 줄은 언제나 약사 확인
    assert NO_API_WARNING not in order.warnings


def test_an_api_failure_does_not_stop_the_graph():
    from agent.drug_api import DrugApiError

    api = FakeApi(error=DrugApiError("의약품 API에 연결하지 못했습니다 (URLError)."))
    update = match_node(structured("rx_002"), Deps(dictionary=DICT, structure=rule_only, api=api))
    item = DispensingOrder.model_validate(update["order"]).items[2]
    assert update["status"] == "matched" and item.item_code is None
    assert "(E_EXT_API)" in item.check_reasons[0]
    # API를 썼는데 못 찾은 것이라 'API를 쓰지 않았다'는 경고는 붙지 않는다
    assert NO_API_WARNING not in DispensingOrder.model_validate(update["order"]).warnings


def test_without_the_api_an_unmatched_drug_leaves_a_warning_once():
    state = structured("rx_002")
    update = match_node(state, DEPS)
    order = DispensingOrder.model_validate(update["order"])
    assert order.warnings.count(NO_API_WARNING) == 1
    again = DispensingOrder.model_validate(match_node(merged(state, update), DEPS)["order"])
    assert again.warnings.count(NO_API_WARNING) == 1           # 다시 매칭해도 한 번만


def test_without_the_api_no_warning_when_every_drug_is_in_the_dictionary():
    state = structured("rx_002")
    order = DispensingOrder.model_validate(state["order"])
    state["order"] = order.model_copy(update={"items": order.items[:2]}).model_dump(mode="json")
    update = match_node(state, DEPS)
    assert NO_API_WARNING not in DispensingOrder.model_validate(update["order"]).warnings


def test_deps_have_no_api_unless_given():
    assert Deps(dictionary=DICT).api is None
