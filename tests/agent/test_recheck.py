"""복용 약 등록 · 재대조 (15단계): agent/recheck.py, register_meds · recheck · recheck_review 노드, HITL ③, AgentApi.register_meds.

수용 기준 (REQ-AG-05, 11/11 팀 결정): 처방약 + 복용 약 병용금기 2쌍 검출
    이부프로펜 계열(트리스펜 · 아르펜) × 메토트렉세이트, 이부프로펜 계열 × 케토롤락
진짜 DUR API는 부르지 않는다 (가짜 FakeDur. 메토트렉세이트 · 케토롤락 제품 코드는 지어낸 것). 실제 API 검출은 show_dur.py --cases가 본다.
"""
import json

import pytest

import fake_langgraph
from agent.api import AgentApi
from agent.checkpoint import memory_checkpointer
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.dur_api import TabooPartner
from agent.graph import build_graph
from agent.mcp_client import in_process
from agent.pipeline import structure_prescription
from agent.recheck import (
    NO_NEW_ANSWER,
    REVIEW_TAIL,
    MedsError,
    add_meds,
    consult_result,
    read_meds,
    recheck,
)
from agent.schema import DispensingOrder
from fake_dur import ARPEN, KETOROLAC, KETOROLAC_NAME, TABOO_REASON, TABOOS, TRISPEN, FakeDur

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
SAMPLES = RESOURCES_DIR / "samples"
METHOTREXATE, METHOTREXATE_NAME = "199900002", "가상메토트렉세이트정"        # 지어낸 제품 (사전 밖)
MTX_REASON = "메토트렉세이트 독성 증가"
TAB = {
    **TABOOS,
    TRISPEN: [*TABOOS[TRISPEN], TabooPartner(METHOTREXATE, METHOTREXATE_NAME, "메토트렉세이트", MTX_REASON)],
    ARPEN: [*TABOOS[ARPEN], TabooPartner(METHOTREXATE, METHOTREXATE_NAME, "메토트렉세이트", MTX_REASON)],
}
MTX = {"item_code": METHOTREXATE, "name": METHOTREXATE_NAME}
KETO = {"item_code": KETOROLAC, "name": KETOROLAC_NAME}


def load(kind, name):
    return json.loads((SAMPLES / kind / f"{name}.json").read_text(encoding="utf-8"))


def no_llm(*args):
    pytest.fail("LLM을 부르면 안 됩니다")


def order_of(name):
    """시연용 처방 (승인 화면의 조제지시서: 매칭 · 처방약끼리 대조까지 끝난 것)."""
    from agent.drug_match import match_order
    from agent.risk import check_order

    order = structure_prescription("P0001", load("rx8_a1", name), llm_fn=no_llm)
    return check_order(match_order(order, DICT), DICT, FakeDur(TAB))


# ------------------------------------------------------------ 복용 약 목록


def test_meds_are_read_from_codes_and_names():
    assert read_meds([{"item_code": " 199900002 ", "name": " 가상메토트렉세이트정 "}]) == [MTX]


@pytest.mark.parametrize("data, words", [
    ([], "비어"), (None, "비어"),
    ([{"item_code": "12345", "name": "x"}], "9자리"), ([{"name": "게보린"}], "9자리"),        # 이름만 있는 약은 등록하지 않는다
    ([{"item_code": "199900002", "name": " "}], "이름"), (["199900002"], "9자리"),
])
def test_wrong_meds_are_refused(data, words):
    with pytest.raises(MedsError, match=words):
        read_meds(data)


def test_the_same_med_is_registered_once():
    merged, added = add_meds([MTX], [MTX, KETO, KETO])
    assert merged == [MTX, KETO] and added == [KETO]


# ------------------------------------------------------------ 재대조 (수용 기준: 병용금기 2쌍)


@pytest.mark.parametrize("rx, rx_drug, med, reason", [
    ("rx8_17", TRISPEN, MTX, MTX_REASON),        # 트리스펜 × 메토트렉세이트
    ("rx8_16", ARPEN, KETO, TABOO_REASON),       # 아르펜 × 케토롤락
    ("rx8_17", TRISPEN, KETO, TABOO_REASON),
    ("rx8_16", ARPEN, MTX, MTX_REASON),
])
def test_the_two_demo_taboo_pairs_are_found_as_danger(rx, rx_drug, med, reason):
    result = recheck(order_of(rx), [med], DICT, FakeDur(TAB))
    taboo = [f for f in result.new_findings if f.type == "combination_taboo"]
    assert len(taboo) == 1 and set(taboo[0].item_codes) == {rx_drug, med["item_code"]}
    assert taboo[0].level == "danger" and reason in taboo[0].message
    assert result.risk_level == "danger" and result.needs_ack


def test_only_new_findings_are_shown_not_those_between_prescription_drugs():
    order = order_of("rx8_02")                                   # 트리스펜 + 아르펜: 이부프로펜 성분 중복은 A.5에 이미 있다
    assert [f.type for f in order.risk_check.findings] == ["ingredient_dup"]
    result = recheck(order, [MTX], DICT, FakeDur(TAB))
    assert all(METHOTREXATE in f.item_codes for f in result.new_findings)
    assert len([f for f in result.new_findings if f.type == "combination_taboo"]) == 2      # 두 처방약 모두 메토트렉세이트와 금기


def test_an_ingredient_duplicate_with_a_current_med_is_a_warning_without_stopping():
    trispen = {"item_code": TRISPEN, "name": "트리스펜연질캡슐"}
    result = recheck(order_of("rx8_16"), [trispen], DICT, FakeDur(TAB))      # 아르펜 처방 + 트리스펜 복용: 이부프로펜
    dup = [f for f in result.new_findings if f.type == "ingredient_dup"]
    assert dup and dup[0].level == "warning" and "이부프로펜" in dup[0].message
    assert not result.needs_ack


def test_the_same_drug_prescribed_and_taken_is_a_new_finding():
    result = recheck(order_of("rx8_16"), [{"item_code": ARPEN, "name": "아르펜정"}], DICT, FakeDur(TAB))
    same = [f for f in result.new_findings if f.item_codes == [ARPEN]]
    assert same and same[0].level == "warning" and "같은 약" in same[0].message


def test_nothing_new_keeps_the_prescription_level():
    order = order_of("rx8_02")
    result = recheck(order, [{"item_code": "199900099", "name": "가상비타민정"}], DICT, FakeDur(TAB))
    assert result.new_findings == () and result.risk_level == order.risk_check.level == "warning"


def test_without_dur_the_taboos_are_not_checked_and_it_says_so():
    result = recheck(order_of("rx8_17"), [MTX], DICT, None)
    assert not [f for f in result.new_findings if f.type == "combination_taboo"]
    assert any("DUR API를 쓰지 않아" in w for w in result.warnings)             # '해당 없음'이 안전함으로 읽히지 않게
    assert any(METHOTREXATE_NAME in w for w in result.warnings)                # 주성분을 모르는 복용 약도 알린다


# ------------------------------------------------------------ A.6 상담 결과


@pytest.mark.parametrize("status, after", [("approved", False), ("planned", True), ("inspected", True)])
def test_a_finding_after_dispensing_asks_for_a_review(status, after):
    result = recheck(order_of("rx8_17"), [MTX], DICT, FakeDur(TAB))
    consult = consult_result("P0001", "P0001", result, [MTX], status)
    assert consult.risk_scope == "rx_with_current_meds" and consult.recheck_after_dispense is after
    assert consult.risk_level == "danger" and consult.answer.endswith(REVIEW_TAIL) is after
    assert "[위험 · 병용금기]" in consult.answer


def test_no_new_finding_is_said_plainly():
    result = recheck(order_of("rx8_17"), [{"item_code": "199900099", "name": "가상비타민정"}], DICT, FakeDur(TAB))
    consult = consult_result("P0001", "P0001", result, [{"item_code": "199900099", "name": "가상비타민정"}], "planned")
    assert consult.answer == NO_NEW_ANSWER.format(n=1) and consult.recheck_after_dispense is False


# ------------------------------------------------------------ 그래프와 입구 (AgentApi)


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


def make_api(dur=None, tools=None):
    graph = build_graph(dictionary=DICT, api=None, dur=dur if dur is not None else FakeDur(TAB), tools=tools,
                        checkpointer=memory_checkpointer(),
                        structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=no_llm))
    return AgentApi(graph, DICT)


def approved(api, rx="rx8_17", cid="P0001"):
    pause = api.start(cid, load("rx8_a1", rx), prescription_id=cid)
    assert api.resume(cid, "rx_approval", {"action": "approve", "order": pause.payload["order"]}) is None
    return api


def test_a_danger_taboo_stops_at_hitl_3_until_the_pharmacist_acknowledges(backend):
    api = approved(make_api())
    pause = api.register_meds("P0001", [MTX], message_id="m3")
    assert pause.hitl_point == "risk" and pause.payload["choices"] == ["acknowledge"]
    assert pause.payload["consult_result"]["risk_level"] == "danger" and "재검토" in pause.payload["notice"]
    assert api.resume("P0001", "risk", {"action": "acknowledge"}) is None
    result = api.result("P0001")
    assert result["current_meds"] == [MTX] and result["consult_result"]["recheck_after_dispense"] is True
    assert result["notes"][-1].startswith("위험 신호 확인(약사):") and result["status"] == "planned"   # 처방 상태는 그대로


def test_a_warning_only_recheck_does_not_stop(backend):
    api = approved(make_api(), rx="rx8_16")
    assert api.register_meds("P0001", [{"item_code": TRISPEN, "name": "트리스펜연질캡슐"}]) is None
    consult = api.result("P0001")["consult_result"]
    assert consult["risk_level"] == "warning" and consult["new_findings"]


def test_meds_add_up_over_messages_and_each_recheck_uses_them_all(backend):
    api = approved(make_api())
    api.register_meds("P0001", [{"item_code": "199900099", "name": "가상비타민정"}])
    assert api.result("P0001")["consult_result"]["new_findings"] == []
    api.register_meds("P0001", [KETO])
    api.resume("P0001", "risk", {"action": "acknowledge"})
    assert [m["item_code"] for m in api.result("P0001")["current_meds"]] == ["199900099", KETOROLAC]


def test_a_wrong_answer_to_hitl_3_leaves_the_graph_waiting(backend):
    api = approved(make_api())
    api.register_meds("P0001", [MTX])
    with pytest.raises(ValueError, match="acknowledge"):
        api.resume("P0001", "risk", {"action": "confirm"})
    assert api.pending("P0001").hitl_point == "risk"


def test_meds_cannot_be_registered_while_waiting_or_without_a_conversation(backend):
    api = make_api()
    with pytest.raises(ValueError, match="없는 대화"):
        api.register_meds("P0001", [MTX])
    api.start("P0001", load("rx8_a1", "rx8_17"), prescription_id="P0001")      # 승인 대기 중
    with pytest.raises(ValueError, match="약사 응답을 기다리는"):
        api.register_meds("P0001", [MTX])
    with pytest.raises(MedsError):
        api.register_meds("P0001", [{"name": "메토트렉세이트"}])


def test_meds_in_a_conversation_without_a_prescription_are_only_registered(backend):
    api = make_api()
    api.start("C0001", None, None, "이부프로펜 먹어도 되나요?")
    assert api.register_meds("C0001", [MTX]) is None
    result = api.result("C0001")
    assert result["current_meds"] == [MTX] and "처방이 없는 대화" in result["notice"]
    assert result["consult_result"] is None                              # 대조할 처방이 없으니 재대조 결과도 없다


def test_a_text_about_current_meds_only_proposes_them(backend):
    api = approved(make_api())
    api.send("P0001", None, None, "게보린 먹고 있어요")
    result = api.result("P0001")
    assert result["current_meds"] == [] and result["proposals"] and "자동으로 등록하지 않습니다" in result["notice"]


def test_the_mcp_tools_give_the_same_recheck(backend):
    with in_process(DICT, None, FakeDur(TAB)) as tools:
        by_mcp = approved(make_api(tools=tools))
        by_mcp.register_meds("P0001", [MTX])
        by_function = approved(make_api())
        by_function.register_meds("P0001", [MTX])
        assert by_mcp.result("P0001")["consult_result"] == by_function.result("P0001")["consult_result"]
