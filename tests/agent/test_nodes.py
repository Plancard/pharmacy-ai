"""그래프의 노드 (agent/nodes.py). LangGraph 없이 함수로 불러 시험한다."""
import copy
import dataclasses
import json
import typing

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_api import ApiDrug
from agent.drug_dictionary import load_dictionary
from agent.errors import E_LLM, E_OCR_EMPTY, E_PLAN_RANGE, ConfigError, StructureError
from agent.drug_match import match_order
from agent.mcp_client import McpToolError
from agent.nodes import (
    MCP_FAILED_WARNING,
    APPROVED_NOTICE,
    ASK_TYPE_NOTICE,
    EDITED_NOTICE,
    HITL_APPROVAL,
    NO_API_WARNING,
    NOT_READY_NOTICES,
    Deps,
    approve_node,
    ask_type_node,
    decide_flow,
    consult_node,
    match_node,
    plan_node,
    read_reply,
    recheck_node,
    register_meds_node,
    risk_node,
    structure_node,
    supervisor_node,
    tray_node,
)
from agent.pipeline import structure_prescription
from agent.risk import UNCHECKED_TAIL
from agent.schema import DispensingOrder, DrugLine, Plan, RiskCheck, RiskFinding
from agent.state import GraphState, initial_state

SAMPLES = RESOURCES_DIR / "samples"
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL = "202106092"
GEVORIN = "197900277"
TRISPEN = "201110120"
ARPEN = "200201239"


def load(kind, sample_id):
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback):
    raise AssertionError("이 시험에서는 LLM을 부르면 안 됩니다")


def real_structure(prescription_id, ocr):
    """실제 구조화에서 LLM만 막아 둔 것. 샘플은 규칙 경로로 읽히므로 LLM은 불리지 않는다."""
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


def approve_as_shown(payload):
    """약사가 화면의 조제지시서를 그대로 승인한다."""
    return {"action": "approve", "order": payload["order"]}


class Pharmacist:
    """승인 대기에서 받은 화면(payload)을 기록하고 정해 둔 응답을 돌려준다 (그래프 밖에서 approve_node를 시험할 때)."""

    def __init__(self, answer=approve_as_shown):
        self.answer = answer
        self.seen = []

    def __call__(self, payload):
        self.seen.append(payload)
        return self.answer(payload)


def deps(*, structure=real_structure, api=None, ask=approve_as_shown):
    return Deps(dictionary=DICT, structure=structure, api=api, ask=ask)


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


def test_supervisor_sets_the_route_when_the_kind_is_clear():
    state = initial_state("P0001", load("cv_a1", "rx_001"))
    assert supervisor_node(state, deps()) == {"route": ["prescription"], "flow": "prescription", "proposals": [], "memo": None}


def test_supervisor_also_leaves_the_question_when_it_asks():
    update = supervisor_node(initial_state("P0001"), deps())
    assert (update["route"], update["flow"]) == (["ask_type"], "ask_type")
    assert update["notice"].startswith("사진 종류를 정하지 못했습니다.")


def test_supervisor_reads_the_kind_from_the_ocr_of_a_state_without_photo_kind():
    state = {"prescription_id": "P0001", "ocr": {"doc_type": "med_bag"}}      # 칸이 몇 개 없어도 읽는다
    assert supervisor_node(state, deps())["route"] == ["recheck"]


def test_supervisor_puts_the_text_of_a_prescription_photo_into_proposals():
    state = initial_state("P0001", load("rx8_a1", "rx8_01"), "prescription", "게보린 빼 주세요")
    update = supervisor_node(state, deps())
    assert update["route"] == ["prescription"]
    assert update["proposals"] == [{"kind": "edit", "text": "게보린 빼 주세요", "drugs": [{"item_code": GEVORIN, "name": "게보린정"}]}]


def test_supervisor_ends_with_noted_when_the_text_is_only_a_memo():
    update = supervisor_node(initial_state("P0001", None, None, "메모: 오후 수령"), deps())
    assert (update["route"], update["status"], update["memo"]) == ([], "noted", "메모: 오후 수령")


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


def matched(sample_id):
    state = structured(sample_id)
    return merged(state, match_node(state, deps()))


def checked(sample_id):
    state = matched(sample_id)
    return merged(state, risk_node(state, deps()))


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


class FakeTools:
    """MCP 클라이언트 대신. 서버가 하는 일(match_order)을 그대로 하고, 불린 횟수와 실패를 흉내 낸다."""

    def __init__(self, *, api_used=False, fail=None):
        self.calls, self.api_used, self.fail = 0, api_used, fail

    def match_drug(self, order):
        self.calls += 1
        if self.fail:
            raise McpToolError(self.fail)
        return match_order(order, DICT), self.api_used


def tools_deps(tools, api=None):
    return dataclasses.replace(deps(api=api), tools=tools)


def test_with_mcp_tools_the_match_goes_through_the_tool_and_gives_the_same_order():
    tools = FakeTools()
    by_mcp = match_node(structured("rx_004"), tools_deps(tools))
    assert tools.calls == 1 and by_mcp == match_node(structured("rx_004"), deps())


def test_with_mcp_tools_the_no_api_warning_follows_the_server():
    without = DispensingOrder.model_validate(match_node(structured("rx_005"), tools_deps(FakeTools(api_used=False)))["order"])
    assert without.warnings.count(NO_API_WARNING) == 1
    with_api = DispensingOrder.model_validate(match_node(structured("rx_005"), tools_deps(FakeTools(api_used=True)))["order"])
    assert NO_API_WARNING not in with_api.warnings                 # 서버가 API를 썼다면 사전만 썼다는 경고는 틀린 말이다


def test_when_the_mcp_tool_fails_the_node_matches_by_itself_and_says_so():
    tools = FakeTools(fail="연결이 끊겼습니다")
    update = match_node(structured("rx_004"), tools_deps(tools))
    order = DispensingOrder.model_validate(update["order"])
    direct = DispensingOrder.model_validate(match_node(structured("rx_004"), deps())["order"])
    assert tools.calls == 1 and update["status"] == "matched"
    assert [d.item_code for d in order.items] == [d.item_code for d in direct.items]
    assert f"{MCP_FAILED_WARNING}: 연결이 끊겼습니다" in order.warnings


def test_without_mcp_tools_the_function_is_called_directly():
    assert Deps(dictionary=DICT, structure=real_structure).tools is None


def test_a_failing_api_does_not_stop_the_match():
    from agent.errors import DrugApiError

    def api(term):
        raise DrugApiError("연결할 수 없습니다.")

    state = state_with(order_of(line("가나다정")))
    update = match_node(state, deps(api=api))
    order = DispensingOrder.model_validate(update["order"])
    assert update["status"] == "matched" and order.items[0].needs_pharmacist_check


# ------------------------------------------------------------ 위험 신호 대조


def after_match(*lines):
    """약품 이름만 있는 줄들을 약품 매칭까지 시킨 상태."""
    state = state_with(order_of(*lines))
    return merged(state, match_node(state, deps()))


def test_risk_fills_the_risk_check_and_nothing_else_in_the_order():
    state = matched("rx_004")
    update = risk_node(state, deps())
    assert set(update) == {"order", "status"} and update["status"] == "checked"
    before, after = DispensingOrder.model_validate(state["order"]), DispensingOrder.model_validate(update["order"])
    assert before.risk_check is None and after.risk_check is not None
    assert after.items == before.items                               # 약품 줄은 그대로다 (매칭 결과도 확인 표시도)
    json.dumps(update)


def test_risk_finds_the_ingredient_duplicate_of_demo_combination_a():
    update = risk_node(after_match(line("타이레놀정500mg"), line("게보린정")), deps())
    risk = DispensingOrder.model_validate(update["order"]).risk_check
    assert (risk.scope, risk.level) == ("rx_only", "warning")
    [finding] = risk.findings
    assert (finding.type, finding.level, finding.item_codes) == ("ingredient_dup", "warning", [TYLENOL, GEVORIN])
    assert "아세트아미노펜" in finding.message


def test_risk_finds_the_ingredient_duplicate_of_demo_combination_b():
    update = risk_node(after_match(line("트리스펜연질캡슐"), line("아르펜정")), deps())
    risk = DispensingOrder.model_validate(update["order"]).risk_check
    [finding] = risk.findings
    assert (risk.level, finding.item_codes) == ("warning", [TRISPEN, ARPEN])
    assert "이부프로펜" in finding.message


def test_risk_says_none_when_the_prescription_drugs_share_no_ingredient():
    update = risk_node(after_match(line("프론크논캡슐"), line("뮤코다인캡슐200mg")), deps())
    risk = DispensingOrder.model_validate(update["order"]).risk_check
    assert (risk.scope, risk.level, risk.findings) == ("rx_only", "none", [])


def test_a_drug_that_could_not_be_checked_is_named_in_the_warnings():
    update = risk_node(after_match(line("타이레놀정500mg"), line("가나다정")), deps())
    order = DispensingOrder.model_validate(update["order"])
    assert order.risk_check.level == "none"                          # 같이 볼 짝이 없으면 '해당 없음'이다. 그래서 경고가 꼭 필요하다
    assert f"가나다정: 품목기준코드를 정하지 못해 {UNCHECKED_TAIL}" in order.warnings


def test_risk_gives_the_same_order_when_it_runs_again():
    state = after_match(line("타이레놀정500mg"), line("게보린정"), line("가나다정"))
    first = risk_node(state, deps())
    second = risk_node({**state, "order": first["order"]}, deps())
    assert second["order"] == first["order"]


def test_risk_does_not_pile_up_warnings_of_the_last_run():
    state = after_match(line("타이레놀정500mg"), line("가나다정"))
    first = risk_node(state, deps())
    again = risk_node({**state, "order": first["order"]}, deps())
    assert DispensingOrder.model_validate(again["order"]).warnings.count(f"가나다정: 품목기준코드를 정하지 못해 {UNCHECKED_TAIL}") == 1


def test_risk_keeps_the_warnings_that_other_nodes_left():
    state = matched("rx_005")                                         # 매칭이 남긴 'API 없음' 경고가 있다
    after = DispensingOrder.model_validate(risk_node(state, deps())["order"])
    assert NO_API_WARNING in after.warnings


# ------------------------------------------------------------ 승인 대기 (HITL ①)


def asked(state):
    """승인 대기에서 약사에게 보인 화면과 approve_node가 돌려준 값."""
    pharmacist = Pharmacist()
    update = approve_node(state, deps(ask=pharmacist))
    [payload] = pharmacist.seen
    return payload, update


def test_approve_shows_the_order_the_ocr_lines_and_a_notice():
    state = checked("rx_001")
    payload, _ = asked(state)
    assert set(payload) == {"hitl_point", "order", "ocr_lines", "notice", "proposals", "memo"}
    assert payload["hitl_point"] == HITL_APPROVAL == "rx_approval"
    assert (payload["proposals"], payload["memo"]) == ([], None)
    assert payload["order"] == state["order"]
    used = {i for item in state["order"]["items"] for i in item["source_line_ids"]}
    assert set(payload["ocr_lines"]) == {str(i) for i in used}
    some = next(iter(payload["ocr_lines"].values()))
    assert set(some) == {"text", "conf"}
    json.dumps(payload)


def test_approve_shows_the_proposals_and_the_memo_of_the_message():
    proposal = {"kind": "edit", "text": "게보린 빼 주세요", "drugs": []}
    payload, update = asked({**checked("rx_001"), "proposals": [proposal], "memo": "오후 수령"})
    assert (payload["proposals"], payload["memo"]) == ([proposal], "오후 수령")
    assert update["status"] == "approved"                            # 제안은 보이기만 하고 지시서를 바꾸지 않는다


def test_approve_says_how_many_lines_need_a_check():
    payload, _ = asked(checked("rx_001"))
    assert payload["notice"] == "약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다."


def test_approve_has_nothing_to_add_when_no_line_needs_a_check():
    # 인쇄된 코드가 사전의 품목기준코드와 같고 이름도 맞으면 자동으로 확정된다 (약사 확인 없음)
    state = state_with(order_of(line("타이레놀정500mg", printed_code=TYLENOL)))
    state = merged(state, match_node(state, deps()))
    state = merged(state, risk_node(state, deps()))
    assert asked(state)[0]["notice"] == "약사 승인을 기다립니다."


def test_approve_says_a_warning_must_be_confirmed_before_approval():
    state = after_match(line("타이레놀정500mg"), line("게보린정"))
    state = merged(state, risk_node(state, deps()))
    assert asked(state)[0]["notice"] == (
        "약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다."
        " 위험 신호가 1건 있습니다(경고). 약사가 확인한 뒤에 승인할 수 있습니다."
    )


def order_with_risk(level):
    codes = [TYLENOL, GEVORIN]
    items = [line("타이레놀정500mg"), line("게보린정")]
    for item, code in zip(items, codes):
        item.item_code, item.match_method = code, "code"
    risk = RiskCheck(scope="rx_only", level=level, findings=[
        RiskFinding(type="ingredient_dup", item_codes=codes, level=level, message="가상의 설명입니다."),
    ])
    return DispensingOrder(prescription_id="P0001", items=items, risk_check=risk).model_dump(mode="json")


@pytest.mark.parametrize("level, label, confirm", [("caution", "주의", False), ("warning", "경고", True), ("danger", "위험", True)])
def test_approve_asks_for_a_confirmation_from_warning_up_but_only_shows_a_caution(level, label, confirm):
    notice = asked(state_with(order_with_risk(level)))[0]["notice"]
    assert f"위험 신호가 1건 있습니다({label})." in notice
    assert ("약사가 확인한 뒤에 승인할 수 있습니다." in notice) is confirm


def test_approve_refuses_an_order_that_skipped_the_risk_check():
    with pytest.raises(ValueError, match="risk_check"):
        approve_node(matched("rx_001"), deps())


def test_approving_the_order_as_shown_approves_it():
    _, update = asked(checked("rx_001"))
    assert update == {"status": "approved", "notice": APPROVED_NOTICE, "error_code": None}


def edit(change):
    """약사가 조제지시서를 고쳐서 돌려준다. change(order dict)가 그 자리에서 고친다."""
    def answer(payload):
        order = copy.deepcopy(payload["order"])
        change(order)
        return {"action": "edit", "order": order}
    return answer


def test_editing_puts_the_new_lines_in_and_asks_for_matching_and_checking_again():
    state = after_match(line("타이레놀정500mg"), line("게보린정"))
    state = merged(state, risk_node(state, deps()))

    def two_days(order):
        order["items"][0]["days"] = 2

    update = approve_node(state, deps(ask=edit(two_days)))
    assert set(update) == {"order", "status", "notice", "error_code"}
    assert (update["status"], update["notice"], update["error_code"]) == ("edited", EDITED_NOTICE, None)
    assert update["order"]["items"][0]["days"] == 2
    assert update["order"]["risk_check"] is None                     # 고친 뒤에는 다시 대조한다
    assert update["order"]["issue_no"] == state["order"]["issue_no"]   # 지시서의 나머지는 그대로


def test_approving_with_changed_lines_counts_as_an_edit():
    state = checked("rx_001")

    def approve_changed(payload):
        order = copy.deepcopy(payload["order"])
        order["items"][0]["count_per_dose"] = 2
        return {"action": "approve", "order": order}

    assert approve_node(state, deps(ask=approve_changed))["status"] == "edited"


def test_an_edit_that_drops_a_drug_is_fine_even_with_the_old_risk_check_in_the_reply():
    state = after_match(line("타이레놀정500mg"), line("게보린정"))
    state = merged(state, risk_node(state, deps()))

    def drop_gevorin(order):
        order["items"] = order["items"][:1]                          # risk_check는 아직 게보린을 가리킨다

    update = approve_node(state, deps(ask=edit(drop_gevorin)))
    assert [i["name"] for i in update["order"]["items"]] == ["타이레놀정500mg"]


@pytest.mark.parametrize("reply, said", [
    (None, "action"),
    ("approve", "action"),
    ({"action": "reject", "order": {}}, "action"),
    ({"action": "approve"}, "order"),
    ({"action": "approve", "order": "지시서"}, "order"),
    ({"action": "approve", "order": {"prescription_id": "P0001", "items": []}}, "items"),
])
def test_a_bad_reply_is_refused(reply, said):
    with pytest.raises(ValueError, match=said):
        read_reply(reply, DispensingOrder.model_validate(checked("rx_001")["order"]))


def test_a_line_the_pharmacist_filled_in_loses_its_failed_matching():
    order = checked("rx_001")["order"]
    reply = copy.deepcopy(order)
    reply["items"][1]["item_code"] = GEVORIN                          # match_method는 아직 none
    _, read = read_reply({"action": "edit", "order": reply}, DispensingOrder.model_validate(order))
    assert (read.items[1].item_code, read.items[1].match_method, read.items[1].match_candidates) == (GEVORIN, None, [])
    assert read.items[0].match_method == "name"                       # 다른 줄은 그대로


def test_a_reply_for_another_prescription_is_refused():
    order = checked("rx_001")["order"]
    with pytest.raises(ValueError, match="다른 처방"):
        read_reply({"action": "approve", "order": {**order, "prescription_id": "P0002"}}, DispensingOrder.model_validate(order))


def test_without_an_ask_function_the_node_uses_langgraph_interrupt(monkeypatch):
    import fake_langgraph

    fake_langgraph.install(monkeypatch)
    with pytest.raises(Exception, match="paused"):                    # 그래프 밖에서 interrupt를 부르면 멈춤 신호가 그대로 나온다
        approve_node(checked("rx_001"), Deps(dictionary=DICT, structure=real_structure))


def test_after_a_plan_failure_the_notice_starts_with_the_reason():
    state = {**checked("rx_001"), "status": "failed", "notice": "조제 계획을 만들 수 없습니다. 이유."}
    assert asked(state)[0]["notice"].startswith("조제 계획을 만들 수 없습니다. 이유. 고친 뒤 다시 승인해 주세요. 약사 승인을 기다립니다.")


# ------------------------------------------------------------ 고친 뒤 다시 매칭


def edited_state(*lines):
    return {**state_with(order_of(*lines)), "status": "edited"}


def test_after_an_edit_a_drug_the_pharmacist_chose_is_kept():
    chosen = DrugLine(item_code=GEVORIN, name="가나다정", printed_code=None, count_per_dose=1, times=["morning"], days=3)
    order = DispensingOrder.model_validate(match_node(edited_state(chosen), deps())["order"])
    item = order.items[0]
    assert (item.item_code, item.match_method, item.needs_pharmacist_check, item.check_reasons) == (GEVORIN, None, False, [])


def test_after_an_edit_a_line_the_matching_agrees_with_gets_the_matching_result():
    same = DrugLine(item_code=GEVORIN, name="게보린정", printed_code=GEVORIN, count_per_dose=1, times=["morning"], days=3)
    item = DispensingOrder.model_validate(match_node(edited_state(same), deps())["order"]).items[0]
    assert (item.item_code, item.match_method) == (GEVORIN, "code")


def test_after_an_edit_a_line_without_a_code_is_matched_again():
    item = DispensingOrder.model_validate(match_node(edited_state(line("게보린정")), deps())["order"]).items[0]
    assert (item.item_code, item.match_method) == (GEVORIN, "name")


def test_without_an_edit_matching_decides_every_line_again():
    chosen = DrugLine(item_code=GEVORIN, name="가나다정", printed_code=None, count_per_dose=1, times=["morning"], days=3)
    item = DispensingOrder.model_validate(match_node(state_with(order_of(chosen)), deps())["order"]).items[0]
    assert item.item_code is None                                     # 구조화 직후의 줄에는 약사의 선택이 없다


# ------------------------------------------------------------ 조제 계획 (승인 뒤)


def approved(*lines):
    """약품 매칭 · 위험 신호 대조를 마치고 약사가 승인한 상태."""
    state = after_match(*lines)
    state = merged(state, risk_node(state, deps()))
    return {**state, "status": "approved"}


def test_plan_makes_the_a4_plan_of_the_approved_order():
    update = plan_node(approved(line("타이레놀정500mg"), line("게보린정")), deps())
    assert set(update) == {"plan", "status", "notice"} and update["status"] == "planned"
    plan = Plan.model_validate(update["plan"])
    assert [(c.day, c.time, [(i.item_code, i.count) for i in c.items]) for c in plan.plan] == [
        (day, "morning", [(TYLENOL, 1), (GEVORIN, 1)]) for day in (1, 2, 3)
    ]
    assert update["notice"] == "조제 계획을 만들었습니다 (3일, 약이 든 칸 3개)."
    json.dumps(update)


@pytest.mark.parametrize("status", ["edited", "checked", "matched", "planned", "failed", None])
def test_plan_refuses_an_order_that_was_not_approved(status):
    state = {**approved(line("타이레놀정500mg")), "status": status}
    with pytest.raises(ValueError, match="승인되지 않은"):
        plan_node(state, deps())


def test_a_plan_beyond_the_tray_is_a_failure_with_the_range_code():
    long = DrugLine(item_code=None, name="타이레놀정500mg", printed_code=None, count_per_dose=1, times=["morning"], days=7)
    state = approved(long)
    update = plan_node(state, deps())
    assert (update["status"], update["plan"], update["error_code"]) == ("failed", None, E_PLAN_RANGE)
    assert update["errors"] == [f"[plan] E_PLAN_RANGE {update['notice']}"]
    assert "order" not in update                                     # 조제지시서는 그대로 남는다


def test_a_line_without_a_drug_cannot_be_planned():
    update = plan_node(approved(line("타이레놀정500mg"), line("가나다정")), deps())
    assert (update["status"], update["error_code"]) == ("failed", None)
    assert "약을 정하지 못한 줄이 있습니다: 2번째 약품(가나다정)" in update["notice"]


def test_a_cell_with_more_than_two_pills_is_a_failure_with_the_range_code():
    two = DrugLine(item_code=None, name="뮤코다인캡슐200mg", printed_code=None, count_per_dose=2, times=["morning"], days=1)
    update = plan_node(approved(two, line("프론크논캡슐")), deps())
    assert (update["status"], update["plan"], update["error_code"]) == ("failed", None, E_PLAN_RANGE)
    assert update["notice"].endswith("트레이 칸당 최대 2알을 넘습니다: 1일 아침 3알")


def test_plan_does_not_change_the_state_it_was_given():
    state = approved(line("타이레놀정500mg"), line("게보린정"))
    before = copy.deepcopy(state)
    plan_node(state, deps())
    assert state == before


# ------------------------------------------------------------ 다른 흐름의 자리


def choose(value):
    return Pharmacist(lambda payload: {"choice": value})


def test_ask_type_shows_the_question_the_supervisor_left_and_the_photo_kinds():
    state = merged(initial_state("P0001"), supervisor_node(initial_state("P0001"), deps()))
    pharmacist = choose("tray")
    ask_type_node(state, deps(ask=pharmacist))
    [shown] = pharmacist.seen
    assert shown == {
        "hitl_point": "type_select", "about": "photo", "choices": ["prescription", "med_bag", "tray"],
        "labels": {"prescription": "처방전", "med_bag": "약 포장", "tray": "트레이"}, "notice": state["notice"],
    }
    assert "사진 종류 정보가 없습니다." in shown["notice"]


def test_ask_type_has_a_question_of_its_own_when_none_was_left():
    pharmacist = choose("tray")
    ask_type_node(initial_state("P0001"), deps(ask=pharmacist))
    assert pharmacist.seen[0]["notice"] == ASK_TYPE_NOTICE


@pytest.mark.parametrize("kind, route", [("prescription", ["prescription"]), ("med_bag", ["recheck"]), ("tray", ["tray"])])
def test_the_photo_kind_the_pharmacist_chose_decides_the_route(kind, route):
    update = ask_type_node(initial_state("P0001"), deps(ask=choose(kind)))
    assert (update["route"], update["flow"], update["photo_kind"], update["notice"]) == (route, route[0], kind, None)


def test_the_chosen_kind_wins_over_a_disagreeing_ocr():
    state = initial_state("P0001", {"doc_type": "med_bag", "lines": []}, "prescription")
    assert ask_type_node(state, deps(ask=choose("prescription")))["route"] == ["prescription"]


def test_ask_type_asks_about_the_text_when_only_text_came():
    state = initial_state("P0001", None, None, "오후 수령 예정")
    pharmacist = choose("question")
    update = ask_type_node(state, deps(ask=pharmacist))
    assert pharmacist.seen[0]["about"] == "text" and pharmacist.seen[0]["choices"] == ["current_med", "question", "memo"]
    assert update["route"] == ["consult"] and "photo_kind" not in update


def test_choosing_memo_for_the_text_ends_with_noted():
    update = ask_type_node(initial_state("P0001", None, None, "오후 수령 예정"), deps(ask=choose("memo")))
    assert (update["route"], update["status"], update["memo"]) == ([], "noted", "오후 수령 예정")


@pytest.mark.parametrize("reply", [None, {}, {"choice": "receipt"}, {"choice": "question"}, "tray"])
def test_a_choice_that_cannot_be_chosen_is_refused(reply):
    with pytest.raises(ValueError, match="고를 수 있는 값이 아닙니다"):
        ask_type_node(initial_state("P0001"), deps(ask=lambda payload: reply))


def test_the_flows_that_are_not_built_yet_say_so():
    for node, flow in ((register_meds_node, "register_meds"), (recheck_node, "recheck"), (tray_node, "tray"), (consult_node, "consult")):
        assert node(initial_state("P0001"), deps()) == {
            "notes": [NOT_READY_NOTICES[flow]], "status": "not_ready", "notice": NOT_READY_NOTICES[flow],
        }
    assert len(set(NOT_READY_NOTICES.values())) == 4


def test_a_flow_that_is_not_built_yet_does_not_hide_what_the_prescription_did():
    state = {**initial_state("P0001"), "status": "planned", "notice": "조제 계획을 만들었습니다."}
    assert consult_node(state, deps()) == {"notes": [NOT_READY_NOTICES["consult"]]}


# ------------------------------------------------------------ 공통


NODE_FUNCTIONS = [
    supervisor_node, structure_node, match_node, risk_node, approve_node, register_meds_node, recheck_node, tray_node, consult_node,
]


@pytest.mark.parametrize("node", NODE_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_node_does_not_change_the_state_it_was_given(node):
    state = checked("rx_001")
    before = copy.deepcopy(state)
    node(state, deps())
    assert state == before


@pytest.mark.parametrize("node", NODE_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_node_only_writes_fields_the_state_has(node):
    state = checked("rx_001")
    assert set(node(state, deps())) <= set(typing.get_type_hints(GraphState))


def test_the_tools_cannot_be_swapped_while_the_graph_runs():
    with pytest.raises(dataclasses.FrozenInstanceError):
        deps().api = lambda term: []


def test_a_sample_goes_through_the_nodes_in_order_to_the_plan():
    state = initial_state("P0001", load("rx8_a1", "rx8_01"))
    for node in (supervisor_node, structure_node, match_node, risk_node, approve_node):
        state = merged(state, node(state, deps()))
    assert (state["flow"], state["status"], state["error_code"], state["errors"]) == ("prescription", "approved", None, [])
    assert state["plan"] is None
    state = merged(state, plan_node(state, deps()))
    assert state["status"] == "planned" and len(state["plan"]["plan"]) == 9
    json.dumps(state)
