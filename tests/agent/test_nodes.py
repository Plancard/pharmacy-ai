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
from agent.dur import COMBINATION_TABOO, EFFICACY_DUP, find_all
from fake_dur import KETOROLAC, KETOROLAC_NAME, FakeDur
from agent.mcp_client import McpToolError
from agent.nodes import (
    approval_notice,
    MCP_DUR_FAILED_WARNING,
    MCP_FAILED_WARNING,
    APPROVED_NOTICE,
    EDITED_NOTICE,
    HITL_APPROVAL,
    NO_API_WARNING,
    Deps,
    approve_node,
    match_node,
    plan_node,
    read_reply,
    risk_node,
    structure_node,
    router_node,
    tray_node,
)
from agent.pipeline import structure_prescription
from agent.risk import NO_DUR_WARNING, UNCHECKED_TAIL
from agent.schema import DispensingOrder, DrugLine, Plan, RiskCheck, RiskFinding
from agent.state import GraphState
from graph_calls import first_state, send

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
    return {**first_state("P0001"), "order": order, **extra}


# ------------------------------------------------------------ 라우터: 사진 종류로 흐름 정하기


# ------------------------------------------------------------ 구조화


def test_structure_makes_the_order_from_a_sample():
    state = first_state("P0001", load("cv_a1", "rx_001"))
    update = structure_node(state, deps())
    assert set(update) == {"order", "status", "notice", "failed_line_ids"} and update["failed_line_ids"] == []
    assert update["status"] == "structured" and update["notice"] is None
    order = DispensingOrder.model_validate(update["order"])
    assert order.prescription_id == "P0001" and len(order.items) == 2
    json.dumps(update)                                              # 체크포인터가 저장할 수 있는 값이다


def test_structure_hands_the_id_and_the_ocr_to_the_structure_function():
    seen = []

    def spy(prescription_id, ocr):
        seen.append((prescription_id, ocr))
        return DispensingOrder.model_validate(order_of(line("게보린정")))

    from agent.structure import agent_view

    ocr = load("cv_a1", "rx_002")
    structure_node(first_state("P0007", ocr), deps(structure=spy))
    assert seen == [("P0007", agent_view(ocr))]                    # 상태에는 표 밖 글자를 비운 OCR이 들어 있다


@pytest.mark.parametrize("code", [E_OCR_EMPTY, E_LLM, None])
def test_a_structure_failure_is_recorded_in_the_state(code):
    def fails(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다.", code=code)

    update = structure_node(first_state("P0001", {}), deps(structure=fails))
    assert update == {
        "order": None, "status": "failed", "notice": "자동 구조화에 실패했습니다.", "error_code": code,
        "errors": [f"[structure] {code or '-'} 자동 구조화에 실패했습니다."], "failed_line_ids": [],   # 표가 없는 OCR
    }


def test_a_structure_failure_points_at_the_drug_table_on_the_photo():
    """11/17 회의 Q6: LLM까지 실패하면 웹이 처방전 사진에서 약품 표 칸을 강조한다 (원본 좌표)."""
    def fails(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다.", code=E_LLM)

    ocr = load("cv_a1", "rx_002")
    update = structure_node(first_state("P0001", ocr), deps(structure=fails))
    ids = update["failed_line_ids"]
    by_id = {l["line_id"]: l for l in ocr["lines"]}
    texts = [by_id[i]["text"] for i in ids]
    assert ids and all(isinstance(i, int) for i in ids)
    assert any("알모란" in t for t in texts)                                     # 약 줄이 들어 있다
    assert not any("주사제" in t for t in texts)                                 # 표 밖은 아니다


def test_a_state_without_ocr_fails_with_the_empty_ocr_code():
    update = structure_node(first_state("P0001", None), deps())
    assert update["status"] == "failed" and update["error_code"] == E_OCR_EMPTY and update["order"] is None


def test_a_config_problem_is_not_turned_into_a_failure_of_one_prescription():
    def broken(prescription_id, ocr):
        raise ConfigError("OPENAI_API_KEY가 없습니다.")

    with pytest.raises(ConfigError):
        structure_node(first_state("P0001", {}), deps(structure=broken))


# ------------------------------------------------------------ 약품 매칭


def structured(sample_id):
    return merged(first_state("P0001", load("cv_a1", sample_id)), structure_node(first_state("P0001", load("cv_a1", sample_id)), deps()))


def matched(sample_id):
    state = structured(sample_id)
    return merged(state, match_node(state, deps()))


def checked(sample_id):
    state = matched(sample_id)
    return merged(state, risk_node(state, deps()))


def with_plan(state):
    """그래프 순서대로 조제 계획까지 만든 상태 (v0.8: 계획은 승인 전에 만든다)."""
    return merged(state, plan_node(state, deps()))


def ready(*lines):
    """약품 매칭 · 위험 신호 대조 · 조제 계획까지 마쳐 승인 대기에 들어갈 상태."""
    state = after_match(*lines)
    return with_plan(merged(state, risk_node(state, deps())))


def ready_order(order):
    """주어진 조제지시서(위험 신호 대조 포함)에 조제 계획을 붙인 상태."""
    return with_plan(state_with(order))


def test_match_finds_the_item_codes_and_marks_the_lines_to_check():
    update = match_node(structured("rx_001"), deps())
    assert set(update) == {"order", "status", "changed_items"} and update["status"] == "matched"
    assert update["changed_items"] is None                             # 처음 매칭 (고친 것 아님)
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

    def __init__(self, *, api_used=False, fail=None, dur=None):
        self.calls, self.api_used, self.fail, self.dur = 0, api_used, fail, dur

    def match_drug(self, order):
        self.calls += 1
        if self.fail:
            raise McpToolError(self.fail)
        return match_order(order, DICT), self.api_used

    def check_dur(self, items):
        self.calls += 1
        if self.fail:
            raise McpToolError(self.fail)
        return find_all(items, DICT, self.dur)


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


def asked(state, answer=approve_as_shown):
    """승인 대기에서 약사에게 보인 화면과 approve_node가 돌려준 값."""
    pharmacist = Pharmacist(answer)
    update = approve_node(state, deps(ask=pharmacist))
    [payload] = pharmacist.seen
    return payload, update


def test_approve_shows_the_order_the_plan_the_ocr_lines_and_a_notice():
    state = with_plan(checked("rx_001"))
    payload, _ = asked(state, edit(lambda order: None))
    assert set(payload) == {"hitl_point", "order", "plan", "error_code", "ocr_lines", "ocr_size", "notice", "changed_items"}
    assert payload["changed_items"] == []
    assert payload["hitl_point"] == HITL_APPROVAL == "rx_approval"
    assert payload["order"] == state["order"]
    used = {i for item in state["order"]["items"] for i in item["source_line_ids"]}
    assert set(payload["ocr_lines"]) == {str(i) for i in used}
    some = next(iter(payload["ocr_lines"].values()))
    assert set(some) == {"text", "conf", "bbox"}                         # bbox: 원본 이미지 픽셀 좌표 (F-WEB-06)
    json.dumps(payload)


def test_approve_says_how_many_lines_need_a_check():
    payload, _ = asked(ready(line("타이레놀정500mg"), line("뮤코다인캡슐200mg")))
    assert payload["notice"] == "약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다."


def test_the_approval_card_shows_the_plan_made_before_approval():
    payload, _ = asked(ready(line("타이레놀정500mg"), line("게보린정")), manual_with("확인함"))
    plan = Plan.model_validate(payload["plan"])
    assert len(plan.plan) == 3 and payload["error_code"] is None


def test_without_a_plan_the_card_shows_the_code_and_only_an_edit_goes_on():
    """3일 초과(E_PLAN_RANGE)면 계획 없이 승인 카드를 보이고, 그대로 승인 · 사유 진행은 받지 않는다."""
    from agent.nodes import NO_PLAN_CANNOT_APPROVE

    long = DrugLine(item_code=None, name="타이레놀정500mg", printed_code=None, count_per_dose=1, times=["morning"], days=4)
    state = ready(long)
    with pytest.raises(ValueError) as error:
        asked(state)
    assert str(error.value) == NO_PLAN_CANNOT_APPROVE
    with pytest.raises(ValueError):
        asked(state, manual_with("확인함"))

    def three_days(order):
        order["items"][0]["days"] = 3

    payload, update = asked(state, edit(three_days))
    assert (payload["plan"], payload["error_code"]) == (None, E_PLAN_RANGE)
    assert update["status"] == "edited" and update["plan"] is None


def test_a_line_without_an_item_code_cannot_be_approved_or_passed_with_a_reason():
    from agent.nodes import UNMATCHED_CANNOT_APPROVE

    state = with_plan(checked("rx_001"))                              # 알모란정은 품목기준코드를 정하지 못했다
    for answer in (approve_as_shown, manual_with("확인함")):
        with pytest.raises(ValueError) as error:
            asked(state, answer)
        assert str(error.value) == UNMATCHED_CANNOT_APPROVE


def test_approve_has_nothing_to_add_when_no_line_needs_a_check():
    # 인쇄된 코드가 사전의 품목기준코드와 같고 이름도 맞으면 자동으로 확정된다 (약사 확인 없음)
    state = state_with(order_of(line("타이레놀정500mg", printed_code=TYLENOL)))
    state = merged(state, match_node(state, deps()))
    state = with_plan(merged(state, risk_node(state, deps())))
    assert asked(state)[0]["notice"] == "약사 승인을 기다립니다."


def test_approve_says_a_warning_must_be_confirmed_before_approval():
    state = ready(line("타이레놀정500mg"), line("게보린정"))
    assert asked(state, manual_with("확인함"))[0]["notice"] == (
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
    notice = asked(ready_order(order_with_risk(level)), manual_with("확인함"))[0]["notice"]
    assert f"위험 신호가 1건 있습니다({label})." in notice
    assert ("약사가 확인한 뒤에 승인할 수 있습니다." in notice) is confirm


def manual_with(reason):
    return lambda payload: {"action": "manual", "order": payload["order"], "reason": reason}


def test_a_danger_cannot_be_approved_as_shown():
    """명세서 2.8절 · 기능분석서 F-WEB-07: 위험은 고치거나 사유를 적어야 진행한다."""
    from agent.nodes import DANGER_NEEDS_REASON

    with pytest.raises(ValueError) as error:
        asked(ready_order(order_with_risk("danger")))
    assert str(error.value) == DANGER_NEEDS_REASON
    with pytest.raises(ValueError):
        read_reply({"action": "approve", "order": order_with_risk("danger")}, DispensingOrder.model_validate(order_with_risk("danger")))


@pytest.mark.parametrize("level", ["none", "caution", "warning", "danger"])
def test_manual_with_a_reason_goes_on_and_keeps_the_reason(level):
    state = ready_order(order_with_risk(level)) if level != "none" else ready(line("타이레놀정500mg"))
    _, update = asked(state, manual_with("  처방의와 통화해 확인함 "))
    assert update["status"] == "planned"
    assert update["notice"] == "약사가 사유를 적고 진행했습니다. 사유: 처방의와 통화해 확인함"
    assert update["notes"] == ["위험 신호 확인 후 진행 사유: 처방의와 통화해 확인함"]


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_manual_without_a_reason_is_refused(reason):
    order = order_with_risk("danger")
    with pytest.raises(ValueError, match="사유"):
        read_reply({"action": "manual", "order": order, "reason": reason}, DispensingOrder.model_validate(order))


def test_a_danger_order_that_was_changed_is_checked_again():
    order = order_with_risk("danger")
    changed = {**order, "items": order["items"][:1]}
    action, answered = read_reply({"action": "approve", "order": changed}, DispensingOrder.model_validate(order))
    assert action == "approve" and len(answered.items) == 1             # 고친 승인은 다시 대조로 간다 (approve_node가 edited로)


def test_approve_refuses_an_order_that_skipped_the_risk_check():
    with pytest.raises(ValueError, match="risk_check"):
        approve_node(matched("rx_001"), deps())


def test_approving_the_order_as_shown_confirms_the_plan_on_the_card():
    state = ready(line("타이레놀정500mg"))
    _, update = asked(state)
    assert update == {"status": "planned", "notice": APPROVED_NOTICE, "error_code": None,   # 계획은 새로 만들지 않는다
                      "previous_order": None, "changed_items": None}


def edit(change):
    """약사가 조제지시서를 고쳐서 돌려준다. change(order dict)가 그 자리에서 고친다."""
    def answer(payload):
        order = copy.deepcopy(payload["order"])
        change(order)
        return {"action": "edit", "order": order}
    return answer


def test_editing_puts_the_new_lines_in_and_asks_for_matching_and_checking_again():
    state = ready(line("타이레놀정500mg"), line("게보린정"))

    def two_days(order):
        order["items"][0]["days"] = 2

    update = approve_node(state, deps(ask=edit(two_days)))
    assert set(update) == {"order", "plan", "status", "notice", "error_code", "previous_order"}
    assert update["previous_order"] == state["order"]                # 고치기 직전 (안 바뀐 줄 · 설명을 다시 쓴다, Q7)
    assert update["plan"] is None                                    # 고친 뒤에는 계획도 다시 만든다
    assert (update["status"], update["notice"], update["error_code"]) == ("edited", EDITED_NOTICE, None)
    assert update["order"]["items"][0]["days"] == 2
    assert update["order"]["risk_check"] is None                     # 고친 뒤에는 다시 대조한다
    assert update["order"]["issue_no"] == state["order"]["issue_no"]   # 지시서의 나머지는 그대로


def test_approving_with_changed_lines_counts_as_an_edit():
    state = with_plan(checked("rx_001"))

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
    state = {**checked("rx_001"), "plan": None, "notice": "조제 계획을 만들 수 없습니다. 이유."}
    assert asked(state, edit(lambda order: None))[0]["notice"].startswith("조제 계획을 만들 수 없습니다. 이유. 고친 뒤 다시 승인해 주세요. 약사 승인을 기다립니다.")


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


# ------------------------------------------------------------ 조제 계획 (승인 전, v0.8)


def risk_checked(*lines):
    """약품 매칭 · 위험 신호 대조를 마친 상태 (조제 계획 바로 앞)."""
    state = after_match(*lines)
    return merged(state, risk_node(state, deps()))


def test_plan_makes_the_a4_plan_before_the_approval():
    state = risk_checked(line("타이레놀정500mg"), line("게보린정"))
    update = plan_node(state, deps())
    assert set(update) == {"plan", "notice", "error_code"}            # 상태는 checked 그대로 (승인 전)
    assert (update["notice"], update["error_code"]) == (None, None)
    plan = Plan.model_validate(update["plan"])
    assert [(c.day, c.time, [(i.item_code, i.count) for i in c.items]) for c in plan.plan] == [
        (day, "morning", [(TYLENOL, 1), (GEVORIN, 1)]) for day in (1, 2, 3)
    ]
    json.dumps(update)


def test_plan_refuses_an_order_that_skipped_the_risk_check():
    state = after_match(line("타이레놀정500mg"))
    with pytest.raises(ValueError, match="위험 신호 대조"):
        plan_node(state, deps())


def test_a_plan_beyond_the_tray_is_a_failure_with_the_range_code():
    long = DrugLine(item_code=None, name="타이레놀정500mg", printed_code=None, count_per_dose=1, times=["morning"], days=7)
    update = plan_node(risk_checked(long), deps())
    assert (update["plan"], update["error_code"]) == (None, E_PLAN_RANGE)
    assert update["errors"] == [f"[plan] E_PLAN_RANGE {update['notice']}"]
    assert "order" not in update and "status" not in update          # 조제지시서와 상태는 그대로, 승인 대기로 간다


def test_a_line_without_a_drug_cannot_be_planned():
    update = plan_node(risk_checked(line("타이레놀정500mg"), line("가나다정")), deps())
    assert (update["plan"], update["error_code"]) == (None, None)
    assert "약을 정하지 못한 줄이 있습니다: 2번째 약품(가나다정)" in update["notice"]


def test_a_cell_can_hold_more_than_two_pills():
    """11/12 결정: 칸당 알 수 제한은 없앴다."""
    two = DrugLine(item_code=None, name="뮤코다인캡슐200mg", printed_code=None, count_per_dose=2, times=["morning"], days=1)
    update = plan_node(risk_checked(two, line("프론크논캡슐")), deps())
    assert update["error_code"] is None and update["plan"] is not None


def test_plan_does_not_change_the_state_it_was_given():
    state = risk_checked(line("타이레놀정500mg"), line("게보린정"))
    before = copy.deepcopy(state)
    plan_node(state, deps())
    assert state == before


# ------------------------------------------------------------ 라우터 (v0.8: 처방전 / 트레이)


def test_router_sends_a_prescription_photo_to_the_prescription_flow():
    got = router_node(first_state("P1", load("cv_a1", "rx_001")), deps())
    assert (got["route"], got["flow"]) == (["prescription"], "prescription") and "status" not in got


def test_router_reads_a_prescription_from_the_ocr_when_no_kind_is_given():
    got = router_node(first_state("P1", load("cv_a1", "rx_001"), photo_kind=None), deps())
    assert got["route"] == ["prescription"]


def test_router_fills_the_prescription_id_when_the_web_left_it_out():
    state = {**first_state("P1", load("cv_a1", "rx_001")), "prescription_id": None}
    assert router_node(state, deps())["prescription_id"] == "P1"


def test_router_ends_with_noted_and_a_note_when_nothing_can_run():
    got = router_node(first_state("P1", None, photo_kind=None), deps())
    assert got["route"] == [] and got["status"] == "noted" and got["notes"] == [got["notice"]]


def test_router_keeps_the_status_of_an_open_prescription_when_a_second_prescription_photo_comes():
    state = {**checked("rx_001"), "status": "planned"}
    got = router_node({**state, "photo_kind": "prescription"}, deps())
    assert got["route"] == [] and got["status"] == "planned" and "새 처방" in got["notice"]


# ------------------------------------------------------------ 공통


NODE_FUNCTIONS = [
    router_node, structure_node, match_node, risk_node, plan_node, approve_node, tray_node,
]


@pytest.mark.parametrize("node", NODE_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_node_does_not_change_the_state_it_was_given(node):
    state = ready(line("타이레놀정500mg"), line("게보린정"))
    before = copy.deepcopy(state)
    node(state, deps())
    assert state == before


@pytest.mark.parametrize("node", NODE_FUNCTIONS, ids=lambda f: f.__name__)
def test_a_node_only_writes_fields_the_state_has(node):
    state = ready(line("타이레놀정500mg"), line("게보린정"))
    assert set(node(state, deps())) <= set(typing.get_type_hints(GraphState))


def test_the_tools_cannot_be_swapped_while_the_graph_runs():
    with pytest.raises(dataclasses.FrozenInstanceError):
        deps().api = lambda term: []


def test_a_sample_goes_through_the_nodes_in_order_to_the_plan():
    state = first_state("P0001", load("rx8_a1", "rx8_01"))
    for node in (router_node, structure_node, match_node, risk_node, plan_node):
        state = merged(state, node(state, deps()))
    assert (state["flow"], state["status"], state["error_code"], state["errors"]) == ("prescription", "checked", None, [])
    shown = state["plan"]
    assert len(shown["plan"]) == 9                                    # 승인 전에 만든 계획
    state = merged(state, approve_node(state, deps(ask=manual_with("확인함"))))
    assert state["status"] == "planned" and state["plan"] == shown   # 승인은 보인 계획을 확정만 한다
    json.dumps(state)



# ------------------------------------------------------------ 위험 신호: DUR 조회 (13단계)
def dur_deps(dur):
    return dataclasses.replace(deps(), dur=dur)


def coded(*pairs):
    """품목기준코드가 정해진 줄들 (사전 밖의 약도 쓸 수 있다)."""
    return state_with(DispensingOrder(prescription_id="P0001", items=[
        DrugLine(item_code=c, name=n, printed_code=c, count_per_dose=1, times=["morning"], days=3, match_method="code")
        for c, n in pairs]).model_dump(mode="json"))


def test_risk_with_dur_finds_a_combination_taboo_as_danger():
    state = coded((TRISPEN, "트리스펜연질캡슐"), (KETOROLAC, KETOROLAC_NAME))
    order = DispensingOrder.model_validate(risk_node(state, dur_deps(FakeDur()))["order"])
    assert [f.type for f in order.risk_check.findings] == [COMBINATION_TABOO]
    assert order.risk_check.level == "danger"
    assert "약사가 확인한 뒤에 승인할 수 있습니다" in approval_notice(order, {})


def test_risk_with_dur_finds_an_efficacy_duplicate_as_warning():
    order = DispensingOrder.model_validate(
        risk_node(after_match(line("타이레놀정500mg"), line("트리스펜연질캡슐")), dur_deps(FakeDur()))["order"])
    assert [(f.type, f.level) for f in order.risk_check.findings] == [(EFFICACY_DUP, "warning")]
    assert NO_DUR_WARNING not in order.warnings


def test_risk_without_dur_says_taboos_were_not_checked():
    order = DispensingOrder.model_validate(risk_node(after_match(line("타이레놀정500mg")), deps())["order"])
    assert NO_DUR_WARNING in order.warnings


def test_risk_through_mcp_tools_is_the_same_as_the_function():
    state = coded((TRISPEN, "트리스펜연질캡슐"), (KETOROLAC, KETOROLAC_NAME), (TYLENOL, "타이레놀정500mg"))
    tools = FakeTools(dur=FakeDur())
    assert risk_node(state, tools_deps(tools)) == risk_node(state, dur_deps(FakeDur()))
    assert tools.calls == 1


def test_when_the_mcp_dur_tool_fails_the_node_checks_by_itself_and_says_so():
    state = after_match(line("타이레놀정500mg"), line("게보린정"))
    order = DispensingOrder.model_validate(risk_node(state, tools_deps(FakeTools(fail="연결이 끊겼습니다")))["order"])
    assert order.risk_check.level == "warning"                         # 성분 중복은 직접 찾았다
    assert f"{MCP_DUR_FAILED_WARNING}: 연결이 끊겼습니다" in order.warnings




# ------------------------------------------------------------ 수정 후 바뀐 항목만 다시 (11/17 회의 Q7)


def _edited(state, change):
    """승인 카드에서 약사가 고친 뒤의 상태 (approve_node가 돌려준 칸을 합친 것)."""
    return merged(state, approve_node(state, deps(ask=edit(change))))


def test_only_the_changed_lines_are_matched_again(monkeypatch):
    import agent.nodes as nodes

    state = ready(line("타이레놀정500mg"), line("게보린정"))
    before = state["order"]["items"]
    seen = []
    real = nodes.match_order
    monkeypatch.setattr(nodes, "match_order", lambda order, *a, **k: seen.append([d.name for d in order.items]) or real(order, *a, **k))

    def two_days(order):
        order["items"][1]["days"] = 2

    update = match_node(_edited(state, two_days), deps())
    assert seen == [["게보린정"]]                                              # 바뀐 줄만 다시 매칭한다
    items = update["order"]["items"]
    assert items[0] == before[0]                                                # 안 바뀐 줄은 그대로
    assert items[1]["item_code"] == GEVORIN and items[1]["days"] == 2
    assert update["notes"] == ["약사 수정 후 다시 검수한 줄: 2번째(게보린정)"]
    assert update["changed_items"] == [1]                                       # 승인 카드가 표시할 줄 번호 (0부터)
    from agent.nodes import approval_payload
    again = merged(_edited(state, two_days), update)
    again = with_plan(merged(again, risk_node(again, deps())))
    assert approval_payload(again)["changed_items"] == [1]


def test_deleting_a_line_matches_nothing_again(monkeypatch):
    import agent.nodes as nodes

    state = ready(line("타이레놀정500mg"), line("게보린정"))
    monkeypatch.setattr(nodes, "match_order", lambda *a, **k: pytest.fail("다시 매칭하면 안 됩니다"))
    update = match_node(_edited(state, lambda order: order["items"].pop(1)), deps())
    assert [d["name"] for d in update["order"]["items"]] == ["타이레놀정500mg"]
    assert update["notes"] == ["약사 수정 후 다시 검수한 줄: 없음 (지운 줄만 있음)"]


def test_a_risk_explained_before_the_edit_is_not_explained_again():
    from agent.nodes import Deps

    state = ready(line("타이레놀정500mg"), line("게보린정"))
    finding = state["order"]["risk_check"]["findings"][0]
    finding.update(explanation="두 약 모두 아세트아미노펜이 들어 있습니다.",
                   evidence=[{"title": "타이레놀정500mg · 주의사항", "source": "e약은요", "excerpt": "…"}])

    def two_days(order):
        order["items"][1]["days"] = 2

    after = _edited(state, two_days)
    after = merged(after, match_node(after, deps()))
    never = Deps(dictionary=DICT, structure=real_structure, explain=lambda *a: pytest.fail("다시 설명하면 안 됩니다"))
    again = risk_node(after, never)["order"]["risk_check"]["findings"]
    assert again[0]["explanation"] == finding["explanation"] and again[0]["evidence"] == finding["evidence"]


def test_a_new_risk_after_the_edit_is_explained():
    from agent.nodes import Deps

    state = ready(line("타이레놀정500mg"), line("뮤코다인캡슐200mg"))
    assert state["order"]["risk_check"]["findings"] == []

    def add_gevorin(order):
        order["items"][1].update(name="게보린정", item_code=None, match_method=None, match_candidates=[])

    after = _edited(state, add_gevorin)
    after = merged(after, match_node(after, deps()))
    asked = []
    explain = lambda text: asked.append(text) or {"items": [{"index": 1, "text": "아세트아미노펜이 겹칩니다. 하루 총량을 확인하세요."}]}
    found = risk_node(after, Deps(dictionary=DICT, structure=real_structure, explain=explain))["order"]["risk_check"]["findings"]
    assert len(asked) == 1 and found[0]["type"] == "ingredient_dup"


def test_approving_clears_the_order_before_the_edit():
    state = ready(line("타이레놀정500mg"))
    state["previous_order"] = state["order"]
    assert approve_node(state, deps())["previous_order"] is None


# ------------------------------------------------------------ 실패 위치: 규칙이 실패한 줄만 · 머리글 없으면 빈 목록 (11/18 결정)


def _without(ocr, *drop):
    """A.1에서 몇 칸을 지우고 line_id를 0부터 다시 매긴다 (돌려준 두 번째 값: 옛 line_id -> 새 line_id)."""
    kept = [l for l in ocr["lines"] if l["line_id"] not in drop]
    renum = {l["line_id"]: i for i, l in enumerate(kept)}
    return {**ocr, "lines": [{**l, "line_id": i} for i, l in enumerate(kept)]}, renum


def _llm_also_fails(prescription_id, ocr):
    from agent.pipeline import structure_prescription

    def broken(text, feedback):
        raise StructureError("응답 없음")
    return structure_prescription(prescription_id, ocr, llm_fn=broken)


def test_when_the_rules_fail_on_one_drug_line_only_that_line_is_marked():
    ocr, renum = _without(load("rx8_a1", "rx8_01"), 67)              # 2번째 약(게보린정) 줄의 '3회' 칸을 지운다
    update = structure_node(first_state("P0001", ocr), deps(structure=_llm_also_fails))
    assert update["error_code"] == E_LLM and "2번째 약품 줄" in update["notice"]
    assert update["failed_line_ids"] == sorted(renum[i] for i in (66, 68, 69, 70))   # 게보린정 줄의 남은 칸들만
    assert renum[61] not in update["failed_line_ids"]                                  # 1번째 약 줄은 아니다


def test_without_the_table_header_no_line_is_marked():
    ocr, _ = _without(load("rx8_a1", "rx8_01"), 54, 57, 58, 59, 60)    # 머리글(명칭 · 투약량 · 횟수 · 일수 · 구분코드)을 지운다
    update = structure_node(first_state("P0001", ocr), deps(structure=_llm_also_fails))
    assert update["status"] == "failed" and update["failed_line_ids"] == []          # 줄을 특정할 수 없다 (요청서 1번)


def test_a_stray_cell_in_the_table_is_the_failed_spot():
    from agent.structure import extract_rule

    ocr = load("rx8_a1", "rx8_01")
    stray = {"line_id": len(ocr["lines"]), "text": "얼룩", "bbox": [1000, 1430, 1100, 1460], "conf": 0.5}
    with pytest.raises(StructureError) as error:
        extract_rule({**ocr, "lines": [*ocr["lines"], stray]})
    assert "어느 약품 줄에도 속하지 않는" in str(error.value) and error.value.line_ids == [stray["line_id"]]
