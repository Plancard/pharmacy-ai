"""위험 신호 대조 결과 만들기 (agent/risk.py)."""
import copy
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import dictionary_from_data, load_dictionary
from agent.dur import (
    COMBINATION_TABOO,
    DUR_FAILED,
    EFFICACY_DUP,
    INGREDIENT_DUP,
    NO_INGREDIENTS,
    NO_ITEM_CODE,
    TABOO_PARTIAL,
    DurHit,
    DurReport,
    Unchecked,
)
from agent.risk import (
    CONFIRM_LEVELS,
    LEVEL_BY_TYPE,
    LEVEL_LABELS,
    DUR_UNCHECKED_TAIL,
    NO_DUR_WARNING,
    TABOO_PARTIAL_TAIL,
    TYPE_LABELS,
    UNCHECKED_TAIL,
    apply_report,
    check_order,
    combination_taboo_message,
    efficacy_dup_message,
    finding_from_hit,
    finding_level,
    ingredient_dup_message,
    is_unchecked_warning,
    requires_confirmation,
    risk_check_from_hits,
    unchecked_warning,
)
from agent.schema import RISK_LEVELS, DispensingOrder, DrugLine

SAMPLES = RESOURCES_DIR / "samples"
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL, GEVORIN, TRISPEN, ARPEN = "202106092", "197900277", "201110120", "200201239"

BLANK = "100000009"
# 주성분 칸이 빈 약이 든 사전 (실제 사전의 8종은 모두 채워져 있다)
WITH_BLANK = dictionary_from_data({"drugs": [
    {"item_code": TYLENOL, "name": "타이레놀정500mg", "ingredients": ["아세트아미노펜"]},
    {"item_code": BLANK, "name": "성분모름정", "ingredients": []},
]})

HIT_A = DurHit(INGREDIENT_DUP, (TYLENOL, GEVORIN), ("타이레놀정500mg", "게보린정"), ("아세트아미노펜",))
HIT_B = DurHit(INGREDIENT_DUP, (TRISPEN, ARPEN), ("트리스펜연질캡슐", "아르펜정"), ("이부프로펜",))


class EmptyDur:
    """DUR 조회 대신: 어느 약도 병용금기 · 효능군이 없다."""

    def taboo(self, item_code):
        from agent.dur_api import TabooList
        return TabooList((), True)

    def efficacy(self, item_code):
        return ()


EMPTY = EmptyDur()


def line(code, name):
    return DrugLine(item_code=code, name=name, printed_code=None, count_per_dose=1, times=["morning"], days=3)


def order_of(*lines):
    return DispensingOrder(prescription_id="P0001", items=list(lines))


# ------------------------------------------------------------ 단계 규칙 (명세서 2.8절)


def test_the_levels_follow_the_spec_table():
    """명세서 2.8절: 병용금기는 위험, 성분 · 효능군 중복은 경고. 1차는 이 셋만 판정한다."""
    assert LEVEL_BY_TYPE == {"combination_taboo": "danger", "ingredient_dup": "warning", "efficacy_dup": "warning"}
    assert finding_level(COMBINATION_TABOO) == "danger" and finding_level(INGREDIENT_DUP) == "warning"
    assert set(TYPE_LABELS) == set(LEVEL_BY_TYPE)


@pytest.mark.parametrize("unknown", ["contraindication", "age_caution", "", "INGREDIENT_DUP"])
def test_a_type_without_a_rule_is_not_guessed(unknown):
    with pytest.raises(ValueError, match="단계 규칙이 없는 DUR 유형"):
        finding_level(unknown)


@pytest.mark.parametrize("level, confirm", [("none", False), ("caution", False), ("warning", True), ("danger", True)])
def test_from_warning_up_the_pharmacist_must_confirm(level, confirm):
    assert requires_confirmation(level) is confirm


def test_the_levels_have_labels_and_the_confirm_levels_are_levels():
    assert set(LEVEL_LABELS) == set(RISK_LEVELS)
    assert LEVEL_LABELS == {"none": "해당 없음", "caution": "주의", "warning": "경고", "danger": "위험"}
    assert set(CONFIRM_LEVELS) <= set(RISK_LEVELS)


# ------------------------------------------------------------ 항목과 risk_check


def test_the_message_names_the_drugs_and_the_ingredients():
    assert ingredient_dup_message(HIT_A) == "타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요."
    two = DurHit(INGREDIENT_DUP, ("1", "2"), ("가정", "나정"), ("성분X", "성분Y"))
    assert ingredient_dup_message(two) == "가정 · 나정에 성분X, 성분Y 성분이 함께 들어 있습니다. 하루 총량을 확인하세요."


def test_a_finding_carries_the_codes_the_level_and_the_message():
    finding = finding_from_hit(HIT_A)
    assert (finding.type, finding.item_codes, finding.level) == (INGREDIENT_DUP, [TYLENOL, GEVORIN], "warning")
    assert finding.message == ingredient_dup_message(HIT_A)


def test_no_hit_gives_none_with_no_findings():
    risk = risk_check_from_hits([])
    assert (risk.scope, risk.level, risk.findings) == ("rx_only", "none", [])


def test_hits_give_the_highest_level_of_the_findings():
    risk = risk_check_from_hits([HIT_A, HIT_B])
    assert (risk.scope, risk.level) == ("rx_only", "warning")
    assert [f.item_codes for f in risk.findings] == [[TYLENOL, GEVORIN], [TRISPEN, ARPEN]]


# ------------------------------------------------------------ 대조하지 못했다는 경고


def test_the_warning_for_a_line_without_a_code():
    text = unchecked_warning(Unchecked("알모탄정", NO_ITEM_CODE))
    assert text == "알모탄정: 품목기준코드를 정하지 못해 성분 중복을 대조하지 못했습니다."


def test_the_warning_for_a_drug_with_unknown_ingredients():
    text = unchecked_warning(Unchecked("성분모름정", NO_INGREDIENTS))
    assert text == "성분모름정: 주성분을 알 수 없어(약품 사전에 없거나 주성분 칸이 비어 있어) 성분 중복을 대조하지 못했습니다."


@pytest.mark.parametrize("reason", [NO_ITEM_CODE, NO_INGREDIENTS])
def test_every_unchecked_warning_is_recognised_by_its_tail(reason):
    assert is_unchecked_warning(unchecked_warning(Unchecked("가정", reason)))
    assert UNCHECKED_TAIL == "성분 중복을 대조하지 못했습니다."


@pytest.mark.parametrize("other", ["", "의약품 API를 쓰지 않아(DRUG_API_KEY 없음) 약품 사전(대상 8종)에서만 품목기준코드를 찾았습니다.",
                                    "1번째 약품: 복용 시점이 처방전에 없어 하루 3회 기준으로 배정했습니다."])
def test_other_warnings_are_not_taken_for_unchecked_ones(other):
    assert not is_unchecked_warning(other)


# ------------------------------------------------------------ check_order


def test_check_order_fills_the_risk_check_of_a_new_order():
    order = order_of(line(TYLENOL, "타이레놀정"), line(GEVORIN, "게보린정"))
    checked = check_order(order, DICT, EMPTY)
    assert order.risk_check is None                                  # 받은 조제지시서는 바뀌지 않는다
    assert checked.risk_check == risk_check_from_hits([HIT_A])
    assert checked.items == order.items and checked.warnings == []
    checked.check_ready_for_web()                                    # 웹에 넘겨도 되는 모양이다 (약을 다 정했으니)


def test_check_order_keeps_the_warnings_that_are_not_about_the_check():
    order = order_of(line(TYLENOL, "타이레놀정"))
    order.warnings = ["다른 경고"]
    assert check_order(order, DICT, EMPTY).warnings == ["다른 경고"]


def test_check_order_adds_a_warning_for_each_drug_it_could_not_check():
    order = order_of(line(TYLENOL, "타이레놀정"), line(None, "알모탄정"), line(BLANK, "성분 모름 정"))
    order.items[1].needs_pharmacist_check = True
    checked = check_order(order, WITH_BLANK, EMPTY)
    assert checked.warnings == [
        f"알모탄정: 품목기준코드를 정하지 못해 {UNCHECKED_TAIL}",
        f"성분모름정: 주성분을 알 수 없어(약품 사전에 없거나 주성분 칸이 비어 있어) {UNCHECKED_TAIL}",
    ]
    assert checked.risk_check.level == "none"


def test_check_order_twice_gives_the_same_order():
    order = order_of(line(TYLENOL, "타이레놀정"), line(GEVORIN, "게보린정"), line(None, "알모탄정"))
    order.items[2].needs_pharmacist_check = True
    once = check_order(order, DICT, EMPTY)
    assert check_order(once, DICT, EMPTY) == once


def test_check_order_replaces_the_last_unchecked_warnings_instead_of_adding():
    order = order_of(line(TYLENOL, "타이레놀정"), line(BLANK, "성분 모름 정"))
    order.warnings = ["다른 경고"]
    once = check_order(order, WITH_BLANK, EMPTY)
    assert len(once.warnings) == 2 and once.risk_check.level == "none"
    now_known = dictionary_from_data({"drugs": [
        {"item_code": TYLENOL, "name": "타이레놀정500mg", "ingredients": ["아세트아미노펜"]},
        {"item_code": BLANK, "name": "성분모름정", "ingredients": ["아세트아미노펜"]},
    ]})
    again = check_order(once, now_known, EMPTY)
    assert again.warnings == ["다른 경고"]                              # 이제 대조할 수 있으니 지난 경고가 사라진다
    assert again.risk_check.level == "warning"


def test_check_order_replaces_a_risk_check_made_before():
    first = check_order(order_of(line(TYLENOL, "타이레놀정"), line(GEVORIN, "게보린정")), DICT, EMPTY)
    changed = first.model_copy(update={"items": [first.items[0]]})
    assert check_order(changed, DICT, EMPTY).risk_check.level == "none"


def test_check_order_does_not_change_the_order_it_was_given():
    order = order_of(line(TYLENOL, "타이레놀정"), line(GEVORIN, "게보린정"), line(None, "알모탄정"))
    before = copy.deepcopy(order)
    check_order(order, DICT, EMPTY)
    assert order == before


def test_the_result_can_be_stored_as_json():
    checked = check_order(order_of(line(TYLENOL, "타이레놀정"), line(GEVORIN, "게보린정")), DICT)
    json.dumps(checked.model_dump(mode="json"))


# ------------------------------------------------------------ 팀 샘플(A.5)과 같은 결과인가


def a5(name):
    return json.loads((SAMPLES / "a5_order" / f"{name}.json").read_text(encoding="utf-8"))


def shape(risk):
    """문장을 뺀 모양. 샘플의 문장은 예시라서 같을 필요가 없다."""
    return risk["scope"], risk["level"], [(f["type"], f["level"], f["item_codes"]) for f in risk["findings"]]


# s04_flagged는 병용금기 · 연령주의 항목(13단계 이후의 일)이라 여기서는 다루지 않는다
@pytest.mark.parametrize("name", ["s01_combo_a", "s02_combo_b", "s03_no_risk", "s05_no_code"])
def test_the_risk_check_equals_the_one_in_the_team_sample(name):
    sample = a5(name)
    without = {k: v for k, v in sample.items() if k != "risk_check"}
    checked = check_order(DispensingOrder.model_validate(without), DICT)
    assert shape(checked.risk_check.model_dump(mode="json")) == shape(sample["risk_check"])


@pytest.mark.parametrize("name, ingredient", [("s01_combo_a", "아세트아미노펜"), ("s02_combo_b", "이부프로펜")])
def test_the_demo_combinations_say_which_ingredient_is_shared(name, ingredient):
    sample = a5(name)
    checked = check_order(DispensingOrder.model_validate({k: v for k, v in sample.items() if k != "risk_check"}), DICT)
    [finding] = checked.risk_check.findings
    assert ingredient in finding.message
    for item in checked.items:
        assert item.name in finding.message or item.name.replace(" ", "") in finding.message


# ------------------------------------------------------------ 병용금기 · 효능군 중복 · DUR 미사용 (13단계)
TABOO = DurHit(COMBINATION_TABOO, (TRISPEN, "199900001"), ("트리스펜연질캡슐", "가상케토롤락정"), ("케토롤락트로메타민",),
               "위장관 출혈 위험 증가")
EFFICACY = DurHit(EFFICACY_DUP, (TYLENOL, ARPEN), ("타이레놀정500mg", "아르펜정"), ("해열진통소염제",))


def test_a_combination_taboo_is_danger_and_shows_the_reason():
    finding = finding_from_hit(TABOO)
    assert finding.level == "danger"
    assert finding.message == "트리스펜연질캡슐 · 가상케토롤락정는 함께 쓰면 안 되는 조합입니다(병용금기). 사유: 위장관 출혈 위험 증가"
    assert combination_taboo_message(DurHit(COMBINATION_TABOO, ("1" * 9, "2" * 9), ("가", "나"), ())).endswith("(병용금기).")
    assert requires_confirmation(risk_check_from_hits([TABOO, HIT_A]).level)


def test_an_efficacy_duplicate_is_a_warning_naming_the_group():
    finding = finding_from_hit(EFFICACY)
    assert finding.level == "warning"
    assert finding.message == efficacy_dup_message(EFFICACY) == \
        "타이레놀정500mg · 아르펜정는 같은 효능군(해열진통소염제)입니다. 함께 쓸 필요가 있는지 확인하세요."


def test_without_dur_the_order_says_taboos_were_not_checked():
    checked = check_order(order_of(line(TYLENOL, "타이레놀정")), DICT)          # DUR 조회 없음
    assert checked.warnings == [NO_DUR_WARNING]
    assert is_unchecked_warning(NO_DUR_WARNING)
    assert check_order(checked, DICT).warnings == [NO_DUR_WARNING]             # 다시 대조해도 한 번만
    assert check_order(checked, DICT, EMPTY).warnings == []                    # DUR로 대조하면 사라진다


def test_dur_failures_and_partial_lists_become_warnings_that_are_replaced_later():
    failed = unchecked_warning(Unchecked("트리스펜연질캡슐", DUR_FAILED, "의약품 API에 연결하지 못했습니다 (URLError)."))
    assert failed == f"트리스펜연질캡슐부터 DUR API를 쓰지 못해(의약품 API에 연결하지 못했습니다 (URLError).) {DUR_UNCHECKED_TAIL}"
    partial = unchecked_warning(Unchecked("가상케토롤락정", TABOO_PARTIAL))
    assert partial == f"가상케토롤락정: {TABOO_PARTIAL_TAIL}"
    assert is_unchecked_warning(failed) and is_unchecked_warning(partial)
    order = apply_report(order_of(line(TYLENOL, "타이레놀정")),
                         DurReport((), (Unchecked("타이레놀정500mg", DUR_FAILED, "x"),), dur_used=True))
    assert len(order.warnings) == 1
    assert apply_report(order, DurReport((), (), dur_used=True)).warnings == []


def test_apply_report_orders_the_findings_as_given():
    lines = (line(TRISPEN, "트리스펜"), line("199900001", "가상케토롤락정"), line(TYLENOL, "타이레놀정"), line(ARPEN, "아르펜정"))
    order = apply_report(order_of(*lines), DurReport((TABOO, EFFICACY), (), True))
    assert [f.type for f in order.risk_check.findings] == [COMBINATION_TABOO, EFFICACY_DUP]
    assert order.risk_check.level == "danger"
