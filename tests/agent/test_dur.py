"""처방약끼리 성분 중복 찾기 (agent/dur.py)."""
import copy
import dataclasses

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import dictionary_from_data, load_dictionary
from agent.dur import INGREDIENT_DUP, NO_INGREDIENTS, NO_ITEM_CODE, DurHit, DurReport, Unchecked, find_ingredient_duplicates
from agent.schema import DrugLine

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL, GEVORIN, TRISPEN, PRONCON, ARPEN, YOBISIN, TRIPLINE, MUCODYNE = (
    "202106092", "197900277", "201110120", "200200099", "200201239", "200713170", "200002714", "200308774",
)


def line(code, name="처방전에 적힌 이름"):
    return DrugLine(item_code=code, name=name, printed_code=None, count_per_dose=1, times=["morning"], days=3)


def lines(*codes):
    return [line(code, f"약{i}") for i, code in enumerate(codes, start=1)]


# 주성분 칸이 빈 약이 든 사전 (실제 사전의 8종은 모두 채워져 있다)
BLANK = "100000009"


def with_blank():
    return dictionary_from_data({"drugs": [
        {"item_code": TYLENOL, "name": "타이레놀정500mg", "ingredients": ["아세트아미노펜"]},
        {"item_code": GEVORIN, "name": "게보린정", "ingredients": ["아세트아미노펜", "카페인무수물"]},
        {"item_code": BLANK, "name": "성분모름정", "ingredients": []},
    ]})


def small(*drugs):
    """(품목기준코드, 이름, 주성분 목록)으로 만든 작은 사전."""
    return dictionary_from_data({"drugs": [{"item_code": c, "name": n, "ingredients": list(i)} for c, n, i in drugs]})


def codes_of(report):
    return [hit.item_codes for hit in report.hits]


# ------------------------------------------------------------ 시연 조합 A · B


def test_demo_combination_a_shares_acetaminophen():
    report = find_ingredient_duplicates(lines(TYLENOL, GEVORIN), DICT)
    assert report == DurReport(
        hits=(DurHit(INGREDIENT_DUP, (TYLENOL, GEVORIN), ("타이레놀정500mg", "게보린정"), ("아세트아미노펜",)),),
        unchecked=(),
    )


def test_demo_combination_b_shares_ibuprofen():
    report = find_ingredient_duplicates(lines(TRISPEN, ARPEN), DICT)
    assert report == DurReport(
        hits=(DurHit(INGREDIENT_DUP, (TRISPEN, ARPEN), ("트리스펜연질캡슐", "아르펜정"), ("이부프로펜",)),),
        unchecked=(),
    )


def test_the_order_of_the_lines_decides_the_order_inside_a_hit():
    forward = find_ingredient_duplicates(lines(TYLENOL, GEVORIN), DICT).hits[0]
    backward = find_ingredient_duplicates(lines(GEVORIN, TYLENOL), DICT).hits[0]
    assert forward.item_codes == (TYLENOL, GEVORIN) and backward.item_codes == (GEVORIN, TYLENOL)
    assert backward.names == ("게보린정", "타이레놀정500mg")


def test_all_eight_target_drugs_together_give_exactly_the_two_demo_hits():
    report = find_ingredient_duplicates(lines(*(e.item_code for e in DICT.entries)), DICT)
    assert codes_of(report) == [(TYLENOL, GEVORIN), (TRISPEN, ARPEN)]
    assert report.unchecked == ()                                    # 8종 모두 주성분을 안다


def test_drugs_that_share_nothing_give_no_hit():
    report = find_ingredient_duplicates(lines(PRONCON, MUCODYNE, TRIPLINE, TYLENOL), DICT)
    assert report == DurReport(hits=(), unchecked=())


def test_one_drug_and_no_drug_give_no_hit():
    assert find_ingredient_duplicates(lines(GEVORIN), DICT) == DurReport((), ())
    assert find_ingredient_duplicates([], DICT) == DurReport((), ())


# ------------------------------------------------------------ 겹침의 모양


def test_three_drugs_with_one_shared_ingredient_make_one_hit():
    dictionary = small(("100000001", "가정", ["성분X"]), ("100000002", "나정", ["성분X", "성분Y"]), ("100000003", "다정", ["성분X"]))
    report = find_ingredient_duplicates(lines("100000001", "100000002", "100000003"), dictionary)
    assert report.hits == (DurHit(INGREDIENT_DUP, ("100000001", "100000002", "100000003"), ("가정", "나정", "다정"), ("성분X",)),)


def test_two_drugs_that_share_two_ingredients_make_one_hit_with_both():
    dictionary = small(("100000001", "가정", ["성분X", "성분Y", "성분Z"]), ("100000002", "나정", ["성분Y", "성분X"]))
    [hit] = find_ingredient_duplicates(lines("100000001", "100000002"), dictionary).hits
    assert hit.ingredients == ("성분X", "성분Y")                  # 사전에 적힌 순서를 따른다


def test_two_separate_overlaps_make_two_hits():
    dictionary = small(("100000001", "가정", ["성분X"]), ("100000002", "나정", ["성분X", "성분Y"]), ("100000003", "다정", ["성분Y"]))
    report = find_ingredient_duplicates(lines("100000001", "100000002", "100000003"), dictionary)
    assert codes_of(report) == [("100000001", "100000002"), ("100000002", "100000003")]
    assert [h.ingredients for h in report.hits] == [("성분X",), ("성분Y",)]


def test_the_same_ingredient_written_with_other_spaces_still_counts():
    dictionary = small(("100000001", "가정", ["아세트 아미노펜"]), ("100000002", "나정", ["아세트아미노펜"]))
    [hit] = find_ingredient_duplicates(lines("100000001", "100000002"), dictionary).hits
    assert hit.ingredients == ("아세트 아미노펜",)                 # 먼저 나온 표기를 보인다


def test_the_same_drug_on_two_lines_is_not_a_duplicate_of_itself():
    report = find_ingredient_duplicates(lines(TYLENOL, TYLENOL), DICT)
    assert report == DurReport((), ())


def test_the_same_drug_twice_next_to_a_sharing_drug_gives_one_hit_with_each_drug_once():
    report = find_ingredient_duplicates(lines(TYLENOL, GEVORIN, TYLENOL), DICT)
    assert codes_of(report) == [(TYLENOL, GEVORIN)]


def test_the_same_result_every_time():
    items = lines(GEVORIN, TRISPEN, TYLENOL, ARPEN, YOBISIN)
    assert find_ingredient_duplicates(items, DICT) == find_ingredient_duplicates(items, DICT)


def test_the_herbal_drug_with_fourteen_ingredients_shares_none_with_the_other_seven():
    report = find_ingredient_duplicates(lines(YOBISIN, *(e.item_code for e in DICT.entries if e.item_code != YOBISIN)), DICT)
    assert all(YOBISIN not in hit.item_codes for hit in report.hits) and report.unchecked == ()


# ------------------------------------------------------------ 대조하지 못한 약


def test_a_line_without_an_item_code_is_left_out_and_named():
    report = find_ingredient_duplicates([line(TYLENOL, "타이레놀정"), line(None, "알모탄정")], DICT)
    assert report == DurReport(hits=(), unchecked=(Unchecked("알모탄정", NO_ITEM_CODE),))


def test_two_lines_without_codes_are_both_named():
    report = find_ingredient_duplicates([line(None, "가나다정"), line(None, "라마바정")], DICT)
    assert [u.name for u in report.unchecked] == ["가나다정", "라마바정"]


def test_a_drug_the_dictionary_does_not_know_is_left_out_and_named_by_its_line():
    report = find_ingredient_duplicates([line(TYLENOL), line("199900001", "식약처에서 찾은 약정")], DICT)
    assert report.unchecked == (Unchecked("식약처에서 찾은 약정", NO_INGREDIENTS),)      # 사전에 없으니 처방전에 적힌 이름을 쓴다


def test_a_drug_with_unknown_ingredients_is_named_by_the_dictionary():
    report = find_ingredient_duplicates([line(BLANK, "성분 모름 정")], with_blank())
    assert report.unchecked == (Unchecked("성분모름정", NO_INGREDIENTS),)


def test_a_drug_left_out_twice_is_named_once():
    report = find_ingredient_duplicates([line(BLANK), line(BLANK), line("199900001", "가정"), line("199900001", "가정")], with_blank())
    assert [u.name for u in report.unchecked] == ["성분모름정", "가정"]


def test_a_drug_left_out_does_not_hide_the_hits_of_the_others():
    report = find_ingredient_duplicates([line(None, "알모탄정"), line(TYLENOL), line(GEVORIN), line(BLANK)], with_blank())
    assert codes_of(report) == [(TYLENOL, GEVORIN)]
    assert [u.reason for u in report.unchecked] == [NO_ITEM_CODE, NO_INGREDIENTS]


# ------------------------------------------------------------ 값의 성질


def test_the_lines_that_were_given_are_not_changed():
    items = lines(TYLENOL, GEVORIN, None)
    before = copy.deepcopy(items)
    find_ingredient_duplicates(items, DICT)
    assert items == before


def test_a_report_cannot_be_changed():
    report = find_ingredient_duplicates(lines(TYLENOL, GEVORIN), DICT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.hits = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.hits[0].type = "other"


def test_the_type_key_of_the_hit_is_the_one_risk_levels_are_keyed_by():
    from agent.risk import LEVEL_BY_TYPE

    assert find_ingredient_duplicates(lines(TYLENOL, GEVORIN), DICT).hits[0].type in LEVEL_BY_TYPE


# ------------------------------------------------------------ 병용금기 · 효능군 중복 (13단계, 가짜 DUR 조회)
from agent.dur import (  # noqa: E402
    COMBINATION_TABOO,
    DUR_FAILED,
    EFFICACY_DUP,
    TABOO_PARTIAL,
    Ref,
    check_duplicate,
    check_interaction,
    combine_checks,
    find_all,
    find_combination_taboos,
    find_efficacy_duplicates,
    merge_reports,
    report_from_json,
    report_to_json,
)
from fake_dur import KETOROLAC, KETOROLAC_NAME, TABOO_REASON, FakeDur  # noqa: E402


def test_a_combination_taboo_is_found_from_the_list_of_the_first_drug():
    report = find_combination_taboos([line(TRISPEN), line(KETOROLAC, KETOROLAC_NAME)], FakeDur(), DICT)
    assert report == DurReport(
        hits=(DurHit(COMBINATION_TABOO, (TRISPEN, KETOROLAC), ("트리스펜연질캡슐", KETOROLAC_NAME), ("케토롤락트로메타민",),
                     TABOO_REASON),),
        unchecked=(), dur_used=True)


def test_a_combination_taboo_is_found_from_either_side_and_counted_once():
    one_sided = FakeDur(taboos={KETOROLAC: FakeDur().taboos[KETOROLAC]})       # 케토롤락 쪽 목록에만 있다
    report = find_combination_taboos([line(TRISPEN), line(KETOROLAC, KETOROLAC_NAME)], one_sided, DICT)
    assert codes_of(report) == [(TRISPEN, KETOROLAC)]
    both = find_combination_taboos([line(KETOROLAC, KETOROLAC_NAME), line(TRISPEN)], FakeDur(), DICT)
    assert codes_of(both) == [(KETOROLAC, TRISPEN)]                            # 처방 순서를 따른다, 한 번만


def test_no_taboo_among_the_demo_drugs():
    assert find_combination_taboos(lines(TYLENOL, GEVORIN, TRISPEN, ARPEN), FakeDur(), DICT).hits == ()


def test_each_drug_is_asked_once_and_lines_without_a_code_are_skipped():
    dur = FakeDur()
    find_combination_taboos([line(TRISPEN), line(TRISPEN), line(None, "알모탄정")], dur, DICT)
    assert dur.asked == [("taboo", TRISPEN)]


def test_a_partial_taboo_list_is_reported():
    report = find_combination_taboos(lines(TRISPEN, TYLENOL), FakeDur(partial={TRISPEN}), DICT)
    assert report.unchecked == (Unchecked("트리스펜연질캡슐", TABOO_PARTIAL),)


def test_after_a_failure_nothing_more_is_asked_and_the_failure_is_reported_once():
    dur = FakeDur(fail={GEVORIN})
    report = find_combination_taboos(lines(TYLENOL, GEVORIN, TRISPEN), dur, DICT)
    assert dur.asked == [("taboo", TYLENOL), ("taboo", GEVORIN)]
    assert report.unchecked == (Unchecked("게보린정", DUR_FAILED, "의약품 API에 연결하지 못했습니다 (URLError)."),)
    assert report.dur_used is True


def test_an_efficacy_duplicate_groups_drugs_of_the_same_group():
    report = find_efficacy_duplicates(lines(TYLENOL, TRISPEN, MUCODYNE), FakeDur(), DICT)
    assert report.hits == (DurHit(EFFICACY_DUP, (TYLENOL, TRISPEN), ("타이레놀정500mg", "트리스펜연질캡슐"), ("해열진통소염제",)),)


def test_an_efficacy_duplicate_already_found_as_an_ingredient_duplicate_is_not_repeated():
    ingredient = find_ingredient_duplicates(lines(TYLENOL, GEVORIN), DICT)
    assert find_efficacy_duplicates(lines(TYLENOL, GEVORIN), FakeDur(), DICT, already=ingredient.hits).hits == ()
    three = find_efficacy_duplicates(lines(TYLENOL, GEVORIN, TRISPEN), FakeDur(), DICT, already=ingredient.hits)
    assert codes_of(three) == [(TYLENOL, GEVORIN, TRISPEN)]                    # 새 약이 끼면 알린다


def test_check_duplicate_without_dur_is_the_ingredient_check():
    assert check_duplicate(lines(TYLENOL, GEVORIN), DICT) == find_ingredient_duplicates(lines(TYLENOL, GEVORIN), DICT)
    assert check_duplicate(lines(TYLENOL, GEVORIN), DICT).dur_used is False


def test_demo_b_is_one_ingredient_duplicate_even_with_dur():
    report = check_duplicate(lines(TRISPEN, ARPEN), DICT, FakeDur())
    assert [h.type for h in report.hits] == [INGREDIENT_DUP] and report.dur_used


def test_find_all_puts_taboos_first_and_reports_one_failure():
    report = find_all([line(TRISPEN), line(KETOROLAC, KETOROLAC_NAME), line(TYLENOL)], DICT, FakeDur())
    assert [h.type for h in report.hits] == [COMBINATION_TABOO, EFFICACY_DUP]
    failed = find_all(lines(TYLENOL, GEVORIN), DICT, FakeDur(fail={TYLENOL}))
    assert [u.reason for u in failed.unchecked] == [DUR_FAILED]              # 병용금기 · 효능군에서 두 번 실패해도 한 번
    assert [h.type for h in failed.hits] == [INGREDIENT_DUP]                 # 사전으로 하는 성분 중복은 그대로


def test_find_all_without_dur_is_the_ingredient_check_and_says_so():
    report = find_all(lines(TYLENOL, GEVORIN), DICT)
    assert report.dur_used is False and [h.type for h in report.hits] == [INGREDIENT_DUP]


def test_check_interaction_is_the_taboo_search():
    items = [line(TRISPEN), line(KETOROLAC, KETOROLAC_NAME)]
    assert check_interaction(items, FakeDur(), DICT) == find_combination_taboos(items, FakeDur(), DICT)


def test_combine_and_merge_keep_order_and_drop_repeats():
    a = DurReport((DurHit(COMBINATION_TABOO, ("1" * 9, "2" * 9), ("가", "나"), ()),), (Unchecked("가", DUR_FAILED, "x"),), True)
    b = DurReport((DurHit(INGREDIENT_DUP, ("1" * 9, "3" * 9), ("가", "다"), ("성분",)),), (Unchecked("나", DUR_FAILED, "y"),), False)
    assert merge_reports(a, a).hits == a.hits
    combined = combine_checks(a, b)
    assert [h.type for h in combined.hits] == [COMBINATION_TABOO, INGREDIENT_DUP]
    assert combined.unchecked == (Unchecked("가", DUR_FAILED, "x"),) and combined.dur_used


def test_a_report_survives_json():
    import json

    report = find_all([line(TRISPEN), line(KETOROLAC, KETOROLAC_NAME), line(None, "알모탄정")], DICT, FakeDur(partial={TRISPEN}))
    assert report_from_json(json.loads(json.dumps(report_to_json(report)))) == report


def test_refs_with_only_a_code_and_a_name_are_enough():
    assert find_all([Ref(TYLENOL, "가"), Ref(GEVORIN, "나")], DICT, FakeDur()) == find_all(lines(TYLENOL, GEVORIN), DICT, FakeDur())
