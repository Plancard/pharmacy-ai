"""약품 매칭 (agent/drug_match.py, REQ-AG-03)."""
import json

import pytest

from agent.config import DATA_DIR
from agent.drug_dictionary import DrugDictionary, dictionary_from_data, load_dictionary
from agent.drug_names import similarity
from agent.drug_match import (
    CANDIDATE_MIN,
    LOW_CONFIDENCE,
    MAX_CANDIDATES,
    NAME_MATCH_MIN,
    NAME_ONLY_MARGIN,
    NAME_ONLY_MIN,
    match_line,
    match_order,
    strengths_conflict,
)
from agent.schema import TRAY_DAYS, DispensingOrder, DrugLine
from agent.structure import extract_rule
from agent.validate import order_from_extracted

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary()


def load(kind, sample_id):
    return json.loads((DATA_DIR / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def line(name="게보린정", printed_code="900000021", days=3, ocr_conf=0.9, **extra):
    return DrugLine(
        name=name, printed_code=printed_code, dose_per_time=1, times_per_day=1,
        days=days, time_slots=["morning"], ocr_conf=ocr_conf, **extra,
    )


def small_dict(*pairs):
    return dictionary_from_data({"drugs": [{"item_code": c, "name": n} for c, n in pairs]})


# ------------------------------------------------------------ 기준값


def test_thresholds_are_the_agreed_values():
    assert (NAME_MATCH_MIN, NAME_ONLY_MIN, NAME_ONLY_MARGIN) == (0.7, 0.9, 0.1)
    assert (CANDIDATE_MIN, MAX_CANDIDATES, LOW_CONFIDENCE) == (0.6, 3, 0.5)


# ------------------------------------------------------------ 1. 코드로 찾기


def test_code_in_the_dictionary_with_the_same_name_needs_no_check():
    got = match_line(line(), DICT)
    assert (got.item_code, got.match_method) == ("900000021", "code")
    assert not got.needs_pharmacist_check and got.check_reasons == [] and got.match_candidates == []


@pytest.mark.parametrize("name", ["프론크논캠술", "프론크논램술", "프론크논캡슐", "프론크논 캡슐"])
def test_ocr_misreadings_of_the_name_still_pass(name):
    got = match_line(line(name, "900000051"), DICT)
    assert got.item_code == "900000051" and not got.needs_pharmacist_check


def test_ocr_letter_o_in_the_strength_is_read_as_zero():
    got = match_line(line("타이레놀정 5oOmg", "900000011"), DICT)
    assert got.item_code == "900000011" and not got.needs_pharmacist_check


def test_a_different_name_for_the_code_keeps_the_code_but_asks_the_pharmacist():
    # 코드가 먼저다. 다만 이름이 다르면 약사가 확인한다
    got = match_line(line("타이레놀정 500mg", "900000021"), DICT)
    assert (got.item_code, got.match_method) == ("900000021", "code")
    assert got.needs_pharmacist_check
    assert got.check_reasons == ["처방전의 약품명이 코드(900000021)의 약 '게보린정'과 다릅니다."]
    # 후보: 코드의 약을 먼저, 그다음 이름이 비슷한 약
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [
        ("900000021", "처방전 코드"), ("900000011", "이름이 비슷함"),
    ]
    assert got.match_candidates[1].score == 1.0
    assert got.match_candidates[0].score < NAME_MATCH_MIN


def test_a_different_strength_for_the_code_asks_the_pharmacist():
    # 이름은 아주 비슷하지만(0.94) 함량이 다르면 다른 약이다
    got = match_line(line("타이레놀정 650mg", "900000011"), DICT)
    assert got.item_code == "900000011" and got.needs_pharmacist_check
    assert got.check_reasons == ["처방전의 함량이 코드(900000011)의 약 '타이레놀정 500mg'과 다릅니다."]
    assert [c.reason for c in got.match_candidates] == ["처방전 코드"]     # 사전에 같은 코드 말고 비슷한 약은 없다


def test_a_name_without_strength_is_not_a_strength_conflict():
    got = match_line(line("타이레놀정", "900000011"), DICT)
    assert got.item_code == "900000011" and not got.needs_pharmacist_check


def test_the_name_threshold_for_a_code_match_is_inclusive(monkeypatch):
    import agent.drug_match as dm

    d = small_dict(("900000001", "가나다라"))
    score = similarity("가나다", "가나다라")
    monkeypatch.setattr(dm, "NAME_MATCH_MIN", score)                  # 딱 기준값이면 통과
    assert not match_line(line("가나다", "900000001"), d).needs_pharmacist_check
    monkeypatch.setattr(dm, "NAME_MATCH_MIN", score + 0.001)          # 조금이라도 모자라면 확인
    assert match_line(line("가나다", "900000001"), d).needs_pharmacist_check


# ------------------------------------------------------------ 2. 이름으로 찾기


def test_without_a_code_a_clear_name_match_is_chosen_but_always_checked():
    got = match_line(line("게보린정", None), DICT)
    assert (got.item_code, got.match_method) == ("900000021", "name")
    assert got.check_reasons == ["처방전에서 코드를 읽지 못했습니다. 이름으로 '게보린정'을(를) 골랐습니다."]
    assert got.match_candidates == []


def test_a_code_missing_from_the_dictionary_falls_back_to_the_name():
    got = match_line(line("트리스펜연질캡슐", "900000099"), DICT)
    assert (got.item_code, got.match_method) == ("900000041", "name")
    assert got.check_reasons[0].startswith("처방전 코드(900000099)가 약품 사전에 없습니다.")


def test_a_name_below_the_name_only_threshold_is_not_chosen():
    # '게보린'은 '게보린정'과 0.82: 코드로 찾았다면 충분하지만 이름만으로 정하기엔 모자라다
    got = match_line(line("게보린", None), DICT)
    assert (got.item_code, got.match_method) == (None, "none")
    assert got.check_reasons == ["처방전에서 코드를 읽지 못했습니다. 이름으로도 사전의 약을 정하지 못했습니다."]
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [("900000021", "이름이 비슷함")]


def test_a_name_with_a_different_strength_is_not_chosen_by_name():
    got = match_line(line("타이레놀정 650mg", None), DICT)
    assert got.item_code is None and got.match_method == "none"
    assert [c.item_code for c in got.match_candidates] == ["900000011"]


def test_two_close_names_are_not_chosen_and_both_are_offered():
    d = small_dict(("900000001", "가나다라마바정"), ("900000002", "가나다라마사정"))
    got = match_line(line("가나다라마바정", None), d)
    assert got.item_code is None                      # 1등과 2등의 차이가 0.1보다 작다
    assert [c.item_code for c in got.match_candidates] == ["900000001", "900000002"]
    assert got.match_candidates[0].score == 1.0


def test_a_clear_winner_needs_the_margin():
    d = small_dict(("900000001", "가나다라마바정"), ("900000002", "하하하하하하"))
    assert match_line(line("가나다라마바정", None), d).item_code == "900000001"


def test_a_one_drug_dictionary_can_match_by_name():
    d = small_dict(("900000001", "가나다라마바정"))
    assert match_line(line("가나다라마바정", None), d).item_code == "900000001"


def test_the_name_only_threshold_is_inclusive(monkeypatch):
    import agent.drug_match as dm

    d = small_dict(("900000001", "가나다라마바정"))
    score = similarity("가나다라마바사정", "가나다라마바정")
    monkeypatch.setattr(dm, "NAME_ONLY_MIN", score)
    assert match_line(line("가나다라마바사정", None), d).item_code == "900000001"
    monkeypatch.setattr(dm, "NAME_ONLY_MIN", score + 0.001)
    assert match_line(line("가나다라마바사정", None), d).item_code is None


def test_the_margin_is_inclusive(monkeypatch):
    import agent.drug_match as dm

    d = small_dict(("900000001", "가나다라마바정"), ("900000002", "가나다라"))
    gap = similarity("가나다라마바정", "가나다라마바정") - similarity("가나다라마바정", "가나다라")
    monkeypatch.setattr(dm, "NAME_ONLY_MARGIN", gap)
    assert match_line(line("가나다라마바정", None), d).item_code == "900000001"
    monkeypatch.setattr(dm, "NAME_ONLY_MARGIN", gap + 0.001)
    assert match_line(line("가나다라마바정", None), d).item_code is None


def test_the_candidate_threshold_is_inclusive(monkeypatch):
    import agent.drug_match as dm

    score = similarity("게보린", "게보린정")
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score)
    assert [c.item_code for c in match_line(line("게보린", None), DICT).match_candidates] == ["900000021"]
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score + 0.001)
    assert match_line(line("게보린", None), DICT).match_candidates == []


def test_candidate_scores_are_rounded_to_three_places():
    got = match_line(line("게보린", None), DICT)
    assert got.match_candidates[0].score == round(similarity("게보린", "게보린정"), 3) == 0.824


def test_equal_scores_keep_the_dictionary_order():
    d = small_dict(("900000002", "가나다라정"), ("900000001", "가나다마정"))
    assert similarity("가나다정", "가나다라정") == similarity("가나다정", "가나다마정")
    got = match_line(line("가나다정", None), d)
    assert [c.item_code for c in got.match_candidates] == ["900000002", "900000001"]


def test_a_drug_not_in_the_dictionary_has_no_code_and_no_candidates():
    got = match_line(line("알모란정", "900000031"), DICT)
    assert (got.item_code, got.match_method, got.match_candidates) == (None, "none", [])
    assert got.check_reasons == ["처방전 코드(900000031)가 약품 사전에 없습니다. 이름으로도 사전의 약을 정하지 못했습니다."]


def test_candidates_are_limited_and_sorted_by_score():
    d = small_dict(*[(f"90000000{i}", "가나다라마" + "바사아자차"[:i]) for i in range(1, 6)])
    got = match_line(line("가나다라마", None), d)
    scores = [c.score for c in got.match_candidates]
    assert len(scores) == MAX_CANDIDATES and scores == sorted(scores, reverse=True)
    assert all(s >= CANDIDATE_MIN for s in scores)


def test_an_empty_dictionary_matches_nothing():
    got = match_line(line(), DrugDictionary([]))
    assert got.item_code is None and got.match_candidates == [] and got.needs_pharmacist_check


# ------------------------------------------------------------ 3. 그 밖의 확인 사유


@pytest.mark.parametrize("conf, flagged", [(0.49, True), (0.0, True), (LOW_CONFIDENCE, False), (0.9, False), (None, False)])
def test_low_ocr_confidence_asks_the_pharmacist(conf, flagged):
    got = match_line(line(ocr_conf=conf), DICT)
    assert got.item_code == "900000021" and got.needs_pharmacist_check is flagged
    assert got.check_reasons == ([f"글자 인식 신뢰도가 낮습니다({conf:.2f})."] if flagged else [])


@pytest.mark.parametrize("days, flagged", [(TRAY_DAYS, False), (TRAY_DAYS + 1, True), (1, False), (30, True)])
def test_more_days_than_the_tray_asks_the_pharmacist(days, flagged):
    got = match_line(line(days=days), DICT)
    assert got.needs_pharmacist_check is flagged
    assert got.check_reasons == ([f"투약 일수 {days}일이 트레이 3일보다 깁니다."] if flagged else [])


def test_reasons_add_up_in_a_fixed_order():
    got = match_line(line("타이레놀정 500mg", "900000021", days=5, ocr_conf=0.1), DICT)
    assert [r.split("(")[0] for r in got.check_reasons] == [
        "처방전의 약품명이 코드", "글자 인식 신뢰도가 낮습니다", "투약 일수 5일이 트레이 3일보다 깁니다.",
    ]


def test_old_match_results_are_replaced_not_kept():
    old = line("알모란정", "900000031", item_code="900000021", match_method="code",
               needs_pharmacist_check=True, check_reasons=["예전 이유"])
    got = match_line(old, DICT)
    assert got.item_code is None and got.check_reasons == [
        "처방전 코드(900000031)가 약품 사전에 없습니다. 이름으로도 사전의 약을 정하지 못했습니다."
    ]


def test_the_input_line_is_not_changed():
    original = line("알모란정", "900000031")
    before = original.model_dump()
    match_line(original, DICT)
    assert original.model_dump() == before


def test_other_fields_are_kept():
    original = line("게보린정 ", days=2, ocr_conf=0.77)
    got = match_line(original, DICT)
    assert {k: v for k, v in got.model_dump().items() if k not in (
        "item_code", "match_method", "match_candidates", "needs_pharmacist_check", "check_reasons")} == {
        k: v for k, v in original.model_dump().items() if k not in (
        "item_code", "match_method", "match_candidates", "needs_pharmacist_check", "check_reasons")}


# ------------------------------------------------------------ strengths_conflict


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("타이레놀정500mg", "타이레놀정500mg", False),
        ("타이레놀정650mg", "타이레놀정500mg", True),
        ("타이레놀정", "타이레놀정500mg", False),
        ("게보린정300mg", "게보린정", False),
        ("가", "나", False),
    ],
)
def test_strengths_conflict(a, b, expected):
    assert strengths_conflict(a, b) is expected


# ------------------------------------------------------------ 실제 샘플 5장


# 약사 확인 표시가 붙는 줄 (0부터). 이유: 낮은 신뢰도, 사전에 없는 약(알모탄정·rx_005의 실제 약), 4일 처방(rx_003)
FLAGGED = {
    "rx_001": [0, 1],
    "rx_002": [2],
    "rx_003": [0, 1],
    "rx_004": [0, 2, 3],
    "rx_005": [0, 1, 2, 3],
}


def matched(sample_id):
    order = order_from_extracted("P0001", extract_rule(load("cv_samples", sample_id)), source="rule")
    return match_order(order, DICT)


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_item_codes_of_the_real_samples_are_the_expected_ones(sample_id):
    assert [d.item_code for d in matched(sample_id).drugs] == [d["item_code"] for d in load("cv_expected", sample_id)["drugs"]]


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_drugs_outside_the_target_list_are_never_given_a_code(sample_id):
    for got, want in zip(matched(sample_id).drugs, load("cv_expected", sample_id)["drugs"]):
        if not want["in_target"]:
            assert got.item_code is None and got.match_method == "none" and got.needs_pharmacist_check


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_flagged_lines_of_the_real_samples(sample_id):
    assert matched(sample_id).flagged_indices == FLAGGED[sample_id]


def test_a_four_day_prescription_flags_every_line():
    assert all("투약 일수 4일" in r for d in matched("rx_003").drugs for r in d.check_reasons[-1:])


def test_match_order_keeps_the_order_fields_and_the_input():
    order = order_from_extracted("P0001", extract_rule(load("cv_samples", "rx_004")), source="rule")
    before = order.model_dump()
    got = match_order(order, DICT)
    assert order.model_dump() == before
    assert isinstance(got, DispensingOrder)
    assert (got.prescription_id, got.source, got.issue_no, got.warnings) == (
        order.prescription_id, order.source, order.issue_no, order.warnings)
    assert [d.name for d in got.drugs] == [d.name for d in order.drugs]
    assert not got.all_matched          # 사전에 없는 알모탄정이 있다


def test_all_matched_when_every_drug_is_in_the_dictionary():
    order = order_from_extracted("P0001", extract_rule(load("cv_samples", "rx_002")), source="rule")
    order = order.model_copy(update={"drugs": order.drugs[:2]})
    assert match_order(order, DICT).all_matched


# ------------------------------------------------------------ 4. 의약품 API로 찾기 (사전에서 못 정했을 때만)

from agent.drug_api import ApiDrug, DrugApiError  # noqa: E402


class FakeApi:
    def __init__(self, *products, error=None):
        self.products, self.error, self.calls = list(products), error, []

    def __call__(self, name):
        self.calls.append(name)
        if self.error:
            raise self.error
        return self.products


ARPEN = ApiDrug("199900001", "아르펜정(이부프로펜)", "가상제약")


def test_a_drug_missing_from_the_dictionary_is_found_through_the_api():
    api = FakeApi(ARPEN, ApiDrug("199900002", "게보린정", "가상제약"))
    got = match_line(line("아르펜정", "900000099"), DICT, api=api)
    assert (got.item_code, got.match_method) == ("199900001", "api")
    assert got.check_reasons == [
        "처방전 코드(900000099)가 약품 사전에 없습니다. 사전에 없어 의약품 API에서 이름으로 '아르펜정(이부프로펜)'(199900001)을(를) 골랐습니다."
    ]
    assert got.needs_pharmacist_check and got.match_candidates == []
    assert api.calls == ["아르펜정"]                    # API에는 처방전에서 읽은 이름 그대로 넘긴다 (검색어는 drug_api가 만든다)


@pytest.mark.parametrize("drug", [line(), line("게보린정", None), line("프론크논캠술", "900000051")])
def test_the_api_is_not_called_when_the_dictionary_decides(drug):
    api = FakeApi(error=AssertionError("부르면 안 된다"))
    assert match_line(drug, DICT, api=api).match_method in ("code", "name")
    assert api.calls == []


def test_without_an_api_nothing_changes():
    assert match_line(line("아르펜정", None), DICT).check_reasons == [
        "처방전에서 코드를 읽지 못했습니다. 이름으로도 사전의 약을 정하지 못했습니다."
    ]


def test_api_results_that_are_not_clear_become_candidates():
    api = FakeApi(ApiDrug("199900001", "아르펜정200mg", "가"), ApiDrug("199900003", "아르펜정400mg", "나"))
    got = match_line(line("아르펜정", None), DICT, api=api)
    assert got.item_code is None and got.match_method == "none"           # 함량이 다른 두 제품: 2등과 차이가 없다
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [
        ("199900001", "의약품 API 검색"), ("199900003", "의약품 API 검색"),
    ]
    assert got.check_reasons == ["처방전에서 코드를 읽지 못했습니다. 사전과 의약품 API 어디에서도 약을 정하지 못했습니다."]


def test_the_strength_written_on_the_prescription_picks_the_product():
    api = FakeApi(ApiDrug("199900001", "아르펜정200밀리그램", "가"), ApiDrug("199900003", "아르펜정400밀리그램", "가"))
    got = match_line(line("아르펜정 4OOmg", None), DICT, api=api)
    assert got.item_code == "199900003" and got.match_method == "api"


def test_a_strength_conflict_with_the_only_api_product_is_not_chosen():
    got = match_line(line("아르펜정 400mg", None), DICT, api=FakeApi(ApiDrug("199900001", "아르펜정200mg", "가")))
    assert got.item_code is None and got.match_candidates[0].item_code == "199900001"     # 고르지는 않지만 후보로는 보인다


def test_one_clear_api_product_is_enough():
    got = match_line(line("아르펜정", None), DICT, api=FakeApi(ARPEN))
    assert got.item_code == "199900001"


def test_no_api_result_leaves_the_line_unmatched():
    got = match_line(line("알모란정", "900000031"), DICT, api=FakeApi())
    assert (got.item_code, got.match_candidates) == (None, [])
    assert got.check_reasons == ["처방전 코드(900000031)가 약품 사전에 없습니다. 사전과 의약품 API 어디에서도 약을 정하지 못했습니다."]


def test_an_api_failure_is_a_reason_not_a_crash():
    got = match_line(line("아르펜정", None, days=5), DICT, api=FakeApi(error=DrugApiError("의약품 API에 연결하지 못했습니다 (URLError).")))
    assert got.item_code is None and got.needs_pharmacist_check
    assert got.check_reasons == [
        "처방전에서 코드를 읽지 못했습니다. 사전에 없어 의약품 API로 찾으려 했지만 실패했습니다: 의약품 API에 연결하지 못했습니다 (URLError).",
        "투약 일수 5일이 트레이 3일보다 깁니다.",
    ]


def test_other_api_errors_are_not_hidden():
    from agent.errors import ConfigError

    with pytest.raises(ConfigError):
        match_line(line("아르펜정", None), DICT, api=FakeApi(error=ConfigError("DRUG_API_KEY가 설정되지 않았습니다.")))


def test_dictionary_and_api_candidates_are_shown_together_by_score():
    api = FakeApi(ApiDrug("199900005", "게보린정에스", "가"), ApiDrug("199900006", "게보", "나"))
    got = match_line(line("게보린", None), DICT, api=api)
    assert got.item_code is None
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [
        ("900000021", "이름이 비슷함"), ("199900006", "의약품 API 검색"), ("199900005", "의약품 API 검색"),
    ]                                       # 0.824 / '게보' 0.727 / '게보린정에스' 0.667
    scores = [c.score for c in got.match_candidates]
    assert scores == sorted(scores, reverse=True)


def test_api_candidates_are_limited_too():
    api = FakeApi(*[ApiDrug(f"19990000{i}", "아르펜정" + "가나다라마"[:i], "가") for i in range(1, 6)])
    assert len(match_line(line("아르펜정", None), DICT, api=api).match_candidates) == MAX_CANDIDATES


def test_api_candidates_below_the_threshold_are_dropped():
    got = match_line(line("아르펜정", None), DICT, api=FakeApi(ApiDrug("199900009", "전혀다른캡슐", "가")))
    assert got.match_candidates == []


def test_the_api_candidate_threshold_is_inclusive_and_scores_are_rounded(monkeypatch):
    import agent.drug_match as dm

    api = FakeApi(ApiDrug("199900009", "아르펜", "가"))
    score = similarity("아르펜정", "아르펜")
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score)
    got = match_line(line("아르펜정", None), DICT, api=api)
    assert [(c.item_code, c.score) for c in got.match_candidates] == [("199900009", round(score, 3))]
    assert round(score, 3) != score                          # 반올림이 실제로 일어나는 값이다
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score + 0.001)
    assert match_line(line("아르펜정", None), DICT, api=api).match_candidates == []


def test_equal_api_scores_keep_the_api_order():
    api = FakeApi(ApiDrug("199900002", "가나다라정", "가"), ApiDrug("199900001", "가나다마정", "가"))
    got = match_line(line("가나다정", None), DICT, api=api)
    assert [c.item_code for c in got.match_candidates] == ["199900002", "199900001"]


def test_the_api_margin_and_threshold_are_the_same_as_for_the_dictionary(monkeypatch):
    import agent.drug_match as dm

    api = FakeApi(ApiDrug("199900001", "가나다라마바정", "가"), ApiDrug("199900002", "가나다라", "가"))
    gap = 1.0 - similarity("가나다라마바정", "가나다라")
    monkeypatch.setattr(dm, "NAME_ONLY_MARGIN", gap)
    assert match_line(line("가나다라마바정", None), DICT, api=api).item_code == "199900001"
    monkeypatch.setattr(dm, "NAME_ONLY_MARGIN", gap + 0.001)
    assert match_line(line("가나다라마바정", None), DICT, api=api).item_code is None
    monkeypatch.setattr(dm, "NAME_ONLY_MARGIN", 0.1)
    monkeypatch.setattr(dm, "NAME_ONLY_MIN", 1.001)
    assert match_line(line("가나다라마바정", None), DICT, api=FakeApi(api.products[0])).item_code is None


def test_match_order_passes_the_api_on():
    order = order_from_extracted("P0001", extract_rule(load("cv_samples", "rx_001")), source="rule")
    api = FakeApi(ApiDrug("199900031", "알모탄정", "가"))
    got = match_order(order, DICT, api=api)
    # OCR의 '알모란정'과 '알모탄정'은 0.909라 API가 이 제품을 돌려주면 고른다 (언제나 약사 확인 표시와 함께)
    assert [d.item_code for d in got.drugs] == ["900000011", "199900031"]
    assert [d.match_method for d in got.drugs] == ["code", "api"] and got.drugs[1].needs_pharmacist_check
    assert api.calls == ["알모란정"]                     # 사전에 있는 타이레놀정은 API를 부르지 않는다


def test_a_different_strength_in_the_dictionary_is_not_a_rival_either():
    d = small_dict(("900000001", "가나다정 200mg"), ("900000002", "가나다정 400mg"))
    assert match_line(line("가나다정 400mg", None), d).item_code == "900000002"
    assert match_line(line("가나다정", None), d).item_code is None          # 함량을 모르면 둘 다 경쟁자다


def test_the_api_method_needs_an_item_code():
    with pytest.raises(Exception, match="item_code가 있어야"):
        DrugLine(name="가", dose_per_time=1, times_per_day=1, days=1, time_slots=["morning"], match_method="api")
