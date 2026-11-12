"""약품 매칭 (agent/drug_match.py, REQ-AG-03) — 인쇄 코드 → 이름 → 의약품 API → 못 정함."""
import itertools
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_api import ApiDrug
from agent.drug_dictionary import DrugDictionary, dictionary_from_data, load_dictionary
from agent.drug_match import (
    CANDIDATE_MIN,
    CODE_DIFFERENT,
    CODE_FOUND,
    CODE_NOT_FOUND,
    LOW_CONFIDENCE,
    MAX_CANDIDATES,
    NAME_MATCH_MIN,
    NAME_ONLY_MARGIN,
    NAME_ONLY_MIN,
    check_item_code,
    match_line,
    match_order,
    strengths_conflict,
)
from agent.drug_names import match_key, similarity
from agent.errors import ConfigError, DrugApiError
from agent.schema import TRAY_DAYS, DispensingOrder, DrugLine
from agent.structure import extract_rule
from agent.validate import order_from_extracted

SAMPLES = RESOURCES_DIR / "samples"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")

# 품목기준코드 (사전 8종). 처방전에 인쇄되는 가상 코드(9000000xx)는 품목기준코드가 아니라서 이름 단계로 넘어간다.
TYLENOL, GEVORIN, TRISPEN, FRONK = "202106092", "197900277", "201110120", "200200099"
ARPEN, YOVISIN, TRIPLINE, MUCODYNE = "200201239", "200713170", "200002714", "200308774"
FAKE_CODE = {"타이레놀": "900000011", "게보린": "900000021", "알모탄": "900000031", "트리스펜": "900000041", "프론크논": "900000051"}
CHANGED = ("item_code", "match_method", "match_candidates", "needs_pharmacist_check", "check_reasons")


def load(kind, sample_id):
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def line(name="게보린정", printed_code=GEVORIN, days=3, ocr_conf=0.9, **extra):
    extra.setdefault("item_code", None)            # 매칭 전의 줄은 약이 정해지지 않았다
    return DrugLine(
        name=name, printed_code=printed_code, count_per_dose=1,
        days=days, times=["morning"], ocr_conf=ocr_conf, **extra,
    )


def small_dict(*pairs):
    return dictionary_from_data({"drugs": [{"item_code": c, "name": n} for c, n in pairs]})


# ------------------------------------------------------------ 기준값


def test_thresholds_are_the_agreed_values():
    assert (NAME_MATCH_MIN, NAME_ONLY_MIN, NAME_ONLY_MARGIN) == (0.7, 0.9, 0.1)
    assert (CANDIDATE_MIN, MAX_CANDIDATES, LOW_CONFIDENCE) == (0.6, 3, 0.5)


def _read_names():
    """샘플 5장에서 OCR이 읽은 약 이름. (읽은 이름, 정답 품목기준코드 또는 None)"""
    return [(d["name"], d["item_code"]) for rx in SAMPLE_IDS for d in load("cv_expected", rx)["items"]]


def test_thresholds_fit_what_was_measured_on_the_samples_and_the_dictionary():
    """기준값은 사전 8종과 샘플 5장에서 잰 값 사이에 놓여 있다 (11/10 측정, 11/12 OCR 한글 보정 뒤 다시 잼). 사전이나 샘플이 바뀌면 다시 재서 맞춘다.

    OCR이 읽은 대상 약 이름 vs 사전의 같은 약: 보정 전 0.812 ~ 1.0, 보정 뒤 모두 1.0 (캠술 · 램술 · 편이 이 샘플의 오인식 전부였다)
    대상이 아닌 약 vs 사전 어느 약이든: 최대 0.667 / 사전의 약끼리: 최대 0.541
    """
    target = [similarity(match_key(n), DICT.by_item_code(code).key) for n, code in _read_names() if code]
    others = [max(similarity(match_key(n), e.key) for e in DICT.entries) for n, code in _read_names() if not code]
    pairs = [similarity(a.key, b.key) for a, b in itertools.combinations(DICT.entries, 2)]
    assert (round(min(target), 3), round(max(others), 3), round(max(pairs), 3)) == (1.0, 0.667, 0.541)
    assert max(others) < NAME_MATCH_MIN <= min(target)          # 코드로 찾은 약과 이름이 맞는지 가르는 선
    assert max(pairs) < CANDIDATE_MIN                            # 사전 약끼리는 서로의 후보가 되지 않는다
    assert CANDIDATE_MIN <= min(target) and CANDIDATE_MIN <= max(others)   # 오독한 대상 약은 후보로 보이고, 비슷한 다른 약도 후보까지는 간다
    assert max(others) < NAME_ONLY_MIN                           # 대상이 아닌 약은 이름만으로 정해지지 않는다
    assert NAME_ONLY_MIN <= min(target)                          # 보정한 대상 약 이름은 이름으로 정한다 (언제나 약사 확인 표시)
    # 보정 목록에 없는 오인식은 여전히 후보로만 남는다: test_a_similar_name_is_not_chosen_only_offered_as_a_candidate


# ------------------------------------------------------------ 사전 8종 찾기


@pytest.mark.parametrize("entry", DICT.entries, ids=lambda e: e.name)
def test_every_dictionary_drug_is_found_by_its_item_code(entry):
    got = match_line(line(entry.name, entry.item_code), DICT)
    assert (got.item_code, got.match_method) == (entry.item_code, "code")
    assert not got.needs_pharmacist_check and got.check_reasons == [] and got.match_candidates == []


@pytest.mark.parametrize("entry", DICT.entries, ids=lambda e: e.name)
@pytest.mark.parametrize("printed", [None, "900000099"])
def test_every_dictionary_drug_is_found_by_its_name_but_always_checked(entry, printed):
    got = match_line(line(entry.name, printed), DICT)
    assert (got.item_code, got.match_method) == (entry.item_code, "name")
    assert got.needs_pharmacist_check and got.match_candidates == []


@pytest.mark.parametrize(
    "written, expected",
    [
        ("타이레놀정 500mg", TYLENOL), ("타이레놀정/500mg", TYLENOL), ("타이레놀정 5oOmg", TYLENOL), ("타이레놀정500밀리그램", TYLENOL),
        ("뮤코다인캡슐 200mg", MUCODYNE), ("뮤코다인캡슐200 mg", MUCODYNE),
        ("아이월드요비신정/416.7mg", YOVISIN), ("아이월드요비신정 416.7 mg", YOVISIN),
    ],
)
def test_spelling_variants_are_found_by_name(written, expected):
    assert match_line(line(written, None), DICT).item_code == expected


# ------------------------------------------------------------ 1. 인쇄 코드로 찾기


def test_code_in_the_dictionary_with_the_same_name_needs_no_check():
    got = match_line(line(), DICT)
    assert (got.item_code, got.match_method) == (GEVORIN, "code")
    assert not got.needs_pharmacist_check and got.check_reasons == [] and got.match_candidates == []


@pytest.mark.parametrize("name", ["프론크논캠술", "프론크논램술", "프론크논캡슐", "프론크논 캡슐"])
def test_ocr_misreadings_of_the_name_still_pass_when_the_code_is_right(name):
    got = match_line(line(name, FRONK), DICT)
    assert got.item_code == FRONK and not got.needs_pharmacist_check


def test_ocr_letter_o_in_the_strength_is_read_as_zero():
    got = match_line(line("타이레놀정 5oOmg", TYLENOL), DICT)
    assert got.item_code == TYLENOL and not got.needs_pharmacist_check


def test_a_different_name_for_the_code_keeps_the_code_but_asks_the_pharmacist():
    # 코드가 먼저다. 다만 이름이 다르면 약사가 확인한다
    got = match_line(line("타이레놀정500mg", GEVORIN), DICT)
    assert (got.item_code, got.match_method) == (GEVORIN, "code")
    assert got.needs_pharmacist_check
    assert got.check_reasons == [f"처방전의 약품명이 코드({GEVORIN})의 약 '게보린정'과 다릅니다."]
    # 후보: 코드의 약을 먼저, 그다음 이름이 비슷한 약
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [(GEVORIN, "처방전 코드"), (TYLENOL, "이름이 비슷함")]
    assert got.match_candidates[1].score == 1.0
    assert got.match_candidates[0].score < NAME_MATCH_MIN


def test_a_different_strength_for_the_code_asks_the_pharmacist():
    # 이름은 아주 비슷하지만(0.94) 함량이 다르면 다른 약이다
    got = match_line(line("타이레놀정 650mg", TYLENOL), DICT)
    assert got.item_code == TYLENOL and got.needs_pharmacist_check
    assert got.check_reasons == [f"처방전의 함량이 코드({TYLENOL})의 약 '타이레놀정500mg'과 다릅니다."]
    assert [c.reason for c in got.match_candidates] == ["처방전 코드"]     # 사전에 같은 코드 말고 비슷한 약은 없다


def test_a_name_without_strength_is_not_a_strength_conflict():
    got = match_line(line("타이레놀정", TYLENOL), DICT)
    assert got.item_code == TYLENOL and not got.needs_pharmacist_check


@pytest.mark.parametrize("fake", sorted(FAKE_CODE.values()))
def test_the_dummy_prescription_codes_are_not_item_codes(fake):
    """가상 처방전의 9000000xx(실제 처방전이면 보험코드 자리)는 품목기준코드가 아니다. 이 코드로는 자동 확정하지 않고 이름 단계로 간다."""
    got = match_line(line("게보린정", fake), DICT)
    assert (got.item_code, got.match_method) == (GEVORIN, "name")
    assert got.needs_pharmacist_check
    assert got.check_reasons == [f"처방전 코드({fake})로는 사전의 약을 찾지 못했습니다. 이름으로 '게보린정'을(를) 골랐습니다."]


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
    assert (got.item_code, got.match_method) == (GEVORIN, "name")
    assert got.check_reasons == ["처방전에서 코드를 읽지 못했습니다. 이름으로 '게보린정'을(를) 골랐습니다."]
    assert got.match_candidates == []


def test_a_code_missing_from_the_dictionary_falls_back_to_the_name():
    got = match_line(line("트리스펜연질캡슐", "900000099"), DICT)
    assert (got.item_code, got.match_method) == (TRISPEN, "name")
    assert got.check_reasons[0].startswith("처방전 코드(900000099)로는 사전의 약을 찾지 못했습니다.")


def test_a_similar_name_is_not_chosen_only_offered_as_a_candidate():
    # 보정 목록에 없는 오인식(프룬그논, 0.875)은 이름만으로 확정하지 않는다. 사전의 약을 후보로 보여 준다
    got = match_line(line("프룬그논캡슐", "900000051"), DICT)
    assert (got.item_code, got.match_method) == (None, "none")
    assert [(c.item_code, c.score, c.reason) for c in got.match_candidates] == [(FRONK, 0.875, "이름이 비슷함")]
    assert got.needs_pharmacist_check
    assert got.check_reasons == ["처방전 코드(900000051)로는 사전의 약을 찾지 못했습니다. 이름이 비슷한 약은 있지만 확정하지 않고 후보로만 보여 드립니다."]


@pytest.mark.parametrize("read, expected", [("프론크논캠술", FRONK), ("프론크논램술", FRONK), ("트리스편연질캠술", TRISPEN)])
def test_known_ocr_misreadings_are_chosen_by_name_but_always_checked(read, expected):
    # 실제 OCR에서 나온 오인식(캡슐 -> 캠술 · 램술, 펜 -> 편)은 비교 열쇠에서 맞춰 이름으로 정한다. 보이는 이름은 그대로다
    got = match_line(line(read, "900000051"), DICT)
    assert (got.item_code, got.match_method, got.name) == (expected, "name", read)
    assert got.needs_pharmacist_check and got.match_candidates == []


def test_a_name_below_the_name_only_threshold_is_not_chosen():
    # '게보린'은 '게보린정'과 0.82: 코드로 찾았다면 충분하지만 이름만으로 정하기엔 모자라다
    got = match_line(line("게보린", None), DICT)
    assert (got.item_code, got.match_method) == (None, "none")
    assert got.check_reasons == ["처방전에서 코드를 읽지 못했습니다. 이름이 비슷한 약은 있지만 확정하지 않고 후보로만 보여 드립니다."]
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [(GEVORIN, "이름이 비슷함")]


def test_a_name_with_a_different_strength_is_not_chosen_by_name():
    got = match_line(line("타이레놀정 650mg", None), DICT)
    assert got.item_code is None and got.match_method == "none"
    assert [c.item_code for c in got.match_candidates] == [TYLENOL]


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
    assert [c.item_code for c in match_line(line("게보린", None), DICT).match_candidates] == [GEVORIN]
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score + 0.001)
    got = match_line(line("게보린", None), DICT)
    assert got.match_candidates == []
    assert got.check_reasons == ["처방전에서 코드를 읽지 못했습니다. 이름으로도 사전의 약을 정하지 못했습니다."]


def test_candidate_scores_are_rounded_to_three_places():
    got = match_line(line("게보린", None), DICT)
    assert got.match_candidates[0].score == round(similarity("게보린", "게보린정"), 3) == 0.824


def test_equal_scores_keep_the_dictionary_order():
    d = small_dict(("900000002", "가나다라정"), ("900000001", "가나다마정"))
    assert similarity("가나다정", "가나다라정") == similarity("가나다정", "가나다마정")
    got = match_line(line("가나다정", None), d)
    assert [c.item_code for c in got.match_candidates] == ["900000002", "900000001"]


def test_a_drug_not_in_the_dictionary_is_never_given_a_code():
    got = match_line(line("알모란정", "900000031"), DICT)
    assert (got.item_code, got.match_method) == (None, "none")
    # OCR이 알모탄정을 '알모란정'으로 읽었다. 사전의 아르펜정과 0.667이라 고르지는 않고 후보로만 보인다
    assert [(c.item_code, c.score) for c in got.match_candidates] == [(ARPEN, 0.667)]
    assert got.needs_pharmacist_check


def test_a_drug_that_looks_like_nothing_in_the_dictionary_has_no_candidates():
    got = match_line(line("알모탄정", "900000031"), DICT)            # OCR이 제대로 읽은 알모탄정은 사전 어느 약과도 0.6보다 덜 비슷하다
    assert (got.item_code, got.match_method, got.match_candidates) == (None, "none", [])
    assert got.check_reasons == ["처방전 코드(900000031)로는 사전의 약을 찾지 못했습니다. 이름으로도 사전의 약을 정하지 못했습니다."]


def test_candidates_are_limited_and_sorted_by_score():
    d = small_dict(*[(f"90000000{i}", "가나다라마" + "바사아자차"[:i]) for i in range(1, 6)])
    got = match_line(line("가나다라마", None), d)
    scores = [c.score for c in got.match_candidates]
    assert len(scores) == MAX_CANDIDATES and scores == sorted(scores, reverse=True)
    assert all(s >= CANDIDATE_MIN for s in scores)


def test_an_empty_dictionary_matches_nothing():
    got = match_line(line(), DrugDictionary([]))
    assert got.item_code is None and got.match_candidates == [] and got.needs_pharmacist_check


def test_a_different_strength_in_the_dictionary_is_not_a_rival_either():
    d = small_dict(("900000001", "가나다정 200mg"), ("900000002", "가나다정 400mg"))
    assert match_line(line("가나다정 400mg", None), d).item_code == "900000002"
    assert match_line(line("가나다정", None), d).item_code is None          # 함량을 모르면 둘 다 경쟁자다


def test_only_a_code_match_is_confirmed_without_the_pharmacist():
    """자동 확정은 인쇄된 코드가 품목기준코드와 맞을 때뿐이다. 이름으로 정한 줄도, 못 정한 줄도 확인 표시가 붙는다."""
    for entry in DICT.entries:
        assert not match_line(line(entry.name, entry.item_code), DICT).needs_pharmacist_check
        assert match_line(line(entry.name, None), DICT).needs_pharmacist_check
        assert match_line(line(entry.name, "900000099"), DICT).needs_pharmacist_check
    for name in ["알모란정", "게보린", "타이레놀정 650mg", "처음 보는 약정"]:
        assert match_line(line(name, None), DICT).needs_pharmacist_check


# ------------------------------------------------------------ 3. 그 밖의 확인 사유


@pytest.mark.parametrize("conf, flagged", [(0.49, True), (0.0, True), (LOW_CONFIDENCE, False), (0.9, False), (None, False)])
def test_low_ocr_confidence_asks_the_pharmacist(conf, flagged):
    got = match_line(line(ocr_conf=conf), DICT)
    assert got.item_code == GEVORIN and got.needs_pharmacist_check is flagged
    assert got.check_reasons == ([f"글자 인식 신뢰도가 낮습니다({conf:.2f})."] if flagged else [])


@pytest.mark.parametrize("days, flagged", [(TRAY_DAYS, False), (TRAY_DAYS + 1, True), (1, False), (30, True)])
def test_more_days_than_the_tray_asks_the_pharmacist(days, flagged):
    got = match_line(line(days=days), DICT)
    assert got.needs_pharmacist_check is flagged
    assert got.check_reasons == ([f"투약 일수 {days}일이 트레이 3일보다 깁니다."] if flagged else [])


def test_reasons_add_up_in_a_fixed_order():
    got = match_line(line("타이레놀정500mg", GEVORIN, days=5, ocr_conf=0.1), DICT)
    assert [r.split("(")[0] for r in got.check_reasons] == [
        "처방전의 약품명이 코드", "글자 인식 신뢰도가 낮습니다", "투약 일수 5일이 트레이 3일보다 깁니다.",
    ]


def test_old_match_results_are_replaced_not_kept():
    old = line(
        "알모탄정", "900000031", item_code=GEVORIN, match_method="code", needs_pharmacist_check=True,
        check_reasons=["예전 이유"],
        match_candidates=[{"item_code": GEVORIN, "name": "게보린정", "score": 0.9, "reason": "예전 후보"}],
    )
    got = match_line(old, DICT)
    assert got.item_code is None and got.match_method == "none" and got.match_candidates == []
    assert got.check_reasons == ["처방전 코드(900000031)로는 사전의 약을 찾지 못했습니다. 이름으로도 사전의 약을 정하지 못했습니다."]


def test_the_input_line_is_not_changed():
    original = line("알모란정", "900000031")
    before = original.model_dump()
    match_line(original, DICT)
    assert original.model_dump() == before


def test_other_fields_are_kept():
    original = line("게보린정 ", days=2, ocr_conf=0.77, source_line_ids=[3, 4])
    got = match_line(original, DICT)
    assert {k: v for k, v in got.model_dump().items() if k not in CHANGED} == {
        k: v for k, v in original.model_dump().items() if k not in CHANGED}


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


# ------------------------------------------------------------ 실제 샘플 5장 (CV가 보낸 가상 처방전)


def matched(sample_id, *, print_item_codes=False):
    order = order_from_extracted("P0001", extract_rule(load("cv_a1", sample_id)), source="rule")
    if print_item_codes:       # 인쇄된 코드를 품목기준코드로 바꿔 본다 (코드 단계가 하는 일을 보려는 것)
        wanted = [d["item_code"] for d in load("cv_expected", sample_id)["items"]]
        order = order.model_copy(update={"items": [d.model_copy(update={"printed_code": c or d.printed_code})
                                                   for d, c in zip(order.items, wanted)]})
    return match_order(order, DICT)


# 인쇄된 코드가 가상 코드(9000000xx)·보험코드라서 코드 단계는 건너뛰고 이름 단계로 간다: (품목기준코드, 방법)
BY_NAME = {
    "rx_001": [(TYLENOL, "name"), (None, "none")],
    "rx_002": [(GEVORIN, "name"), (FRONK, "name"), (None, "none")],       # 프론크논캠술은 OCR 한글 보정으로 이름이 같아진다
    "rx_003": [(TRISPEN, "name"), (None, "none")],                        # 트리스편연질캠술도 (보정 전에는 0.857이라 후보로만)
    "rx_004": [(TYLENOL, "name"), (GEVORIN, "name"), (FRONK, "name"), (None, "none")],
    "rx_005": [(None, "none")] * 4,
}
# 인쇄된 코드를 품목기준코드로 바꾸면 코드 단계가 약을 정한다. 확인 표시가 붙는 줄(0부터): 낮은 신뢰도, 사전에 없는 약, 4일 처방
BY_CODE_FLAGGED = {"rx_001": [0, 1], "rx_002": [2], "rx_003": [0, 1], "rx_004": [0, 2, 3], "rx_005": [0, 1, 2, 3]}


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_real_samples_by_name_when_the_printed_codes_are_not_item_codes(sample_id):
    got = matched(sample_id)
    assert [(d.item_code, d.match_method) for d in got.items] == BY_NAME[sample_id]
    assert got.flagged_indices == list(range(len(got.items)))      # 코드로 확정한 줄이 없으니 모든 줄을 약사가 본다


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_a_chosen_code_is_never_a_wrong_one_and_the_right_drug_is_offered_otherwise(sample_id):
    for got, want in zip(matched(sample_id).items, load("cv_expected", sample_id)["items"]):
        if got.item_code is not None:
            assert got.item_code == want["item_code"]
        elif want["item_code"] is not None:
            assert want["item_code"] in [c.item_code for c in got.match_candidates]     # 못 정해도 맞는 약이 후보에는 있다
        assert got.item_code is not None or got.needs_pharmacist_check


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_drugs_outside_the_target_list_are_never_given_a_code(sample_id):
    for print_item_codes in (False, True):
        for got, want in zip(matched(sample_id, print_item_codes=print_item_codes).items, load("cv_expected", sample_id)["items"]):
            if not want["in_target"]:
                assert got.item_code is None and got.match_method == "none" and got.needs_pharmacist_check


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_real_samples_with_item_codes_printed_are_found_by_code(sample_id):
    got = matched(sample_id, print_item_codes=True)
    assert [d.item_code for d in got.items] == [d["item_code"] for d in load("cv_expected", sample_id)["items"]]
    assert got.flagged_indices == BY_CODE_FLAGGED[sample_id]
    assert all(d.match_method == ("code" if d.item_code else "none") for d in got.items)


def test_a_four_day_prescription_flags_every_line():
    assert all("투약 일수 4일이 트레이 3일보다 깁니다." in d.check_reasons for d in matched("rx_003").items)


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_a_line_without_a_code_is_always_flagged_so_the_web_can_show_it(sample_id):
    for print_item_codes in (False, True):
        got = matched(sample_id, print_item_codes=print_item_codes)
        assert all(d.needs_pharmacist_check for d in got.items if d.item_code is None)
        DispensingOrder.model_validate(got.model_dump())             # 모양도 그대로 통과한다 (extra="forbid")


# ------------------------------------------------------------ 우리 가상 처방전 (a1_ocr 4장 + LLM 경로의 s05)


DUMMY_IDS = ["s01_combo_a", "s02_combo_b", "s03_no_risk", "s04_flagged", "s05_no_code"]
# s04의 프론크논은 OCR이 '프론크논램술'로 읽은 것이지만 OCR 한글 보정으로 이름 단계가 정한다. 나머지도 이름이 사전과 같다
DUMMY_UNCHOSEN = {"s04_flagged": [3]}              # 약을 못 정하는 줄 (0부터): 알모탄정(대상 아님)


def dummy_lines(sample_id):
    if sample_id == "s05_no_code":          # 규칙 읽기가 못 읽는 처방전: LLM이 읽은 이름과 같다고 보고 정답의 이름으로 대신한다
        return [line(d["name"], None) for d in load("a5_order", sample_id)["items"]]
    return order_from_extracted("P", extract_rule(load("a1_ocr", sample_id)), source="rule").items


@pytest.mark.parametrize("sample_id", DUMMY_IDS)
def test_our_dummy_prescriptions_are_matched_by_name(sample_id):
    answer = load("a5_order", sample_id)["items"]
    got = [match_line(d, DICT) for d in dummy_lines(sample_id)]
    assert len(got) == len(answer)
    for g, want in zip(got, answer):
        if g.item_code is not None:
            assert (g.item_code, g.match_method) == (want["item_code"], "name")      # 고른 약은 정답과 같다
        elif want["item_code"] is not None:
            assert want["item_code"] in [c.item_code for c in g.match_candidates]    # 못 골라도 맞는 약이 후보에 있다
        assert g.needs_pharmacist_check                                               # 인쇄된 코드가 품목기준코드가 아니라서 모두 약사가 본다
    assert [i for i, g in enumerate(got) if g.item_code is None] == DUMMY_UNCHOSEN.get(sample_id, [])


# ------------------------------------------------------------ 조제지시서 전체


def test_match_order_keeps_the_order_fields_and_the_input():
    order = order_from_extracted("P0001", extract_rule(load("cv_a1", "rx_004")), source="rule")
    before = order.model_dump()
    got = match_order(order, DICT)
    assert order.model_dump() == before
    assert isinstance(got, DispensingOrder)
    assert (got.prescription_id, got.source, got.issue_no, got.warnings) == (
        order.prescription_id, order.source, order.issue_no, order.warnings)
    assert [d.name for d in got.items] == [d.name for d in order.items]
    assert not got.all_matched          # 사전에 없는 알모탄정이 있다


def test_all_matched_when_every_drug_is_in_the_dictionary():
    order = order_from_extracted("P0001", extract_rule(load("cv_a1", "rx_001")), source="rule")
    order = order.model_copy(update={"items": order.items[:1]})            # 타이레놀정만
    assert match_order(order, DICT).all_matched


# ------------------------------------------------------------ 3. 의약품 API로 찾기 (사전에서 못 정했을 때만)


class FakeApi:
    """API 조회 함수의 가짜. 받은 검색어를 기록하고, 정해 둔 제품(또는 오류)을 돌려준다."""

    def __init__(self, *products, error=None):
        self.products, self.error, self.calls = list(products), error, []

    def __call__(self, term):
        self.calls.append(term)
        if self.error:
            raise self.error
        return self.products


FESTAL = ApiDrug("199900001", "훼스탈정(소화효소)", "가상제약")
NOT_WHAT_IT_SAYS = "처방전에서 코드를 읽지 못했습니다."


def test_a_drug_missing_from_the_dictionary_is_found_through_the_api():
    api = FakeApi(FESTAL, ApiDrug("199900002", "게보린정", "가상제약"))
    got = match_line(line("훼스탈정", "900000099"), DICT, api=api)
    assert (got.item_code, got.match_method) == ("199900001", "api")
    assert got.check_reasons == [
        "처방전 코드(900000099)로는 사전의 약을 찾지 못했습니다. 사전에 없어 의약품 API에서 이름으로 '훼스탈정(소화효소)'(199900001)을(를) 골랐습니다."
    ]
    assert got.needs_pharmacist_check and got.match_candidates == []              # API로 정한 줄도 언제나 약사 확인
    assert api.calls == ["훼스탈정"]                                               # API에는 함량 등을 뺀 검색어를 보낸다


@pytest.mark.parametrize("drug", [line(), line("게보린정", None), line("프론크논캡슐", "900000051"), line("타이레놀정 5oOmg", "900000011")])
def test_the_api_is_not_called_when_the_dictionary_decides(drug):
    api = FakeApi(error=AssertionError("부르면 안 된다"))
    assert match_line(drug, DICT, api=api).match_method in ("code", "name")
    assert api.calls == []


@pytest.mark.parametrize("entry", DICT.entries, ids=lambda e: e.name)
def test_the_api_is_never_asked_about_a_drug_the_dictionary_knows(entry):
    api = FakeApi(error=AssertionError("부르면 안 된다"))
    for printed in (entry.item_code, None, "900000099"):
        assert match_line(line(entry.name, printed), DICT, api=api).item_code == entry.item_code
    assert api.calls == []


def test_without_an_api_nothing_changes():
    # 훼스탈정은 사전 8종 어느 것과도 0.6보다 덜 비슷하다
    assert match_line(line("훼스탈정", None), DICT).check_reasons == [
        "처방전에서 코드를 읽지 못했습니다. 이름으로도 사전의 약을 정하지 못했습니다."
    ]


def test_a_misread_name_is_still_a_candidate_when_the_api_finds_nothing():
    # OCR이 '프론크논캡슐'을 '프룬그논캡슐'로 읽었다(보정 목록에 없음). API는 이 이름으로 아무것도 찾지 못한다. 사전의 약이 후보로 남는다
    api = FakeApi()
    got = match_line(line("프룬그논캡슐", "900000051"), DICT, api=api)
    assert (got.item_code, got.match_method) == (None, "none")
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [(FRONK, "이름이 비슷함")]
    assert got.check_reasons == ["처방전 코드(900000051)로는 사전의 약을 찾지 못했습니다. 사전과 의약품 API 어디에서도 약을 정하지 못했습니다."]
    assert api.calls == ["프룬그논캡슐"]


def test_api_results_that_are_not_clear_become_candidates():
    api = FakeApi(ApiDrug("199900001", "훼스탈정200mg", "가"), ApiDrug("199900003", "훼스탈정400mg", "나"))
    got = match_line(line("훼스탈정", None), DICT, api=api)
    assert got.item_code is None and got.match_method == "none"           # 함량이 다른 두 제품: 2등과 차이가 없다
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [
        ("199900001", "의약품 API 검색"), ("199900003", "의약품 API 검색"),
    ]
    assert got.check_reasons == [f"{NOT_WHAT_IT_SAYS} 사전과 의약품 API 어디에서도 약을 정하지 못했습니다."]


def test_the_strength_written_on_the_prescription_picks_the_product():
    api = FakeApi(ApiDrug("199900001", "훼스탈정200밀리그램", "가"), ApiDrug("199900003", "훼스탈정400밀리그램", "가"))
    got = match_line(line("훼스탈정 4OOmg", None), DICT, api=api)
    assert got.item_code == "199900003" and got.match_method == "api"
    assert api.calls == ["훼스탈정"]


def test_the_api_spelling_of_the_unit_is_understood():
    # 식약처 제품명은 '밀리그람'으로 적기도 하고 성분명을 괄호로 붙인다
    api = FakeApi(ApiDrug("199900010", "훼스탈정200밀리그람(소화효소)", "가"), ApiDrug("199900011", "훼스탈정400밀리그람(소화효소)", "가"))
    assert match_line(line("훼스탈정 200mg", None), DICT, api=api).item_code == "199900010"


def test_a_strength_conflict_with_the_only_api_product_is_not_chosen():
    got = match_line(line("훼스탈정 400mg", None), DICT, api=FakeApi(ApiDrug("199900001", "훼스탈정200mg", "가")))
    assert got.item_code is None and got.match_candidates[0].item_code == "199900001"     # 고르지는 않지만 후보로는 보인다


def test_one_clear_api_product_is_enough():
    got = match_line(line("훼스탈정", None), DICT, api=FakeApi(FESTAL))
    assert got.item_code == "199900001"


def test_no_api_result_leaves_the_line_unmatched():
    got = match_line(line("알모란정", "900000031"), DICT, api=FakeApi())
    assert got.item_code is None
    assert [c.item_code for c in got.match_candidates] == [ARPEN]       # 사전 후보는 그대로 보인다
    assert got.check_reasons == ["처방전 코드(900000031)로는 사전의 약을 찾지 못했습니다. 사전과 의약품 API 어디에서도 약을 정하지 못했습니다."]


def test_an_api_failure_is_a_reason_not_a_crash():
    api = FakeApi(error=DrugApiError("의약품 API에 연결하지 못했습니다 (URLError)."))
    got = match_line(line("훼스탈정", None, days=5), DICT, api=api)
    assert got.item_code is None and got.needs_pharmacist_check
    assert got.check_reasons == [
        "처방전에서 코드를 읽지 못했습니다. 사전에 없어 의약품 API로 찾으려 했지만 실패했습니다(E_EXT_API): 의약품 API에 연결하지 못했습니다 (URLError).",
        "투약 일수 5일이 트레이 3일보다 깁니다.",
    ]


def test_dictionary_candidates_are_kept_when_the_api_fails():
    got = match_line(line("게보린", None), DICT, api=FakeApi(error=DrugApiError("x")))
    assert got.item_code is None and [c.item_code for c in got.match_candidates] == [GEVORIN]


def test_other_errors_are_not_hidden():
    with pytest.raises(ConfigError):
        match_line(line("훼스탈정", None), DICT, api=FakeApi(error=ConfigError("DRUG_API_KEY가 설정되지 않았습니다.")))


def test_dictionary_and_api_candidates_are_shown_together_by_score():
    api = FakeApi(ApiDrug("199900005", "게보린정에스", "가"), ApiDrug("199900006", "게보", "나"))
    got = match_line(line("게보린", None), DICT, api=api)
    assert got.item_code is None
    assert [(c.item_code, c.reason) for c in got.match_candidates] == [
        (GEVORIN, "이름이 비슷함"), ("199900006", "의약품 API 검색"), ("199900005", "의약품 API 검색"),
    ]                                       # 0.824 / '게보' 0.727 / '게보린정에스' 0.667
    scores = [c.score for c in got.match_candidates]
    assert scores == sorted(scores, reverse=True)


def test_api_candidates_are_limited_too():
    api = FakeApi(*[ApiDrug(f"19990000{i}", "훼스탈정" + "가나다라마"[:i], "가") for i in range(1, 6)])
    assert len(match_line(line("훼스탈정", None), DICT, api=api).match_candidates) == MAX_CANDIDATES


def test_api_candidates_below_the_threshold_are_dropped():
    got = match_line(line("훼스탈정", None), DICT, api=FakeApi(ApiDrug("199900009", "전혀다른캡슐", "가")))
    assert got.match_candidates == []


def test_the_api_candidate_threshold_is_inclusive_and_scores_are_rounded(monkeypatch):
    import agent.drug_match as dm

    api = FakeApi(ApiDrug("199900009", "훼스탈", "가"))
    score = similarity("훼스탈정", "훼스탈")
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score)
    got = match_line(line("훼스탈정", None), DICT, api=api)
    assert [(c.item_code, c.score) for c in got.match_candidates] == [("199900009", round(score, 3))]
    assert round(score, 3) != score                          # 반올림이 실제로 일어나는 값이다
    monkeypatch.setattr(dm, "CANDIDATE_MIN", score + 0.001)
    assert match_line(line("훼스탈정", None), DICT, api=api).match_candidates == []


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


def test_a_name_with_nothing_to_search_does_not_call_the_api():
    api = FakeApi(error=AssertionError("부르면 안 된다"))
    got = match_line(line("500mg", None), DICT, api=api)
    assert got.item_code is None and api.calls == []
    assert got.check_reasons == [f"{NOT_WHAT_IT_SAYS} 사전과 의약품 API 어디에서도 약을 정하지 못했습니다."]


def test_the_api_result_is_never_the_only_thing_a_pharmacist_relies_on():
    """API로 정한 줄도, 못 정한 줄도, API가 실패한 줄도 모두 약사 확인이 붙는다."""
    for api in (FakeApi(FESTAL), FakeApi(), FakeApi(error=DrugApiError("x"))):
        assert match_line(line("훼스탈정", None), DICT, api=api).needs_pharmacist_check


# ------------------------------------------------------------ 3-1. 조제지시서 전체에서의 API


def order_with(*names_and_codes):
    from agent.schema import DispensingOrder

    return DispensingOrder(
        prescription_id="P0001", warnings=["처음부터 있던 경고"],
        items=[line(n, c, item_code=None).model_copy(update={"times": ["morning"]}) for n, c in names_and_codes],
    )


def test_match_order_passes_the_api_on_and_asks_only_about_unknown_drugs():
    order = order_from_extracted("P0001", extract_rule(load("cv_a1", "rx_001")), source="rule")
    api = FakeApi(ApiDrug("199900031", "알모탄정", "가"))
    got = match_order(order, DICT, api=api)
    # OCR의 '알모란정'과 '알모탄정'은 0.909라 API가 이 제품을 돌려주면 고른다 (언제나 약사 확인과 함께)
    assert [d.item_code for d in got.items] == [TYLENOL, "199900031"]
    assert [d.match_method for d in got.items] == ["name", "api"] and got.items[1].needs_pharmacist_check
    assert api.calls == ["알모란정"]                     # 사전에 있는 타이레놀정은 API를 부르지 않는다
    assert got.warnings == order.warnings                # 실패하지 않았으면 경고를 더하지 않는다


def test_an_api_failure_adds_one_warning_to_the_order_and_stops_asking():
    api = FakeApi(error=DrugApiError("의약품 API에 연결하지 못했습니다 (URLError)."))
    order = order_with(("훼스탈정", None), ("게보린정", None), ("판피린정", None), ("알마겔정", None))
    got = match_order(order, DICT, api=api)
    assert api.calls == ["훼스탈정"]                      # 첫 줄에서 실패한 뒤로는 부르지 않는다 (줄마다 제한 시간을 기다리지 않게)
    assert [d.item_code for d in got.items] == [None, GEVORIN, None, None]          # 사전이 정한 게보린정은 그대로
    assert got.warnings == [
        "처음부터 있던 경고",
        "의약품 API를 쓰지 못해 약품 사전만으로 약을 찾았습니다(E_EXT_API): 의약품 API에 연결하지 못했습니다 (URLError).",
    ]
    for i in (0, 2, 3):                                  # 나머지 줄도 같은 사유를 이유로 갖는다
        assert "실패했습니다(E_EXT_API): 의약품 API에 연결하지 못했습니다 (URLError)." in got.items[i].check_reasons[0]
        assert got.items[i].needs_pharmacist_check
    assert got.items[1].match_method == "name" and got.items[1].needs_pharmacist_check


def test_the_service_continues_with_the_dictionary_when_the_api_is_down():
    order = order_from_extracted("P0001", extract_rule(load("cv_a1", "rx_004")), source="rule")
    with_api = match_order(order, DICT, api=FakeApi(error=DrugApiError("x")))
    without_api = match_order(order, DICT)
    assert [d.item_code for d in with_api.items] == [d.item_code for d in without_api.items]     # 사전이 정한 것은 같다
    assert with_api.flagged_indices == without_api.flagged_indices == [0, 1, 2, 3]


def test_a_recovered_api_is_asked_again_in_the_next_order():
    order = order_with(("훼스탈정", None))
    assert match_order(order, DICT, api=FakeApi(error=DrugApiError("x"))).items[0].item_code is None
    assert match_order(order, DICT, api=FakeApi(FESTAL)).items[0].item_code == "199900001"


def test_match_order_with_an_api_that_works_adds_no_warning():
    order = order_with(("훼스탈정", None))
    assert match_order(order, DICT, api=FakeApi()).warnings == ["처음부터 있던 경고"]


def test_match_order_keeps_working_without_an_api():
    order = order_with(("훼스탈정", None))
    assert match_order(order, DICT).items[0].match_method == "none"


# ------------------------------------------------------------ 4. 코드 재대조 (사전 8종 대조와 MCP 도구 check_item_code가 쓴다)


def test_the_code_is_confirmed_when_the_api_has_a_product_with_that_code():
    products = [ApiDrug("202005623", "타이레놀정160밀리그람", "가"), ApiDrug(TYLENOL, "타이레놀정500밀리그람(아세트아미노펜)", "(주)한국얀센")]
    got = check_item_code(TYLENOL, "타이레놀정500mg", products)
    assert got.verdict == CODE_FOUND == "일치"
    assert [p.item_code for _, p in got.similar] == [TYLENOL, "202005623"]            # 이름이 비슷한 순서
    assert got.similar[0][0] == 1.0 > got.similar[1][0]


def test_a_different_code_is_reported_with_the_similar_products():
    got = check_item_code(ARPEN, "아르펜정", [ApiDrug("200299999", "아르펜정", "가"), ApiDrug("200211111", "아르펜정400밀리그램", "가")])
    assert got.verdict == CODE_DIFFERENT == "불일치"
    assert [p.item_code for _, p in got.similar] == ["200299999", "200211111"]


def test_no_product_means_not_found():
    assert check_item_code(ARPEN, "아르펜정", []) == (CODE_NOT_FOUND, []) and CODE_NOT_FOUND == "없음"


def test_check_item_code_does_not_call_the_network():
    # 제품 목록은 부른 쪽이 가져와 넘긴다. 이 함수는 계산만 한다
    assert check_item_code(GEVORIN, "게보린정", [ApiDrug(GEVORIN, "게보린정", None)]).verdict == "일치"
