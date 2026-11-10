import pytest

from agent.drug_names import (
    extract_strengths, fix_ocr_digits, jamo, match_key, normalize_name, product_key, search_term, similarity,
)

# 눈으로 구별하기 어려운 글자는 코드 값으로 적는다
SQUARE_UG = "㎍"        # ㎍
SQUARE_MG = "㎎"        # ㎎
SQUARE_ML = "㎖"        # ㎖
MICRO_SIGN = "µ"       # µ (마이크로 기호)
GREEK_MU = "μ"         # μ (그리스 문자)


# ------------------------------------------------------------ normalize_name


@pytest.mark.parametrize(
    "written",
    [
        "타이레놀정500mg",
        "타이레놀정 500mg",
        "타이레놀정 500 mg",
        "타이레놀정500밀리그램",
        "타이레놀정 500 밀리그램",
        "타이레놀정 500 밀리 그램",     # OCR이 단위를 끊어 읽은 경우
        "타이레놀정(500 mg)",
        "[타이레놀정] 500mg",
        "타이레놀정-500mg",
        f"타이레놀정500{SQUARE_MG}",
        "타이레놀정500MG",
        "타이레놀정５００ｍｇ",          # 전각 숫자·영문
        "  타이레놀정500mg  ",
    ],
)
def test_spelling_variants_of_the_same_name_become_one_key(written):
    assert normalize_name(written) == "타이레놀정500mg"


@pytest.mark.parametrize(
    "written",
    ["A정 100마이크로그램", "A정 100 마이크로 그램", f"A정100{SQUARE_UG}", f"A정100{MICRO_SIGN}g", f"A정100{GREEK_MU}g", "A정100mcg"],
)
def test_microgram_spellings_become_mcg(written):
    assert normalize_name(written) == "a정100mcg"


@pytest.mark.parametrize(
    "written, expected",
    [
        ("시럽 5밀리리터", "시럽5ml"),
        (f"시럽 5{SQUARE_ML}", "시럽5ml"),
        ("A정 0.5그램", "a정0.5g"),
        ("A정 500밀리그램", "a정500mg"),       # "그램"만 바꿔 "500밀리g"가 되면 안 된다
        ("A크림 0.5%", "a크림0.5%"),
        ("비타민D 1000IU", "비타민d1000iu"),
    ],
)
def test_other_units_are_unified(written, expected):
    assert normalize_name(written) == expected


def test_unit_words_are_changed_only_right_after_a_number():
    assert normalize_name("프로그램정") == "프로그램정"
    assert normalize_name("프로그램정 500밀리그램") == "프로그램정500mg"


def test_strength_is_kept_so_different_products_stay_different():
    assert normalize_name("부루펜정200mg") != normalize_name("부루펜정400mg")
    assert normalize_name("A정0.5mg") != normalize_name("A정5mg")        # 소수점이 지워지면 안 된다
    assert normalize_name("A정 1,000mg") == "a정1000mg"


def test_dots_are_kept_inside_numbers_but_not_at_the_ends():
    assert normalize_name("A정 0.5g") == "a정0.5g"
    assert normalize_name(".정.") == "정"


@pytest.mark.parametrize("empty", ["", "   ", None, "()", "-", "..."])
def test_nothing_left_gives_an_empty_key(empty):
    assert normalize_name(empty) == ""


def test_fullwidth_and_case():
    assert normalize_name("ＡＢＣ 정") == "abc정"
    assert normalize_name("Tylenol ER") == "tylenoler"


def test_the_whole_hangul_syllable_range_is_kept():
    assert normalize_name("가힣") == "가힣"       # 첫 글자와 마지막 글자


@pytest.mark.parametrize("name", ["타이레놀정 500 밀리 그램", f"A정100{SQUARE_UG}", "[부루펜정] 200mg", "ＡＢＣ 정", "A정 0.5%"])
def test_normalizing_twice_changes_nothing(name):
    once = normalize_name(name)
    assert normalize_name(once) == once


def test_different_products_do_not_collapse_into_one_key():
    names = ["타이레놀정500mg", "게보린정", "알마겔정", "베아제정", "부루펜정200mg", "부루펜정400mg", "판피린티정", "훼스탈플러스정"]
    assert len({normalize_name(n) for n in names}) == len(names)


# ------------------------------------------------------------ extract_strengths


@pytest.mark.parametrize(
    "name, expected",
    [
        ("부루펜정200mg", {"200mg"}),
        ("부루펜정 200 밀리그램", {"200mg"}),
        ("A정 0.5g", {"0.5g"}),
        ("A정 100마이크로그램", {"100mcg"}),      # mg로 잘못 읽으면 안 된다
        ("A정 100mcg", {"100mcg"}),
        ("시럽 5ml", {"5ml"}),
        ("비타민D 1000IU", {"1000iu"}),
        ("A크림 0.5%", {"0.5%"}),
        ("복합제 10mg/5ml", {"10mg", "5ml"}),
        ("비타민C 1,000mg", {"1000mg"}),
        ("비타민B12정", set()),
        ("게보린정", set()),
        ("", set()),
        (None, set()),
    ],
)
def test_extract_strengths(name, expected):
    assert extract_strengths(name) == expected


# ------------------------------------------------------------ fix_ocr_digits / match_key


@pytest.mark.parametrize(
    "written, fixed",
    [
        ("타이레놀정 5oOmg", "타이레놀정 500mg"),          # 실제 OCR 오류: 소문자 o, 대문자 O가 섞여 있다
        ("에스코렌정 1Omg", "에스코렌정 10mg"),
        ("킬렉시아정 25Omg", "킬렉시아정 250mg"),
        ("타세늘8시간이말서방정 65Omg", "타세늘8시간이말서방정 650mg"),
        ("타이레놀정5oOmg", "타이레놀정500mg"),            # 이름 바로 뒤에 붙어 있어도
        ("타이레놀정 5oO mg", "타이레놀정 500 mg"),         # 숫자와 단위 사이에 공백이 있어도
        ("A정 5oO㎎", "A정 500mg"),                       # ㎎는 NFKC가 mg로 바꾼다
        ("A정 1o㎍", "A정 10μg"),
        ("타이레놀정 5o0밀리그램", "타이레놀정 500밀리그램"),
        ("비타민D 1OOOIU", "비타민D 1000IU"),
        ("A정 5oO MG", "A정 500 MG"),
        ("A정 o5mg", "A정 05mg"),                         # 숫자가 하나 있으면 앞의 o도 0
        ("A정 １Ｏmg", "A정 10mg"),                        # 전각 숫자·알파벳
    ],
)
def test_o_inside_a_strength_is_turned_back_into_zero(written, fixed):
    assert fix_ocr_digits(written) == fixed


@pytest.mark.parametrize(
    "text",
    [
        "Omg",                  # 숫자가 하나도 없으면 단위 앞의 O라도 건드리지 않는다
        "Cipro500mg",           # 영문 이름 안의 o
        "Tylenol 500mg",
        "필루비서방정 145mg",      # 숫자를 잘못 읽은 것은 고칠 수 없다
        "알모란정",
        "오메가3 1000mg",
        "아모정 5omm",           # 단위가 아니면 그대로
        "A정 5o",               # 단위가 없으면 그대로
        "",
    ],
)
def test_text_that_is_not_a_misread_strength_is_left_alone(text):
    assert fix_ocr_digits(text) == text


def test_fix_ocr_digits_accepts_none():
    assert fix_ocr_digits(None) == ""


def test_the_fix_changes_only_the_zeros():
    assert fix_ocr_digits("타이레놀정 5oOmg 환자용 o") == "타이레놀정 500mg 환자용 o"


def test_fixing_twice_changes_nothing():
    once = fix_ocr_digits("타이레놀정 5oOmg")
    assert fix_ocr_digits(once) == once


def test_match_key_makes_a_misread_name_equal_to_the_dictionary_spelling():
    assert match_key("타이레놀정 5oOmg") == match_key("타이레놀정/500mg") == "타이레놀정500mg"


def test_match_key_keeps_other_misreadings_so_they_are_visible_to_similarity():
    assert match_key("프론크논램술") == "프론크논램술"
    assert match_key(None) == ""


# ------------------------------------------------------------ jamo / similarity


def test_jamo_splits_hangul_syllables_and_leaves_other_characters():
    assert jamo("탄") == "\u1110\u1161\u11ab"          # ㅌ ㅏ ㄴ
    assert jamo("가a1") == "\u1100\u1161a1"
    assert jamo("") == ""


def test_identical_names_are_fully_similar():
    assert similarity("게보린정", "게보린정") == 1.0


def test_one_jamo_apart_is_still_very_similar():
    # 팀 안내: 알모탄정 / 알모란정은 자모 유사도 약 0.909다. 비슷해 보여도 서로 다른 약이다.
    assert round(similarity("알모탄정", "알모란정"), 3) == 0.909


def test_the_same_name_as_a_dictionary_spelling_with_a_misread_syllable_stays_above_point_seven():
    # 실제 샘플의 OCR 오류가 있는 이름들. 매칭의 '같은 이름' 기준(0.7)을 넘어야 한다.
    assert round(similarity("프론크논램술", "프론크논캡슐"), 3) == 0.812
    assert round(similarity("프론크논캠술", "프론크논캡슐"), 3) == 0.875
    assert similarity("트리스편연질캠술", "트리스펜연질캡슐") > 0.8


def test_a_jamo_apart_scores_higher_than_a_whole_name_apart():
    assert similarity("알모탄정", "알모란정") > similarity("알모탄정", "알마겔정")


def test_unrelated_names_are_not_similar():
    assert similarity("가나다정", "게보린정") < 0.6
    assert similarity("알모란정", "게보린정") < 0.6          # 샘플에서 사전의 약과 가장 비슷한 대상 외 약


def test_very_short_names_are_not_similar_to_longer_ones():
    assert similarity("정", "게보린정") < 0.6
    assert similarity("정제", "타이레놀정제") < 0.6


def test_a_name_inside_a_longer_name_is_only_partly_similar():
    # 한쪽이 다른 쪽에 통째로 들어 있어도 자동으로 높은 점수를 주지 않는다 (이름만으로 약을 정하지 않는다)
    assert 0.6 < similarity("부루펜정", "부루펜정200mg") < 0.85
    assert similarity("베아제", "베아제정500mg") < 0.7


def test_order_of_the_two_names_does_not_matter():
    assert similarity("알모탄정", "알모란정") == similarity("알모란정", "알모탄정")


def test_empty_name_has_zero_similarity():
    assert similarity("", "abc") == 0.0
    assert similarity("abc", "") == 0.0
    assert similarity("", "") == 0.0


def test_same_name_with_a_different_strength_looks_very_similar():
    # 이름만 보면 거의 같다. 그래서 매칭 단계에서 함량을 따로 비교한다 (agent/drug_match.py)
    assert similarity(normalize_name("타이레놀정500mg"), normalize_name("타이레놀정400mg")) >= 0.9


def test_similarity_stays_between_zero_and_one():
    for a, b in [("abc", "abd"), ("부루펜정", "부루펜"), ("가", "나다라")]:
        assert 0.0 <= similarity(a, b) <= 1.0


# ------------------------------------------------------------ search_term / product_key (의약품 API용, 5단계)


@pytest.mark.parametrize(
    "name, term",
    [
        ("타이레놀정 5oOmg", "타이레놀정"),             # OCR의 o를 0으로 고친 뒤 함량을 뺀다
        ("타이레놀정/500mg", "타이레놀정"),
        ("타이레놀정 500MG", "타이레놀정"),             # 대문자 단위
        ("타이레놀정500밀리그램(아세트아미노펜)", "타이레놀정"),
        ("뮤코다인캡슐200mg", "뮤코다인캡슐"),
        ("아이월드요비신정 416.7mg", "아이월드요비신정"),   # 소수점 함량
        ("에스코렌정 / 1Omg", "에스코렌정"),
        ("타세놀8시간이알서방정 650mg", "타세놀8시간이알서방정"),   # 이름 속 숫자(8시간)는 함량이 아니라 남는다
        ("게보린정", "게보린정"),
        ("아르펜 정", "아르펜정"),
        ("", ""),
        (None, ""),
        ("500mg", ""),                                 # 함량뿐이면 검색할 이름이 없다
    ],
)
def test_search_term(name, term):
    assert search_term(name) == term


def test_the_search_term_of_every_dictionary_drug_is_its_name_without_the_strength():
    from agent.config import RESOURCES_DIR
    from agent.drug_dictionary import load_dictionary

    terms = [search_term(e.name) for e in load_dictionary(RESOURCES_DIR / "drug_dictionary.json").entries]
    assert terms == ["타이레놀정", "게보린정", "트리스펜연질캡슐", "프론크논캡슐", "아르펜정", "아이월드요비신정", "트립라인정", "뮤코다인캡슐"]


def test_product_key_drops_the_ingredient_in_parentheses():
    assert product_key("타이레놀정500밀리그램(아세트아미노펜)") == "타이레놀정500mg"
    assert product_key("트립라인정(브로멜라인)") == "트립라인정"
    assert product_key("게보린정") == "게보린정"
    assert product_key("") == "" and product_key(None) == ""


@pytest.mark.parametrize("api_name", [
    "타이레놀정500밀리그램(아세트아미노펜)",
    "타이레놀정500밀리그람(아세트아미노펜)",       # 식약처 자료는 '밀리그람'으로 적기도 한다
    "타이레놀정 500 밀리그람",
])
def test_an_api_product_name_equals_the_dictionary_name_after_the_key(api_name):
    assert product_key(api_name) == match_key("타이레놀정500mg") == "타이레놀정500mg"


@pytest.mark.parametrize(
    "written, expected",
    [
        ("A정 500밀리그람", "a정500mg"), ("A정 100마이크로그람", "a정100mcg"), ("A정 0.5그람", "a정0.5g"),
        ("프로그램정", "프로그램정"), ("프로그람정 500밀리그람", "프로그람정500mg"),     # 숫자 뒤에서만 단위다
    ],
)
def test_the_other_spelling_of_unit_words_is_unified_too(written, expected):
    assert normalize_name(written) == expected
    assert extract_strengths("부루펜정 200밀리그람") == {"200mg"}


def test_the_other_spelling_works_after_an_ocr_zero_fix_as_well():
    assert fix_ocr_digits("타이레놀정 5oO밀리그람") == "타이레놀정 500밀리그람"
    assert match_key("타이레놀정 5oO밀리그람") == "타이레놀정500mg"
