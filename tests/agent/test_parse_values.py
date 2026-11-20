import pytest

from agent.structure import parse_times, unread_times_text

# 용법 글자 -> 복용 시점. (숫자 칸은 test_split_columns.py의 classify_cell이 읽는다)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("아침 점심 저녁", ["morning", "noon", "evening"]),
        ("아침 저녁", ["morning", "evening"]),
        ("저녁 아침", ["morning", "evening"]),         # 적힌 순서와 상관없이 고정 순서
        ("점심", ["noon"]),
        ("아침/저녁 식후", ["morning", "evening"]),
        ("아 침 저 녁", ["morning", "evening"]),        # 글자 사이 공백
        ("조식 중식 석식", ["morning", "noon", "evening"]),
        ("아침 아침", ["morning"]),                     # 중복은 하나로
        ("취침 전", ["bedtime"]),
        ("취침전", ["bedtime"]),
        ("자기 전", ["bedtime"]),
        ("잠자기 전", ["bedtime"]),
        ("저녁 식후 취침 전", ["evening", "bedtime"]),
        ("취침 전 저녁", ["evening", "bedtime"]),       # 적힌 순서와 상관없이 고정 순서
        ("아침 점심 저녁 취침 전", ["morning", "noon", "evening", "bedtime"]),
        ("PO 취짐 전", ["bedtime"]),                    # 11/19 시험: OCR이 '취침'을 '취짐'으로 읽음
        ("PO 취칭 전", ["bedtime"]),
        ("취 짐 전", ["bedtime"]),                      # 글자 사이 공백
        ("저녁 취짐 전", ["evening", "bedtime"]),
        ("취전", []),                                   # 가운데 글자가 없으면 취침으로 보지 않는다
        ("취하여 전", []),                              # 글자 하나가 아니면 아니다
        ("아침저녁", ["morning", "evening"]),
        ("매 식후", []),
        ("PO 식후 30분", []),                           # 실제 처방전의 용법: 시점이 적혀 있지 않다
        ("식후 30분", []),
        ("", []),
    ],
)
def test_parse_times(text, expected):
    assert parse_times(text) == expected


def test_full_width_spaces_inside_a_word_are_ignored():
    assert parse_times("자기　전") == ["bedtime"]                  # 전각 공백. 반각으로 바꿔야 '자기전'이 된다
    assert parse_times("아침　저녁") == ["morning", "evening"]


def test_every_slot_has_words_and_nothing_else_does():
    # 복용 시점 값이 한 곳(agent.schema.TIME_SLOTS)에서만 정해지도록, 여기 단어표가 그 값들과 정확히 같아야 한다.
    from agent.schema import TIME_SLOTS
    from agent.structure import SLOT_WORDS

    assert list(SLOT_WORDS) == list(TIME_SLOTS)


# 시점을 못 찾았을 때, 정상적인 말을 빼고도 한글이 남으면 시점 글자를 못 알아본 것으로 본다 (11/20)
@pytest.mark.parametrize(
    "text, expected",
    [
        ("PO 식후 30분", None),            # 원래 시점이 없는 처방 (rx8 샘플 전부)
        ("PO 식전 30분", None),
        ("PO 식후30분", None),
        ("", None),
        ("PO 취침 전", None),                # 시점을 찾았다
        ("PO 점심 식후", None),
        ("PO 취짐 전", None),                # '취짐'도 시점으로 읽히니(취 + 글자 + 전) 해당 없다
        ("PO 수면 전", "수면 전"),           # 시점 단어가 아니고 정상적인 말도 아니다
        ("PO 식후 30분 쥐 짐", "쥐 짐"),
        ("P0 4후 3O분", "후 분"),            # 심하게 깨진 글자의 한글 조각
        ("PO", None),                       # 한글이 없다
        ("%$ 12", None),
    ],
)
def test_unread_times_text_is_the_hangul_left_after_the_plain_words(text, expected):
    assert unread_times_text(text) == expected
