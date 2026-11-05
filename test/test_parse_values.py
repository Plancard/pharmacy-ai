import pytest

from agent.structure import parse_time_slots

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
        ("아침저녁", ["morning", "evening"]),
        ("매 식후", []),
        ("PO 식후 30분", []),                           # 실제 처방전의 용법: 시점이 적혀 있지 않다
        ("식후 30분", []),
        ("", []),
    ],
)
def test_parse_time_slots(text, expected):
    assert parse_time_slots(text) == expected


def test_full_width_spaces_inside_a_word_are_ignored():
    assert parse_time_slots("자기　전") == ["bedtime"]                  # 전각 공백. 반각으로 바꿔야 '자기전'이 된다
    assert parse_time_slots("아침　저녁") == ["morning", "evening"]


def test_every_slot_has_words_and_nothing_else_does():
    # 복용 시점 값이 한 곳(agent.schema.SLOT_ORDER)에서만 정해지도록, 여기 단어표가 그 값들과 정확히 같아야 한다.
    from agent.schema import SLOT_ORDER
    from agent.structure import SLOT_WORDS

    assert list(SLOT_WORDS) == list(SLOT_ORDER)
