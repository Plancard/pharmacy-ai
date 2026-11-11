"""글의 의도 가르기 (agent/intent.py) — 규칙."""
import pytest

from agent.intent import CURRENT_MED, EDIT, INTENT_LABELS, INTENTS, MEMO, QUESTION, WORDS, classify, compact


@pytest.mark.parametrize("text, intents", [
    ("타이레놀 먹고 있어요", [CURRENT_MED]),
    ("복용 중인 약: 게보린", [CURRENT_MED]),
    ("원래 먹는 약이 있어요", [CURRENT_MED]),
    ("게보린 빼 주세요", [EDIT]),
    ("2일분으로 바꿔 주세요", [EDIT]),
    ("뮤코다인 대신 트리스펜", [EDIT]),
    ("타이레놀 추가해 주세요", [EDIT]),
    ("같이 먹어도 되나요", [QUESTION]),
    ("부작용 알려 주세요", [QUESTION]),
    ("이거 왜 두 개예요?", [QUESTION]),
    ("메모: 오후 수령", [MEMO]),
    ("참고로 할머니 약입니다", [MEMO]),
    ("타이레놀 먹고 있는데 같이 먹어도 되나요?", [CURRENT_MED, QUESTION]),
    ("게보린 빼 주세요. 괜찮을까요?", [EDIT, QUESTION]),
])
def test_the_intents_of_a_text(text, intents):
    assert classify(text) == intents


@pytest.mark.parametrize("text", ["오후 3시", "할머니", "타이레놀", "ㅇㅋ"])
def test_a_text_without_a_known_word_is_unclear(text):
    assert classify(text) == []


@pytest.mark.parametrize("text", [None, "", "   ", "\n"])
def test_an_empty_text_has_no_intent(text):
    assert classify(text) == []


def test_a_question_mark_alone_makes_a_question():
    assert classify("타이레놀?") == [QUESTION]
    assert classify("타이레놀 먹고 있어요?") == [CURRENT_MED, QUESTION]      # 순서는 늘 INTENTS 순서


def test_spaces_and_full_width_letters_do_not_matter():
    assert classify("먹 고 있 어 요") == [CURRENT_MED]
    assert classify("같이 먹어도 되나요？") == [QUESTION]                      # 전각 물음표
    assert compact(" 가 나\t다 ") == "가나다"


def test_the_order_and_labels_of_the_intents():
    assert INTENTS == ("current_med", "edit", "question", "memo")
    assert set(INTENT_LABELS) == set(INTENTS) == set(WORDS)
    assert all(word == compact(word) for words in WORDS.values() for word in words)   # 찾는 말에는 공백이 없다
