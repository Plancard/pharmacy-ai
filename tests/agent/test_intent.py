"""글의 의도 가르기 (agent/intent.py) — 규칙."""
import pytest

from agent.intent import PATIENT_INFO, EDIT_REQUEST, INTENT_LABELS, INTENTS, NOTE, QUESTION, WORDS, classify, compact


@pytest.mark.parametrize("text, intents", [
    ("타이레놀 먹고 있어요", [PATIENT_INFO]),
    ("복용 중인 약: 게보린", [PATIENT_INFO]),
    ("원래 먹는 약이 있어요", [PATIENT_INFO]),
    ("게보린 빼 주세요", [EDIT_REQUEST]),
    ("2일분으로 바꿔 주세요", [EDIT_REQUEST]),
    ("뮤코다인 대신 트리스펜", [EDIT_REQUEST]),
    ("타이레놀 추가해 주세요", [EDIT_REQUEST]),
    ("같이 먹어도 되나요", [QUESTION]),
    ("부작용 알려 주세요", [QUESTION]),
    ("이거 왜 두 개예요?", [QUESTION]),
    ("메모: 오후 수령", [NOTE]),
    ("참고로 할머니 약입니다", [NOTE]),
    ("타이레놀 먹고 있는데 같이 먹어도 되나요?", [PATIENT_INFO, QUESTION]),
    ("게보린 빼 주세요. 괜찮을까요?", [EDIT_REQUEST, QUESTION]),
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
    assert classify("타이레놀 먹고 있어요?") == [PATIENT_INFO, QUESTION]      # 순서는 늘 INTENTS 순서


def test_spaces_and_full_width_letters_do_not_matter():
    assert classify("먹 고 있 어 요") == [PATIENT_INFO]
    assert classify("같이 먹어도 되나요？") == [QUESTION]                      # 전각 물음표
    assert compact(" 가 나\t다 ") == "가나다"


def test_the_order_and_labels_of_the_intents():
    assert INTENTS == ("patient_info", "edit_request", "question", "note")  # 기능분석서 v0.3 4.2절
    assert set(INTENT_LABELS) == set(INTENTS) == set(WORDS)
    assert all(word == compact(word) for words in WORDS.values() for word in words)   # 찾는 말에는 공백이 없다
