"""글을 제안으로 만들기 (agent/proposal.py)."""
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import dictionary_from_data, load_dictionary
from agent.intent import PATIENT_INFO, EDIT_REQUEST, NOTE, QUESTION
from agent.proposal import PROPOSAL_KINDS, PROPOSAL_LABELS, make_proposals, mentioned_drugs, stem

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")


def names(drugs):
    return [d["name"] for d in drugs]


def test_the_stem_drops_strength_brackets_and_form():
    assert [stem(e.name) for e in DICT.entries] == ["타이레놀", "게보린", "트리스펜", "프론크논", "아르펜", "아이월드요비신", "트립라인", "뮤코다인"]


@pytest.mark.parametrize("text, found", [
    ("타이레놀 먹고 있어요", ["타이레놀정500mg"]),
    ("게보린정 빼 주세요", ["게보린정"]),
    ("요비신 복용 중", ["아이월드요비신정 416.7mg"]),                     # 이름의 뒷부분(3글자 이상)
    ("뮤코다인 대신 트리스펜", ["트리스펜연질캡슐", "뮤코다인캡슐200mg"]),     # 사전 순서
    ("타 이 레 놀", ["타이레놀정500mg"]),
    ("2일분으로 바꿔 주세요", []),
    ("레놀", []),                                                       # 두 글자로는 정하지 않는다
])
def test_drugs_whose_name_appears_in_the_text(text, found):
    assert names(mentioned_drugs(text, DICT)) == found


def test_mentioned_drugs_carry_the_item_code():
    assert mentioned_drugs("게보린", DICT) == [{"item_code": "197900277", "name": "게보린정"}]


def test_a_proposal_per_usable_intent():
    got = make_proposals(" 게보린 먹고 있는데 타이레놀 빼 주세요 ", [PATIENT_INFO, EDIT_REQUEST, QUESTION], DICT)
    assert [p["kind"] for p in got] == [EDIT_REQUEST, PATIENT_INFO]
    assert all(p["text"] == "게보린 먹고 있는데 타이레놀 빼 주세요" for p in got)
    assert names(got[0]["drugs"]) == ["타이레놀정500mg", "게보린정"]
    json.dumps(got)


@pytest.mark.parametrize("intents", [[], [QUESTION], [NOTE], [QUESTION, NOTE]])
def test_no_proposal_without_an_edit_or_a_current_med(intents):
    assert make_proposals("게보린 빼 주세요", intents, DICT) == []


@pytest.mark.parametrize("text", [None, "", "  "])
def test_no_proposal_for_an_empty_text(text):
    assert make_proposals(text, [EDIT_REQUEST], DICT) == []


def test_another_dictionary_finds_its_own_drugs():
    small = dictionary_from_data({"drugs": [{"item_code": "100000001", "name": "가나다라정"}]})
    assert names(mentioned_drugs("나다라 먹어요", small)) == ["가나다라정"]


def test_the_kinds_and_labels():
    assert PROPOSAL_KINDS == ("edit_request", "patient_info")
    assert PROPOSAL_LABELS == {"edit_request": "수정 제안", "patient_info": "복용 약 후보"}
