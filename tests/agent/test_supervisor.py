"""슈퍼바이저의 계획 (agent/supervisor.py) — 메시지 하나의 흐름과 순서. 그래프 없이 함수로 시험한다."""
import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.supervisor import (
    ASK_TEXT_NOTICE,
    EDIT_WITHOUT_ORDER_NOTICE,
    FLOW_ORDER,
    NO_RX_PHOTO_NOTICE,
    MessagePlan,
    TEXT_CHOICES,
    choice_labels,
    has_photo,
    ordered,
    plan_after_choice,
    plan_message,
)

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
RX = {"doc_type": "prescription", "lines": []}
GEVORIN = {"item_code": "197900277", "name": "게보린정"}


def plan(photo_kind=None, ocr=None, text=None):
    return plan_message(photo_kind, ocr, text, DICT)


def test_the_order_of_the_flows():
    assert FLOW_ORDER == ("ask_type", "prescription", "register_meds", "recheck", "tray", "consult")
    assert ordered(["consult", "prescription", "consult", "ask_type"]) == ("ask_type", "prescription", "consult")


# ------------------------------------------------------------ 사진만


@pytest.mark.parametrize("photo_kind, ocr, route", [
    ("prescription", None, ("prescription",)), (None, RX, ("prescription",)), ("med_bag", None, ("recheck",)),
    ("tray", None, ("tray",)), (None, None, ("ask_type",)), ("receipt", None, ("ask_type",)),
    ("prescription", {"doc_type": "med_bag"}, ("ask_type",)),
])
def test_a_photo_alone(photo_kind, ocr, route):
    got = plan(photo_kind, ocr)
    assert (got.route, got.flow, got.proposals, got.memo) == (route, route[0], (), None)


def test_asking_about_a_photo_says_why_and_what_about():
    got = plan("receipt")
    assert got.ask_about == "photo" and "알 수 없는 사진 종류입니다." in got.notice


# ------------------------------------------------------------ 사진 + 글


def test_an_edit_request_with_a_prescription_photo_becomes_a_proposal_only():
    got = plan("prescription", RX, "게보린 빼 주세요")
    assert got.route == ("prescription",) and got.memo is None
    assert got.proposals == ({"kind": "edit_request", "text": "게보린 빼 주세요", "drugs": [GEVORIN]},)


def test_a_current_med_with_a_prescription_photo_becomes_a_candidate_not_a_registration():
    got = plan("prescription", RX, "게보린 먹고 있어요")
    assert got.route == ("prescription",)                              # 등록하지 않는다 (약사가 확인한 것만)
    assert [p["kind"] for p in got.proposals] == ["patient_info"]


def test_a_question_with_a_prescription_photo_adds_the_consult_after_it():
    got = plan("prescription", RX, "게보린 먹고 있는데 같이 먹어도 되나요?")
    assert got.route == ("prescription", "consult") and [p["kind"] for p in got.proposals] == ["patient_info"]


@pytest.mark.parametrize("text", ["메모: 오후 수령", "오후 3시"])
def test_a_memo_or_an_unclear_text_with_a_prescription_photo_is_kept_as_a_memo(text):
    got = plan("prescription", RX, text)
    assert (got.route, got.memo, got.proposals) == (("prescription",), text, ())


@pytest.mark.parametrize("text", ["게보린 빼 주세요", "같이 먹어도 되나요?", "게보린 먹고 있어요", "오후 3시"])
def test_any_text_with_a_tray_photo_is_only_a_memo(text):
    got = plan("tray", None, text)
    assert (got.route, got.memo, got.proposals) == (("tray",), text, ())


def test_a_med_bag_photo_takes_a_current_med_as_a_candidate_and_an_edit_as_a_memo():
    assert [p["kind"] for p in plan("med_bag", None, "게보린 먹고 있어요").proposals] == ["patient_info"]
    got = plan("med_bag", None, "게보린 빼 주세요")
    assert (got.route, got.proposals, got.memo) == (("recheck",), (), "게보린 빼 주세요")
    assert plan("med_bag", None, "괜찮나요?").route == ("recheck", "consult")


def test_an_unclear_photo_with_text_asks_about_the_photo_first():
    got = plan(None, None, None)
    assert got.route == ("ask_type",) and got.ask_about == "photo"
    got = plan("receipt", None, "게보린 빼 주세요")
    assert (got.route, got.ask_about, got.proposals) == (("ask_type",), "photo", ())


# ------------------------------------------------------------ 글만


@pytest.mark.parametrize("text, route, kinds, memo, notice", [
    ("같이 먹어도 되나요?", ("consult",), [], None, None),
    ("게보린 먹고 있어요", ("register_meds",), ["patient_info"], None, None),
    ("게보린 먹고 있는데 괜찮나요?", ("register_meds", "consult"), ["patient_info"], None, None),
    ("메모: 오후 수령", (), [], "메모: 오후 수령", None),
    ("게보린 빼 주세요", (), [], "게보린 빼 주세요", EDIT_WITHOUT_ORDER_NOTICE),
])
def test_text_alone(text, route, kinds, memo, notice):
    got = plan(None, None, text)
    assert (got.route, [p["kind"] for p in got.proposals], got.memo, got.notice) == (route, kinds, memo, notice)
    assert got.flow == (route[0] if route else None)


def test_text_alone_that_cannot_be_read_asks_its_meaning():
    got = plan(None, None, "오후 3시")
    assert (got.route, got.ask_about, got.notice) == (("ask_type",), "text", ASK_TEXT_NOTICE)


def test_what_counts_as_a_photo():
    assert has_photo("tray", None, "글") and has_photo(None, RX, "글") and has_photo(None, None, None) and has_photo(None, None, " ")
    assert not has_photo(None, None, "글")


# ------------------------------------------------------------ 되묻기 뒤


@pytest.mark.parametrize("choice, route", [("prescription", ("prescription",)), ("med_bag", ("recheck",)), ("tray", ("tray",))])
def test_after_choosing_a_photo_kind(choice, route):
    assert plan_after_choice("photo", choice, None, DICT).route == route


def test_after_choosing_the_prescription_kind_the_text_is_still_used():
    got = plan_after_choice("photo", "prescription", "게보린 빼 주세요", DICT)
    assert [p["kind"] for p in got.proposals] == ["edit_request"]


@pytest.mark.parametrize("choice, route, memo", [("question", ("consult",), None), ("patient_info", ("register_meds",), None), ("note", (), "오후 3시")])
def test_after_choosing_a_text_meaning(choice, route, memo):
    got = plan_after_choice("text", choice, "오후 3시", DICT)
    assert (got.route, got.memo) == (route, memo)


@pytest.mark.parametrize("about, choice", [("photo", "question"), ("text", "tray"), ("photo", None), ("text", "edit")])
def test_a_choice_from_the_other_list_is_refused(about, choice):
    with pytest.raises(ValueError, match="고를 수 있는 값이 아닙니다"):
        plan_after_choice(about, choice, "글", DICT)


def test_the_text_choices_and_their_labels():
    assert TEXT_CHOICES == ("patient_info", "question", "note")
    assert choice_labels("text") == {"patient_info": "복용 약 정보", "question": "질문", "note": "메모"}
    assert choice_labels("photo") == {"prescription": "처방전", "tray": "트레이", "med_bag": "약봉투", "package": "포장"}


def test_a_plan_becomes_the_fields_of_the_state():
    got = MessagePlan(route=("prescription", "consult"), flow="prescription", proposals=({"kind": "edit_request", "text": "t", "drugs": []},), memo="m")
    assert got.as_update() == {"route": ["prescription", "consult"], "flow": "prescription",
                               "proposals": [{"kind": "edit_request", "text": "t", "drugs": []}], "memo": "m", "intents": []}
    assert MessagePlan(route=(), notice="n").as_update()["notice"] == "n"


# ------------------------------------------------------------ 처방이 없는 대화 (상담 대화, 11/12 결정)


def chat(photo_kind=None, ocr=None, text=None):
    return plan_message(photo_kind, ocr, text, DICT, has_prescription=False)


@pytest.mark.parametrize("text", ["타이레놀이랑 게보린 같이 먹어도 되나요?", "게보린 먹고 있어요", "게보린 빼 주세요", "오후 3시"])
def test_text_alone_without_a_prescription_is_a_consultation(text):
    # 고칠 처방도 대조할 처방약도 없으므로 의도와 상관없이 상담으로 간다. 제안 · 메모는 만들지 않는다
    got = chat(text=text)
    assert (got.route, got.flow, got.proposals, got.memo, got.ask_about) == (("consult",), "consult", (), None, None)


def test_the_intents_are_still_recorded_in_a_consultation():
    assert chat(text="게보린 먹고 있는데 괜찮나요?").intents == ("patient_info", "question")


@pytest.mark.parametrize("photo_kind, ocr", [("prescription", None), (None, RX)])
def test_a_prescription_photo_without_a_prescription_starts_the_prescription_flow(photo_kind, ocr):
    got = chat(photo_kind, ocr, "같이 먹어도 되나요?")
    assert (got.route, got.flow) == (("prescription", "consult"), "prescription")


@pytest.mark.parametrize("photo_kind, label", [("tray", "트레이"), ("med_bag", "약봉투"), ("package", "포장")])
def test_other_photos_without_a_prescription_are_not_run_and_the_pharmacist_is_told(photo_kind, label):
    got = chat(photo_kind, None, "메모입니다")
    assert (got.route, got.flow, got.memo) == ((), None, "메모입니다")
    assert got.notice == NO_RX_PHOTO_NOTICE.format(kind=label) and "처방 대화에서 올려 주세요" in got.notice


def test_an_unclear_photo_without_a_prescription_is_still_asked_and_the_choice_follows_the_same_rule():
    assert chat().route == ("ask_type",)
    assert plan_after_choice("photo", "prescription", None, DICT, has_prescription=False).route == ("prescription",)
    assert plan_after_choice("photo", "tray", None, DICT, has_prescription=False).route == ()


def test_with_a_prescription_nothing_changes():
    assert plan(text="게보린 먹고 있는데 괜찮나요?").route == ("register_meds", "consult")
    assert plan("tray").route == ("tray",)
