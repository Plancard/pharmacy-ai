"""그래프의 상태 (agent/state.py)."""
import json
import operator
import typing

from agent.schema import DocType
import pytest

from agent.state import FLOWS, MESSAGE_KEYS, PHOTO_KINDS, STATUSES, GraphState, initial_state, message_input, read_message


def hints():
    return typing.get_type_hints(GraphState, include_extras=True)


def test_the_first_state_has_every_field_of_the_graph_state():
    state = initial_state("P0001")
    assert set(state) == set(hints())


def test_the_first_state_values():
    assert initial_state("C0001") == {
        "conversation_id": "C0001", "prescription_id": None,
        "message_id": None, "image_id": None, "photo_kind": None, "ocr": None, "text": None, "intents": [],
        "route": [], "flow": None, "proposals": [], "memo": None, "order": None, "plan": None, "inspection": None,
        "status": "received", "notice": None, "error_code": None, "errors": [], "notes": [],
    }


def test_a_prescription_conversation_carries_its_prescription_id():
    assert initial_state("P0001", None, "P0001")["prescription_id"] == "P0001"


def test_the_message_is_carried_into_the_state():
    message = {"message_id": "m1", "text": "게보린 빼 주세요", "image_id": "img1", "kind": "prescription"}
    state = initial_state("P0001", message, "P0001")
    assert (state["message_id"], state["text"], state["image_id"], state["photo_kind"]) == ("m1", "게보린 빼 주세요", "img1", "prescription")


def test_ocr_and_photo_kind_are_carried_into_the_state():
    ocr = {"image_id": "x", "doc_type": "med_bag", "lines": []}
    state = initial_state("P0001", {"kind": "med_bag"}, "P0001", ocr)
    assert state["ocr"] is ocr and state["photo_kind"] == "med_bag"
    assert initial_state("P0001", None, None, ocr)["photo_kind"] is None    # 사진 종류를 안 주면 None (OCR의 doc_type으로 가린다)


def test_a_message_with_an_unknown_key_is_refused():
    assert MESSAGE_KEYS == ("message_id", "text", "image_id", "kind")
    with pytest.raises(ValueError, match="모르는 키"):
        read_message({"kind": "tray", "photo": b"..."})          # 원본 바이트는 담지 않는다 (image_id만)


def test_the_next_message_does_not_carry_the_order_or_the_plan():
    # 같은 대화의 다음 메시지 입력에 order · plan · status가 있으면 앞서 승인된 조제지시서와 계획이 지워진다 (11/12 요청서)
    data = message_input({"message_id": "m2", "kind": "tray", "image_id": "img2"})
    assert not {"order", "plan", "status", "conversation_id", "errors", "notes"} & set(data)
    assert "prescription_id" not in data
    assert (data["message_id"], data["photo_kind"], data["image_id"], data["intents"]) == ("m2", "tray", "img2", [])


def test_a_message_that_opens_a_prescription_carries_its_id():
    assert message_input({"kind": "prescription"}, prescription_id="C0001")["prescription_id"] == "C0001"


def test_each_first_state_has_its_own_lists():
    a, b = initial_state("P0001"), initial_state("P0002")
    for key in ("errors", "notes", "route", "proposals", "intents"):
        a[key].append("x")
        assert b[key] == []


def test_the_first_state_can_be_stored_as_json():
    json.dumps(initial_state("P0001", {"kind": "prescription"}, "P0001", {"image_id": "x", "lines": []}))


def test_errors_and_notes_are_added_up_and_other_fields_are_overwritten():
    h = hints()
    assert h["errors"].__metadata__ == (operator.add,) and h["notes"].__metadata__ == (operator.add,)
    assert not any(hasattr(h[key], "__metadata__") for key in h if key not in ("errors", "notes"))


def test_the_values_the_state_can_take():
    assert PHOTO_KINDS == ("prescription", "tray", "med_bag", "package")
    assert FLOWS == ("prescription", "recheck", "tray", "ask_type", "register_meds", "consult")
    assert STATUSES == (
        "received", "structured", "matched", "checked", "approved", "edited", "planned", "inspecting", "inspected",
        "not_ready", "noted", "failed",
    )


def test_every_flow_has_a_place_in_the_order_of_the_supervisor():
    from agent.supervisor import FLOW_ORDER

    assert sorted(FLOW_ORDER) == sorted(FLOWS)


def test_a_photo_kind_exists_for_every_document_type_the_ocr_can_read():
    assert set(typing.get_args(DocType)) <= set(PHOTO_KINDS)
