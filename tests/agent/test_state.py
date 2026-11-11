"""그래프의 상태 (agent/state.py)."""
import json
import operator
import typing

from agent.schema import DocType
from agent.state import FLOWS, PHOTO_KINDS, STATUSES, GraphState, initial_state


def hints():
    return typing.get_type_hints(GraphState, include_extras=True)


def test_the_first_state_has_every_field_of_the_graph_state():
    state = initial_state("P0001")
    assert set(state) == set(hints())


def test_the_first_state_values():
    assert initial_state("P0001") == {
        "prescription_id": "P0001", "photo_kind": None, "ocr": None, "text": None, "route": [], "flow": None,
        "proposals": [], "memo": None, "order": None, "plan": None,
        "status": "received", "notice": None, "error_code": None, "errors": [], "notes": [],
    }


def test_the_text_of_the_message_is_carried_into_the_state():
    state = initial_state("P0001", None, None, "게보린 빼 주세요")
    assert state["text"] == "게보린 빼 주세요" and state["ocr"] is None and state["photo_kind"] is None


def test_ocr_and_photo_kind_are_carried_into_the_state():
    ocr = {"image_id": "x", "doc_type": "med_bag", "lines": []}
    state = initial_state("P0001", ocr, "med_bag")
    assert state["ocr"] is ocr and state["photo_kind"] == "med_bag"
    assert initial_state("P0001", ocr)["photo_kind"] is None       # 사진 종류를 안 주면 None (OCR의 doc_type으로 가린다)


def test_each_first_state_has_its_own_lists():
    a, b = initial_state("P0001"), initial_state("P0002")
    for key in ("errors", "notes", "route", "proposals"):
        a[key].append("x")
        assert b[key] == []


def test_the_first_state_can_be_stored_as_json():
    json.dumps(initial_state("P0001", {"image_id": "x", "lines": []}, "prescription"))


def test_errors_and_notes_are_added_up_and_other_fields_are_overwritten():
    h = hints()
    assert h["errors"].__metadata__ == (operator.add,) and h["notes"].__metadata__ == (operator.add,)
    assert not any(hasattr(h[key], "__metadata__") for key in h if key not in ("errors", "notes"))


def test_the_values_the_state_can_take():
    assert PHOTO_KINDS == ("prescription", "tray", "med_bag", "package")
    assert FLOWS == ("prescription", "recheck", "tray", "ask_type", "register_meds", "consult")
    assert STATUSES == (
        "received", "structured", "matched", "checked", "approved", "edited", "planned", "not_ready", "noted", "failed",
    )


def test_every_flow_has_a_place_in_the_order_of_the_supervisor():
    from agent.supervisor import FLOW_ORDER

    assert sorted(FLOW_ORDER) == sorted(FLOWS)


def test_a_photo_kind_exists_for_every_document_type_the_ocr_can_read():
    assert set(typing.get_args(DocType)) <= set(PHOTO_KINDS)
