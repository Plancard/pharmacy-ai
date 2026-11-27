"""라우터의 계획 (agent/router.py) — 사진 한 장의 흐름 (명세서: 처방전 / 트레이). 그래프 없이 함수로 시험한다."""
import pytest

from agent.router import (
    FLOW_ORDER,
    HAS_ORDER_NOTICE,
    NO_KIND_NOTICE,
    MessagePlan,
    plan_message,
)

RX = {"doc_type": "prescription", "lines": []}


def test_the_order_of_the_flows():
    assert FLOW_ORDER == ("prescription", "tray")      # 상담 · 복용 약 · 되묻기 흐름 없음


@pytest.mark.parametrize("photo_kind, ocr, route", [
    ("prescription", None, ("prescription",)),
    ("prescription", RX, ("prescription",)),
    (None, RX, ("prescription",)),                     # 종류가 없어도 OCR이 있으면 처방전
    ("tray", None, ("tray",)),
])
def test_a_photo_goes_to_its_flow(photo_kind, ocr, route):
    got = plan_message(photo_kind, ocr)
    assert got.route == route and got.flow == route[0] and got.notice is None


def test_no_kind_and_no_ocr_is_only_a_notice():
    got = plan_message(None, None)
    assert got.route == () and got.flow is None and got.notice == NO_KIND_NOTICE


@pytest.mark.parametrize("kind", ["med_bag", "package", "photo", ""])
def test_kinds_out_of_scope_are_not_run(kind):
    got = plan_message(kind, None)
    assert got.route == () and kind in got.notice


def test_an_ocr_that_is_not_a_prescription_is_not_run():
    got = plan_message("prescription", {"doc_type": "med_bag", "lines": []})
    assert got.route == () and "med_bag" in got.notice


def test_a_second_prescription_photo_for_the_same_prescription_is_not_run():
    got = plan_message("prescription", RX, has_order=True)
    assert got.route == () and got.notice == HAS_ORDER_NOTICE
    assert plan_message("tray", None, has_order=True).route == ("tray",)    # 트레이 사진은 조제지시서가 있어야 의미가 있다


def test_as_update_carries_only_route_flow_and_notice():
    assert MessagePlan(route=("tray",), flow="tray").as_update() == {"route": ["tray"], "flow": "tray"}
    assert MessagePlan(route=(), notice="안내").as_update() == {"route": [], "flow": None, "notice": "안내"}
