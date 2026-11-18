"""HITL ① 수정(edit)으로 만든 조제지시서가 진짜 Agent의 조제지시서 규칙(agent.schema.DispensingOrder)을 통과하는가.

11/18 통합 확인: 실제 OCR이 '트립라인정'을 '트럭라인정'으로 읽어 이름을 고치자, 웹이 item_code만 비우고
match_method='code'를 남겨 Agent가 'match_method=code이면 item_code가 있어야 합니다'로 거절했다.
"""
import pytest

pytest.importorskip("pydantic")

from agent.schema import DispensingOrder  # noqa: E402
from web.workflow import build_edited_order  # noqa: E402

TRIPLINE = "200002714"


def agent_order():
    """실제 Agent가 승인 카드에 넘긴 모양 (코드로 매칭된 줄, OCR이 이름을 틀리게 읽음)."""
    return {
        "prescription_id": "rx-1",
        "items": [
            {"item_code": "197900277", "name": "게보린정", "count_per_dose": 1, "times": ["morning", "evening"], "days": 3,
             "source_line_ids": [40], "printed_code": "197900277", "ocr_conf": 0.9, "match_method": "code"},
            {"item_code": TRIPLINE, "name": "트럭라인정", "count_per_dose": 7, "times": ["bedtime"], "days": 3,
             "source_line_ids": [44], "printed_code": TRIPLINE, "ocr_conf": 0.8, "match_method": "code",
             "needs_pharmacist_check": True, "check_reasons": ["코드와 이름이 다릅니다"],
             "match_candidates": [{"item_code": TRIPLINE, "name": "트립라인정", "score": 0.8, "reason": "처방전 코드"}]},
        ],
        "warnings": [],
    }


def edits(order, **change):
    rows = [{"index": i, "name": it["name"], "count_per_dose": it["count_per_dose"], "times": it["times"], "days": it["days"]}
            for i, it in enumerate(order["items"])]
    rows[1].update(change)
    return rows


def test_renamed_item_passes_agent_schema_and_is_rematched():
    order = agent_order()
    new = build_edited_order(order, edits(order, name="트립라인정", count_per_dose=1))
    DispensingOrder.model_validate(new)                    # 예전에는 여기서 ValueError
    line = new["items"][1]
    assert line["item_code"] is None and line["count_per_dose"] == 1
    for key in ("match_method", "match_candidates", "printed_code", "needs_pharmacist_check", "check_reasons"):
        assert key not in line
    assert line["source_line_ids"] == [44]                 # 원래 줄 정보는 남긴다


def test_count_only_edit_keeps_the_match():
    order = agent_order()
    new = build_edited_order(order, edits(order, count_per_dose=1))
    DispensingOrder.model_validate(new)
    line = new["items"][1]
    assert line["item_code"] == TRIPLINE and line["match_method"] == "code" and line["printed_code"] == TRIPLINE


def test_new_item_passes_agent_schema():
    order = agent_order()
    rows = edits(order) + [{"index": None, "name": "타이레놀정500mg", "count_per_dose": 1, "times": ["noon"], "days": 3}]
    DispensingOrder.model_validate(build_edited_order(order, rows))
