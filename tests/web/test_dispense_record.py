"""조제 기록 저장 (11/20, 기능분석서 F-WEB-12 ②, REQ-WEB-06): 완료될 때 최종 조제지시서·조제 계획·검수 결과를 남긴다.

웹 DB는 test_prescriptions.py의 가짜(FakeRx)라 SQL(트랜잭션·COALESCE·CHECK)은 여기서 확인하지 못한다.
새 SQL은 개발 모드에서 직접 처방을 끝내 보고 DB에서 확인한다 (웹_도커_실행_절차의 '자주 막히는 곳').
실행: python -m pytest tests/web/test_dispense_record.py -q
"""
from __future__ import annotations

import uuid

import pytest

import web.rx as rx
from test_prescriptions import approve, client, fake_rx, make_client, upload  # noqa: F401  (client · fake_rx는 fixture)
from test_tray_upload import agent_gateway, at_hitl_2, backend, detail, inspect_reply, tray, waiting_tray  # noqa: F401  (backend는 fixture)
from web.agent_gateway import FakeGateway


def test_all_ok_saves_one_record_with_order_plan_and_inspection(client, fake_rx):
    rx_id = waiting_tray(client)
    image_id = tray(client, rx_id).get_json()["image_id"]
    assert detail(client, rx_id)["status"] == "done"

    [rec] = fake_rx.dispense_rows
    assert rec["rx"] == uuid.UUID(rx_id)
    assert rec["order"]["prescription_id"] == rx_id and rec["order"]["items"]       # 승인된 조제지시서(A.5)
    assert rec["plan"]["prescription_id"] == rx_id and rec["plan"]["plan"]          # 조제 계획(A.4)
    assert rec["inspection"]["image_id"] == image_id and len(rec["inspection"]["cells"]) == 12   # 검수 결과(A.3)
    assert rec["risk_level"] == rec["order"]["risk_check"]["level"]                 # 위험 단계는 조제지시서 값
    assert rec["user_id"] == fake_rx.approvals[-1]["user_id"]                       # 마지막 HITL 응답자


def test_confirm_after_hitl_2_saves_the_record_with_the_confirming_user(client, fake_rx, monkeypatch):
    rx_id = at_hitl_2(client, monkeypatch)
    assert fake_rx.dispense_rows == []                       # HITL ②에서 멈춰 있는 동안은 기록이 없다
    assert inspect_reply(client, rx_id, action="confirm").status_code == 202
    [rec] = fake_rx.dispense_rows
    assert rec["inspection"]["summary"]["hitl_required"] is True      # 불일치가 있던 검수 결과 그대로
    assert rec["user_id"] == fake_rx.approvals[-1]["user_id"] and fake_rx.approvals[-1]["action"] == "confirm"


def test_redo_then_a_new_photo_saves_a_single_record_with_the_last_inspection(client, fake_rx, monkeypatch):
    rx_id = at_hitl_2(client, monkeypatch)
    assert inspect_reply(client, rx_id, action="redo").status_code == 202
    assert fake_rx.dispense_rows == []                       # 재조제는 완료가 아니다
    monkeypatch.setenv("CV_STUB_SCENARIO", "ok")
    image_id = tray(client, rx_id).get_json()["image_id"]
    [rec] = fake_rx.dispense_rows
    assert rec["inspection"]["image_id"] == image_id and rec["inspection"]["summary"]["hitl_required"] is False


def test_cancelled_while_processing_saves_no_record(client, fake_rx):
    """처리 중에 취소했으면 늦게 끝난 검수가 완료·조제 기록을 만들지 않는다(조건부 UPDATE 계약)."""
    rx_id = waiting_tray(client)
    key = uuid.UUID(rx_id)
    fake_rx.rows[key]["status"] = "cancelled"
    assert fake_rx.set_done(key, {"x": 1}, {"y": 1}, {"z": 1}) is False
    assert fake_rx.dispense_rows == [] and fake_rx.rows[key]["status"] == "cancelled"


def test_a_failed_save_leaves_a_db_failure_and_no_record(client, fake_rx):
    rx_id = waiting_tray(client)
    fake_rx.fail_done = True
    assert tray(client, rx_id).status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "failed" and d["error_message"].startswith("저장 중 오류")    # E_DB 안내문
    assert fake_rx.dispense_rows == []


def test_a_missing_plan_is_a_db_failure_not_a_half_record(client, fake_rx, monkeypatch):
    """조제 계획을 못 가져오면 완료로 두지 않는다 (plan_json은 NOT NULL)."""
    monkeypatch.setattr(FakeGateway, "approved_plan", lambda self, rx_id: None)
    rx_id = waiting_tray(client)
    tray(client, rx_id)
    d = detail(client, rx_id)
    assert d["status"] == "failed" and fake_rx.dispense_rows == []


def test_record_on_the_real_agent_has_the_agents_order_plan_and_inspection(backend, monkeypatch, tmp_path, fake_rx):
    """진짜 Agent(가짜 LangGraph, langgraph가 있으면 진짜도): 승인 → 트레이 전 칸 일치 → 조제 기록."""
    c = make_client(monkeypatch, tmp_path)
    c.application.extensions["agent"] = agent_gateway(None)
    monkeypatch.setenv("CV_STUB_SCENARIO", "ok")
    rx_id = upload(c).get_json()["id"]
    assert approve(c, rx_id, confirmed=True).status_code == 202
    image_id = tray(c, rx_id).get_json()["image_id"]
    assert detail(c, rx_id)["status"] == "done"

    [rec] = fake_rx.dispense_rows
    assert rec["order"]["items"] and rec["order"]["prescription_id"] == rx_id
    assert rec["plan"]["plan"] and rec["plan"]["prescription_id"] == rx_id
    assert rec["inspection"]["image_id"] == image_id
    assert rec["risk_level"] in rx.RISK_LEVELS or rec["risk_level"] is None


@pytest.mark.parametrize("order, expected", [
    ({"risk_check": {"level": "danger"}}, "danger"),
    ({"risk_check": {"level": "none"}}, "none"),
    ({"risk_check": {"level": "unknown"}}, None),     # 목록 밖 값은 CHECK에 걸리므로 NULL로 둔다
    ({"risk_check": None}, None),
    ({}, None),
    (None, None),
])
def test_risk_level_of(order, expected):
    assert rx.risk_level_of(order) == expected
