"""처방 취소 (11/19, F-5, REQ-WEB-07). 웹 DB는 test_prescriptions.py의 가짜(FakeRx)라 실제 SQL·DB 제약은 보지 않는다.

실제 DB 쪽(set_cancelled의 조건부 UPDATE, error_code CHECK)은 개발 모드에서 직접 눌러 확인한다(11/19 확인함).
실행: python -m pytest tests/web/test_cancel.py -q
"""
from __future__ import annotations

import uuid

from test_prescriptions import approve, client, fake_rx, make_client, upload  # noqa: F401  (client · fake_rx는 fixture)
from test_tray_upload import detail, tray, waiting_tray


def cancel(client, rx_id):
    return client.post(f"/api/prescriptions/{rx_id}/cancel", headers={"X-CSRF-Token": client.token})


def test_cancel_while_waiting_approval(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    assert detail(client, rx_id)["status"] == "waiting_approval"
    r = cancel(client, rx_id)
    assert r.status_code == 200 and r.get_json()["status"] == "cancelled"
    d = detail(client, rx_id)
    assert d["status"] == "cancelled" and d["hitl_point"] is None


def test_cancel_while_waiting_tray(client, fake_rx):
    rx_id = waiting_tray(client)
    assert cancel(client, rx_id).status_code == 200
    assert detail(client, rx_id)["status"] == "cancelled"


def test_cancel_a_failed_prescription_clears_error_code(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="fail")
    rx_id = upload(c).get_json()["id"]
    assert detail(c, rx_id)["status"] == "failed"
    assert cancel(c, rx_id).status_code == 200            # 실패 카드의 [삭제] = 취소됨 처리
    d = detail(c, rx_id)
    assert d["status"] == "cancelled" and d["error_code"] is None


def test_cannot_cancel_a_done_prescription(client, fake_rx):
    rx_id = waiting_tray(client)
    assert tray(client, rx_id).status_code == 202
    assert detail(client, rx_id)["status"] == "done"
    assert cancel(client, rx_id).status_code == 409       # 이력에 넘어간 건 취소할 수 없다
    assert detail(client, rx_id)["status"] == "done"


def test_cancel_twice_is_rejected(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    assert cancel(client, rx_id).status_code == 200
    assert cancel(client, rx_id).status_code == 409       # 더블 클릭


def test_cancelled_prescription_rejects_approve_and_tray(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    assert cancel(client, rx_id).status_code == 200
    assert approve(client, rx_id, confirmed=True).status_code == 409
    assert fake_rx.approvals == []                         # 승인 기록이 남지 않는다
    assert tray(client, rx_id).status_code == 409
    assert detail(client, rx_id)["status"] == "cancelled"


def test_cancelled_after_approval_cannot_receive_a_tray(client, fake_rx):
    rx_id = waiting_tray(client)
    assert cancel(client, rx_id).status_code == 200
    assert tray(client, rx_id).status_code == 409
    assert fake_rx.tray_images(uuid.UUID(rx_id)) == []     # 트레이 사진이 붙지 않는다


def test_late_background_result_does_not_overwrite_cancelled(client, fake_rx):
    """처리 중에 취소한 뒤 백그라운드 작업이 늦게 끝나도 취소됨을 덮어쓰지 않는다(조건부 UPDATE 계약)."""
    rx_id = upload(client).get_json()["id"]
    key = uuid.UUID(rx_id)
    fake_rx.rows[key]["status"] = "processing"             # 처리 중 상태로 되돌려 둔다
    assert cancel(client, rx_id).status_code == 200
    assert fake_rx.set_waiting(key, "rx_approval") is False
    assert fake_rx.set_waiting_tray(key) is False
    assert fake_rx.set_failed(key, "E_LLM") is False
    assert fake_rx.set_done(key) is False
    assert fake_rx.rows[key]["status"] == "cancelled"


def test_cancel_needs_csrf(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    r = client.post(f"/api/prescriptions/{rx_id}/cancel")
    assert r.status_code >= 400
    assert detail(client, rx_id)["status"] == "waiting_approval"


def test_cancel_unknown_or_other_pharmacy_is_404(client, fake_rx):
    assert cancel(client, str(uuid.uuid4())).status_code == 404
    rx_id = upload(client).get_json()["id"]
    fake_rx.rows[uuid.UUID(rx_id)]["pharmacy_id"] += 1000
    assert cancel(client, rx_id).status_code == 404
    assert fake_rx.rows[uuid.UUID(rx_id)]["status"] == "waiting_approval"
