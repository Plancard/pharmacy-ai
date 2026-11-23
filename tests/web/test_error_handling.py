"""오류 처리 화면 (11/23, 기능분석서 6장 · REQ-NF-06): 오류별 원인 표시와 약사 선택지.

  E_CV_INFER(트레이)  다시 올리기 + 육안 검수 후 수동 진행(사유 필수, 검수 결과 없이 완료)
  E_CV_INFER(처방전)  [다시 시도] (같은 사진으로 새 처방)
  E_DB               [저장 다시 시도] (같은 처방, 조제 기록 저장만)
  그 밖의 실패        [새 사진으로 다시 올리기] (새 처방, 실패한 처방은 취소됨). E_DB는 제외

DB는 test_prescriptions.FakeRx, Agent는 가짜 연결부(FakeGateway) + cv/stub.
실행: python -m pytest -q tests/web/test_error_handling.py
"""
from __future__ import annotations

import io
import uuid

import pytest

from test_prescriptions import JPG, PNG, client, fake_rx, make_client, upload  # noqa: F401  (fixture)
from test_tray_upload import detail, tray, waiting_tray
from web import workflow


def post(client, path, **kw):
    return client.post(path, headers={"X-CSRF-Token": client.token}, **kw)


def retake(client, rx_id, data=JPG, name="rx2.jpg"):
    return post(client, f"/api/prescriptions/{rx_id}/retake", data={"image": (io.BytesIO(data), name)},
                content_type="multipart/form-data")


def manual(client, rx_id, reason):
    return post(client, f"/api/prescriptions/{rx_id}/manual-inspection", json={"reason": reason})


@pytest.fixture
def cv_error(monkeypatch):
    """트레이 검수에서 검출 모델이 실패한다 (CVError = E_CV_INFER)."""
    from cv import stub

    def broken(*a, **k):
        raise stub.CVError("모델 가중치를 읽지 못함")

    monkeypatch.setattr(stub, "inspect", broken)


def stuck_on_cv_error(client):
    rx_id = waiting_tray(client)
    assert tray(client, rx_id).status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "waiting_tray"                       # 실패가 아니라 트레이 대기 (TRAY_RETRY_CODES)
    return rx_id, d


# ------------------------------------------------------------ 트레이 검출 모델 오류 → 육안 검수 후 수동 진행
def test_cv_error_on_tray_offers_retake_and_manual_inspection(client, cv_error):
    _, d = stuck_on_cv_error(client)
    retake_info = d["card"]["retake"]
    assert retake_info["code"] == "E_CV_INFER" and retake_info["manual_allowed"] is True
    assert "육안" in retake_info["message"]


def test_tray_not_found_does_not_offer_manual_inspection(client, monkeypatch):
    rx_id = waiting_tray(client)
    monkeypatch.setenv("CV_STUB_SCENARIO", "no_tray")
    tray(client, rx_id)
    retake_info = detail(client, rx_id)["card"]["retake"]
    assert retake_info["code"] == "E_TRAY_NOT_FOUND" and retake_info["manual_allowed"] is False
    assert manual(client, rx_id, "눈으로 봤음").status_code == 409     # 다시 찍으면 되는 오류는 수동 진행 없음


def test_manual_inspection_finishes_with_reason_and_no_inspection(client, fake_rx, cv_error):
    rx_id, _ = stuck_on_cv_error(client)
    r = manual(client, rx_id, "  조제 계획과 12칸 육안 대조, 이상 없음  ")
    assert r.status_code == 200 and r.get_json()["status"] == "done"

    d = detail(client, rx_id)
    assert d["status"] == "done" and d["card"]["inspection"] is None and d["card"]["order"]
    assert d["timeline"][-1]["event"] == "response" and d["timeline"][-1]["action"] == "manual"
    a = fake_rx.approvals[-1]
    assert (a["hitl_point"], a["action"], a["reason"]) == ("inspection", "manual", "조제 계획과 12칸 육안 대조, 이상 없음")
    [rec] = fake_rx.dispense_rows
    assert rec["inspection"] is None and rec["order"]["prescription_id"] == rx_id and rec["plan"]["plan"]
    assert manual(client, rx_id, "두 번째").status_code == 409        # 이미 완료


@pytest.mark.parametrize("reason", [None, "", "   ", 3, "가" * 501])
def test_manual_inspection_needs_a_reason(client, fake_rx, cv_error, reason):
    rx_id, _ = stuck_on_cv_error(client)
    assert manual(client, rx_id, reason).status_code == 400
    assert detail(client, rx_id)["status"] == "waiting_tray" and fake_rx.dispense_rows == []


def test_manual_inspection_only_while_waiting_tray(client, fake_rx):
    rx_id = upload(client).get_json()["id"]                    # 승인 대기 (트레이 오류 없음)
    assert manual(client, rx_id, "사유").status_code == 409


def test_manual_inspection_of_other_pharmacy_is_404(client, fake_rx, cv_error):
    rx_id, _ = stuck_on_cv_error(client)
    fake_rx.rows[uuid.UUID(rx_id)]["pharmacy_id"] = 2
    assert manual(client, rx_id, "사유").status_code == 404


# ------------------------------------------------------------ 조제 기록 저장 실패(E_DB) → 저장 다시 시도
def test_db_failure_offers_save_retry_not_retake(client, fake_rx):
    rx_id = waiting_tray(client)
    fake_rx.fail_done = True
    tray(client, rx_id)
    d = detail(client, rx_id)
    assert d["status"] == "failed" and d["error_code"] == "E_DB"
    assert (d["save_retryable"], d["retake_allowed"], d["retryable"]) == (True, False, False)
    assert retake(client, rx_id).status_code == 409           # 검수까지 끝난 처방을 처음부터 다시 하지 않는다

    fake_rx.fail_done = False
    r = post(client, f"/api/prescriptions/{rx_id}/save-retry")
    assert r.status_code == 202
    assert detail(client, rx_id)["status"] == "done"
    [rec] = fake_rx.dispense_rows
    assert rec["inspection"]["summary"]["hitl_required"] is False      # 같은 검수 결과로 저장
    assert post(client, f"/api/prescriptions/{rx_id}/save-retry").status_code == 409   # 두 번째는 거절


def test_save_retry_that_fails_again_stays_e_db(client, fake_rx):
    rx_id = waiting_tray(client)
    fake_rx.fail_done = True
    tray(client, rx_id)
    assert post(client, f"/api/prescriptions/{rx_id}/save-retry").status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "failed" and d["error_code"] == "E_DB" and d["save_retryable"] is True


def test_save_retry_only_for_db_failures(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="llm_fail")
    rx_id = upload(c).get_json()["id"]
    assert post(c, f"/api/prescriptions/{rx_id}/save-retry").status_code == 409


# ------------------------------------------------------------ 실패한 처방 → 새 사진으로 다시 올리기
def test_ocr_empty_failure_can_be_retaken_with_a_new_photo(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="fail")      # E_OCR_EMPTY
    old = upload(c).get_json()["id"]
    d = detail(c, old)
    assert d["error_code"] == "E_OCR_EMPTY"
    assert (d["retryable"], d["retake_allowed"], d["save_retryable"]) == (False, True, False)

    c.application.extensions["agent"].scenario = "warning"
    r = retake(c, old, PNG, "new.png")
    assert r.status_code == 202
    new = r.get_json()
    assert new["id"] != old and detail(c, old)["status"] == "cancelled"
    assert detail(c, new["id"])["status"] == "waiting_approval"
    key = fake_rx.first_image(uuid.UUID(new["id"]))["storage_key"]
    assert key.endswith(".png") and (tmp_path / key).read_bytes() == PNG   # 새로 올린 사진이 붙었다
    assert retake(c, old).status_code == 409                               # 이미 취소됨


def test_retake_needs_an_image_and_a_failed_rx(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="fail")
    rx_id = upload(c).get_json()["id"]
    assert retake(c, rx_id, b"GIF89a", "a.gif").status_code == 400
    assert detail(c, rx_id)["status"] == "failed"                          # 거절되면 그대로
    c2 = make_client(monkeypatch, tmp_path / "b", scenario="warning")
    ok = upload(c2).get_json()["id"]
    assert retake(c2, ok).status_code == 409                               # 실패가 아닌 처방


def test_cv_infer_at_ocr_is_retryable():
    assert "E_CV_INFER" in workflow.RETRY_CODES and "E_DB" not in workflow.RETRY_CODES


# ------------------------------------------------------------ 조제 이력: 육안 검수 기록
def test_history_marks_manual_inspection(client, fake_rx, cv_error, monkeypatch):
    import web.rx as rx

    rx_id, _ = stuck_on_cv_error(client)
    manual(client, rx_id, "육안 대조 이상 없음")
    rid = uuid.UUID(rx_id)
    rec = fake_rx.dispense_rows[0]
    monkeypatch.setattr(rx, "dispense_record", lambda r, p: {
        "id": rid, "display_no": "P0001", "created_at": rec["completed_at"], "completed_at": rec["completed_at"],
        "username": "ph1", "risk_level": rec["risk_level"], "rx_json": rec["order"], "plan_json": rec["plan"],
        "inspection_json": None} if r == rid else None)
    monkeypatch.setattr(rx, "approval_log", lambda r: [
        {**a, "username": "ph1"} for a in fake_rx.approvals if a["rx"] == r])
    html = client.get(f"/history/{rx_id}").get_data(as_text=True)
    assert "육안 검수(수동 진행)" in html and "육안 대조 이상 없음" in html and "올린 트레이 사진" in html
    assert "트레이 촬영" not in html                                       # 검수 결과 표는 그리지 않는다
