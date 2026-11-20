"""실패 카드 (11/17 회의 Q6): 구조화에 실패한 줄 강조와 [다시 시도].

DB는 test_prescriptions.FakeRx, Agent는 가짜 연결부의 llm_fail 시나리오(E_LLM + 강조할 줄)를 쓴다.
실행: python -m pytest tests/web -q
"""
from __future__ import annotations

import uuid

import pytest

from test_prescriptions import JPG, PNG, fake_rx, make_client, upload   # noqa: F401  (fixture 이름 그대로 쓴다)
from web import workflow


@pytest.fixture
def llm_client(monkeypatch, tmp_path, fake_rx):
    return make_client(monkeypatch, tmp_path, scenario="llm_fail")


def set_scenario(client, scenario):
    client.application.extensions["agent"].scenario = scenario      # 가짜 연결부의 시나리오를 중간에 바꾼다


def detail(client, rx_id):
    return client.get(f"/api/prescriptions/{rx_id}").get_json()


def retry(client, rx_id):
    return client.post(f"/api/prescriptions/{rx_id}/retry", headers={"X-CSRF-Token": client.token})


def test_llm_failure_card_has_failed_lines_and_retry(llm_client):
    rx_id = upload(llm_client).get_json()["id"]
    d = detail(llm_client, rx_id)
    assert d["status"] == "failed" and d["error_code"] == "E_LLM"
    assert d["retryable"] is True
    fl = d["failed_lines"]
    assert fl["ocr_size"]["width"] > 0 and fl["ocr_size"]["height"] > 0
    assert fl["lines"] and all(len(l["bbox"]) == 4 for l in fl["lines"])


@pytest.mark.parametrize("scenario", ["fail", "range", "warning"])
def test_other_outcomes_have_no_failed_lines_or_retry(monkeypatch, tmp_path, fake_rx, scenario):
    """E_OCR_EMPTY(fail)는 다시 찍어야 하므로 [다시 시도]가 없고, 실패가 아닌 처방은 강조할 줄도 없다."""
    c = make_client(monkeypatch, tmp_path, scenario=scenario)
    d = detail(c, upload(c).get_json()["id"])
    assert d["retryable"] is False and d["failed_lines"] is None


def test_retry_creates_new_rx_and_cancels_failed_one(llm_client, fake_rx, tmp_path):
    old = upload(llm_client, JPG).get_json()
    assert detail(llm_client, old["id"])["retryable"]
    set_scenario(llm_client, "warning")      # 이번에는 AI가 응답한다
    r = retry(llm_client, old["id"])
    assert r.status_code == 202
    new = r.get_json()
    assert new["id"] != old["id"] and new["display_no"] != old["display_no"]
    assert detail(llm_client, old["id"])["status"] == "cancelled"
    d = detail(llm_client, new["id"])
    assert d["status"] == "waiting_approval"           # 처음부터 다시 돌아 승인 대기까지 왔다
    # 같은 처방전 사진이 새 처방에 붙었다
    old_key = fake_rx.first_image(uuid.UUID(old["id"]))["storage_key"]
    new_key = fake_rx.first_image(uuid.UUID(new["id"]))["storage_key"]
    assert old_key != new_key
    assert (tmp_path / old_key).read_bytes() == (tmp_path / new_key).read_bytes() == JPG
    # 확장자도 그대로
    assert new_key.endswith(".jpg")


def test_retry_keeps_png_extension(llm_client, fake_rx, tmp_path):
    old = upload(llm_client, PNG, "rx.png").get_json()
    new = retry(llm_client, old["id"]).get_json()
    assert fake_rx.first_image(uuid.UUID(new["id"]))["storage_key"].endswith(".png")


def test_retry_twice_makes_only_one_new_rx(llm_client, fake_rx):
    old = upload(llm_client).get_json()["id"]
    assert retry(llm_client, old).status_code == 202
    assert retry(llm_client, old).status_code == 409           # 이미 취소됨 → 새 처방은 하나만
    assert len(fake_rx.rows) == 2


@pytest.mark.parametrize("code", ["E_OCR_EMPTY", "E_PLAN_RANGE", "E_DB", None])
def test_retry_refused_for_non_transient_errors(llm_client, fake_rx, code):
    rx_id = upload(llm_client).get_json()["id"]
    fake_rx.rows[uuid.UUID(rx_id)].update(error_code=code)
    assert retry(llm_client, rx_id).status_code == 409
    assert detail(llm_client, rx_id)["status"] == "failed"      # 그대로 둔다


@pytest.mark.parametrize("code", sorted(workflow.RETRY_CODES))
def test_retry_allowed_for_transient_errors(llm_client, fake_rx, code):
    rx_id = upload(llm_client).get_json()["id"]
    fake_rx.rows[uuid.UUID(rx_id)].update(error_code=code)
    set_scenario(llm_client, "none")
    assert retry(llm_client, rx_id).status_code == 202


def test_retry_refused_when_not_failed(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="warning")
    rx_id = upload(c).get_json()["id"]                          # 승인 대기
    assert retry(c, rx_id).status_code == 409


def test_retry_missing_photo_is_refused_and_keeps_failed(llm_client, fake_rx, tmp_path):
    rx_id = upload(llm_client).get_json()["id"]
    (tmp_path / fake_rx.first_image(uuid.UUID(rx_id))["storage_key"]).unlink()
    assert retry(llm_client, rx_id).status_code == 409
    assert detail(llm_client, rx_id)["status"] == "failed"      # 취소하지 않고 그대로 (사진이 없으면 다시 올려야 한다)
