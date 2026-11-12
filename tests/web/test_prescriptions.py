"""처방 접수 → 승인 대기 → HITL ① 승인 흐름 테스트.

DB는 web.rx 함수를 메모리 가짜로 바꾸고(SQL은 PG16에서 따로 확인), Agent는 가짜 연결부(FakeGateway)를
그대로 쓴다. 처리 작업은 JOBS_SYNC로 요청 안에서 바로 돈다.
실행: python -m pytest tests/web -q
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone

import pytest

import web.rx as rx
from test_app import FakeDB, csrf, login   # 같은 폴더의 로그인 도우미
import web.db as db
from web import create_app

JPG = b"\xff\xd8\xff\xe0" + b"0" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


class FakeRx:
    """web.rx와 같은 함수 이름. 상태 바꾸기의 '지금 상태가 이것일 때만' 조건도 똑같이 흉내 낸다."""

    def __init__(self):
        self.rows: dict[uuid.UUID, dict] = {}
        self.images: dict[uuid.UUID, dict] = {}
        self.approvals: list[dict] = []
        self.fail_create = False

    def create(self, pharmacy_id, user_id, rx_id, storage_key, kind="prescription"):
        if self.fail_create:
            raise RuntimeError("db down")
        n = sum(1 for r in self.rows.values() if r["pharmacy_id"] == pharmacy_id) + 1
        row = {"id": rx_id, "pharmacy_id": pharmacy_id, "display_no": f"P{n:04d}", "status": "processing",
               "hitl_point": None, "error_code": None, "created_at": datetime.now(timezone.utc)}
        self.rows[rx_id] = row
        img_id = uuid.uuid4()
        self.images[img_id] = {"id": img_id, "rx": rx_id, "kind": kind, "storage_key": storage_key}
        return {**row, "image_id": img_id}

    def get(self, rx_id, pharmacy_id):
        r = self.rows.get(rx_id)
        return dict(r) if r and r["pharmacy_id"] == pharmacy_id else None

    def first_image(self, rx_id, kind="prescription"):
        return next((i for i in self.images.values() if i["rx"] == rx_id and i["kind"] == kind), None)

    def image(self, rx_id, image_id, pharmacy_id):
        i = self.images.get(image_id)
        if i and i["rx"] == rx_id and self.rows[rx_id]["pharmacy_id"] == pharmacy_id:
            return i
        return None

    def set_waiting(self, rx_id, hitl_point):
        r = self.rows[rx_id]
        if r["status"] != "processing":
            return False
        r.update(status="waiting_approval", hitl_point=hitl_point)
        return True

    def set_waiting_tray(self, rx_id):
        r = self.rows[rx_id]
        if r["status"] != "processing":
            return False
        r.update(status="waiting_tray", hitl_point=None)
        return True

    def set_failed(self, rx_id, error_code):
        r = self.rows[rx_id]
        if r["status"] != "processing":
            return False
        r.update(status="failed", hitl_point=None, error_code=error_code)
        return True

    def respond(self, rx_id, pharmacy_id, user_id, hitl_point, action, before, after, reason=None):
        r = self.rows.get(rx_id)
        if not r or r["pharmacy_id"] != pharmacy_id or r["status"] != "waiting_approval" or r["hitl_point"] != hitl_point:
            return False
        r.update(status="processing", hitl_point=None)
        self.approvals.append({"rx": rx_id, "hitl_point": hitl_point, "action": action, "user_id": user_id,
                               "before": before, "after": after})
        return True


@pytest.fixture
def fake_rx(monkeypatch):
    f = FakeRx()
    for name in ("create", "get", "first_image", "image", "set_waiting", "set_waiting_tray", "set_failed", "respond"):
        monkeypatch.setattr(rx, name, getattr(f, name))
    return f


def make_client(monkeypatch, tmp_path, scenario="warning"):
    f = FakeDB()
    monkeypatch.setattr(db, "query_one", f.query_one)
    monkeypatch.setattr(db, "query_all", f.query_all)
    monkeypatch.setattr(db, "execute", f.execute)
    app = create_app({"SECRET_KEY": "test", "DATABASE_URL": "postgresql://unused", "VAR_DIR": tmp_path,
                      "TESTING": True, "JOBS_SYNC": True, "FAKE_AGENT_SCENARIO": scenario})
    client = app.test_client()
    login(client)
    client.token = csrf(client, "/")
    return client


@pytest.fixture
def client(monkeypatch, tmp_path, fake_rx):
    return make_client(monkeypatch, tmp_path)


def upload(client, data=JPG, name="rx.jpg"):
    return client.post("/api/prescriptions", data={"image": (io.BytesIO(data), name)},
                       headers={"X-CSRF-Token": client.token}, content_type="multipart/form-data")


def approve(client, rx_id, **body):
    return client.post(f"/api/prescriptions/{rx_id}/approve", json=body, headers={"X-CSRF-Token": client.token})


# ------------------------------------------------------------ 업로드
def test_upload_then_waiting_approval(client, fake_rx, tmp_path):
    r = upload(client)
    assert r.status_code == 202
    body = r.get_json()
    assert body["display_no"] == "P0001" and body["status"] == "processing"

    d = client.get(f"/api/prescriptions/{body['id']}").get_json()
    assert d["status"] == "waiting_approval" and d["hitl_point"] == "rx_approval"
    order = d["card"]["order"]
    assert [it["name"] for it in order["items"]] == ["타이레놀정500mg", "게보린정", "프론크논캡슐"]
    assert order["risk_check"] == {"scope": "rx_only", "level": "warning", "findings": order["risk_check"]["findings"]}
    assert d["card"]["ocr_lines"]["58"]["text"].startswith("(1)900000011")   # 원문 줄이 함께 온다
    # F-WEB-06: 줄마다 원본 위치(bbox)와 원본 크기가 와서 항목을 누르면 위치를 표시할 수 있다
    assert d["card"]["ocr_lines"]["58"]["bbox"] == [229, 1246, 799, 1309]
    assert d["card"]["ocr_size"] == {"width": 2480, "height": 3507}
    assert d["card"]["blockers"] == [] and d["card"]["needs_confirm"] is True   # 프론크논 확인 필요 표시
    # 원본 이미지는 같은 약국 사용자만 받는다
    img = client.get(d["image_url"])
    assert img.status_code == 200 and img.data == JPG
    assert len(list((tmp_path / "uploads").rglob("*.jpg"))) == 1


def test_numbers_increase(client):
    assert upload(client).get_json()["display_no"] == "P0001"
    assert upload(client, PNG, "b.png").get_json()["display_no"] == "P0002"


@pytest.mark.parametrize("data,name", [(b"GIF89a....", "a.gif"), (b"not an image", "a.jpg"), (JPG, "a.exe")])
def test_upload_rejects_non_images(client, fake_rx, data, name):
    r = upload(client, data, name)
    assert r.status_code == 400 and "JPG 또는 PNG" in r.get_json()["error"]
    assert fake_rx.rows == {}


def test_upload_needs_file(client):
    r = client.post("/api/prescriptions", data={}, headers={"X-CSRF-Token": client.token})
    assert r.status_code == 400 and "선택" in r.get_json()["error"]


def test_upload_needs_csrf(client):
    r = client.post("/api/prescriptions", data={"image": (io.BytesIO(JPG), "a.jpg")})
    assert r.status_code == 400


def test_api_needs_login(monkeypatch, tmp_path, fake_rx):
    app = create_app({"SECRET_KEY": "test", "DATABASE_URL": "x", "VAR_DIR": tmp_path, "TESTING": True})
    c = app.test_client()
    r = c.get(f"/api/prescriptions/{uuid.uuid4()}")
    assert r.status_code == 401 and "로그인" in r.get_json()["error"]


def test_db_failure_removes_saved_file(client, fake_rx, tmp_path):
    fake_rx.fail_create = True
    with pytest.raises(RuntimeError):
        upload(client)
    assert not list((tmp_path / "uploads").rglob("*.jpg"))


def test_ocr_failure_marks_failed(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="fail")
    rx_id = upload(c).get_json()["id"]
    d = c.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "failed" and d["error_code"] == "E_OCR_EMPTY"
    assert "다시 찍어" in d["error_message"]


def test_unknown_or_other_pharmacy_is_404(client, fake_rx):
    assert client.get(f"/api/prescriptions/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/prescriptions/not-a-uuid").status_code == 404
    rx_id = upload(client).get_json()["id"]
    fake_rx.rows[uuid.UUID(rx_id)]["pharmacy_id"] = 99
    r = client.get(f"/api/prescriptions/{rx_id}")
    assert r.status_code == 404 and r.get_json()["error"] == "처방을 찾을 수 없습니다."


# ------------------------------------------------------------ 승인 (HITL ①)
def test_approve_records_history_and_moves_on(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    r = approve(client, rx_id)                       # 확인 필요 항목이 있어 체크 없이는 안 된다
    assert r.status_code == 409 and "체크" in r.get_json()["error"]
    assert fake_rx.approvals == []

    r = approve(client, rx_id, confirmed=True)
    assert r.status_code == 202
    a = fake_rx.approvals[0]
    assert a["action"] == "approve" and a["hitl_point"] == "rx_approval" and a["before"] == a["after"]

    d = client.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "waiting_tray" and d["hitl_point"] is None and "stage" not in d
    assert d["card"]["order"]["items"][0]["item_code"] == "202106092"


def test_tray_wait_survives_restart(monkeypatch, tmp_path, fake_rx):
    """승인 뒤 트레이 대기 카드도 서버를 다시 켜면 그대로 보인다 (REQ-WEB-03)."""
    c1 = make_client(monkeypatch, tmp_path)
    rx_id = upload(c1).get_json()["id"]
    assert approve(c1, rx_id, confirmed=True).status_code == 202
    c2 = make_client(monkeypatch, tmp_path)
    d = c2.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "waiting_tray" and d["card"]["order"]["items"]
    # 트레이 대기에서는 다시 승인할 수 없다
    assert approve(c2, rx_id, confirmed=True).status_code == 409


def test_status_labels_cover_every_db_status():
    """schema.sql의 처방 상태 값과 화면 라벨(STATUS_LABELS)이 어긋나지 않는다."""
    import re
    from pathlib import Path
    from web import STATUS_LABELS
    sql = (Path(__file__).resolve().parents[2] / "web" / "schema.sql").read_text(encoding="utf-8")
    m = re.search(r"prescriptions_status_check\s+CHECK \(status IN \(([^)]*)\)\)", sql)
    assert m, "schema.sql에서 prescriptions_status_check를 찾지 못함"
    assert set(re.findall(r"'(\w+)'", m.group(1))) == set(STATUS_LABELS)


def test_double_approve_is_rejected(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    assert approve(client, rx_id, confirmed=True).status_code == 202
    r = approve(client, rx_id, confirmed=True)
    assert r.status_code == 409 and len(fake_rx.approvals) == 1


def test_none_scenario_still_has_flagged_item(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="none")
    rx_id = upload(c).get_json()["id"]
    # none 시나리오에도 프론크논 확인 필요 표시가 있어 체크가 필요하다
    assert c.get(f"/api/prescriptions/{rx_id}").get_json()["card"]["needs_confirm"] is True


def test_unmatched_item_blocks_approval(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="flagged")
    rx_id = upload(c).get_json()["id"]
    d = c.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["card"]["blockers"] and "알모란정" in d["card"]["blockers"][0]
    r = approve(c, rx_id, confirmed=True)
    assert r.status_code == 409 and "알모란정" in r.get_json()["error"]
    assert fake_rx.approvals == []


def test_danger_level(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="danger")
    rx_id = upload(c).get_json()["id"]
    d = c.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["card"]["order"]["risk_check"]["level"] == "danger"
    assert approve(c, rx_id).status_code == 409
    assert approve(c, rx_id, confirmed=True).status_code == 202


def test_waiting_state_survives_restart(monkeypatch, tmp_path, fake_rx):
    """새 앱(서버 재시작)에서도 승인 대기 카드가 그대로 보인다 (REQ-WEB-03)."""
    c1 = make_client(monkeypatch, tmp_path)
    rx_id = upload(c1).get_json()["id"]
    c2 = make_client(monkeypatch, tmp_path)
    d = c2.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "waiting_approval" and d["card"]["order"]["items"]


def test_main_page_has_upload_form(client):
    html = client.get("/").get_data(as_text=True)
    assert 'id="upload-form"' in html and "app.js" in html and 'data-id="x"' in html


def test_error_codes_match_schema():
    """schema.sql의 실패 코드 목록과 화면 안내(workflow.ERROR_CODES)가 어긋나지 않는다."""
    import re
    from pathlib import Path
    from web import workflow
    sql = (Path(__file__).resolve().parents[2] / "web" / "schema.sql").read_text(encoding="utf-8")
    m = re.search(r"prescriptions_error_code_check\s+CHECK \(error_code IN \(([^)]*)\)\)", sql)
    assert m, "schema.sql에서 prescriptions_error_code_check를 찾지 못함"
    assert set(re.findall(r"'(\w+)'", m.group(1))) == workflow.ERROR_CODES
    assert "E_INTERRUPTED" in workflow.ERROR_MESSAGES


def test_fail_stale_command(monkeypatch, tmp_path):
    """서버 시작 때 '처리 중'에 남은 처방을 정리하는 CLI (Dockerfile CMD에서 부른다)."""
    from web import workflow
    calls = []
    monkeypatch.setattr(rx, "fail_stale", lambda minutes: calls.append(minutes) or ["P0003", "P0004"])
    app = create_app({"SECRET_KEY": "test", "DATABASE_URL": "x", "VAR_DIR": tmp_path, "TESTING": True})
    out = app.test_cli_runner().invoke(args=["fail-stale"])
    assert out.exit_code == 0 and "2건" in out.output and "P0003, P0004" in out.output
    assert calls == [workflow.STALE_MINUTES]
    out = app.test_cli_runner().invoke(args=["fail-stale", "--minutes", "0"])
    assert calls[-1] == 0


def test_interrupted_shows_message(client, fake_rx):
    rx_id = upload(client).get_json()["id"]
    fake_rx.rows[uuid.UUID(rx_id)].update(status="failed", hitl_point=None, error_code="E_INTERRUPTED")
    d = client.get(f"/api/prescriptions/{rx_id}").get_json()
    assert d["status"] == "failed" and "서버가 다시 시작" in d["error_message"]
