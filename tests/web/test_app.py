"""웹 뼈대 테스트. DB 없이 돌도록 web.db의 질의 함수를 가짜로 바꾼다.

실제 PostgreSQL 연결은 `docker compose up` 후 /api/health 로 확인한다.
실행: python -m pytest tests/web -q
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from werkzeug.security import generate_password_hash

import web.db as db
from web import create_app


class FakeDB:
    def __init__(self):
        self.pharmacies = {1: "테스트약국"}
        self.users = {1: {"id": 1, "username": "ph1", "password_hash": generate_password_hash("password123"),
                          "role": "pharmacist", "pharmacy_id": 1}}
        self.up = True

    def query_one(self, sql, params=None):
        if "FROM users u JOIN pharmacies" in sql:
            u = self.users.get(params[0])
            return {**u, "pharmacy_name": self.pharmacies[u["pharmacy_id"]]} if u else None
        if "SELECT id, password_hash FROM users" in sql:
            return next(({"id": u["id"], "password_hash": u["password_hash"]}
                         for u in self.users.values() if u["username"] == params[0]), None)
        if "SELECT 1 FROM users" in sql:
            return {"?column?": 1} if any(u["username"] == params[0] for u in self.users.values()) else None
        if "FROM pharmacies WHERE name" in sql:
            return next(({"id": k} for k, v in self.pharmacies.items() if v == params[0]), None)
        if "SELECT 1 AS ok" in sql:
            if not self.up:
                raise ConnectionError("db down")
            return {"ok": 1}
        raise AssertionError(sql)

    def query_all(self, sql, params=None):
        if "FROM prescriptions" in sql and "dispense_records" not in sql:
            return [{"id": "x", "display_no": "P0001", "status": "waiting_approval", "hitl_point": "rx_approval",
                     "created_at": datetime(2026, 11, 6, 3, 0, tzinfo=timezone.utc)}]
        if "FROM dispense_records" in sql:
            return []
        raise AssertionError(sql)

    def execute(self, sql, params=None):
        if sql.startswith("INSERT INTO pharmacies"):
            new = max(self.pharmacies) + 1
            self.pharmacies[new] = params[0]
            return {"id": new}
        if sql.startswith("INSERT INTO users"):
            new = max(self.users) + 1
            self.users[new] = {"id": new, "pharmacy_id": params[0], "username": params[1],
                               "password_hash": params[2], "role": params[3]}
            return None
        raise AssertionError(sql)


@pytest.fixture
def fake(monkeypatch):
    f = FakeDB()
    monkeypatch.setattr(db, "query_one", f.query_one)
    monkeypatch.setattr(db, "query_all", f.query_all)
    monkeypatch.setattr(db, "execute", f.execute)
    return f


@pytest.fixture
def app(fake, tmp_path):
    return create_app({"SECRET_KEY": "test", "DATABASE_URL": "postgresql://unused", "VAR_DIR": tmp_path, "TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


def csrf(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def login(client, username="ph1", password="password123", next_=""):
    return client.post("/login", data={"csrf_token": csrf(client), "username": username,
                                       "password": password, "next": next_})


def test_missing_config_is_reported(monkeypatch):
    from web.config import ConfigError
    with pytest.raises(ConfigError, match="FLASK_SECRET_KEY"):
        create_app({"SECRET_KEY": "", "DATABASE_URL": "x"})


def test_pages_need_login(client):
    for path in ("/", "/history"):
        r = client.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_login_success_and_main(client):
    r = login(client)
    assert r.status_code == 302 and r.headers["Location"] == "/"
    html = client.get("/").get_data(as_text=True)
    assert "테스트약국 · ph1" in html and "P0001" in html and "승인 대기" in html
    assert "2026-11-06 12:00" in html          # 03:00 UTC → 12:00 KST


def test_wrong_password(client):
    r = login(client, password="nope")
    assert r.status_code == 200 and "올바르지 않습니다" in r.get_data(as_text=True)
    assert client.get("/").status_code == 302


def test_unknown_user_same_message(client):
    r = login(client, username="ghost")
    assert "올바르지 않습니다" in r.get_data(as_text=True)


def test_next_redirect_only_same_site(client):
    assert login(client, next_="/history").headers["Location"] == "/history"
    client.post("/logout", data={"csrf_token": csrf(client, "/")})
    assert login(client, next_="https://evil.example/").headers["Location"] == "/"
    client.post("/logout", data={"csrf_token": csrf(client, "/")})
    assert login(client, next_="//evil.example").headers["Location"] == "/"


def test_csrf_required(client):
    client.get("/login")
    r = client.post("/login", data={"username": "ph1", "password": "password123"})
    assert r.status_code == 400


def test_logout_requires_post_and_clears(client):
    login(client)
    assert client.get("/logout").status_code == 405
    client.post("/logout", data={"csrf_token": csrf(client, "/")})
    assert client.get("/").status_code == 302


def test_deleted_user_session_cleared(client, fake):
    login(client)
    fake.users.clear()
    assert client.get("/").status_code == 302


def test_history_empty(client):
    login(client)
    assert "저장된 조제 기록이 없습니다" in client.get("/history").get_data(as_text=True)


def test_health(client, fake, monkeypatch):
    monkeypatch.setattr(db, "ping", lambda: fake.query_one("SELECT 1 AS ok")["ok"] == 1)
    assert client.get("/api/health").get_json() == {"status": "ok", "db": True}


def test_health_db_down(client, fake, app):
    fake.up = False
    r = client.get("/api/health")      # 진짜 ping()이 예외를 잡아 503
    assert r.status_code == 503 and r.get_json()["db"] is False


def test_create_user_cli(app, fake):
    runner = app.test_cli_runner()
    r = runner.invoke(args=["create-user", "ph2", "--pharmacy", "새약국"], input="longpassword\nlongpassword\n")
    assert r.exit_code == 0, r.output
    assert "새약국" in fake.pharmacies.values() and any(u["username"] == "ph2" for u in fake.users.values())
    r = runner.invoke(args=["create-user", "ph2", "--pharmacy", "새약국"], input="longpassword\nlongpassword\n")
    assert r.exit_code != 0 and "이미 있는 아이디" in r.output
    r = runner.invoke(args=["create-user", "ph3", "--pharmacy", "새약국"], input="short\nshort\n")
    assert r.exit_code != 0 and "8자 이상" in r.output


def test_storage_roundtrip_and_traversal(app):
    from web import storage
    with app.app_context():
        key = storage.save("8a2f0d8e-6a8b-4d0e-9a3c-1f2b3c4d5e6f", b"img", ".JPG")
        assert key.startswith("uploads/8a2f0d8e-6a8b-4d0e-9a3c-1f2b3c4d5e6f/") and key.endswith(".jpg")
        assert storage.read(key) == b"img"
        with pytest.raises(ValueError):
            storage.path_of("../../etc/passwd")
        with pytest.raises(ValueError):
            storage.save("not-a-uuid", b"x", ".jpg")
        with pytest.raises(ValueError):
            storage.save("8a2f0d8e-6a8b-4d0e-9a3c-1f2b3c4d5e6f", b"x", ".exe")


def test_404_page(client):
    r = client.get("/nope")
    assert r.status_code == 404 and "404" in r.get_data(as_text=True)
