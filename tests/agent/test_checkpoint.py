"""체크포인터 고르기 (agent/checkpoint.py).

PostgreSQL에 실제로 저장되고 '다시 켠 뒤' 이어 가는지는 AGENT_TEST_DATABASE_URL을 줄 때만 본다 (맨 아래 시험).
    $env:AGENT_TEST_DATABASE_URL="postgresql://user:password@localhost:5432/dbname"   (PowerShell)
    python -m pytest tests/agent/test_checkpoint.py -q
"""
import os
import sys
import types
import uuid

import pytest

import fake_langgraph
from agent import checkpoint
from agent.errors import ConfigError


def test_memory_is_used_without_a_database_url(monkeypatch):
    fake_langgraph.install(monkeypatch)
    saver = checkpoint.default_checkpointer()
    assert type(saver).__name__ == "InMemorySaver"
    assert checkpoint.describe(saver) == "메모리 (프로세스가 끝나면 사라진다)"


def test_postgres_is_used_when_the_database_url_is_set(monkeypatch):
    seen = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")
    monkeypatch.setattr(checkpoint, "postgres_checkpointer", lambda url: seen.append(url) or "pg")
    assert checkpoint.default_checkpointer() == "pg" and seen == ["postgresql://u:p@db:5432/x"]


def test_a_blank_database_url_counts_as_none(monkeypatch):
    fake_langgraph.install(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "   ")
    monkeypatch.setattr(checkpoint, "postgres_checkpointer", lambda url: pytest.fail("빈 주소로 연결하면 안 됩니다"))
    assert type(checkpoint.default_checkpointer()).__name__ == "InMemorySaver"


class FakePool:
    made = []

    def __init__(self, conninfo, *, kwargs, open):
        self.conninfo, self.kwargs, self.opened = conninfo, kwargs, open
        FakePool.made.append(self)


class FakeSaver:
    def __init__(self, conn):
        self.conn = conn
        self.ready = False

    def setup(self):
        self.ready = True


def install_postgres_modules(monkeypatch):
    postgres = types.ModuleType("langgraph.checkpoint.postgres")
    postgres.PostgresSaver = FakeSaver
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = "dict_row"
    pool = types.ModuleType("psycopg_pool")
    pool.ConnectionPool = FakePool
    psycopg = types.ModuleType("psycopg")
    psycopg.rows = rows
    for name, module in (("langgraph.checkpoint.postgres", postgres), ("psycopg", psycopg), ("psycopg.rows", rows), ("psycopg_pool", pool)):
        monkeypatch.setitem(sys.modules, name, module)


def test_the_postgres_checkpointer_opens_a_pool_with_the_settings_it_needs_and_makes_its_tables(monkeypatch):
    install_postgres_modules(monkeypatch)
    FakePool.made.clear()
    saver = checkpoint.postgres_checkpointer("postgresql://u:p@db:5432/x")
    [pool] = FakePool.made
    assert pool.conninfo == "postgresql://u:p@db:5432/x" and pool.opened is True
    assert pool.kwargs == {"autocommit": True, "prepare_threshold": 0, "row_factory": "dict_row"}
    assert isinstance(saver, FakeSaver) and saver.conn is pool and saver.ready


def test_a_missing_postgres_package_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", None)
    with pytest.raises(ConfigError, match="langgraph-checkpoint-postgres"):
        checkpoint.postgres_checkpointer("postgresql://u:p@db:5432/x")


def test_the_postgres_name_never_shows_the_address():
    pg = type("PostgresSaver", (), {})()
    assert checkpoint.describe(pg) == "PostgreSQL (DATABASE_URL, 프로세스를 꺼도 남는다)"
    assert checkpoint.describe(type("Other", (), {})()) == "Other"


# ------------------------------------------------------------ 진짜 PostgreSQL (주소를 준 때만)

TEST_URL = os.getenv("AGENT_TEST_DATABASE_URL", "").strip()


@pytest.mark.skipif(not TEST_URL, reason="AGENT_TEST_DATABASE_URL이 없다 (진짜 PostgreSQL 시험은 건너뛴다)")
def test_a_real_postgres_keeps_the_stop_across_a_restart():
    pytest.importorskip("langgraph.checkpoint.postgres")
    import json

    from agent.api import AgentApi
    from agent.config import RESOURCES_DIR
    from agent.drug_dictionary import load_dictionary
    from agent.graph import build_graph
    from agent.pipeline import structure_prescription

    dictionary = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
    ocr = json.loads((RESOURCES_DIR / "samples" / "rx8_a1" / "rx8_01.json").read_text(encoding="utf-8"))
    rx_id = f"TEST-{uuid.uuid4().hex[:8]}"

    def rule_only(pid, o):
        return structure_prescription(pid, o, llm_fn=lambda *a: pytest.fail("LLM"))

    first = AgentApi(build_graph(dictionary=dictionary, structure=rule_only, api=None, checkpointer=checkpoint.postgres_checkpointer(TEST_URL)))
    pause = first.start(rx_id, ocr)
    # 다시 켠 것처럼: 연결 풀부터 새로 만든다
    second = AgentApi(build_graph(dictionary=dictionary, structure=lambda pid, o: pytest.fail("다시 구조화하면 안 됩니다"), api=None,
                                  checkpointer=checkpoint.postgres_checkpointer(TEST_URL)))
    assert second.pending(rx_id) == pause
    assert second.resume(rx_id, "rx_approval", {"action": "approve", "order": pause.payload["order"]}) is None
    assert second.stage(rx_id) == "await_tray"
