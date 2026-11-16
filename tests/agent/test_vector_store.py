"""pgvector 저장소 (agent/vector_store.py)와 RAG 검색 연결 (11/16 결정: 벡터 DB는 pgvector).

평소에는 SQL을 적어 두는 가짜 연결로 시험한다. 진짜 DB 시험은 시험용 DB 주소를 줄 때만 돈다 (운영 DB는 쓰지 않는다):
    $env:AGENT_TEST_PGVECTOR_URL = "postgresql://user:pw@localhost:5433/agent_test"; python -m pytest tests/agent/test_vector_store.py -v
    (pgvector가 깔린 DB: docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg16)
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from agent import config
from agent.errors import ConfigError, VectorStoreError
from agent.rag import RagIndex, chunk_docs, load_index
from agent.vector_store import TABLE, PgVectorStore, default_store, vector_literal

TYLENOL, GEVORIN = "202106092", "197900277"


def doc(code, name, section, text):
    return {"doc_id": f"{code}:t:{section}", "item_code": code, "item_name": name, "source": "시험용", "section": section,
            "url": "", "text": text, "updated": None}


DOCS = [doc(TYLENOL, "타이레놀정500mg", "상호작용", "(시험용) 다른 해열진통제와 함께 먹지 마십시오."),
        doc(TYLENOL, "타이레놀정500mg", "보관법", "(시험용) 실온에 보관하십시오."),
        doc(GEVORIN, "게보린정", "주의사항", "(시험용) 아세트아미노펜이 든 약과 함께 먹지 마십시오.")]
CHUNKS = chunk_docs(DOCS)


class FakeCursor:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.db.sql.append((" ".join(sql.split()), params))
        if self.db.fail_on and self.db.fail_on in sql:
            raise self.db.error

    def executemany(self, sql, rows):
        self.db.sql.append((" ".join(sql.split()), list(rows)))

    def fetchall(self):
        return self.db.rows

    def fetchone(self):
        return self.db.rows[0]


class FakeDb:
    """psycopg 연결 흉내. 보낸 SQL과 커밋 · 되돌리기를 적어 둔다."""

    def __init__(self, rows=(), fail_on=None, error=None):
        self.sql, self.rows, self.fail_on, self.error = [], list(rows), fail_on, error
        self.commits = self.rollbacks = self.closed = 0
        self.urls = []

    def connect(self, url):
        self.urls.append(url)
        return self

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed += 1


def store(db):
    return PgVectorStore("postgresql://u:secret@db/x", connect=db.connect)


def test_vectors_are_sent_as_text_literals():
    assert vector_literal([0.1, 2, -0.000001234]) == "[0.1,2,-1.234e-06]"


def test_setup_creates_the_extension_the_table_and_the_indexes():
    db = FakeDb()
    store(db).setup(1536)
    sql = [s for s, _ in db.sql]
    assert sql[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert f"CREATE TABLE IF NOT EXISTS {TABLE} (chunk_id text PRIMARY KEY" in sql[1] and "embedding vector(1536) NOT NULL" in sql[1]
    assert "USING hnsw (embedding vector_cosine_ops)" in sql[3]
    assert (db.commits, db.closed) == (1, 1)


def test_without_permission_for_the_extension_the_fix_is_named():
    db = FakeDb(fail_on="CREATE EXTENSION", error=RuntimeError("permission denied to create extension \"vector\""))
    with pytest.raises(VectorStoreError, match="CREATE EXTENSION IF NOT EXISTS vector"):
        store(db).setup(8)
    assert (db.rollbacks, db.closed) == (1, 1)


def test_replace_all_deletes_the_model_then_inserts_in_one_transaction():
    db = FakeDb()
    count = store(db).replace_all("m1", [("c1", TYLENOL, [1.0, 0.0]), ("c2", GEVORIN, [0.0, 1.0])])
    assert count == 2 and db.commits == 1
    (delete, params), (insert, rows) = db.sql
    assert delete == f"DELETE FROM {TABLE} WHERE model = %s" and params == ("m1",)
    assert insert.startswith(f"INSERT INTO {TABLE}") and "%s::vector" in insert and "ON CONFLICT (chunk_id)" in insert
    assert rows == [("c1", TYLENOL, "m1", "[1,0]"), ("c2", GEVORIN, "m1", "[0,1]")]


def test_search_filters_by_model_and_drugs_and_orders_by_cosine_distance():
    db = FakeDb(rows=[("c2", 0.91), ("c1", 0.5)])
    got = store(db).search([1.0, 0.0], [TYLENOL], 5, "m1")
    sql, params = db.sql[0]
    assert "1 - (embedding <=> %s::vector) AS similarity" in sql and "WHERE model = %s AND item_code = ANY(%s)" in sql
    assert "ORDER BY embedding <=> %s::vector LIMIT %s" in sql and params == ["[1,0]", "m1", [TYLENOL], "[1,0]", 5]
    assert got == [("c2", 0.91), ("c1", 0.5)]
    store(db).search([1.0], [], 3, "m1")
    assert "ANY" not in db.sql[-1][0]                                          # 약을 정하지 않으면 전부에서


def test_db_errors_do_not_show_the_address_or_password():
    def refuse(url):
        raise OSError(f"could not connect to {url}")

    with pytest.raises(VectorStoreError) as e:
        PgVectorStore("postgresql://u:secret@db/x", connect=refuse).count("m1")
    assert "secret" not in str(e.value) and "OSError" in str(e.value)


def test_a_strange_table_name_is_refused():
    with pytest.raises(ValueError):
        PgVectorStore("x", connect=FakeDb().connect, table="t; DROP TABLE x")


def test_the_store_is_used_only_when_configured(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@db/x")
    monkeypatch.setattr(config, "RAG_VECTOR_STORE", "file")
    assert default_store() is None
    monkeypatch.setattr(config, "RAG_VECTOR_STORE", "pgvector")
    assert isinstance(default_store(), PgVectorStore)
    monkeypatch.delenv("DATABASE_URL")
    assert default_store() is None


def test_a_missing_driver_is_a_config_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name, *a, **kw):
        if name == "psycopg":
            raise ModuleNotFoundError(name="psycopg")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    with pytest.raises(ConfigError, match="psycopg"):
        PgVectorStore("postgresql://x").count("m1")


# ------------------------------------------------------------ RagIndex와 연결


def test_the_index_ranks_with_the_db_search_and_keeps_bm25():
    seen = []

    def db_search(vector, codes, k):
        seen.append((vector, codes, k))
        return [(CHUNKS[1].chunk_id, 0.9), ("없는조각", 0.8), (CHUNKS[0].chunk_id, 0.1)]

    index = RagIndex(CHUNKS, None, db_search)
    assert index.has_vectors
    found = index.search("보관", focus=[TYLENOL], k=2, query_vector=[1.0, 0.0])
    assert [p.chunk.section for p in found] == ["보관법", "상호작용"]
    assert seen == [([1.0, 0.0], [TYLENOL], 50)]                               # 물은 약만, 후보 50개


def test_a_failing_db_falls_back_to_the_file_vectors_then_bm25():
    def down(vector, codes, k):
        raise VectorStoreError("pgvector DB에 연결하지 못했습니다 (OperationalError).")

    vectors = {CHUNKS[0].chunk_id: [1.0, 0.0], CHUNKS[1].chunk_id: [0.0, 1.0]}
    index = RagIndex(CHUNKS, vectors, down)
    found = index.search("약", focus=[TYLENOL], k=1, query_vector=[0.0, 1.0])
    assert found[0].chunk.section == "보관법" and index.vector_errors == 1     # 파일 임베딩으로 찾았다
    bm25_only = RagIndex(CHUNKS, None, down)
    assert bm25_only.search("보관", focus=[TYLENOL], k=1, query_vector=[1.0])[0].chunk.section == "보관법"


def test_load_index_passes_the_db_search(tmp_path):
    (tmp_path / "docs.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in DOCS), encoding="utf-8")
    search = lambda v, c, k: []                                                # noqa: E731
    assert load_index(tmp_path, search).vector_search is search


# ------------------------------------------------------------ build_rag_index.py --pgvector


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "build_rag_index.py"
_spec = importlib.util.spec_from_file_location("build_rag_index_pg", SCRIPT)
build_rag_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_rag_index)


def write_index(folder, model="m1"):
    (folder / "docs.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in DOCS), encoding="utf-8")
    vectors = {c.chunk_id: [float(i), 1.0] for i, c in enumerate(CHUNKS)}
    (folder / "embeddings.json").write_text(json.dumps({"model": model, "vectors": vectors}), encoding="utf-8")


def test_push_only_sends_the_existing_embeddings_without_calling_the_api(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@db/x")
    monkeypatch.setattr(config, "EMBED_MODEL", "m1")
    write_index(tmp_path)
    db, lines = FakeDb(), []
    code = build_rag_index.main(["--pgvector", "--out", str(tmp_path)], out=lines.append,
                                fetch=lambda *a: pytest.fail("API를 부르면 안 됩니다"),
                                store_factory=lambda url: PgVectorStore(url, connect=db.connect))
    assert code == 0 and f"{TABLE} 테이블에 조각 {len(CHUNKS)}개" in lines[-1] and "2차원" in lines[-1]
    inserted = next(rows for sql, rows in db.sql if sql.startswith("INSERT"))
    assert [r[0] for r in inserted] == [c.chunk_id for c in CHUNKS] and inserted[0][2] == "m1"


@pytest.mark.parametrize("setup, words", [
    ("no_url", "DATABASE_URL이 없어"),
    ("no_files", "먼저 --embed로 만드세요"),
    ("db_error", "pgvector에 넣지 못했습니다: pgvector 확장"),
])
def test_push_problems_are_explained(monkeypatch, tmp_path, setup, words):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@db/x")
    if setup == "no_url":
        monkeypatch.delenv("DATABASE_URL")
    if setup != "no_files":
        write_index(tmp_path)
    db = FakeDb(fail_on="CREATE EXTENSION", error=RuntimeError("extension \"vector\" is not available"))
    lines = []
    code = build_rag_index.main(["--pgvector", "--out", str(tmp_path)], out=lines.append,
                                store_factory=lambda url: PgVectorStore(url, connect=db.connect))
    assert code == 1 and words in "\n".join(lines)


# ------------------------------------------------------------ 진짜 DB (시험용 DB 주소를 줄 때만)


TEST_URL = os.getenv("AGENT_TEST_PGVECTOR_URL", "").strip()


@pytest.mark.skipif(not TEST_URL, reason="AGENT_TEST_PGVECTOR_URL(시험용 pgvector DB)이 없어 건너뜀")
def test_a_real_pgvector_round_trip():
    assert TEST_URL != (config.database_url() or ""), "운영 DB(DATABASE_URL)로는 시험하지 않습니다. 시험용 DB를 따로 주세요"
    db = PgVectorStore(TEST_URL, table="agent_rag_vectors_test")
    db.setup(2)
    assert db.replace_all("test-model", [("a", TYLENOL, [1.0, 0.0]), ("b", TYLENOL, [0.0, 1.0]), ("c", GEVORIN, [1.0, 0.1])]) == 3
    assert db.count("test-model") == 3
    hits = db.search([1.0, 0.0], [TYLENOL], 2, "test-model")
    assert [h[0] for h in hits] == ["a", "b"] and hits[0][1] == pytest.approx(1.0)
    assert [h[0] for h in db.search([1.0, 0.0], [], 1, "test-model")] == ["a"]
    db.replace_all("test-model", [])
    assert db.count("test-model") == 0
