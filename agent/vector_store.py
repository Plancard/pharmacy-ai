"""RAG 임베딩 저장소: PostgreSQL + pgvector (가장자리, 11/16 결정 '벡터 DB는 pgvector').

    store = PgVectorStore(url)                     # 웹과 같은 DB (DATABASE_URL)
    store.setup(dim)                               # 확장 · 테이블 · 색인 (여러 번 해도 안전하다)
    store.replace_all(model, rows)                 # 색인을 다시 만들 때 통째로 바꾼다. rows: [(chunk_id, item_code, vector)]
    store.search(vector, item_codes, k, model)     # [(chunk_id, 코사인 유사도)] 가까운 순서
    store.count(model)

rag.RagIndex는 이 저장소를 search_fn(vector_search)으로 받는다. 글(docs.jsonl)과 BM25는 그대로 파일에 있고,
임베딩 검색만 DB에서 한다. DB가 안 되면 VectorStoreError를 올리고, RagIndex는 파일 임베딩(embeddings.json) · BM25로 계속한다.

DB 준비 (웹 · DB 담당이 한 번)
- pgvector가 깔린 PostgreSQL이어야 한다 (docker 이미지 pgvector/pgvector:pg16 등).
- 확장은 DB 관리자 권한이 필요하다: CREATE EXTENSION IF NOT EXISTS vector;  (setup이 해 보고, 권한이 없으면 이 문장을 알려 준다)
- 테이블 이름은 agent_rag_vectors (웹 테이블과 겹치지 않게 agent_ 로 시작한다). 공개 허가정보의 조각 번호와 숫자만 있다 (개인정보 없음).

필요한 패키지: psycopg (체크포인터와 같다). 벡터는 '[0.1,0.2,…]' 글로 보내고 ::vector로 바꾼다 (pgvector 파이썬 패키지 없이).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from agent.errors import ConfigError, VectorStoreError

TABLE = "agent_rag_vectors"
NEED_EXTENSION = "DB 관리자가 한 번 실행해야 합니다: CREATE EXTENSION IF NOT EXISTS vector;"
Connect = Callable[[str], Any]


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(float(x), ".7g") for x in vector) + "]"


def _psycopg_connect(url: str) -> Any:
    try:
        import psycopg
    except ModuleNotFoundError:
        raise ConfigError("pgvector 저장소에 필요한 psycopg가 없습니다. pip install \"psycopg[binary]\" 로 설치하세요.") from None
    return psycopg.connect(url, connect_timeout=5)


class PgVectorStore:
    def __init__(self, url: str, *, connect: Connect | None = None, table: str = TABLE):
        if not table.replace("_", "").isalnum():
            raise ValueError(f"테이블 이름이 이상합니다: {table}")
        self.url, self.table = url, table
        self._connect = connect or _psycopg_connect

    def _run(self, work: Callable[[Any], Any]) -> Any:
        """연결 하나를 열어 work(커서)를 하고 커밋한다. DB 오류는 VectorStoreError (주소 · 비밀번호는 문장에 넣지 않는다)."""
        try:
            conn = self._connect(self.url)
        except ConfigError:
            raise
        except Exception as e:
            raise VectorStoreError(f"pgvector DB에 연결하지 못했습니다 ({type(e).__name__}).") from None
        try:
            with conn.cursor() as cur:
                result = work(cur)
            conn.commit()
            return result
        except VectorStoreError:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            text = str(e).lower()
            if "vector" in text and ("does not exist" in text or "permission" in text or "extension" in text):
                raise VectorStoreError(f"pgvector 확장이 없습니다. {NEED_EXTENSION}") from None
            raise VectorStoreError(f"pgvector DB 작업에 실패했습니다 ({type(e).__name__}).") from None
        finally:
            conn.close()

    def setup(self, dim: int) -> None:
        if not (isinstance(dim, int) and 0 < dim <= 16000):
            raise ValueError(f"임베딩 차원이 이상합니다: {dim}")
        t = self.table

        def work(cur):
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                raise VectorStoreError(f"pgvector 확장을 만들지 못했습니다. {NEED_EXTENSION}") from None
            cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (chunk_id text PRIMARY KEY, item_code text NOT NULL, "
                        f"model text NOT NULL, embedding vector({dim}) NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_item_idx ON {t} (model, item_code)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_hnsw_idx ON {t} USING hnsw (embedding vector_cosine_ops)")
        self._run(work)

    def replace_all(self, model: str, rows: Iterable[tuple[str, str, Sequence[float]]]) -> int:
        """이 모델의 임베딩을 통째로 바꾼다 (한 트랜잭션: 중간에 실패하면 예전 것이 남는다). 넣은 줄 수."""
        data = [(chunk_id, item_code, model, vector_literal(v)) for chunk_id, item_code, v in rows]
        t = self.table

        def work(cur):
            cur.execute(f"DELETE FROM {t} WHERE model = %s", (model,))
            if data:
                cur.executemany(f"INSERT INTO {t} (chunk_id, item_code, model, embedding) VALUES (%s, %s, %s, %s::vector) "
                                f"ON CONFLICT (chunk_id) DO UPDATE SET item_code = EXCLUDED.item_code, model = EXCLUDED.model, "
                                f"embedding = EXCLUDED.embedding, updated_at = now()", data)
            return len(data)
        return self._run(work)

    def search(self, vector: Sequence[float], item_codes: Sequence[str], k: int, model: str) -> list[tuple[str, float]]:
        """가까운 조각 k개 [(chunk_id, 코사인 유사도)]. item_codes가 있으면 그 약의 조각만."""
        literal, t = vector_literal(vector), self.table
        codes = list(item_codes)

        def work(cur):
            where = "model = %s" + (" AND item_code = ANY(%s)" if codes else "")
            params: list[Any] = [literal, model, *([codes] if codes else []), literal, int(k)]
            cur.execute(f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS similarity FROM {t} WHERE {where} "
                        f"ORDER BY embedding <=> %s::vector LIMIT %s", params)
            return [(str(row[0]), float(row[1])) for row in cur.fetchall()]
        return self._run(work)

    def count(self, model: str) -> int:
        t = self.table

        def work(cur):
            cur.execute(f"SELECT count(*) FROM {t} WHERE model = %s", (model,))
            return int(cur.fetchone()[0])
        return self._run(work)

    def searcher(self, model: str) -> Callable[[Sequence[float], Sequence[str], int], list[tuple[str, float]]]:
        """RagIndex에 넘길 검색 함수 (모델을 묶어 둔다)."""
        return lambda vector, item_codes, k: self.search(vector, item_codes, k, model)


def default_store() -> PgVectorStore | None:
    """설정을 따른 저장소: RAG_VECTOR_STORE=pgvector이고 DATABASE_URL이 있으면 저장소, 아니면 None (파일 임베딩)."""
    from agent import config

    url = config.database_url()
    if config.RAG_VECTOR_STORE != "pgvector" or not url:
        return None
    return PgVectorStore(url)
