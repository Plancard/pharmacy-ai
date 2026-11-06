"""PostgreSQL 연결 (psycopg 3 + 연결 풀).

- 테이블 정의는 web/schema.sql 하나가 기준이다. ORM은 쓰지 않는다.
- 연결은 첫 사용 때 연다. 그래서 DB가 꺼져 있어도 `flask --help` 같은 명령은 동작한다.
- 세션 시간대는 UTC로 고정한다. 화면에서만 KST로 바꾼다 (ERD 설계 규칙 6).
- LangGraph PostgresSaver도 같은 DATABASE_URL을 쓴다 (checkpointer_conninfo()).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from flask import Flask, current_app

SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def init_app(app: Flask) -> None:
    app.extensions["db_pool"] = None
    app.cli.add_command(init_db_command)


def _pool():
    pool = current_app.extensions.get("db_pool")
    if pool is None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            current_app.config["DATABASE_URL"],
            min_size=1,
            max_size=current_app.config["DB_POOL_MAX"],
            kwargs={"row_factory": dict_row, "options": "-c TimeZone=UTC"},
            open=False,
        )
        pool.open(wait=True, timeout=10)
        current_app.extensions["db_pool"] = pool
    return pool


# ------------------------------------------------------------ 짧은 질의용 함수
def query_one(sql: str, params: Any = None) -> dict | None:
    with _pool().connection() as conn:
        return conn.execute(sql, params).fetchone()


def query_all(sql: str, params: Any = None) -> list[dict]:
    with _pool().connection() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql: str, params: Any = None) -> dict | None:
    """INSERT … RETURNING 처럼 결과가 한 행 있으면 돌려준다. 블록을 벗어나면 커밋된다."""
    with _pool().connection() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchone() if cur.description else None


def ping() -> bool:
    try:
        return query_one("SELECT 1 AS ok")["ok"] == 1
    except Exception:   # 상태 확인용이라 원인은 로그로만 남긴다
        current_app.logger.exception("DB 연결 확인 실패")
        return False


def checkpointer_conninfo() -> str:
    """LangGraph PostgresSaver.from_conn_string()에 그대로 넘긴다."""
    return current_app.config["DATABASE_URL"]


# ------------------------------------------------------------ 스키마
def run_schema() -> None:
    """schema.sql 전체를 한 번에 실행한다. 여러 번 실행해도 안전하다."""
    sql = SCHEMA.read_text(encoding="utf-8")
    with _pool().connection() as conn:
        conn.execute(sql)   # 파라미터 없이 실행하면 여러 문장(함수 $$ 포함)을 한 번에 보낼 수 있다


@click.command("init-db")
def init_db_command() -> None:
    """web/schema.sql로 테이블을 만든다 (이미 있으면 건너뜀)."""
    run_schema()
    click.echo("DB 스키마 적용 완료")
