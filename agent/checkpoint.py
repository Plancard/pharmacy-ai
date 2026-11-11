"""체크포인터 (10단계, 일정표 11/12, REQ-WEB-03 · REQ-NF-06) — 그래프가 멈춘 곳을 저장하는 곳.

승인 대기(HITL ①)에서 멈춘 그래프는 처방 ID(thread_id)마다 여기 저장된다. 그래서 프로세스를 껐다 켜도 같은 곳에서 이어 간다.

- .env에 DATABASE_URL이 있으면 PostgreSQL에 저장한다 (PostgresSaver). 웹의 서비스 테이블과 같은 DB를 쓴다 (명세서 2.5절).
  처음 연결할 때 체크포인트 테이블을 만든다 (setup, 여러 번 해도 안전하다).
- 없으면 메모리에 저장한다 (InMemorySaver). 프로세스가 끝나면 사라지므로 개발 · 시험용이다.

필요한 패키지: PostgreSQL을 쓸 때만 langgraph-checkpoint-postgres와 psycopg · psycopg-pool (웹이 이미 쓴다).
가장자리 층이다: DB에 닿는 일만 하고, 무엇을 저장할지는 그래프가 정한다.
"""
from __future__ import annotations

from typing import Any

from agent import config
from agent.errors import ConfigError

POSTGRES_PACKAGES = "pip install langgraph-checkpoint-postgres \"psycopg[binary]\" psycopg-pool"


def memory_checkpointer() -> Any:
    """메모리 체크포인터. 프로세스가 끝나면 저장한 것이 사라진다."""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def postgres_checkpointer(url: str) -> Any:
    """PostgreSQL 체크포인터. 연결 풀을 열고 체크포인트 테이블을 만든다 (이미 있으면 그대로 둔다)."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ModuleNotFoundError as e:
        raise ConfigError(f"PostgreSQL 체크포인터에 필요한 패키지({e.name})가 없습니다. {POSTGRES_PACKAGES} 로 설치하세요.") from None
    # PostgresSaver가 요구하는 연결 설정: 자동 커밋, 준비된 문장 끄기, 줄을 dict로 받기
    pool = ConnectionPool(url, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}, open=True)
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def default_checkpointer() -> Any:
    """설정에 맞는 체크포인터: DATABASE_URL이 있으면 PostgreSQL, 없으면 메모리."""
    url = config.database_url()
    return postgres_checkpointer(url) if url else memory_checkpointer()


def describe(checkpointer: Any) -> str:
    """화면에 보일 체크포인터 이름 (주소 · 비밀번호는 보이지 않는다)."""
    name = type(checkpointer).__name__
    if name == "PostgresSaver":
        return "PostgreSQL (DATABASE_URL, 프로세스를 꺼도 남는다)"
    if name in ("InMemorySaver", "MemorySaver"):
        return "메모리 (프로세스가 끝나면 사라진다)"
    return name
