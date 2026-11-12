"""처방·이미지·승인 테이블 읽기와 쓰기. 처방 관련 SQL은 이 파일에만 둔다.

상태 바꾸기는 모두 '지금 상태가 이것일 때만' 조건을 건 UPDATE로 한다. 승인 버튼을 두 번 누르거나
처리 작업이 늦게 끝나도 상태가 거꾸로 가지 않는다 (바뀐 행이 없으면 False).
"""
from __future__ import annotations

import uuid
from typing import Any

from web import db

# 약국별 처방 번호 P0001, P0002 … 같은 약국 안에서 동시에 만들 때는 advisory lock으로 줄 세운다.
_INSERT_RX = """
INSERT INTO prescriptions (id, pharmacy_id, user_id, display_no)
SELECT %s, %s, %s,
       'P' || lpad((COALESCE(MAX(substring(display_no FROM 2)::int), 0) + 1)::text, 4, '0')
FROM prescriptions WHERE pharmacy_id = %s
RETURNING id, display_no, status, created_at
"""

_RX_COLUMNS = "id, display_no, status, hitl_point, error_code, created_at, updated_at"


def _jsonb(value: Any):
    from psycopg.types.json import Jsonb   # psycopg는 실제 DB에 쓸 때만 필요하다
    return None if value is None else Jsonb(value)


def create(pharmacy_id: int, user_id: int, rx_id: uuid.UUID, storage_key: str,
           kind: str = "prescription") -> dict:
    """처방 1건과 첫 이미지를 한 트랜잭션으로 만든다. 처방은 '처리 중'으로 시작한다.

    처방은 같은 ID의 대화(conversations)에 속한다. 대화가 없으면 새로 만들고, 상담 대화(같은 ID)가
    이미 있으면 그 대화가 처방 대화가 된다.
    """
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations (id, pharmacy_id, user_id) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (rx_id, pharmacy_id, user_id),
        )
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (pharmacy_id,))
        rx = conn.execute(_INSERT_RX, (rx_id, pharmacy_id, user_id, pharmacy_id)).fetchone()
        img = conn.execute(
            "INSERT INTO images (prescription_id, kind, kind_source, storage_key) "
            "VALUES (%s, %s, 'manual', %s) RETURNING id",
            (rx_id, kind, storage_key),
        ).fetchone()
    return {**rx, "image_id": img["id"]}


def get(rx_id: uuid.UUID, pharmacy_id: int) -> dict | None:
    """다른 약국의 처방은 없는 것으로 본다."""
    return db.query_one(
        f"SELECT {_RX_COLUMNS} FROM prescriptions WHERE id = %s AND pharmacy_id = %s",
        (rx_id, pharmacy_id),
    )


def first_image(rx_id: uuid.UUID, kind: str = "prescription") -> dict | None:
    return db.query_one(
        "SELECT id, kind, storage_key FROM images WHERE prescription_id = %s AND kind = %s "
        "ORDER BY created_at LIMIT 1",
        (rx_id, kind),
    )


def image(rx_id: uuid.UUID, image_id: uuid.UUID, pharmacy_id: int) -> dict | None:
    return db.query_one(
        "SELECT i.id, i.storage_key FROM images i JOIN prescriptions p ON p.id = i.prescription_id "
        "WHERE i.id = %s AND i.prescription_id = %s AND p.pharmacy_id = %s",
        (image_id, rx_id, pharmacy_id),
    )


def set_waiting(rx_id: uuid.UUID, hitl_point: str) -> bool:
    """처리 중 → 승인 대기."""
    row = db.execute(
        "UPDATE prescriptions SET status = 'waiting_approval', hitl_point = %s "
        "WHERE id = %s AND status = 'processing' RETURNING id",
        (hitl_point, rx_id),
    )
    return row is not None


def set_waiting_tray(rx_id: uuid.UUID) -> bool:
    """처리 중 → 트레이 대기. 조제 계획까지 끝나 약사가 조제 후 트레이 사진을 올리기를 기다린다.

    HITL(interrupt)이 아니라 그래프 실행이 끝난 상태다. 트레이 사진은 새 메시지로 들어와 새 실행이 된다.
    """
    row = db.execute(
        "UPDATE prescriptions SET status = 'waiting_tray', hitl_point = NULL "
        "WHERE id = %s AND status = 'processing' RETURNING id",
        (rx_id,),
    )
    return row is not None


def set_failed(rx_id: uuid.UUID, error_code: str | None) -> bool:
    """처리 중 → 실패. error_code는 명세서 A.7 코드 또는 None(원인 미상, 로그 확인)."""
    row = db.execute(
        "UPDATE prescriptions SET status = 'failed', hitl_point = NULL, error_code = %s "
        "WHERE id = %s AND status = 'processing' RETURNING id",
        (error_code, rx_id),
    )
    return row is not None


def fail_stale(min_age_minutes: int) -> list[str]:
    """처리 중인 채로 min_age_minutes분 넘게 바뀌지 않은 처방을 실패(E_INTERRUPTED)로 바꾼다. → 바꾼 display_no 목록.

    처리는 웹 프로세스 안의 백그라운드 스레드(web/jobs.py)라서 서버가 다시 시작되면 하던 일이 사라지고 처방만
    '처리 중'으로 남는다. 서버 시작 때(Dockerfile CMD) 이것을 불러 정리한다.
    """
    rows = db.query_all(
        "UPDATE prescriptions SET status = 'failed', hitl_point = NULL, error_code = 'E_INTERRUPTED' "
        "WHERE status = 'processing' AND updated_at < now() - make_interval(mins => %s) "
        "RETURNING display_no",
        (min_age_minutes,),
    )
    return [r["display_no"] for r in rows]


def respond(rx_id: uuid.UUID, pharmacy_id: int, user_id: int, hitl_point: str, action: str,
            before: Any, after: Any, reason: str | None = None) -> bool:
    """HITL 응답: 승인 대기 → 처리 중으로 되돌리고 approvals에 이력을 남긴다 (한 트랜잭션).

    이미 다른 요청이 응답했거나 다른 지점에서 기다리는 중이면 False (아무것도 바꾸지 않음).
    """
    with db.transaction() as conn:
        row = conn.execute(
            "UPDATE prescriptions SET status = 'processing', hitl_point = NULL "
            "WHERE id = %s AND pharmacy_id = %s AND status = 'waiting_approval' AND hitl_point = %s "
            "RETURNING id",
            (rx_id, pharmacy_id, hitl_point),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "INSERT INTO approvals (prescription_id, hitl_point, action, user_id, before_json, after_json, reason) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (rx_id, hitl_point, action, user_id, _jsonb(before), _jsonb(after), reason),
        )
    return True
