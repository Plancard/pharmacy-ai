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
    """처방 1건과 첫 이미지를 한 트랜잭션으로 만든다. 처방은 '처리 중'으로 시작한다."""
    with db.transaction() as conn:
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
