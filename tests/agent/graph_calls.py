"""시험용 줄임 함수: 처방(대화 ID = 처방 ID)에 사진 한 장을 넣는 호출 모양 (id, ocr, 사진 종류).

메시지는 {image_id, kind}로 들어온다 (명세서: 글 입력 없음).
"""
from __future__ import annotations

from typing import Any

from agent.graph import run_message
from agent.state import GraphState, initial_state


def message(photo_kind: str | None = None) -> dict[str, Any]:
    return {"kind": photo_kind}


def first_state(rx_id: str, ocr: dict[str, Any] | None = None, photo_kind: str | None = None) -> GraphState:
    """처방의 첫 상태 (prescription_id = 대화 ID)."""
    return initial_state(rx_id, message(photo_kind), rx_id, ocr)


def send(graph, rx_id: str, ocr: dict[str, Any] | None = None, photo_kind: str | None = None) -> dict[str, Any]:
    """처방에 사진 한 장을 넣어 돌린다."""
    return run_message(graph, rx_id, message(photo_kind), ocr, prescription_id=rx_id)
