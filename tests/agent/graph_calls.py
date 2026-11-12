"""시험용 줄임 함수: 처방 대화(대화 ID = 처방 ID)에 메시지 하나를 넣는 옛 호출 모양 (id, ocr, 사진 종류, 글).

11/12 결정으로 state가 대화 기준이 되고 메시지가 {message_id, text, image_id, kind}로 들어오게 바뀌었다.
처방 흐름을 보는 시험들은 처방 대화를 그대로 쓰므로 이 함수로 부른다. 상담 대화(처방 없음)는 시험에서 직접 만든다.
"""
from __future__ import annotations

from typing import Any

from agent.graph import run_message
from agent.state import GraphState, initial_state


def message(photo_kind: str | None = None, text: str | None = None) -> dict[str, Any]:
    return {"kind": photo_kind, "text": text}


def first_state(rx_id: str, ocr: dict[str, Any] | None = None, photo_kind: str | None = None,
                text: str | None = None) -> GraphState:
    """처방 대화의 첫 상태 (prescription_id = 대화 ID)."""
    return initial_state(rx_id, message(photo_kind, text), rx_id, ocr)


def send(graph, rx_id: str, ocr: dict[str, Any] | None = None, photo_kind: str | None = None,
         text: str | None = None) -> dict[str, Any]:
    """처방 대화에 메시지 하나를 넣어 돌린다."""
    return run_message(graph, rx_id, message(photo_kind, text), ocr, prescription_id=rx_id)
