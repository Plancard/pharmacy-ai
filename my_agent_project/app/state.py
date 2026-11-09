"""LangGraph 그래프 상태 (11/6 뼈대).

State에는 JSON으로 바꿀 수 있는 값(dict·list·str·None)만 담는다.
조제지시서는 dict로 담고, 노드 안에서 DispensingOrder로 다시 읽어 검증한다.
(체크포인터가 상태를 저장·복원할 때 Pydantic 객체보다 안전하다)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# 진행 단계. 다음 주에 'approved', 'planned', 'rejected'가 늘어난다.
Status = Literal["received", "structured", "matched", "awaiting_approval", "failed"]


class GraphState(TypedDict, total=False):
    prescription_id: str                      # 처방 ID (= 나중에 체크포인터의 thread_id)
    ocr: dict[str, Any]                       # CV가 보낸 A.1 형식 OCR 결과
    order: dict[str, Any] | None              # 조제지시서 (DispensingOrder.model_dump(mode="json"))
    status: Status                            # 지금 어느 단계인지
    notice: str | None                        # 화면에 보여 줄 안내 (실패 원인, 승인 대기 안내)
    error_code: str | None                    # 실패했을 때 명세서 A.7 오류 코드 (E_OCR_EMPTY, E_LLM …)
    errors: Annotated[list[str], operator.add]   # 노드 실패 기록. 덮어쓰지 않고 쌓는다


def initial_state(prescription_id: str, ocr: dict[str, Any]) -> GraphState:
    """그래프에 넣을 첫 상태."""
    return {
        "prescription_id": prescription_id,
        "ocr": ocr,
        "order": None,
        "status": "received",
        "notice": None,
        "error_code": None,
        "errors": [],
    }
