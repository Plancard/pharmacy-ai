"""상위 그래프의 상태 (7단계, 일정표 11/9, REQ-AG-01).

State에는 JSON으로 바꿀 수 있는 값(dict · list · str · None)만 담는다.
조제지시서는 dict로 담고, 노드 안에서 DispensingOrder로 다시 읽어 검증한다.
(10단계에서 체크포인터가 상태를 저장·복원할 때 Pydantic 객체보다 안전하다)

메시지 하나가 들어오면 슈퍼바이저가 사진 종류를 보고 흐름(flow)을 한 번 정한다.

    photo_kind(들어온 사진 종류) + ocr.doc_type(OCR이 읽은 문서 종류)  →  flow(슈퍼바이저가 정한 흐름)
    처방전(prescription)  →  prescription  처방 구조화 → 약품 매칭 → 승인 대기
    약 포장(med_bag)      →  recheck       복용 약 재대조 (자리만 있다)
    트레이(tray)          →  tray          트레이 검수 (자리만 있다)
    그 밖 · 애매함         →  ask_type      사진 종류를 되묻는다 (HITL type_select 자리)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict, get_args

# 들어온 사진의 종류. 처방전 · 약 포장은 A.1의 doc_type 값과 같고, 트레이는 OCR을 거치지 않는다.
PhotoKind = Literal["prescription", "med_bag", "tray"]
# 슈퍼바이저가 정한 흐름. 흐름마다 상위 그래프의 갈래가 하나씩 있다.
Flow = Literal["prescription", "recheck", "tray", "ask_type"]
# 진행 단계. 승인 이후('approved', 'planned', 'rejected')는 8단계부터 늘어난다.
# not_ready는 아직 만들지 않은 흐름(복용 약 재대조 · 트레이 검수)에 들어왔다는 뜻이다. 그 흐름이 만들어지면 사라진다.
Status = Literal[
    "received", "structured", "matched", "awaiting_approval", "awaiting_type", "not_ready", "failed",
]

PHOTO_KINDS: tuple[str, ...] = get_args(PhotoKind)
FLOWS: tuple[str, ...] = get_args(Flow)
STATUSES: tuple[str, ...] = get_args(Status)


class GraphState(TypedDict, total=False):
    prescription_id: str                      # 처방 ID (= 10단계부터 체크포인터의 thread_id)
    photo_kind: PhotoKind | None              # 들어온 사진 종류. 모르면 None (OCR의 doc_type으로 가린다)
    ocr: dict[str, Any] | None                # CV가 보낸 A.1 형식 OCR 결과. 트레이 사진에는 없다
    flow: Flow | None                         # 슈퍼바이저가 정한 흐름. 슈퍼바이저를 지나기 전에는 None
    order: dict[str, Any] | None              # 조제지시서 (DispensingOrder.model_dump(mode="json"))
    status: Status                            # 지금 어느 단계인지
    notice: str | None                        # 화면에 보여 줄 안내 (실패 원인, 승인 대기 안내, 사진 종류 질문)
    error_code: str | None                    # 실패했을 때 명세서 A.7 오류 코드 (E_OCR_EMPTY, E_LLM …)
    errors: Annotated[list[str], operator.add]   # 노드 실패 기록. 덮어쓰지 않고 쌓는다


def initial_state(
    prescription_id: str,
    ocr: dict[str, Any] | None = None,
    photo_kind: str | None = None,
) -> GraphState:
    """그래프에 넣을 첫 상태."""
    return {
        "prescription_id": prescription_id,
        "photo_kind": photo_kind,
        "ocr": ocr,
        "flow": None,
        "order": None,
        "status": "received",
        "notice": None,
        "error_code": None,
        "errors": [],
    }
