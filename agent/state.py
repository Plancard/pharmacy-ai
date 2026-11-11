"""그래프의 상태 (7 ~ 11단계, 일정표 11/9 ~ 11/13, REQ-AG-01 · 04).

State에는 JSON으로 바꿀 수 있는 값(dict · list · str · None)만 담는다.
조제지시서는 dict로 담고, 노드 안에서 DispensingOrder로 다시 읽어 검증한다.
(체크포인터가 상태를 저장·복원할 때 Pydantic 객체보다 안전하다)

메시지 하나(사진 1장 · 글 · 둘 다)가 들어오면 슈퍼바이저(supervisor.py)가 실행할 흐름과 순서(route)를 한 번 정한다.

    photo_kind(들어온 사진 종류) + ocr.doc_type(OCR이 읽은 문서 종류) + text(글의 의도)  →  route
    ask_type       사진 종류나 글의 뜻을 되묻는다 (HITL type_select에서 멈춘다)
    prescription   처방 서브그래프: 구조화 → 약품 매칭 → 위험 신호 대조 → 승인 대기(멈춤) → 조제 계획
    register_meds  복용 약 등록 (자리만 있다. 15단계)
    recheck        복용 약 재대조 (자리만 있다. 15단계)
    tray           트레이 검수 (자리만 있다. 14단계)
    consult        상담 답변 (자리만 있다. 16단계)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict, get_args

# 들어온 사진의 종류. 처방전 · 약 포장은 A.1의 doc_type 값과 같고, 트레이는 OCR을 거치지 않는다.
PhotoKind = Literal["prescription", "med_bag", "tray"]
# 흐름. 흐름마다 상위 그래프의 노드가 하나씩 있다. 실행 순서는 supervisor.FLOW_ORDER가 정한다.
Flow = Literal["prescription", "recheck", "tray", "ask_type", "register_meds", "consult"]
# 진행 단계. 멈춘 것(승인 대기 · 되묻기)은 상태 값이 아니라 그래프가 멈춘 것으로 안다 (checkpointer의 interrupts. api.py).
# approved는 약사가 승인한 상태, edited는 약사가 고쳐 매칭 · 대조를 다시 하는 상태, planned는 조제 계획을 만든 상태다.
# not_ready는 아직 만들지 않은 흐름(복용 약 · 트레이 · 상담)만 돈 것이고, noted는 글을 메모로만 남기고 끝난 것이다.
Status = Literal[
    "received", "structured", "matched", "checked", "approved", "edited", "planned", "not_ready", "noted", "failed",
]

PHOTO_KINDS: tuple[str, ...] = get_args(PhotoKind)
FLOWS: tuple[str, ...] = get_args(Flow)
STATUSES: tuple[str, ...] = get_args(Status)


class GraphState(TypedDict, total=False):
    prescription_id: str                      # 처방 ID (= 체크포인터의 thread_id)
    photo_kind: PhotoKind | None              # 들어온 사진 종류. 모르면 None (OCR의 doc_type으로 가린다)
    ocr: dict[str, Any] | None                # CV가 보낸 A.1 형식 OCR 결과. 트레이 사진 · 글만 온 메시지에는 없다
    text: str | None                          # 메시지의 글. 없으면 None
    route: list[str]                          # 슈퍼바이저가 정한 실행할 흐름 (실행 순서). 지나기 전에는 빈 목록
    flow: Flow | None                         # 대표 흐름 (사진의 흐름, 글만 오면 첫 흐름). 메모만 남기면 None
    proposals: list[dict[str, Any]]           # 글로 받은 수정 요청 · 복용 약 정보 (승인 카드의 제안. 자동으로 반영하지 않는다)
    memo: str | None                          # 메모로만 남긴 글
    order: dict[str, Any] | None              # 조제지시서 (DispensingOrder.model_dump(mode="json"))
    plan: dict[str, Any] | None               # 조제 계획 A.4 (Plan.model_dump(mode="json")). 승인 뒤에만 생긴다
    status: Status                            # 지금 어느 단계인지
    notice: str | None                        # 화면에 보여 줄 안내 (실패 원인, 승인 대기 안내, 되묻는 말)
    error_code: str | None                    # 실패했을 때 명세서 A.7 오류 코드 (E_OCR_EMPTY, E_LLM …)
    errors: Annotated[list[str], operator.add]   # 노드 실패 기록. 덮어쓰지 않고 쌓는다
    notes: Annotated[list[str], operator.add]    # 자리만 있는 흐름이 남긴 안내 (복용 약 등록 · 상담 등). 쌓는다


def initial_state(
    prescription_id: str,
    ocr: dict[str, Any] | None = None,
    photo_kind: str | None = None,
    text: str | None = None,
) -> GraphState:
    """그래프에 넣을 첫 상태."""
    return {
        "prescription_id": prescription_id,
        "photo_kind": photo_kind,
        "ocr": ocr,
        "text": text,
        "route": [],
        "flow": None,
        "proposals": [],
        "memo": None,
        "order": None,
        "plan": None,
        "status": "received",
        "notice": None,
        "error_code": None,
        "errors": [],
        "notes": [],
    }
