"""그래프의 상태 (7 ~ 11단계, 명세서 v0.8 2.2절, REQ-AG-01 · 04).

State에는 JSON으로 바꿀 수 있는 값(dict · list · str · None)만 담는다.
조제지시서는 dict로 담고, 노드 안에서 DispensingOrder로 다시 읽어 검증한다.
(체크포인터가 상태를 저장·복원할 때 Pydantic 객체보다 안전하다)

처방 1건 = 대화 1개 = LangGraph thread 1개다 (thread_id = conversation_id = prescription_id, 명세서 v0.8 REQ-WEB-07).
첫 메시지(처방전 사진)는 initial_state, 같은 처방의 다음 메시지(트레이 사진)는 message_input(이번 메시지 칸만)으로 넣는다
(graph.run_message가 고른다).

메시지는 사진 1장이다 (v0.8: 글 · 복합 입력 · 약봉투 · 포장 사진 · 사진 종류 판별 제외). 라우터(router.py)가 흐름을 정한다.

    prescription   처방 서브그래프: 구조화 → 약품 매칭 → 위험 신호 대조 → 조제 계획 → 승인 대기(멈춤, 승인하면 계획 확정)
    tray           트레이 검수: CV 검수 결과(A.3)를 받아, ok가 아닌 칸이 있으면 약사 확인(HITL ② inspection)에서 멈춘다 (14단계)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict, get_args

from agent.structure import agent_view

# 들어온 사진의 종류. 웹이 처방 상태로 정해 넘긴다 (새 처방이면 처방전, 승인된 처방이면 트레이. v0.8: 종류 판별 없음).
# 처방전은 A.1의 doc_type 값과 같다. 트레이는 OCR을 거치지 않는다.
PhotoKind = Literal["prescription", "tray"]
# 흐름. 흐름마다 상위 그래프의 노드가 하나씩 있다. 실행 순서는 router.FLOW_ORDER가 정한다.
Flow = Literal["prescription", "tray"]
# 진행 단계. 멈춘 것(승인 대기 · 검수 확인)은 상태 값이 아니라 그래프가 멈춘 것으로 안다 (checkpointer의 interrupts. api.py).
# edited는 약사가 고쳐 매칭 · 대조 · 계획을 다시 하는 상태, planned는 약사가 승인해 조제 계획이 확정된 상태다 (웹: waiting_tray).
# 조제 계획은 승인 전에 만들어 state.plan에 담기지만(v0.8), 승인 전 상태는 checked 그대로다.
# noted는 흐름을 돌리지 않고 안내만 남기고 끝난 것이다 (사진 종류가 없음, 조제 계획 전의 트레이 사진 등).
# inspecting은 트레이 검수 결과에 약사 확인이 필요해 HITL ②에서 멈춘 것, inspected는 검수가 끝난 것이다(전 칸 일치 또는 약사 확인).
# 트레이 검수가 실패하거나 재조제(redo)면 planned로 남는다 (웹: waiting_tray, 다시 찍은 사진을 기다린다).
Status = Literal[
    "received", "structured", "matched", "checked", "edited", "planned", "inspecting", "inspected",
    "noted", "failed",
]

PHOTO_KINDS: tuple[str, ...] = get_args(PhotoKind)
FLOWS: tuple[str, ...] = get_args(Flow)
STATUSES: tuple[str, ...] = get_args(Status)


class GraphState(TypedDict, total=False):
    conversation_id: str                      # 대화 ID (= 체크포인터의 thread_id). 대화 1개 = thread 1개 (11/12 결정)
    prescription_id: str | None               # 처방 ID. conversation_id와 같다 (웹이 넣어 준다. 빠지면 라우터가 채운다)
    # --- 이번 메시지 (사진 1장 = 라우터 한 번 실행). 같은 처방의 다음 메시지는 이 칸들만 바꾼다 (message_input)
    image_id: str | None                      # 사진의 ID (images 테이블). 사진이 없으면 None. 원본 바이트는 담지 않는다
    photo_kind: PhotoKind | None              # 들어온 사진 종류 (웹이 정한다). None이면 OCR이 있을 때만 처방전으로 본다
    ocr: dict[str, Any] | None                # CV가 보낸 A.1 형식 OCR 결과. 트레이 사진에는 없다
    # --- 흐름과 결과
    route: list[str]                          # 라우터가 정한 실행할 흐름 (실행 순서). 지나기 전에는 빈 목록
    flow: Flow | None                         # 이번 메시지의 흐름. 흐름을 돌리지 않으면 None
    order: dict[str, Any] | None              # 조제지시서 (DispensingOrder.model_dump(mode="json"))
    plan: dict[str, Any] | None               # 조제 계획 A.4 (Plan.model_dump(mode="json")). 승인 전에 만들어 승인 카드에 보인다 (v0.8)
    # 약사가 고치기(edit) 직전의 조제지시서. 바뀌지 않은 줄의 매칭과 위험 항목의 AI 설명 · 근거를 다시 쓴다 (11/17 회의 Q7). 승인하면 비운다
    previous_order: dict[str, Any] | None
    changed_items: list[int] | None           # 약사가 고친 뒤 다시 검수한 줄 번호(0부터). 승인 카드가 표시한다 (Q7). 고치기 전에는 None
    inspection: dict[str, Any] | None         # 마지막 트레이 검수 결과 A.3 (CV가 준 것을 확인한 것). 트레이 사진을 검수한 뒤에만 생긴다
    status: Status                            # 지금 어느 단계인지
    notice: str | None                        # 화면에 보여 줄 안내 (실패 원인, 승인 대기 안내, 되묻는 말)
    error_code: str | None                    # 실패했을 때 명세서 A.7 오류 코드 (E_OCR_EMPTY, E_LLM …)
    # 구조화에 실패했을 때 처방전 사진에서 강조할 OCR 줄 번호(A.1 line_id, 약품 표 본문, 11/17 회의 Q6). 특정할 수 없으면 빈 목록
    failed_line_ids: list[int]
    errors: Annotated[list[str], operator.add]   # 노드 실패 기록. 덮어쓰지 않고 쌓는다
    notes: Annotated[list[str], operator.add]    # 조제 기록에 남길 안내 (사유 입력 · 검수 확인 · AI 조사 등). 쌓는다


MESSAGE_KEYS = ("image_id", "kind")     # 웹이 넘기는 메시지 {image_id, kind}. 없는 키는 None (v0.8: 글 입력 · message_id 없음, 변경요청 11/16)


def read_message(message: dict[str, Any] | None) -> dict[str, Any]:
    """웹이 넘긴 메시지에서 상태에 담을 칸만 꺼낸다. 모르는 키가 있으면 ValueError (오타를 조용히 넘기지 않는다)."""
    message = dict(message or {})
    unknown = sorted(set(message) - set(MESSAGE_KEYS))
    if unknown:
        raise ValueError(f"메시지에 모르는 키가 있습니다: {', '.join(unknown)} (가능: {', '.join(MESSAGE_KEYS)})")
    return {
        "image_id": message.get("image_id"),
        "photo_kind": message.get("kind"),
    }


def message_input(
    message: dict[str, Any] | None,
    ocr: dict[str, Any] | None = None,
    prescription_id: str | None = None,
) -> dict[str, Any]:
    """같은 처방의 두 번째 이후 메시지(트레이 사진)로 그래프에 넣을 입력: 이번 메시지 칸만.

    order · plan · inspection · status는 넣지 않는다. 넣으면 앞서 승인된 조제지시서와 조제 계획이 지워진다 (체크포인터가 입력을 덮어쓴다).
    """
    data: dict[str, Any] = {
        **read_message(message),
        "ocr": agent_view(ocr) if ocr is not None else None,    # 표 밖 칸(환자 · 의료인 정보)의 글자는 비운다 (체크포인트에 남지 않게)
        "route": [],
        "flow": None,
        "notice": None,
        "error_code": None,
    }
    if prescription_id is not None:
        data["prescription_id"] = prescription_id
    return data


def initial_state(
    conversation_id: str,
    message: dict[str, Any] | None = None,
    prescription_id: str | None = None,
    ocr: dict[str, Any] | None = None,
) -> GraphState:
    """처방의 첫 메시지(처방전 사진)로 그래프에 넣을 상태. 웹이 prescription_id(= conversation_id)를 넣어 준다."""
    return {
        "conversation_id": conversation_id,
        "prescription_id": prescription_id,
        **message_input(message, ocr),
        "order": None,
        "plan": None,
        "previous_order": None,
        "changed_items": None,
        "failed_line_ids": [],
        "inspection": None,
        "status": "received",
        "errors": [],
        "notes": [],
    }
