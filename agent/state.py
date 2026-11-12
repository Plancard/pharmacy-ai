"""그래프의 상태 (7 ~ 11단계, 일정표 11/9 ~ 11/13, REQ-AG-01 · 04).

State에는 JSON으로 바꿀 수 있는 값(dict · list · str · None)만 담는다.
조제지시서는 dict로 담고, 노드 안에서 DispensingOrder로 다시 읽어 검증한다.
(체크포인터가 상태를 저장·복원할 때 Pydantic 객체보다 안전하다)

대화 1개 = LangGraph thread 1개다 (thread_id = conversation_id, 11/12 결정). 처방 대화는 conversation_id = prescription_id이고,
상담 대화(처방전 없이 글로 시작)는 prescription_id가 None이다. 상담 대화에 처방전 사진이 오면 같은 ID로 처방 대화가 된다.
첫 메시지는 initial_state, 같은 대화의 다음 메시지는 message_input(이번 메시지 칸만)으로 넣는다 (graph.run_message가 고른다).

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

# 들어온 사진의 종류 (기능분석서 v0.3 F-WEB-05의 버튼 순서: 처방전 · 트레이 · 약봉투 · 포장).
# 처방전 · 약봉투는 A.1의 doc_type 값과 같다. 트레이는 OCR을 거치지 않고, 포장은 바코드(F-CV-07)로 읽는다.
PhotoKind = Literal["prescription", "tray", "med_bag", "package"]
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
    conversation_id: str                      # 대화 ID (= 체크포인터의 thread_id). 대화 1개 = thread 1개 (11/12 결정)
    prescription_id: str | None               # 처방 ID. 처방 대화에서는 conversation_id와 같고, 상담 대화(처방 없음)에서는 None
    # --- 이번 메시지 (메시지 하나 = 슈퍼바이저 한 번 실행). 같은 대화의 다음 메시지는 이 칸들만 바꾼다 (message_input)
    message_id: str | None                    # 웹이 발급한 메시지 ID (messages 테이블). 없으면 None
    image_id: str | None                      # 사진의 ID (images 테이블). 사진이 없으면 None. 원본 바이트는 담지 않는다
    photo_kind: PhotoKind | None              # 들어온 사진 종류. 모르면 None (OCR의 doc_type으로 가린다)
    ocr: dict[str, Any] | None                # CV가 보낸 A.1 형식 OCR 결과. 트레이 사진 · 글만 온 메시지에는 없다
    text: str | None                          # 메시지의 글. 없으면 None
    intents: list[str]                        # 글의 의도 (intent.py: patient_info · edit_request · question · note). 슈퍼바이저가 채운다
    # --- 흐름과 결과
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


MESSAGE_KEYS = ("message_id", "text", "image_id", "kind")     # 웹이 넘기는 메시지 {message_id, text, image_id, kind}. 없는 키는 None


def read_message(message: dict[str, Any] | None) -> dict[str, Any]:
    """웹이 넘긴 메시지에서 상태에 담을 칸만 꺼낸다. 모르는 키가 있으면 ValueError (오타를 조용히 넘기지 않는다)."""
    message = dict(message or {})
    unknown = sorted(set(message) - set(MESSAGE_KEYS))
    if unknown:
        raise ValueError(f"메시지에 모르는 키가 있습니다: {', '.join(unknown)} (가능: {', '.join(MESSAGE_KEYS)})")
    return {
        "message_id": message.get("message_id"),
        "image_id": message.get("image_id"),
        "photo_kind": message.get("kind"),
        "text": message.get("text"),
    }


def message_input(
    message: dict[str, Any] | None,
    ocr: dict[str, Any] | None = None,
    prescription_id: str | None = None,
) -> dict[str, Any]:
    """같은 대화의 두 번째 이후 메시지로 그래프에 넣을 입력: 이번 메시지 칸만.

    order · plan · status는 넣지 않는다. 넣으면 앞서 승인된 조제지시서와 조제 계획이 지워진다 (체크포인터가 입력을 덮어쓴다).
    상담 대화가 처방 대화로 바뀌는 메시지(처방전 사진)면 prescription_id도 넣는다.
    """
    data: dict[str, Any] = {
        **read_message(message),
        "ocr": ocr,
        "intents": [],
        "route": [],
        "flow": None,
        "proposals": [],
        "memo": None,
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
    """대화의 첫 메시지로 그래프에 넣을 상태. 처방 대화면 웹이 prescription_id(= conversation_id)를 넣어 준다."""
    return {
        "conversation_id": conversation_id,
        "prescription_id": prescription_id,
        **message_input(message, ocr),
        "order": None,
        "plan": None,
        "status": "received",
        "errors": [],
        "notes": [],
    }
