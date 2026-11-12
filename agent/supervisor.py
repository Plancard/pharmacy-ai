"""슈퍼바이저 (11단계, 일정표 11/13, REQ-AG-01) — 메시지 하나를 흐름 여러 개로 나눠 정해진 순서로 실행할 계획을 세운다.

메시지 하나는 사진 1장, 글, 또는 둘 다다 (명세서 2.2절 복합 입력). 사진은 종류를, 글은 의도(intent.py)를 가른다.
흐름은 늘 이 순서로 실행한다 (FLOW_ORDER). 위험 신호 대조는 처방 흐름 안에서 한 번 돈다.

    ask_type → prescription → register_meds → recheck → tray → consult
    되묻기     처방 구조화     복용 약 등록      재대조    트레이   상담 답변

규칙 (사진이 있을 때)
- 처방전 사진: 처방 흐름. 글의 수정 요청 · 복용 약 정보는 승인 카드의 제안(proposal.py)으로만 붙는다. 질문이면 상담을 뒤에 더한다.
  그 밖의 글(메모 · 애매함)은 메모로 남긴다.
- 약봉투 · 포장 사진: 복용 약 재대조. 글의 복용 약 정보는 후보 제안, 질문이면 상담. 그 밖의 글은 메모.
- 트레이 사진: 트레이 검수. 글은 무엇이든 메모로만 남긴다 (검수 판정은 사진 기준).
- 사진 종류가 애매하면 되묻는다 (ask_type, HITL type_select). 약사가 고르면 그 종류로 다시 계획한다 (plan_after_choice).
규칙 (글만 올 때, 새 대화)
- 질문 → 상담. 복용 약 정보 → 복용 약 등록(후보 제안). 메모 → 메모만 남기고 끝.
- 수정 요청 → 고칠 처방이 없으므로 메모로 남기고 그렇게 알린다.
- 의도를 못 가르면 되묻는다 (ask_type, 글의 의도를 고른다).
이미 열린 처방에 글만 들어오면 그래프를 다시 돌리지 않는다. api.py의 handle_text가 같은 규칙으로 답한다.
규칙 (처방이 없는 대화 = 상담 대화, 11/12 결정)
- 글만 → 상담 (대조 없이 근거 기반 답변). 의도는 기록만 한다.
- 처방전 사진 → 처방 흐름. 대화가 처방 대화로 바뀐다 (prescription_id = conversation_id).
- 약봉투 · 포장 · 트레이 사진 → 1차 미지원. 흐름을 돌리지 않고 "처방 대화에서 올려 주세요"라고 알린다.

복용 약 등록 · 재대조 · 트레이 검수 · 상담은 아직 자리만 있다 (15 · 14 · 16단계). 그래프 층이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.drug_dictionary import DrugDictionary
from agent.intent import PATIENT_INFO, EDIT_REQUEST, INTENT_LABELS, NOTE, QUESTION, classify
from agent.proposal import make_proposals
from agent.state import PHOTO_KINDS, Flow

# 실행 순서 (명세서 2.2절: 처방 구조화 → 복용 약 등록 → 위험 신호 대조 한 번 → 상담 답변). 되묻기가 맨 앞이다.
FLOW_ORDER: tuple[str, ...] = ("ask_type", "prescription", "register_meds", "recheck", "tray", "consult")

# 사진 종류 -> 흐름
KIND_FLOW: dict[str, Flow] = {"prescription": "prescription", "tray": "tray", "med_bag": "recheck", "package": "recheck"}
KIND_LABELS = {"prescription": "처방전", "tray": "트레이", "med_bag": "약봉투", "package": "포장"}
DOC_KINDS = ("prescription", "med_bag")     # OCR(A.1)의 doc_type이 가질 수 있는 값

# 되묻기(type_select)의 선택지: 사진이면 종류, 글이면 의도
PHOTO_CHOICES: tuple[str, ...] = PHOTO_KINDS
TEXT_CHOICES: tuple[str, ...] = (PATIENT_INFO, QUESTION, NOTE)
ASK_TEXT_NOTICE = "글의 뜻을 정하지 못했습니다. 복용 약 정보 · 질문 · 메모 중에서 골라 주세요."
EDIT_WITHOUT_ORDER_NOTICE = "고칠 처방이 없어 글을 메모로 남겼습니다."
NO_RX_PHOTO_NOTICE = "처방이 없는 대화입니다. {kind} 사진은 처방 대화에서 올려 주세요."      # 상담 대화에 온 약봉투 · 포장 · 트레이 사진


@dataclass(frozen=True)
class MessagePlan:
    """메시지 하나를 어떻게 처리할지. 그래프의 상태 칸으로 그대로 옮긴다 (as_update)."""

    route: tuple[str, ...]                  # 실행할 흐름 (FLOW_ORDER 순서)
    flow: str | None = None                 # 대표 흐름 (사진의 흐름, 글만 오면 첫 흐름). 없으면 None (메모만)
    notice: str | None = None               # 되묻거나 알릴 말
    proposals: tuple[dict, ...] = ()        # 승인 카드에 보일 제안 (자동으로 반영하지 않는다)
    memo: str | None = None                 # 메모로만 남긴 글
    ask_about: str | None = None            # 되물을 때: 'photo'(사진 종류) 또는 'text'(글 의도)
    intents: tuple[str, ...] = field(default_factory=tuple)

    def as_update(self) -> dict[str, Any]:
        update: dict[str, Any] = {
            "route": list(self.route), "flow": self.flow, "proposals": [dict(p) for p in self.proposals], "memo": self.memo,
            "intents": list(self.intents),
        }
        if self.notice is not None:
            update["notice"] = self.notice
        return update


def _ask(reason: str) -> tuple[Flow, str]:
    return "ask_type", f"사진 종류를 정하지 못했습니다. {reason} 처방전 · 트레이 · 약봉투 · 포장 중에서 골라 주세요."


def decide_flow(photo_kind: Any, ocr: Any) -> tuple[Flow, str | None]:
    """사진 종류로 흐름을 정한다. (흐름, 사진 종류를 되물을 때의 안내) 를 돌려준다.

    두 단서를 본다: 들어온 사진 종류(photo_kind)와 OCR이 읽은 문서 종류(ocr의 doc_type).
    - 한쪽만 있으면 그것을 따른다. 둘 다 있고 같으면 그것을 따른다.
    - 둘 다 없거나, 알 수 없는 값이거나, 서로 다르면 맞혀 보지 않고 되묻는다 (ask_type).
      처방전을 약봉투로 잘못 다루면 약사가 모르고 지나가기 때문이다.
    """
    if photo_kind is not None and photo_kind not in PHOTO_KINDS:
        return _ask("알 수 없는 사진 종류입니다.")
    doc_type = ocr.get("doc_type") if isinstance(ocr, dict) else None
    if doc_type is not None and doc_type not in DOC_KINDS:
        return _ask("OCR이 읽은 문서 종류를 알 수 없습니다.")
    if photo_kind is not None and doc_type is not None and photo_kind != doc_type:
        return _ask(f"지정한 종류({KIND_LABELS[photo_kind]})와 OCR이 읽은 종류({KIND_LABELS[doc_type]})가 다릅니다.")
    kind = photo_kind or doc_type
    if kind is None:
        return _ask("사진 종류 정보가 없습니다.")
    return KIND_FLOW[kind], None


def ordered(flows) -> tuple[str, ...]:
    """흐름을 실행 순서로 (겹친 것은 한 번만)."""
    wanted = set(flows)
    return tuple(f for f in FLOW_ORDER if f in wanted)


def _with_text(flow: Flow, text: str | None, intents: list[str], dictionary: DrugDictionary) -> MessagePlan:
    """사진의 흐름이 정해진 뒤, 같이 온 글을 붙인다."""
    if not (text or "").strip():
        return MessagePlan(route=(flow,), flow=flow)
    if flow == "tray":                                   # 트레이 사진과 온 글은 메모로만
        return MessagePlan(route=(flow,), flow=flow, memo=text.strip(), intents=tuple(intents))
    usable = {"prescription": (EDIT_REQUEST, PATIENT_INFO), "recheck": (PATIENT_INFO,)}[flow]
    proposals = tuple(make_proposals(text, [i for i in intents if i in usable], dictionary))
    route = [flow] + (["consult"] if QUESTION in intents else [])
    used = set(usable) | {QUESTION}
    memo = text.strip() if not any(i in used for i in intents) else None
    return MessagePlan(route=ordered(route), flow=flow, proposals=proposals, memo=memo, intents=tuple(intents))


def _text_only(text: str, intents: list[str], dictionary: DrugDictionary) -> MessagePlan:
    if not intents:
        return MessagePlan(route=("ask_type",), flow="ask_type", notice=ASK_TEXT_NOTICE, ask_about="text")
    route = (["consult"] if QUESTION in intents else []) + (["register_meds"] if PATIENT_INFO in intents else [])
    proposals = tuple(make_proposals(text, [PATIENT_INFO] if PATIENT_INFO in intents else [], dictionary))
    notice = None
    memo = None
    if not route:                                        # 수정 요청 · 메모만: 메모로 남기고 끝
        memo = text.strip()
        notice = EDIT_WITHOUT_ORDER_NOTICE if EDIT_REQUEST in intents else None
    route_t = ordered(route)
    return MessagePlan(route=route_t, flow=route_t[0] if route_t else None, notice=notice, proposals=proposals,
                       memo=memo, intents=tuple(intents))


def has_photo(photo_kind: Any, ocr: Any, text: Any) -> bool:
    """사진이 왔는가. 사진 종류나 OCR이 있으면 사진이 온 것이고, 아무것도 없이 글만 있으면 글만 온 메시지다.

    사진도 글도 단서가 없으면 사진이 왔는데 종류를 모르는 것으로 본다 (되묻는다).
    """
    return photo_kind is not None or ocr is not None or not (text or "").strip()


def _consult_only(text: str | None, intents: list[str]) -> MessagePlan:
    """처방이 없는 대화에 글만 왔을 때: 상담. 고칠 처방도 대조할 처방약도 없으므로 제안 · 메모를 만들지 않는다."""
    return MessagePlan(route=("consult",), flow="consult", intents=tuple(intents))


def _photo_without_prescription(kind: str, text: str | None, intents: list[str]) -> MessagePlan:
    """처방이 없는 대화에 처방전이 아닌 사진(약봉투 · 포장 · 트레이)이 왔을 때. 1차는 돌리지 않고 알린다. 글은 메모로 남긴다."""
    memo = text.strip() if (text or "").strip() else None
    return MessagePlan(route=(), notice=NO_RX_PHOTO_NOTICE.format(kind=KIND_LABELS[kind]), memo=memo, intents=tuple(intents))


def _plan_photo(kind: str, text: str | None, intents: list[str], dictionary: DrugDictionary,
                has_prescription: bool) -> MessagePlan:
    flow = KIND_FLOW[kind]
    if not has_prescription and flow != "prescription":
        return _photo_without_prescription(kind, text, intents)
    return _with_text(flow, text, intents, dictionary)


def plan_message(photo_kind: Any, ocr: Any, text: str | None, dictionary: DrugDictionary,
                 has_prescription: bool = True) -> MessagePlan:
    """메시지 하나(사진 · 글)를 어떤 흐름으로 어떤 순서로 실행할지 정한다.

    has_prescription은 이 대화에 처방이 있는가다 (state의 prescription_id). 없으면 상담 대화의 규칙을 따른다.
    """
    intents = classify(text)
    if not has_photo(photo_kind, ocr, text):
        return _text_only(text, intents, dictionary) if has_prescription else _consult_only(text, intents)
    flow, notice = decide_flow(photo_kind, ocr)
    if flow == "ask_type":
        return MessagePlan(route=("ask_type",), flow="ask_type", notice=notice, ask_about="photo", intents=tuple(intents))
    kind = photo_kind or ocr.get("doc_type")
    return _plan_photo(kind, text, intents, dictionary, has_prescription)


def ask_choices(about: str) -> tuple[str, ...]:
    return PHOTO_CHOICES if about == "photo" else TEXT_CHOICES


def choice_labels(about: str) -> dict[str, str]:
    return dict(KIND_LABELS) if about == "photo" else {c: INTENT_LABELS[c] for c in TEXT_CHOICES}


def plan_after_choice(about: str, choice: str, text: str | None, dictionary: DrugDictionary,
                      has_prescription: bool = True) -> MessagePlan:
    """되묻기에서 약사가 고른 뒤의 계획. 약사가 고른 것을 따른다 (OCR의 doc_type과 달라도 다시 묻지 않는다)."""
    if choice not in ask_choices(about):
        raise ValueError(f"고를 수 있는 값이 아닙니다: {choice!r} (가능: {', '.join(ask_choices(about))})")
    if about == "photo":
        return _plan_photo(choice, text, classify(text), dictionary, has_prescription)
    return _text_only(text or "", [choice], dictionary)
