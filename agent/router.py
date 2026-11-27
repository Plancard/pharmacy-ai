"""라우터 (11단계, 명세서 2.2절 · REQ-AG-01) — 사진 한 장을 어느 흐름으로 보낼지 정한다.

v0.8에서 범위가 처방 단일 흐름으로 줄었다 (11/16). 글 입력 · 복합 입력 · 상담 · 복용 약 재대조 · 약봉투 · 포장 사진 ·
사진 종류 판별(되묻기)은 없다. 흐름은 둘이고 이 순서다 (FLOW_ORDER).

    prescription → tray
    처방 흐름      트레이 검수

규칙
- 사진 종류(photo_kind)는 웹이 처방 상태로 정해 넘긴다: 새 처방이면 처방전, 승인된 처방(트레이 대기)이면 트레이.
- 종류가 없으면 OCR(A.1)이 있을 때만 처방전으로 본다. 둘 다 없거나 모르는 값이면 흐름을 돌리지 않고 알린다.
- 처방전 사진은 조제지시서가 아직 없는 처방에만 받는다. 이미 처방전이 있는 처방에 또 오면 돌리지 않고 알린다
  (새 처방전은 새 처방으로 연다).
- 트레이 사진은 트레이 흐름으로 보낸다. 조제 계획이 없으면(승인 전) 트레이 노드가 안내만 남긴다.

그래프 층이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.state import PHOTO_KINDS

# 실행 순서 (명세서 2.2절: 처방 흐름 1~6단계 → 트레이 검수 7~9단계)
FLOW_ORDER: tuple[str, ...] = ("prescription", "tray")

DOC_KINDS = ("prescription",)                # OCR(A.1)의 doc_type이 가질 수 있는 값

NO_KIND_NOTICE = "사진 종류 정보가 없습니다. 처방전 사진은 새 처방으로, 트레이 사진은 승인된 처방에 올려 주세요."
UNKNOWN_KIND_NOTICE = "받을 수 없는 사진 종류입니다: {kind} (처방전 · 트레이만 받습니다)."
UNKNOWN_DOC_NOTICE = "OCR이 읽은 문서 종류가 처방전이 아닙니다: {doc_type}."
HAS_ORDER_NOTICE = "이 처방에는 이미 처방전이 있습니다. 새 처방전은 새 처방으로 올려 주세요."


@dataclass(frozen=True)
class MessagePlan:
    """사진 한 장을 어떻게 처리할지. 그래프의 상태 칸으로 그대로 옮긴다 (as_update)."""

    route: tuple[str, ...]                  # 실행할 흐름 (FLOW_ORDER 순서). 비면 흐름을 돌리지 않는다
    flow: str | None = None                 # 이번 사진의 흐름. 돌리지 않으면 None
    notice: str | None = None               # 흐름을 돌리지 않을 때 알릴 말

    def as_update(self) -> dict[str, Any]:
        update: dict[str, Any] = {"route": list(self.route), "flow": self.flow}
        if self.notice is not None:
            update["notice"] = self.notice
        return update


def _skip(notice: str) -> MessagePlan:
    return MessagePlan(route=(), notice=notice)


def plan_message(photo_kind: Any, ocr: Any, has_order: bool = False) -> MessagePlan:
    """사진 한 장을 어느 흐름으로 보낼지 정한다. has_order는 이 처방에 조제지시서가 이미 있는가다."""
    doc_type = ocr.get("doc_type") if isinstance(ocr, dict) else None
    kind = photo_kind
    if kind is None:
        if ocr is None:
            return _skip(NO_KIND_NOTICE)
        kind = "prescription"
    if kind not in PHOTO_KINDS:
        return _skip(UNKNOWN_KIND_NOTICE.format(kind=kind))
    if kind == "prescription":
        if doc_type is not None and doc_type not in DOC_KINDS:
            return _skip(UNKNOWN_DOC_NOTICE.format(doc_type=doc_type))
        if has_order:
            return _skip(HAS_ORDER_NOTICE)
    return MessagePlan(route=(kind,), flow=kind)
