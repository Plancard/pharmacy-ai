"""트레이 검수 결과 다루기 (14단계, 일정표 11/13, REQ-CV-04 · 05, REQ-WEB-04, 명세서 2.2절 HITL ②).

검수 판정(칸마다 ok · mismatch · need_review)은 CV가 한다 (cv.inspect(image, plan) -> A.3, 기능분석서 F-CV-05).
Agent는 판정을 바꾸지 않는다. 여기서 하는 일은 세 가지다.

1. 받은 A.3이 지금 조제 계획에 대한 결과인지 확인한다 (read_inspection).
   - A.3 스키마(12칸, summary와 칸 수가 맞음)
   - 처방 ID가 같고, 칸마다 expected가 지금 계획과 같을 것. 다르면 CV가 다른(옛) 계획으로 판정한 것이라 믿지 않는다.
   - hitl_required가 칸 판정과 맞을 것 (ok가 아닌 칸이 있으면 반드시 true). 어긋나면 약사 확인 쪽으로 본다.
2. 약사에게 보일 내용과 안내 (inspection_payload · inspection_notice).
3. HITL ② 응답을 읽는다 (read_inspection_reply): {"action": "confirm"} 확인 완료 · {"action": "redo"} 재조제 (기능분석서 F-WEB-09).

판단 층이다. 파일도 네트워크도 쓰지 않는다.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agent.errors import E_CV_INFER
from agent.plan import TIME_LABELS
from agent.schema import Inspection, Plan

HITL_INSPECTION = "inspection"           # HITL ② 지점 이름 (기능분석서 5.2절)
CONFIRM, REDO = "confirm", "redo"        # 약사 응답의 action (기능분석서 F-WEB-09)
STATUS_LABELS = {"ok": "일치", "mismatch": "불일치", "need_review": "확인 필요"}
REASON_LABELS = {"low_conf": "신뢰도 낮음", "boundary": "칸 경계에 걸침", "overlap": "알약 겹침", "rotated": "트레이가 뒤집혀 찍힘"}


class InspectionError(Exception):
    """CV가 준 검수 결과를 쓸 수 없음. code는 A.7 오류 코드, 메시지는 화면에 그대로 보여도 되는 문장이다."""

    def __init__(self, message: str, code: str | None = E_CV_INFER):
        super().__init__(message)
        self.code = code


def plan_counts(plan: Plan) -> dict[tuple[int, str], dict[str, int]]:
    """조제 계획을 칸마다 {품목기준코드: 알 수}로. 계획에 없는 칸은 빠진다 (빈 칸)."""
    return {(c.day, c.time): {i.item_code: i.count for i in c.items} for c in plan.plan}


def read_inspection(data: Any, plan: Plan) -> Inspection:
    """CV가 준 A.3을 읽고 지금 조제 계획에 대한 결과인지 확인한다. 쓸 수 없으면 InspectionError(E_CV_INFER).

    hitl_required가 false인데 ok가 아닌 칸이 있으면 true로 고쳐 읽는다 (약사 확인을 건너뛰지 않는다, REQ-NF-03).
    """
    try:
        inspection = Inspection.model_validate(data)
    except ValidationError as e:
        raise InspectionError(f"검수 결과(A.3) 형식이 아닙니다: {e.error_count()}곳이 틀립니다.") from None
    if inspection.prescription_id != plan.prescription_id:
        raise InspectionError(f"다른 처방의 검수 결과입니다 ({inspection.prescription_id} != {plan.prescription_id}).")
    expected = plan_counts(plan)
    differ = [f"{c.day}일 {TIME_LABELS[c.time]}" for c in inspection.cells
              if {k: v for k, v in c.expected.items() if v} != expected.get((c.day, c.time), {})]
    if differ:
        raise InspectionError("검수 결과가 지금 조제 계획과 맞지 않습니다 (다른 계획으로 판정함): " + ", ".join(differ))
    not_ok = inspection.summary.mismatch + inspection.summary.need_review
    if not_ok and not inspection.summary.hitl_required:
        inspection = inspection.model_copy(update={
            "summary": inspection.summary.model_copy(update={"hitl_required": True})})
    return inspection


def needs_pharmacist(inspection: Inspection) -> bool:
    """HITL ②를 열어야 하는가: ok가 아닌 칸이 하나라도 있으면 (명세서 2.2절 8단계)."""
    return inspection.summary.hitl_required


def cell_lines(inspection: Inspection, names: dict[str, str] | None = None) -> list[str]:
    """ok가 아닌 칸마다 한 줄: '1일 아침 불일치: 예상 타이레놀정500mg 1 / 실제 없음'."""
    names = names or {}

    def content(counts: dict[str, int]) -> str:
        shown = [f"{names.get(code, code)} {n}" for code, n in counts.items() if n]
        return " · ".join(shown) if shown else "없음"

    lines = []
    for c in inspection.cells:
        if c.status == "ok":
            continue
        line = f"{c.day}일 {TIME_LABELS[c.time]} {STATUS_LABELS[c.status]}: 예상 {content(c.expected)} / 실제 {content(c.actual)}"
        if c.reasons:
            line += f" ({', '.join(REASON_LABELS[r] for r in c.reasons)})"
        lines.append(line)
    return lines


def inspection_notice(inspection: Inspection) -> str:
    s = inspection.summary
    if not s.hitl_required:
        return f"트레이 검수: 전 칸 일치 ({s.ok}칸)."
    rotated = any("rotated" in c.reasons for c in inspection.cells)
    if rotated:
        return "트레이가 180° 뒤집혀 찍힌 것 같습니다. 아침 줄이 위로 오게 다시 찍어 주세요."
    return (f"트레이 검수: 불일치 {s.mismatch}칸, 확인 필요 {s.need_review}칸. "
            "약사가 확인해 주세요 (확인 완료 또는 재조제).")


def inspection_payload(inspection: Inspection, plan: Plan, names: dict[str, str] | None = None) -> dict[str, Any]:
    """HITL ②에서 웹에 보일 내용 (기능분석서 F-WEB-09: 칸별 색 표시, 예상 · 실제와 사유)."""
    return {
        "hitl_point": HITL_INSPECTION,
        "inspection": inspection.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "names": dict(names or {}),
        "cells": cell_lines(inspection, names),
        "notice": inspection_notice(inspection),
        "choices": [CONFIRM, REDO],
    }


def read_inspection_reply(reply: Any) -> str:
    """HITL ② 응답 {"action": "confirm" | "redo"}을 읽는다. 잘못된 응답이면 ValueError."""
    action = reply.get("action") if isinstance(reply, dict) else None
    if action not in (CONFIRM, REDO):
        raise ValueError("검수 확인 응답의 action은 confirm(확인 완료) · redo(재조제) 중 하나여야 합니다.")
    return action
