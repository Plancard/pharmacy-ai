"""웹이 부르는 Agent 입구 (10 · 11단계, 명세서 v0.8 2.2절, REQ-AG-01, REQ-WEB-02 · 03) — 처방 하나를 시작 · 멈춤 확인 · 이어 가기.

웹의 web/agent_gateway.py가 이미 이 모양을 쓴다 (지금은 FakeGateway). 웹 담당이 LangGraphGateway를 만들어 이것을 부르면
AGENT_BACKEND=langgraph로 같은 화면이 돈다.

    start(rx_id, ocr, photo_kind)      처방전 사진으로 처방을 시작한다. 승인 대기에서 멈추면 Pause를 돌려준다
    send(rx_id, ocr, photo_kind)       사진 한 장을 보낸다. 새 처방이면 시작하고, 있는 처방이면 이번 메시지 칸만 넣어 이어 간다
    pending(rx_id)                     지금 멈춰 있는 지점과 화면에 보일 내용 (없으면 None)
    resume(rx_id, hitl_point, reply)   약사 응답으로 이어 간다. 다시 멈추면 Pause, 끝나면 None
    stage(rx_id)                       멈춘 곳이 아닐 때 지금 단계 (조제 계획까지 끝났으면 'await_tray')
    result(rx_id)                      지금 상태의 조제지시서 · 조제 계획 · 검수 결과 · 안내
    approved_order(rx_id)              약사가 승인한 조제지시서 (트레이 대기 카드)
    check_reply(rx_id, hitl_point, reply)  응답이 지금 멈춘 지점에 맞는지만 본다 (웹이 요청을 받을 때 400으로 거르기)

처방 1건 = 대화 1개 = thread 1개다 (명세서 v0.8 REQ-WEB-07). 아래의 rx_id는 처방 ID이자 thread_id다.
v0.8에서 글 입력 · 상담 · 복용 약 등록 · 재대조(HITL ③) · 사진 종류 되묻기는 범위에서 빠졌다 (11/16).
메서드의 image_id · prescription_id는 웹이 아는 값을 넘긴다 (없으면 None).
멈춘 곳은 체크포인터(checkpoint.py)에 저장되므로 프로세스를 꺼도 같은 곳에서 이어 간다.
HITL ①(rx_approval)의 응답은 {"action": "approve" | "edit" | "manual", "order": 조제지시서, "reason": 사유}다.
  manual: 위험(danger) 항목이 있을 때 사유를 적고 진행한다 (기능분석서 F-WEB-07 ③). 위험이 있으면 approve는 거절된다.
  승인 카드(Pause.payload)에는 조제 계획(plan, A.4)이 함께 있다 (v0.8: 계획은 승인 앞). 만들지 못했으면 plan은 None이고
  error_code(E_PLAN_RANGE 등)와 notice에 이유가 있다.
  approve: 약품 줄이 그대로면 승인 카드의 계획을 그대로 확정하고 끝난다 → None, stage는 await_tray.
           약을 정하지 못한 줄이 있거나 계획이 없으면 거절된다 (ValueError, 고치기만 가능).
  edit (또는 줄을 바꾼 approve): 고친 값으로 약품 매칭 · 위험 신호 대조 · 조제 계획을 다시 하고 다시 승인 대기로 멈춘다.
           LLM은 다시 부르지 않는다.
트레이 사진(조제 계획이 있는 처방에 send(rx_id, None, "tray", image_id=...))은 CV로 검수한다 (14단계).
  전 칸 일치면 None, stage는 inspected. ok가 아닌 칸이 있으면 HITL ②(inspection)에서 멈춘다 → Pause.
  HITL ② 응답은 {"action": "confirm"}(확인 완료 → inspected) 또는 {"action": "redo"}(재조제 → await_tray, 다시 찍은 사진을 기다림).
  검수하지 못하면(트레이 못 찾음 E_TRAY_NOT_FOUND 등) AgentRunError. 단계는 await_tray 그대로라 다시 찍어 보내면 된다.
  검수 결과(A.3)는 result()의 inspection에 있다.
처방 처리가 실패하면(구조화 실패 등) 오류 코드(명세서 A.7)가 붙은 AgentRunError를 올린다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent import config
from agent.graph import build_graph, resume_run, run_message, thread_config
from agent.mcp_client import DrugTools, from_settings as mcp_from_settings
from agent.inspection import HITL_INSPECTION, read_inspection_reply
from agent.nodes import HITL_APPROVAL, read_reply
from agent.schema import DispensingOrder

# 승인된 조제지시서가 있는 단계 (approved_order가 돌려준다)
APPROVED_STATUSES = ("planned", "inspecting", "inspected")
STAGE_AFTER_PLAN = "await_tray"     # 조제 계획까지 끝나 트레이 사진을 기다리는 단계 (웹의 FakeGateway와 같은 이름)


@dataclass(frozen=True)
class Pause:
    hitl_point: str                 # 멈춘 지점 (rx_approval 등. 웹의 agent_gateway.HITL_POINTS)
    payload: dict[str, Any]         # rx_approval: {"order": A.5, "plan": A.4 | None, "error_code", "ocr_lines": {line_id: {text, conf, bbox}}, "ocr_size", "notice"}


class AgentRunError(Exception):
    """처방 처리 실패. code는 명세서 A.7 오류 코드, 메시지는 화면에 그대로 보여도 되는 문장이다."""

    def __init__(self, code: str | None, message: str, area: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.area = area     # 처방전 사진에서 강조할 곳 {width, height, lines: [{line_id, bbox}]} 또는 None (11/17 회의 Q6)


class AgentApi:
    def __init__(self, graph, tools: DrugTools | None = None):
        self.graph = graph
        self.tools = tools                     # 그래프가 쓰는 MCP 연결 (있으면 close()에서 닫는다)

    @classmethod
    def from_settings(cls, **kwargs) -> "AgentApi":
        """설정대로 만든 그래프로 (약품 사전 · 의약품 API · 체크포인터). 인자는 graph.build_graph와 같다.

        .env AGENT_USE_MCP(기본 1)이 켜져 있으면 약품 매칭 · DUR 대조 · 조제 계획 · 근거 검색을 MCP 도구로 부른다 (0이면 함수로 직접). AGENT_MCP_URL이 있으면 따로 띄운 서버에 그 주소로 붙고,
        없으면 서버(python -m agent.mcp_server)를 하위 프로세스로 띄운다.
        api=None을 주면 서버도 의약품 API 없이 띄운다.
        """
        if kwargs.get("tools") is None and config.use_mcp():
            kwargs["tools"] = mcp_from_settings(no_api="api" in kwargs and kwargs["api"] is None)
        return cls(build_graph(**kwargs), kwargs.get("tools"))

    def close(self) -> None:
        """MCP 연결을 닫는다 (없으면 아무 일도 하지 않는다)."""
        if self.tools is not None:
            self.tools.close()

    def _snapshot(self, rx_id: str):
        return self.graph.get_state(thread_config(rx_id))

    def start(self, rx_id: str, ocr: dict[str, Any] | None, photo_kind: str | None = "prescription", *,
              image_id: str | None = None,
              prescription_id: str | None = None) -> Pause | None:
        """처방전 사진으로 처방을 시작한다. 같은 ID로 두 번 시작하면 ValueError (새 처방전은 새 처방으로 연다)."""
        if self._snapshot(rx_id).values:
            raise ValueError(f"이미 시작한 처방입니다: {rx_id} (같은 처방의 다음 사진은 send로 보내세요)")
        return self.send(rx_id, ocr, photo_kind, image_id=image_id,
                         prescription_id=prescription_id)

    def send(self, rx_id: str, ocr: dict[str, Any] | None = None, photo_kind: str | None = None, *,
             image_id: str | None = None,
             prescription_id: str | None = None) -> Pause | None:
        """사진 한 장을 보낸다. 새 처방이면 시작하고, 있는 처방이면 이번 메시지 칸만 넣어 이어 간다 (승인된 조제지시서는 그대로).

        멈춰 있는 처방(승인 대기 · 검수 확인)에는 보내지 않는다: ValueError. 약사 응답은 resume으로.
        """
        pause = self.pending(rx_id) if self._snapshot(rx_id).values else None
        if pause is not None:
            raise ValueError(f"약사 응답을 기다리는 처방입니다: {rx_id} ({pause.hitl_point}). resume을 쓰세요")
        message = {"image_id": image_id, "kind": photo_kind}
        run_message(self.graph, rx_id, message, ocr, prescription_id)
        return self._after_run(rx_id)

    def pending(self, rx_id: str) -> Pause | None:
        interrupts = self._snapshot(rx_id).interrupts
        if not interrupts:
            return None
        value = dict(interrupts[0].value)
        return Pause(value.pop("hitl_point"), value)

    def resume(self, rx_id: str, hitl_point: str, reply: Any) -> Pause | None:
        """약사 응답으로 이어 간다. 멈춰 있지 않거나 지점이 다르거나 응답이 잘못되었으면 ValueError (그래프는 그대로 멈춰 있다)."""
        self.check_reply(rx_id, hitl_point, reply)          # 이어 가기 전에 거른다
        resume_run(self.graph, rx_id, reply)
        return self._after_run(rx_id)

    def check_reply(self, rx_id: str, hitl_point: str, reply: Any) -> None:
        """약사 응답이 지금 멈춘 지점에 맞는지만 본다 (그래프는 돌리지 않는다). 아니면 ValueError, 메시지는 화면에 보여도 되는 문장.

        웹은 응답을 백그라운드에서 이어 가므로(web/workflow.run_resume), 요청을 받을 때 이것으로 먼저 걸러 400으로 돌려준다.
        잘못된 응답이 백그라운드에서 처방을 '실패'로 만들지 않게 한다.
        """
        pause = self.pending(rx_id)
        if pause is None or pause.hitl_point != hitl_point:
            raise ValueError(f"멈춰 있는 지점이 없거나 다릅니다: {rx_id} ({hitl_point})")
        if hitl_point == HITL_APPROVAL:
            read_reply(reply, DispensingOrder.model_validate(pause.payload["order"]), plan_ready=pause.payload.get("plan") is not None)
        elif hitl_point == HITL_INSPECTION:
            read_inspection_reply(reply)

    def approved_order(self, rx_id: str) -> dict[str, Any] | None:
        """약사가 승인한 조제지시서 (트레이 대기 카드에 보인다). 아직 승인 전이거나 없는 대화면 None."""
        values = self._snapshot(rx_id).values
        if values.get("status") not in APPROVED_STATUSES:
            return None
        return values.get("order")

    def stage(self, rx_id: str) -> str | None:
        """멈춰 있으면 그 지점, 조제 계획까지 끝났으면 await_tray, 그 밖에는 상태 값. 없는 처방이면 None."""
        snapshot = self._snapshot(rx_id)
        if not snapshot.values:
            return None
        if snapshot.interrupts:
            return snapshot.interrupts[0].value["hitl_point"]
        status = snapshot.values.get("status")
        return STAGE_AFTER_PLAN if status == "planned" else status

    def result(self, rx_id: str) -> dict[str, Any]:
        """지금 저장된 상태에서 웹이 쓰는 칸만. 멈춰 있는 동안의 조제지시서는 pending의 payload에 있다."""
        values = self._snapshot(rx_id).values
        keys = ("conversation_id", "prescription_id", "status", "flow", "route", "order", "plan",
                "inspection", "notice", "error_code", "error_area", "notes")
        return {key: values.get(key) for key in keys}

    def _after_run(self, rx_id: str) -> Pause | None:
        pause = self.pending(rx_id)
        if pause is not None:
            return pause
        values = self._snapshot(rx_id).values
        if values.get("status") == "failed" or values.get("error_code"):     # 트레이 검수 실패는 상태가 planned 그대로다
            raise AgentRunError(values.get("error_code"), values.get("notice") or "처방 처리에 실패했습니다.",
                                values.get("error_area"))
        return None
