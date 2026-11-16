"""웹이 부르는 Agent 입구 (10 · 11단계, 일정표 11/12 · 11/13, REQ-AG-01, REQ-WEB-02 · 03) — 처방 하나를 시작 · 멈춤 확인 · 이어 가기.

웹의 web/agent_gateway.py가 이미 이 모양을 쓴다 (지금은 FakeGateway). 웹 담당이 LangGraphGateway를 만들어 이것을 부르면
AGENT_BACKEND=langgraph로 같은 화면이 돈다.

    start(rx_id, ocr, photo_kind, text) 메시지 하나(사진 · 글)로 대화를 시작한다. 승인 대기 · 되묻기에서 멈추면 Pause를 돌려준다
    send(conv_id, ocr, photo_kind, text) 메시지 하나를 보낸다. 새 대화면 시작하고, 있는 대화면 이번 메시지 칸만 넣어 이어 간다
    pending(rx_id)                     지금 멈춰 있는 지점과 화면에 보일 내용 (없으면 None)
    resume(rx_id, hitl_point, reply)   약사 응답으로 이어 간다. 다시 멈추면 Pause, 끝나면 None
    stage(rx_id)                       멈춘 곳이 아닐 때 지금 단계 (조제 계획까지 끝났으면 'await_tray')
    result(rx_id)                      지금 상태의 조제지시서 · 조제 계획 · 안내 · 제안 · 메모
    handle_text(rx_id, text)           이미 열린 처방에 글만 들어왔을 때: 그래프를 돌리지 않고 제안 · 재검토 표시 · 메모 · 안내를 돌려준다
    register_meds(conv_id, meds)       복용 약 [{item_code, name}]을 등록하고 처방약과 다시 대조한다 (HITL ③ risk에서 멈출 수 있다)
    search_drugs(name)                 복용 약 수동 입력의 약 검색 (REQ-WEB-09). MCP search_drug와 같은 결과를 후보 목록으로
    approved_order(rx_id)              약사가 승인한 조제지시서 (트레이 대기 카드)
    check_reply(rx_id, hitl_point, reply)  응답이 지금 멈춘 지점에 맞는지만 본다 (웹이 요청을 받을 때 400으로 거르기)

thread_id는 대화 ID다 (11/12 결정: 대화 1개 = thread 1개). 처방 대화에서는 대화 ID = 처방 ID라서 아래의 rx_id는 둘 다를 뜻한다.
처방 없이 글로 시작한 대화는 prescription_id가 None이고 글은 메모로 남는다 (11/16 상담 답변 제외). 처방전 사진이 오면 같은 ID로 처방 대화가 된다.
메서드의 message_id · image_id · prescription_id는 웹이 아는 값을 넘긴다 (없으면 None).
멈춘 곳은 체크포인터(checkpoint.py)에 저장되므로 프로세스를 꺼도 같은 곳에서 이어 간다.
HITL ①(rx_approval)의 응답은 {"action": "approve" | "edit" | "manual", "order": 조제지시서, "reason": 사유}다.
  manual: 위험(danger) 항목이 있을 때 사유를 적고 진행한다 (기능분석서 F-WEB-07 ③). 위험이 있으면 approve는 거절된다.
  approve: 약품 줄이 그대로면 조제 계획을 만들고 끝난다 → None, stage는 await_tray.
           계획을 만들 수 없으면(3일 초과 · 칸당 2알 초과 등) 이유를 안내에 담아 다시 승인 대기로 멈춘다.
  edit (또는 줄을 바꾼 approve): 고친 값으로 약품 매칭 · 위험 신호 대조를 다시 하고 다시 승인 대기로 멈춘다. LLM은 다시 부르지 않는다.
HITL type_select(되묻기)의 응답은 {"choice": 값}이다. 고를 수 있는 값은 Pause.payload의 choices에 있다.
트레이 사진(조제 계획이 있는 처방 대화에 send(conv_id, None, "tray", image_id=...))은 CV로 검수한다 (14단계).
  전 칸 일치면 None, stage는 inspected. ok가 아닌 칸이 있으면 HITL ②(inspection)에서 멈춘다 → Pause.
  HITL ② 응답은 {"action": "confirm"}(확인 완료 → inspected) 또는 {"action": "redo"}(재조제 → await_tray, 다시 찍은 사진을 기다림).
  검수하지 못하면(트레이 못 찾음 E_TRAY_NOT_FOUND 등) AgentRunError. 단계는 await_tray 그대로라 다시 찍어 보내면 된다.
  검수 결과(A.3)는 result()의 inspection에 있다.
복용 약 등록(register_meds(conv_id, [{item_code, name}]))은 처방약과 다시 대조해 새 항목만 result()의 consult_result(A.6)에 담는다 (15단계).
  새 항목에 위험(danger)이 있으면 HITL ③(risk)에서 멈춘다. 응답은 {"action": "acknowledge"}.
처방 처리가 실패하면(구조화 실패 등) 오류 코드(명세서 A.7)가 붙은 AgentRunError를 올린다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent import config
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.graph import build_graph, resume_run, run_message, thread_config
from agent.drug_api import default_api
from agent.drug_match import ApiSearch, search_drug_result
from agent.errors import ConfigError, DrugApiError
from agent.mcp_client import DrugTools, McpToolError, from_settings as mcp_from_settings
from agent.intent import PATIENT_INFO, EDIT_REQUEST, NOTE, QUESTION, classify
from agent.inspection import HITL_INSPECTION, read_inspection_reply
from agent.nodes import HITL_APPROVAL, HITL_RISK, HITL_TYPE, read_choice, read_reply, read_risk_reply
from agent.recheck import read_meds
from agent.proposal import make_proposals
from agent.schema import DispensingOrder
from agent.supervisor import NO_CONSULT_NOTICE, TEXT_CHOICES

# 승인된 조제지시서가 있는 단계 (approved_order가 돌려준다). 처방 대화에서 재대조 · 상담 뒤에도 이 단계는 그대로다
APPROVED_STATUSES = ("approved", "planned", "inspecting", "inspected")
STAGE_AFTER_PLAN = "await_tray"     # 조제 계획까지 끝나 트레이 사진을 기다리는 단계 (웹의 FakeGateway와 같은 이름)
REVIEW_NEEDED_NOTICE = "조제 내용 재검토 필요: 승인이 끝난 처방에 수정 요청이 들어왔습니다."     # 명세서 2.2절


@dataclass(frozen=True)
class Pause:
    hitl_point: str                 # 멈춘 지점 (rx_approval 등. 웹의 agent_gateway.HITL_POINTS)
    payload: dict[str, Any]         # rx_approval: {"order": A.5, "ocr_lines": {line_id: {text, conf, bbox}}, "ocr_size", "notice", "proposals", "memo"}


class AgentRunError(Exception):
    """처방 처리 실패. code는 명세서 A.7 오류 코드, 메시지는 화면에 그대로 보여도 되는 문장이다."""

    def __init__(self, code: str | None, message: str):
        super().__init__(message)
        self.code = code


class AgentApi:
    def __init__(self, graph, dictionary: DrugDictionary | None = None, tools: DrugTools | None = None,
                 drug_api: ApiSearch | None = None):
        self.graph = graph
        self._dictionary = dictionary          # 글에서 약 이름을 찾을 때 쓴다 (handle_text). 없으면 처음 쓸 때 설정의 사전을 읽는다
        self.tools = tools                     # 그래프가 쓰는 MCP 연결 (있으면 close()에서 닫는다)
        self.drug_api = drug_api               # 약 검색(search_drugs)에 쓰는 의약품 API. MCP가 있으면 서버의 것을 쓴다

    @classmethod
    def from_settings(cls, **kwargs) -> "AgentApi":
        """설정대로 만든 그래프로 (약품 사전 · 의약품 API · 체크포인터). 인자는 graph.build_graph와 같다.

        .env에 AGENT_USE_MCP=1 이면 약품 매칭 · DUR 대조를 MCP 도구로 부른다. AGENT_MCP_URL이 있으면 따로 띄운 서버에 그 주소로 붙고,
        없으면 서버(python -m agent.mcp_server)를 하위 프로세스로 띄운다.
        api=None을 주면 서버도 의약품 API 없이 띄운다.
        """
        if kwargs.get("tools") is None and config.use_mcp():
            kwargs["tools"] = mcp_from_settings(no_api="api" in kwargs and kwargs["api"] is None)
        drug_api = kwargs["api"] if "api" in kwargs else default_api()
        return cls(build_graph(**kwargs), kwargs.get("dictionary"), kwargs.get("tools"), drug_api)

    def close(self) -> None:
        """MCP 연결을 닫는다 (없으면 아무 일도 하지 않는다)."""
        if self.tools is not None:
            self.tools.close()

    @property
    def dictionary(self) -> DrugDictionary:
        if self._dictionary is None:
            self._dictionary = load_dictionary()
        return self._dictionary

    def _snapshot(self, rx_id: str):
        return self.graph.get_state(thread_config(rx_id))

    def start(self, rx_id: str, ocr: dict[str, Any] | None, photo_kind: str | None = "prescription",
              text: str | None = None, *, message_id: str | None = None, image_id: str | None = None,
              prescription_id: str | None = None) -> Pause | None:
        """메시지 하나(사진 · 글)로 대화를 시작한다. 같은 ID로 두 번 시작하면 ValueError (새 처방전은 새 대화로 연다).

        글만 보낼 때는 ocr=None, photo_kind=None, text=글. 처방이 없는 대화의 글은 메모로 남고 처방전 사진을 올리라고 안내한다.
        """
        if self._snapshot(rx_id).values:
            raise ValueError(f"이미 시작한 대화입니다: {rx_id} (같은 대화의 다음 메시지는 send로 보내세요)")
        return self.send(rx_id, ocr, photo_kind, text, message_id=message_id, image_id=image_id,
                         prescription_id=prescription_id)

    def send(self, conv_id: str, ocr: dict[str, Any] | None = None, photo_kind: str | None = None,
             text: str | None = None, *, message_id: str | None = None, image_id: str | None = None,
             prescription_id: str | None = None) -> Pause | None:
        """메시지 하나를 보낸다. 새 대화면 시작하고, 있는 대화면 이번 메시지 칸만 넣어 이어 간다 (승인된 조제지시서는 그대로).

        멈춰 있는 대화(승인 대기 · 되묻기)에는 보내지 않는다: ValueError. 약사 응답은 resume, 승인 대기 중의 글은 handle_text로.
        처방 없던 대화가 처방 대화로 바뀌는 메시지(처방전 사진)면 웹이 prescription_id(= conv_id)를 넣어 준다.
        """
        pause = self.pending(conv_id) if self._snapshot(conv_id).values else None
        if pause is not None:
            raise ValueError(f"약사 응답을 기다리는 대화입니다: {conv_id} ({pause.hitl_point}). resume 또는 handle_text를 쓰세요")
        message = {"message_id": message_id, "text": text, "image_id": image_id, "kind": photo_kind}
        run_message(self.graph, conv_id, message, ocr, prescription_id)
        return self._after_run(conv_id)

    def register_meds(self, conv_id: str, meds: list[dict[str, str]], *, message_id: str | None = None) -> Pause | None:
        """복용 약을 등록하고 처방약과 다시 대조한다 (REQ-WEB-09 → REQ-AG-05). meds는 검색으로 고른 [{item_code, name}].

        대조 결과(A.6)는 result()의 consult_result에 있다. 새 항목에 위험(danger)이 있으면 HITL ③(risk)에서 멈춘다 → Pause.
        응답은 {"action": "acknowledge"}. 없는 대화 · 멈춰 있는 대화 · 잘못된 목록이면 ValueError (그래프는 그대로).
        """
        meds = read_meds(meds)
        if not self._snapshot(conv_id).values:
            raise ValueError(f"없는 대화입니다: {conv_id} (복용 약은 열린 대화에 등록합니다)")
        pause = self.pending(conv_id)
        if pause is not None:
            raise ValueError(f"약사 응답을 기다리는 대화입니다: {conv_id} ({pause.hitl_point}). 응답한 뒤에 등록하세요")
        run_message(self.graph, conv_id, {"message_id": message_id}, meds=meds)
        return self._after_run(conv_id)

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
            read_reply(reply, DispensingOrder.model_validate(pause.payload["order"]))
        elif hitl_point == HITL_TYPE:
            read_choice(reply, pause.payload["about"])
        elif hitl_point == HITL_INSPECTION:
            read_inspection_reply(reply)
        elif hitl_point == HITL_RISK:
            read_risk_reply(reply)

    def search_drugs(self, name: str) -> list[dict[str, Any]]:
        """복용 약 수동 입력의 약 검색 (REQ-WEB-09 · F-WEB-10, 기능분석서 '복용 약 검색은 F-AG-08 도구 재사용').

        [{item_code, name, company, source}] 사전 → 의약품 API 순서, 같은 약은 한 번. source는 dictionary · api.
        MCP 서버가 있으면 서버의 search_drug를, 없으면 같은 함수(drug_match.search_drug_result)를 직접 부른다.
        API가 실패하면 사전 결과만 돌려준다 (서비스는 멈추지 않는다). 이름이 비면 ValueError.
        """
        name = " ".join(str(name or "").split())
        if not name:
            raise ValueError("찾을 약 이름이 비었습니다")
        try:
            found = (self.tools.call("search_drug", name=name) if self.tools is not None
                     else search_drug_result(name, self.dictionary, self.drug_api))
        except (McpToolError, DrugApiError, ConfigError):
            found = search_drug_result(name, self.dictionary, None)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in ("dictionary", "api"):
            for row in found.get(source) or []:
                if row["item_code"] not in seen:
                    seen.add(row["item_code"])
                    out.append({"item_code": row["item_code"], "name": row["name"], "company": row.get("company"),
                                "source": source})
        return out

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
        keys = ("conversation_id", "prescription_id", "message_id", "status", "flow", "route", "intents", "order", "plan",
                "inspection", "current_meds", "consult_result", "notice", "error_code", "proposals", "memo", "notes")
        return {key: values.get(key) for key in keys}

    def handle_text(self, rx_id: str, text: str) -> dict[str, Any]:
        """이미 열린 처방에 글만 들어왔을 때 (명세서 2.2절 복합 입력). 그래프는 돌리지 않는다.

        - 승인 대기 중: 수정 요청 · 복용 약 정보는 제안(proposals)으로. 웹이 승인 카드에 보이고, 약사가 확인한 것만 고쳐 반영한다.
        - 승인이 끝난 뒤: 수정 요청은 '조제 내용 재검토 필요'(review_needed). 복용 약 정보는 재대조 후보 제안.
        - 질문은 답하지 않는다 (11/16 상담 제외): 메모로 남기고 notice로 알린다. 메모나 쓸 데 없는 글도 memo로 남긴다.
        - 뜻을 못 가르면 ask=True와 고를 수 있는 의도(choices). 웹이 되묻는 카드를 띄운다.
        없는 처방이면 ValueError (새 대화는 start로 연다).
        """
        stage = self.stage(rx_id)
        if stage is None:
            raise ValueError(f"열린 처방이 없습니다: {rx_id} (새 대화는 start로 시작하세요)")
        if self._snapshot(rx_id).values.get("prescription_id") is None:
            raise ValueError(f"처방이 없는 대화입니다: {rx_id} (상담 대화의 글은 send로 보내세요)")
        intents = classify(text)
        waiting = stage == HITL_APPROVAL
        usable = [EDIT_REQUEST, PATIENT_INFO] if waiting else [PATIENT_INFO]
        review = (not waiting) and EDIT_REQUEST in intents and stage in (STAGE_AFTER_PLAN, "approved", "planned")
        answer: dict[str, Any] = {
            "intents": intents,
            "proposals": make_proposals(text, [i for i in intents if i in usable], self.dictionary),
            "review_needed": review,
            "memo": None,
            "ask": not intents,
            "choices": list(TEXT_CHOICES) if not intents else [],
            "notice": REVIEW_NEEDED_NOTICE if review else (NO_CONSULT_NOTICE if QUESTION in intents else None),
        }
        if intents and not (answer["proposals"] or review):
            answer["memo"] = text.strip()                    # 메모, 또는 지금 쓸 데가 없는 글
        elif NOTE in intents and not answer["memo"]:
            answer["memo"] = text.strip()
        return answer

    def _after_run(self, rx_id: str) -> Pause | None:
        pause = self.pending(rx_id)
        if pause is not None:
            return pause
        values = self._snapshot(rx_id).values
        if values.get("status") == "failed" or values.get("error_code"):     # 트레이 검수 실패는 상태가 planned 그대로다
            raise AgentRunError(values.get("error_code"), values.get("notice") or "처방 처리에 실패했습니다.")
        return None
