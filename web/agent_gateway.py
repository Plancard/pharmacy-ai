"""웹 ↔ Agent 연결부. 웹은 이 파일의 메서드만 부르고, Agent 내부(LangGraph)는 모른다.

    start(rx_id, image_path)          처방전 처리 시작. HITL에서 멈추면 Pause를 돌려준다
    pending(rx_id)                    지금 멈춰 있는 지점과 화면에 보여 줄 내용 (없으면 None)
    resume(rx_id, hitl_point, reply)  약사 응답으로 이어 간다. 다시 멈추면 Pause
    approved_order(rx_id)             승인된 조제지시서 (트레이 대기 카드에 보여 줌)

start·resume은 셋 중 하나를 돌려준다.
    Pause  HITL에서 멈춤 → 처방 상태 waiting_approval
    Wait   이번 실행이 끝났고 약사의 다음 사진을 기다림 → 처방 상태 Wait.status (지금은 waiting_tray뿐)
    None   아직 처리 중 (상태를 바꾸지 않음)

실패는 code 속성(명세서 A.7 오류 코드)이 있는 예외로 올린다 (cv.stub.CVError, AgentError).

지금은 FakeGateway를 쓴다: cv/stub.py로 A.1을 만들고, 정해 둔 A.5 샘플로 HITL ①에서 멈춘다.
멈춘 내용은 VAR_DIR/fake_agent/<처방 ID>.json에 두어 새로고침·재시작해도 남는다 (REQ-WEB-03).
Agent 그래프에 interrupt와 PostgresSaver가 붙으면 LangGraphGateway를 만들어 바꿔 끼운다.
  - start  → graph.invoke(initial_state, config={"configurable": {"thread_id": rx_id}})
  - pending→ graph.get_state(config).interrupts[0].value
  - resume → graph.invoke(Command(resume=reply), config)
  - 그래프가 멈추지 않고 끝나면 마지막 state.status로 Wait를 정한다: 'planned' → Wait('waiting_tray')
    (Agent 상태와 DB 처방 상태는 이름이 다르다. 대응표는 기능분석서 5.2절)
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, current_app

HITL_POINTS = ("rx_approval", "inspection")   # 11/17: type_select(종류 선택)·risk(HITL ③) 삭제


class AgentError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class Pause:
    hitl_point: str
    payload: dict[str, Any]     # rx_approval: {"order": A.2, "plan": A.4 | None, "error_code": str | None,
                                #               "notice": str | None, "ocr_lines": {line_id: {text, conf, bbox}},
                                #               "ocr_size": {width, height} | None}
                                #   plan이 None이면(3일 초과·약을 정하지 못한 줄) 승인은 막고 edit만 받는다 (11/17)


WAIT_STATUSES = ("waiting_tray",)


@dataclass(frozen=True)
class Wait:
    """그래프가 멈추지 않고 이번 실행을 끝냈고, 약사의 다음 입력(사진)을 기다린다."""
    status: str                 # DB 처방 상태. WAIT_STATUSES 중 하나

    def __post_init__(self):
        if self.status not in WAIT_STATUSES:
            raise ValueError(f"Wait.status={self.status} (가능: {', '.join(WAIT_STATUSES)})")


def init_app(app: Flask) -> None:
    backend = app.config["AGENT_BACKEND"]
    if backend == "fake":
        app.extensions["agent"] = FakeGateway(Path(app.config["VAR_DIR"]) / "fake_agent",
                                              app.config["FAKE_AGENT_SCENARIO"])
    else:
        raise RuntimeError(f"AGENT_BACKEND={backend}는 아직 없습니다. 지금은 fake만 씁니다.")


def gateway() -> "FakeGateway":
    return current_app.extensions["agent"]


# ------------------------------------------------------------ 가짜 연결부
# rx_004 샘플(cv/fixtures/a1/rx_004.json)의 약품 줄을 기준으로 만든 A.5. item_code는 명세서 2.6절.
_ITEMS = [
    {"item_code": "202106092", "name": "타이레놀정500mg", "count_per_dose": 1,
     "times": ["morning", "noon", "evening"], "days": 3, "source_line_ids": [58, 59, 60, 62]},
    {"item_code": "197900277", "name": "게보린정", "count_per_dose": 1,
     "times": ["morning", "noon", "evening"], "days": 3, "source_line_ids": [63, 64, 65, 67]},
    {"item_code": "200200099", "name": "프론크논캡슐", "count_per_dose": 1,
     "times": ["morning", "noon", "evening"], "days": 3, "source_line_ids": [68, 69, 71, 72],
     "needs_pharmacist_check": True, "check_reasons": ["OCR 신뢰도 낮음 (프론크논램술 → 프론크논캡슐)"]},
]
_UNMATCHED = {"item_code": None, "name": "알모란정", "count_per_dose": 1,
              "times": ["morning", "noon", "evening"], "days": 3, "source_line_ids": [73, 74, 75, 77],
              "needs_pharmacist_check": True, "check_reasons": ["대상 8종이 아닌 약품 (품목기준코드 없음)"]}
_DUP = {"type": "ingredient_dup", "item_codes": ["202106092", "197900277"], "level": "warning",
        "message": "타이레놀정500mg과 게보린정에 아세트아미노펜이 함께 들어 있습니다. 하루 총량을 확인하세요."}
_DANGER = {"type": "contraindication", "item_codes": ["202106092", "200200099"], "level": "danger",
           "message": "(가짜 데이터) 병용금기 조합입니다. 처방 의사에게 확인한 뒤 진행하세요."}
SCENARIOS = ("warning", "none", "danger", "flagged", "fail", "range")
_TIMES = ("morning", "noon", "evening")


class FakeGateway:
    def __init__(self, store: Path, scenario: str = "warning"):
        if scenario not in SCENARIOS:
            raise RuntimeError(f"FAKE_AGENT_SCENARIO={scenario} (가능: {', '.join(SCENARIOS)})")
        self.store = store
        self.scenario = scenario

    def _path(self, rx_id: str) -> Path:
        return self.store / f"{rx_id}.json"

    def _load(self, rx_id: str) -> dict[str, Any] | None:
        p = self._path(rx_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _save(self, rx_id: str, data: dict[str, Any]) -> None:
        self.store.mkdir(parents=True, exist_ok=True)
        tmp = self._path(rx_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(rx_id))

    def _order(self, rx_id: str) -> dict[str, Any]:
        items = copy.deepcopy(_ITEMS)
        findings: list[dict] = []
        if self.scenario in ("warning", "flagged"):
            findings.append(copy.deepcopy(_DUP))
        if self.scenario == "danger":
            findings += [copy.deepcopy(_DUP), copy.deepcopy(_DANGER)]
        if self.scenario == "flagged":
            items.append(copy.deepcopy(_UNMATCHED))
        order_rank = ["none", "caution", "warning", "danger"]
        level = max((f["level"] for f in findings), key=order_rank.index, default="none")
        return {"prescription_id": rx_id, "items": items,
                "risk_check": {"scope": "rx_only", "level": level, "findings": findings}}

    def _rematch(self, rx_id: str, order: dict[str, Any]) -> dict[str, Any]:
        """edit 뒤 약품 다시 찾기·위험 신호 다시 보기의 가짜 버전: 이름으로 샘플 약품을 찾고, 없는 약의 경고는 뺀다."""
        known = {it["name"]: it["item_code"] for it in _ITEMS}
        items = copy.deepcopy(order["items"])
        for it in items:
            if it.get("item_code") is None:
                if it["name"] in known:
                    it["item_code"] = known[it["name"]]
                else:
                    it.update(needs_pharmacist_check=True, check_reasons=["대상 8종이 아닌 약품 (품목기준코드 없음)"])
        present = {it["item_code"] for it in items if it.get("item_code")}
        risk = order.get("risk_check") or {"scope": "rx_only", "level": "none", "findings": []}
        findings = [f for f in risk.get("findings", []) if set(f["item_codes"]) <= present]
        rank = ["none", "caution", "warning", "danger"]
        level = max((f["level"] for f in findings), key=rank.index, default="none")
        return {**order, "items": items, "risk_check": {**risk, "level": level, "findings": findings}}

    def _plan(self, rx_id: str, order: dict[str, Any]) -> tuple[dict | None, str | None, str | None]:
        """조제 계획(A.4)과, 못 만들었을 때의 (error_code, 이유). 실제 Agent는 HITL ① 앞에서 계획을 만든다 (11/17)."""
        items = order["items"]
        if any(it["days"] > 3 for it in items):
            return None, "E_PLAN_RANGE", "복용 일수가 3일을 넘어 조제 계획을 만들지 못했습니다."
        if any(not it.get("item_code") for it in items):
            return None, None, "약을 정하지 못한 줄이 있어 조제 계획을 만들지 못했습니다."
        days = max(it["days"] for it in items)
        cells = []
        for d in range(1, days + 1):
            for t in _TIMES:
                its = [{"item_code": it["item_code"], "count": it["count_per_dose"]}
                       for it in items if d <= it["days"] and t in it["times"]]
                if its:
                    cells.append({"day": d, "time": t, "items": its})
        return {"prescription_id": rx_id, "days": days, "plan": cells}, None, None

    def start(self, rx_id: str, image_path: Path) -> Pause | Wait | None:
        from cv import stub   # CV 가짜 모듈. 실제 모듈이 나오면 cv.pipeline으로 바꾼다

        if self.scenario == "fail":
            raise stub.OcrEmpty(Path(image_path).name)
        a1 = stub.ocr(Path(image_path).with_name("rx_004.jpg"))   # 업로드 파일 이름과 무관하게 rx_004 샘플
        order = self._order(rx_id)
        if self.scenario == "range":
            for it in order["items"]:
                it["days"] = 4
        plan, plan_error, why = self._plan(rx_id, order)
        used = {i for it in order["items"] for i in it["source_line_ids"]}
        lines = {str(l["line_id"]): {"text": l["text"], "conf": l["conf"], "bbox": l.get("bbox")}
                 for l in a1["lines"] if l["line_id"] in used}
        size = {"width": a1["width"], "height": a1["height"]} if a1.get("width") and a1.get("height") else None
        pause = Pause("rx_approval", {"order": order, "plan": plan, "error_code": plan_error, "notice": why,
                                      "ocr_lines": lines, "ocr_size": size})
        self._save(rx_id, {"pause": {"hitl_point": pause.hitl_point, "payload": pause.payload},
                           "approved_order": None})
        return pause

    def pending(self, rx_id: str) -> Pause | None:
        data = self._load(rx_id)
        if not data or not data.get("pause"):
            return None
        return Pause(data["pause"]["hitl_point"], data["pause"]["payload"])

    def resume(self, rx_id: str, hitl_point: str, reply: dict[str, Any]) -> Pause | Wait | None:
        data = self._load(rx_id)
        if not data or not data.get("pause") or data["pause"]["hitl_point"] != hitl_point:
            raise AgentError("E_LLM", "멈춰 있는 지점이 없거나 다릅니다.")
        if hitl_point == "rx_approval" and reply.get("action") == "edit":
            # 실제 그래프: 매칭 → 처방약 대조 → 조제 계획을 다시 계산하고 다시 멈춘다 (잠정, 구조화는 다시 안 함)
            old = data["pause"]["payload"]
            order = self._rematch(rx_id, reply["order"])
            plan, plan_error, why = self._plan(rx_id, order)
            pause = Pause("rx_approval", {**old, "order": order, "plan": plan, "error_code": plan_error, "notice": why})
            data.update(pause={"hitl_point": pause.hitl_point, "payload": pause.payload})
            self._save(rx_id, data)
            return pause
        if hitl_point == "rx_approval":
            # 실제 그래프: 승인된 조제지시서 → 조제 계획(A.4) → state.status = 'planned'로 끝난다
            data.update(pause=None, approved_order=reply["order"])
            self._save(rx_id, data)
            return Wait("waiting_tray")
        self._save(rx_id, data)
        return None

    def approved_order(self, rx_id: str) -> dict[str, Any] | None:
        data = self._load(rx_id)
        return data.get("approved_order") if data else None
