"""웹 ↔ Agent 연결부. 웹은 이 파일의 메서드만 부르고, Agent 내부(LangGraph)는 모른다.

    start(rx_id, image_path)          처방전 처리 시작. HITL에서 멈추면 Pause를 돌려준다
    pending(rx_id)                    지금 멈춰 있는 지점과 화면에 보여 줄 내용 (없으면 None)
    resume(rx_id, hitl_point, reply)  약사 응답으로 이어 간다. 다시 멈추면 Pause
    approved_order(rx_id)             승인된 조제지시서 (트레이 대기 카드에 보여 줌)
    approved_plan(rx_id)              그 조제지시서의 조제 계획 A.4 (조제 기록 저장에 씀, 없으면 None)
    failed_lines(rx_id)               구조화 실패 때 처방전에서 강조할 줄 {ocr_size, lines: [{line_id, bbox}]} (Q6, 모르면 None)
    tray(rx_id, image_id, image_path) 트레이 사진 검수 (11/18). 불일치·확인 필요 칸이 있으면 HITL ②(inspection)에서 멈춘다
    inspection(rx_id)                 마지막 검수 결과 A.3 (없으면 None)
    tray_error(rx_id)                 트레이 대기로 돌아온 이유 {code, notice} (트레이 못 찾음 등, 없으면 None)

start·resume·tray는 셋 중 하나를 돌려준다.
    Pause  HITL에서 멈춤 → 처방 상태 waiting_approval
    Wait   이번 실행이 끝남 → 처방 상태 Wait.status
           waiting_tray: 약사가 트레이 사진을 올리기를 기다림 (승인 끝, 또는 HITL ② 재조제)
           done: 트레이 검수 끝 (전 칸 일치, 또는 HITL ② 확인 완료)
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
                                #               "ocr_size": {width, height} | None,
                                #               "changed_items": [고친 뒤 다시 검수한 줄 번호(0부터)] (고치기 전엔 없거나 빈 목록)}
                                #   plan이 None이면(3일 초과·약을 정하지 못한 줄) 승인은 막고 edit만 받는다 (11/17)


WAIT_STATUSES = ("waiting_tray", "done")


@dataclass(frozen=True)
class Wait:
    """그래프가 멈추지 않고 이번 실행을 끝냈다. 다음 사진을 기다리거나(waiting_tray) 처방 흐름이 끝났다(done)."""
    status: str                 # DB 처방 상태. WAIT_STATUSES 중 하나

    def __post_init__(self):
        if self.status not in WAIT_STATUSES:
            raise ValueError(f"Wait.status={self.status} (가능: {', '.join(WAIT_STATUSES)})")


def init_app(app: Flask) -> None:
    backend = app.config["AGENT_BACKEND"]
    if backend == "fake":
        app.extensions["agent"] = FakeGateway(Path(app.config["VAR_DIR"]) / "fake_agent",
                                              app.config["FAKE_AGENT_SCENARIO"])
    elif backend == "langgraph":
        from web.langgraph_gateway import from_settings   # agent · langgraph 패키지는 이때만 읽는다

        app.extensions["agent"] = from_settings()
    else:
        raise RuntimeError(f"AGENT_BACKEND={backend} (가능: fake · langgraph)")


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
_SLOT_WARNING = "3번째 약품: 복용 시점이 처방전에 없어 하루 3회 기준으로 배정했습니다."   # agent/validate.py의 문구와 같은 꼴
SCENARIOS = ("warning", "none", "danger", "flagged", "fail", "range", "llm_fail")
_TIMES = ("morning", "noon", "evening")
_TIME_LABELS = {"morning": "아침", "noon": "점심", "evening": "저녁", "bedtime": "취침 전"}
_STATUS_LABELS = {"ok": "일치", "mismatch": "불일치", "need_review": "확인 필요"}
_REASON_LABELS = {"boundary": "칸 경계에 걸침", "overlap": "알약 겹침", "rotated": "트레이가 뒤집혀 찍힘"}
INSPECTION_ACTIONS = ("confirm", "redo")   # HITL ② 응답 (기능분석서 F-WEB-09)


def inspection_payload(a3: dict[str, Any], plan: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    """HITL ② 카드 내용. 진짜 Agent의 agent.inspection.inspection_payload와 같은 칸 (가짜 연결부용, agent 패키지 없이)."""
    def content(counts: dict[str, int]) -> str:
        shown = [f"{names.get(code, code)} {n}" for code, n in counts.items() if n]
        return " · ".join(shown) if shown else "없음"

    lines = []
    for c in a3["cells"]:
        if c["status"] == "ok":
            continue
        line = (f"{c['day']}일 {_TIME_LABELS[c['time']]} {_STATUS_LABELS[c['status']]}: "
                f"예상 {content(c['expected'])} / 실제 {content(c['actual'])}")
        if c["reasons"]:
            line += f" ({', '.join(_REASON_LABELS.get(r, r) for r in c['reasons'])})"
        lines.append(line)
    s = a3["summary"]
    if any("rotated" in c["reasons"] for c in a3["cells"]):
        notice = "트레이가 180° 뒤집혀 찍힌 것 같습니다. 아침 줄이 위로 오게 다시 찍어 주세요."
    else:
        notice = (f"트레이 검수: 불일치 {s['mismatch']}칸, 확인 필요 {s['need_review']}칸. "
                  "약사가 확인해 주세요 (확인 완료 또는 재조제).")
    return {"inspection": a3, "plan": plan, "names": names, "cells": lines, "notice": notice,
            "choices": list(INSPECTION_ACTIONS)}


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
        order = {"prescription_id": rx_id, "items": items, "warnings": [],
                 "risk_check": {"scope": "rx_only", "level": level, "findings": findings}}
        if self.scenario == "warning":
            order["warnings"] = [copy.deepcopy(_SLOT_WARNING)]   # 화면 확인용: 복용 시점 누락 경고
        return order

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
        if self.scenario == "llm_fail":   # 화면 확인용(Q6): LLM이 첫 약품 줄을 구조화하지 못함(E_LLM) → 그 줄을 강조하고 [다시 시도]
            ids = set(order["items"][0]["source_line_ids"])
            lines = [{"line_id": l["line_id"], "bbox": l["bbox"]} for l in a1["lines"] if l["line_id"] in ids and l.get("bbox")]
            size = {"width": a1["width"], "height": a1["height"]} if a1.get("width") and a1.get("height") else None
            self._save(rx_id, {"pause": None, "approved_order": None,
                               "failed_lines": {"ocr_size": size, "lines": lines} if size and lines else None})
            raise AgentError("E_LLM", "AI가 처방 줄을 구조화하지 못했습니다.")
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
            before = {(it["name"], it["count_per_dose"], tuple(it["times"]), it["days"]) for it in old["order"]["items"]}
            changed = [i for i, it in enumerate(order["items"])    # 가짜: 고치기 전에 없던 줄 = 바뀐 줄 (Q7)
                       if (it["name"], it["count_per_dose"], tuple(it["times"]), it["days"]) not in before]
            pause = Pause("rx_approval", {**old, "order": order, "plan": plan, "error_code": plan_error, "notice": why,
                                          "changed_items": changed})
            data.update(pause={"hitl_point": pause.hitl_point, "payload": pause.payload})
            self._save(rx_id, data)
            return pause
        if hitl_point == "rx_approval":
            # 실제 그래프: 승인된 조제지시서 → 조제 계획(A.4) → state.status = 'planned'로 끝난다
            data.update(pause=None, approved_order=reply["order"])
            self._save(rx_id, data)
            return Wait("waiting_tray")
        # HITL ② 검수 확인: confirm → 완료(inspected), redo → 재조제(planned, 다시 찍은 트레이 사진을 기다림)
        data.update(pause=None)
        self._save(rx_id, data)
        return Wait("done") if reply.get("action") == "confirm" else Wait("waiting_tray")

    def check_reply(self, rx_id: str, hitl_point: str, reply: dict[str, Any]) -> None:
        """응답이 지금 멈춘 지점에 맞는지 본다 (가짜는 지점과 HITL ②의 action만 확인. 응답 모양 검사는 진짜 Agent가 한다)."""
        data = self._load(rx_id)
        if not data or not data.get("pause") or data["pause"]["hitl_point"] != hitl_point:
            raise ValueError(f"멈춰 있는 지점이 없거나 다릅니다: {rx_id} ({hitl_point})")
        if hitl_point == "inspection" and (not isinstance(reply, dict) or reply.get("action") not in INSPECTION_ACTIONS):
            raise ValueError("검수 확인 응답의 action은 confirm(확인 완료) · redo(재조제) 중 하나여야 합니다.")

    def tray(self, rx_id: str, image_id: str, image_path: Path) -> Pause | Wait | None:
        """트레이 사진 검수의 가짜 버전: cv/stub.py inspect(시나리오는 .env CV_STUB_SCENARIO, 기본 ok)로 A.3을 만든다.

        진짜 그래프와 같이 검수하지 못하면(E_TRAY_NOT_FOUND 등) 코드가 있는 예외를 올리고 트레이 대기에 머문다.
        """
        from cv import stub

        data = self._load(rx_id)
        order = (data or {}).get("approved_order")
        if not order:
            raise AgentError("E_CV_INFER", "조제 계획이 아직 없습니다. 조제지시서를 승인한 뒤에 트레이 사진을 올려 주세요.")
        plan, _, why = self._plan(rx_id, order)
        if plan is None:
            raise AgentError("E_CV_INFER", why or "조제 계획이 없습니다.")
        try:
            a3 = stub.inspect(Path(image_path), plan)
        except stub.CVError as e:
            data["tray_error"] = {"code": e.code, "notice": str(e)}
            self._save(rx_id, data)
            raise
        a3["image_id"] = image_id                    # stub은 파일 이름을 쓴다. 진짜 Agent처럼 웹의 이미지 ID로 맞춘다
        data.update(inspection=a3, tray_error=None)
        if a3["summary"]["hitl_required"]:
            names = {it["item_code"]: it["name"] for it in order["items"] if it.get("item_code")}
            pause = Pause("inspection", inspection_payload(a3, plan, names))
            data["pause"] = {"hitl_point": pause.hitl_point, "payload": pause.payload}
            self._save(rx_id, data)
            return pause
        self._save(rx_id, data)
        return Wait("done")                          # 전 칸 일치: HITL ②를 건너뛰고 바로 완료 (E-1)

    def inspection(self, rx_id: str) -> dict[str, Any] | None:
        data = self._load(rx_id)
        return data.get("inspection") if data else None

    def tray_error(self, rx_id: str) -> dict[str, Any] | None:
        data = self._load(rx_id)
        return data.get("tray_error") if data else None

    def approved_order(self, rx_id: str) -> dict[str, Any] | None:
        data = self._load(rx_id)
        return data.get("approved_order") if data else None

    def failed_lines(self, rx_id: str) -> dict[str, Any] | None:
        data = self._load(rx_id)
        return data.get("failed_lines") if data else None

    def approved_plan(self, rx_id: str) -> dict[str, Any] | None:
        order = self.approved_order(rx_id)
        return self._plan(rx_id, order)[0] if order else None
