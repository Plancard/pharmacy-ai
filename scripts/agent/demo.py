"""뼈대 그래프 동작 확인 (11단계, 팀 체크포인트 '뼈대 그래프' 11/13) — 손으로 돌려 보는 스크립트.

    python scripts/agent/demo.py            # 시나리오를 차례로 돌리고 기대와 같은지 보여 준다
    python scripts/agent/demo.py --quiet    # 시나리오마다 결과 한 줄만

pytest가 아니다. 웹이 부르는 입구(agent/api.py)를 그대로 써서, 웹 화면에서 일어날 일을 글로 보여 준다.
시연용 가상 처방전(agent/resources/samples/rx8_a1)을 쓰고, 멈춘 곳은 메모리에 저장한다. 의약품 API와 LLM은 부르지 않는다
(시연용 처방전은 규칙 경로로 읽힌다). 명세서의 처방 단일 흐름만 돈다 (글 입력 · 상담 · 복용 약 재대조는 범위 밖).
트레이 검수는 CV 가짜 모듈(cv/stub.py)로 한다. 사진 ID 끝이 결과를 정한다: tray-ok(전 칸 일치) · tray-mismatch(1칸 불일치) ·
tray-no_tray(트레이 못 찾음) 등 (cv.stub.SCENARIOS).
기대와 다른 시나리오가 있으면 1로 끝난다. langgraph가 설치되어 있어야 한다 (pip install -r agent/requirements.txt).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import config  # noqa: E402
from agent.api import AgentApi, AgentRunError, Pause  # noqa: E402
from agent.checkpoint import memory_checkpointer  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.pipeline import structure_prescription  # noqa: E402
from agent.plan import plan_lines  # noqa: E402
from agent.rag import load_index  # noqa: E402
from agent.schema import Plan  # noqa: E402

RX8 = config.RESOURCES_DIR / "samples" / "rx8_a1"
Out = Callable[[str], None]


def load(name: str) -> dict[str, Any]:
    return json.loads((RX8 / f"{name}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback):
    raise RuntimeError("시연용 처방전은 규칙 경로로 읽혀야 합니다 (LLM을 부르지 않는다)")


def stub_tray(image_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    """CV 가짜 검수. 사진 ID 'tray-<시나리오>'의 시나리오대로 (tray-ok, tray-mismatch, tray-no_tray …)."""
    from cv import stub

    return stub.inspect(image_id, plan, scenario=image_id.rsplit("-", 1)[-1])


def make_api() -> AgentApi:
    dictionary = load_dictionary()
    graph = build_graph(dictionary=dictionary, structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=no_llm),
                        api=None, checkpointer=memory_checkpointer(), inspect=stub_tray,
                        rag=load_index(config.RAG_DIR), embed=None)
    return AgentApi(graph)


def approve(pause: Pause) -> dict[str, Any]:
    return {"action": "approve", "order": pause.payload["order"]}


def show_pause(pause: Pause | None, out: Out) -> None:
    if pause is None:
        out("      (멈추지 않고 끝남)")
        return
    out(f"      멈춤: {pause.hitl_point} - {pause.payload['notice']}")
    if pause.hitl_point == "inspection":
        for line in pause.payload["cells"]:
            out(f"      칸 {line}")
        out("      고를 수 있는 것: 확인 완료(confirm) · 재조제(redo)")
        return
    out("      조제지시서: " + " · ".join(f"{d['name']} {d['count_per_dose']}개 x {len(d['times'])}회 x {d['days']}일"
                                    for d in pause.payload["order"]["items"]))
    risk = pause.payload["order"].get("risk_check") or {}
    for f in risk.get("findings", []):
        out(f"      위험 신호({f['level']}): {f['message']}")
        for e in f.get("evidence", []):
            out(f"        근거: {e['title']} ({e['source']})")
    plan = pause.payload.get("plan")
    if plan is not None:                                            # 조제 계획은 승인 전에 승인 카드에 보인다
        names = {d["item_code"]: d["name"] for d in pause.payload["order"]["items"] if d.get("item_code")}
        for line in plan_lines(Plan.model_validate(plan), names):
            out(f"      계획(승인 카드) {line}")
    else:
        out(f"      계획: 만들 수 없음 ({pause.payload.get('error_code') or '코드 없음'}) - 고치기(edit)만 할 수 있음")


def show_result(api: AgentApi, rx_id: str, out: Out) -> dict[str, Any]:
    result = api.result(rx_id)
    out(f"      단계: {api.stage(rx_id)} | 상태: {result['status']} | 순서: {' → '.join(result['route']) or '-'}")
    if result["plan"]:
        names = {d["item_code"]: d["name"] for d in result["order"]["items"]}
        for line in plan_lines(Plan.model_validate(result["plan"]), names):
            out(f"      계획 {line}")
    for note in result["notes"]:
        out(f"      기록: {note}")
    return result


# ------------------------------------------------------------ 시나리오. 각각 (제목, 함수). 함수는 기대와 같으면 True


def s1_prescription_approve_then_plan(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_01 타이레놀 · 게보린)")
    pause = api.start("D1", load("rx8_01"))
    show_pause(pause, out)
    ok = pause is not None and pause.hitl_point == "rx_approval" and len(pause.payload["order"]["items"]) == 2
    ok &= pause is not None and pause.payload["plan"] is not None
    out("   약사: 승인 카드의 조제지시서 · 조제 계획을 보고 그대로 승인 → 계획 확정")
    ok &= api.resume("D1", "rx_approval", approve(pause)) is None
    result = show_result(api, "D1", out)
    return ok and result["status"] == "planned" and result["plan"] == pause.payload["plan"]


def s2_duplicate_warning_on_approval(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_02 트리스펜 · 아르펜, 이부프로펜 중복)")
    pause = api.start("D2", load("rx8_02"))
    show_pause(pause, out)
    out("   약사: 경고를 보고 승인 → 조제 계획")
    api.resume("D2", "rx_approval", approve(pause))
    result = show_result(api, "D2", out)
    risk = pause.payload["order"]["risk_check"] if pause else None
    return (risk is not None and risk["level"] == "warning" and result["route"] == ["prescription"]
            and result["status"] == "planned")


def s3_edit_after_a_plan_that_cannot_be_made(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_18, 투약 4일 - 트레이는 3일까지)")
    pause = api.start("D3", load("rx8_18"))
    show_pause(pause, out)
    out("   약사: 그대로 승인 → 거절됨")
    try:
        api.resume("D3", "rx_approval", approve(pause))
        refused = None
    except ValueError as e:
        refused = e
        out(f"      거절: {e}")
    order = copy.deepcopy(pause.payload["order"])
    for item in order["items"]:
        item["days"] = 3
    out("   약사: 투약일수를 3일로 고치기(edit) → 다시 매칭 · 대조 · 계획")
    again = api.resume("D3", "rx_approval", {"action": "edit", "order": order})
    show_pause(again, out)
    api.resume("D3", "rx_approval", approve(again))
    result = show_result(api, "D3", out)
    return (pause.payload["plan"] is None and pause.payload["error_code"] == "E_PLAN_RANGE" and refused is not None
            and again is not None and again.payload["plan"] is not None and result["status"] == "planned")


def s4_second_prescription_photo(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전(rx8_04)을 승인한 뒤, 같은 처방에 처방전 사진을 또 올렸다")
    api.resume("D4", "rx_approval", approve(api.start("D4", load("rx8_04"))))
    before = api.result("D4")
    again = api.send("D4", load("rx8_05"), "prescription")
    result = show_result(api, "D4", out)
    out(f"      안내: {result['notice']}")
    return (again is None and result["order"] == before["order"] and result["plan"] == before["plan"]
            and result["status"] == "planned" and "새 처방" in result["notice"])


def s5_tray_photo_all_ok(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전(rx8_01)을 승인해 조제 계획까지 만든 뒤, 같은 처방에 트레이 사진")
    pause = api.start("D5", load("rx8_01"))
    api.resume("D5", "rx_approval", approve(pause))
    planned = api.result("D5")
    out("   (같은 처방의 두 번째 메시지: 이번 메시지 칸만 넣는다)")
    again = api.send("D5", None, "tray", image_id="tray-ok")
    show_pause(again, out)
    result = show_result(api, "D5", out)
    kept = result["order"] == planned["order"] and result["plan"] == planned["plan"]
    out(f"      승인된 조제지시서 · 조제 계획: {'그대로' if kept else '바뀜'} | 검수: {result['notice']}")
    return (again is None and kept and result["status"] == "inspected" and result["inspection"]["summary"]["ok"] == 12)


def s9_tray_mismatch_redo_then_ok(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전(rx8_02)을 승인한 뒤 트레이 사진: 1칸에 1알이 빠짐")
    api.resume("D9", "rx_approval", approve(api.start("D9", load("rx8_02"))))
    pause = api.send("D9", None, "tray", image_id="tray-mismatch")
    show_pause(pause, out)
    out("   약사: 재조제(redo) → 다시 채워 찍은 사진")
    back = api.resume("D9", "inspection", {"action": "redo"})
    stage = api.stage("D9")
    out(f"      단계: {stage} (다시 찍은 사진을 기다림)")
    done = api.send("D9", None, "tray", image_id="tray-ok")
    result = show_result(api, "D9", out)
    return (pause is not None and pause.hitl_point == "inspection" and len(pause.payload["cells"]) == 1
            and back is None and stage == "await_tray" and done is None and result["status"] == "inspected")


def s10_tray_not_found_then_confirm(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전(rx8_03)을 승인한 뒤 트레이가 안 보이는 사진")
    api.resume("D10", "rx_approval", approve(api.start("D10", load("rx8_03"))))
    try:
        api.send("D10", None, "tray", image_id="tray-no_tray")
        failed = None
    except AgentRunError as e:
        failed = e
        out(f"      오류: {e.code} - {e}")
    stage = api.stage("D10")
    out(f"      단계: {stage} (조제 계획은 그대로, 다시 찍으면 된다)")
    out("   약사: 다시 찍은 사진이 180° 뒤집혀 찍힘 → 검수 확인(HITL ②)에서 다시 찍기(redo)")
    pause = api.send("D10", None, "tray", image_id="tray-rotated")
    show_pause(pause, out)
    api.resume("D10", "inspection", {"action": "redo"})
    out("   약사: 바로 놓고 다시 찍은 사진")
    done = api.send("D10", None, "tray", image_id="tray-ok")
    result = show_result(api, "D10", out)
    return (failed is not None and failed.code == "E_TRAY_NOT_FOUND" and stage == "await_tray"
            and pause is not None and "뒤집혀" in pause.payload["cells"][0] and done is None
            and result["status"] == "inspected")


SCENARIOS: list[tuple[str, Callable[[AgentApi, Out], bool]]] = [
    ("처방전 사진 → 승인 카드에 조제 계획 → 승인으로 확정", s1_prescription_approve_then_plan),
    ("성분 중복(이부프로펜) → 승인 화면에 경고와 근거, 승인 → 조제 계획 확정", s2_duplicate_warning_on_approval),
    ("4일 처방 → 승인 카드에 계획 없음(E_PLAN_RANGE), 승인 거절 → 고치기", s3_edit_after_a_plan_that_cannot_be_made),
    ("이미 처방전이 있는 처방에 처방전 사진 → 안내만, 조제지시서 · 계획은 그대로", s4_second_prescription_photo),
    ("승인 뒤 같은 처방에 트레이 사진 → 전 칸 일치, 조제지시서 · 계획은 그대로", s5_tray_photo_all_ok),
    ("트레이 불일치 → 검수 확인(HITL ②)에서 재조제 → 다시 찍어 전 칸 일치", s9_tray_mismatch_redo_then_ok),
    ("트레이를 못 찾음 → 오류 코드, 계획은 그대로 → 뒤집힌 사진은 다시 찍기 → 전 칸 일치", s10_tray_not_found_then_confirm),
]


def main(argv: list[str] | None = None, *, out: Out = print, api_factory: Callable[[], AgentApi] = make_api) -> int:
    parser = argparse.ArgumentParser(description="뼈대 그래프 동작 확인 (명세서 처방 흐름)")
    parser.add_argument("--quiet", action="store_true", help="시나리오마다 결과 한 줄만")
    args = parser.parse_args(argv)
    try:
        api = api_factory()
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] != "langgraph":
            raise
        out("langgraph가 설치되어 있지 않습니다. pip install -r agent/requirements.txt 로 설치한 뒤 다시 돌려 주세요.")
        return 1
    passed = 0
    for number, (title, scenario) in enumerate(SCENARIOS, start=1):
        lines: list[str] = []
        ok = scenario(api, lines.append)
        passed += ok
        out(f"[{number}] {title}: {'기대대로' if ok else '기대와 다름'}")
        if not args.quiet or not ok:
            for line in lines:
                out(line)
    out(f"전체 {len(SCENARIOS)}개 중 {passed}개가 기대대로 돌았습니다.")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
