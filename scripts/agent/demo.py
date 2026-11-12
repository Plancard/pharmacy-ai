"""뼈대 그래프 동작 확인 (11단계, 팀 체크포인트 '뼈대 그래프' 11/13) — 손으로 돌려 보는 스크립트.

    python scripts/agent/demo.py            # 시나리오 10개를 차례로 돌리고 기대와 같은지 보여 준다
    python scripts/agent/demo.py --quiet    # 시나리오마다 결과 한 줄만

pytest가 아니다. 웹이 부르는 입구(agent/api.py)를 그대로 써서, 웹 화면에서 일어날 일을 글로 보여 준다.
시연용 가상 처방전(agent/resources/samples/rx8_a1)을 쓰고, 멈춘 곳은 메모리에 저장한다. 의약품 API와 LLM은 부르지 않는다
(시연용 처방전은 규칙 경로로 읽힌다). 복용 약 등록 · 재대조 · 상담은 아직 자리만 있어 '준비 중'으로 나온다.
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
from agent.proposal import PROPOSAL_LABELS  # noqa: E402
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
                        api=None, checkpointer=memory_checkpointer(), inspect=stub_tray)
    return AgentApi(graph, dictionary)


def approve(pause: Pause) -> dict[str, Any]:
    return {"action": "approve", "order": pause.payload["order"]}


def show_pause(pause: Pause | None, out: Out) -> None:
    if pause is None:
        out("      (멈추지 않고 끝남)")
        return
    out(f"      멈춤: {pause.hitl_point} - {pause.payload['notice']}")
    if pause.hitl_point == "type_select":
        out("      고를 수 있는 것: " + " · ".join(pause.payload["labels"][c] for c in pause.payload["choices"]))
        return
    if pause.hitl_point == "inspection":
        for line in pause.payload["cells"]:
            out(f"      칸 {line}")
        out("      고를 수 있는 것: 확인 완료(confirm) · 재조제(redo)")
        return
    out("      조제지시서: " + " · ".join(f"{d['name']} {d['count_per_dose']}개 x {len(d['times'])}회 x {d['days']}일"
                                    for d in pause.payload["order"]["items"]))
    for p in pause.payload["proposals"]:
        out(f"      제안({PROPOSAL_LABELS[p['kind']]}): {p['text']}  ← 약사가 확인해야 반영")
    if pause.payload.get("memo"):
        out(f"      메모: {pause.payload['memo']}")


def show_result(api: AgentApi, rx_id: str, out: Out) -> dict[str, Any]:
    result = api.result(rx_id)
    out(f"      단계: {api.stage(rx_id)} | 상태: {result['status']} | 순서: {' → '.join(result['route']) or '-'}")
    if result["plan"]:
        names = {d["item_code"]: d["name"] for d in result["order"]["items"]}
        for line in plan_lines(Plan.model_validate(result["plan"]), names):
            out(f"      계획 {line}")
    for note in result["notes"]:
        out(f"      자리: {note}")
    if result["memo"]:
        out(f"      메모: {result['memo']}")
    return result


# ------------------------------------------------------------ 시나리오. 각각 (제목, 함수). 함수는 기대와 같으면 True


def s1_prescription_with_edit_request(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_01 타이레놀 · 게보린) + 글 '게보린 빼 주세요'")
    pause = api.start("D1", load("rx8_01"), "prescription", "게보린 빼 주세요")
    show_pause(pause, out)
    ok = pause.hitl_point == "rx_approval" and len(pause.payload["order"]["items"]) == 2 and len(pause.payload["proposals"]) == 1
    out("   약사: 제안은 두고 그대로 승인")
    ok &= api.resume("D1", "rx_approval", approve(pause)) is None
    result = show_result(api, "D1", out)
    return ok and result["status"] == "planned" and len(result["plan"]["plan"]) == 9


def s2_prescription_with_question(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_02 트리스펜 · 아르펜) + 글 '같이 먹어도 되나요?'")
    pause = api.start("D2", load("rx8_02"), "prescription", "같이 먹어도 되나요?")
    show_pause(pause, out)
    out("   약사: 승인 → 조제 계획 다음에 상담이 이어진다")
    api.resume("D2", "rx_approval", approve(pause))
    result = show_result(api, "D2", out)
    return result["route"] == ["prescription", "consult"] and result["status"] == "planned" and len(result["notes"]) == 1


def s3_edit_after_a_plan_that_cannot_be_made(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_07, 한 칸에 3알이 들어가는 처방)")
    pause = api.start("D3", load("rx8_07"))
    out("   약사: 그대로 승인")
    back = api.resume("D3", "rx_approval", approve(pause))
    show_pause(back, out)
    order = copy.deepcopy(back.payload["order"])
    order["items"] = order["items"][:2]
    out("   약사: 뮤코다인을 빼고 고치기(edit) → 다시 대조한 지시서를 승인")
    again = api.resume("D3", "rx_approval", {"action": "edit", "order": order})
    show_pause(again, out)
    api.resume("D3", "rx_approval", approve(again))
    result = show_result(api, "D3", out)
    return back is not None and "칸당" in back.payload["notice"] and result["status"] == "planned"


def s4_unclear_photo_kind(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_04)을 '약봉투'라고 올렸다 (OCR은 처방전으로 읽음)")
    pause = api.start("D4", load("rx8_04"), "med_bag")
    show_pause(pause, out)
    out("   약사: '처방전'을 고른다")
    again = api.resume("D4", "type_select", {"choice": "prescription"})
    show_pause(again, out)
    return pause.hitl_point == "type_select" and again is not None and again.hitl_point == "rx_approval"


def s5_tray_photo_with_text(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전(rx8_01)을 승인해 조제 계획까지 만든 뒤, 같은 대화에 트레이 사진 + 글 '게보린 빼 주세요'")
    pause = api.start("D5", load("rx8_01"))
    api.resume("D5", "rx_approval", approve(pause))
    planned = api.result("D5")
    out("   (같은 대화의 두 번째 메시지: 이번 메시지 칸만 넣는다. 트레이와 온 글은 메모로만)")
    again = api.send("D5", None, "tray", "게보린 빼 주세요", message_id="m2", image_id="tray-ok")
    show_pause(again, out)
    result = show_result(api, "D5", out)
    kept = result["order"] == planned["order"] and result["plan"] == planned["plan"]
    out(f"      승인된 조제지시서 · 조제 계획: {'그대로' if kept else '바뀜'} | 검수: {result['notice']}")
    return (again is None and kept and result["memo"] == "게보린 빼 주세요" and result["proposals"] == []
            and result["status"] == "inspected" and result["inspection"]["summary"]["ok"] == 12)


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
    out("   약사: 다시 찍은 사진: 저신뢰 칸 1개 → 약사가 보고 확인 완료(confirm)")
    pause = api.send("D10", None, "tray", image_id="tray-review")
    show_pause(pause, out)
    api.resume("D10", "inspection", {"action": "confirm"})
    result = show_result(api, "D10", out)
    return (failed is not None and failed.code == "E_TRAY_NOT_FOUND" and stage == "await_tray"
            and pause is not None and "신뢰도 낮음" in pause.payload["cells"][0] and result["status"] == "inspected")


def s6_text_alone(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방 없이 새 대화에 글만 '타이레놀 먹고 있는데 괜찮나요?' (상담 대화)")
    pause = api.start("D6", None, None, "타이레놀 먹고 있는데 괜찮나요?")
    show_pause(pause, out)
    result = show_result(api, "D6", out)
    out(f"      처방 ID: {result['prescription_id']} | 의도: {', '.join(result['intents'])}")
    return result["route"] == ["consult"] and result["prescription_id"] is None and result["proposals"] == []


def s8_consultation_becomes_a_prescription(api: AgentApi, out: Out) -> bool:
    out("   약사: 상담 대화(D8)에서 글로 묻다가, 같은 대화에 처방전 사진(rx8_02)을 올린다")
    api.start("D8", None, None, "이부프로펜 하루에 몇 번 먹어요?")
    pause = api.send("D8", load("rx8_02"), "prescription", prescription_id="D8")
    show_pause(pause, out)
    out("      같은 ID로 처방 대화가 되었다 (앞의 상담 기록이 이어진다)")
    return pause is not None and pause.hitl_point == "rx_approval" and pause.payload["order"]["prescription_id"] == "D8"


def s7_text_for_an_open_prescription(api: AgentApi, out: Out) -> bool:
    out("   약사: 처방전 사진(rx8_03)을 올려 승인 대기 중에, 글만 '뮤코다인 빼 주세요'")
    pause = api.start("D7", load("rx8_03"))
    got = api.handle_text("D7", "뮤코다인 빼 주세요")
    for p in got["proposals"]:
        out(f"      제안({PROPOSAL_LABELS[p['kind']]}): {p['text']} (글에 보인 약: {', '.join(d['name'] for d in p['drugs'])})")
    out("      그래프는 그대로 승인 대기에 멈춰 있다 (자동 반영 없음)")
    return api.pending("D7") == pause and [p["kind"] for p in got["proposals"]] == ["edit_request"]


SCENARIOS: list[tuple[str, Callable[[AgentApi, Out], bool]]] = [
    ("처방전 사진 + 수정 요청 글 → 제안으로만, 승인 → 조제 계획", s1_prescription_with_edit_request),
    ("처방전 사진 + 질문 → 처방 흐름 다음에 상담 (정해진 순서)", s2_prescription_with_question),
    ("계획을 못 만들면 승인 대기로 돌아와 고치기", s3_edit_after_a_plan_that_cannot_be_made),
    ("사진 종류가 애매하면 되묻기 (type_select)", s4_unclear_photo_kind),
    ("승인 뒤 같은 대화에 트레이 사진 + 글 → 전 칸 일치, 글은 메모, 조제지시서 · 계획은 그대로", s5_tray_photo_with_text),
    ("처방 없는 대화에 글만 → 상담", s6_text_alone),
    ("열린 처방에 글만 → 승인 카드의 제안", s7_text_for_an_open_prescription),
    ("상담 대화에 처방전 사진 → 같은 ID로 처방 대화", s8_consultation_becomes_a_prescription),
    ("트레이 불일치 → 검수 확인(HITL ②)에서 재조제 → 다시 찍어 전 칸 일치", s9_tray_mismatch_redo_then_ok),
    ("트레이를 못 찾음 → 오류 코드, 계획은 그대로 → 다시 찍어 확인 완료", s10_tray_not_found_then_confirm),
]


def main(argv: list[str] | None = None, *, out: Out = print, api_factory: Callable[[], AgentApi] = make_api) -> int:
    parser = argparse.ArgumentParser(description="뼈대 그래프 동작 확인 (시나리오 10개)")
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
