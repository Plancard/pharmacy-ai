"""뼈대 그래프 동작 확인 (11단계, 팀 체크포인트 '뼈대 그래프' 11/13) — 손으로 돌려 보는 스크립트.

    python scripts/agent/demo.py            # 시나리오 7개를 차례로 돌리고 기대와 같은지 보여 준다
    python scripts/agent/demo.py --quiet    # 시나리오마다 결과 한 줄만

pytest가 아니다. 웹이 부르는 입구(agent/api.py)를 그대로 써서, 웹 화면에서 일어날 일을 글로 보여 준다.
시연용 가상 처방전(agent/resources/samples/rx8_a1)을 쓰고, 멈춘 곳은 메모리에 저장한다. 의약품 API와 LLM은 부르지 않는다
(시연용 처방전은 규칙 경로로 읽힌다). 복용 약 등록 · 재대조 · 트레이 검수 · 상담은 아직 자리만 있어 '준비 중'으로 나온다.
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
from agent.api import AgentApi, Pause  # noqa: E402
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


def make_api() -> AgentApi:
    dictionary = load_dictionary()
    graph = build_graph(dictionary=dictionary, structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=no_llm),
                        api=None, checkpointer=memory_checkpointer())
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
    out("   약사: 트레이 사진 + 글 '게보린 빼 주세요' (트레이와 온 글은 메모로만)")
    pause = api.start("D5", None, "tray", "게보린 빼 주세요")
    show_pause(pause, out)
    result = show_result(api, "D5", out)
    return pause is None and result["memo"] == "게보린 빼 주세요" and result["proposals"] == []


def s6_text_alone(api: AgentApi, out: Out) -> bool:
    out("   약사: 글만 '타이레놀 먹고 있는데 괜찮나요?'")
    pause = api.start("D6", None, None, "타이레놀 먹고 있는데 괜찮나요?")
    show_pause(pause, out)
    result = show_result(api, "D6", out)
    for p in result["proposals"]:
        out(f"      제안({PROPOSAL_LABELS[p['kind']]}): {', '.join(d['name'] for d in p['drugs'])}")
    return result["route"] == ["register_meds", "consult"]


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
    ("트레이 사진 + 글 → 글은 메모로만", s5_tray_photo_with_text),
    ("글만 → 복용 약 등록 · 상담 자리", s6_text_alone),
    ("열린 처방에 글만 → 승인 카드의 제안", s7_text_for_an_open_prescription),
]


def main(argv: list[str] | None = None, *, out: Out = print, api_factory: Callable[[], AgentApi] = make_api) -> int:
    parser = argparse.ArgumentParser(description="뼈대 그래프 동작 확인 (시나리오 7개)")
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
