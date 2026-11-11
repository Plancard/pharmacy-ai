"""그래프에 샘플 처방전을 넣어 돌려 보고 결과를 보여 준다 (7 ~ 11단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/run_graph.py                       # rx_001 한 장: 승인 대기에서 멈춘 모습
    python scripts/agent/run_graph.py rx8_01 --approve      # 승인 대기에서 약사가 그대로 승인 → 조제 계획까지
    python scripts/agent/run_graph.py --all                 # 샘플 전부 (CV 처방전 5장 + 시연용 처방전 8장)
    python scripts/agent/run_graph.py --no-api              # .env에 DRUG_API_KEY가 있어도 식약처 API를 쓰지 않는다 (약품 사전만)
    python scripts/agent/run_graph.py --subgraph --all      # 처방 서브그래프만 단독으로 (슈퍼바이저를 거치지 않는다). 결과는 상위 그래프와 같다
    python scripts/agent/run_graph.py rx8_01 --db           # 멈춘 곳을 .env의 DATABASE_URL(PostgreSQL)에 저장한다 (기본은 메모리)

사진과 함께 글을 보내려면 --text를 준다. 글만 보내려면 --no-ocr와 함께 쓴다.
    python scripts/agent/run_graph.py rx8_01 --text "게보린 빼 주세요"      # 수정 요청 -> 승인 카드의 제안으로만
    python scripts/agent/run_graph.py --no-ocr --text "타이레놀 같이 먹어도 되나요?"   # 글만: 상담 자리
    python scripts/agent/run_graph.py --no-ocr --choose tray      # 사진 종류를 되묻는 곳에서 '트레이'를 고른다

슈퍼바이저의 분기를 보려면 사진 종류를 지정한다. --no-ocr 은 OCR 없이 사진 종류만 보낸다 (트레이 사진처럼).
    python scripts/agent/run_graph.py --kind med_bag --no-ocr   # 약 포장 -> 복용 약 재대조 자리
    python scripts/agent/run_graph.py --kind tray --no-ocr      # 트레이 -> 검수 자리
    python scripts/agent/run_graph.py --no-ocr                  # 종류를 알 수 없음 -> 되묻는 자리
    python scripts/agent/run_graph.py --kind med_bag            # 처방전 OCR인데 약 포장이라고 함 -> 되묻는 자리

pytest가 아니다. 샘플은 agent/resources/samples의 cv_a1(rx_001 ~ rx_005, CV가 보낸 OCR)과 rx8_a1(rx8_01 ~ rx8_08, 대상 8종 시연용)이다.
규칙 경로가 성공하면 OpenAI는 부르지 않는다. 화면에는 환자 정보를 찍지 않는다 (약품 표와 위험 신호, 조제 계획만).
처방 흐름은 승인 대기(HITL ①)에서 멈춘다. --approve를 주면 화면의 조제지시서를 그대로 승인해 이어 간다.
langgraph가 설치되어 있어야 한다 (pip install -r agent/requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/run_graph.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.checkpoint import default_checkpointer, describe, memory_checkpointer  # noqa: E402
from agent.errors import ConfigError  # noqa: E402
from agent.graph import build_graph, build_prescription_graph, resume_run, run_message, run_prescription  # noqa: E402
from agent.nodes import APPROVE, HITL_TYPE  # noqa: E402
from agent.proposal import PROPOSAL_LABELS  # noqa: E402
from agent.plan import plan_lines  # noqa: E402
from agent.risk import LEVEL_LABELS  # noqa: E402
from agent.schema import DispensingOrder, Plan  # noqa: E402

SAMPLE_DIRS = [config.RESOURCES_DIR / "samples" / "cv_a1", config.RESOURCES_DIR / "samples" / "rx8_a1"]


def sample_ids() -> list[str]:
    return [p.stem for folder in SAMPLE_DIRS for p in sorted(folder.glob("rx*.json"))]


def load(sample_id: str) -> dict[str, Any]:
    path = next(folder / f"{sample_id}.json" for folder in SAMPLE_DIRS if (folder / f"{sample_id}.json").exists())
    return json.loads(path.read_text(encoding="utf-8"))


def paused_value(state: dict[str, Any]) -> dict[str, Any] | None:
    """그래프가 멈췄으면 멈춘 내용 (interrupt 값), 아니면 None."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value if interrupts else None


def show(sample_id: str, state: dict[str, Any], out: Callable[[str], None]) -> None:
    paused = paused_value(state)
    status = f"멈춤({paused['hitl_point']})" if paused and paused["hitl_point"] == HITL_TYPE else (
        f"승인 대기({paused['hitl_point']})" if paused else state["status"])
    route = " → ".join(state.get("route") or []) or "-"
    out(f"[{sample_id}] 상태: {status} | 흐름: {state.get('flow') or '-'} | 순서: {route}")
    notice = paused["notice"] if paused else state.get("notice")
    if notice:
        out(f"    안내: {notice}")
    if state.get("error_code") and not paused:
        out(f"    오류 코드: {state['error_code']}")
    for error in state.get("errors", []):
        out(f"    오류: {error}")
    for note in state.get("notes", []):
        out(f"    자리: {note}")
    if paused and paused["hitl_point"] == HITL_TYPE:
        out("    고를 수 있는 것: " + " · ".join(f"{c}({paused['labels'][c]})" for c in paused["choices"]))
        return
    proposals = paused["proposals"] if paused else state.get("proposals") or []
    for p in proposals:
        drugs = ", ".join(d["name"] for d in p["drugs"]) or "-"
        out(f"    제안: [{PROPOSAL_LABELS[p['kind']]}] {p['text']} (글에 보인 약: {drugs}). 약사가 확인해야 반영됩니다")
    memo = paused.get("memo") if paused else state.get("memo")
    if memo:
        out(f"    메모: {memo}")
    order_data = paused["order"] if paused else state.get("order")
    if order_data is None:
        return
    order = DispensingOrder.model_validate(order_data)
    out(f"    경로: {order.source} | 교부번호: {order.issue_no}")
    for i, d in enumerate(order.items, start=1):
        mark = "확인" if d.needs_pharmacist_check else "    "
        out(
            f"    {i}. [{mark}] {d.name} | 코드 {d.printed_code} -> 품목 {d.item_code or '-'} ({d.match_method})"
            f" | {d.count_per_dose}개 x {len(d.times)}회 x {d.days}일 | {', '.join(d.times)}"
        )
        for reason in d.check_reasons:
            out(f"         - {reason}")
    risk = order.risk_check
    if risk is not None:
        out(f"    위험 신호: {LEVEL_LABELS[risk.level]} (처방약끼리 대조, {len(risk.findings)}건)")
        for finding in risk.findings:
            out(f"         - [{LEVEL_LABELS[finding.level]}] {finding.message}")
    for warning in order.warnings:
        out(f"    경고: {warning}")
    if state.get("plan") is not None:
        names = {d.item_code: d.name for d in order.items if d.item_code}
        out("    조제 계획:")
        for line in plan_lines(Plan.model_validate(state["plan"]), names):
            out(f"      {line}")
    elif paused:
        out("    조제 계획: 약사가 승인한 뒤에 만듭니다")


def main(argv: list[str] | None = None, *, graph=None, out: Callable[[str], None] = print) -> int:
    ids = sample_ids()
    parser = argparse.ArgumentParser(description="그래프 실행 (가상 처방전)")
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="처방전 번호 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="샘플 전부")
    parser.add_argument("--no-api", action="store_true", help="의약품 API를 쓰지 않는다 (약품 사전만)")
    parser.add_argument("--kind", help="사진 종류를 지정한다: prescription · med_bag · tray (그 밖의 값은 알 수 없는 종류)")
    parser.add_argument("--no-ocr", action="store_true", help="OCR 결과 없이 보낸다 (트레이 사진처럼)")
    parser.add_argument("--subgraph", action="store_true", help="처방 서브그래프만 단독으로 돌린다 (슈퍼바이저를 거치지 않는다)")
    parser.add_argument("--approve", action="store_true", help="승인 대기에서 화면의 조제지시서를 그대로 승인해 이어 간다")
    parser.add_argument("--db", action="store_true", help="멈춘 곳을 DATABASE_URL(PostgreSQL)에 저장한다 (기본은 메모리)")
    parser.add_argument("--text", help="사진과 함께(또는 --no-ocr로 글만) 보낼 글")
    parser.add_argument("--choose", help="되묻는 곳(type_select)에서 고를 값 (사진: prescription · med_bag · tray, 글: current_med · question · memo)")
    args = parser.parse_args(argv)
    if args.subgraph and args.kind is not None:
        parser.error("--subgraph는 슈퍼바이저를 거치지 않으므로 사진 종류(--kind)를 줄 수 없습니다")

    if graph is None:
        try:
            checkpointer = default_checkpointer() if args.db else memory_checkpointer()
            build = build_prescription_graph if args.subgraph else build_graph
            graph = build(checkpointer=checkpointer, **({"api": None} if args.no_api else {}))
        except ConfigError as e:
            out(f"설정 문제: {e}")
            return 1
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] != "langgraph":
                raise
            out("langgraph가 설치되어 있지 않습니다. pip install -r agent/requirements.txt 로 설치한 뒤 다시 돌려 주세요.")
            return 1
        if args.no_api:
            out("약품 API: 사용 안 함 (--no-api, 약품 사전 8종만 씁니다)")
        elif config.drug_api_key():
            out("약품 API: 사용 (사전에 없는 약은 식약처 API에서 찾습니다)")
        else:
            out("약품 API: 사용 안 함 (.env에 DRUG_API_KEY가 없어 약품 사전 8종만 씁니다)")
        out(f"멈춘 곳 저장: {describe(checkpointer)}")

    chosen = ids if args.all else [args.sample]
    failed, reached = 0, 0
    for sample_id in chosen:
        rx_id = f"RUN-{sample_id}"
        try:
            ocr = None if args.no_ocr else load(sample_id)
            if args.subgraph:
                state = run_prescription(graph, rx_id, ocr)
            else:
                state = run_message(graph, rx_id, ocr, args.kind, args.text)
            show(sample_id, state, out)
            if args.choose and paused_value(state) is not None and paused_value(state)["hitl_point"] == HITL_TYPE:
                out(f"    -> 약사가 '{args.choose}'을(를) 고릅니다")
                choices = paused_value(state)["choices"]
                if args.choose not in choices:
                    out(f"    고를 수 있는 값이 아닙니다: {args.choose} (가능: {', '.join(choices)})")
                    return 1
                state = resume_run(graph, rx_id, {"choice": args.choose})
                show(sample_id, state, out)
            reached += paused_value(state) is not None and paused_value(state)["hitl_point"] != HITL_TYPE
            if args.approve and paused_value(state) is not None and paused_value(state)["hitl_point"] != HITL_TYPE:
                order = paused_value(state)["order"]
                out("    -> 약사가 화면의 조제지시서를 그대로 승인합니다")
                state = resume_run(graph, rx_id, {"action": APPROVE, "order": order})
                show(sample_id, state, out)
        except ConfigError as e:
            out(f"설정 문제: {e}")
            return 1
        failed += state.get("status") == "failed" and paused_value(state) is None
    if args.all:
        out(f"전체 {len(chosen)}장 중 {reached}장이 약사 승인 대기까지 왔습니다.")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
