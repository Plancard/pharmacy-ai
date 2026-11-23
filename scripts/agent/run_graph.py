"""그래프에 샘플 처방전을 넣어 돌려 보고 결과를 보여 준다 (7 ~ 11단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/run_graph.py                       # rx_001 한 장: 승인 대기에서 멈춘 모습
    python scripts/agent/run_graph.py rx8_01 --approve      # 승인 대기에서 약사가 그대로 승인 → 조제 계획 확정
    python scripts/agent/run_graph.py --all                 # 샘플 전부 (CV 처방전 5장 + 시연용 처방전 8장)
    python scripts/agent/run_graph.py --no-api              # .env에 DRUG_API_KEY가 있어도 식약처 API를 쓰지 않는다 (약품 사전만)
    python scripts/agent/run_graph.py --subgraph --all      # 처방 서브그래프만 단독으로 (슈퍼바이저를 거치지 않는다). 결과는 상위 그래프와 같다
    python scripts/agent/run_graph.py rx8_01 --db           # 멈춘 곳을 .env의 DATABASE_URL(PostgreSQL)에 저장한다 (기본은 메모리)
    python scripts/agent/run_graph.py --all --mcp           # 약품 매칭을 MCP 도구로 부른다 (python -m agent.mcp_server를 띄운다)
    python scripts/agent/run_graph.py --all --mcp-url http://127.0.0.1:8765/mcp   # 따로 띄운 MCP 서버에 붙는다

슈퍼바이저의 분기를 보려면 사진 종류를 지정한다 (명세서 v0.8: 처방전 · 트레이만). --no-ocr 은 OCR 없이 사진 종류만 보낸다.
    python scripts/agent/run_graph.py --kind tray --no-ocr      # 조제 계획 전의 트레이 사진 -> 안내만
    python scripts/agent/run_graph.py --no-ocr                  # 종류도 OCR도 없음 -> 안내만

pytest가 아니다. 샘플은 agent/resources/samples의 cv_a1(rx_001 ~ rx_005, CV가 보낸 OCR)과 rx8_a1(rx8_01 ~ rx8_08, 대상 8종 시연용)이다.
규칙 경로가 성공하면 OpenAI는 부르지 않는다. 화면에는 환자 정보를 찍지 않는다 (약품 표와 위험 신호, 조제 계획만).
처방 흐름은 조제 계획까지 만든 뒤 승인 대기(HITL ①)에서 멈춘다. --approve를 주면 화면의 조제지시서를 그대로 승인해 계획을 확정한다
(약을 정하지 못한 줄이 있거나 계획을 만들지 못했으면 승인할 수 없다고 보여 주고 넘어간다).
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
from agent.mcp_client import McpToolError, http, stdio  # noqa: E402
from agent.graph import build_graph, build_prescription_graph, resume_run, run_message, run_prescription  # noqa: E402
from agent.nodes import APPROVE, read_reply  # noqa: E402
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
    status = f"승인 대기({paused['hitl_point']})" if paused else state["status"]
    route = " → ".join(state.get("route") or []) or "-"
    out(f"[{sample_id}] 상태: {status} | 흐름: {state.get('flow') or '-'} | 순서: {route}")
    notice = paused["notice"] if paused else state.get("notice")
    if notice:
        out(f"    안내: {notice}")
    if state.get("error_code") and not paused:
        out(f"    오류 코드: {state['error_code']}")
    if not paused:                         # 멈춘 동안의 이유는 안내(승인 카드)에 있다. 상위 그래프는 서브그래프의 기록을 아직 받지 않았다
        for error in state.get("errors", []):
            out(f"    오류: {error}")
    for note in state.get("notes", []):
        out(f"    기록: {note}")
    order_data = paused["order"] if paused else state.get("order")
    if order_data is None:
        return
    order = DispensingOrder.model_validate(order_data)
    out(f"    경로: {order.source} | 교부번호: {order.issue_no}")
    for i, d in enumerate(order.items, start=1):
        mark = "확인" if d.needs_pharmacist_check else "    "
        out(
            f"    {i}. [{mark}] {d.name}{f' (처방전: {d.printed_name})' if d.printed_name else ''} | 코드 {d.printed_code} -> 품목 {d.item_code or '-'} ({d.match_method})"
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
    plan = paused.get("plan") if paused else state.get("plan")     # v0.8: 계획은 승인 전에 만들어 승인 카드에 보인다
    if plan is not None:
        names = {d.item_code: d.name for d in order.items if d.item_code}
        out("    조제 계획:" + (" (승인 카드에 보임)" if paused else ""))
        for line in plan_lines(Plan.model_validate(plan), names):
            out(f"      {line}")
    elif paused:
        code = paused.get("error_code")
        out(f"    조제 계획: 만들 수 없음{f' ({code})' if code else ''}. 고친 뒤(edit) 승인할 수 있습니다")


def main(argv: list[str] | None = None, *, graph=None, out: Callable[[str], None] = print) -> int:
    ids = sample_ids()
    parser = argparse.ArgumentParser(description="그래프 실행 (가상 처방전)")
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="처방전 번호 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="샘플 전부")
    parser.add_argument("--no-api", action="store_true", help="의약품 API를 쓰지 않는다 (약품 사전만)")
    parser.add_argument("--kind", help="사진 종류를 지정한다: prescription · tray (그 밖의 값은 받지 않는 종류)")
    parser.add_argument("--no-ocr", action="store_true", help="OCR 결과 없이 보낸다 (트레이 사진처럼)")
    parser.add_argument("--subgraph", action="store_true", help="처방 서브그래프만 단독으로 돌린다 (슈퍼바이저를 거치지 않는다)")
    parser.add_argument("--approve", action="store_true", help="승인 대기에서 화면의 조제지시서를 그대로 승인해 이어 간다")
    parser.add_argument("--db", action="store_true", help="멈춘 곳을 DATABASE_URL(PostgreSQL)에 저장한다 (기본은 메모리)")
    parser.add_argument("--mcp", action="store_true", help="약품 매칭을 MCP 도구(match_drug)로 부른다. 서버를 하위 프로세스로 띄운다")
    parser.add_argument("--mcp-url", help="따로 띄운 MCP 서버(python -m agent.mcp_server --http)의 주소로 붙는다. --mcp를 겸한다")
    args = parser.parse_args(argv)
    if args.subgraph and args.kind is not None:
        parser.error("--subgraph는 슈퍼바이저를 거치지 않으므로 사진 종류(--kind)를 줄 수 없습니다")

    tools = None
    if graph is None and (args.mcp or args.mcp_url):
        try:
            tools = http(args.mcp_url) if args.mcp_url else stdio(no_api=args.no_api)
        except McpToolError as e:
            out(f"MCP 서버 문제: {e}")
            out("mcp 패키지가 없다면 pip install -r agent/requirements.txt 로 설치해 주세요.")
            return 1
        where = args.mcp_url or "python -m agent.mcp_server"
        out(f"약품 매칭: MCP 도구로 부름 ({where}, 도구 {len(tools.tool_names())}개: {', '.join(tools.tool_names())})")
    try:
        return _run(args, ids, graph, tools, out)
    finally:
        if tools is not None:
            tools.close()


def _run(args: argparse.Namespace, ids: list[str], graph, tools, out: Callable[[str], None]) -> int:
    if graph is None:
        try:
            checkpointer = default_checkpointer() if args.db else memory_checkpointer()
            build = build_prescription_graph if args.subgraph else build_graph
            options = ({"api": None} if args.no_api else {}) | ({"tools": tools} if tools is not None else {})
            graph = build(checkpointer=checkpointer, **options)
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
                state = run_message(graph, rx_id, {"kind": args.kind}, ocr, prescription_id=rx_id)
            show(sample_id, state, out)
            reached += paused_value(state) is not None
            if args.approve and paused_value(state) is not None:
                shown = paused_value(state)
                reply = {"action": APPROVE, "order": shown["order"]}
                try:                                   # 웹(api.check_reply)처럼 이어 가기 전에 거른다
                    read_reply(reply, DispensingOrder.model_validate(shown["order"]), plan_ready=shown.get("plan") is not None)
                except ValueError as e:
                    out(f"    -> 그대로 승인할 수 없습니다: {e}")
                else:
                    out("    -> 약사가 화면의 조제지시서를 그대로 승인합니다")
                    state = resume_run(graph, rx_id, reply)
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
