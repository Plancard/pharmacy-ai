"""진입점 및 실행 테스트 (11/6 LangGraph 뼈대).

    python main.py            # rx_001 한 장
    python main.py rx_005     # 지정한 한 장
    python main.py --all      # 실제 양식 샘플 5장 전부

data/agent/cv_samples의 가상 처방전을 그래프에 넣어 끝까지 돌리고, 마지막 상태를 보여 준다.
규칙 경로가 성공하면 OpenAI는 부르지 않는다. 화면에는 환자 정보를 찍지 않는다 (약품 표만).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from agent.errors import ConfigError
from agent.schema import DispensingOrder

DATA = Path(__file__).resolve().parent / "data" / "agent"


def sample_ids() -> list[str]:
    return sorted(p.stem for p in (DATA / "cv_samples").glob("rx_*.json"))


def show(sample_id: str, state: dict[str, Any], out: Callable[[str], None]) -> None:
    out(f"[{sample_id}] 상태: {state['status']}")
    if state.get("notice"):
        out(f"    안내: {state['notice']}")
    for error in state.get("errors", []):
        out(f"    오류: {error}")
    if state.get("order") is None:
        return
    order = DispensingOrder.model_validate(state["order"])
    out(f"    경로: {order.source} | 교부번호: {order.issue_no}")
    for i, d in enumerate(order.drugs, start=1):
        mark = "확인" if d.needs_pharmacist_check else "    "
        out(
            f"    {i}. [{mark}] {d.name} | 코드 {d.printed_code} -> 품목 {d.item_code or '-'} ({d.match_method})"
            f" | {d.dose_per_time}개 x {d.times_per_day}회 x {d.days}일 | {', '.join(d.time_slots)}"
        )
        for reason in d.check_reasons:
            out(f"         - {reason}")


def main(argv: list[str] | None = None, *, graph=None, out: Callable[[str], None] = print) -> int:
    ids = sample_ids()
    parser = argparse.ArgumentParser(description="LangGraph 뼈대 실행 (가상 처방전)")
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="처방전 번호 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="샘플 전부")
    args = parser.parse_args(argv)

    if graph is None:
        try:
            from agent_project.my_agent_project.app.graph import build_graph
            graph = build_graph()
        except ImportError as e:
            out(f"설정 문제: langgraph를 불러오지 못했습니다 ({e.name}). `pip install langgraph`로 설치하세요.")
            return 1
        except ConfigError as e:
            out(f"설정 문제: {e}")
            return 1

    from agent_project.my_agent_project.app.graph import run_prescription

    targets = ids if args.all else [args.sample]
    results = []
    for sample_id in targets:
        ocr = json.loads((DATA / "cv_samples" / f"{sample_id}.json").read_text(encoding="utf-8"))
        state = run_prescription(graph, f"P-{sample_id}", ocr)
        show(sample_id, state, out)
        results.append(state["status"] == "awaiting_approval")
    if len(targets) > 1:
        out(f"전체 {len(targets)}장 중 {sum(results)}장이 약사 승인 대기까지 왔습니다")
    return 0 if all(results) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
