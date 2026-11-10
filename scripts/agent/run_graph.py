"""상위 그래프에 샘플 처방전을 넣어 끝까지 돌려 보고 마지막 상태를 보여 준다 (7단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/run_graph.py                       # rx_001 한 장
    python scripts/agent/run_graph.py rx_005                # 지정한 한 장
    python scripts/agent/run_graph.py --all                 # 샘플 5장 전부
    python scripts/agent/run_graph.py --no-api              # .env에 DRUG_API_KEY가 있어도 식약처 API를 쓰지 않는다 (약품 사전만)

슈퍼바이저의 분기를 보려면 사진 종류를 지정한다. --no-ocr 은 OCR 없이 사진 종류만 보낸다 (트레이 사진처럼).
    python scripts/agent/run_graph.py --kind med_bag --no-ocr   # 약 포장 -> 복용 약 재대조 자리
    python scripts/agent/run_graph.py --kind tray --no-ocr      # 트레이 -> 검수 자리
    python scripts/agent/run_graph.py --no-ocr                  # 종류를 알 수 없음 -> 되묻는 자리
    python scripts/agent/run_graph.py --kind med_bag            # 처방전 OCR인데 약 포장이라고 함 -> 되묻는 자리

pytest가 아니다. agent/resources/samples/cv_a1 의 가상 처방전을 그래프에 넣는다.
규칙 경로가 성공하면 OpenAI는 부르지 않는다. 화면에는 환자 정보를 찍지 않는다 (약품 표만).
langgraph가 설치되어 있어야 한다 (pip install langgraph).
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
from agent.errors import ConfigError  # noqa: E402
from agent.graph import build_graph, run_message  # noqa: E402
from agent.schema import DispensingOrder  # noqa: E402

SAMPLES = config.RESOURCES_DIR / "samples" / "cv_a1"


def sample_ids() -> list[str]:
    return sorted(p.stem for p in SAMPLES.glob("rx_*.json"))


def load(sample_id: str) -> dict[str, Any]:
    return json.loads((SAMPLES / f"{sample_id}.json").read_text(encoding="utf-8"))


def show(sample_id: str, state: dict[str, Any], out: Callable[[str], None]) -> None:
    out(f"[{sample_id}] 상태: {state['status']} | 흐름: {state['flow']}")
    if state.get("notice"):
        out(f"    안내: {state['notice']}")
    if state.get("error_code"):
        out(f"    오류 코드: {state['error_code']}")
    for error in state.get("errors", []):
        out(f"    오류: {error}")
    if state.get("order") is None:
        return
    order = DispensingOrder.model_validate(state["order"])
    out(f"    경로: {order.source} | 교부번호: {order.issue_no}")
    for i, d in enumerate(order.items, start=1):
        mark = "확인" if d.needs_pharmacist_check else "    "
        out(
            f"    {i}. [{mark}] {d.name} | 코드 {d.printed_code} -> 품목 {d.item_code or '-'} ({d.match_method})"
            f" | {d.count_per_dose}개 x {len(d.times)}회 x {d.days}일 | {', '.join(d.times)}"
        )
        for reason in d.check_reasons:
            out(f"         - {reason}")
    for warning in order.warnings:
        out(f"    경고: {warning}")


def main(argv: list[str] | None = None, *, graph=None, out: Callable[[str], None] = print) -> int:
    ids = sample_ids()
    parser = argparse.ArgumentParser(description="상위 그래프 실행 (가상 처방전)")
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="처방전 번호 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="샘플 전부")
    parser.add_argument("--no-api", action="store_true", help="의약품 API를 쓰지 않는다 (약품 사전만)")
    parser.add_argument("--kind", help="사진 종류를 지정한다: prescription · med_bag · tray (그 밖의 값은 알 수 없는 종류)")
    parser.add_argument("--no-ocr", action="store_true", help="OCR 결과 없이 보낸다 (트레이 사진처럼)")
    args = parser.parse_args(argv)

    if graph is None:
        try:
            graph = build_graph(**({"api": None} if args.no_api else {}))
        except ConfigError as e:
            out(f"설정 문제: {e}")
            return 1
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] != "langgraph":
                raise
            out("langgraph가 설치되어 있지 않습니다. pip install langgraph 로 설치한 뒤 다시 돌려 주세요.")
            return 1
        if args.no_api:
            out("약품 API: 사용 안 함 (--no-api, 약품 사전 8종만 씁니다)")
        elif config.drug_api_key():
            out("약품 API: 사용 (사전에 없는 약은 식약처 API에서 찾습니다)")
        else:
            out("약품 API: 사용 안 함 (.env에 DRUG_API_KEY가 없어 약품 사전 8종만 씁니다)")

    chosen = ids if args.all else [args.sample]
    failed, reached = 0, 0
    for sample_id in chosen:
        try:
            state = run_message(graph, f"RUN-{sample_id}", None if args.no_ocr else load(sample_id), args.kind)
        except ConfigError as e:
            out(f"설정 문제: {e}")
            return 1
        show(sample_id, state, out)
        failed += state["status"] == "failed"
        reached += state["status"] == "awaiting_approval"
    if args.all:
        out(f"전체 {len(chosen)}장 중 {reached}장이 약사 승인 대기까지 왔습니다.")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
