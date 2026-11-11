"""팀 샘플 조제지시서(A.5)로 조제 계획(A.4)을 만들어 보여 준다 (9단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/show_plan.py                 # s01_combo_a 한 장
    python scripts/agent/show_plan.py s02_combo_b     # 지정한 한 장
    python scripts/agent/show_plan.py --all           # 샘플 5장 전부

pytest가 아니다. agent/resources/samples/a5_order 의 조제지시서는 매칭 · 대조까지 끝난 최종 정답이라, 약사가 승인한 지시서로 보고
조제 계획을 만든다 (실제 그래프에서는 승인 뒤에만 만든다). 같은 이름의 A.4 샘플(a4_plan)이 있으면 그것과 같은지도 본다.
계획을 만들 수 없는 지시서(약을 정하지 못한 줄 · 3일 초과 · 칸당 2알 초과)는 그 이유를 보여 준다. 이것은 실패가 아니라 그 지시서의 답이다.
A.4 샘플과 다를 때만 1로 끝난다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/show_plan.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.plan import PlanError, build_plan, plan_lines  # noqa: E402
from agent.schema import DispensingOrder, Plan  # noqa: E402

SAMPLES = config.RESOURCES_DIR / "samples"
ORDERS = SAMPLES / "a5_order"
PLANS = SAMPLES / "a4_plan"


def sample_ids() -> list[str]:
    return sorted(p.stem for p in ORDERS.glob("*.json"))


def load(folder: Path, sample_id: str) -> dict[str, Any]:
    return json.loads((folder / f"{sample_id}.json").read_text(encoding="utf-8"))


def show(sample_id: str, out: Callable[[str], None]) -> bool:
    """한 장을 보여 준다. A.4 샘플과 다르면 False."""
    order = DispensingOrder.model_validate(load(ORDERS, sample_id))
    out(f"[{sample_id}] 약품 {len(order.items)}줄: " + " · ".join(d.name for d in order.items))
    try:
        plan = build_plan(order)
    except PlanError as e:
        out(f"    계획을 만들 수 없음 ({e.code or '코드 없음'}): {e}")
        return True
    out(f"    조제 계획: {plan.days}일, 약이 든 칸 {len(plan.plan)}개")
    for line in plan_lines(plan, {d.item_code: d.name for d in order.items if d.item_code}):
        out(f"      {line}")
    if (PLANS / f"{sample_id}.json").exists():
        same = plan == Plan.model_validate(load(PLANS, sample_id))
        out(f"    A.4 샘플(a4_plan/{sample_id}.json)과 " + ("같습니다" if same else "다릅니다"))
        return same
    return True


def main(argv: list[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    ids = sample_ids()
    parser = argparse.ArgumentParser(description="샘플 조제지시서(A.5)로 조제 계획(A.4)을 만들어 본다")
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="샘플 이름 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="샘플 전부")
    args = parser.parse_args(argv)
    results = [show(sample_id, out) for sample_id in (ids if args.all else [args.sample])]
    return 0 if all(results) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
