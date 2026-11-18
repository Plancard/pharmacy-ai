"""칸별 조제 계획 (9단계, 명세서 v0.8 2.2절 5단계 · REQ-AG-04) — 조제지시서를 A.4 조제 계획으로 펼친다.

조제지시서의 약품 줄(1회 개수 · 복용 시점 · 일수)을 트레이 칸(일자 × 복용 시점)마다 무슨 약을 몇 알 담는지로 바꾼다.

- HITL ① 승인 전에 조제지시서와 함께 만들어 승인 카드에 보인다 (v0.8). 약사가 승인하면 그 계획을 그대로 확정하고,
  약사가 지시서를 고치면 고친 지시서로 다시 만든다 (graph.py · nodes.py가 지킨다).
- 약이 없는 칸은 적지 않는다 (A.4: 계획에 없는 칸은 빈 칸).
- 같은 약이 두 줄에 나오면 같은 칸에서 개수를 더한다.
- 칸 순서는 1일 아침부터 일자 순 · 시점 순(아침 → 점심 → 저녁 → 취침 전)이고, 칸 안의 약은 처방전에 적힌 순서다.

만들 수 없으면 PlanError를 낸다. 트레이 규격(명세서 2.7절: 4시점 x 3일)을 넘으면 E_PLAN_RANGE(명세서 A.7)이다.
칸당 알 수는 제한하지 않는다 (명세서 2.7절, 11/12 삭제).
- 투약 일수가 트레이(3일)를 넘는 줄: E_PLAN_RANGE. 약사에게 돌아가 일수를 나누거나 고친다.
- 약을 정하지 못한 줄(item_code 없음): 코드 없음. 승인 화면에서 약을 정해야 하는 일이다.
문제가 여럿이면 한 번에 모두 알린다. 트레이 범위 문제가 하나라도 있으면 코드는 E_PLAN_RANGE다.

판단 층이다. 파일도 네트워크도 쓰지 않는다.
"""
from __future__ import annotations

from agent.errors import E_PLAN_RANGE
from agent.schema import TIME_SLOTS, TRAY_DAYS, DispensingOrder, Plan, PlanCell, PlanItem

TIME_LABELS = {"morning": "아침", "noon": "점심", "evening": "저녁", "bedtime": "취침 전"}


class PlanError(Exception):
    """조제 계획을 만들 수 없음. 메시지는 화면에 그대로 보여도 되는 문장이다.

    code는 A.7 오류 코드다. 트레이 범위(일수)를 넘으면 E_PLAN_RANGE, 약을 정하지 못한 줄 때문이면 None.
    """

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


def _label(index: int, name: str) -> str:
    return f"{index}번째 약품({name})"


NOT_TARGET_PLAN = "조제 계획을 만들 수 없습니다. 대상 8종이 아닌 약이 있습니다"


def not_target_lines(order: DispensingOrder, dictionary) -> list[str]:
    """품목기준코드는 있지만 약품 사전(대상 8종)에 없는 줄 (의약품 API로 찾은 약 · 약사가 넣은 코드).

    CV가 알아보지 못하는 약이라 트레이로 조제 · 검수할 수 없다 → 조제 계획을 만들지 않는다 (명세서 v0.9 REQ-AG-03, 회의 Q4).
    그래프의 계획 노드와 MCP build_plan 도구가 같이 쓴다.
    """
    return [_label(i, d.name) for i, d in enumerate(order.items, start=1)
            if d.item_code is not None and dictionary.by_item_code(d.item_code) is None]


def build_plan(order: DispensingOrder) -> Plan:
    """조제지시서로 A.4 조제 계획을 만든다. 만들 수 없으면 PlanError."""
    unmatched = [_label(i, d.name) for i, d in enumerate(order.items, start=1) if d.item_code is None]
    too_long = [f"{_label(i, d.name)} {d.days}일" for i, d in enumerate(order.items, start=1) if d.days > TRAY_DAYS]
    if unmatched or too_long:
        problems = []
        if too_long:
            problems.append(f"투약 일수가 트레이({TRAY_DAYS}일)를 넘습니다: {', '.join(too_long)}")
        if unmatched:
            problems.append(f"약을 정하지 못한 줄이 있습니다: {', '.join(unmatched)}")
        code = E_PLAN_RANGE if too_long else None
        raise PlanError("조제 계획을 만들 수 없습니다. " + " / ".join(problems), code=code)

    days = max(d.days for d in order.items)
    cells: list[PlanCell] = []
    for day in range(1, days + 1):
        for time in TIME_SLOTS:
            counts: dict[str, int] = {}
            for d in order.items:
                if time in d.times and day <= d.days:
                    counts[d.item_code] = counts.get(d.item_code, 0) + d.count_per_dose
            if counts:
                cells.append(PlanCell(day=day, time=time, items=[PlanItem(item_code=c, count=n) for c, n in counts.items()]))
    return Plan(prescription_id=order.prescription_id, days=days, plan=cells)


def plan_lines(plan: Plan, names: dict[str, str] | None = None) -> list[str]:
    """화면에 보일 조제 계획 (시점마다 한 줄, 일자별 칸). names는 품목기준코드 -> 약품 이름."""
    names = names or {}
    by_cell = {(c.day, c.time): c for c in plan.plan}
    lines = []
    for time in TIME_SLOTS:
        if not any((day, time) in by_cell for day in range(1, plan.days + 1)):
            continue
        parts = []
        for day in range(1, plan.days + 1):
            cell = by_cell.get((day, time))
            content = " · ".join(f"{names.get(i.item_code, i.item_code)} {i.count}" for i in cell.items) if cell else "-"
            parts.append(f"{day}일 {content}")
        lines.append(f"{TIME_LABELS[time]}: " + " | ".join(parts))
    return lines
