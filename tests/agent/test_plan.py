"""칸별 조제 계획 (agent/plan.py)."""
import copy
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.errors import E_PLAN_RANGE
from agent.plan import MAX_PER_CELL, TIME_LABELS, PlanError, build_plan, plan_lines
from agent.schema import TIME_SLOTS, TRAY_DAYS, DispensingOrder, DrugLine, Plan

SAMPLES = RESOURCES_DIR / "samples"
TYLENOL, GEVORIN, MUCODYNE = "202106092", "197900277", "200308774"


def line(code=TYLENOL, name="타이레놀정500mg", count=1, times=("morning",), days=3):
    return DrugLine(item_code=code, name=name, printed_code=None, count_per_dose=count, times=list(times), days=days)


def order_of(*lines):
    return DispensingOrder(prescription_id="P0001", items=list(lines))


def cells(plan):
    return [(c.day, c.time, [(i.item_code, i.count) for i in c.items]) for c in plan.plan]


def load(kind, name):
    return json.loads((SAMPLES / kind / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------ 팀 샘플


def test_the_plan_of_sample_a_equals_the_team_a4_sample():
    plan = build_plan(DispensingOrder.model_validate(load("a5_order", "s01_combo_a")))
    assert plan == Plan.model_validate(load("a4_plan", "s01_combo_a"))


@pytest.mark.parametrize("name", ["s01_combo_a", "s02_combo_b", "s05_no_code"])
def test_every_plannable_sample_gives_what_its_order_asks_for(name):
    order = DispensingOrder.model_validate(load("a5_order", name))
    plan = build_plan(order)
    want = {}
    for day in range(1, TRAY_DAYS + 1):
        for t in TIME_SLOTS:
            got = {d.item_code: d.count_per_dose for d in order.items if t in d.times and day <= d.days}
            if got:
                want[(day, t)] = got
    assert {(c.day, c.time): {i.item_code: i.count for i in c.items} for c in plan.plan} == want
    Plan.model_validate(plan.model_dump(mode="json"))                # A.4 검사를 통과한다


def test_the_flagged_sample_cannot_be_planned_for_two_reasons():
    """s04는 알모탄정의 약을 정하지 못했고, 아침 · 저녁 칸에 세 약이 1알씩 들어가 칸당 2알도 넘는다."""
    with pytest.raises(PlanError) as error:
        build_plan(DispensingOrder.model_validate(load("a5_order", "s04_flagged")))
    assert error.value.code == E_PLAN_RANGE
    assert str(error.value).endswith("/ 약을 정하지 못한 줄이 있습니다: 4번째 약품(알모탄정)")
    assert "트레이 칸당 최대 2알을 넘습니다: 1~3일 아침·저녁 3알" in str(error.value)


def test_sample_s03_cannot_be_planned_because_three_pills_go_in_a_cell():
    """s03은 프론크논 1알 + 뮤코다인 2알이 아침 · 저녁 칸에 함께 들어가 칸당 2알(명세서 2.7절)을 넘는다."""
    with pytest.raises(PlanError) as error:
        build_plan(DispensingOrder.model_validate(load("a5_order", "s03_no_risk")))
    assert error.value.code == E_PLAN_RANGE
    assert str(error.value) == (
        "조제 계획을 만들 수 없습니다. 트레이 칸당 최대 2알을 넘습니다: 1~3일 아침·저녁 3알"
    )


# ------------------------------------------------------------ 펼치기


def test_one_line_fills_its_times_for_its_days():
    plan = build_plan(order_of(line(times=("morning", "evening"), days=2)))
    assert (plan.prescription_id, plan.days) == ("P0001", 2)
    assert cells(plan) == [
        (1, "morning", [(TYLENOL, 1)]), (1, "evening", [(TYLENOL, 1)]),
        (2, "morning", [(TYLENOL, 1)]), (2, "evening", [(TYLENOL, 1)]),
    ]


def test_cells_go_by_day_then_by_time_of_the_tray_rows():
    plan = build_plan(order_of(line(times=("bedtime", "morning", "evening", "noon"), days=1)))
    assert [c.time for c in plan.plan] == ["morning", "noon", "evening", "bedtime"]


def test_lines_with_different_days_stop_at_their_own_day():
    plan = build_plan(order_of(line(days=3), line(GEVORIN, "게보린정", days=1)))
    assert plan.days == 3
    assert cells(plan) == [
        (1, "morning", [(TYLENOL, 1), (GEVORIN, 1)]), (2, "morning", [(TYLENOL, 1)]), (3, "morning", [(TYLENOL, 1)]),
    ]


def test_the_count_per_dose_is_the_count_in_the_cell():
    plan = build_plan(order_of(line(MUCODYNE, "뮤코다인캡슐200mg", count=2, days=1)))
    assert cells(plan) == [(1, "morning", [(MUCODYNE, 2)])]


def test_the_same_drug_on_two_lines_is_added_up_in_a_shared_cell():
    plan = build_plan(order_of(line(times=("morning",), days=1), line(times=("morning", "evening"), days=1)))
    assert cells(plan) == [(1, "morning", [(TYLENOL, 2)]), (1, "evening", [(TYLENOL, 1)])]


def test_the_order_inside_a_cell_follows_the_prescription():
    plan = build_plan(order_of(line(GEVORIN, "게보린정", days=1), line(days=1)))
    assert cells(plan) == [(1, "morning", [(GEVORIN, 1), (TYLENOL, 1)])]


def test_empty_cells_are_not_written():
    plan = build_plan(order_of(line(times=("noon",), days=3)))
    assert len(plan.plan) == 3 and {c.time for c in plan.plan} == {"noon"}


def test_three_days_is_the_most_the_tray_holds():
    assert build_plan(order_of(line(days=TRAY_DAYS))).days == 3


# ------------------------------------------------------------ 만들 수 없을 때


@pytest.mark.parametrize("days", [4, 7, 30])
def test_more_days_than_the_tray_is_the_range_error(days):
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(days=days)))
    assert error.value.code == E_PLAN_RANGE
    assert str(error.value) == f"조제 계획을 만들 수 없습니다. 투약 일수가 트레이(3일)를 넘습니다: 1번째 약품(타이레놀정500mg) {days}일"


def test_a_line_without_a_drug_cannot_be_planned():
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(), line(None, "가나다정")))
    assert error.value.code is None
    assert str(error.value) == "조제 계획을 만들 수 없습니다. 약을 정하지 못한 줄이 있습니다: 2번째 약품(가나다정)"


def test_every_problem_is_listed_and_the_range_code_wins():
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(None, "가나다정"), line(days=5), line(None, "라마바정", days=4)))
    assert error.value.code == E_PLAN_RANGE
    assert str(error.value) == (
        "조제 계획을 만들 수 없습니다. 투약 일수가 트레이(3일)를 넘습니다: 2번째 약품(타이레놀정500mg) 5일, 3번째 약품(라마바정) 4일"
        " / 트레이 칸당 최대 2알을 넘습니다: 1~3일 아침 3알"
        " / 약을 정하지 못한 줄이 있습니다: 1번째 약품(가나다정), 3번째 약품(라마바정)"
    )


# ------------------------------------------------------------ 칸당 알 수 (명세서 2.7절)


def test_two_pills_in_a_cell_is_the_most_a_cell_holds():
    assert MAX_PER_CELL == 2
    plan = build_plan(order_of(line(days=1), line(GEVORIN, "게보린정", days=1)))
    assert cells(plan) == [(1, "morning", [(TYLENOL, 1), (GEVORIN, 1)])]
    assert cells(build_plan(order_of(line(MUCODYNE, "뮤코다인캡슐200mg", count=2, days=1)))) == [(1, "morning", [(MUCODYNE, 2)])]


def test_three_pills_of_one_drug_in_a_cell_is_the_range_error():
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(count=3, times=("morning", "bedtime"), days=1)))
    assert error.value.code == E_PLAN_RANGE
    assert str(error.value) == "조제 계획을 만들 수 없습니다. 트레이 칸당 최대 2알을 넘습니다: 1일 아침·취침 전 3알"


def test_pills_of_different_drugs_in_the_same_cell_are_added_up():
    with pytest.raises(PlanError, match="1일 아침 3알"):
        build_plan(order_of(line(days=1), line(GEVORIN, "게보린정", days=1), line(MUCODYNE, "뮤코다인캡슐200mg", days=1)))


def test_only_the_shared_cells_are_too_full():
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(count=2, times=("morning", "noon"), days=2), line(GEVORIN, "게보린정", times=("noon",), days=1)))
    assert str(error.value).endswith("트레이 칸당 최대 2알을 넘습니다: 1일 점심 3알")


def test_a_full_cell_from_an_unknown_drug_still_counts():
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(count=2, days=1), line(None, "가나다정", days=1)))
    assert error.value.code == E_PLAN_RANGE
    assert str(error.value) == (
        "조제 계획을 만들 수 없습니다. 트레이 칸당 최대 2알을 넘습니다: 1일 아침 3알"
        " / 약을 정하지 못한 줄이 있습니다: 2번째 약품(가나다정)"
    )


def test_days_beyond_the_tray_are_not_counted_as_full_cells():
    with pytest.raises(PlanError) as error:
        build_plan(order_of(line(days=5)))
    assert "칸당" not in str(error.value)


@pytest.mark.parametrize("lines, said", [
    ((("morning", 3, 3), ("evening", 3, 3)), "1~3일 아침·저녁 3알"),
    ((("morning", 3, 1),), "1일 아침 3알"),
    ((("morning", 3, 2), ("noon", 4, 3)), "1~2일 아침 3알, 1~3일 점심 4알"),
    ((("noon", 3, 3), ("morning", 3, 3)), "1~3일 아침·점심 3알"),          # 시점 순서로
])
def test_crowded_cells_are_grouped_by_days_and_pills(lines, said):
    order = order_of(*(line(count=count, times=(time,), days=days) for time, count, days in lines))
    with pytest.raises(PlanError) as error:
        build_plan(order)
    assert str(error.value).endswith(f"트레이 칸당 최대 2알을 넘습니다: {said}")


def test_days_that_do_not_follow_each_other_are_listed():
    from agent.plan import _days_text

    assert (_days_text([1, 3]), _days_text([2]), _days_text([1, 2, 3]), _days_text([2, 3])) == ("1·3일", "2일", "1~3일", "2~3일")


# ------------------------------------------------------------ 값의 성질


def test_the_order_that_was_given_is_not_changed():
    order = order_of(line(), line(GEVORIN, "게보린정"))
    before = copy.deepcopy(order)
    build_plan(order)
    assert order == before


def test_the_plan_can_be_stored_as_json_and_read_back():
    plan = build_plan(order_of(line(times=("morning", "evening"))))
    assert Plan.model_validate(json.loads(json.dumps(plan.model_dump(mode="json")))) == plan


def test_every_time_slot_has_a_label():
    assert set(TIME_LABELS) == set(TIME_SLOTS)


# ------------------------------------------------------------ 화면에 보이는 모양


def test_plan_lines_show_one_row_per_time_with_every_day():
    plan = build_plan(order_of(line(times=("morning", "evening"), days=2), line(GEVORIN, "게보린정", days=1)))
    assert plan_lines(plan, {TYLENOL: "타이레놀", GEVORIN: "게보린"}) == [
        "아침: 1일 타이레놀 1 · 게보린 1 | 2일 타이레놀 1",
        "저녁: 1일 타이레놀 1 | 2일 타이레놀 1",
    ]


def test_plan_lines_show_an_empty_day_as_a_dash_and_codes_without_names():
    plan = build_plan(order_of(line(days=2), line(GEVORIN, "게보린정", times=("bedtime",), days=1)))
    assert plan_lines(plan) == [
        f"아침: 1일 {TYLENOL} 1 | 2일 {TYLENOL} 1",
        f"취침 전: 1일 {GEVORIN} 1 | 2일 -",
    ]


def test_plan_lines_of_an_empty_plan():
    assert plan_lines(Plan(prescription_id="P0001", days=1, plan=[])) == []
