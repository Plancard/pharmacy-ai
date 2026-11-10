"""agent/schema.py (부록 A 형식)와 agent/resources/samples/ 의 더미 JSON을 검사한다.

1. 샘플이 모두 형식을 통과한다 (A.1 · A.3 · A.4 · A.5 · A.6, 위험 신호 필드 포함)
2. 샘플끼리 서로 맞는다 (A.1 줄 번호 ↔ A.5 근거, A.5 ↔ A.4 계획 ↔ A.3 검수, A.5 ↔ A.6 위험 수준)
3. 어긋난 값은 거부한다
"""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.config import RESOURCES_DIR
from agent.schema import (
    TIME_SLOTS, TRAY_CELLS, ConsultResult, DispensingOrder, DrugLine, Inspection, OcrResult, Plan, RiskCheck,
    RiskFinding, max_risk, risk_rank,
)

SAMPLES = RESOURCES_DIR / "samples"

# 대상 약품 8종 (품목기준코드 → 이름). 이름은 cv/classes.yaml과 같다 (test_schema_contract.py가 대조한다)
TARGET_8 = {
    "202106092": "타이레놀정500mg", "197900277": "게보린정", "201110120": "트리스펜연질캡슐", "200200099": "프론크논캡슐",
    "200201239": "아르펜정", "200713170": "아이월드요비신정 416.7mg", "200002714": "트립라인정", "200308774": "뮤코다인캡슐200mg",
}
TYLENOL, GEVORIN, TRISPEN, ARPEN = "202106092", "197900277", "201110120", "200201239"


def load(kind: str, name: str) -> dict:
    return json.loads((SAMPLES / kind / f"{name}.json").read_text(encoding="utf-8"))


def names(kind: str) -> list[str]:
    return sorted(p.stem for p in (SAMPLES / kind).glob("*.json"))


RX_NAMES = ["s01_combo_a", "s02_combo_b", "s03_no_risk", "s04_flagged", "s05_no_code"]


# ------------------------------------------------------------------ 1. 샘플이 형식을 통과한다

@pytest.mark.parametrize("kind, model", [
    ("a1_ocr", OcrResult), ("a5_order", DispensingOrder), ("a4_plan", Plan),
    ("a3_inspection", Inspection), ("a6_consult", ConsultResult),
])
def test_every_sample_passes_its_format(kind, model):
    found = names(kind)
    assert found, f"{kind}에 샘플이 없습니다"
    for name in found:
        model.model_validate(load(kind, name))


def test_sample_set_is_complete():
    assert names("a1_ocr") == names("a5_order") == sorted(RX_NAMES)   # OCR 하나마다 정답 하나
    assert names("a4_plan") == ["s01_combo_a"]
    assert names("a3_inspection") == ["i01_ok", "i02_mismatch"]
    assert names("a6_consult") == ["c01_combo_a_rx_only", "c02_no_risk_with_current_meds"]


def test_round_trip_keeps_values():
    for kind, model in [("a5_order", DispensingOrder), ("a6_consult", ConsultResult), ("a3_inspection", Inspection)]:
        for name in names(kind):
            obj = model.model_validate(load(kind, name))
            assert model.model_validate(obj.model_dump(mode="json")) == obj


# ------------------------------------------------------------------ 2. 샘플끼리 맞는다

@pytest.mark.parametrize("name", RX_NAMES)
def test_order_source_lines_exist_in_its_ocr(name):
    ocr = OcrResult.model_validate(load("a1_ocr", name))
    order = DispensingOrder.model_validate(load("a5_order", name))
    ids = {ln.line_id for ln in ocr.lines}
    used: list[int] = []
    for d in order.items:
        assert d.source_line_ids, f"{d.name}: 근거 줄이 없습니다"
        assert set(d.source_line_ids) <= ids
        used += d.source_line_ids
    assert len(used) == len(set(used)), "한 OCR 줄이 두 약품의 근거로 쓰였습니다"


@pytest.mark.parametrize("name", RX_NAMES)
def test_order_name_is_read_from_the_named_line(name):
    """근거 줄 중 약품명 줄(가장 번호가 작은 줄)에, 정답 이름의 앞 두 글자가 들어 있다."""
    ocr = OcrResult.model_validate(load("a1_ocr", name))
    text = {ln.line_id: ln.text for ln in ocr.lines}
    for d in DispensingOrder.model_validate(load("a5_order", name)).items:
        assert d.name[:2] in text[min(d.source_line_ids)], d.name


def test_samples_use_all_eight_target_drugs():
    used = {d.item_code for n in RX_NAMES for d in DispensingOrder.model_validate(load("a5_order", n)).items}
    assert set(TARGET_8) <= used
    for n in RX_NAMES:                       # 사전의 이름과 같다 (못 정한 약은 제외)
        for d in DispensingOrder.model_validate(load("a5_order", n)).items:
            if d.item_code:
                assert d.name == TARGET_8[d.item_code]


def test_demo_combinations_a_and_b():
    a = DispensingOrder.model_validate(load("a5_order", "s01_combo_a"))
    b = DispensingOrder.model_validate(load("a5_order", "s02_combo_b"))
    assert a.risk_check.level == b.risk_check.level == "warning"
    assert {tuple(sorted(f.item_codes)) for f in a.risk_check.findings} == {tuple(sorted([TYLENOL, GEVORIN]))}
    assert {tuple(sorted(f.item_codes)) for f in b.risk_check.findings} == {tuple(sorted([TRISPEN, ARPEN]))}


def test_samples_cover_every_risk_level():
    levels = set()
    for n in RX_NAMES:
        rc = DispensingOrder.model_validate(load("a5_order", n)).risk_check
        levels |= {rc.level} | {f.level for f in rc.findings}
    for n in names("a6_consult"):
        c = ConsultResult.model_validate(load("a6_consult", n))
        levels |= {c.risk_level} | {f.level for f in c.new_findings}
    assert levels == {"none", "caution", "warning", "danger"}


def test_flagged_sample_marks_pharmacist_check():
    order = DispensingOrder.model_validate(load("a5_order", "s04_flagged"))
    assert order.flagged_indices == [0, 3]
    assert not order.all_matched                         # 알모탄정은 대상 8종이 아니라 item_code가 없다
    assert DispensingOrder.model_validate(load("a5_order", "s03_no_risk")).flagged_indices == []


def test_plan_is_what_the_order_asks_for():
    order = DispensingOrder.model_validate(load("a5_order", "s01_combo_a"))
    plan = Plan.model_validate(load("a4_plan", "s01_combo_a"))
    assert plan.prescription_id == order.prescription_id
    want = {(day, t): {d.item_code: d.count_per_dose for d in order.items if t in d.times and day <= d.days}
            for day in range(1, 4) for t in TIME_SLOTS}
    got = {(c.day, c.time): {i.item_code: i.count for i in c.items} for c in plan.plan}
    assert got == {k: v for k, v in want.items() if v}


@pytest.mark.parametrize("name", ["i01_ok", "i02_mismatch"])
def test_inspection_expected_equals_plan(name):
    plan = Plan.model_validate(load("a4_plan", "s01_combo_a"))
    insp = Inspection.model_validate(load("a3_inspection", name))
    assert insp.prescription_id == plan.prescription_id
    assert len(insp.cells) == TRAY_CELLS
    expect = {(c.day, c.time): {i.item_code: i.count for i in c.items} for c in plan.plan}
    for c in insp.cells:
        assert c.expected == expect.get((c.day, c.time), {})


def test_inspection_samples_show_ok_and_trouble():
    ok = Inspection.model_validate(load("a3_inspection", "i01_ok"))
    bad = Inspection.model_validate(load("a3_inspection", "i02_mismatch"))
    assert ok.summary.hitl_required is False and ok.summary.ok == 12
    assert bad.summary.hitl_required is True
    assert (bad.summary.mismatch, bad.summary.need_review) == (1, 1)


def test_consult_risk_level_is_not_lower_than_the_order():
    """A.6 risk_level = A.5 risk_check와 재대조 중 높은 쪽."""
    for name in names("a6_consult"):
        c = ConsultResult.model_validate(load("a6_consult", name))
        orders = {DispensingOrder.model_validate(load("a5_order", n)).prescription_id: n for n in RX_NAMES}
        order = DispensingOrder.model_validate(load("a5_order", orders[c.prescription_id]))
        assert risk_rank(c.risk_level) >= risk_rank(order.risk_check.level)


def test_new_findings_are_not_already_shown_in_the_order():
    c = ConsultResult.model_validate(load("a6_consult", "c02_no_risk_with_current_meds"))
    order = DispensingOrder.model_validate(load("a5_order", "s03_no_risk"))
    shown = {(f.type, tuple(sorted(f.item_codes))) for f in order.risk_check.findings}
    assert c.new_findings
    for f in c.new_findings:
        assert (f.type, tuple(sorted(f.item_codes))) not in shown


# ------------------------------------------------------------------ 3. 어긋난 값은 거부한다 — 공통

def test_common_constants():
    assert TIME_SLOTS == ("morning", "noon", "evening", "bedtime")
    assert TRAY_CELLS == 12


def test_risk_order_and_max():
    assert [risk_rank(x) for x in ("none", "caution", "warning", "danger")] == [0, 1, 2, 3]
    assert max_risk([]) == "none"
    assert max_risk(["caution", "danger", "warning"]) == "danger"


def test_unknown_keys_are_rejected():
    d = load("a5_order", "s01_combo_a")
    d["items"][0]["rx_code"] = "900000011"          # 예전 사전의 칸. 형식에 없는 키다
    with pytest.raises(ValidationError):
        DispensingOrder.model_validate(d)
    d = load("a1_ocr", "s01_combo_a")
    d["engine"] = "easyocr"
    with pytest.raises(ValidationError):
        OcrResult.model_validate(d)


# ------------------------------------------------------------------ A.1

def test_ocr_empty_lines_is_still_a_valid_shape():
    d = load("a1_ocr", "s01_combo_a")
    d["lines"] = []
    assert OcrResult.model_validate(d).lines == []


@pytest.mark.parametrize("mutate, why", [
    (lambda d: d["lines"].__setitem__(1, {**d["lines"][1], "line_id": 7}), "line_id가 0부터 차례가 아님"),
    (lambda d: d["lines"].reverse(), "정렬 순서가 아님"),
    (lambda d: d["lines"][0].__setitem__("bbox", [206, 144, 9999, 191]), "bbox가 이미지 밖"),
    (lambda d: d["lines"][0].__setitem__("bbox", [468, 144, 206, 191]), "x1 >= x2"),
    (lambda d: d["lines"][0].__setitem__("bbox", [206, 144, 468]), "bbox가 4개가 아님"),
    (lambda d: d["lines"][0].__setitem__("conf", 1.2), "conf가 1을 넘음"),
    (lambda d: d["lines"][0].__setitem__("text", "  "), "글자가 비어 있음"),
    (lambda d: d.__setitem__("doc_type", "receipt"), "doc_type이 아님"),
    (lambda d: d.__setitem__("width", 0), "width가 0"),
])
def test_ocr_rejects(mutate, why):
    d = load("a1_ocr", "s01_combo_a")
    mutate(d)
    with pytest.raises(ValidationError):
        OcrResult.model_validate(d)


def test_ocr_accepts_med_bag():
    d = load("a1_ocr", "s01_combo_a")
    d["doc_type"] = "med_bag"
    assert OcrResult.model_validate(d).doc_type == "med_bag"


# ------------------------------------------------------------------ A.5 약품 줄

def line(**kw) -> dict:
    base = {"item_code": TYLENOL, "name": "타이레놀정500mg", "count_per_dose": 1,
            "times": ["morning"], "days": 3, "source_line_ids": [1]}
    return {**base, **kw}


@pytest.mark.parametrize("kw", [
    {"item_code": "20210609"},                       # 8자리
    {"item_code": "2021060920"},                     # 10자리
    {"item_code": "20210609A"},                      # 글자 섞임
    {"item_code": "２０２１０６０９２"},                  # 전각 숫자
    {"name": ""},
    {"count_per_dose": 0},
    {"days": 0},
    {"times": []},
    {"times": ["midnight"]},
    {"times": ["morning", "morning"]},
    {"source_line_ids": [-1]},
    {"source_line_ids": [3, 3]},
    {"ocr_conf": 1.5},
    {"ocr_conf": -0.1},
    {"check_reasons": ["이유만 있음"]},                # 이유는 있는데 확인 표시가 없음
])
def test_drug_line_rejects(kw):
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(**kw))


def test_drug_line_accepts_unmatched_when_flagged():
    d = DrugLine.model_validate(line(item_code=None, needs_pharmacist_check=True, check_reasons=["대상 아님"]))
    assert d.item_code is None


def test_drug_line_without_item_code_is_fine_before_matching():
    """구조화 직후에는 아직 약을 정하지 않았다. 웹에 넘기기 전 검사는 check_ready_for_web이 한다."""
    assert DrugLine.model_validate(line(item_code=None)).needs_pharmacist_check is False


def test_drug_line_internal_fields():
    d = DrugLine.model_validate(line(printed_code="900000011", ocr_conf=0.43))
    assert (d.printed_code, d.ocr_conf) == ("900000011", 0.43)
    assert (DrugLine.model_validate(line()).printed_code, DrugLine.model_validate(line()).ocr_conf) == (None, None)


def test_drug_line_without_source_lines_is_ok():
    """LLM 경로는 어느 줄을 읽었는지 모른다."""
    d = line()
    del d["source_line_ids"]
    assert DrugLine.model_validate(d).source_line_ids == []


def test_days_over_tray_is_a_valid_order():
    """4일 이상 처방은 조제지시서로는 올바르다. 트레이에 담을 수 없는 문제는 조제 계획 단계(E_PLAN_RANGE)가 다룬다."""
    assert DrugLine.model_validate(line(days=5)).days == 5


# ------------------------------------------------------------------ A.5 약품 줄: 약품 매칭 결과 (4단계)

def candidate(item_code=GEVORIN, **kw) -> dict:
    return {"item_code": item_code, "name": "게보린정", "score": 0.824, "reason": "이름이 비슷함", **kw}


def test_match_fields_default_to_not_matched_yet():
    d = DrugLine.model_validate(line())
    assert (d.match_method, d.match_candidates) == (None, [])


@pytest.mark.parametrize("method", ["code", "name", "api"])
def test_a_method_that_found_a_drug_needs_an_item_code(method):
    assert DrugLine.model_validate(line(match_method=method)).match_method == method
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(item_code=None, match_method=method))


def test_method_none_means_the_drug_was_not_chosen():
    d = DrugLine.model_validate(line(item_code=None, match_method="none", needs_pharmacist_check=True))
    assert (d.item_code, d.match_method) == (None, "none")
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(match_method="none"))          # 약을 골랐는데 고르지 못했다고 적을 수 없다


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(match_method="pharmacist"))


def test_candidates_are_shown_only_on_a_flagged_line():
    d = DrugLine.model_validate(line(item_code=None, match_method="none", needs_pharmacist_check=True,
                                     match_candidates=[candidate()]))
    assert [(c.item_code, c.score) for c in d.match_candidates] == [(GEVORIN, 0.824)]
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(item_code=None, match_candidates=[candidate()]))     # 확인 표시가 없다


def test_at_most_three_distinct_candidates():
    codes = [GEVORIN, TYLENOL, TRISPEN, ARPEN]
    ok = line(item_code=None, needs_pharmacist_check=True, match_candidates=[candidate(c) for c in codes[:3]])
    assert len(DrugLine.model_validate(ok).match_candidates) == 3
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(item_code=None, needs_pharmacist_check=True, match_candidates=[candidate(c) for c in codes]))
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(item_code=None, needs_pharmacist_check=True, match_candidates=[candidate(), candidate()]))


@pytest.mark.parametrize("bad", [
    {"item_code": "19790027"}, {"name": ""}, {"score": 1.01}, {"score": -0.1}, {"reason": ""}, {"extra": 1},
])
def test_a_candidate_rejects(bad):
    with pytest.raises(ValidationError):
        DrugLine.model_validate(line(item_code=None, needs_pharmacist_check=True, match_candidates=[candidate(**bad)]))


# ------------------------------------------------------------------ A.5 risk_check

def order(items=None, risk_check=None) -> dict:
    d = {"prescription_id": "P0001",
         "items": items or [line(), line(item_code=GEVORIN, name="게보린정", source_line_ids=[2])]}
    if risk_check is not None:
        d["risk_check"] = risk_check
    return d


def finding(**kw) -> dict:
    base = {"type": "ingredient_dup", "item_codes": [TYLENOL, GEVORIN], "level": "warning", "message": "예시"}
    return {**base, **kw}


def check(level="warning", findings=None, scope="rx_only") -> dict:
    return {"scope": scope, "level": level, "findings": [finding()] if findings is None else findings}


def test_order_without_risk_check_is_allowed_before_the_check():
    o = DispensingOrder.model_validate(order())
    assert o.risk_check is None


def test_order_internal_fields_default_to_empty():
    o = DispensingOrder.model_validate(order())
    assert (o.issue_no, o.source, o.warnings) == (None, None, [])
    o = DispensingOrder.model_validate(order() | {"issue_no": "2026년11월09일 00061호", "source": "llm", "warnings": ["확인"]})
    assert (o.issue_no, o.source, o.warnings) == ("2026년11월09일 00061호", "llm", ["확인"])
    with pytest.raises(ValidationError):
        DispensingOrder.model_validate(order() | {"source": "manual"})


def test_samples_are_ready_for_web():
    for n in RX_NAMES:
        DispensingOrder.model_validate(load("a5_order", n)).check_ready_for_web()


def test_check_ready_for_web_needs_risk_check():
    with pytest.raises(ValueError, match="risk_check"):
        DispensingOrder.model_validate(order()).check_ready_for_web()
    DispensingOrder.model_validate(order(risk_check=check())).check_ready_for_web()


def test_check_ready_for_web_needs_flag_on_unmatched_line():
    items = [line(), line(item_code=None, name="알모탄정")]
    with pytest.raises(ValueError, match="알모탄정"):
        DispensingOrder.model_validate(order(items=items, risk_check=check(level="none", findings=[]))).check_ready_for_web()
    items[1] |= {"needs_pharmacist_check": True, "check_reasons": ["대상 아님"]}
    DispensingOrder.model_validate(order(items=items, risk_check=check(level="none", findings=[]))).check_ready_for_web()


def test_order_needs_at_least_one_item():
    with pytest.raises(ValidationError):
        DispensingOrder.model_validate({"prescription_id": "P0001", "items": []})


def test_risk_check_none_level_with_no_findings():
    rc = RiskCheck.model_validate({"scope": "rx_only", "level": "none"})
    assert rc.findings == []


@pytest.mark.parametrize("rc", [
    check(level="none"),                                                   # 항목이 있는데 none
    check(level="danger"),                                                 # 가장 높은 수준(warning)보다 높음
    check(level="caution"),                                                # 더 낮음
    check(level="warning", findings=[]),                                   # 항목이 없는데 warning
    check(findings=[finding(level="none")]),                               # 항목의 수준은 none일 수 없음
    check(findings=[finding(item_codes=[])]),
    check(findings=[finding(item_codes=[TYLENOL, TYLENOL])]),
    check(findings=[finding(type="")]),
    check(findings=[finding(message="")]),
    check(scope="everything"),
])
def test_risk_check_rejects(rc):
    with pytest.raises(ValidationError):
        RiskCheck.model_validate(rc)


def test_risk_check_level_is_the_highest_finding():
    fs = [finding(level="caution"), finding(level="danger", item_codes=[GEVORIN])]
    assert RiskCheck.model_validate(check(level="danger", findings=fs)).level == "danger"


def test_order_risk_check_must_point_at_items_in_the_order():
    rc = check(findings=[finding(item_codes=[TYLENOL, ARPEN])])           # 아르펜은 이 처방에 없음
    with pytest.raises(ValidationError):
        DispensingOrder.model_validate(order(risk_check=rc))


def test_order_risk_check_is_prescription_only():
    with pytest.raises(ValidationError):
        DispensingOrder.model_validate(order(risk_check=check(scope="rx_with_current_meds")))


def test_risk_finding_type_is_not_restricted_yet():
    """type의 영어 키는 13단계(DUR)에서 확정한다. 그 전까지는 빈 문자열만 막는다."""
    assert RiskFinding.model_validate(finding(type="anything_new")).type == "anything_new"


# ------------------------------------------------------------------ A.4

def plan(cells=None, days=3) -> dict:
    cells = cells if cells is not None else [{"day": 1, "time": "morning", "items": [{"item_code": TYLENOL, "count": 1}]}]
    return {"prescription_id": "P0001", "days": days, "plan": cells}


def cell(**kw) -> dict:
    return {**{"day": 1, "time": "morning", "items": [{"item_code": TYLENOL, "count": 1}]}, **kw}


@pytest.mark.parametrize("p", [
    plan([cell(day=4)], days=3),                                           # 4일차: E_PLAN_RANGE 대상
    plan([cell(day=0)]),
    plan([cell(time="midnight")]),
    plan(days=4),
    plan(days=0),
    plan([cell(), cell()]),                                                # 같은 칸 두 번
    plan([cell(day=3)], days=2),                                           # days를 넘는 칸
    plan([cell(items=[])]),                                                # 빈 칸은 적지 않는다
    plan([cell(items=[{"item_code": TYLENOL, "count": 0}])]),
    plan([cell(items=[{"item_code": TYLENOL, "count": 1}, {"item_code": TYLENOL, "count": 1}])]),
    plan([cell(items=[{"item_code": "123", "count": 1}])]),
])
def test_plan_rejects(p):
    with pytest.raises(ValidationError):
        Plan.model_validate(p)


def test_plan_may_be_empty_of_cells_only_if_nothing_to_dispense():
    assert Plan.model_validate(plan([])).plan == []


# ------------------------------------------------------------------ A.3

def test_inspection_rejects_wrong_cell_count():
    d = load("a3_inspection", "i01_ok")
    d["cells"].pop()
    d["summary"]["ok"] = 11
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)


def test_inspection_rejects_extra_cell():
    d = load("a3_inspection", "i01_ok")
    d["cells"].append(copy.deepcopy(d["cells"][0]))
    d["summary"]["ok"] = 13
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)


def test_inspection_rejects_duplicate_cell():
    d = load("a3_inspection", "i01_ok")
    d["cells"][1] = copy.deepcopy(d["cells"][0])
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)


def test_inspection_rejects_summary_that_disagrees_with_cells():
    d = load("a3_inspection", "i02_mismatch")
    d["summary"]["mismatch"] = 0
    d["summary"]["ok"] = 11
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)


def test_inspection_rejects_status_that_disagrees_with_counts():
    d = load("a3_inspection", "i01_ok")
    d["cells"][0]["actual"] = {TYLENOL: 0}                                 # 있어야 할 약이 없는데 ok
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)
    d = load("a3_inspection", "i02_mismatch")
    bad = next(c for c in d["cells"] if c["status"] == "mismatch")
    bad["actual"] = dict(bad["expected"])                                  # 같은데 mismatch
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)


def test_inspection_rejects_unknown_reason_and_status():
    d = load("a3_inspection", "i02_mismatch")
    review = next(c for c in d["cells"] if c["status"] == "need_review")
    review["reasons"] = ["blurry"]
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)
    d = load("a3_inspection", "i01_ok")
    d["cells"][0]["status"] = "fine"
    with pytest.raises(ValidationError):
        Inspection.model_validate(d)


def test_inspection_allows_every_reason():
    d = load("a3_inspection", "i02_mismatch")
    review = next(c for c in d["cells"] if c["status"] == "need_review")
    review["reasons"] = ["low_conf", "boundary", "overlap", "rotated"]
    assert Inspection.model_validate(d).cells


# ------------------------------------------------------------------ A.6

def consult(**kw) -> dict:
    base = {"prescription_id": "P0001", "risk_scope": "rx_only", "new_findings": [], "recheck_after_dispense": False,
            "risk_level": "none", "answer": "예시 답변", "evidence": []}
    return {**base, **kw}


def test_consult_minimal_is_valid():
    assert ConsultResult.model_validate(consult()).evidence == []


@pytest.mark.parametrize("c", [
    consult(risk_scope="rx_only", new_findings=[finding()], risk_level="warning"),   # 처방약끼리만 봤는데 새 항목
    consult(risk_scope="rx_with_current_meds", new_findings=[finding(level="danger")], risk_level="warning"),   # 항목보다 낮음
    consult(risk_scope="rx_with_current_meds", new_findings=[finding()], risk_level="none"),
    consult(risk_scope="whole_world"),
    consult(risk_level="severe"),
    consult(answer=""),
    consult(recheck_after_dispense="maybe"),
    consult(evidence=[{"title": "제목", "source": "출처"}]),                         # excerpt 없음
])
def test_consult_rejects(c):
    with pytest.raises(ValidationError):
        ConsultResult.model_validate(c)


def test_consult_risk_level_may_exceed_new_findings():
    """A.5의 위험이 더 높으면 risk_level은 그쪽을 따른다 (둘 중 높은 쪽)."""
    c = consult(risk_scope="rx_with_current_meds", new_findings=[finding(level="caution")], risk_level="danger")
    assert ConsultResult.model_validate(c).risk_level == "danger"


def test_consult_new_findings_may_point_at_current_meds_not_in_the_order():
    f = finding(item_codes=[TYLENOL, ARPEN], level="caution")                # 복용 중인 약(아르펜)은 처방에 없다
    c = consult(risk_scope="rx_with_current_meds", new_findings=[f], risk_level="caution", recheck_after_dispense=True)
    assert ConsultResult.model_validate(c).recheck_after_dispense is True


# 이 파일의 경로 가정 점검: 샘플 폴더가 git에 올라가는 위치인지 (data/ 아래면 .gitignore에 걸린다)
def test_samples_are_not_under_a_data_folder():
    assert "data" not in Path(SAMPLES).relative_to(RESOURCES_DIR.parent.parent).parts
