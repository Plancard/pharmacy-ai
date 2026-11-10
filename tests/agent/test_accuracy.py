"""정확도 계산 (agent/accuracy.py)."""
import json

import pytest

from agent.accuracy import (
    CANDIDATE,
    CELL_FIELDS,
    CONFIRMED,
    HEADLINE_FIELDS,
    MATCH_LABELS,
    MATCH_OUTCOMES,
    MISSING,
    NOT_FOUND,
    NOT_TARGET,
    TARGET_RATE,
    TRUE_CELL_FIELDS,
    WRONG,
    Check,
    Summary,
    classify_match,
    compare,
    lines_all_correct,
    match_checks,
    name_misread_lines,
    score_sample,
    summarize,
)
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.drug_match import match_order
from agent.schema import DrugLine, MatchCandidate
from agent.structure import extract_rule
from agent.validate import order_from_extracted

SAMPLES = RESOURCES_DIR / "samples"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary()


def load(kind, sample_id):
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def rule_order(sample_id):
    return match_order(order_from_extracted("P1", extract_rule(load("cv_a1", sample_id)), source="rule"), DICT)


def order_from_answer(sample_id, changes=None):
    """정답과 같은 추출 결과에서 일부 칸만 바꿔 만든 조제지시서."""
    expected = load("cv_expected", sample_id)
    extracted = {
        "issue_no": expected["issue_no"],
        "items": [
            {k: d[k] for k in ("name", "printed_code", "count_per_dose", "times_per_day", "days")} | {"times": []}
            for d in expected["items"]
        ],
    }
    for (i, key), value in (changes or {}).items():
        if i is None:
            extracted[key] = value
        else:
            extracted["items"][i][key] = value
    return match_order(order_from_extracted("P1", extracted, source="llm"), DICT)


def wrong(checks):
    return [(c.line, c.field) for c in checks if not c.ok]


def drug_line(item_code=None, candidates=(), name="약"):
    """약품 매칭 결과만 다른 약품 한 줄."""
    method = None if item_code is None and not candidates else ("name" if item_code else "none")
    return DrugLine(
        item_code=item_code, name=name, count_per_dose=1, times=["morning"], days=1,
        needs_pharmacist_check=bool(candidates) or item_code is None, match_method=method,
        match_candidates=[
            MatchCandidate(item_code=c, name="후보", score=0.8, reason="이름이 비슷함") for c in candidates
        ],
    )


# ------------------------------------------------------------ 칸 비교


def test_fields_compared():
    assert CELL_FIELDS == ("printed_code", "name", "count_per_dose", "times_per_day", "days")
    assert TRUE_CELL_FIELDS == ("printed_code", "count_per_dose", "times_per_day", "days")
    assert HEADLINE_FIELDS == ("name", "count_per_dose", "times_per_day", "days")      # 일정표 기록: 약품명·1회량·횟수·일수
    assert set(HEADLINE_FIELDS) <= set(CELL_FIELDS) and TARGET_RATE == 0.9


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_the_rule_path_matches_every_cell_of_the_ocr_answer(sample_id):
    checks = compare(rule_order(sample_id), load("cv_expected", sample_id))
    items = len(load("cv_expected", sample_id)["items"])
    assert len(checks) == 1 + items * len(CELL_FIELDS)        # 교부번호 + 약품 줄마다 5칸
    assert wrong(checks) == []


def test_by_the_true_values_only_the_misread_issue_number_of_rx_004_is_wrong():
    for sample_id in SAMPLE_IDS:
        checks = compare(rule_order(sample_id), load("cv_expected", sample_id), truth=True)
        assert wrong(checks) == ([(None, "issue_no")] if sample_id == "rx_004" else [])
    checks = compare(rule_order("rx_004"), load("cv_expected", "rx_004"), truth=True)
    assert checks[0] == Check(None, "issue_no", "2026년11월03일 00058호", "2026년11월03일 000583", False)


def test_names_are_not_compared_by_the_true_values():
    checks = compare(rule_order("rx_001"), load("cv_expected", "rx_001"), truth=True)
    assert "name" not in {c.field for c in checks}


def test_a_wrong_cell_is_reported_with_both_values():
    checks = compare(order_from_answer("rx_002", {(1, "days"): 7}), load("cv_expected", "rx_002"))
    assert wrong(checks) == [(1, "days")]
    bad = next(c for c in checks if not c.ok)
    assert (bad.want, bad.got) == (2, 7)
    assert bad.describe() == "2번째 약품 투약 일수: 정답 2, 결과 7"


def test_the_daily_count_is_the_number_of_times():
    # 하루 횟수는 복용 시점의 수다 (조제지시서에 따로 칸이 없다). 3회를 2회로 읽으면 틀린 것이다
    checks = compare(order_from_answer("rx_002", {(1, "times_per_day"): 2}), load("cv_expected", "rx_002"))
    assert wrong(checks) == [(1, "times_per_day")]
    assert next(c for c in checks if not c.ok).got == 2


def test_a_wrong_printed_code_is_wrong_only_in_its_own_cell():
    checks = compare(order_from_answer("rx_001", {(0, "printed_code"): "900000021"}), load("cv_expected", "rx_001"))
    assert wrong(checks) == [(0, "printed_code")]


def test_names_that_differ_only_by_spaces_or_slashes_are_equal():
    checks = compare(order_from_answer("rx_001", {(0, "name"): "타이레놀정/5oOmg"}), load("cv_expected", "rx_001"))
    assert wrong(checks) == []


def test_a_corrected_name_is_wrong_by_the_ocr_answer():
    # LLM이 OCR 오류를 '고쳐' 오면 OCR 기준으로는 틀린 것이다 (적힌 그대로 옮기라고 했다)
    checks = compare(order_from_answer("rx_001", {(1, "name"): "알모탄정"}), load("cv_expected", "rx_001"))
    assert wrong(checks) == [(1, "name")]


def test_a_wrong_issue_number():
    checks = compare(order_from_answer("rx_001", {(None, "issue_no"): None}), load("cv_expected", "rx_001"))
    assert wrong(checks) == [(None, "issue_no")]
    assert checks[0].describe() == "교부번호: 정답 '2026년11월03일 00012호', 결과 None"


def test_a_missing_drug_makes_all_its_cells_wrong():
    order = order_from_answer("rx_002")
    order = order.model_copy(update={"items": order.items[:2]})
    checks = compare(order, load("cv_expected", "rx_002"))
    assert wrong(checks) == [(2, f) for f in CELL_FIELDS]
    assert all(c.got is None for c in checks if c.line == 2)


def test_an_extra_drug_is_counted_as_one_wrong_cell():
    order = order_from_answer("rx_002")
    order = order.model_copy(update={"items": [*order.items, order.items[0]]})
    checks = compare(order, load("cv_expected", "rx_002"))
    assert wrong(checks) == [(3, "extra")]
    assert checks[-1].describe() == "4번째 약품 정답에 없는 약품: 정답 None, 결과 '게보린정'"


def test_no_order_makes_every_cell_wrong():
    expected = load("cv_expected", "rx_001")
    checks = compare(None, expected)
    assert len(checks) == 1 + 2 * len(CELL_FIELDS) and not any(c.ok for c in checks)
    assert all(c.got is None for c in checks)


# ------------------------------------------------------------ 줄·장 단위


def test_lines_all_correct():
    checks = compare(order_from_answer("rx_004", {(1, "days"): 9, (3, "name"): "가"}), load("cv_expected", "rx_004"))
    assert lines_all_correct(checks, 4) == 2


def test_an_extra_drug_does_not_spoil_the_correct_lines():
    order = order_from_answer("rx_002")
    order = order.model_copy(update={"items": [*order.items, order.items[0]]})
    assert lines_all_correct(compare(order, load("cv_expected", "rx_002")), 3) == 3


@pytest.mark.parametrize("sample_id, count", [("rx_001", 1), ("rx_002", 2), ("rx_003", 2), ("rx_004", 2), ("rx_005", 4)])
def test_name_misread_lines(sample_id, count):
    # 타이레놀정 5oOmg는 o -> 0 보정 뒤 진짜 이름과 같아 틀리게 읽은 줄로 세지 않는다
    assert name_misread_lines(load("cv_expected", sample_id)) == count


# ------------------------------------------------------------ 약품 매칭


def test_match_outcomes_have_a_label_each():
    assert MATCH_OUTCOMES == (CONFIRMED, CANDIDATE, NOT_TARGET, NOT_FOUND, WRONG, MISSING)
    assert set(MATCH_LABELS) == set(MATCH_OUTCOMES)


@pytest.mark.parametrize(
    "want, line, outcome",
    [
        ("202106092", drug_line("202106092"), CONFIRMED),
        ("202106092", drug_line("197900277"), WRONG),                          # 다른 약을 정했다
        (None, drug_line("197900277"), WRONG),                                 # 대상이 아닌 약을 대상 약으로 정했다
        (None, drug_line(None), NOT_TARGET),
        (None, drug_line(None, candidates=["200201239"]), NOT_TARGET),         # 대상 약이 아닌 줄에 엉뚱한 후보가 있어도 정하지는 않았다
        ("200200099", drug_line(None, candidates=["200200099"]), CANDIDATE),
        ("200200099", drug_line(None, candidates=["200201239", "200200099"]), CANDIDATE),
        ("200200099", drug_line(None, candidates=["200201239"]), NOT_FOUND),    # 후보는 있지만 정답이 없다
        ("200200099", drug_line(None), NOT_FOUND),
        ("200200099", None, MISSING),
        (None, None, MISSING),
    ],
)
def test_classify_match(want, line, outcome):
    assert classify_match(want, line) == outcome


def test_a_confirmed_line_is_confirmed_even_when_other_candidates_exist():
    line = drug_line("202106092", candidates=["197900277"])
    assert classify_match("202106092", line) == CONFIRMED


def test_the_rule_path_confirms_only_what_the_names_support():
    # 인쇄 코드(9000000xx)는 사전의 품목기준코드가 아니라서 모두 이름 단계다.
    # 이름이 맞게 읽힌 타이레놀·게보린은 정하고, OCR이 이름을 틀리게 읽은 대상 약(프론크논·트리스펜)은 후보로만 보인다.
    # 대상이 아닌 약(알모탄정, rx_005)은 정하지 않는다.
    outcomes = [m.outcome for sample_id in SAMPLE_IDS for m in match_checks(rule_order(sample_id), load("cv_expected", sample_id))]
    assert outcomes.count(CONFIRMED) == 4 and outcomes.count(CANDIDATE) == 3 and outcomes.count(NOT_TARGET) == 8
    assert WRONG not in outcomes and NOT_FOUND not in outcomes and MISSING not in outcomes


def test_a_wrongly_chosen_drug_is_found():
    # 타이레놀 줄을 게보린이라고 읽어 오면 이름 단계가 게보린을 정한다. 정답은 타이레놀이라 잘못 정한 것이다
    checks = match_checks(order_from_answer("rx_001", {(0, "name"): "게보린정"}), load("cv_expected", "rx_001"))
    assert [m.outcome for m in checks] == [WRONG, NOT_TARGET]
    assert checks[0].describe() == "1번째 약품 '게보린정': 정답 202106092, 결과 197900277 (잘못 정함)"


def test_a_missing_line_is_counted_per_line():
    order = order_from_answer("rx_002")
    order = order.model_copy(update={"items": order.items[:2]})
    assert [m.outcome for m in match_checks(order, load("cv_expected", "rx_002"))] == [CONFIRMED, CANDIDATE, MISSING]


def test_without_an_order_every_line_is_missing_and_keeps_the_answer_name():
    checks = match_checks(None, load("cv_expected", "rx_001"))
    assert [m.outcome for m in checks] == [MISSING, MISSING]
    assert checks[0].name == "타이레놀정 5oOmg" and checks[0].got is None
    assert checks[1].describe() == "2번째 약품 '알모란정': 정답 없음(대상 약 아님), 결과 정하지 않음 (줄 못 읽음)"


# ------------------------------------------------------------ 요약


def test_score_and_summarize():
    good = score_sample("rx_002", "rule", rule_order("rx_002"), load("cv_expected", "rx_002"), seconds=0.5)
    bad = score_sample("rx_001", "llm", None, load("cv_expected", "rx_001"), seconds=2.0, error="실패")
    assert (good.all_correct, good.flagged, good.line_count, good.ocr_lines_correct) == (True, 3, 3, 3)
    assert (bad.all_correct, bad.flagged, bad.ocr_lines_correct) == (False, 0, 0)

    s = summarize("rule", [good, bad])
    assert (s.samples, s.failed, s.all_correct) == (2, 1, 1)
    assert (s.ocr_correct, s.ocr_total) == (16, 27)
    assert (s.true_correct, s.true_total) == (13, 22)
    assert (s.lines_correct, s.lines, s.flagged, s.seconds) == (3, 5, 3, [0.5, 2.0])
    assert s.ocr_rate == 16 / 27 and s.true_rate == 13 / 22 and s.line_rate == 3 / 5
    # 칸 이름별: rx_002는 모두 맞고 rx_001은 모두 틀렸다
    assert (s.field_correct["name"], s.field_total["name"]) == (3, 5)
    assert (s.field_correct["issue_no"], s.field_total["issue_no"]) == (1, 2)
    assert s.field_rate("days") == 3 / 5 and s.field_rate("없는칸") == 0.0
    assert s.matches == {CONFIRMED: 1, CANDIDATE: 1, NOT_TARGET: 1, NOT_FOUND: 0, WRONG: 0, MISSING: 2}
    assert s.wrong_matches == 0


def test_only_the_lines_with_the_pharmacist_check_mark_are_flagged():
    # 인쇄 코드가 사전의 품목기준코드와 같으면 이름을 보지 않고 정하고 약사 확인 표시도 붙이지 않는다 (4단계).
    # 그래서 그 줄을 뺀 나머지만 표시된다. 샘플의 가상 코드(9000000xx)는 모두 이름 단계라 표시가 전부 붙는다
    expected = load("cv_expected", "rx_001")
    by_code = score_sample("rx_001", "llm", order_from_answer("rx_001", {(0, "printed_code"): "202106092"}), expected, seconds=0)
    assert by_code.flagged == 1 and by_code.match_checks[0].outcome == CONFIRMED
    by_name = score_sample("rx_001", "rule", rule_order("rx_001"), expected, seconds=0)
    assert by_name.flagged == 2


def test_wrong_matches_are_counted_in_the_summary():
    r = score_sample("rx_001", "llm", order_from_answer("rx_001", {(0, "name"): "게보린정"}), load("cv_expected", "rx_001"), seconds=0)
    assert summarize("llm", [r]).wrong_matches == 1


def test_an_order_with_every_cell_right_but_an_error_is_not_all_correct():
    r = score_sample("rx_002", "rule", rule_order("rx_002"), load("cv_expected", "rx_002"), seconds=0, error="x")
    assert not r.all_correct


def test_empty_summary_rates_are_zero():
    s = summarize("rule", [])
    assert (s.ocr_rate, s.true_rate, s.line_rate) == (0.0, 0.0, 0.0)
    assert not s.meets_target and s.matches == dict.fromkeys(MATCH_OUTCOMES, 0)


@pytest.mark.parametrize("correct, total, meets", [(9, 10, True), (8, 10, False), (90, 100, True), (89, 100, False), (80, 80, True)])
def test_the_target_is_ninety_percent_or_more(correct, total, meets):
    assert Summary("rule", ocr_correct=correct, ocr_total=total).meets_target is meets


def test_the_rule_path_meets_the_target_on_the_five_samples():
    results = [score_sample(s, "rule", rule_order(s), load("cv_expected", s), seconds=0) for s in SAMPLE_IDS]
    summary = summarize("rule", results)
    assert (summary.ocr_correct, summary.ocr_total) == (80, 80) and summary.meets_target
    assert (summary.true_correct, summary.true_total) == (64, 65)
    assert summary.flagged == 15 and summary.wrong_matches == 0
