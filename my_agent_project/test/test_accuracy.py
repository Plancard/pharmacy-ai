"""정확도 계산 (agent/accuracy.py)."""
import json

import pytest

from agent.accuracy import (
    OCR_FIELDS,
    TRUE_FIELDS,
    Check,
    compare,
    lines_all_correct,
    name_misread_lines,
    score_sample,
    summarize,
)
from agent.config import DATA_DIR
from agent.drug_dictionary import load_dictionary
from agent.drug_match import match_order
from agent.structure import extract_rule
from agent.validate import order_from_extracted

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary()


def load(kind, sample_id):
    return json.loads((DATA_DIR / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def rule_order(sample_id):
    return match_order(order_from_extracted("P1", extract_rule(load("cv_samples", sample_id)), source="rule"), DICT)


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


# ------------------------------------------------------------ compare


def test_fields_compared():
    assert OCR_FIELDS == ("printed_code", "name", "count_per_dose", "times_per_day", "days", "item_code")
    assert TRUE_FIELDS == ("printed_code", "count_per_dose", "times_per_day", "days", "item_code")


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_the_rule_path_matches_every_cell_of_the_ocr_answer(sample_id):
    checks = compare(rule_order(sample_id), load("cv_expected", sample_id))
    items = len(load("cv_expected", sample_id)["items"])
    assert len(checks) == 1 + items * len(OCR_FIELDS)        # 교부번호 + 약품 줄마다 6칸
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


def test_a_wrong_code_also_makes_the_item_code_wrong():
    checks = compare(order_from_answer("rx_001", {(0, "printed_code"): "900000021"}), load("cv_expected", "rx_001"))
    assert wrong(checks) == [(0, "printed_code"), (0, "item_code")]


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
    assert wrong(checks) == [(2, f) for f in OCR_FIELDS]
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
    assert len(checks) == 1 + 2 * len(OCR_FIELDS) and not any(c.ok for c in checks)
    assert all(c.got is None for c in checks)


def test_a_wrong_answer_that_happens_to_equal_none_is_still_wrong_without_an_order():
    # 정답의 item_code가 None인 줄(사전에 없는 약)도, 조제지시서가 없으면 맞은 것이 아니다
    checks = compare(None, load("cv_expected", "rx_001"))
    item = next(c for c in checks if c.line == 1 and c.field == "item_code")
    assert item.want is None and item.got is None and not item.ok


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


def test_score_and_summarize():
    good = score_sample("rx_002", "rule", rule_order("rx_002"), load("cv_expected", "rx_002"), seconds=0.5)
    bad = score_sample("rx_001", "llm", None, load("cv_expected", "rx_001"), seconds=2.0, error="실패")
    assert (good.all_correct, good.flagged, good.line_count, good.ocr_lines_correct) == (True, 1, 3, 3)
    assert (bad.all_correct, bad.flagged, bad.ocr_lines_correct) == (False, 0, 0)

    s = summarize("rule", [good, bad])
    assert (s.samples, s.failed, s.all_correct) == (2, 1, 1)
    assert (s.ocr_correct, s.ocr_total) == (19, 32)
    assert (s.true_correct, s.true_total) == (16, 27)
    assert (s.lines_correct, s.lines, s.flagged, s.seconds) == (3, 5, 1, [0.5, 2.0])
    assert s.ocr_rate == 19 / 32 and s.true_rate == 16 / 27 and s.line_rate == 3 / 5


def test_an_order_with_every_cell_right_but_an_error_is_not_all_correct():
    r = score_sample("rx_002", "rule", rule_order("rx_002"), load("cv_expected", "rx_002"), seconds=0, error="x")
    assert not r.all_correct


def test_empty_summary_rates_are_zero():
    s = summarize("rule", [])
    assert (s.ocr_rate, s.true_rate, s.line_rate) == (0.0, 0.0, 0.0)
