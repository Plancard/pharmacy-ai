"""규칙으로 읽기(structure.py → validate.py)를 A.1 샘플 전체에 돌려 A.5 정답과 맞춘다.

- 가상 처방전 s01~s05 (agent/resources/samples/a1_ocr): 정답은 a5_order
- CV가 보낸 실제 형식 샘플 rx_001~005 (agent/resources/samples/cv_a1): line_id가 붙은 A.1
"""
import copy
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.errors import StructureError
from agent.schema import DispensingOrder, OcrResult
from agent.structure import extract_rule, llm_input
from agent.validate import order_from_extracted

SAMPLES = RESOURCES_DIR / "samples"
RULE_OK = ["s01_combo_a", "s02_combo_b", "s03_no_risk", "s04_flagged"]
CV_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]


def load(kind: str, name: str) -> dict:
    return json.loads((SAMPLES / kind / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", RULE_OK)
def test_rule_path_reads_the_dummy_prescription_like_the_answer(name):
    ocr = load("a1_ocr", name)
    OcrResult.model_validate(ocr)                                  # 입구: A.1 형식
    order = order_from_extracted("P0001", extract_rule(ocr), source="rule")
    DispensingOrder.model_validate(order.model_dump())             # 출구: A.5 형식 (다시 만들어도 같다)
    want = load("a5_order", name)["items"]
    assert len(order.items) == len(want)
    for got, ans in zip(order.items, want):
        assert (got.count_per_dose, got.times, got.days) == (ans["count_per_dose"], ans["times"], ans["days"])
        assert got.source_line_ids == ans["source_line_ids"]       # 그 약을 읽은 줄 번호까지 정답과 같다
    assert order.issue_no == next(ln["text"] for ln in ocr["lines"] if ln["text"].startswith("2026년"))
    assert order.source == "rule"


@pytest.mark.parametrize("name", RULE_OK)
def test_rule_result_is_not_ready_for_the_web_yet(name):
    """약품 매칭(item_code)과 위험 신호 대조(risk_check)가 아직이다. 그래서 웹에 넘기는 검사는 통과하지 못한다."""
    order = order_from_extracted("P0001", extract_rule(load("a1_ocr", name)), source="rule")
    assert order.all_matched is False
    assert order.risk_check is None
    with pytest.raises(ValueError):
        order.check_ready_for_web()


def test_rule_path_keeps_the_ocr_misreading_for_the_matching_step():
    """OCR이 '프론크논램술'로 읽은 이름은 여기서 고치지 않는다 (약품 매칭의 일)."""
    order = order_from_extracted("P0001", extract_rule(load("a1_ocr", "s04_flagged")), source="rule")
    assert order.items[0].name == "프론크논램술" and order.items[0].printed_code == "900000051"
    assert order.items[0].ocr_conf == 0.45                         # 낮은 신뢰도가 그대로 넘어간다
    assert order.items[1].times == ["bedtime"]                     # 용법의 "취침 전"


def test_rule_path_gives_up_when_no_drug_code_is_printed():
    """코드가 인쇄되지 않은 처방전은 약품 줄을 알아볼 수 없다 → 3단계에서 LLM이 읽는다."""
    with pytest.raises(StructureError, match="약품 줄"):
        extract_rule(load("a1_ocr", "s05_no_code"))


def test_the_llm_path_can_still_be_given_the_table_of_a_codeless_prescription():
    issue_no, rows = llm_input(load("a1_ocr", "s05_no_code"))
    assert issue_no == "2026년11월09일 00065호"
    assert any("타이레놀" in r for r in rows) and any("뮤코다인" in r for r in rows)
    assert not any("테스트병원" in r or "별지" in r for r in rows)   # 표 밖의 글자는 LLM에 가지 않는다


@pytest.mark.parametrize("name", ["s01_combo_a", "s05_no_code"])
def test_llm_never_sees_anything_outside_the_table(name):
    text = " ".join(llm_input(load("a1_ocr", name))[1])
    for outside in ("가상의료재단", "약국제출용", "OCR 테스트용"):
        assert outside not in text


# ------------------------------------------------------------------ CV가 보낸 실제 형식 (line_id 포함)

@pytest.mark.parametrize("sample_id", CV_IDS)
def test_cv_sample_is_a1_and_has_line_ids(sample_id):
    ocr = OcrResult.model_validate(load("cv_a1", sample_id))
    assert [ln.line_id for ln in ocr.lines] == list(range(len(ocr.lines)))


@pytest.mark.parametrize("sample_id", CV_IDS)
def test_source_line_ids_follow_the_line_id_of_the_ocr(sample_id):
    ocr = load("cv_a1", sample_id)
    text = {ln["line_id"]: ln["text"] for ln in ocr["lines"]}
    for item in extract_rule(ocr)["items"]:
        assert item["source_line_ids"] and set(item["source_line_ids"]) <= set(text)
        assert any(item["printed_code"] in text[i] for i in item["source_line_ids"])   # 코드가 인쇄된 약품 줄이 근거에 들어 있다


@pytest.mark.parametrize("sample_id", CV_IDS)
def test_line_ids_equal_list_order_for_a_sorted_a1(sample_id):
    """A.1의 lines는 정렬되어 있고 line_id가 순서와 같다. line_id가 없던 옛 OCR과 읽은 결과가 똑같다."""
    ocr = load("cv_a1", sample_id)
    bare = copy.deepcopy(ocr)
    for ln in bare["lines"]:
        del ln["line_id"]
    assert extract_rule(ocr) == extract_rule(bare)


@pytest.mark.parametrize("sample_id", CV_IDS)
def test_cv_sample_matches_our_expected_answer(sample_id):
    got = extract_rule(load("cv_a1", sample_id))
    want = load("cv_expected", sample_id)
    assert got["issue_no"] == want["issue_no"]
    assert [(d["printed_code"], d["name"], d["count_per_dose"], d["times_per_day"], d["days"]) for d in got["items"]] == [
        (w["printed_code"], w["name"], w["count_per_dose"], w["times_per_day"], w["days"]) for w in want["items"]
    ]
