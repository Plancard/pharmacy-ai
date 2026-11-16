"""LLM에 보내기 직전의 개인정보 가리기 (agent/redact.py, 11/13 두 번째 안전장치).

첫 번째 안전장치(structure.llm_input: 약품 표 범위만 보냄)가 표 범위를 잘못 잡아도 주민등록번호 · 전화번호 · 이름이 새지 않는지 본다.
시험에 쓰는 번호는 모두 지어낸 것이다.
"""
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.pipeline import REDACTED_WARNING, structure_prescription
from agent.redact import MASK, is_personal_row, mask_text, redact_issue_no, redact_rows
from agent.structure import llm_input, ocr_to_llm_text, prepare_llm_input

SAMPLES = RESOURCES_DIR / "samples"
ALL_SAMPLES = [p for folder in ("rx8_a1", "cv_a1", "a1_ocr") for p in sorted((SAMPLES / folder).glob("*.json"))
               if "lines" in json.loads(p.read_text(encoding="utf-8"))]      # summary.json 같은 OCR 결과가 아닌 파일은 뺀다


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------ 모양으로 가리기


@pytest.mark.parametrize("text", [
    "900101-1234567", "900101 - 2234567", "9001011234567", "900101—1234567", "900101~1234567",
    "900101-1******", "９００１０１－１２３４５６７",      # 뒷자리를 별표로 가려 인쇄 · 전각 숫자
    "000101-3234567", "850505-5123456",                     # 2000년대생 · 외국인등록번호
])
def test_resident_registration_numbers_are_masked(text):
    masked, n = mask_text(f"앞 {text} 뒤")
    assert (masked, n) == (f"앞 {MASK} 뒤", 1)


@pytest.mark.parametrize("text", ["010-1234-5678", "02-123-4567", "031)123-4567", "010.1234.5678", "02 - 000 - 0000"])
def test_phone_numbers_are_masked(text):
    assert mask_text(text) == (MASK, 1)


def test_a_patient_name_with_its_number_is_masked():
    assert mask_text("김가상(000000101) 900000011 타이레놀정") == (f"{MASK} 900000011 타이레놀정", 1)


@pytest.mark.parametrize("text", [
    "(1)201110120 트리스펜연질캡슐 1 CAP 3회 3일 PO 식후 30분",     # 약품 코드 9자리와 값 칸
    "200713170 아이월드요비신정/416.7mg 1 TAB 3회 1일",
    "900000011 타이레놀정 5oOmg 1 TAB 3회 3일",
    "2026년11월11일 00012호",                                     # 교부번호
    "200308774 뮤코다인캡슐 200308775 게보린정",                     # 코드 두 개가 한 줄에
    "123456 7",                                                   # 6자리 + 한 자리
])
def test_drug_rows_are_not_touched(text):
    assert mask_text(text) == (text, 0)


# ------------------------------------------------------------ 칸 이름으로 줄 빼기


@pytest.mark.parametrize("row", [
    "성명 김가상", "주민등록번호 [가림]", "주 민 등 록 번 호", "환자 전화번호", "의료기관 명칭 가상의원", "면허번호 12345",
    "요양기관기호 12345678", "주소 서울시",
])
def test_rows_with_a_personal_field_name_are_dropped(row):
    assert is_personal_row(row)
    assert redact_rows([row, "900000011 타이레놀정 3회"]).rows == ["900000011 타이레놀정 3회"]


def test_redaction_counts_what_it_did():
    done = redact_rows(["성명 김가상 900101-1234567", "900000011 타이레놀정 010-1234-5678", "900000021 게보린정"])
    assert (done.rows, done.masked, done.dropped, done.changed) == (
        [f"900000011 타이레놀정 {MASK}", "900000021 게보린정"], 1, 1, True)
    assert redact_rows(["900000021 게보린정"]).changed is False


def test_an_issue_number_with_personal_data_is_masked_or_dropped():
    assert redact_issue_no("2026년11월03일 00012호") == ("2026년11월03일 00012호", 0)
    assert redact_issue_no(None) == (None, 0)
    assert redact_issue_no("2026년11월03일 900101-1234567") == (f"2026년11월03일 {MASK}", 1)
    assert redact_issue_no("성명 김가상") == (None, 1)


# ------------------------------------------------------------ LLM 입력 전체


@pytest.mark.parametrize("path", ALL_SAMPLES, ids=lambda p: p.stem)
def test_every_sample_is_sent_exactly_as_before(path):
    # 가상 처방전 · 실제 OCR 샘플의 약품 표에는 가릴 것이 없다: 보내는 글이 바뀌지 않는다
    assert prepare_llm_input(load(path))[2] == 0


def box(text, y, x1=100, x2=None):
    return {"text": text, "bbox": [x1, y, x2 or x1 + 300, y + 40], "conf": 0.9}


def test_personal_data_that_slipped_into_the_table_is_not_sent():
    # 머리글을 못 찾고, 표 범위가 환자 칸 쪽으로 넓게 잡힌 경우 (흐린 사진)
    lines = [
        box("(1)900000011 타이레놀정", 300), box("3회", 300, x1=500, x2=550),
        box("주민등록번호", 360), box("900101-1234567", 360, x1=500, x2=900),
        box("900000021 게보린정 010-0000-0000", 420),
        box("김가상(000000101)", 480),
    ]
    issue_no, rows, changed = prepare_llm_input({"lines": lines})
    text = "\n".join(rows)
    assert "1234567" not in text and "010-0000-0000" not in text and "김가상" not in text and "주민등록" not in text
    assert rows == ["(1)900000011 타이레놀정 3회", f"900000021 게보린정 {MASK}", MASK]
    assert changed == 3
    assert llm_input({"lines": lines}) == (issue_no, rows)
    assert "1234567" not in ocr_to_llm_text({"lines": lines})       # 스크립트(try_llm)가 보내는 글도 같다


def test_the_structured_order_warns_when_something_was_masked():
    lines = [box("(1)900000011 타이레놀정 1 TAB", 300), box("900000021 게보린정 010-0000-0000", 360)]
    sent = []

    def llm(text, feedback):
        sent.append(text)
        return {"issue_no": None, "items": [
            {"name": "타이레놀정", "printed_code": "900000011", "count_per_dose": 1, "times_per_day": 3, "days": 3, "times": []}]}

    order = structure_prescription("P0001", {"lines": lines}, llm_fn=llm)
    assert "010-0000-0000" not in sent[0] and MASK in sent[0]
    assert REDACTED_WARNING in order.warnings


def test_no_warning_when_nothing_was_masked():
    lines = [box("(1)900000011 타이레놀정 1 TAB", 300), box("3회", 300, x1=500, x2=550)]
    order = structure_prescription("P0001", {"lines": lines}, llm_fn=lambda text, feedback: {"issue_no": None, "items": [
        {"name": "타이레놀정", "printed_code": "900000011", "count_per_dose": 1, "times_per_day": 3, "days": 3, "times": []}]})
    assert REDACTED_WARNING not in order.warnings


def test_when_only_personal_rows_were_found_nothing_is_sent():
    lines = [box("900000011 김가상 주민등록번호 900101-1234567", 300), box("3회", 300, x1=500, x2=550)]
    called = []
    from agent.errors import E_OCR_EMPTY, StructureError

    with pytest.raises(StructureError) as e:
        structure_prescription("P0001", {"lines": lines}, llm_fn=lambda *a: called.append(a))
    assert called == [] and e.value.code == E_OCR_EMPTY
