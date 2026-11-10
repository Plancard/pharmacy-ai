"""표 본문의 범위 찾기 (find_table). 머리글 낱말 아래 ~ '주사제 처방명세' 위."""
import json
import math
import re
from pathlib import Path

import pytest

from agent.config import RESOURCES_DIR
from agent.structure import StructureError, find_table

DATA = RESOURCES_DIR / "samples"   # CV 샘플은 cv_a1/, 우리 정답은 cv_expected/ 에 동결해 둔다


def _plain(ocr):
    """CV 샘플에서 line_id를 뗀다. 이 파일의 시험은 "line_id가 없는 OCR은 lines 안의 순서가 줄 번호"라는 전제로 쓰였다.
    line_id가 붙은 실제 A.1은 test_structure_a1.py가 시험한다."""
    for line in ocr["lines"]:
        line.pop("line_id", None)
    return ocr
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]


def load_ocr(sample_id):
    return _plain(json.loads((DATA / "cv_a1" / f"{sample_id}.json").read_text(encoding="utf-8")))


def box(text, y1, y2=None, x1=100, x2=300):
    return {"text": text, "bbox": [x1, y1, x2, y2 if y2 is not None else y1 + 40], "conf": 0.9}


# ------------------------------------------------------------ 실제 샘플


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_table_range_of_the_real_samples(sample_id):
    top, bottom = find_table(load_ocr(sample_id)["lines"])
    assert top == 1208        # '횟수'·'일수' 칸의 아래쪽 (가장 아래에 있는 머리글 낱말)
    assert bottom == 2793     # '주사제 처방명세' 칸의 위쪽


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_every_drug_line_is_inside_the_range(sample_id):
    ocr = load_ocr(sample_id)
    top, bottom = find_table(ocr["lines"])
    drug_lines = [l for l in ocr["lines"] if re.match(r"^(\(\d+\))?\d{9}\s", l["text"])]   # "(1)900000011 타이레놀정"
    assert drug_lines and all(top < l["bbox"][1] and l["bbox"][3] < bottom for l in drug_lines)


def test_the_misread_dose_header_does_not_matter():
    # rx_002는 '투약량'을 '투악량'으로 읽었다. 나머지 낱말(횟수·일수·구분코드 …)로 충분하다.
    ocr = load_ocr("rx_002")
    assert "투악량" in [l["text"] for l in ocr["lines"]] and "투약량" not in [l["text"] for l in ocr["lines"]]
    assert find_table(ocr["lines"])[0] == 1208


def test_patient_number_above_the_table_is_not_inside_it():
    ocr = load_ocr("rx_001")
    top, _ = find_table(ocr["lines"])
    patient = next(l for l in ocr["lines"] if "000000101" in l["text"])
    assert patient["bbox"][3] < top


# ------------------------------------------------------------ 머리글 낱말


def test_two_header_words_are_enough():
    top, _ = find_table([box("횟수", 100, 140), box("일수", 110, 150)])
    assert top == 150          # 가장 아래에 있는 낱말의 아래쪽


def test_header_words_split_by_spaces_are_found():
    top, _ = find_table([box("횟 수", 100, 140), box("구분 코드", 110, 150)])
    assert top == 150


def test_the_long_name_header_counts_as_a_word():
    top, _ = find_table([box("처방 의약품의 명칭", 100, 160), box("횟수", 100, 140)])
    assert top == 160


def test_one_header_word_is_not_enough():
    with pytest.raises(StructureError, match="머리글"):
        find_table([box("횟수", 100)])


def test_no_header_word_fails():
    with pytest.raises(StructureError, match="머리글"):
        find_table([box("알마겔정", 100), box("3회", 150)])


def test_the_same_word_twice_counts_once():
    with pytest.raises(StructureError, match="머리글"):
        find_table([box("횟수", 100), box("횟수", 200)])


def test_header_word_inside_a_long_sentence_is_not_a_header():
    # 긴 문장 속에 우연히 들어 있는 낱말은 머리글이 아니다
    with pytest.raises(StructureError, match="머리글"):
        find_table([box("총 처방 일수를 확인하세요", 100), box("횟수를 센다", 150)])


def test_a_word_with_stray_characters_is_still_a_header_up_to_two_of_them():
    # OCR이 낱말 옆에 점이나 기호를 붙여 읽어도 머리글이다 (붙은 글자 2개까지)
    assert find_table([box("횟수:", 100, 140), box("|일수.", 110, 150)])[0] == 150
    assert find_table([box("횟수::", 100, 140), box("일수::", 110, 150)])[0] == 150
    with pytest.raises(StructureError, match="머리글"):     # 3개부터는 문장으로 본다
        find_table([box("횟수:::", 100, 140), box("일수:::", 110, 150)])


# ------------------------------------------------------------ 표의 끝


def test_table_ends_where_the_injection_section_starts():
    lines = [box("횟수", 100, 140), box("일수", 100, 140), box("주사제 처방명세([ ]원 내 조제,", 900, 950)]
    assert find_table(lines) == (140, 900)


def test_the_end_marker_is_found_even_with_spaces_between_letters():
    lines = [box("횟수", 100, 140), box("일수", 100, 140), box("주 사 제 처방명세", 800, 850)]
    assert find_table(lines)[1] == 800


def test_the_highest_end_marker_wins():
    lines = [box("횟수", 100, 140), box("일수", 100, 140), box("주사제 처방명세", 900), box("주사제", 800)]
    assert find_table(lines)[1] == 800


def test_without_an_end_marker_the_table_has_no_end():
    assert find_table([box("횟수", 100, 140), box("일수", 100, 140)]) == (140, math.inf)


def test_header_words_below_the_end_marker_are_ignored():
    # 표 아래쪽에 '횟수' '일수'가 또 나와도 그것을 머리글로 오해하지 않는다
    lines = [box("주사제 처방명세", 500), box("횟수", 800), box("일수", 850)]
    with pytest.raises(StructureError, match="머리글"):
        find_table(lines)


def test_a_header_word_centered_exactly_on_the_end_marker_line_is_outside_the_table():
    # 마커 칸의 위쪽 y가 500이면, 중심이 500인 칸(480~520)은 표 밖이다
    lines = [box("주사제 처방명세", 500), box("횟수", 480, 520), box("일수", 480, 520)]
    with pytest.raises(StructureError, match="머리글"):
        find_table(lines)
    # 중심이 499인 칸이면 표 안이다
    assert find_table([box("주사제 처방명세", 500), box("횟수", 479, 519), box("일수", 479, 519)]) == (519, 500)


def test_input_order_does_not_matter():
    lines = load_ocr("rx_004")["lines"]
    assert find_table(list(reversed(lines))) == find_table(lines)
