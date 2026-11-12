"""LLM에 보낼 텍스트 만들기 (llm_input, format_llm_text, ocr_to_llm_text).

규칙 경로가 실패했을 때 LLM에 보내는 것은 교부번호와 약품 표 본문뿐이다.
처방전 위쪽의 환자·의료인 정보는 어떤 경우에도 보내지 않는다.
"""
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.structure import (
    ISSUE_NOT_FOUND,
    StructureError,
    format_llm_text,
    llm_input,
    ocr_to_llm_text,
    ocr_to_text,
    table_end,
)

DATA = RESOURCES_DIR / "samples"   # CV 샘플은 cv_a1/, 우리 정답은 cv_expected/ 에 동결해 둔다


def _plain(ocr):
    """CV 샘플에서 line_id를 뗀다. 이 파일의 시험은 "line_id가 없는 OCR은 lines 안의 순서가 줄 번호"라는 전제로 쓰였다.
    line_id가 붙은 실제 A.1은 test_structure_a1.py가 시험한다."""
    for line in ocr["lines"]:
        line.pop("line_id", None)
    return ocr
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
HEADER_TEXTS = ("횟수", "일수", "구분코드", "투약량", "투악량", "처방의약품의명칭")

# 처방전 위쪽(표 위)에 있는 환자·의료인 정보 칸의 글자 일부
PRIVATE_TEXTS = [
    "000000101", "000000102", "000000103", "000000104", "000000105",       # 환자 번호
    "김가상", "박샘플", "정데모", "윤시험", "강가명",                            # 환자 이름
    "900101-1", "02-000-0000", "02-000-0001", "hosp",                         # 주민등록번호, 전화·팩스, 홈페이지
    "가상의료재단", "이테스트", "오가상", "면허", "성   명", "주민등록번호",       # 의료기관, 의사, 칸 이름
]


def load_ocr(sample_id):
    return _plain(json.loads((DATA / "cv_a1" / f"{sample_id}.json").read_text(encoding="utf-8")))


def load_expected(sample_id):
    return json.loads((DATA / "cv_expected" / f"{sample_id}.json").read_text(encoding="utf-8"))


def without_header(ocr):
    for line in ocr["lines"]:
        if "".join(line["text"].split()) in HEADER_TEXTS:
            line["text"] = "?"
    return ocr


def box(text, y1, y2=None, x1=100, x2=300):
    return {"text": text, "bbox": [x1, y1, x2, y2 if y2 is not None else y1 + 40], "conf": 0.9}


# ------------------------------------------------------------ 실제 샘플


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
@pytest.mark.parametrize("header", [True, False], ids=["with_header", "without_header"])
def test_only_the_issue_number_and_the_drug_rows_are_sent(sample_id, header):
    ocr = load_ocr(sample_id) if header else without_header(load_ocr(sample_id))
    issue_no, rows = llm_input(ocr)
    expected = load_expected(sample_id)
    assert issue_no == expected["issue_no"]
    assert len(rows) == len(expected["items"])                    # 약품 한 줄 = 텍스트 한 줄
    for row, drug in zip(rows, expected["items"]):
        assert drug["printed_code"] in row and "PO 식후 30분" in row


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
@pytest.mark.parametrize("header", [True, False], ids=["with_header", "without_header"])
def test_no_patient_or_doctor_information_is_in_the_text(sample_id, header):
    ocr = load_ocr(sample_id) if header else without_header(load_ocr(sample_id))
    text = ocr_to_llm_text(ocr)
    for private in PRIVATE_TEXTS:
        assert private not in text


def test_the_text_of_rx_001():
    assert ocr_to_llm_text(load_ocr("rx_001")) == (
        "교부번호: 2026년11월03일 00012호\n"
        "약품 표:\n"
        "(1)900000011 타이레놀정/ 5oOmg 1 TAB 3회 3일 PO 식후 30분\n"
        "900000031 알모란정 1 TAB 3회 3일 PO 식후 30분"
    )


def test_a_split_name_piece_stays_on_its_row():
    rows = llm_input(load_ocr("rx_005"))[1]
    assert rows[0] == "(1)671805860 필루비서방정 145mg 1 TAB 2회 3일 PO 식후 30분"


def test_nothing_below_the_table_is_sent():
    text = ocr_to_llm_text(load_ocr("rx_001"))
    for below in ("주사제", "사용기간", "출력시간", "조제약사"):
        assert below not in text


# ------------------------------------------------------------ 머리글이 없을 때


def test_without_a_header_the_text_starts_at_the_first_code_line():
    lines = [
        box("김가상(000000101)", 100),          # 환자 번호: 9자리지만 약품 코드가 아니다
        box("02-000-0000", 160),
        box("2026년11월03일 00012호", 40),
        box("900000011 타이레놀정", 300),
        box("3회", 300, x1=400, x2=450),
        box("900000021 게보린정", 360),
    ]
    assert llm_input({"lines": lines}) == (
        "2026년11월03일 00012호", ["900000011 타이레놀정 3회", "900000021 게보린정"],
    )


def test_without_a_header_a_patient_number_without_parentheses_is_not_a_starting_point():
    # OCR이 괄호를 빠뜨려도 같은 행에 값 칸(3회 등)이 없으니 약품 줄이 아니다
    lines = [
        box("김가상 000000101", 100),
        box("02-000-0000", 160),
        box("900000011 타이레놀정", 300),
        box("3회", 300, x1=400, x2=450),
    ]
    assert llm_input({"lines": lines}) == (None, ["900000011 타이레놀정 3회"])


def test_without_a_header_or_value_cells_a_line_that_starts_with_a_code_and_a_name_is_still_sent():
    # 11/12 촬영본 점검: 흐린 사진에서 머리글도 값 칸도 같은 행에 못 잡으면 LLM을 부르지도 못했다 (어려운 단계 5장).
    # '(순번) 9자리 코드 + 약 이름'으로 시작하는 줄은 약품 줄이다. LLM이 값을 못 읽으면 0을 내고 검증이 거절한다 (추측으로 채우지 않는다).
    lines = [box("김가상 000000101", 100), box("(1)900000011 타이레놀정", 300), box("900000021 게보린정", 360)]
    assert llm_input({"lines": lines}) == (None, ["(1)900000011 타이레놀정", "900000021 게보린정"])


@pytest.mark.parametrize("text", ["900000011", "김가상 000000101", "000000101 (김가상)", "전화 900000011 타이레놀정"])
def test_without_a_header_a_number_that_is_not_followed_by_a_name_or_does_not_start_the_line_is_not_a_drug_line(text):
    # 9자리 숫자만 있는 칸은 괄호가 빠진 환자 번호일 수 있다. 환자 줄은 이름으로 시작한다. 이런 줄부터는 보내지 않는다.
    assert llm_input({"lines": [box(text, 300)]}) == (None, [])


@pytest.mark.parametrize(
    "value_box, starts",
    [
        ((328, 368), True),       # 코드 줄(300~340)과 12 겹침 = 높이 40의 30%: 같은 행
        ((329, 369), False),      # 11 겹침: 다른 행
    ],
)
def test_without_a_header_the_value_cell_must_overlap_the_code_line_by_thirty_percent(value_box, starts):
    lines = [box("900000011", 300), box("3회", *value_box, x1=400, x2=450)]      # 코드만 있는 칸: 값 칸이 같은 행에 있어야 약품 줄이다
    assert bool(llm_input({"lines": lines})[1]) is starts


def test_without_a_header_a_cell_slightly_above_the_first_code_line_is_kept():
    # 첫 코드 줄이 300~340(높이 40)이면 중심이 280(=300-20)보다 아래인 칸부터 보낸다
    lines = [box("900000011 가정", 300), box("3회", 300, x1=500, x2=550), box("1 TAB", 265, 305, x1=400, x2=450)]  # 중심 285
    assert "1 TAB" in " ".join(llm_input({"lines": lines})[1])
    lines = [box("900000011 가정", 300), box("3회", 300, x1=500, x2=550), box("1 TAB", 260, 300, x1=400, x2=450)]  # 중심 280
    assert "1 TAB" not in " ".join(llm_input({"lines": lines})[1])


def test_without_a_header_a_patient_number_next_to_a_value_like_cell_is_not_a_starting_point():
    # 실제 양식에는 '1회 1일 본인' 같은 칸이 있다. 환자 번호 옆에 그런 칸이 있어도 환자 줄부터 보내면 안 된다.
    lines = [
        box("김가상(000000101)", 100),
        box("1일", 100, x1=400, x2=450),
        box("02-000-0000", 160),
        box("900000011 타이레놀정", 300),
        box("3회", 300, x1=400, x2=450),
    ]
    assert llm_input({"lines": lines}) == (None, ["900000011 타이레놀정 3회"])


def test_without_a_header_a_patient_number_alone_is_not_a_starting_point():
    lines = [box("김가상(000000101)", 100), box("알마겔정 3회", 300)]
    assert llm_input({"lines": lines}) == (None, [])


def test_without_a_header_code_lines_below_the_end_marker_are_ignored():
    lines = [box("주사제 처방명세", 200), box("900000011 타이레놀정", 300), box("3회", 300, x1=400, x2=450)]
    assert llm_input({"lines": lines}) == (None, [])


def test_without_a_header_the_end_marker_still_cuts_the_table():
    lines = [box("900000011 타이레놀정", 100), box("3회", 100, x1=400, x2=450),
             box("주사제 처방명세", 200), box("조제약사 성명", 300)]
    assert llm_input({"lines": lines}) == (None, ["900000011 타이레놀정 3회"])


def test_without_a_header_the_issue_number_is_read_only_above_the_first_code_line():
    lines = [box("900000011 타이레놀정", 100), box("3회", 100, x1=400, x2=450), box("2026년11월03일 00012호", 300)]
    issue_no, rows = llm_input({"lines": lines})
    assert issue_no is None and rows == ["900000011 타이레놀정 3회", "2026년11월03일 00012호"]


def test_blank_cells_are_not_sent():
    lines = [box("900000011 가정", 300), box("3회", 300, x1=500, x2=550), box("   ", 300, x1=400, x2=450)]
    assert llm_input({"lines": lines})[1] == ["900000011 가정 3회"]


# ------------------------------------------------------------ 형식


def test_format_llm_text():
    assert format_llm_text("2026년11월03일 00012호", ["a", "b"]) == "교부번호: 2026년11월03일 00012호\n약품 표:\na\nb"
    assert format_llm_text(None, []) == f"교부번호: {ISSUE_NOT_FOUND}\n약품 표:"
    assert format_llm_text("", ["a"]) == f"교부번호: {ISSUE_NOT_FOUND}\n약품 표:\na"


@pytest.mark.parametrize("bad", [None, {}, {"lines": []}, {"lines": [{"text": "a"}]}])
def test_malformed_ocr_is_refused(bad):
    with pytest.raises(StructureError):
        llm_input(bad)


def test_table_end():
    import math

    assert table_end([box("주사제 처방명세", 500), box("주 사 제", 400), box("가", 100)]) == 400
    assert table_end([box("가", 100)]) == math.inf


# ------------------------------------------------------------ ocr_to_text (사람이 훑어보는 전체 텍스트)


def test_ocr_to_text_writes_one_line_per_row():
    rows = ocr_to_text(load_ocr("rx_001")).split("\n")
    assert rows[0] == "[별지 제9호 서식]"
    assert "(1)900000011 타이레놀정/ 5oOmg 1 TAB 3회 3일 PO 식후 30분" in rows
    assert "900000031 알모란정 1 TAB 3회 3일 PO 식후 30분" in rows
    assert any("김가상(000000101)" in r for r in rows)     # 전체 텍스트에는 환자 정보도 있다. 그래서 LLM에는 쓰지 않는다


def test_ocr_to_text_rejects_malformed_ocr():
    with pytest.raises(StructureError):
        ocr_to_text({"lines": []})
