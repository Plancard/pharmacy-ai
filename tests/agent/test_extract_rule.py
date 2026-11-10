"""규칙 기반 구조화 (extract_rule). 실제 형식 샘플 5건을 읽고, 샘플을 고쳐서 실패 사례도 만든다."""
import copy
import json
import random
import re
from pathlib import Path

import pytest

from agent.config import RESOURCES_DIR
from agent.structure import StructureError, extract_rule, find_issue_no

DATA = RESOURCES_DIR / "samples"   # CV 샘플은 cv_a1/, 우리 정답은 cv_expected/ 에 동결해 둔다


def _plain(ocr):
    """CV 샘플에서 line_id를 뗀다. 이 파일의 시험은 "line_id가 없는 OCR은 lines 안의 순서가 줄 번호"라는 전제로 쓰였다.
    line_id가 붙은 실제 A.1은 test_structure_a1.py가 시험한다."""
    for line in ocr["lines"]:
        line.pop("line_id", None)
    return ocr
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DRUG_LINE = re.compile(r"^(\(\d+\))?\d{9}\s")      # "(1)900000011 타이레놀정/ 5oOmg"


def load_ocr(sample_id):
    return _plain(json.loads((DATA / "cv_a1" / f"{sample_id}.json").read_text(encoding="utf-8")))


def load_expected(sample_id):
    return json.loads((DATA / "cv_expected" / f"{sample_id}.json").read_text(encoding="utf-8"))


def center_y(line):
    return (line["bbox"][1] + line["bbox"][3]) / 2


def anchors_of(ocr):
    return sorted((l for l in ocr["lines"] if DRUG_LINE.match(l["text"])), key=center_y)


def cell(ocr, row, pattern):
    """row번째(0부터) 약품 줄의 칸 중 글자가 pattern과 통째로 맞는 것. 코드 줄 위치로 행을 고른다."""
    anchor_y = center_y(anchors_of(ocr)[row])
    found = [
        l for l in ocr["lines"]
        if anchor_y - 25 <= center_y(l) <= anchor_y + 45 and not DRUG_LINE.match(l["text"]) and re.fullmatch(pattern, l["text"])
    ]
    assert len(found) == 1, (row, pattern, found)
    return found[0]


def box(text, x1, y1, x2=None, y2=None):
    return {"text": text, "bbox": [x1, y1, x2 or x1 + 100, y2 or y1 + 40], "conf": 0.9}


def row_confs(ocr):
    """코드 줄마다, 같은 행(중심 y가 코드 줄 위 25 ~ 아래 45)에 있는 모든 칸의 최저 신뢰도. 파서와 따로 만든 검사용 구현."""
    return [
        min(l["conf"] for l in ocr["lines"] if center_y(a) - 25 <= center_y(l) <= center_y(a) + 45)
        for a in anchors_of(ocr)
    ]


def row_line_ids(ocr):
    """코드 줄마다, 같은 행(위와 같은 범위)에 있는 칸들의 lines 안 순서. 파서와 따로 만든 검사용 구현."""
    return [
        [i for i, l in enumerate(ocr["lines"]) if center_y(a) - 25 <= center_y(l) <= center_y(a) + 45]
        for a in anchors_of(ocr)
    ]


def with_line_ids(ocr, start=0):
    """A.1 v0.3처럼 줄마다 line_id를 붙인다 (lines 안 순서 + start)."""
    ocr = copy.deepcopy(ocr)
    for i, line in enumerate(ocr["lines"]):
        line["line_id"] = start + i
    return ocr


def expected_drugs(sample_id):
    confs = row_confs(load_ocr(sample_id))
    ids = row_line_ids(load_ocr(sample_id))
    return [
        {
            "name": d["name"],
            "printed_code": d["printed_code"],
            "count_per_dose": d["count_per_dose"],
            "times_per_day": d["times_per_day"],
            "days": d["days"],
            "times": [],          # 용법이 모두 'PO 식후 30분'이라 시점이 적혀 있지 않다
            "source_line_ids": line_ids,
            "ocr_conf": conf,
        }
        for d, conf, line_ids in zip(load_expected(sample_id)["items"], confs, ids)
    ]


# ------------------------------------------------------------ 실제 형식 샘플 5건


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_drugs_match_the_expected_answer(sample_id):
    assert extract_rule(load_ocr(sample_id))["items"] == expected_drugs(sample_id)


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_issue_no_matches_the_expected_answer(sample_id):
    assert extract_rule(load_ocr(sample_id))["issue_no"] == load_expected(sample_id)["issue_no"]


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_only_the_agreed_keys_are_returned(sample_id):
    result = extract_rule(load_ocr(sample_id))
    assert set(result) == {"issue_no", "items"}
    for drug in result["items"]:
        assert set(drug) == {
            "name", "printed_code", "count_per_dose", "times_per_day", "days", "times", "source_line_ids", "ocr_conf",
        }


# 약품 줄마다 그 줄을 읽은 칸들(코드 줄·이름 조각·값 칸·용법)의 최저 신뢰도. 눈으로 센 값이다.
LOWEST_CONF = {
    "rx_001": [0.43, 0.55],
    "rx_002": [0.57, 0.57, 0.57],
    "rx_003": [0.30, 0.55],
    "rx_004": [0.43, 0.80, 0.45, 0.30],
    "rx_005": [0.32, 0.29, 0.21, 0.24],        # 첫 줄은 코드 줄이 0.94지만 떨어져 나온 '145mg'이 0.32다
}


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_ocr_conf_is_the_lowest_confidence_among_the_cells_of_the_row(sample_id):
    assert [d["ocr_conf"] for d in extract_rule(load_ocr(sample_id))["items"]] == LOWEST_CONF[sample_id]


def test_a_low_confidence_cell_other_than_the_name_line_lowers_the_row_confidence():
    ocr = load_ocr("rx_002")
    cell(ocr, 1, "3회")["conf"] = 0.12
    assert [d["ocr_conf"] for d in extract_rule(ocr)["items"]] == [0.57, 0.12, 0.57]


def test_ocr_conf_is_none_when_no_cell_of_the_row_has_a_confidence():
    ocr = load_ocr("rx_001")
    for line in ocr["lines"]:
        del line["conf"]
    assert [d["ocr_conf"] for d in extract_rule(ocr)["items"]] == [None, None]


def test_cells_without_a_confidence_are_skipped_not_counted_as_zero():
    ocr = load_ocr("rx_001")
    del cell(ocr, 0, "3회")["conf"]
    assert extract_rule(ocr)["items"][0]["ocr_conf"] == 0.43


def test_a_non_numeric_confidence_is_ignored():
    ocr = load_ocr("rx_001")
    cell(ocr, 0, "3회")["conf"] = False           # bool은 숫자로 세지 않는다 (False를 0으로 세면 최저값이 된다)
    cell(ocr, 0, "3일")["conf"] = "0.01"
    assert extract_rule(ocr)["items"][0]["ocr_conf"] == 0.43


def test_the_name_is_the_ocr_text_not_a_corrected_one():
    # 약품명 고치기는 약품 매칭 단계의 일이다. 규칙 읽기는 OCR이 읽은 글자 그대로 넘긴다.
    names = [d["name"] for d in extract_rule(load_ocr("rx_001"))["items"]]
    assert names == ["타이레놀정 5oOmg", "알모란정"]


def test_a_name_split_into_two_boxes_is_joined():
    # rx_005 첫 약품: '(1)671805860 필루비서방정'과 같은 행에 떨어져 나온 '145mg'
    assert extract_rule(load_ocr("rx_005"))["items"][0]["name"] == "필루비서방정 145mg"


def test_the_patient_number_with_nine_digits_is_not_a_drug():
    ocr = load_ocr("rx_001")
    assert any("000000101" in l["text"] for l in ocr["lines"])       # 표 위에 9자리 숫자가 있다
    assert [d["printed_code"] for d in extract_rule(ocr)["items"]] == ["900000011", "900000031"]


def test_more_than_three_days_is_read_as_it_is():
    # 트레이는 3일이라 약사 확인이 필요하지만, 그 판단은 약품 매칭 단계에서 한다
    assert [d["days"] for d in extract_rule(load_ocr("rx_003"))["items"]] == [4, 4]


# ------------------------------------------------------------ 순서·원본


@pytest.mark.parametrize("sample_id", ["rx_002", "rx_005"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_input_order_does_not_matter(sample_id, seed):
    ocr = with_line_ids(load_ocr(sample_id))        # CV가 붙인 line_id는 순서가 바뀌어도 그대로다
    shuffled = dict(ocr, lines=random.Random(seed).sample(ocr["lines"], len(ocr["lines"])))
    assert extract_rule(shuffled) == extract_rule(ocr)


def test_sorting_by_x_instead_of_y_gives_the_same_result():
    ocr = with_line_ids(load_ocr("rx_004"))
    by_x = dict(ocr, lines=sorted(ocr["lines"], key=lambda l: l["bbox"][0]))
    assert extract_rule(by_x) == extract_rule(ocr)


def test_the_input_is_not_modified():
    ocr = load_ocr("rx_005")
    before = copy.deepcopy(ocr)
    extract_rule(ocr)
    assert ocr == before


# ------------------------------------------------------------ 샘플을 고쳐 만든 정상 사례


def test_times_written_in_the_usage_are_read():
    ocr = load_ocr("rx_001")
    cell(ocr, 0, "PO 식후 30분")["text"] = "아침 점심 저녁 취침 전"
    cell(ocr, 0, "3회")["text"] = "4회"
    drug = extract_rule(ocr)["items"][0]
    assert drug["times_per_day"] == 4 and drug["times"] == ["morning", "noon", "evening", "bedtime"]
    assert extract_rule(ocr)["items"][1]["times"] == []          # 다른 줄은 그대로


def test_full_width_digits_are_read():
    ocr = load_ocr("rx_001")
    cell(ocr, 0, "3회")["text"] = "３회"
    cell(ocr, 0, "1 TAB")["text"] = "２ TAB"
    drug = extract_rule(ocr)["items"][0]
    assert (drug["times_per_day"], drug["count_per_dose"]) == (3, 2)


def test_capsule_dose_is_read_like_a_tablet():
    drug = extract_rule(load_ocr("rx_002"))["items"][1]
    assert drug["name"] == "프론크논캠술" and drug["count_per_dose"] == 1


def test_more_name_pieces_are_joined_left_to_right():
    ocr = load_ocr("rx_001")
    anchor = anchors_of(ocr)[0]
    y1, y2 = anchor["bbox"][1], anchor["bbox"][3]
    ocr["lines"] += [box("뒤", 900, y1, 950, y2), box("앞", 820, y1, 870, y2)]
    assert extract_rule(ocr)["items"][0]["name"] == "타이레놀정 5oOmg 앞 뒤"


def test_blank_text_boxes_inside_the_table_are_ignored():
    ocr = load_ocr("rx_001")
    ocr["lines"].append(box("   ", 1000, 1600))       # 어느 약품 줄과도 겹치지 않는 빈 칸
    assert extract_rule(ocr) == extract_rule(load_ocr("rx_001"))


def test_a_cell_between_the_dose_and_days_columns_is_not_a_name_piece():
    # 이름 조각은 '값 칸이 시작하는 가장 왼쪽 x'보다 왼쪽에 있는 글자뿐이다
    ocr = load_ocr("rx_001")
    anchor = anchors_of(ocr)[0]
    ocr["lines"].append(box("메모", 1420, anchor["bbox"][1], 1460, anchor["bbox"][3]))
    assert extract_rule(ocr)["items"][0]["name"] == "타이레놀정 5oOmg"


def test_name_pieces_are_split_by_x_not_by_y():
    # 값 칸이 x 800부터 시작하고 y는 205다. 기준이 x가 아니라 y로 잡히면 x 700의 조각이 이름에서 빠진다.
    lines = [
        box("횟수", 1000, 100), box("일수", 1100, 100),
        box("(1)900000011 타이레놀정", 100, 200, 600, 240), box("500mg", 650, 200, 750, 240),
        box("2 TAB", 800, 205), box("3회", 1000, 205), box("5일", 1100, 205),
    ]
    assert extract_rule({"lines": lines})["items"][0]["name"] == "타이레놀정 500mg"


def test_a_name_piece_with_a_usage_word_is_not_read_as_usage():
    ocr = load_ocr("rx_001")
    anchor = anchors_of(ocr)[0]
    ocr["lines"].append(box("아침", 820, anchor["bbox"][1], 870, anchor["bbox"][3]))
    drug = extract_rule(ocr)["items"][0]
    assert drug["name"] == "타이레놀정 5oOmg 아침" and drug["times"] == []


def test_full_width_characters_in_a_name_piece_are_normalized():
    ocr = load_ocr("rx_001")
    anchor = anchors_of(ocr)[0]
    ocr["lines"].append(box("５００ｍｇ", 820, anchor["bbox"][1], 1000, anchor["bbox"][3]))
    assert extract_rule(ocr)["items"][0]["name"] == "타이레놀정 5oOmg 500mg"


def test_a_full_width_code_is_returned_as_ascii_digits():
    ocr = load_ocr("rx_001")
    anchors_of(ocr)[0]["text"] = "(1)９００００００１１ 타이레놀정"
    drug = extract_rule(ocr)["items"][0]
    assert drug["printed_code"] == "900000011" and drug["name"] == "타이레놀정"


@pytest.mark.parametrize(
    "text, name",
    [
        ("(1)900000051 아르펜정", "아르펜정"),
        ("( 2 )900000051 아르펜정", "아르펜정"),
        ("900000051 아르펜정(200)mg", "아르펜정(200)mg"),        # 줄 앞이 아닌 (숫자)는 이름의 일부다
        ("(1)900000051 아르펜정 (2)", "아르펜정 (2)"),
        ("900000051/아르펜정/200mg", "아르펜정 200mg"),
        ("900000051 아르펜정 123456789", "아르펜정 123456789"),    # 코드는 첫 9자리 하나만 뺀다
    ],
)
def test_name_is_what_is_left_after_removing_the_row_number_and_the_code(text, name):
    ocr = load_ocr("rx_001")
    anchors_of(ocr)[0]["text"] = text
    assert extract_rule(ocr)["items"][0]["name"] == name


def test_a_line_centered_exactly_on_the_header_bottom_is_not_in_the_table():
    # 머리글 낱말의 아래쪽 y가 140이고 중심이 140인 칸(120~160)은 표 밖이다
    lines = [box("횟수", 100, 100, y2=140), box("일수", 100, 100, y2=140), box("(1)900000011 가", 100, 120, y2=160)]
    with pytest.raises(StructureError, match="약품 줄을 찾지"):
        extract_rule({"lines": lines})


def test_a_line_centered_exactly_on_the_end_marker_line_is_not_in_the_table():
    lines = [
        box("횟수", 1000, 100), box("일수", 1100, 100),
        box("(1)900000011 타이레놀정", 100, 200, 600, 240),
        box("2 TAB", 800, 205), box("3회", 1000, 205), box("5일", 1100, 205),
        box("주사제 처방명세", 100, 500),
        box("기타", 100, 480, y2=520),            # 중심 y 500 = 마커 칸의 위쪽 y
    ]
    assert extract_rule({"lines": lines})["items"][0]["name"] == "타이레놀정"


def test_a_nine_digit_number_below_the_table_is_ignored():
    ocr = load_ocr("rx_001")
    ocr["lines"].append(box("900000099", 300, 3000))       # '주사제 처방명세' 아래
    assert extract_rule(ocr) == extract_rule(load_ocr("rx_001"))


def test_two_header_words_are_enough_to_read_the_table():
    ocr = with_line_ids(load_ocr("rx_001"))
    ocr["lines"] = [l for l in ocr["lines"] if l["text"] not in ("횟수", "일수", "구분코드")]   # 투약량·명칭만 남는다
    assert extract_rule(ocr) == extract_rule(with_line_ids(load_ocr("rx_001")))


def test_a_small_prescription_made_by_hand():
    lines = [
        box("횟수", 1000, 100), box("일수", 1100, 100),
        box("(1)900000011 타이레놀정", 100, 200, 600, 240),
        box("2 TAB", 800, 205), box("3회", 1000, 205), box("5일", 1100, 205), box("아침 저녁", 1300, 205, 1500, 245),
        box("주사제 처방명세", 100, 400),
    ]
    assert extract_rule({"lines": lines}) == {
        "issue_no": None,
        "items": [{
            "name": "타이레놀정", "printed_code": "900000011", "count_per_dose": 2,
            "times_per_day": 3, "days": 5, "times": ["morning", "evening"], "source_line_ids": [2, 3, 4, 5, 6],
            "ocr_conf": 0.9,
        }],
    }


# ------------------------------------------------------------ 원문 줄 번호 (source_line_ids, A.5)

# 약품 줄마다 그 줄을 읽은 칸들의 lines 안 순서. rx_001을 눈으로 본 값이다:
# 1번째 줄 = 코드 줄 '(1)900000011 타이레놀정/ 5oOmg'와 '1 TAB' '3회' '3일' 'PO 식후 30분'
RX_001_LINES = [
    ["(1)900000011 타이레놀정/ 5oOmg", "1 TAB", "3회", "3일", "PO 식후 30분"],
    ["900000031 알모란정", "1 TAB", "3회", "3일", "PO 식후 30분"],
]


def test_source_line_ids_point_to_the_cells_of_each_row():
    ocr = load_ocr("rx_001")
    for drug, texts in zip(extract_rule(ocr)["items"], RX_001_LINES):
        assert sorted(ocr["lines"][i]["text"] for i in drug["source_line_ids"]) == sorted(texts)
        assert drug["source_line_ids"] == sorted(drug["source_line_ids"])


def test_a_split_name_piece_is_part_of_the_row_lines():
    ocr = load_ocr("rx_005")
    ids = extract_rule(ocr)["items"][0]["source_line_ids"]
    assert "145mg" in [ocr["lines"][i]["text"] for i in ids]


def test_line_ids_from_the_ocr_are_used_when_present():
    ocr = with_line_ids(load_ocr("rx_001"), start=100)
    plain = extract_rule(load_ocr("rx_001"))["items"]
    got = extract_rule(ocr)["items"]
    assert [d["source_line_ids"] for d in got] == [[100 + i for i in d["source_line_ids"]] for d in plain]


def test_line_ids_need_not_follow_the_list_order():
    ocr = with_line_ids(load_ocr("rx_002"))
    for line in ocr["lines"]:
        line["line_id"] = 1000 - line["line_id"]          # 거꾸로 매겨도 그 값을 그대로 쓴다
    ids = extract_rule(ocr)["items"][0]["source_line_ids"]
    texts = {l["line_id"]: l["text"] for l in ocr["lines"]}
    assert ids == sorted(ids) and "(1)900000021 게보린정" in [texts[i] for i in ids]


def test_line_ids_on_only_some_lines_are_refused():
    ocr = load_ocr("rx_001")
    ocr["lines"][0]["line_id"] = 0
    with pytest.raises(StructureError, match="일부 줄에만 line_id"):
        extract_rule(ocr)


@pytest.mark.parametrize("bad", [-1, 1.0, "3", None, True])
def test_a_line_id_that_is_not_a_non_negative_integer_is_refused(bad):
    ocr = with_line_ids(load_ocr("rx_001"))
    ocr["lines"][5]["line_id"] = bad
    with pytest.raises(StructureError, match="0 이상의 정수"):
        extract_rule(ocr)


def test_duplicate_line_ids_are_refused():
    ocr = with_line_ids(load_ocr("rx_001"))
    ocr["lines"][5]["line_id"] = ocr["lines"][6]["line_id"]
    with pytest.raises(StructureError, match="같은 line_id"):
        extract_rule(ocr)


# ------------------------------------------------------------ 발급번호


def test_issue_no_is_none_when_the_prescription_has_none():
    ocr = load_ocr("rx_001")
    ocr["lines"] = [l for l in ocr["lines"] if "00012호" not in l["text"]]
    assert extract_rule(ocr)["issue_no"] is None


def test_an_issue_number_below_the_table_does_not_count():
    ocr = load_ocr("rx_001")
    for l in ocr["lines"]:
        if "00012호" in l["text"]:
            l["bbox"] = [775, 3000, 1227, 3050]       # 표 아래로 옮긴다
    assert extract_rule(ocr)["issue_no"] is None


@pytest.mark.parametrize(
    "texts, expected",
    [
        (["2026년11월03일 00012호"], "2026년11월03일 00012호"),
        (["  2026년 11월 3일 12호  "], "2026년 11월 3일 12호"),           # 공백은 달라도 되고, 돌려줄 때는 앞뒤만 다듬는다
        (["２０２６년１１월０３일 ００１２호"], "２０２６년１１월０３일 ００１２호"),   # 전각 숫자도 알아본다 (글자는 그대로)
        (["2026년11월03일 000583"], "2026년11월03일 000583"),       # OCR이 '호'를 '3'으로 읽은 경우
        (["2026년11월03일 00012"], "2026년11월03일 00012"),
        (["2026년11월03일"], None),                                   # 번호가 없다
        (["발급일부터("], None),
        (["출력시간 : 2026.11.03 09.12.40"], None),
        (["가2026년11월03일 00012호"], None),                         # 줄 앞에 다른 글자가 붙어 있으면 다른 칸이다
        ([], None),
    ],
)
def test_find_issue_no(texts, expected):
    lines = [box(t, 100, 100 + 100 * i) for i, t in enumerate(texts)]
    assert find_issue_no(lines, table_top=2000) == expected


def test_the_topmost_issue_number_wins():
    lines = [box("2026년11월03일 00002호", 100, 500), box("2026년11월03일 00001호", 100, 300)]
    assert find_issue_no(lines, table_top=2000) == "2026년11월03일 00001호"


def test_on_the_same_height_the_left_one_wins():
    lines = [box("2026년11월03일 00002호", 900, 300), box("2026년11월03일 00001호", 100, 300)]
    assert find_issue_no(lines, table_top=2000) == "2026년11월03일 00001호"


def test_only_lines_whose_center_is_above_the_table_top_are_looked_at():
    line = box("2026년11월03일 00001호", 100, 1960, y2=2000)      # 중심 y = 1980
    assert find_issue_no([line], table_top=1981) == "2026년11월03일 00001호"
    assert find_issue_no([line], table_top=1980) is None             # 중심이 표 위쪽 선과 같으면 표 안이다


# ------------------------------------------------------------ 읽을 수 없는 처방전: 추측하지 않고 실패


def read_error(ocr):
    with pytest.raises(StructureError) as error:
        extract_rule(ocr)
    return str(error.value)


def test_no_header_fails():
    ocr = load_ocr("rx_001")
    ocr["lines"] = [l for l in ocr["lines"] if l["text"] not in ("횟수", "일수", "구분코드", "투약량", "처방 의약품의 명칭")]
    assert "머리글" in read_error(ocr)


def test_no_drug_lines_in_the_table_fails():
    ocr = load_ocr("rx_001")
    ocr["lines"] = [l for l in ocr["lines"] if not DRUG_LINE.match(l["text"])]
    message = read_error(ocr)
    assert "약품 줄을 찾지 못했습니다" in message and "속하지" not in message     # 값 칸이 남아 있어도 이 이유가 먼저다


def test_a_table_with_only_a_header_fails_instead_of_returning_no_drugs():
    lines = [box("횟수", 1000, 100), box("일수", 1100, 100)]
    assert "약품 줄을 찾지 못했습니다" in read_error({"lines": lines})


def test_a_drug_whose_code_was_misread_is_not_skipped_silently():
    # 코드의 숫자 하나를 O로 읽으면 그 줄은 코드 줄이 아니게 되고, 그 줄의 칸들도 갈 곳이 없다. 약품을 빼먹지 않고 실패한다.
    ocr = load_ocr("rx_001")
    anchors_of(ocr)[1]["text"] = "9000000З1 알모란정"        # З는 키릴 문자라서 숫자가 아니다
    assert "속하지 않는" in read_error(ocr)


def test_text_in_the_table_that_belongs_to_no_row_fails_without_quoting_it():
    ocr = load_ocr("rx_001")
    ocr["lines"].append(box("비밀약품", 300, 1700))
    message = read_error(ocr)
    assert "속하지 않는" in message and "비밀약품" not in message


def test_losing_the_end_marker_makes_the_footer_part_of_the_table_and_fails():
    ocr = load_ocr("rx_001")
    ocr["lines"] = [l for l in ocr["lines"] if not l["text"].startswith("주사제")]
    assert "속하지 않는" in read_error(ocr)


def test_a_missing_times_cell_names_the_row_and_the_column():
    ocr = load_ocr("rx_001")
    cell(ocr, 1, "3회")["text"] = "3호"             # '회'를 '호'로 잘못 읽었다
    message = read_error(ocr)
    assert "2번째 약품 줄" in message and "횟수" in message and "찾지 못했습니다" in message


def test_a_missing_dose_cell_fails():
    ocr = load_ocr("rx_001")
    ocr["lines"].remove(cell(ocr, 0, "1 TAB"))
    message = read_error(ocr)
    assert "1번째 약품 줄" in message and "1회량" in message


def test_a_missing_days_cell_fails():
    ocr = load_ocr("rx_002")
    ocr["lines"].remove(cell(ocr, 2, "2일"))
    message = read_error(ocr)
    assert "3번째 약품 줄" in message and "일수" in message


def test_two_cells_of_the_same_kind_in_one_row_fail():
    ocr = load_ocr("rx_001")
    twin = copy.deepcopy(cell(ocr, 0, "3회"))
    twin["bbox"][0] += 200
    twin["bbox"][2] += 200
    ocr["lines"].append(twin)
    message = read_error(ocr)
    assert "1번째 약품 줄" in message and "횟수" in message and "여러 개" in message


@pytest.mark.parametrize(
    "pattern, new_text, column",
    [("1 TAB", "0.5 TAB", "1회량"), ("1 TAB", "1/2 TAB", "1회량"), ("3회", "2.5회", "횟수"), ("3일", "3.5일", "일수")],
)
def test_a_value_that_is_not_a_whole_number_fails(pattern, new_text, column):
    ocr = load_ocr("rx_001")
    cell(ocr, 0, pattern)["text"] = new_text
    message = read_error(ocr)
    assert "1번째 약품 줄" in message and column in message and "정수" in message
    assert new_text not in message


def test_a_row_without_a_name_fails():
    ocr = load_ocr("rx_001")
    anchors_of(ocr)[0]["text"] = "(1)900000011"
    message = read_error(ocr)
    assert "1번째 약품 줄" in message and "약품명" in message


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"lines": []},
        {"lines": "텍스트"},
        {"lines": [{"text": "a"}]},                                   # bbox 없음
        {"lines": [{"text": "a", "bbox": [1, 2, 3]}]},                # bbox가 4개가 아님
        {"lines": [{"text": "a", "bbox": [1, 2, 3, "x"]}]},           # 좌표가 숫자가 아님
        {"lines": [{"bbox": [1, 2, 3, 4]}]},                          # text 없음
        {"lines": ["문자열"]},
    ],
)
def test_malformed_ocr_fails_with_structure_error(bad):
    with pytest.raises(StructureError):
        extract_rule(bad)
