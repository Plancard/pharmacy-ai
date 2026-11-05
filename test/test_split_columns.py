"""약품 줄 알아보기(is_drug_line), 값 칸 알아보기(classify_cell), 열별로 나누기(split_columns), 행에 붙이기(attach_cells)."""
import pytest

from agent.structure import StructureError, attach_cells, classify_cell, is_drug_line, split_columns


def box(text, x1=0, y1=0, x2=100, y2=40):
    return {"text": text, "bbox": [x1, y1, x2, y2], "conf": 0.9}


# ------------------------------------------------------------ is_drug_line


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(1)900000011 타이레놀정/ 5oOmg", True),
        ("900000051 프론크논캠술", True),
        ("(1)671805860 필루비서방정", True),
        ("900000011", True),
        ("타이레놀정 900000011", True),
        ("９００００００１１ 가정", True),                 # 전각 숫자
        ("김가상(000000101)", True),                      # 9자리 숫자가 있으면 줄 자체는 코드 줄이다. 표 밖인지는 find_table이 가른다
        ("90000001", False),                              # 8자리
        ("요양기관기호 99990001", False),
        ("9000000110", False),                            # 10자리: 앞 9자리만 보고 코드로 착각하면 안 된다
        ("1900000011", False),                            # 10자리: 뒤 9자리만 보고 코드로 착각하면 안 된다
        ("0200000000001", False),                         # 긴 번호
        ("900000011 900000021", True),
        ("타이레놀정", False),
        ("", False),
    ],
)
def test_a_drug_line_has_a_nine_digit_code(text, expected):
    assert is_drug_line(box(text)) is expected


# ------------------------------------------------------------ classify_cell


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 TAB", ("dose", "1")),
        ("1TAB", ("dose", "1")),
        ("2 CAP", ("dose", "2")),
        ("1 tab", ("dose", "1")),               # 대소문자는 상관없다
        ("1정", ("dose", "1")),
        ("2 캡슐", ("dose", "2")),
        ("1알", ("dose", "1")),
        ("3회", ("times", "3")),
        ("3 회", ("times", "3")),
        ("３회", ("times", "3")),                # 전각 숫자
        ("3일", ("days", "3")),
        ("10일", ("days", "10")),
        ("  3일  ", ("days", "3")),
        # 정수가 아닌 숫자 글자도 값 칸으로 알아본다. 정수로 읽을 수 없다는 판단은 한 줄을 읽을 때 한다.
        ("0.5 TAB", ("dose", "0.5")),
        ("1/2 TAB", ("dose", "1/2")),
        ("3.5일", ("days", "3.5")),
    ],
)
def test_value_cells_are_recognized_by_their_shape(text, expected):
    assert classify_cell(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "PO 식후 30분",     # 용법
        "145mg",           # 약품명이 쪼개진 조각
        "타이레놀정",
        "아르펜정",         # 단위('정')로 끝나도 앞에 숫자가 없으면 값 칸이 아니다
        "",
        "   ",
        "3",               # 단위가 없다
        "회",
        "일",
        "TAB",
        "30분",
        "3회분",
        "1 TAB 2",
        "1일 3회",
        "O회",             # OCR이 0을 O로 읽은 경우: 값으로 우기지 않는다
        "3호",             # '회'를 '호'로 잘못 읽은 경우
    ],
)
def test_other_text_is_not_a_value_cell(text):
    assert classify_cell(text) is None


# ------------------------------------------------------------ split_columns


def test_cells_are_split_into_columns_whatever_their_list_order():
    cells = [
        box("PO 식후 30분", x1=1921),
        box("3일", x1=1617),
        box("145mg", x1=680),
        box("1 TAB", x1=1303),
        box("3회", x1=1476),
    ]
    assert split_columns(cells, name_edge=1301) == {
        "dose": ["1"], "times": ["3"], "days": ["3"], "name": ["145mg"], "other": ["PO 식후 30분"],
    }


def test_name_pieces_and_other_texts_are_kept_in_left_to_right_order():
    cells = [box("뒤", x1=900), box("앞", x1=500), box("용법2", x1=2100), box("용법1", x1=1900)]
    columns = split_columns(cells, name_edge=1301)
    assert columns["name"] == ["앞", "뒤"] and columns["other"] == ["용법1", "용법2"]


def test_a_cell_starting_exactly_at_the_edge_is_not_a_name_piece():
    columns = split_columns([box("기타", x1=1301), box("조각", x1=1300)], name_edge=1301)
    assert columns["other"] == ["기타"] and columns["name"] == ["조각"]


def test_without_an_edge_nothing_is_a_name_piece():
    columns = split_columns([box("145mg", x1=100)], name_edge=None)
    assert columns["name"] == [] and columns["other"] == ["145mg"]


def test_blank_cells_are_dropped_and_text_is_stripped():
    columns = split_columns([box("   ", x1=2000), box("  용법  ", x1=2100)], name_edge=1301)
    assert columns["other"] == ["용법"]


def test_two_cells_of_the_same_kind_are_both_kept_so_the_reader_can_refuse():
    columns = split_columns([box("1 TAB", x1=1300), box("2 TAB", x1=1400), box("3회", x1=1500)], name_edge=1300)
    assert columns["dose"] == ["1", "2"] and columns["times"] == ["3"] and columns["days"] == []


def test_no_cells_gives_empty_columns():
    assert split_columns([], name_edge=None) == {"dose": [], "times": [], "days": [], "name": [], "other": []}


# ------------------------------------------------------------ attach_cells

# 코드 줄 두 개: y 100~140, y 200~240 (높이 40)
A1 = box("(1)900000011 가", y1=100, y2=140)
A2 = box("(2)900000021 나", y1=200, y2=240)


def attach(*cells):
    return attach_cells([A1, A2], list(cells))


def test_cells_go_to_the_row_they_overlap():
    c1, c2 = box("a", y1=110, y2=150), box("b", y1=210, y2=250)
    assert attach(c1, c2) == [[c1], [c2]]


def test_result_has_one_list_per_code_line_even_when_empty():
    assert attach() == [[], []]
    c = box("a", y1=110, y2=150)
    assert attach(c) == [[c], []]


def test_a_cell_touching_two_rows_goes_to_the_one_it_overlaps_more():
    upper = box("a", y1=120, y2=205)      # 위 줄과 20, 아래 줄과 5 겹침
    lower = box("b", y1=135, y2=225)      # 위 줄과 5, 아래 줄과 25 겹침
    assert attach(upper) == [[upper], []]
    assert attach(lower) == [[], [lower]]


def test_a_cell_that_only_grazes_the_next_row_stays_with_its_own_row():
    # 실제 샘플의 '3일'은 자기 줄과 39픽셀, 아랫줄과 1픽셀 겹친다
    cell = box("3일", y1=270, y2=325)
    anchors = [box("a", y1=250, y2=309), box("b", y1=324, y2=387)]
    assert attach_cells(anchors, [cell]) == [[cell], []]


def test_overlap_of_thirty_percent_of_the_smaller_box_is_enough():
    cell = box("x", y1=128, y2=168)      # A1(100~140)과 12 겹침 = 높이 40의 30%
    assert attach(cell) == [[cell], []]


def test_overlap_below_thirty_percent_is_refused():
    with pytest.raises(StructureError, match="속하지 않는"):
        attach(box("x", y1=129, y2=169))      # 11 겹침 = 27.5%


def test_the_ratio_uses_the_smaller_box():
    # 칸이 아주 작으면(높이 10) 3픽셀만 겹쳐도 30%다
    assert attach(box("점", y1=137, y2=147)) == [[box("점", y1=137, y2=147)], []]
    # 칸이 아주 크면(높이 200) 기준은 코드 줄(높이 40)이다. 12픽셀 겹치면 충분하다.
    big = box("큰", y1=128, y2=328)
    assert attach_cells([A1], [big]) == [[big]]


def test_a_cell_outside_every_row_is_refused_without_quoting_it():
    with pytest.raises(StructureError) as error:
        attach(box("비밀약품", y1=500, y2=540))
    assert "비밀약품" not in str(error.value)


def test_a_cell_just_touching_a_row_is_refused():
    with pytest.raises(StructureError):
        attach(box("x", y1=140, y2=180))      # 겹침 0


def test_a_cell_with_no_code_lines_at_all_is_refused():
    with pytest.raises(StructureError):
        attach_cells([], [box("x")])


def test_equal_overlap_goes_to_the_upper_row():
    cell = box("t", y1=128, y2=212)       # 위 줄과 12, 아래 줄과 12: 같다
    assert attach(cell) == [[cell], []]
