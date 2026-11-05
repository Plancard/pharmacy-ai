"""규칙 기반 처방 구조화 (11/5 실제 처방전 양식으로 다시 작성).

입력은 부록 A.1의 lines (칸마다 text, bbox=[x1, y1, x2, y2], conf).

실제 처방전(별지 제9호 서식)은 열 위치로 읽을 수 없다. 표 머리글이 여러 조각으로 쪼개지고
('1회' / '투약량', '투악량'처럼 틀리게 읽히기도 한다), 같은 행의 칸이 y로 조금씩 어긋나며,
목록 순서도 x 순서가 아니다. 그래서 다음 순서로 읽는다.
  1. find_table      표 본문의 세로 범위: 머리글 낱말(횟수·일수 …)의 아래 ~ '주사제 처방명세'의 위
  2. 약품 줄         표 본문에서 9자리 코드가 든 줄. (환자 번호 같은 다른 9자리 숫자는 표 밖이다)
  3. attach_cells    나머지 칸을 세로로 가장 많이 겹치는 약품 줄에 붙인다
  4. split_columns   한 약품 줄의 칸을 글자 모양('1 TAB', '3회', '3일')으로 열별로 나눈다
  5. extract_rule    위를 합쳐 처방전 하나를 읽는다

읽을 수 없으면 추측하지 않고 StructureError로 알린다. (그러면 구조화 함수가 LLM 경로로 넘어간다)
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from agent.errors import StructureError   # 다른 파일도 같은 에러를 쓰려고 errors.py로 옮겼다

Line = dict[str, Any]   # A.1의 lines 한 칸: {"text": ..., "bbox": [x1, y1, x2, y2], "conf": ...}


def _center_y(line: Line) -> float:
    _, y1, _, y2 = line["bbox"]
    return (y1 + y2) / 2


def _height(line: Line) -> float:
    return line["bbox"][3] - line["bbox"][1]


def _overlap(a: Line, b: Line) -> float:
    """두 칸이 세로로 겹치는 길이. 안 겹치면 0 이하."""
    return min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1])


def _nfkc(text: str) -> str:
    """전각 문자를 반각으로 (３회 -> 3회, ㎎ -> mg)."""
    return unicodedata.normalize("NFKC", text)


def _key(line: Line) -> str:
    """공백을 모두 없앤 글자. OCR이 띄어 읽기 쉬운 말('주사제 처방명세')을 찾을 때 쓴다."""
    return "".join(_nfkc(line["text"]).split())


# ---------------------------------------------------------------- LLM 경로에 넘길 텍스트용: 같은 줄 묶기


def _reading_order(line: Line):
    """세로 중심, 그다음 왼쪽 x. 그래도 같으면 박스와 글자로 순서를 고정한다."""
    return (_center_y(line), line["bbox"][0], tuple(line["bbox"]), line["text"])


def group_rows(lines: list[Line]) -> list[list[Line]]:
    """OCR 칸들을 같은 줄끼리 묶는다.

    - 줄은 위에서 아래 순서, 한 줄 안의 칸은 왼쪽에서 오른쪽 순서로 돌려준다.
    - 세로 중심이 비슷하면 같은 줄이다: 지금 칸의 높이의 절반 이내.
      (OCR은 같은 줄이어도 y가 몇 픽셀씩 달라서 y가 완전히 같을 때만 묶으면 안 된다.)
    - 입력 순서는 상관없다. OCR 엔진이 칸을 섞어서 내보내도 같은 결과가 나온다.
      (세로 중심이 똑같은 칸은 왼쪽부터 처리한다. 처리 순서에 따라 줄 평균이 달라져 묶음이 바뀌기 때문이다.)
    """
    rows: list[list[Line]] = []
    row_cy = 0.0   # 지금 줄에 모인 칸들의 세로 중심 평균
    for line in sorted(lines, key=_reading_order):
        _, y1, _, y2 = line["bbox"]
        cy = _center_y(line)
        if rows and abs(cy - row_cy) <= (y2 - y1) / 2:
            rows[-1].append(line)
            row_cy = sum(_center_y(l) for l in rows[-1]) / len(rows[-1])
        else:
            rows.append([line])
            row_cy = cy
    return [sorted(row, key=lambda l: l["bbox"][0]) for row in rows]


def row_text(row: list[Line]) -> str:
    """한 줄의 칸들을 공백으로 이어 붙인 문자열."""
    return " ".join(line["text"] for line in row)


# ---------------------------------------------------------------- 1) 표 본문의 범위

# 표 머리글에 들어 있는 낱말. 공백을 뺀 글자로 찾는다.
HEADER_WORDS = ("처방의약품의명칭", "투약량", "횟수", "일수", "구분코드")
# 이 중 이만큼은 보여야 머리글로 인정한다. (OCR이 '투약량'을 '투악량'으로 읽는 일이 있어 전부를 요구하지 않는다)
MIN_HEADER_WORDS = 2
# 머리글 낱말 칸에 낱말 말고 더 붙어 있어도 되는 글자 수. 긴 문장 속의 우연한 일치를 막는다.
HEADER_EXTRA_CHARS = 2
# 이 말로 시작하는 줄부터는 약품 표 밖이다.
TABLE_END_WORD = "주사제"


def table_end(lines: list[Line]) -> float:
    """약품 표가 끝나는 y: '주사제 처방명세' 줄의 위. 그 줄이 없으면 끝이 없다(math.inf)."""
    return min((l["bbox"][1] for l in lines if _key(l).startswith(TABLE_END_WORD)), default=math.inf)


def find_table(lines: list[Line]) -> tuple[float, float]:
    """약품 표 본문의 세로 범위 (위, 아래)를 돌려준다.

    위는 머리글 낱말 칸들의 맨 아래, 아래는 '주사제 처방명세' 줄의 위다. (그 줄이 없으면 끝이 없다)
    머리글을 못 찾으면 StructureError. 못 읽었다는 사실을 숨기지 않는다.
    """
    bottom = table_end(lines)
    found: dict[str, float] = {}   # 머리글 낱말 -> 그 낱말 칸의 아래쪽 y
    for line in lines:
        if _center_y(line) >= bottom:
            continue
        key = _key(line)
        for word in HEADER_WORDS:
            if word in key and len(key) <= len(word) + HEADER_EXTRA_CHARS:
                found[word] = max(found.get(word, -math.inf), line["bbox"][3])
    if len(found) < MIN_HEADER_WORDS:
        raise StructureError("처방 의약품 표의 머리글(횟수·일수 등)을 찾지 못했습니다.")
    return max(found.values()), bottom


# ---------------------------------------------------------------- 2) 약품 줄, 3) 칸 붙이기

_CODE = re.compile(r"(?<!\d)\d{9}(?!\d)")        # 처방전에 인쇄된 9자리 코드
_ROW_NUMBER = re.compile(r"^\s*\(\s*\d+\s*\)\s*")   # 줄 앞의 번호 "(1)"

# 세로로 이 비율 이상 겹쳐야 같은 행으로 본다. (둘 중 낮은 칸의 높이 기준)
# 윗줄 칸이 아랫줄에 1~2픽셀 걸치는 정도는 같은 행이 아니다.
MIN_OVERLAP_RATIO = 0.3


def is_drug_line(line: Line) -> bool:
    return _CODE.search(_nfkc(line["text"])) is not None


def attach_cells(anchors: list[Line], cells: list[Line]) -> list[list[Line]]:
    """코드 줄(anchors)마다 같은 행의 칸들을 붙여 돌려준다. anchors와 같은 순서.

    칸은 세로로 가장 많이 겹치는 코드 줄에 붙는다. 어느 줄과도 충분히 겹치지 않으면 StructureError:
    약품명 뒷부분처럼 읽다 만 글자일 수 있어서, 조용히 버리지 않고 실패로 알린다.
    """
    attached: list[list[Line]] = [[] for _ in anchors]
    for cell in cells:
        best, best_overlap = None, 0.0
        for i, anchor in enumerate(anchors):
            overlap = _overlap(cell, anchor)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        if best is None or best_overlap < MIN_OVERLAP_RATIO * min(_height(cell), _height(anchors[best])):
            raise StructureError("표 안에 어느 약품 줄에도 속하지 않는 글자가 있습니다.")
        attached[best].append(cell)
    return attached


# ---------------------------------------------------------------- 4) 한 약품 줄의 칸 나누기

# 값 칸의 글자 모양: 숫자 + 단위. '1 TAB', '3회', '3일'
_VALUE_CELL = re.compile(r"^([0-9][0-9.,/]*)\s*(TAB|CAP|정|캡슐|알|회|일)$", re.IGNORECASE)
_UNIT_COLUMN = {"tab": "dose", "cap": "dose", "정": "dose", "캡슐": "dose", "알": "dose", "회": "times", "일": "days"}
COLUMN_LABELS = {"dose": "1회량", "times": "횟수", "days": "일수"}


def classify_cell(text: str) -> tuple[str, str] | None:
    """값 칸이면 (열 이름, 숫자 글자), 아니면 None. '1 TAB' -> ('dose', '1'), '３회' -> ('times', '3')

    숫자 글자는 읽은 그대로라서 '0.5'처럼 정수가 아닐 수 있다. 정수로 바꾸는 일은 한 줄을 읽을 때 한다.
    """
    match = _VALUE_CELL.match(_nfkc(text).strip())
    if match is None:
        return None
    return _UNIT_COLUMN[match.group(2).lower()], match.group(1)


def split_columns(cells: list[Line], name_edge: float | None) -> dict[str, list[str]]:
    """한 약품 줄에 붙은 칸들을 열별 글자로 나눈다. 칸은 왼쪽(x1이 작은 것)부터 센다.

    - dose / times / days: 글자 모양이 값 칸인 것의 숫자 글자
    - name: 값 칸이 아닌데 name_edge(값 칸이 시작하는 x)보다 왼쪽에서 시작하는 것. 이름이 둘로 쪼개진 조각이다.
    - other: 나머지 (용법 등)
    열 위치를 머리글에서 가져오지 않는다. 머리글은 쪼개지거나 틀리게 읽혀서 믿을 수 없다.
    """
    columns: dict[str, list[str]] = {"dose": [], "times": [], "days": [], "name": [], "other": []}
    for cell in sorted(cells, key=lambda l: l["bbox"][0]):
        text = cell["text"].strip()
        value = classify_cell(text)
        if value is not None:
            columns[value[0]].append(value[1])
        elif name_edge is not None and cell["bbox"][0] < name_edge:
            columns["name"].append(text)
        elif text:
            columns["other"].append(text)
    return columns


# ---------------------------------------------------------------- 5) 글자 -> 값

# 복용 시점 -> 용법에 쓰일 수 있는 말
SLOT_WORDS: dict[str, tuple[str, ...]] = {
    "morning": ("아침", "조식"),
    "noon": ("점심", "중식"),
    "evening": ("저녁", "석식"),
    "bedtime": ("취침", "자기전"),   # 글자 사이 공백은 지우고 찾으므로 "취침 전", "자기 전"도 잡힌다
}


def parse_time_slots(text: str) -> list[str]:
    """용법 글자에서 복용 시점을 찾는다. 항상 morning, noon, evening, bedtime 순서로 돌려준다.

    "PO 식후 30분"처럼 아침·점심·저녁·취침 전이 없으면 빈 목록이다.
    (빈 목록은 "알 수 없음"이다. 하루 횟수에 맞춰 채우는 일은 나중에 스키마 단계에서 한다.)
    """
    key = _nfkc(text).replace(" ", "")
    return [slot for slot, words in SLOT_WORDS.items() if any(w in key for w in words)]


def _read_number(digits: list[str], column: str) -> int:
    label = COLUMN_LABELS[column]
    if not digits:
        raise StructureError(f"{label} 칸을 찾지 못했습니다.")
    if len(digits) > 1:
        raise StructureError(f"{label} 칸이 여러 개입니다.")
    if not re.fullmatch(r"[0-9]+", digits[0]):
        # 0.5, 1/2 같은 값은 1차 범위 밖이다. 조용히 고치지 않고 실패로 알린다. 범위 검사는 스키마가 한다.
        raise StructureError(f"{label} 값을 정수로 읽지 못했습니다.")
    return int(digits[0])


def _clean_name(anchor_text: str, pieces: list[str]) -> str:
    """코드 줄에서 앞 번호 (n)과 9자리 코드를 빼고, 떨어져 나온 조각을 이어 붙인다. '/'는 공백으로 바꾼다.

    OCR이 읽은 글자를 고치지 않는다 ('5oOmg'는 그대로). 고치는 일은 약품 매칭 단계에서 한다.
    """
    text = _ROW_NUMBER.sub("", _nfkc(anchor_text))   # ^로 시작 위치에 묶여 있어 한 번만 바뀐다
    text = _CODE.sub("", text, count=1)
    return " ".join(" ".join([text, *map(_nfkc, pieces)]).replace("/", " ").split())


def _lowest_conf(lines: list[Line]) -> float | None:
    """칸들의 OCR 신뢰도 중 가장 낮은 값. 신뢰도가 없는 칸은 건너뛰고, 하나도 없으면 None."""
    confs = [
        l["conf"] for l in lines
        if isinstance(l.get("conf"), (int, float)) and not isinstance(l.get("conf"), bool)
    ]
    return min(confs) if confs else None


def _read_row(anchor: Line, cells: list[Line], name_edge: float | None) -> dict[str, Any]:
    columns = split_columns(cells, name_edge)
    numbers = {column: _read_number(columns[column], column) for column in COLUMN_LABELS}
    name = _clean_name(anchor["text"], columns["name"])
    if not name:
        raise StructureError("약품명을 찾지 못했습니다.")
    return {
        "name": name,
        "printed_code": _CODE.search(_nfkc(anchor["text"])).group(),
        "dose_per_time": numbers["dose"],
        "times_per_day": numbers["times"],
        "days": numbers["days"],
        "time_slots": parse_time_slots(" ".join(columns["other"])),
        # 이 줄을 읽는 데 쓴 모든 칸(코드 줄·이름 조각·값 칸·용법)의 최저 신뢰도. 낮으면 약사가 원문과 대조한다.
        "ocr_conf": _lowest_conf([anchor, *cells]),
    }


# ---------------------------------------------------------------- 처방전 하나 읽기

# 발급 연월일 및 번호의 값. "2026년11월03일 00012호". OCR이 끝의 '호'를 다른 글자로 읽는 일이 있어 숫자까지만 본다.
_ISSUE_NO = re.compile(r"^\d{4}년\d{1,2}월\d{1,2}일\d+")


def find_issue_no(lines: list[Line], table_top: float) -> str | None:
    """표 위쪽에서 발급 연월일 및 번호 칸의 글자를 찾는다. 없으면 None. 위에서부터 첫 번째를 쓴다."""
    for line in sorted(lines, key=lambda l: (_center_y(l), l["bbox"][0])):
        if _center_y(line) < table_top and _ISSUE_NO.match(_key(line)):
            return line["text"].strip()
    return None


def _check_lines(ocr: Any) -> list[Line]:
    """A.1의 lines가 있고 칸마다 text와 bbox(숫자 4개)가 있는지 확인한다."""
    lines = ocr.get("lines") if isinstance(ocr, dict) else None
    if not isinstance(lines, list) or not lines:
        raise StructureError("OCR 결과에 글자 줄(lines)이 없습니다.")
    for line in lines:
        bbox = line.get("bbox") if isinstance(line, dict) else None
        valid = (
            isinstance(line, dict)
            and isinstance(line.get("text"), str)
            and isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        )
        if not valid:
            raise StructureError("OCR 결과의 줄 형식(text, bbox)이 올바르지 않습니다.")
    return lines


def extract_rule(ocr: dict[str, Any]) -> dict[str, Any]:
    """규칙 기반 구조화: A.1 형식의 OCR 결과 -> 약품 목록.

    돌려주는 값은 LLM 경로와 같은 모양이다.
        {"issue_no": "...", "drugs": [{"name", "printed_code", "dose_per_time",
                                       "times_per_day", "days", "time_slots", "ocr_conf"}, ...]}
    name은 OCR이 읽은 글자 그대로다. time_slots는 용법에서 읽은 그대로라서 비어 있을 수 있다.
    ocr_conf는 그 약품 줄을 읽은 칸들의 최저 신뢰도(0~1)다. 칸에 신뢰도가 없으면 None. (LLM 경로는 이 값을 내지 않는다)
    (고치는 일은 약품 매칭 단계, 채우기와 범위 검사는 스키마 단계에서 한다)

    읽을 수 없으면 StructureError. 추측으로 채우지 않는다.
    - 머리글을 못 찾음, 표 안에 약품 줄(9자리 코드)이 없음
    - 표 안에 어느 약품 줄에도 속하지 않는 글자가 있음
    - 약품 줄인데 1회량·횟수·일수 칸이 없거나 여러 개, 정수가 아님, 약품명이 비어 있음
    """
    lines = _check_lines(ocr)
    top, bottom = find_table(lines)
    body = [l for l in lines if top < _center_y(l) < bottom]

    anchors = sorted((l for l in body if is_drug_line(l)), key=_center_y)
    if not anchors:
        raise StructureError("처방전에서 약품 줄을 찾지 못했습니다.")
    anchor_ids = {id(a) for a in anchors}
    cells = [l for l in body if id(l) not in anchor_ids and l["text"].strip()]
    attached = attach_cells(anchors, cells)

    # 이름 조각과 값 칸을 가르는 기준: 값 칸이 시작하는 가장 왼쪽 x (표 전체에서 하나)
    name_edge = min(
        (l["bbox"][0] for row in attached for l in row if classify_cell(l["text"]) is not None),
        default=None,
    )

    drugs: list[dict[str, Any]] = []
    for number, (anchor, row) in enumerate(zip(anchors, attached), start=1):
        try:
            drugs.append(_read_row(anchor, row, name_edge))
        except StructureError as e:
            raise StructureError(f"{number}번째 약품 줄: {e}") from e
    return {"issue_no": find_issue_no(lines, top), "drugs": drugs}


# ---------------------------------------------------------------- LLM 경로에 넘길 텍스트

# 환자 이름 옆의 환자 번호 "김가상(000000101)". 9자리라서 약품 코드와 헷갈리지 않게 따로 가린다.
_PATIENT_NUMBER = re.compile(r"\(\d{9}\)")

# 교부번호를 찾지 못했을 때 LLM에 적어 보내는 말. 프롬프트도 이 글자를 쓴다.
ISSUE_NOT_FOUND = "(찾지 못함)"


def llm_input(ocr: dict[str, Any]) -> tuple[str | None, list[str]]:
    """LLM에 보낼 내용: (교부번호 칸의 글자, 약품 표 본문의 줄들).

    처방전 위쪽의 환자·의료인 정보(이름, 주민등록번호, 전화번호, 면허번호 …)는 보내지 않는다.
    - 머리글을 찾으면 표 본문(머리글 아래 ~ '주사제 처방명세' 위)만 보낸다.
    - 머리글을 못 찾으면(규칙 경로가 실패한 흔한 이유) 첫 약품 줄부터 표 끝까지 보낸다.
      약품 줄은 9자리 코드가 있고, 같은 행에 값 칸(1 TAB, 3회, 3일 …)이 하나 이상 있는 줄이다.
      (OCR이 환자 번호의 괄호를 빠뜨려 '김가상 000000101'로 읽어도 그 줄부터 보내지 않게 한다.)
      약품 줄이 하나도 없으면 보낼 줄이 없다(빈 목록).
    OCR 형식이 올바르지 않으면 StructureError.
    """
    lines = _check_lines(ocr)
    try:
        top, bottom = find_table(lines)
    except StructureError:
        bottom = table_end(lines)
        values = [l for l in lines if classify_cell(l["text"]) is not None]
        codes = [
            l for l in lines
            if _center_y(l) < bottom
            and is_drug_line(l)
            and not _PATIENT_NUMBER.search(_nfkc(l["text"]))
            and any(_overlap(l, v) >= MIN_OVERLAP_RATIO * min(_height(l), _height(v)) for v in values)
        ]
        # 첫 코드 줄에서 시작한다. 같은 행의 칸이 조금 위로 어긋나도 들어오게 그 줄 높이의 절반만큼 더 위부터.
        top = min((l["bbox"][1] - _height(l) / 2 for l in codes), default=math.inf)
    body = [l for l in lines if top < _center_y(l) < bottom and l["text"].strip()]
    return find_issue_no(lines, top), [row_text(row) for row in group_rows(body)]


def format_llm_text(issue_no: str | None, rows: list[str]) -> str:
    """llm_input의 결과를 LLM에 보내는 텍스트로 적는다."""
    return "\n".join([f"교부번호: {issue_no or ISSUE_NOT_FOUND}", "약품 표:", *rows])


def ocr_to_llm_text(ocr: dict[str, Any]) -> str:
    """LLM 경로의 입력 텍스트. 교부번호와 약품 표 본문만 담는다 (환자·의료인 정보는 빠진다)."""
    return format_llm_text(*llm_input(ocr))


def ocr_to_text(ocr: dict[str, Any]) -> str:
    """OCR 결과 전체를 같은 줄끼리 이어 붙여 한 줄씩 적은 텍스트. (사람이 OCR 결과를 훑어볼 때 쓴다)

    환자 정보까지 모두 들어 있으니 LLM에는 ocr_to_llm_text를 쓴다.
    OCR 형식이 올바르지 않으면 StructureError.
    """
    return "\n".join(row_text(row) for row in group_rows(_check_lines(ocr)))
