"""기운 사진 읽기 (11/12 촬영본 점검 뒤): structure.estimate_skew · straighten, 흐린 머리글, LLM 추론 강도.

폰으로 찍은 처방전은 종이가 몇 도 기울어 있다. A.1의 bbox는 기울어진 글자를 감싼 수평 상자라서, 바로잡지 않으면
1°만 기울어도 같은 행의 칸이 세로로 겹치지 않아 규칙 경로가 20장 모두 실패했다 (같은 줄 묶기도 흩어져 LLM 입력이 깨졌다).
여기서는 깨끗한 샘플의 좌표를 돌려(글자는 그대로) 기운 사진을 만든다. 원근(사다리꼴)은 CV 전처리가 편다.
"""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from agent import config
from agent.config import RESOURCES_DIR
from agent.llm import LlmPrescription, chat_parse
from agent.structure import (
    SKEW_MIN_LINES,
    estimate_skew,
    extract_rule,
    find_table,
    llm_input,
    straighten,
)
from agent.validate import order_from_extracted

SAMPLES = RESOURCES_DIR / "samples"
RX8 = sorted((SAMPLES / "rx8_a1").glob("rx8_*.json"))
CV = sorted((SAMPLES / "cv_a1").glob("rx_*.json"))            # 실제 EasyOCR이 읽은 5장


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_of(path):
    folder = "rx8_expected" if path.parent.name == "rx8_a1" else "cv_expected"
    return json.loads((SAMPLES / folder / path.name).read_text(encoding="utf-8"))


def tilt(ocr, deg):
    """사진이 deg도(시계 방향 +) 기운 것처럼: 칸마다 네 꼭짓점을 종이 가운데를 축으로 돌리고, 그것을 감싼 수평 상자를 bbox로 둔다.

    순서와 line_id는 그대로 둔다 (OCR이 주는 순서가 달라져도 읽는 결과는 같아야 한다. 그것은 다른 시험이 본다).
    """
    t = math.radians(deg)
    cos, sin = math.cos(t), math.sin(t)
    cx, cy = ocr.get("width", 2480) / 2, ocr.get("height", 3507) / 2
    out = copy.deepcopy(ocr)
    for line in out["lines"]:
        x1, y1, x2, y2 = line["bbox"]
        pts = [((x - cx) * cos - (y - cy) * sin + cx, (x - cx) * sin + (y - cy) * cos + cy)
               for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        line["bbox"] = [int(min(p[0] for p in pts)), int(min(p[1] for p in pts)),
                        int(max(p[0] for p in pts)), int(max(p[1] for p in pts))]
    return out


def read(ocr):
    return order_from_extracted("P0001", extract_rule(ocr), source="rule")


def drugs(order):
    return [(d.printed_code, d.name, d.count_per_dose, d.days, tuple(d.times), tuple(d.source_line_ids)) for d in order.items]


# ------------------------------------------------------------ 각도 재기


@pytest.mark.parametrize("path", RX8 + CV + sorted((SAMPLES / "a1_ocr").glob("*.json")), ids=lambda p: p.stem)
def test_a_level_page_is_left_alone(path):
    ocr = load(path)
    assert estimate_skew(ocr["lines"]) == 0.0
    assert straighten(ocr["lines"]) is ocr["lines"]            # 반듯하면 좌표를 건드리지 않는다


@pytest.mark.parametrize("deg", [-11.0, -5.0, -2.0, -0.75, 0.75, 1.0, 3.0, 7.0, 12.0])
def test_the_tilt_of_a_page_is_measured(deg):
    for path in RX8[:5]:
        assert estimate_skew(tilt(load(path), deg)["lines"]) == pytest.approx(deg, abs=0.15)


def test_too_few_cells_are_not_trusted_for_an_angle():
    lines = tilt(load(RX8[0]), 5.0)["lines"]
    assert estimate_skew(lines[:SKEW_MIN_LINES - 1]) == 0.0 and estimate_skew(lines) != 0.0


def test_a_page_tilted_more_than_the_search_range_is_not_guessed_beyond_it():
    assert abs(estimate_skew(tilt(load(RX8[0]), 25.0)["lines"])) <= 15.0


# ------------------------------------------------------------ 바로잡은 좌표


def test_straightening_keeps_the_order_the_text_and_the_line_ids():
    ocr = tilt(load(RX8[0]), 4.0)
    fixed = straighten(ocr["lines"])
    assert fixed is not ocr["lines"] and len(fixed) == len(ocr["lines"])
    assert [(l["line_id"], l["text"], l["conf"]) for l in fixed] == [(l["line_id"], l["text"], l["conf"]) for l in ocr["lines"]]
    assert ocr == tilt(load(RX8[0]), 4.0)                      # 받은 OCR 결과는 바꾸지 않는다


def test_straightened_cells_are_back_where_they_were_on_the_level_page():
    ocr = load(RX8[0])
    fixed = straighten(tilt(ocr, 6.0)["lines"])

    def cy(line):
        return (line["bbox"][1] + line["bbox"][3]) / 2

    def height(line):
        return line["bbox"][3] - line["bbox"][1]

    # 되돌린 좌표는 통째로 옮겨져 있을 수 있다(축이 다르다). 칸끼리의 세로 간격과 칸의 높이가 반듯한 종이와 같으면 된다.
    for level, back in zip(ocr["lines"], fixed):
        assert (cy(back) - cy(fixed[0])) - (cy(level) - cy(ocr["lines"][0])) == pytest.approx(0, abs=4), level["text"]
        if level["bbox"][2] - level["bbox"][0] > 3 * height(level):          # 긴 글자: 부푼 상자에서 원래 높이를 되찾는다
            assert height(back) == pytest.approx(height(level), abs=6), level["text"]


# ------------------------------------------------------------ 기운 사진을 규칙 경로로 읽는다


@pytest.mark.parametrize("deg", [-7.0, -3.0, -1.0, 0.75, 1.0, 2.0, 5.0, 7.0])
def test_every_virtual_prescription_is_read_the_same_when_tilted(deg):
    for path in RX8:
        ocr = load(path)
        assert drugs(read(tilt(ocr, deg))) == drugs(read(ocr)), f"{path.stem} {deg}°"


@pytest.mark.parametrize("deg", [-5.0, -2.0, 1.0, 3.0])
def test_the_real_ocr_samples_are_read_the_same_when_tilted(deg):
    for path in CV:
        ocr = load(path)
        assert drugs(read(tilt(ocr, deg))) == drugs(read(ocr)), f"{path.stem} {deg}°"


def test_the_source_lines_of_a_tilted_page_still_point_at_the_original_lines():
    ocr = load(RX8[0])
    level, tilted = extract_rule(ocr), extract_rule(tilt(ocr, 3.0))
    assert [i["source_line_ids"] for i in tilted["items"]] == [i["source_line_ids"] for i in level["items"]]
    assert tilted["issue_no"] == level["issue_no"]


@pytest.mark.parametrize("deg", [-12.0, -6.0, 1.0, 4.0, 10.0])
def test_the_rows_sent_to_the_llm_are_one_drug_per_row_again(deg):
    for path in RX8:
        ocr = load(path)
        assert llm_input(tilt(ocr, deg)) == llm_input(ocr), f"{path.stem} {deg}°"


# ------------------------------------------------------------ 흐린 머리글


def box(text, x1, y1, x2=None, y2=None):
    return {"text": text, "bbox": [x1, y1, x2 or x1 + 100, y2 or y1 + 40], "conf": 0.9}


BODY = [box("(1)900000011 타이레놀정", 100, 200, 600, 240), box("2 TAB", 800, 205), box("3회", 1000, 205), box("3일", 1100, 205)]


@pytest.mark.parametrize("headers", [
    ("투악량", "처방의악품의명칭"),              # 받침 · 모음 하나씩 틀리게 읽음
    ("투약랑", "구분코드"),
    ("1회투약량", "처방 의약품의 명칭"),         # 글자가 더 붙거나 띄어 읽음 (원래 되던 것)
])
def test_header_words_that_are_read_a_little_wrong_still_mark_the_table(headers):
    lines = [box(h, 100 + 400 * i, 100, 400 + 400 * i) for i, h in enumerate(headers)] + BODY
    assert extract_rule({"lines": lines})["items"][0]["name"] == "타이레놀정"


@pytest.mark.parametrize("headers", [
    ("횟수", "잇수"),                            # 두 글자 낱말은 글자가 같아야 한다 (횟수 · 일수는 서로 비슷하다)
    ("투여기간", "조제약사"),                    # 다른 낱말
    ("투약량",),                                 # 하나로는 부족하다
])
def test_words_that_are_not_header_words_do_not_mark_the_table(headers):
    from agent.errors import StructureError

    lines = [box(h, 100 + 400 * i, 100, 400 + 400 * i) for i, h in enumerate(headers)] + BODY
    with pytest.raises(StructureError, match="머리글"):
        find_table(lines)


# ------------------------------------------------------------ LLM 추론 강도


class FakeClient:
    def __init__(self):
        self.calls = []
        parsed = LlmPrescription(issue_no=None, items=[])
        message = SimpleNamespace(parsed=parsed, refusal=None)
        answer = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw: self.calls.append(kw) or answer))


def test_the_llm_is_asked_to_think_briefly_by_default():
    assert config.LLM_REASONING_EFFORT == "low"
    client = FakeClient()
    chat_parse([], LlmPrescription, client=client)
    assert client.calls[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize("value, sent", [("minimal", "minimal"), ("medium", "medium")])
def test_the_reasoning_effort_can_be_changed(monkeypatch, value, sent):
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", value)
    client = FakeClient()
    chat_parse([], LlmPrescription, client=client)
    assert client.calls[0]["reasoning_effort"] == sent


def test_an_empty_reasoning_effort_is_not_sent(monkeypatch):
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", "")
    client = FakeClient()
    chat_parse([], LlmPrescription, client=client)
    assert "reasoning_effort" not in client.calls[0]
