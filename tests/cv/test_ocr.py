"""cv/ocr.py (A.1 만들기) · cv/ocr_eval.py (OCR 단계 채점)"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv import ocr_eval
from cv.ocr import read_document, to_lines
from cv.stub import OcrEmpty

ROOT = Path(__file__).resolve().parents[2]
A1_DIR = ROOT / "agent" / "resources" / "samples" / "rx8_a1"
GT = json.loads((ROOT / "agent" / "resources" / "samples" / "rx8_img" / "ground_truth.json").read_text(encoding="utf-8"))


class FakeReader:
    """EasyOCR 대신: 정해 둔 결과를 돌려주고, 받은 이미지 크기를 기록"""
    def __init__(self, results):
        self.results, self.shape = results, None

    def readtext(self, img, detail=1, paragraph=False):
        self.shape = img.shape
        return self.results


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------- cv/ocr.py
def test_to_lines_sorts_and_scales():
    res = [(box(100, 50, 200, 70), "3회", 0.9), (box(10, 52, 90, 68), "타이레놀정", 0.81),
           (box(10, 10, 60, 30), "처방전", 0.99), (box(0, 0, 5, 5), "  ", 0.2)]
    lines = to_lines(res, scale=0.5)
    assert [l["text"] for l in lines] == ["처방전", "타이레놀정", "3회"]       # 위→아래, 같은 줄은 왼→오, 빈 글자 제외
    assert [l["line_id"] for l in lines] == [0, 1, 2]
    assert lines[1]["bbox"] == [20, 104, 180, 136]                          # 축소 비율만큼 되돌림
    assert lines[1]["conf"] == 0.81


def test_read_document_resizes_and_keeps_a1_shape(tmp_path):
    p = tmp_path / "rx_test.png"
    cv2.imwrite(str(p), np.full((3200, 2400, 3), 255, np.uint8))
    fake = FakeReader([(box(10, 20, 110, 40), "게보린정", 0.7)])
    a1, img, info = read_document(p, image_id="abc", flatten=False, max_side=1600, reader=fake)
    assert fake.shape[:2] == (1600, 1200)                                    # 긴 변 1600으로 줄여 OCR
    assert set(a1) == {"image_id", "doc_type", "width", "height", "lines"}   # A.1 형식 그대로
    assert (a1["image_id"], a1["doc_type"], a1["width"], a1["height"]) == ("abc", "prescription", 2400, 3200)
    assert a1["lines"][0]["bbox"] == [20, 40, 220, 80]                       # 원래(OCR에 넣기 전) 이미지 좌표
    assert img.shape[:2] == (3200, 2400) and info["status"] == "off"


def test_read_document_empty_and_bad_type(tmp_path):
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((400, 300, 3), 255, np.uint8))
    with pytest.raises(OcrEmpty):
        read_document(p, flatten=False, reader=FakeReader([]))
    with pytest.raises(ValueError):
        read_document(p, doc_type="tray", reader=FakeReader([]))
    with pytest.raises(OcrEmpty):
        read_document(tmp_path / "없음.png", reader=FakeReader([]))


def test_read_document_flatten_finds_paper(tmp_path):
    """책상 위 흰 종이(A4 비율)를 찍은 사진 → 펴진 A4 크기 이미지에서 OCR"""
    img = np.full((4000, 3000, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[500, 450], [2550, 520], [2480, 3420], [430, 3350]], np.int32)], (245, 245, 245))
    p = tmp_path / "photo.jpg"
    cv2.imwrite(str(p), img)
    a1, out, info = read_document(p, reader=FakeReader([(box(10, 10, 50, 30), "R51", 0.9)]))
    assert info["status"] in ("ok", "rotated90")
    assert (a1["width"], a1["height"]) == (2480, 3508) == (out.shape[1], out.shape[0])


# ---------------------------------------------------------------- cv/ocr_eval.py
def load(name):
    return json.loads((A1_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_eval_clean_samples_all_correct():
    rows = []
    for k, g in GT.items():
        rows += ocr_eval.score_image(load(Path(k).stem), g["drugs"])
    s = ocr_eval.summarize(rows)
    assert s["lines"] == 41 and s["code"] == 1.0 and s["numbers"] == 1.0 and s["all"] == 1.0


def test_eval_counts_misreads():
    a1 = copy.deepcopy(load("rx8_01"))
    g = GT["rx8_01.png"]["drugs"]
    for l in a1["lines"]:
        if l["text"].startswith("(1)202106092"):
            l["text"] = "(1)2021O6O92 타이레놀정/500mg"      # 코드 오독 → 약품명으로 행을 찾음
        if l["text"] == "1 TAB" and l["bbox"][1] > 1300:
            l["text"] = "I TAB"                              # 게보린 1회량 오독
    rows = ocr_eval.score_image(a1, g)
    assert [r["code_ok"] for r in rows] == [False, True]
    assert rows[0]["times_ok"] and rows[0]["days_ok"] and rows[0]["dose_ok"]
    assert not rows[1]["dose_ok"] and rows[1]["times_ok"]


def test_eval_row_shift_breaks_numbers():
    """기운 사진처럼 오른쪽 숫자 칸이 한 줄 아래로 밀리면 그 줄의 숫자는 틀린 것으로 센다"""
    a1 = copy.deepcopy(load("rx8_01"))
    for l in a1["lines"]:
        if l["bbox"][0] > 1200 and 1250 < l["bbox"][1] < 1420:
            l["bbox"] = [l["bbox"][0], l["bbox"][1] + 90, l["bbox"][2], l["bbox"][3] + 90]
    rows = ocr_eval.score_image(a1, GT["rx8_01.png"]["drugs"])
    assert rows[0]["code_ok"] and not rows[0]["all_ok"]


def test_gt_lookup_by_stem():
    assert ocr_eval.gt_for(GT, "data/rx_photo/rx8p_03.jpg") is GT["rx8_03.png"]
    assert ocr_eval.gt_for(GT, "x/rx8_05.json") is GT["rx8_05.png"]


def test_eval_picks_nearest_row_when_rows_crowd():
    """다음 약의 숫자 줄이 같은 행 범위 안에 들어와도 세로로 가장 가까운 줄을 쓴다 (11/16 clean 오판 사례)"""
    a1 = copy.deepcopy(load("rx8_01"))
    texts = {"3회": "2회", "3일": "5일", "1 TAB": "2 TAB"}
    for l in a1["lines"]:
        if l["bbox"][0] > 1200 and 1340 < l["bbox"][1] < 1420 and l["text"] in texts:
            l["text"] = texts[l["text"]]                                  # 게보린 행을 다른 값으로
            l["bbox"] = [l["bbox"][0] - 10, l["bbox"][1] - 50, l["bbox"][2] - 10, l["bbox"][3] - 50]   # 타이레놀 행 쪽으로, 조금 왼쪽에
    rows = ocr_eval.score_image(a1, GT["rx8_01.png"]["drugs"])
    assert rows[0]["all_ok"], rows[0]["read"]
