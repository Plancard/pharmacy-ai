"""cv/detect.py: detect_pills (F-CV-03, 가짜 검출 모델) · detect (보정 → 검출 → 칸 배정)"""
from types import SimpleNamespace

import numpy as np
import pytest

from cv import detect, tray
from cv.stub import TrayNotFound

IMG = np.zeros((1280, 960, 3), np.uint8)


class FakeBoxes:
    def __init__(self, rows):
        a = np.array(rows, float).reshape(-1, 6)       # x1, y1, x2, y2, conf, cls
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]

    def __len__(self):
        return len(self.conf)


class FakeDetModel:
    """ultralytics 검출 모델 대신: 정해 둔 박스를 돌려주고, 받은 인자를 기록"""
    def __init__(self, rows):
        self.rows, self.kw = rows, None

    def predict(self, img, **kw):
        self.kw = kw
        return [SimpleNamespace(boxes=FakeBoxes(self.rows))]


def test_model_called_with_spec_values():
    model = FakeDetModel([])
    assert detect.detect_pills(IMG, model) == []
    assert model.kw["imgsz"] == 960 and model.kw["conf"] == 0.70
    assert (detect.DET_IMGSZ, detect.DET_CONF) == (960, 0.70)


def test_below_baseline_is_dropped_even_if_the_model_returns_it():
    dets = detect.detect_pills(IMG, FakeDetModel([[10, 10, 50, 50, 0.69, 0], [60, 10, 100, 50, 0.70, 1]]))
    assert [d["class_id"] for d in dets] == [1]


def test_item_code_from_class_id_and_unknown_class_is_none():
    dets = detect.detect_pills(IMG, FakeDetModel([[10, 10, 50, 50, 0.9, 0], [60, 10, 100, 50, 0.9, 99]]))
    assert dets[0]["item_code"] == "202106092"           # 타이레놀정500mg
    assert dets[1]["class_id"] == 99 and dets[1]["item_code"] is None
    assert set(dets[0]) == {"class_id", "item_code", "bbox", "conf"}


def test_all_eight_classes_have_item_codes():
    codes = detect.item_codes()
    assert sorted(codes) == list(range(8)) and len(set(codes.values())) == 8


def test_sorted_top_to_bottom_then_left_to_right():
    rows = [[500, 400, 560, 460, 0.9, 1], [100, 400, 160, 460, 0.9, 2], [300, 20, 360, 80, 0.9, 3]]
    assert [d["class_id"] for d in detect.detect_pills(IMG, FakeDetModel(rows))] == [3, 2, 1]


def test_detect_returns_a2_with_cells(monkeypatch):
    monkeypatch.setattr(tray, "rectify", lambda path, model=None: SimpleNamespace(image=IMG))
    model = FakeDetModel([[100, 20, 160, 80, 0.95, 0], [400, 340, 460, 400, 0.88, 4]])
    a2 = detect.detect("var/uploads/tray_001.jpg", model=model)
    assert a2["image_id"] == "tray_001"
    assert a2["tray"] == {"rows": 4, "cols": 3, "cell_px": 320, "corrected": True}
    first, second = a2["detections"]
    assert first["cell"] == {"day": 1, "time": "morning"} and first["flags"] == []
    assert second["cell"] == {"day": 2, "time": "noon"} and second["item_code"] == "200201239"


def test_detect_passes_tray_errors(monkeypatch):
    def no_tray(path, model=None):
        raise TrayNotFound("x")
    monkeypatch.setattr(tray, "rectify", no_tray)
    with pytest.raises(TrayNotFound) as e:
        detect.detect("t.jpg", model=FakeDetModel([]))
    assert e.value.code == "E_TRAY_NOT_FOUND"
