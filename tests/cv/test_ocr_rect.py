import cv2
import numpy as np
import pytest

from cv import ocr


class FakeReader:
    def readtext(self, img, detail=1, paragraph=False):
        return [([[0, 0], [20, 0], [20, 10], [0, 10]], "타이레놀정", 0.9)]


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "rx.jpg"
    cv2.imwrite(str(p), np.full((300, 400, 3), 255, np.uint8))
    return p


def test_rect_out_saved_with_a1_size(photo, tmp_path, monkeypatch):
    flat = np.full((200, 100, 3), 255, np.uint8)
    monkeypatch.setattr(ocr, "rectify", lambda img: (flat, {"status": "ok", "edge": ""}))
    out = tmp_path / "sub" / "rx.jpg.rect.jpg"
    a1, _, _ = ocr.read_document(photo, reader=FakeReader(), rect_out=out)
    saved = cv2.imread(str(out))
    assert (a1["width"], a1["height"]) == (100, 200)
    assert saved.shape[:2] == (200, 100)


@pytest.mark.parametrize("info, expected", [
    ({"status": "ok", "edge": "yes"}, True),
    ({"status": "ok", "edge": ""}, False),
    ({"status": "rotated90", "edge": "yes"}, True),
    ({"status": "skip", "edge": ""}, None),
])
def test_paper_edge_cut(photo, monkeypatch, info, expected):
    monkeypatch.setattr(ocr, "rectify", lambda img: (img, info))
    a1 = ocr.read_document(photo, reader=FakeReader())[0]
    assert a1.get("paper_edge_cut") is expected


def test_no_flatten_has_no_edge_field(photo):
    a1 = ocr.read_document(photo, flatten=False, reader=FakeReader())[0]
    assert "paper_edge_cut" not in a1
