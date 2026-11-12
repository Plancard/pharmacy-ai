"""cv/rectify_doc.py: 책상 위 종이의 네 모서리를 찾아 정면으로 편다."""
import importlib.util
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rectify_doc", ROOT / "cv" / "rectify_doc.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)


def photo(corners, size=(1500, 2000)):
    """어두운 책상 위에 표 선이 있는 흰 종이를 corners(좌상 · 우상 · 우하 · 좌하) 자리에 놓은 사진."""
    w, h = size
    img = np.full((h, w, 3), 70, np.uint8)
    doc = np.full((1414, 1000, 3), 245, np.uint8)
    for y in range(200, 1300, 120):
        cv2.line(doc, (60, y), (940, y), (0, 0, 0), 3)
    cv2.putText(doc, "RX", (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [1000, 0], [1000, 1414], [0, 1414]]), np.float32(corners))
    paper = cv2.warpPerspective(doc, M, (w, h))
    mask = cv2.warpPerspective(np.full((1414, 1000), 255, np.uint8), M, (w, h)) > 0
    img[mask] = paper[mask]
    return img


def test_finds_tilted_paper_corners():
    truth = np.float32([[300, 250], [1180, 330], [1120, 1600], [220, 1520]])
    q, ratio, cut = rd.find_paper(photo(truth))
    assert q is not None and not cut and ratio > 0.3
    assert np.abs(q - truth).max() < 15


def test_rectify_makes_a4_portrait():
    out, info = rd.rectify(photo([[300, 250], [1180, 330], [1120, 1600], [220, 1520]]))
    assert info["status"] == "ok" and out.shape[:2] == (rd.OUT_H, rd.OUT_W)


def test_no_paper_is_skipped():
    img = np.full((2000, 1500, 3), 70, np.uint8)
    out, info = rd.rectify(img)
    assert info["status"] == "skip" and out is img


def test_cut_paper_is_flagged():
    _, info = rd.rectify(photo([[300, -120], [1250, 0], [1150, 1700], [250, 1600]]))
    assert info["edge"] == "yes"


def test_landscape_paper_turns_portrait():
    _, info = rd.rectify(photo([[100, 1400], [100, 400], [1514, 400], [1514, 1400]], size=(1650, 1800)))
    assert info["status"] == "rotated90"
