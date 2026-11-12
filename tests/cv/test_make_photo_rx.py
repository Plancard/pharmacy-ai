"""cv/make_photo_rx.py: 같은 seed면 같은 사진, 정답 파일은 이름만 바꿔 옮긴다."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_photo_rx", ROOT / "cv" / "make_photo_rx.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    img = np.full((350, 248, 3), 255, np.uint8)
    cv2.putText(img, "RX 123", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    for i in (1, 2):
        cv2.imwrite(str(d / f"rx8_0{i}.png"), img)
    (d / "gt.json").write_text(json.dumps({"rx8_01.png": ["A"], "rx8_02.png": ["B"]}), encoding="utf-8")
    return d


def test_same_seed_same_photo(src, tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "CANVAS", (300, 400))
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    ia = mp.make_one(src / "rx8_01.png", a, "hard", 7)
    ib = mp.make_one(src / "rx8_01.png", b, "hard", 7)
    assert ia == ib and a.read_bytes() == b.read_bytes()
    assert 7 <= abs(ia["rotate_deg"]) <= 12 and 65 <= ia["jpeg"] <= 75


def test_main_writes_photos_and_renamed_gt(src, tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "CANVAS", (300, 400))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--out", str(out)])
    mp.main()
    assert sorted(p.name for p in out.glob("*.jpg")) == ["rx8p_01.jpg", "rx8p_02.jpg"]
    assert json.loads((out / "gt.json").read_text(encoding="utf-8")) == {"rx8p_01.jpg": ["A"], "rx8p_02.jpg": ["B"]}
    assert (out / "params.csv").read_text(encoding="utf-8").splitlines()[1].startswith("rx8p_01.jpg,rx8_01.png,easy")
