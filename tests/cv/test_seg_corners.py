"""seg_corners: 마스크 윤곽선 → 꼭짓점 순서·정밀도"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cv"))
from seg_corners import corners_from_polygon  # noqa: E402


def tray_polygon(angle_deg, w=600, h=800, cx=1000, cy=1000, radius=20):
    """모서리가 둥근 세로 사각형을 그려 마스크 윤곽선과 정답 꼭짓점(tl·tr·br·bl)을 돌려준다"""
    img = np.zeros((2000, 2000), np.uint8)
    rect = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = rect @ rot.T + [cx, cy]
    inner = np.array([[-w / 2 + radius, -h / 2 + radius], [w / 2 - radius, -h / 2 + radius],
                      [w / 2 - radius, h / 2 - radius], [-w / 2 + radius, h / 2 - radius]]) @ rot.T + [cx, cy]
    cv2.fillPoly(img, [inner.astype(np.int32)], 255)
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea).reshape(-1, 2), corners


@pytest.mark.parametrize("angle", [0, 7, -7, 25, -25, 43, -43, 57, 180])
def test_corners_and_order(angle):
    poly, gt = tray_polygon(angle)
    q = corners_from_polygon(poly)
    assert q is not None
    if angle == 180:              # 180° 사진도 화면 기준: 위쪽 짧은 변이 tl→tr
        gt = np.roll(gt, 2, axis=0)
    err = np.linalg.norm(q - gt, axis=1)
    # 순서가 맞으면 오차는 몇 px, 틀리면 수백 px
    assert err.max() < 6, (angle, q.round(1), gt.round(1))


def test_pose2seg(tmp_path):
    """pose 분할 목록·꼭짓점 라벨 → seg 이미지·다각형 라벨·seg.yaml"""
    import subprocess
    import yaml
    rf = tmp_path / "tray" / "rf" / "corners_v3" / "exp" / "train"
    (rf / "images").mkdir(parents=True)
    (rf / "labels").mkdir()
    splits = tmp_path / "splits"
    splits.mkdir()
    names = {"train": "g01_s01_jpg.rf.aa", "val": "pose_s05_jpg.rf.bb", "test": "rot_s01_jpg.rf.cc"}
    for s, n in names.items():
        cv2.imwrite(str(rf / "images" / f"{n}.jpg"), np.zeros((40, 30, 3), np.uint8))
        (rf / "labels" / f"{n}.txt").write_text("0 0.5 0.5 0.8 0.8 0.1 0.1 2 0.9 0.1 2 0.9 0.9 2 0.1 0.9 2\n")
        (splits / f"pose_{s}.txt").write_text(f"{(rf / 'images' / f'{n}.jpg').resolve()}\n")
    script = Path(__file__).resolve().parents[2] / "cv" / "pose2seg.py"
    subprocess.run([sys.executable, str(script), "--data", str(tmp_path)], check=True)
    for s, n in names.items():
        lines = (splits / f"seg_{s}.txt").read_text().split()
        assert len(lines) == 1 and Path(lines[0]).exists()
        lbl = (tmp_path / "tray" / "seg" / "labels" / f"{n}.txt").read_text().split()
        assert lbl == ["0", "0.100000", "0.100000", "0.900000", "0.100000", "0.900000", "0.900000", "0.100000", "0.900000"]
    cfg = yaml.safe_load((splits / "seg.yaml").read_text())
    assert cfg["train"] == "splits/seg_train.txt" and cfg["names"] == {0: "tray"}
