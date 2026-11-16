"""cv/tray.py: rectify (F-CV-02, 가짜 seg 모델) · assign_cells (F-CV-04)"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cv import tray
from cv.stub import TrayNotFound

W, H = 1500, 1900
TEMPLATE = np.zeros((1280, 960, 3), np.uint8)
TEMPLATE[:, :, 0] = np.linspace(0, 255, 960, dtype=np.uint8)[None, :]    # B: 가로 위치
TEMPLATE[:, :, 1] = np.linspace(0, 255, 1280, dtype=np.uint8)[:, None]   # G: 세로 위치
TEMPLATE[:, :, 2] = 128


def tray_quad(angle_deg, w=600, h=800, cx=750, cy=950):
    """세로로 긴 트레이의 꼭짓점 (템플릿의 좌상·우상·우하·좌하가 사진에서 놓인 자리)"""
    rect = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return rect @ rot.T + [cx, cy]


def photo_of(quad):
    """템플릿을 quad 자리에 붙인 사진"""
    photo = np.full((H, W, 3), 60, np.uint8)
    src = np.float32([[0, 0], [960, 0], [960, 1280], [0, 1280]])
    m = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    cv2.warpPerspective(TEMPLATE, m, (W, H), dst=photo, borderMode=cv2.BORDER_TRANSPARENT)
    return photo


def dense(quad, n=60):
    """마스크 윤곽선처럼 네 변 위에 점을 촘촘히 놓는다"""
    pts = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
    return np.array(pts)


class FakeBoxes:
    def __init__(self, conf):
        self.conf = np.array(conf)

    def __len__(self):
        return len(self.conf)


class FakeModel:
    """ultralytics seg 모델 대신: 정해 둔 윤곽선·신뢰도를 돌려주고, 받은 인자를 기록"""
    def __init__(self, polys=None, confs=None):
        self.polys, self.confs, self.kw = polys, confs, None

    def predict(self, img, **kw):
        self.kw = kw
        if not self.polys:
            return [SimpleNamespace(masks=None, boxes=FakeBoxes([]))]
        return [SimpleNamespace(masks=SimpleNamespace(xy=self.polys), boxes=FakeBoxes(self.confs))]


# ---------------------------------------------------------------- rectify
@pytest.mark.parametrize("angle", [0, 8, -8, 25, -25])
def test_rectify_restores_template(angle):
    quad = tray_quad(angle)
    model = FakeModel([dense(quad)], [0.93])
    r = tray.rectify_image(photo_of(quad), model)
    assert r.image.shape == (1280, 960, 3) and r.corrected
    assert r.conf == pytest.approx(0.93) and r.ratio == pytest.approx(0.75, abs=0.01)
    diff = np.abs(r.image[60:-60, 60:-60].astype(int) - TEMPLATE[60:-60, 60:-60].astype(int))
    assert diff.mean() < 4, diff.mean()          # 좌우·상하가 뒤바뀌면 100 넘게 나온다
    # 변환 행렬: 원본의 꼭짓점이 보정 이미지의 모서리로 간다
    p = cv2.perspectiveTransform(r.corners.reshape(1, 4, 2).astype(np.float64), r.matrix.astype(np.float64))[0]
    assert np.abs(p - [[0, 0], [960, 0], [960, 1280], [0, 1280]]).max() < 0.5


def test_model_called_with_spec_values():
    quad = tray_quad(0)
    model = FakeModel([dense(quad)], [0.9])
    tray.rectify_image(photo_of(quad), model)
    assert model.kw["imgsz"] == 960 and model.kw["conf"] == 0.5


def test_upside_down_photo_follows_screen_top():
    """180° 사진은 화면 위쪽 짧은 변이 맨 위가 된다 (뒤집힘은 검수 단계 rotated가 잡는다)"""
    quad = tray_quad(180)
    r = tray.rectify_image(photo_of(quad), FakeModel([dense(quad)], [0.9]))
    expect = TEMPLATE[::-1, ::-1]
    diff = np.abs(r.image[60:-60, 60:-60].astype(int) - expect[60:-60, 60:-60].astype(int))
    assert diff.mean() < 4


def test_sideways_tray_is_upright():
    quad = tray_quad(90)                      # 트레이를 옆으로 놓고 찍음 (긴 변이 가로)
    r = tray.rectify_image(photo_of(quad), FakeModel([dense(quad)], [0.9]))
    assert r.image.shape == (1280, 960, 3) and r.ratio == pytest.approx(0.75, abs=0.01)
    # 어느 쪽이 위가 되든 템플릿의 가로·세로 기울기는 유지된다 (0° 또는 180°만 남는다)
    d0 = np.abs(r.image[60:-60, 60:-60].astype(int) - TEMPLATE[60:-60, 60:-60].astype(int)).mean()
    d180 = np.abs(r.image[60:-60, 60:-60].astype(int) - TEMPLATE[::-1, ::-1][60:-60, 60:-60].astype(int)).mean()
    assert min(d0, d180) < 4


def test_upright_rolls_wide_quad():
    wide = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], float)
    assert tray._ratio(wide) == pytest.approx(4 / 3)
    assert tray._ratio(tray._upright(wide)) == pytest.approx(0.75)
    tall = wide[:, ::-1].copy()[[0, 3, 2, 1]]   # 이미 세로
    assert np.array_equal(tray._upright(tall), tall)


def test_no_tray_or_low_conf():
    quad = tray_quad(0)
    with pytest.raises(TrayNotFound) as e:
        tray.rectify_image(photo_of(quad), FakeModel())          # 신뢰도 0.5 미만이면 모델이 아무것도 안 돌려준다
    assert e.value.code == "E_TRAY_NOT_FOUND"


def test_cut_tray_is_rejected():
    quad = tray_quad(0, cx=150)                  # 왼쪽이 사진 밖으로 나감 (일부만 찍힘)
    with pytest.raises(TrayNotFound):
        tray.rectify_image(photo_of(quad), FakeModel([dense(quad)], [0.9]))


def test_bad_aspect_is_rejected():
    quad = tray_quad(0, w=800, h=800)            # 3:4가 아닌 정사각형 (비율 1.0)
    with pytest.raises(TrayNotFound):
        tray.rectify_image(photo_of(quad), FakeModel([dense(quad)], [0.9]))


def test_aspect_within_tolerance_passes():
    quad = tray_quad(0, w=640, h=800)            # 비율 0.8 → 3:4에서 6.7% 차이
    tray.rectify_image(photo_of(quad), FakeModel([dense(quad)], [0.9]))


def test_unreadable_file(tmp_path):
    with pytest.raises(TrayNotFound):
        tray.rectify(tmp_path / "없음.jpg", model=FakeModel())


def test_rectify_reads_file(tmp_path):
    quad = tray_quad(5)
    p = tmp_path / "t.jpg"
    cv2.imwrite(str(p), photo_of(quad))
    r = tray.rectify(p, model=FakeModel([dense(quad)], [0.8]))
    assert r.image.shape == (1280, 960, 3)


# ---------------------------------------------------------------- assign_cells
def det(cx, cy, w=60, h=40, conf=0.9):
    return {"class_id": 0, "item_code": "202106092", "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "conf": conf}


def test_all_twelve_cells():
    times = ["morning", "noon", "evening", "bedtime"]
    dets = [det(160 + 320 * c, 160 + 320 * r) for r in range(4) for c in range(3)]
    out = tray.assign_cells(dets)
    assert [(d["cell"]["day"], d["cell"]["time"]) for d in out] == [(c + 1, times[r]) for r in range(4) for c in range(3)]
    assert all(d["flags"] == [] for d in out)


def test_center_on_border_goes_right_and_down():
    out = tray.assign_cells([det(320, 640, w=20, h=20)])
    assert out[0]["cell"] == {"day": 2, "time": "evening"}


def test_input_is_not_changed():
    src = [det(160, 160)]
    out = tray.assign_cells(src)
    assert "cell" not in src[0] and "flags" not in src[0] and out[0] is not src[0]


def test_boundary_flag():
    # 중심은 1일차 아침 칸(x<320)인데 폭 100 중 40이 오른쪽 칸에 걸침 → 40%
    d = {"item_code": "x", "bbox": [260, 100, 360, 160], "conf": 0.9}
    out = tray.assign_cells([d])
    assert out[0]["cell"] == {"day": 1, "time": "morning"} and out[0]["flags"] == ["boundary"]
    # 10%만 걸치면 플래그 없음
    d2 = {"item_code": "x", "bbox": [230, 100, 330, 160], "conf": 0.9}
    assert tray.assign_cells([d2])[0]["flags"] == []


def test_overlap_flag_only_inside_same_cell():
    a = {"item_code": "x", "bbox": [100, 100, 180, 160], "conf": 0.9}
    b = {"item_code": "y", "bbox": [105, 105, 185, 165], "conf": 0.9}      # 같은 칸, IoU 높음
    c = {"item_code": "z", "bbox": [400, 100, 480, 160], "conf": 0.9}      # 다른 칸
    out = tray.assign_cells([a, b, c])
    assert out[0]["flags"] == ["overlap"] and out[1]["flags"] == ["overlap"] and out[2]["flags"] == []
    far = {"item_code": "y", "bbox": [170, 100, 250, 160], "conf": 0.9}     # 겹침이 작음(IoU 낮음)
    assert tray.assign_cells([a, far])[0]["flags"] == []


def test_outside_tray_has_no_cell():
    out = tray.assign_cells([det(-30, 100), det(500, 1300), det(500, 500)])
    assert out[0]["cell"] is None and out[1]["cell"] is None and out[2]["cell"] == {"day": 2, "time": "noon"}


def test_rows_cols_must_match_tray():
    with pytest.raises(ValueError):
        tray.assign_cells([], rows=3, cols=3)
