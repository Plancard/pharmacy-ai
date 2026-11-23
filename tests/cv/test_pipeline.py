"""트레이 검수 파이프라인 (cv/pipeline.py, F-CV-05 진입점 + 결과 오버레이 그림). 모델 없이 본다.

실행: python -m pytest -q tests/cv/test_pipeline.py
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cv import pipeline as pl
from cv import tray_inspect as ti
from cv.stub import PlanRangeError, TrayNotFound

TYL, GEB = "202106092", "197900277"     # 타이레놀정500mg · 게보린정
CELL = 320                              # 보정 이미지 960×1280, 칸 320px
TIMES = ("morning", "noon", "evening", "bedtime")
WHITE = (255, 255, 255)


def box(day, time, k=0, dx=0, dy=0, w=50, h=36):
    cx = (day - 1) * CELL + 100 + 60 * k + dx
    cy = TIMES.index(time) * CELL + 160 + dy
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def det(code, day, time, k=0, **kw):
    return {"class_id": 0, "item_code": code, "bbox": box(day, time, k, **kw), "conf": 0.95}


def plan_3days():
    cells = [{"day": d, "time": t, "items": [{"item_code": TYL, "count": 1}, {"item_code": GEB, "count": 1}]}
             for d in (1, 2, 3) for t in ("morning", "noon", "evening")]
    return {"prescription_id": "rx1", "days": 3, "plan": cells}


def filled(plan):
    out = []
    for c in plan["plan"]:
        k = 0
        for it in c["items"]:
            for _ in range(it["count"]):
                out.append(det(it["item_code"], c["day"], c["time"], k))
                k += 1
    return out


def blank():
    return np.full((1280, 960, 3), 255, np.uint8)


def center(day, time):
    """칸 가운데 아래쪽 (알약 박스가 없는 자리)"""
    return (day - 1) * CELL + 160, TIMES.index(time) * CELL + 280


def px(img, xy):
    x, y = xy
    return tuple(int(v) for v in img[y, x])


def fake_models(monkeypatch, dets, image=None):
    from cv import detect, tray

    monkeypatch.setattr(tray, "rectify", lambda p, model=None: SimpleNamespace(image=blank() if image is None else image))
    monkeypatch.setattr(detect, "detect_pills", lambda img, model=None: dets)


def one_missing():
    """2일 저녁 게보린을 빼먹은 트레이"""
    plan = plan_3days()
    dets = [d for d in filled(plan) if not (d["item_code"] == GEB and d["bbox"] == box(2, "evening", 1))]
    return plan, dets


# ------------------------------------------------------------ 그림 (render_overlay)
def render(plan, dets):
    from cv import tray

    a3 = ti.judge(dets, plan, "img")
    return pl.render_overlay(blank(), tray.assign_cells(dets), a3), a3


def test_mismatch_cell_is_tinted_red_and_others_are_not():
    plan, dets = one_missing()
    out, a3 = render(plan, dets)
    assert a3["summary"]["mismatch"] == 1
    b, g, r = px(out, center(2, "evening"))
    assert r > 200 and g < 230 and b < 230 and r - g > 40          # 빨강 쪽으로 칠해짐
    assert px(out, center(1, "evening")) == WHITE                  # 일치 칸 안쪽은 칠하지 않는다
    assert px(out, center(3, "bedtime")) == WHITE                  # 빈 칸도 칠하지 않는다


def test_need_review_cell_is_tinted_amber_and_flagged_box_is_drawn_amber():
    plan = plan_3days()
    dets = filled(plan)
    dets[0] = det(TYL, 1, "morning", dy=130, h=120)                # 1일 아침 박스가 점심 칸으로 25% 걸침 → boundary
    out, a3 = render(plan, dets)
    c = next(c for c in a3["cells"] if (c["day"], c["time"]) == (1, "morning"))
    assert c["status"] == "need_review" and c["reasons"] == ["boundary"]
    b, g, r = px(out, center(1, "morning"))
    assert r > 200 and b < 200 and g > b                           # 주황 쪽으로 칠해짐
    x1, y1, x2, y2 = (int(round(v)) for v in dets[0]["bbox"])
    assert px(out, (max(x1, 0) + 40, y1)) == pl.FLAG               # 걸친 박스는 주황 테두리


def test_cell_borders_follow_the_web_grid_order():
    """칸 위치가 웹 판정 표와 같아야 한다: 위→아래 아침·점심·저녁·취침 전, 왼→오른 1·2·3일"""
    assert pl._cell_rect(1, "morning") == (0, 0, 320, 320)
    assert pl._cell_rect(3, "bedtime") == (640, 960, 960, 1280)
    assert pl._cell_rect(2, "noon") == (320, 320, 640, 640)


def test_render_does_not_change_the_input_image():
    plan, dets = one_missing()
    from cv import tray

    img = blank()
    pl.render_overlay(img, tray.assign_cells(dets), ti.judge(dets, plan, "img"))
    assert (img == 255).all()


# ------------------------------------------------------------ 진입점 (run)
def test_run_returns_the_same_a3_as_judge_and_saves_overlay(monkeypatch, tmp_path):
    plan, dets = one_missing()
    fake_models(monkeypatch, dets)
    out = tmp_path / "rx" / "abc.overlay.jpg"
    a3 = pl.run(tmp_path / "abc.jpg", plan, image_id="7f3e", overlay_path=out)
    assert a3 == ti.judge(dets, plan, "7f3e")                       # A.3 형식 그대로
    img = cv2.imread(str(out))
    assert img is not None and img.shape == (1280, 960, 3)
    assert not list(out.parent.glob("*.tmp"))                       # 임시 파일이 남지 않는다


def test_run_without_overlay_path_writes_nothing(monkeypatch, tmp_path):
    plan, dets = one_missing()
    fake_models(monkeypatch, dets)
    a3 = pl.run(tmp_path / "abc.jpg", plan)
    assert a3["image_id"] == "abc"
    assert list(tmp_path.iterdir()) == []


def test_rotated_tray_gets_no_overlay(monkeypatch, tmp_path):
    plan = plan_3days()
    fake_models(monkeypatch, ti.flip_180(filled(plan)))             # 계획대로 담고 180° 돌려 찍음
    out = tmp_path / "abc.overlay.jpg"
    a3 = pl.run(tmp_path / "abc.jpg", plan, overlay_path=out)
    assert pl.is_rotated(a3) and a3["summary"]["need_review"] == 12
    assert not out.exists()


def test_overlay_failure_still_returns_the_result(monkeypatch, tmp_path, caplog):
    plan, dets = one_missing()
    fake_models(monkeypatch, dets)
    (tmp_path / "not_a_dir").write_text("x")
    a3 = pl.run(tmp_path / "abc.jpg", plan, overlay_path=tmp_path / "not_a_dir" / "o.jpg")
    assert a3["summary"]["mismatch"] == 1
    assert "오버레이 그림 저장 실패" in caplog.text


def test_tray_not_found_passes_through_and_writes_nothing(monkeypatch, tmp_path):
    from cv import tray

    def no_tray(p, model=None):
        raise TrayNotFound("x")

    monkeypatch.setattr(tray, "rectify", no_tray)
    out = tmp_path / "o.jpg"
    with pytest.raises(TrayNotFound) as e:
        pl.run(tmp_path / "a.jpg", plan_3days(), overlay_path=out)
    assert e.value.code == "E_TRAY_NOT_FOUND" and not out.exists()


def test_bad_plan_fails_before_the_model_runs(monkeypatch, tmp_path):
    from cv import tray

    monkeypatch.setattr(tray, "rectify", lambda *a, **k: pytest.fail("모델을 돌리면 안 됩니다"))
    with pytest.raises(PlanRangeError):
        pl.run(tmp_path / "a.jpg", {"prescription_id": "r", "days": 4,
                                    "plan": [{"day": 4, "time": "noon", "items": []}]})


def test_inspect_tray_passes_overlay_path_through(monkeypatch, tmp_path):
    plan, dets = one_missing()
    fake_models(monkeypatch, dets)
    out = tmp_path / "o.jpg"
    a3 = ti.inspect_tray(tmp_path / "abc.jpg", plan, image_id="7f3e", overlay_path=out)
    assert a3["image_id"] == "7f3e" and out.exists()
