"""칸별 검수 판정 (cv/tray_inspect.py, F-CV-05). 모델 없이 보정 이미지 좌표의 검출 목록으로 본다.

실행: python -m pytest -q tests/cv/test_tray_inspect.py
"""
from collections import Counter
from types import SimpleNamespace

import pytest

from cv import tray_inspect as ti
from cv.stub import CVError, PlanRangeError, TrayNotFound

TYL, GEB, PRO = "202106092", "197900277", "200200099"     # 타이레놀정500mg · 게보린정 · 프론크논캡슐
CELL = 320                                                # 보정 이미지 960×1280, 칸 320px
TIMES = ("morning", "noon", "evening", "bedtime")


def box(day, time, k=0, dx=0, dy=0, w=50, h=36):
    """칸 안 k번째 알약 박스 (칸 가운데 근처에 가로로 늘어놓음)."""
    cx = (day - 1) * CELL + 100 + 60 * k + dx
    cy = TIMES.index(time) * CELL + 160 + dy
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def det(code, day, time, k=0, **kw):
    return {"class_id": 0, "item_code": code, "bbox": box(day, time, k, **kw), "conf": 0.95}


def plan_3days():
    """타이레놀·게보린 아침·점심·저녁 3일, 취침 전은 빈 칸 (가짜 연결부 샘플과 같은 모양)."""
    cells = [{"day": d, "time": t, "items": [{"item_code": TYL, "count": 1}, {"item_code": GEB, "count": 1}]}
             for d in (1, 2, 3) for t in ("morning", "noon", "evening")]
    return {"prescription_id": "rx1", "days": 3, "plan": cells}


def filled(plan):
    """계획대로 담은 검출 목록."""
    out = []
    for c in plan["plan"]:
        k = 0
        for it in c["items"]:
            for _ in range(it["count"]):
                out.append(det(it["item_code"], c["day"], c["time"], k))
                k += 1
    return out


def cell(a3, day, time):
    return next(c for c in a3["cells"] if (c["day"], c["time"]) == (day, time))


def bad(a3):
    return [(c["day"], c["time"], c["status"], c["reasons"]) for c in a3["cells"] if c["status"] != "ok"]


# ------------------------------------------------------------ 판정 규칙
def test_as_planned_is_all_ok():
    a3 = ti.judge(filled(plan_3days()), plan_3days(), "img1")
    assert a3["summary"] == {"ok": 12, "mismatch": 0, "need_review": 0, "hitl_required": False}
    assert (a3["prescription_id"], a3["image_id"]) == ("rx1", "img1")
    assert [(c["day"], c["time"]) for c in a3["cells"]][:5] == [(1, "morning"), (1, "noon"), (1, "evening"),
                                                                (1, "bedtime"), (2, "morning")]
    assert cell(a3, 1, "bedtime") == {"day": 1, "time": "bedtime", "expected": {}, "actual": {},
                                      "status": "ok", "reasons": []}


def test_missing_wrong_and_extra_pills_are_mismatch():
    plan = plan_3days()
    dets = filled(plan)
    dets = [d for d in dets if not (d["bbox"] == box(1, "morning", 0))]          # 1일 아침 타이레놀 빠짐
    dets = [({**d, "item_code": PRO} if d["bbox"] == box(2, "noon", 1) else d) for d in dets]   # 2일 점심 게보린 → 프론크논
    dets.append(det(TYL, 3, "bedtime"))                                            # 빈 칸에 알약
    a3 = ti.judge(dets, plan)
    assert bad(a3) == [(1, "morning", "mismatch", []), (2, "noon", "mismatch", []), (3, "bedtime", "mismatch", [])]
    assert cell(a3, 1, "morning")["actual"] == {GEB: 1}
    assert cell(a3, 2, "noon")["actual"] == {TYL: 1, PRO: 1}
    assert cell(a3, 3, "bedtime") == {"day": 3, "time": "bedtime", "expected": {}, "actual": {TYL: 1},
                                      "status": "mismatch", "reasons": []}
    assert a3["summary"] == {"ok": 9, "mismatch": 3, "need_review": 0, "hitl_required": True}


def test_boundary_and_overlap_need_review_even_if_counts_match():
    plan = plan_3days()
    dets = filled(plan)
    # 1일 점심 게보린을 칸 오른쪽 벽에 걸치게 (중심은 그대로 1일, 박스 절반 가까이 2일 칸으로)
    dets = [({**d, "bbox": [270, 450, 340, 490]} if d["bbox"] == box(1, "noon", 1) else d) for d in dets]
    # 3일 저녁 두 알을 거의 겹치게
    dets = [({**d, "bbox": box(3, "evening", 0, dx=8)} if d["bbox"] == box(3, "evening", 1) else d) for d in dets]
    a3 = ti.judge(dets, plan)
    assert bad(a3) == [(1, "noon", "need_review", ["boundary"]), (3, "evening", "need_review", ["overlap"])]
    assert cell(a3, 1, "noon")["actual"] == {TYL: 1, GEB: 1}                      # 개수는 맞지만 확인 필요


def test_outside_the_tray_is_left_out():
    plan = plan_3days()
    dets = filled(plan) + [{"class_id": 0, "item_code": TYL, "bbox": [-80, -60, -20, -10], "conf": 0.9}]
    assert ti.judge(dets, plan)["summary"]["ok"] == 12


def test_upside_down_tray_is_rotated_for_all_cells():
    plan = plan_3days()
    a3 = ti.judge(ti.flip_180(filled(plan)), plan)
    assert a3["summary"] == {"ok": 0, "mismatch": 0, "need_review": 12, "hitl_required": True}
    assert all(c["reasons"] == ["rotated"] for c in a3["cells"])
    assert cell(a3, 3, "bedtime")["actual"] == {TYL: 1, GEB: 1}                    # 실제는 돌리지 않은 그대로 보인다


def test_a_plan_that_looks_the_same_upside_down_is_not_rotated():
    """180° 돌려도 계획과 똑같이 맞으면(대칭) 일치 칸 수가 같으므로 뒤집힘으로 보지 않는다."""
    cells = [{"day": d, "time": t, "items": [{"item_code": TYL, "count": 1}]} for d in (1, 2, 3) for t in TIMES]
    plan = {"prescription_id": "rx1", "days": 3, "plan": cells}
    a3 = ti.judge(ti.flip_180(filled(plan)), plan)
    assert a3["summary"]["ok"] == 12


def test_a_few_wrong_cells_are_not_mistaken_for_rotation():
    plan = plan_3days()
    dets = [d for d in filled(plan) if d["bbox"][1] > CELL]                          # 아침 줄 3칸이 통째로 빔
    a3 = ti.judge(dets, plan)
    assert [c["reasons"] for c in a3["cells"] if c["status"] != "ok"] == [[], [], []]
    assert a3["summary"]["mismatch"] == 3


def test_unknown_class_and_plan_outside_the_tray_are_errors():
    plan = plan_3days()
    with pytest.raises(CVError) as e:
        ti.judge(filled(plan) + [{"class_id": 9, "item_code": None, "bbox": box(1, "bedtime"), "conf": 0.9}], plan)
    assert e.value.code == "E_CV_INFER"
    with pytest.raises(PlanRangeError):
        ti.judge([], {"prescription_id": "rx1", "days": 4,
                      "plan": [{"day": 4, "time": "morning", "items": [{"item_code": TYL, "count": 1}]}]})


def test_counts_of_zero_in_the_plan_are_ignored():
    plan = {"prescription_id": "rx1", "days": 1,
            "plan": [{"day": 1, "time": "morning", "items": [{"item_code": TYL, "count": 1}, {"item_code": GEB, "count": 0}]}]}
    a3 = ti.judge([det(TYL, 1, "morning")], plan)
    assert cell(a3, 1, "morning")["expected"] == {TYL: 1} and a3["summary"]["ok"] == 12


# ------------------------------------------------------------ Agent가 받는 형식
@pytest.mark.parametrize("make", ["ok", "mismatch", "review", "rotated"])
def test_the_agent_accepts_the_result(make):
    pytest.importorskip("pydantic")
    from agent.inspection import read_inspection
    from agent.schema import Plan

    plan = plan_3days()
    dets = filled(plan)
    if make == "mismatch":
        dets = dets[1:]
    elif make == "review":
        dets[0] = {**dets[0], "bbox": [280, 140, 350, 180]}
    elif make == "rotated":
        dets = ti.flip_180(dets)
    a3 = ti.judge(dets, plan, "img1")
    got = read_inspection(a3, Plan.model_validate(plan))
    assert got.summary.hitl_required == (make != "ok")


def test_same_shape_as_the_stub():
    from cv import stub

    plan = plan_3days()
    mine, theirs = ti.judge(filled(plan), plan, "img1"), stub.inspect("img1.jpg", plan, scenario="ok")
    assert mine == theirs


# ------------------------------------------------------------ 서비스 함수 (모델은 가짜)
def test_inspect_tray_runs_rectify_then_detection(monkeypatch, tmp_path):
    from cv import detect, tray

    plan = plan_3days()
    calls = []
    monkeypatch.setattr(tray, "rectify", lambda p, model=None: calls.append(("rectify", p.name, model)) or
                        SimpleNamespace(image="IMG"))
    monkeypatch.setattr(detect, "detect_pills", lambda img, model=None: calls.append(("detect", img, model)) or
                        filled(plan))
    a3 = ti.inspect_tray(tmp_path / "abc.jpg", plan, model="M", seg_model="S")
    assert calls == [("rectify", "abc.jpg", "S"), ("detect", "IMG", "M")]
    assert a3["image_id"] == "abc" and a3["summary"]["ok"] == 12
    assert ti.inspect_tray(tmp_path / "abc.jpg", plan, image_id="7f3e")["image_id"] == "7f3e"


def test_inspect_tray_passes_tray_not_found(monkeypatch, tmp_path):
    from cv import tray

    def no_tray(p, model=None):
        raise TrayNotFound("x")

    monkeypatch.setattr(tray, "rectify", no_tray)
    with pytest.raises(TrayNotFound) as e:
        ti.inspect_tray(tmp_path / "a.jpg", plan_3days())
    assert e.value.code == "E_TRAY_NOT_FOUND"


def test_a_bad_plan_fails_before_the_model_runs(monkeypatch, tmp_path):
    from cv import tray

    monkeypatch.setattr(tray, "rectify", lambda *a, **k: pytest.fail("모델을 돌리면 안 됩니다"))
    with pytest.raises(PlanRangeError):
        ti.inspect_tray(tmp_path / "a.jpg", {"prescription_id": "r", "days": 4,
                                             "plan": [{"day": 4, "time": "noon", "items": []}]})


# ------------------------------------------------------------ 정답표 평가
CELLS_CSV = """group,day,time,planned,actual,error,flags
g01,1,morning,0:1;1:1,0:1;1:1,,
g01,1,noon,0:1;1:1,1:1,missing,
g01,1,evening,0:1;1:1,0:1;1:1,,boundary
g01,1,bedtime,,,,
g01,2,morning,0:1,0:1,,
g01,2,noon,0:1,0:1,,
g01,2,evening,0:1,0:1,,
g01,2,bedtime,,,,
g01,3,morning,,,,
g01,3,noon,,,,
g01,3,evening,,,,
g01,3,bedtime,,,,
"""


def test_load_truth_and_plan_from_counts(tmp_path):
    (tmp_path / "cells.csv").write_text(CELLS_CSV + "".join(
        f"g21,{d},{t},,{'3:1' if (d, t) == (1, 'morning') else ''},,\n" for d in (1, 2, 3) for t in TIMES),
        encoding="utf-8")
    truth = ti.load_truth(tmp_path / "cells.csv")
    assert truth["g01"][(1, "noon")]["expected"] == Counter({0: 1, 1: 1})
    assert truth["g01"][(1, "noon")]["actual"] == Counter({1: 1})
    assert truth["g01"][(1, "evening")]["flags"] == {"boundary"}
    assert (truth["g01"][(1, "noon")]["error"], truth["g01"][(1, "morning")]["error"]) == ("missing", "")
    assert truth["g21"][(1, "morning")]["expected"] == Counter({3: 1})       # 추가 촬영분: expected = actual

    plan = ti.plan_from_counts({k: v["expected"] for k, v in truth["g01"].items()}, "g01")
    assert plan["prescription_id"] == "g01" and plan["days"] == 2
    assert plan["plan"][0] == {"day": 1, "time": "morning", "items": [{"item_code": TYL, "count": 1},
                                                                    {"item_code": GEB, "count": 1}]}


def test_evaluate_counts_misses_and_false_alarms(tmp_path):
    (tmp_path / "cells.csv").write_text(CELLS_CSV, encoding="utf-8")
    truth = ti.load_truth(tmp_path / "cells.csv")
    plan = ti.plan_from_counts({k: v["expected"] for k, v in truth["g01"].items()}, "g01")
    good = [d for d in filled(plan) if d["bbox"] != box(1, "noon", 0)]           # 실제 사진처럼 1일 점심 타이레놀 없음
    caught = ti.judge(good, plan)
    missed = ti.judge(filled(plan), plan)                                          # 잘못 담은 칸을 ok로 봄
    noisy = ti.judge(good + [det(TYL, 3, "bedtime")], plan)                        # 정상 빈 칸을 불일치로 봄
    stat, rows = ti.evaluate([("a.jpg", "g01", caught), ("b.jpg", "g01", missed), ("c.jpg", "g01", noisy),
                              ("d.jpg", "g01", None)], truth)
    assert (stat["photos"], stat["retake"]) == (3, 1)
    assert (stat["wrong_cells"], stat["wrong_missed"]) == (3, 1)
    assert (stat["normal_cells"], stat["normal_false_alarm"]) == (30, 1)
    assert stat["flagged_cells"] == 3 and len(rows) == 36
    assert stat["truth_conflict"] == 0 and rows[1]["error"] == "missing"


@pytest.mark.parametrize("path, group", [
    ("data/tray/fixed/g07_s03.jpg", "g07"),                                             # photos.csv 이름 그대로
    ("data/tray/rf/corners_v3/x/valid/images/g07_s03_jpg.rf.4f9a0c1b2d.jpg", "g07"),     # Roboflow 내보내기 이름
    ("data/tray/fixed/g13_s01.jpg", "g13"),                                             # photos.csv에 없어도 이름 앞 그룹
    ("data/tray/fixed/pose_s05.jpg", None), ("data/tray/fixed/empty_s01.jpg", None), ("rot_s01.jpg", None),
])
def test_group_of_photo_names(path, group):
    photos = {"g07_s03": "g07", "empty_s01": "empty", "pose_s05": "pose"}
    assert ti.group_of(path, photos, {"g07": {}, "g13": {}}) == group


def test_evaluate_rotated(tmp_path):
    plan = plan_3days()
    up = ti.judge(filled(plan), plan)
    down = ti.judge(ti.flip_180(filled(plan)), plan)
    stat, rows = ti.evaluate_rotated([("rot_s01.jpg", "g20", down), ("rot_s02.jpg", "g20", up), ("rot_s03.jpg", "g20", None)])
    assert (stat["rotated_hit"], stat["rotated_missed"], stat["retake"]) == (1, 1, 1)
    assert [r["rotated"] for r in rows] == ["yes", "no", ""] and rows[1]["result"] == "놓침: 전 칸 일치로 봄"


def test_old_expected_column_still_reads(tmp_path):
    (tmp_path / "cells.csv").write_text(CELLS_CSV.replace("planned", "expected", 1), encoding="utf-8")
    assert ti.load_truth(tmp_path / "cells.csv")["g01"][(1, "noon")]["expected"] == Counter({0: 1, 1: 1})


@pytest.mark.parametrize("value", ["ok", "OK", "none", "-", "정상"])
def test_normal_markers_in_the_error_column_are_not_errors(tmp_path, value):
    (tmp_path / "cells.csv").write_text(CELLS_CSV.replace("0:1;1:1,0:1;1:1,,\n", f"0:1;1:1,0:1;1:1,{value},\n"),
                                        encoding="utf-8")
    truth = ti.load_truth(tmp_path / "cells.csv")
    assert truth["g01"][(1, "morning")]["error"] == "" and truth["g01"][(1, "noon")]["error"] == "missing"


def test_truth_conflict_is_counted(tmp_path):
    """error 열은 비었는데 planned ≠ actual이면(또는 반대) 정답표 확인용으로 센다."""
    (tmp_path / "cells.csv").write_text(CELLS_CSV.replace("1:1,missing,", "1:1,,"), encoding="utf-8")
    truth = ti.load_truth(tmp_path / "cells.csv")
    plan = ti.plan_from_counts({k: v["expected"] for k, v in truth["g01"].items()}, "g01")
    stat, _ = ti.evaluate([("a.jpg", "g01", ti.judge(filled(plan), plan))], truth)
    assert stat["truth_conflict"] == 1
