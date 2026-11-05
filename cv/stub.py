"""CV 가짜 모듈 (Agent·웹 연동용). 모델 없이 기능분석서 5.1절 A.1~A.3 형식의 결과를 돌려준다.

실제 모듈(cv/pipeline.py)이 나오면 import만 바꾼다. 함수 이름·인자·반환 형식·예외는 그대로 유지한다.

    from cv.stub import ocr, detect, inspect, CVError      # 지금
    from cv.pipeline import ocr, detect, inspect, CVError  # 11/25 이후

함수
  ocr(image_path)                       → A.1 OCR 결과
  detect(image_path, plan=None)         → A.2 알약 검출 결과
  inspect(image_path, plan)             → A.3 검수 결과

오류 (기능분석서 6장) — 모두 CVError를 상속하고 .code에 오류 코드가 들어 있다
  OcrEmpty         E_OCR_EMPTY        글자를 찾지 못함 → 재촬영
  TrayNotFound     E_TRAY_NOT_FOUND   트레이를 찾지 못함 → 재촬영
  PlanRangeError   E_PLAN_RANGE       조제 계획이 트레이 범위(1~3일, 4개 시점)를 넘음 → 조제지시서 수정

시나리오 (scenario 인자, 없으면 환경변수 CV_STUB_SCENARIO, 기본 "ok")
  ok         계획대로 담김 → 모든 칸 ok
  mismatch   약이 든 첫 칸에서 1알이 빠짐 → 그 칸 mismatch (HITL ② 확인용)
  review     약이 든 첫 칸이 저신뢰 → 그 칸 need_review, reasons ["low_conf"]
  no_tray    detect·inspect가 TrayNotFound
  rotated    트레이가 180° 뒤집혀 찍힘 → 12칸 모두 need_review, reasons ["rotated"]
  ocr_empty  ocr가 OcrEmpty

OCR 샘플은 cv/fixtures/a1/<이미지 파일 이름>.json을 읽는다. 없으면 rx_001.json을 쓴다.
"""
import copy
import json
import os
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "a1"
TIMES = ("morning", "noon", "evening", "bedtime")   # 트레이 4행 (위→아래)
DAYS = (1, 2, 3)                                    # 트레이 3열 (왼→오)
CELL_PX = 320                                       # 보정 이미지 960×1280에서 칸 한 변
CONF_MIN = 0.5                                      # 이 미만이면 need_review (low_conf)
SCENARIOS = ("ok", "mismatch", "review", "rotated", "no_tray", "ocr_empty")


class CVError(Exception):
    code = "E_CV_INFER"

    def __init__(self, message=""):
        super().__init__(f"{self.code}: {message}" if message else self.code)


class OcrEmpty(CVError):
    code = "E_OCR_EMPTY"


class TrayNotFound(CVError):
    code = "E_TRAY_NOT_FOUND"


class PlanRangeError(CVError):
    code = "E_PLAN_RANGE"


def _scenario(scenario):
    s = scenario or os.environ.get("CV_STUB_SCENARIO", "ok")
    if s not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {s} (가능: {', '.join(SCENARIOS)})")
    return s


def _image_id(image_path):
    return Path(image_path).stem


# ---------- A.1 ----------

def ocr(image_path, scenario=None):
    """처방전·약봉투 이미지 → A.1 OCR 결과."""
    if _scenario(scenario) == "ocr_empty":
        raise OcrEmpty(_image_id(image_path))
    p = FIXTURES / f"{_image_id(image_path)}.json"
    if not p.exists():
        p = FIXTURES / "rx_001.json"
    a1 = json.loads(p.read_text(encoding="utf-8"))
    a1["image_id"] = _image_id(image_path)
    a1.setdefault("doc_type", "prescription")
    for i, line in enumerate(a1["lines"]):        # 예전 샘플에 line_id가 없어도 동작하게
        line.setdefault("line_id", i)
    return a1


# ---------- A.2 ----------

def _expected(plan):
    """A.4 plan → {(day, time): {item_code: count}}. 트레이 밖이면 PlanRangeError."""
    exp = {}
    for c in plan["plan"]:
        if c["day"] not in DAYS or c["time"] not in TIMES:
            raise PlanRangeError(f"트레이에 없는 칸 day={c['day']}, time={c['time']}")
        cell = exp.setdefault((c["day"], c["time"]), {})
        for it in c["items"]:
            cell[it["item_code"]] = cell.get(it["item_code"], 0) + it["count"]
    return exp


def _order(key):
    return key[0], TIMES.index(key[1])


def _target_cell(exp):
    """시나리오로 바꿀 칸: 약이 든 첫 칸 (1일차 아침부터)."""
    filled = [k for k, v in exp.items() if v]
    return min(filled, key=_order) if filled else None


def _box(day, time, k):
    """칸 안 k번째 알약 박스. 칸 안에 가로로 늘어놓는다."""
    col, row = DAYS.index(day), TIMES.index(time)
    cx = col * CELL_PX + 100 + 60 * (k % 3)
    cy = row * CELL_PX + 120 + 70 * (k // 3)
    return [cx - 25, cy - 18, cx + 25, cy + 18]


def detect(image_path, plan=None, scenario=None):
    """트레이 이미지 → A.2 검출 결과. plan을 주면 계획(과 시나리오)대로 담긴 것처럼 만든다."""
    s = _scenario(scenario)
    if s == "no_tray":
        raise TrayNotFound(_image_id(image_path))

    exp = _expected(plan) if plan else {(1, "morning"): {"000000000": 1}}
    actual = copy.deepcopy(exp)
    target = _target_cell(exp)
    if s == "mismatch" and target:
        code = next(iter(actual[target]))
        actual[target][code] -= 1
        if actual[target][code] == 0:
            del actual[target][code]

    detections = []
    for (day, time), items in sorted(actual.items(), key=lambda kv: _order(kv[0])):
        k = 0
        for code, n in items.items():
            for _ in range(n):
                detections.append({
                    "class_id": None,          # 실제 모듈에서는 classes.yaml의 ID. 약은 item_code로 구분
                    "item_code": code,
                    "bbox": _box(day, time, k),
                    "conf": 0.42 if (s == "review" and (day, time) == target) else 0.91,
                    "cell": {"day": day, "time": time},
                    "flags": [],               # boundary · overlap
                })
                k += 1
    return {
        "image_id": _image_id(image_path),
        "tray": {"rows": len(TIMES), "cols": len(DAYS), "cell_px": CELL_PX, "corrected": True},
        "detections": detections,
    }


# ---------- A.3 ----------

def inspect(image_path, plan, scenario=None):
    """트레이 이미지 + A.4 조제 계획 → A.3 검수 결과. 12칸을 모두 돌려준다."""
    s = _scenario(scenario)
    a2 = detect(image_path, plan, scenario)
    exp = _expected(plan)
    rotated = s == "rotated"   # 실제 모듈: 180° 돌린 배치가 계획과 더 잘 맞으면 True

    actual, reasons = {}, {}
    for d in a2["detections"]:
        key = (d["cell"]["day"], d["cell"]["time"])
        cell = actual.setdefault(key, {})
        cell[d["item_code"]] = cell.get(d["item_code"], 0) + 1
        r = reasons.setdefault(key, set())
        r.update(d["flags"])
        if d["conf"] < CONF_MIN:
            r.add("low_conf")

    cells, summary = [], {"ok": 0, "mismatch": 0, "need_review": 0}
    for day in DAYS:
        for time in TIMES:
            key = (day, time)
            e, a, r = exp.get(key, {}), actual.get(key, {}), sorted(reasons.get(key, ()))
            if rotated:
                r = ["rotated"]
            status = "need_review" if r else ("ok" if e == a else "mismatch")
            summary[status] += 1
            cells.append({"day": day, "time": time, "expected": e, "actual": a,
                          "status": status, "reasons": r})
    summary["hitl_required"] = summary["mismatch"] + summary["need_review"] > 0
    return {"prescription_id": plan["prescription_id"], "image_id": a2["image_id"],
            "summary": summary, "cells": cells}


if __name__ == "__main__":
    demo_plan = {
        "prescription_id": "P0001",
        "days": 2,
        "plan": [
            {"day": d, "time": t, "items": [{"item_code": "900000041", "count": 1}]}
            for d in (1, 2) for t in ("morning", "evening")
        ],
    }
    for sc in ("ok", "mismatch", "review", "rotated", "no_tray"):
        try:
            r = inspect("tray_demo.jpg", demo_plan, sc)
            bad = [(c["day"], c["time"], c["status"], c["reasons"]) for c in r["cells"] if c["status"] != "ok"]
            print(f"{sc:9s}: {r['summary']} {bad[:2]}{' …' if len(bad) > 2 else ''}")
        except CVError as e:
            print(f"{sc:9s}: {type(e).__name__} [{e.code}]")
    try:
        inspect("t.jpg", {"prescription_id": "P0002", "days": 4,
                          "plan": [{"day": 4, "time": "morning", "items": []}]})
    except CVError as e:
        print(f"{'4일차':9s}: {type(e).__name__} [{e.code}]")
    try:
        ocr("rx_003.png", "ocr_empty")
    except CVError as e:
        print(f"{'ocr_empty':9s}: {type(e).__name__} [{e.code}]")
    a1 = ocr("rx_003.png")
    print(f"{'ocr':9s}: {a1['image_id']} {a1['doc_type']} {len(a1['lines'])}줄, 첫 줄 line_id={a1['lines'][0]['line_id']}")
