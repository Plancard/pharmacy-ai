"""칸별 검수 판정(F-CV-05) — 트레이 사진 + 조제 계획(A.4) → 검수 결과(A.3).

    from cv.tray_inspect import inspect_tray, judge
    a3 = inspect_tray("var/uploads/<처방>/<사진>.jpg", plan, image_id="<사진 ID>")   # 보정 → 검출 → 칸 배정 → 판정
    a3 = judge(detections, plan, image_id="…")   # 검출 목록(보정 이미지 픽셀 bbox)만으로 판정 (모델 없이 시험·평가)

파일 이름을 inspect.py로 하지 않은 이유: CV 스크립트는 `python cv/xxx.py`로 돌리는데, 그러면 cv/가 sys.path 맨 앞에
들어가 표준 라이브러리 inspect 대신 이 파일이 읽힌다 (dataclasses·ultralytics가 깨짐). 그래서 tray_inspect다.

처리 절차 (기능분석서 F-CV-05, 기준값은 cv/tray.py·cv/detect.py의 11/17 확정값)
  0. 트레이 보정(F-CV-02, cv.tray.rectify) → 알약 검출(F-CV-03, cv.detect.detect_pills, 기준선 conf 0.70)
  1. 뒤집힘: 검출을 그대로 한 번, 180° 돌려서 한 번 칸에 배정해 계획과 대조한다. 돌린 쪽의 일치 칸이 더 많으면
     12칸 모두 need_review(사유 rotated) — 재촬영 안내. 같으면 뒤집히지 않은 것으로 본다.
  2. 아니면 칸마다 (F-CV-04 cv.tray.assign_cells의 칸·플래그로)
       boundary · overlap 플래그가 있는 알약이 있으면 need_review (사유 boundary · overlap)
       계획의 품목기준코드별 개수(expected) == 검출 개수(actual)이면 ok, 다르면 mismatch
       (계획상 빈 칸에 알약이 있어도 mismatch)
  중심점이 트레이 밖인 검출(cell None)은 판정에서 뺀다.
  기준선 미만 검출은 detect_pills가 버리므로 저신뢰(low_conf) 판정은 없다.

오류: 트레이를 못 찾으면 cv.stub.TrayNotFound(E_TRAY_NOT_FOUND). 계획이 트레이 밖 칸을 가리키면 PlanRangeError,
품목기준코드가 없는 검출 클래스(약품 사전 누락)는 CVError(E_CV_INFER) — 설정 오류라 조용히 넘기지 않는다.
반환 형식은 cv/stub.inspect와 같다 (Agent의 agent.schema.Inspection으로 검사된다).

정답표 평가 (저장소 루트에서, 모델 필요. test 그룹은 11/20 1회 원칙이라 기본으로 막는다)
  python -m cv.tray_inspect --list data/splits/pose_val.txt --csv data/tray/inspect_check.csv
  → 사진 × 칸마다 판정을 cells.csv(계획 planned · 실제 actual · error · flags)와 대조해
    '일부러 잘못 담은 칸을 놓치지 않았는지'(수용 기준: 모두 mismatch 또는 need_review)와
    '정상 칸 오판정 비율'(수용 기준 5% 이하)을 출력한다.

뒤집힘 확인 (rot_s01~05 = g20 배치를 180° 돌려 찍은 사진. 꼭짓점 test라 --allow-test가 있어야 돈다)
  python -m cv.tray_inspect --list rot.txt --group g20 --rotated --allow-test --csv data/tray/inspect_rot.csv
  → 사진마다 12칸 모두 rotated로 판정됐는지 (수용 기준: 전부)
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from cv.stub import CVError, DAYS, TIMES, PlanRangeError
from cv.warp import OUT_H, OUT_W

REASON_ORDER = ("boundary", "overlap")
NO_ERROR = {"", "ok", "none", "normal", "-", "0", "no", "false", "정상", "없음"}   # cells.csv error 열의 '틀린 칸 아님' 값


# ---------------------------------------------------------------- 판정 (모델 없이)
def expected_counts(plan):
    """A.4 → {(day, time): {item_code: 개수}} (계획에 없는 칸은 빠짐). 트레이 밖 칸이면 PlanRangeError."""
    exp = {}
    for c in plan["plan"]:
        if c["day"] not in DAYS or c["time"] not in TIMES:
            raise PlanRangeError(f"트레이에 없는 칸 day={c['day']}, time={c['time']}")
        cell = exp.setdefault((c["day"], c["time"]), {})
        for it in c["items"]:
            if it["count"]:
                cell[it["item_code"]] = cell.get(it["item_code"], 0) + it["count"]
    return exp


def _assign(detections):
    from cv import tray      # numpy만 쓰는 함수지만 tray 모듈이 cv2를 읽으므로 쓸 때 읽는다
    return tray.assign_cells(detections)


def _actual(assigned):
    """칸 배정된 검출 → ({(day, time): {item_code: 개수}}, {(day, time): {사유}}). 트레이 밖 검출은 뺀다."""
    actual, reasons = {}, {}
    for d in assigned:
        if d.get("cell") is None:
            continue
        if not d.get("item_code"):
            raise CVError(f"검출 클래스 {d.get('class_id')}의 품목기준코드가 약품 사전에 없습니다")
        key = (d["cell"]["day"], d["cell"]["time"])
        cell = actual.setdefault(key, {})
        cell[d["item_code"]] = cell.get(d["item_code"], 0) + 1
        reasons.setdefault(key, set()).update(f for f in d.get("flags", []) if f in REASON_ORDER)
    return actual, reasons


def flip_180(detections, size=(OUT_W, OUT_H)):
    """보정 이미지 안에서 박스를 180° 돌린다 (트레이를 거꾸로 놓고 찍었다면 계획과 맞을 배치)."""
    w, h = size
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        out.append({**d, "bbox": [w - x2, h - y2, w - x1, h - y1]})
    return out


def _ok_cells(exp, actual):
    return sum(exp.get((d, t), {}) == actual.get((d, t), {}) for d in DAYS for t in TIMES)


def judge(detections, plan, image_id="tray"):
    """검출 목록(보정 이미지 픽셀 bbox, item_code 포함) + A.4 → A.3. 12칸을 위→아래·1일→3일 순서가 아니라
    cv/stub.inspect와 같은 순서(1일차 아침 … 3일차 취침 전)로 돌려준다."""
    exp = expected_counts(plan)
    assigned = _assign(detections)
    actual, reasons = _actual(assigned)
    flipped_actual, _ = _actual(_assign(flip_180(detections)))
    rotated = _ok_cells(exp, flipped_actual) > _ok_cells(exp, actual)

    cells, summary = [], {"ok": 0, "mismatch": 0, "need_review": 0}
    for day in DAYS:
        for time in TIMES:
            key = (day, time)
            e, a = exp.get(key, {}), actual.get(key, {})
            if rotated:
                why = ["rotated"]
            else:
                why = [r for r in REASON_ORDER if r in reasons.get(key, ())]
            status = "need_review" if why else ("ok" if e == a else "mismatch")
            summary[status] += 1
            cells.append({"day": day, "time": time, "expected": dict(e), "actual": dict(a),
                          "status": status, "reasons": why})
    summary["hitl_required"] = summary["mismatch"] + summary["need_review"] > 0
    return {"prescription_id": plan["prescription_id"], "image_id": str(image_id),
            "summary": summary, "cells": cells}


# ---------------------------------------------------------------- 서비스 함수
def inspect_tray(image_path, plan, *, image_id=None, overlay_path=None, model=None, seg_model=None):
    """F-CV-05 인터페이스: 트레이 사진 + A.4 → A.3 (cv/stub.inspect와 같은 형식·예외).

    image_id를 안 주면 파일 이름(확장자 뺀 것)을 쓴다. 웹은 자기 사진 ID를 넘긴다 (A.3 image_id = images.id).
    트레이를 못 찾으면 TrayNotFound(E_TRAY_NOT_FOUND).
    실제 처리는 cv.pipeline.run이 한다 (11/23). overlay_path를 주면 결과 오버레이 그림도 저장한다.
    """
    from cv.pipeline import run

    return run(image_path, plan, image_id=image_id, overlay_path=overlay_path, model=model, seg_model=seg_model)


# ---------------------------------------------------------------- 정답표 평가
def plan_from_counts(group_cells, rx_id):
    """{(day, time): Counter(class_id)} → A.4 (class_id는 약품 사전의 품목기준코드로)."""
    from cv.detect import item_codes

    codes = item_codes()
    plan = []
    for (day, time), cnt in sorted(group_cells.items(), key=lambda kv: (kv[0][0], TIMES.index(kv[0][1]))):
        items = [{"item_code": codes[c], "count": n} for c, n in sorted(cnt.items()) if n]
        if items:
            plan.append({"day": day, "time": time, "items": items})
    return {"prescription_id": rx_id, "days": max([c["day"] for c in plan], default=1), "plan": plan}


def _parse(s):
    out = Counter()
    for part in (s or "").replace(" ", "").split(";"):
        if part:
            c, n = part.split(":")
            out[int(c)] += int(n)
    return out


def load_truth(path):
    """cells.csv(group,day,time,planned,actual,error,flags) → {group: {(day, time): {"expected", "actual", "error", "flags"}}}.

    expected = planned(계획한 배치), actual = 실제로 담은 것, error = 일부러 틀린 종류(wrong · missing · extra, 정상은 빈칸).
    그룹의 planned가 모두 비어 있으면(추가 촬영분 cells_add: planned 없이 actual만 적음, 틀린 칸 없음) expected = actual.
    """
    groups = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = r["group"].strip()
            planned = r.get("planned", r.get("expected"))
            groups.setdefault(g, {})[(int(r["day"]), r["time"].strip())] = {
                "expected": _parse(planned), "actual": _parse(r.get("actual")),
                "error": "" if (r.get("error") or "").strip().lower() in NO_ERROR else r["error"].strip(),
                "flags": set((r.get("flags") or "").replace(";", " ").replace(",", " ").replace("|", " ").split()),
                "has_expected": bool((planned or "").strip())}
    for cells in groups.values():
        if not any(c["has_expected"] for c in cells.values()):
            for c in cells.values():
                c["expected"] = Counter(c["actual"])
    return groups


def evaluate(results, truth):
    """results: [(사진, group, A.3 | None)] → 집계 dict와 칸별 행.

    정답 칸 분류: wrong(계획 ≠ 실제, 일부러 잘못 담음) · flagged(정답표에 boundary·overlap 표시) · normal(그 밖)
    정답표 error 열과 계획 ≠ 실제가 어긋나는 칸은 stat["truth_conflict"]로 센다 (정답표 확인용)
    수용 기준(F-CV-05): wrong 칸은 모두 ok가 아니어야 하고, normal 칸의 오판정(ok가 아님)은 5% 이하.
    """
    rows, stat = [], Counter()
    for name, group, a3 in results:
        if a3 is None:
            stat["retake"] += 1
            continue
        stat["photos"] += 1
        if any("rotated" in c["reasons"] for c in a3["cells"]):
            stat["rotated_photos"] += 1
        for c in a3["cells"]:
            t = truth[group][(c["day"], c["time"])]
            if bool(t.get("error")) != (t["expected"] != t["actual"]):
                stat["truth_conflict"] += 1
            kind = ("wrong" if t["expected"] != t["actual"] else
                    "flagged" if t["flags"] & set(REASON_ORDER) else "normal")
            stat[f"{kind}_cells"] += 1
            bad = c["status"] != "ok"
            if kind == "wrong" and not bad:
                stat["wrong_missed"] += 1
            if kind == "normal" and bad:
                stat["normal_false_alarm"] += 1
            rows.append({"file": name, "group": group, "day": c["day"], "time": c["time"], "truth": kind,
                         "error": t.get("error", ""),
                         "status": c["status"], "reasons": " ".join(c["reasons"]),
                         "expected": ";".join(f"{k}:{v}" for k, v in sorted(c["expected"].items())),
                         "detected": ";".join(f"{k}:{v}" for k, v in sorted(c["actual"].items()))})
    return stat, rows


_RF_SUFFIX = re.compile(r"_(jpe?g|png)\.rf\.[0-9a-zA-Z]+$")    # Roboflow 내보내기: g07_s01_jpg.rf.<해시>
_GROUP = re.compile(r"^(g\d+)_")


def group_of(path, photo_group, groups):
    """사진 경로 → 정답표 그룹. photos.csv 이름 → Roboflow 이름을 원래 이름으로 되돌려서 → 이름 앞 gNN_ 순으로 찾는다."""
    stem = Path(path).stem
    for name in (stem, _RF_SUFFIX.sub("", stem)):
        if photo_group.get(name) in groups:
            return photo_group[name]
    m = _GROUP.match(_RF_SUFFIX.sub("", stem))
    return m.group(1) if m and m.group(1) in groups else None


def evaluate_rotated(results):
    """뒤집어 찍은 사진: 12칸 모두 rotated로 판정됐는지. → (집계, 사진별 행)"""
    stat, rows = Counter(), []
    for name, group, a3 in results:
        if a3 is None:
            stat["retake"] += 1
            rows.append({"file": name, "group": group, "rotated": "", "result": "재촬영(트레이 못 찾음)"})
            continue
        stat["photos"] += 1
        hit = all(c["reasons"] == ["rotated"] for c in a3["cells"])
        stat["rotated_hit" if hit else "rotated_missed"] += 1
        bad = [f"{c['day']}일 {c['time']} {c['status']}" for c in a3["cells"] if c["status"] != "ok"]
        rows.append({"file": name, "group": group, "rotated": "yes" if hit else "no",
                     "result": "뒤집힘 판정" if hit else ("놓침: " + ", ".join(bad) if bad else "놓침: 전 칸 일치로 봄")})
    return stat, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="검수 판정 평가: 사진마다 inspect_tray를 돌려 cells.csv 정답과 대조")
    ap.add_argument("--list", required=True, help="사진 경로 목록 (data/splits/*.txt 형식, 원본 트레이 사진)")
    ap.add_argument("--tray", default="data/tray", help="cells.csv · photos.csv 위치")
    ap.add_argument("--model", default=None, help="검출 가중치 (기본 cv.detect.DET_MODEL)")
    ap.add_argument("--seg-model", default=None, help="트레이 분할 가중치 (기본 cv.tray.SEG_MODEL)")
    ap.add_argument("--csv", default="data/tray/inspect_check.csv")
    ap.add_argument("--allow-test", action="store_true", help="test 그룹도 돌린다 (11/20 최종 1회에만)")
    ap.add_argument("--group", help="목록의 모든 사진을 이 그룹 배치로 본다 (예: rot 사진은 g20)")
    ap.add_argument("--rotated", action="store_true", help="목록이 180° 뒤집어 찍은 사진: 전부 rotated로 판정됐는지만 본다")
    args = ap.parse_args(argv)

    TEST_GROUPS = {"g03", "g06", "g08", "g10", "g11", "g20"}     # 11/11 확정 분할 (rot 사진은 꼭짓점 test)
    if args.group and args.group in TEST_GROUPS | {"rot"} and not args.allow_test:
        sys.exit(f"{args.group}는 test 그룹입니다 (rot 사진도 꼭짓점 test). 11/20 1회 평가 때만 --allow-test로 돌리세요.")

    from ultralytics import YOLO
    from cv import detect, tray
    from cv.stub import TrayNotFound
    truth = load_truth(Path(args.tray) / "cells.csv")
    with open(Path(args.tray) / "photos.csv", newline="", encoding="utf-8-sig") as f:
        photo_group = {Path(r["file"].strip()).stem: r["group"].strip() for r in csv.DictReader(f)}

    det_model = YOLO(args.model or detect.DET_MODEL)
    seg_model = YOLO(args.seg_model or tray.SEG_MODEL)
    results, skipped, no_truth = [], Counter(), []
    for p in Path(args.list).read_text(encoding="utf-8").split():
        p = Path(p)
        if p.stem.startswith("rot_") and not args.allow_test:
            skipped["rot 사진(꼭짓점 test)"] += 1
            continue
        group = args.group if args.group in truth else group_of(p, photo_group, truth)
        if args.group and args.group not in truth:
            sys.exit(f"cells.csv에 {args.group} 그룹이 없습니다")
        if group is None:
            skipped["정답표 없음(빈 트레이·꼭짓점 전용·회전 등)"] += 1
            no_truth.append(p.name)
            continue
        if group in TEST_GROUPS and not args.allow_test:
            skipped["test 그룹(11/20 1회)"] += 1
            continue
        plan = plan_from_counts({k: v["expected"] for k, v in truth[group].items()}, group)
        try:
            a3 = inspect_tray(p, plan, model=det_model, seg_model=seg_model)
        except TrayNotFound:
            a3 = None
        results.append((p.name, group, a3))

    if args.rotated:
        stat, rows = evaluate_rotated(results)
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=["file", "group", "rotated", "result"])
            wr.writeheader()
            wr.writerows(rows)
        print(f"뒤집어 찍은 사진 {stat['photos'] + stat['retake']}장: 뒤집힘 판정 {stat['rotated_hit']}장, "
              f"놓침 {stat['rotated_missed']}장, 재촬영 {stat['retake']}장 (기준: 전부 뒤집힘 판정)")
        for r in rows:
            print(f"  {r['file']}: {r['result']}")
        return 0 if stat["rotated_missed"] == 0 and stat["retake"] == 0 else 1

    stat, rows = evaluate(results, truth)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "group", "day", "time", "truth", "error", "status", "reasons",
                                           "expected", "detected"])
        wr.writeheader()
        wr.writerows(rows)
    normal = stat["normal_cells"]
    rate = stat["normal_false_alarm"] / normal if normal else 0.0
    print(f"사진 {stat['photos']}장 판정, 재촬영 {stat['retake']}장, 뒤집힘 판정 {stat['rotated_photos']}장"
          + (f", 건너뜀 {dict(skipped)}" if skipped else ""))
    print(f"  잘못 담은 칸 {stat['wrong_cells']}개 중 놓침 {stat['wrong_missed']}개 (기준: 0)")
    print(f"  정상 칸 {normal}개 중 오판정 {stat['normal_false_alarm']}개 = {rate:.1%} (기준: 5% 이하)")
    print(f"  정답표에 경계·겹침 표시된 칸 {stat['flagged_cells']}개 (참고)")
    if stat["truth_conflict"]:
        print(f"  주의: 정답표 error 열과 planned ≠ actual이 어긋나는 칸 {stat['truth_conflict']}개 (cells.csv 확인)")
    print(f"  칸별 결과 → {args.csv}")
    if no_truth:
        print(f"  정답표 없는 사진 예: {', '.join(no_truth[:5])}")
    return 0 if stat["wrong_missed"] == 0 and rate <= 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
