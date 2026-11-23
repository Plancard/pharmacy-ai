"""트레이 검수 파이프라인 (F-CV-05 진입점) — 트레이 사진 + 조제 계획(A.4) → 검수 결과(A.3) + 결과 오버레이 그림.

    from cv.pipeline import run
    a3 = run("var/uploads/<처방>/<사진>.jpg", plan, image_id="<사진 ID>",
             overlay_path="var/uploads/<처방>/<사진>.overlay.jpg")      # overlay_path는 생략 가능

처리 절차
  1. 계획 검사 (트레이 밖 칸이면 PlanRangeError — 모델을 돌리기 전에)
  2. 트레이 보정 (F-CV-02, cv.tray.rectify → 960×1280). 트레이를 못 찾으면 TrayNotFound(E_TRAY_NOT_FOUND) → 웹이 재촬영 요청
  3. 알약 검출 (F-CV-03, cv.detect.detect_pills, 기준선 conf 0.70)
  4. 칸 배정 · 판정 (F-CV-04 · F-CV-05, cv.tray_inspect.judge) → A.3
  5. overlay_path를 주면 결과 오버레이 그림을 저장한다 (아래)

A.3 형식은 바꾸지 않는다 (Agent agent.schema.Inspection은 정해진 필드만 받는다). 그림은 A.3 밖의 파일이고,
웹이 트레이 사진 옆에 둘 경로를 정해 넘긴다. cv.tray_inspect.inspect_tray는 이 run을 부른다 (예전 호출 그대로 동작).

결과 오버레이 그림 (보정된 트레이 960×1280 위에 그린다, JPEG)
  - 4×3 칸 선. 칸 순서는 웹 판정 표와 같다: 위→아래 아침·점심·저녁·취침 전, 왼→오른 1·2·3일
  - 칸 판정 색: 불일치(mismatch) 빨강 · 확인 필요(need_review) 주황은 반투명 칠 + 굵은 테두리,
    일치(ok)는 초록 테두리만, 계획도 검출도 없는 빈 칸은 칠하지 않는다
  - 검출 박스: 흰 테두리(어두운 외곽선). 경계(boundary) · 겹침(overlap)에 걸린 박스는 주황 굵은 테두리
  - 글자는 쓰지 않는다 (OpenCV는 한글을 못 그리고, 판정 글자는 웹 표가 보여 준다. 색만으로 구분하지 않게 하는 것도 표 몫)
  - 뒤집힘(rotated, 12칸 모두 need_review)이면 그리지 않는다 — 재촬영 안내 대상이고 뒤집힌 그림은 헷갈린다
  - 그림 저장이 실패해도 검수 결과는 그대로 돌려준다 (그림은 보조 자료). 실패는 로그만 남긴다

사진 한 장 확인 (저장소 루트, 모델 필요. test 그룹 사진은 쓰지 않는다)
  python -m cv.pipeline <트레이 사진> --plan <A.4 json> --overlay out.jpg
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from cv.stub import DAYS, TIMES
from cv.warp import OUT_H, OUT_W

log = logging.getLogger(__name__)

# BGR
COLOR = {"ok": (76, 175, 80), "mismatch": (60, 60, 220), "need_review": (0, 165, 255)}
GRID = (160, 160, 160)
BOX = (255, 255, 255)
BOX_EDGE = (40, 40, 40)
FLAG = COLOR["need_review"]
FILL_ALPHA = 0.28          # 불일치 · 확인 필요 칸 칠 투명도
JPEG_QUALITY = 85


def _cell_rect(day, time, size=(OUT_W, OUT_H)):
    cw, ch = size[0] / len(DAYS), size[1] / len(TIMES)
    col, row = DAYS.index(day), TIMES.index(time)
    return int(round(col * cw)), int(round(row * ch)), int(round((col + 1) * cw)), int(round((row + 1) * ch))


def _is_empty(cell):
    return not any(cell.get("expected", {}).values()) and not any(cell.get("actual", {}).values())


def is_rotated(a3):
    return bool(a3["cells"]) and all(c["reasons"] == ["rotated"] for c in a3["cells"])


def render_overlay(img, assigned, a3):
    """보정 이미지(BGR) + 칸 배정된 검출(cv.tray.assign_cells 결과) + A.3 → 오버레이 그림 (새 배열, 입력은 안 바꾼다)."""
    h, w = img.shape[:2]
    out = img.copy()
    size = (w, h)

    fill = out.copy()
    filled = False
    for c in a3["cells"]:
        if c["status"] in ("mismatch", "need_review"):
            x1, y1, x2, y2 = _cell_rect(c["day"], c["time"], size)
            cv2.rectangle(fill, (x1, y1), (x2, y2), COLOR[c["status"]], -1)
            filled = True
    if filled:
        out = cv2.addWeighted(fill, FILL_ALPHA, out, 1 - FILL_ALPHA, 0)

    for i in range(1, len(DAYS)):                                   # 칸 선
        x = int(round(i * w / len(DAYS)))
        cv2.line(out, (x, 0), (x, h), GRID, 2)
    for i in range(1, len(TIMES)):
        y = int(round(i * h / len(TIMES)))
        cv2.line(out, (0, y), (w, y), GRID, 2)

    for d in assigned:                                              # 검출 박스
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        flagged = any(f in ("boundary", "overlap") for f in d.get("flags", []))
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_EDGE, 6 if flagged else 4)
        cv2.rectangle(out, (x1, y1), (x2, y2), FLAG if flagged else BOX, 4 if flagged else 2)

    for c in a3["cells"]:                                           # 칸 테두리 (박스 위에 그려 칸 판정이 먼저 보이게)
        if c["status"] == "ok" and _is_empty(c):
            continue
        x1, y1, x2, y2 = _cell_rect(c["day"], c["time"], size)
        t = 3 if c["status"] == "ok" else 8
        cv2.rectangle(out, (x1 + t // 2, y1 + t // 2), (x2 - 1 - t // 2, y2 - 1 - t // 2), COLOR[c["status"]], t)
    return out


def save_overlay(img, path):
    """JPEG로 저장한다. 반쯤 쓴 파일을 웹이 내보내지 않게 임시 파일에 쓰고 이름을 바꾼다."""
    path = Path(path)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("오버레이 그림을 JPEG로 만들지 못했습니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)
    return path


def run(image_path, plan, *, image_id=None, overlay_path=None, model=None, seg_model=None):
    """F-CV-05 진입점: 트레이 사진 + A.4 → A.3. overlay_path를 주면 결과 오버레이 그림도 저장한다.

    image_id를 안 주면 파일 이름(확장자 뺀 것)을 쓴다. 웹은 자기 사진 ID를 넘긴다 (A.3 image_id = images.id).
    트레이를 못 찾으면 TrayNotFound(E_TRAY_NOT_FOUND).
    """
    from cv import detect, tray                 # 모델 모듈은 쓸 때 읽는다 (시험은 모듈 함수를 바꿔 끼운다)
    from cv.tray_inspect import expected_counts, judge

    path = Path(image_path)
    expected_counts(plan)                       # 계획부터 검사한다 (모델을 돌리기 전에)
    r = tray.rectify(path, model=seg_model)
    detections = detect.detect_pills(r.image, model)
    a3 = judge(detections, plan, image_id or path.stem)

    if overlay_path is not None and not is_rotated(a3):
        try:
            save_overlay(render_overlay(r.image, tray.assign_cells(detections), a3), overlay_path)
        except Exception as e:                  # 그림은 보조 자료 — 검수 결과는 그대로 돌려준다
            log.warning("오버레이 그림 저장 실패 (%s): %s", overlay_path, e)
    return a3


def main(argv=None):
    ap = argparse.ArgumentParser(description="트레이 사진 한 장 검수 + 오버레이 그림")
    ap.add_argument("image")
    ap.add_argument("--plan", required=True, help="A.4 조제 계획 json")
    ap.add_argument("--overlay", help="오버레이 그림 저장 경로 (.jpg)")
    args = ap.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    a3 = run(args.image, plan, overlay_path=args.overlay)
    json.dump(a3, sys.stdout, ensure_ascii=False, indent=2)
    print()
    if args.overlay:
        print(f"→ {args.overlay}" if Path(args.overlay).exists() else "오버레이 그림 없음 (뒤집힘 또는 저장 실패)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
