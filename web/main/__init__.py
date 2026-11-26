"""화면 3개 중 로그인 뒤 두 개: 메인(대화)·조제 이력(목록·상세, 11/23)."""
from __future__ import annotations

import math
import uuid
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from flask import Blueprint, abort, current_app, g, render_template, request

from web import db, rx, storage
from web.auth import login_required

bp = Blueprint("main", __name__)


@bp.route("/")
@login_required
def index():
    """메인 대화 화면 (F-WEB-02~11). 11/11부터 업로드·카드를 붙인다."""
    recent = db.query_all(
        "SELECT id, display_no, status, hitl_point, created_at FROM prescriptions "
        "WHERE pharmacy_id = %s ORDER BY created_at DESC LIMIT 20",
        (g.user["pharmacy_id"],),
    )
    return render_template("main/index.html", recent=recent)


# ------------------------------------------------------------ 조제 이력 (F-WEB-13, REQ-WEB-10)
# 화면 문구는 메인 화면(static/app.js)과 같게 둔다
TIMES = {"morning": "아침", "noon": "점심", "evening": "저녁", "bedtime": "취침 전"}
TIME_ORDER = ("morning", "noon", "evening", "bedtime")
DAYS = (1, 2, 3)
RISK = {"none": "해당 없음", "caution": "주의", "warning": "경고", "danger": "위험"}
CELL = {"ok": "일치", "mismatch": "불일치", "need_review": "확인 필요"}
CELL_REASON = {"boundary": "칸 경계에 걸침", "overlap": "알약 겹침", "rotated": "트레이가 뒤집혀 찍힘"}
REVIEW = {"auto": "전 칸 일치", "checked": "약사 확인 후 완료", "manual": "육안 검수(수동 진행)"}
POINT = {"rx_approval": "① 조제지시서 승인", "inspection": "② 검수 확인"}
ACTION = {"approve": "승인", "edit": "수정 후 다시 검수", "manual": "사유 입력 후 진행",
          "confirm": "확인 완료", "redo": "재조제"}
LABELS = {"times": TIMES, "risk": RISK, "cell": CELL, "cell_reason": CELL_REASON, "review": REVIEW,
          "point": POINT, "action": ACTION}


@bp.app_template_filter("label")
def label(key, table: dict):
    """코드 → 화면 문구 (표에 없으면 코드 그대로). 예: {{ times | map('label', labels.times) | join(' · ') }}"""
    return table.get(key, key)


def _day(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


DEFAULT_DAYS = 30   # 기간을 하나도 주지 않으면 오늘부터 한 달 전까지 (11/26)


def _filters(args, today: date | None = None) -> dict:
    """화면 필터(기간 · 위험 단계 · 검수 결과 · 쪽). 잘못된 값은 버린다 (조건 없이 보임).

    기간 칸이 주소에 아예 없으면(처음 들어올 때 · 조건 지우기) 오늘 기준 최근 DEFAULT_DAYS일로 채운다 (defaulted).
    사용자가 날짜 칸을 비우고 조회하면 칸은 있지만 비어 있으므로 기간 없이 전체를 본다.
    """
    defaulted = today is not None and "from" not in args and "to" not in args
    if defaulted:
        since, until = today - timedelta(days=DEFAULT_DAYS), today
    else:
        since, until = _day(args.get("from")), _day(args.get("to"))
    if since and until and since > until:
        since, until = until, since
    try:
        page = max(int(args.get("page", 1)), 1)
    except ValueError:
        page = 1
    risk = args.get("risk") if args.get("risk") in RISK else ""
    review = args.get("review") if args.get("review") in REVIEW else ""
    return {"from": since.isoformat() if since else "", "to": until.isoformat() if until else "",
            "risk": risk, "review": review, "page": page, "defaulted": defaulted}


@bp.route("/history")
@login_required
def history():
    """조제 이력 목록 (F-WEB-13 ①): 기간 · 위험 단계 · 검수 결과로 거른 완료 기록을 최신순 20건씩.

    이력에는 완료된 처방만 있으므로(F-5) 처방 상태 대신 최종 위험 단계와 검수 결과로 거른다 (미확정요소 F-2).
    기간은 화면 시간대(DISPLAY_TZ, 기본 KST)의 날짜로 받아 시작일 0시 ~ 끝일 다음날 0시로 바꾼다.
    """
    tz = ZoneInfo(current_app.config["DISPLAY_TZ"])
    f = _filters(request.args, today=datetime.now(tz).date())
    since = datetime.combine(date.fromisoformat(f["from"]), time(), tz) if f["from"] else None
    until = datetime.combine(date.fromisoformat(f["to"]) + timedelta(days=1), time(), tz) if f["to"] else None
    records, total = rx.dispense_list(g.user["pharmacy_id"], since=since, until=until, risk=f["risk"] or None,
                                      review=f["review"] or None, page=f["page"])
    pages = max(math.ceil(total / rx.HISTORY_PAGE_SIZE), 1)
    # 쪽 이동 주소에 남길 조건. 기본 기간은 넣지 않는다 (다음 쪽에서도 같은 기본값이 다시 채워진다)
    query = {k: v for k, v in f.items() if v and k not in ("page", "defaulted")
             and not (f["defaulted"] and k in ("from", "to"))}
    return render_template("main/history.html", records=records, total=total, f=f, pages=pages, query=query,
                           labels=LABELS, filtered=bool(query))


def _overlay_url(rx_id: str, image_id) -> str | None:
    """검수 결과 그림(cv/pipeline.py)이 있으면 이미지 API 주소 (api._overlay_url과 같은 규칙)."""
    try:
        key = rx.tray_image_key(uuid.UUID(str(image_id))) if image_id else None
    except ValueError:
        key = None
    if key is None or not storage.overlay_path(storage.path_of(key)).is_file():
        return None
    return f"/api/prescriptions/{rx_id}/images/{image_id}?view=overlay"


@bp.route("/history/<rx_id>")
@login_required
def history_detail(rx_id: str):
    """조제 이력 상세 (F-WEB-13 ②): 처방전 원본 · 최종 조제지시서 · 검수 결과 그림과 칸별 판정 · 승인 이력(시간순) ·
    최종 위험 단계와 대조 근거. 검수는 최종 결과만 보이고 트레이 촬영 횟수를 함께 적는다 (미확정요소 E-3)."""
    try:
        rid = uuid.UUID(rx_id)
    except ValueError:
        abort(404)
    rec = rx.dispense_record(rid, g.user["pharmacy_id"])
    if rec is None:
        abort(404)
    sid = str(rec["id"])
    order = rec["rx_json"] or {}
    a3 = rec["inspection_json"] or {}
    names = {it.get("item_code"): it.get("name") for it in order.get("items", []) if it.get("item_code")}
    cells = {(c.get("day"), c.get("time")): c for c in a3.get("cells", [])}
    grid = [(t, [cells.get((d, t)) for d in DAYS]) for t in TIME_ORDER]
    rx_img = rx.first_image(rid)
    tray_id = a3.get("image_id")
    if not tray_id:   # 육안 검수로 끝난 기록은 검수 결과가 없다: 마지막으로 올린 트레이 사진을 보인다
        trays = rx.tray_images(rid)
        tray_id = str(trays[-1]["id"]) if trays else None
    return render_template(
        "main/history_detail.html", rec=rec, order=order, a3=a3, names=names, grid=grid, days=DAYS,
        risk=(order.get("risk_check") or {}), approvals=rx.approval_log(rid), attempts=len(rx.inspections(rid)),
        rx_image_url=f"/api/prescriptions/{sid}/images/{rx_img['id']}" if rx_img else None,
        tray_image_url=f"/api/prescriptions/{sid}/images/{tray_id}" if tray_id else None,
        overlay_url=_overlay_url(sid, tray_id), labels=LABELS)
