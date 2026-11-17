"""JSON API. 메인 화면의 스크립트(static/app.js)가 부른다.

    POST /api/prescriptions                       처방전 이미지 업로드 (multipart, 필드 image)
    GET  /api/prescriptions/<id>                  상태·카드 내용·타임라인 사건 (화면이 2초마다 다시 읽음)
    GET  /api/prescriptions/<id>/images/<img_id>  원본 이미지
    POST /api/prescriptions/<id>/approve          HITL ① 조제지시서 승인

로그인 안 됨은 401, 다른 약국의 처방은 404, 상태가 맞지 않으면 409로 답한다.
"""
from __future__ import annotations

import functools
import uuid
from typing import Any
from pathlib import Path

from flask import Blueprint, abort, current_app, g, jsonify, request, send_file

from web import db, rx, storage, workflow
from web.agent_gateway import gateway

bp = Blueprint("api", __name__, url_prefix="/api")

# 확장자만 믿지 않고 파일 앞부분으로 형식을 확인한다
_MAGIC = {".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff", ".png": b"\x89PNG\r\n\x1a\n"}
TIME_ORDER = ("morning", "noon", "evening", "bedtime")


def _error(status: int, message: str, **extra):
    return jsonify(error=message, **extra), status


def api_login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            return _error(401, "로그인이 필요합니다. 페이지를 새로고침해 주세요.")
        return view(*args, **kwargs)
    return wrapped


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        abort(404)


def _load(rx_id: str) -> dict:
    row = rx.get(_uuid(rx_id), g.user["pharmacy_id"])
    if row is None:
        abort(404)
    return row


@bp.errorhandler(404)
def _not_found(e):
    return _error(404, "처방을 찾을 수 없습니다.")


@bp.errorhandler(413)
def _too_large(e):
    mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return _error(413, f"이미지가 너무 큽니다. {mb}MB 이하로 올려 주세요.")


@bp.get("/health")
def health():
    """Docker healthcheck용. 로그인 없이 열려 있고 DB 연결만 확인한다."""
    ok = db.ping()
    return jsonify(status="ok" if ok else "db_error", db=ok), (200 if ok else 503)


@bp.post("/prescriptions")
@api_login_required
def upload():
    """처방전 업로드 (F-WEB-02, REQ-WEB-07). 바로 202로 답하고 처리는 백그라운드에서 한다."""
    f = request.files.get("image")
    if f is None or not f.filename:
        return _error(400, "처방전 이미지를 선택해 주세요.")
    ext = Path(f.filename).suffix.lower()
    data = f.read()
    magic = _MAGIC.get(ext)
    if magic is None or not data.startswith(magic):
        return _error(400, "JPG 또는 PNG 이미지만 올릴 수 있습니다.")
    row = workflow.receive(g.user["pharmacy_id"], g.user["id"], data, ext)
    return jsonify(id=str(row["id"]), display_no=row["display_no"], status=row["status"]), 202


def _timeline(row: dict) -> list[dict]:
    """처방 타임라인에 시각을 붙일 사건: 접수와 약사의 HITL 응답. 오래된 것부터 (검수 결과는 11/18에 더한다)."""
    events = [{"event": "received", "at": row["created_at"].isoformat()}]
    for h in rx.history(row["id"]):
        events.append({"event": "response", "hitl_point": h["hitl_point"], "action": h["action"],
                       "at": h["created_at"].isoformat()})
    return events


@bp.get("/prescriptions/<rx_id>")
@api_login_required
def detail(rx_id: str):
    row = _load(rx_id)
    sid = str(row["id"])
    img = rx.first_image(row["id"])
    body = {
        "id": sid,
        "display_no": row["display_no"],
        "status": row["status"],
        "hitl_point": row["hitl_point"],
        "error_code": row["error_code"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "timeline": _timeline(row),
        "image_url": f"/api/prescriptions/{sid}/images/{img['id']}" if img else None,
        "card": None,
    }
    if row["status"] == "failed":
        body["error_message"] = workflow.ERROR_MESSAGES.get(row["error_code"], workflow.UNKNOWN_ERROR)
    elif row["status"] == "waiting_approval":
        pause = gateway().pending(sid)
        if pause is not None and pause.hitl_point == row["hitl_point"]:
            order = pause.payload["order"]
            body["card"] = {**pause.payload,
                            "blockers": workflow.approval_blockers(order, pause.payload.get("plan")),
                            "needs_confirm": workflow.needs_confirm(order),
                            "is_danger": workflow.is_danger(order)}
    elif row["status"] == "waiting_tray":
        body["card"] = {"order": gateway().approved_order(sid)}
    return jsonify(body)


@bp.get("/prescriptions/<rx_id>/images/<image_id>")
@api_login_required
def image(rx_id: str, image_id: str):
    row = rx.image(_uuid(rx_id), _uuid(image_id), g.user["pharmacy_id"])
    if row is None:
        abort(404)
    return send_file(storage.path_of(row["storage_key"]), max_age=3600)


@bp.post("/prescriptions/<rx_id>/approve")
@api_login_required
def approve(rx_id: str):
    """HITL ① 조제지시서 승인(기본)·수정(action=edit)·사유 입력 진행(action=manual) (F-WEB-04·07, REQ-WEB-02·03).

    수정하면 다시 계산해 승인 대기로 돌아온다. 위험(danger)은 승인할 수 없고 수정 또는 manual(사유 필수)만 된다.
    """
    row = _load(rx_id)
    if row["status"] != "waiting_approval" or row["hitl_point"] != "rx_approval":
        return _error(409, "승인을 기다리는 처방이 아닙니다. 화면을 새로고침해 주세요.")
    pause = gateway().pending(str(row["id"]))
    if pause is None or pause.hitl_point != "rx_approval":
        return _error(409, "승인할 조제지시서를 찾지 못했습니다. 화면을 새로고침해 주세요.")
    order = pause.payload["order"]
    body = request.get_json(silent=True) or {}
    if body.get("action") == "edit":   # 수정은 계획이 없어도 받는다. 승인 차단 규칙은 승인에만 건다
        try:
            new_order = workflow.build_edited_order(order, body.get("items"))
        except ValueError as e:
            return _error(400, str(e))
        try:
            gateway().check_reply(str(row["id"]), "rx_approval", {"action": "edit", "order": new_order})
        except ValueError as e:
            return _error(400, str(e))
        if not rx.respond(row["id"], g.user["pharmacy_id"], g.user["id"], "rx_approval",
                          "edit", before=order, after=new_order):
            return _error(409, "이미 처리된 요청입니다. 화면을 새로고침해 주세요.")
        workflow.jobs.submit(workflow.run_resume, row["id"], "rx_approval", {"action": "edit", "order": new_order})
        return jsonify(id=str(row["id"]), status="processing"), 202
    blockers = workflow.approval_blockers(order, pause.payload.get("plan"))
    if blockers:
        return _error(409, blockers[0])
    reply: dict[str, Any] = {"order": order, "plan": pause.payload.get("plan")}
    if body.get("action") == "manual":   # 사유를 적어 그대로 진행. 위험(danger)은 이 길이나 수정으로만 간다
        reason = body.get("reason")
        reason = reason.strip() if isinstance(reason, str) else ""
        if not reason or len(reason) > 500:
            return _error(400, "사유를 1~500자로 적어 주세요.")
        action, reply["reason"] = "manual", reason
    else:
        if workflow.is_danger(order):
            return _error(409, "위험 항목은 그대로 승인할 수 없습니다. 항목을 수정하거나 사유를 적어 진행하세요.")
        if workflow.needs_confirm(order) and body.get("confirmed") is not True:
            return _error(409, "위험 신호와 확인 필요 항목을 확인했다는 칸에 체크해 주세요.")
        action, reason = "approve", None

    try:   # 응답이 Agent가 받을 모양인지 미리 본다: 잘못된 응답이 백그라운드에서 처방을 실패로 만들지 않게
        gateway().check_reply(str(row["id"]), "rx_approval", {"action": action, **reply})
    except ValueError as e:
        return _error(400, str(e))
    if not rx.respond(row["id"], g.user["pharmacy_id"], g.user["id"], "rx_approval",
                      action, before=order, after=order, reason=reason):
        return _error(409, "이미 처리된 승인입니다. 화면을 새로고침해 주세요.")
    workflow.jobs.submit(workflow.run_resume, row["id"], "rx_approval", {"action": action, **reply})
    return jsonify(id=str(row["id"]), status="processing"), 202
