"""JSON API. 메인 화면의 스크립트(static/app.js)가 부른다.

    POST /api/prescriptions                       처방전 이미지 업로드 (multipart, 필드 image)
    GET  /api/prescriptions/<id>                  상태·카드 내용·타임라인 사건 (화면이 2초마다 다시 읽음)
    GET  /api/prescriptions/<id>/images/<img_id>  원본 이미지 (?view=rect 펴진 처방전, ?view=overlay 트레이 검수 결과 그림)
    POST /api/prescriptions/<id>/approve          HITL ① 조제지시서 승인
    POST /api/prescriptions/<id>/tray             트레이 사진 업로드 (multipart, 필드 image. 트레이 대기 처방에만)
    POST /api/prescriptions/<id>/inspection       HITL ② 검수 확인 {action: confirm | redo}
    POST /api/prescriptions/<id>/retry            실패: 같은 처방전 사진으로 새 처방 (일시 오류)
    POST /api/prescriptions/<id>/retake           실패: 새로 찍은 처방전 사진으로 새 처방 (multipart, 필드 image)
    POST /api/prescriptions/<id>/save-retry       실패(E_DB): 조제 기록 저장만 다시
    POST /api/prescriptions/<id>/manual-inspection  트레이 검출 모델 오류(E_CV_INFER): 육안 검수 후 수동 진행 {reason}

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


def _read_image(what: str) -> tuple[bytes, str] | tuple[Any, int]:
    """업로드 필드 image를 읽는다. → (바이트, 확장자) 또는 오류 응답 (응답, 상태 코드)."""
    f = request.files.get("image")
    if f is None or not f.filename:
        return _error(400, f"{what} 이미지를 선택해 주세요.")
    ext = Path(f.filename).suffix.lower()
    data = f.read()
    magic = _MAGIC.get(ext)
    if magic is None or not data.startswith(magic):
        return _error(400, "JPG 또는 PNG 이미지만 올릴 수 있습니다.")
    return data, ext


@bp.post("/prescriptions")
@api_login_required
def upload():
    """처방전 업로드 (F-WEB-02, REQ-WEB-07). 바로 202로 답하고 처리는 백그라운드에서 한다."""
    got = _read_image("처방전")
    if not isinstance(got[0], bytes):
        return got
    data, ext = got
    row = workflow.receive(g.user["pharmacy_id"], g.user["id"], data, ext)
    return jsonify(id=str(row["id"]), display_no=row["display_no"], status=row["status"]), 202


NOT_WAITING_TRAY = "조제지시서 승인이 끝난 처방(트레이 대기)에만 트레이 사진을 올릴 수 있습니다. 화면을 새로고침해 주세요."


@bp.post("/prescriptions/<rx_id>/tray")
@api_login_required
def upload_tray(rx_id: str):
    """트레이 사진 업로드 (F-WEB-02 ④, REQ-WEB-04). 트레이 대기 → 처리 중으로 바꾸고 검수는 백그라운드에서 한다.

    승인이 끝나지 않았거나 이미 검수 중인 처방은 409. 검수 결과는 상세 조회(GET)로 본다:
    전 칸 일치면 done, 불일치·확인 필요가 있으면 waiting_approval(inspection), 트레이를 못 찾으면 waiting_tray + retake.
    """
    row = _load(rx_id)
    if row["status"] != "waiting_tray":
        return _error(409, NOT_WAITING_TRAY)
    got = _read_image("트레이")
    if not isinstance(got[0], bytes):
        return got
    data, ext = got
    image_id = workflow.receive_tray(row["id"], g.user["pharmacy_id"], data, ext)
    if image_id is None:
        return _error(409, NOT_WAITING_TRAY)
    return jsonify(id=str(row["id"]), image_id=str(image_id), status="processing"), 202


def _image_url(rx_id: str, image_id: Any) -> str | None:
    return f"/api/prescriptions/{rx_id}/images/{image_id}" if image_id else None


def _overlay_url(rx_id: str, image_id: Any) -> str | None:
    """트레이 검수 결과 그림(cv/pipeline.py, 기능분석서 [K])이 있으면 그 주소. 없으면(가짜 CV·뒤집힘·저장 실패) None."""
    try:
        key = rx.tray_image_key(uuid.UUID(str(image_id))) if image_id else None
    except ValueError:
        key = None
    if key is None or not storage.overlay_path(storage.path_of(key)).is_file():
        return None
    return f"{_image_url(rx_id, image_id)}?view=overlay"


def _timeline(row: dict, inspections: list[dict]) -> list[dict]:
    """처방 타임라인에 시각을 붙일 사건: 접수, 약사의 HITL 응답, 트레이 사진 업로드, 트레이 검수 결과. 오래된 것부터.

    트레이 사진은 올렸는데 검수 결과가 없으면 검수 중이거나 다시 찍어야 하는 사진이다 (화면이 구분해 보인다).
    """
    events = [{"event": "received", "at": row["created_at"]}]
    for t in rx.tray_images(row["id"]):
        events.append({"event": "tray_uploaded", "image_id": str(t["id"]), "at": t["created_at"]})
    for h in rx.history(row["id"]):
        events.append({"event": "response", "hitl_point": h["hitl_point"], "action": h["action"],
                       "at": h["created_at"]})
    for i in inspections:
        events.append({"event": "inspected", "image_id": str(i["image_id"]), "hitl_required": i["hitl_required"],
                       "at": i["created_at"]})
    events.sort(key=lambda e: e["at"])
    return [{**e, "at": e["at"].isoformat()} for e in events]


@bp.get("/prescriptions/<rx_id>")
@api_login_required
def detail(rx_id: str):
    row = _load(rx_id)
    sid = str(row["id"])
    img = rx.first_image(row["id"])
    inspections = rx.inspections(row["id"])
    body = {
        "id": sid,
        "display_no": row["display_no"],
        "status": row["status"],
        "hitl_point": row["hitl_point"],
        "error_code": row["error_code"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "timeline": _timeline(row, inspections),
        "image_url": f"/api/prescriptions/{sid}/images/{img['id']}" if img else None,
        "card": None,
        "failed_lines": None,
        "retryable": False,
    }
    if row["status"] == "failed":
        body["error_message"] = workflow.ERROR_MESSAGES.get(row["error_code"], workflow.UNKNOWN_ERROR)
        body["retryable"] = row["error_code"] in workflow.RETRY_CODES and img is not None   # [재시도] (Q6)
        body["save_retryable"] = row["error_code"] in workflow.SAVE_RETRY_CODES             # [저장 다시 시도] (6장)
        body["retake_allowed"] = row["error_code"] not in workflow.SAVE_RETRY_CODES          # [다른 사진으로 다시 올리기]
        body["failed_lines"] = gateway().failed_lines(sid)   # 구조화 실패 때 처방전 위에 강조할 줄 {ocr_size, lines} (Q6)
    elif row["status"] == "waiting_approval":
        pause = gateway().pending(sid)
        if pause is not None and pause.hitl_point == row["hitl_point"] == "rx_approval":
            order = pause.payload["order"]
            body["card"] = {**pause.payload,
                            "blockers": workflow.approval_blockers(order, pause.payload.get("plan")),
                            "needs_confirm": workflow.needs_confirm(order),
                            "is_danger": workflow.is_danger(order)}
        elif pause is not None and pause.hitl_point == row["hitl_point"] == "inspection":
            # HITL ② 검수 결과 카드 (F-WEB-09): A.3 + 조제 계획 + 칸별 안내 + 트레이 사진
            a3 = pause.payload.get("inspection") or {}
            body["card"] = {**pause.payload, "tray_image_url": _image_url(sid, a3.get("image_id")),
                            "tray_overlay_url": _overlay_url(sid, a3.get("image_id"))}
    elif row["status"] == "waiting_tray":
        # 트레이를 채울 때 볼 칸 배치(A.4)와, 재조제로 돌아왔으면 직전 검수 결과(A.3 · 사진)를 함께 준다 (11/26 화면)
        card: dict[str, Any] = {"order": gateway().approved_order(sid), "plan": gateway().approved_plan(sid),
                                "retake": None, "previous": None}
        if inspections:
            last = inspections[-1]
            card["previous"] = {"inspection": last["result_json"], "tray_image_url": _image_url(sid, last["image_id"]),
                                "tray_overlay_url": _overlay_url(sid, last["image_id"])}
        err = gateway().tray_error(sid)
        if err:   # 트레이를 못 찾는 등으로 다시 찍어야 함. 검출 모델 오류면 육안 검수 후 수동 진행도 열어 준다 (6장)
            code = err.get("code")
            card["retake"] = {"code": code,
                              "message": workflow.TRAY_ERROR_MESSAGES.get(code) or workflow.ERROR_MESSAGES.get(
                                  code, err.get("notice") or workflow.UNKNOWN_ERROR),
                              "manual_allowed": code in workflow.MANUAL_TRAY_CODES}
        body["card"] = card
    elif row["status"] == "done" and inspections:
        last = inspections[-1]
        body["card"] = {"inspection": last["result_json"], "tray_image_url": _image_url(sid, last["image_id"]),
                        "tray_overlay_url": _overlay_url(sid, last["image_id"]),
                        "order": gateway().approved_order(sid)}
    elif row["status"] == "done":   # 육안 검수 후 수동 진행으로 끝난 처방 (검수 결과 없음, 6장)
        body["card"] = {"inspection": None, "order": gateway().approved_order(sid)}
    return jsonify(body)


@bp.get("/prescriptions/<rx_id>/images/<image_id>")
@api_login_required
def image(rx_id: str, image_id: str):
    row = rx.image(_uuid(rx_id), _uuid(image_id), g.user["pharmacy_id"])
    if row is None:
        abort(404)
    from flask import request

    path = storage.path_of(row["storage_key"])
    view = request.args.get("view")
    if view == "rect":    # F-WEB-06: OCR 좌표 기준인 펴진 이미지. 없으면(가짜 연결부 등) 원본
        rect = path.with_name(path.name + ".rect.jpg")
        if rect.is_file():
            path = rect
    elif view == "overlay":   # F-WEB-09·13: 트레이 검수 결과 그림. 원본으로 대신하지 않는다 (판정 색이 없는 사진을 결과로 오인)
        path = storage.overlay_path(path)
        if not path.is_file():
            abort(404)
    return send_file(path, max_age=3600)


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
    # 경고 확인 체크 여부 (Q11): 확인할 것이 있던 승인이면 체크해야만 여기까지 온다. manual(위험)은 사유로 대신한다
    warning_ack = None if action == "manual" else (workflow.needs_confirm(order) and body.get("confirmed") is True)

    try:   # 응답이 Agent가 받을 모양인지 미리 본다: 잘못된 응답이 백그라운드에서 처방을 실패로 만들지 않게
        gateway().check_reply(str(row["id"]), "rx_approval", {"action": action, **reply})
    except ValueError as e:
        return _error(400, str(e))
    if not rx.respond(row["id"], g.user["pharmacy_id"], g.user["id"], "rx_approval",
                      action, before=order, after=order, reason=reason, warning_ack=warning_ack):
        return _error(409, "이미 처리된 승인입니다. 화면을 새로고침해 주세요.")
    workflow.jobs.submit(workflow.run_resume, row["id"], "rx_approval", {"action": action, **reply})
    return jsonify(id=str(row["id"]), status="processing"), 202


@bp.post("/prescriptions/<rx_id>/cancel")
@api_login_required
def cancel(rx_id: str):
    """처방 취소 (F-5, REQ-WEB-07). 완료 전 처방을 '취소됨'으로 남긴다. 실패 카드의 [삭제]도 이 API를 쓴다.

    이후 승인·트레이 업로드는 상태 확인에서 409가 되고, 처리 중이던 결과는 조건부 UPDATE가 무시한다.
    """
    row = _load(rx_id)
    if not rx.set_cancelled(row["id"]):
        return _error(409, "이미 끝났거나 취소된 처방입니다. 화면을 새로고침해 주세요.")
    return jsonify(id=str(row["id"]), status="cancelled")


@bp.post("/prescriptions/<rx_id>/retry")
@api_login_required
def retry(rx_id: str):
    """실패한 처방 다시 시도 (Q6). 일시 오류(E_LLM·E_EXT_API·E_INTERRUPTED·E_CV_INFER)로 실패한 처방만. 같은 처방전 이미지로 새 처방을
    만들어 처음부터 다시 돌리고, 실패한 처방은 취소됨으로 남긴다. 새 처방의 ID·번호를 202로 돌려준다.
    """
    row = _load(rx_id)
    if row["status"] != "failed" or row["error_code"] not in workflow.RETRY_CODES:
        return _error(409, "다시 시도할 수 없는 처방입니다. 화면을 새로고침해 주세요.")
    new = workflow.retry(row, g.user["pharmacy_id"], g.user["id"])
    if new is None:
        return _error(409, "이미 다시 시도했거나 처방전 사진을 찾을 수 없습니다. 화면을 새로고침해 주세요.")
    return jsonify(id=str(new["id"]), display_no=new["display_no"], status=new["status"]), 202


@bp.post("/prescriptions/<rx_id>/retake")
@api_login_required
def retake(rx_id: str):
    """실패한 처방을 새로 찍은 처방전 사진으로 다시 접수 (6장 재촬영). 실패한 처방은 취소됨으로 남고 새 처방 번호가 생긴다.
    조제 기록 저장 실패(E_DB)는 여기서 받지 않는다 (검수까지 끝난 처방이라 [저장 다시 시도])."""
    row = _load(rx_id)
    if row["status"] != "failed" or row["error_code"] in workflow.SAVE_RETRY_CODES:
        return _error(409, "다시 올릴 수 없는 처방입니다. 화면을 새로고침해 주세요.")
    got = _read_image("처방전")
    if not isinstance(got[0], bytes):
        return got
    data, ext = got
    new = workflow.retake(row, g.user["pharmacy_id"], g.user["id"], data, ext)
    if new is None:
        return _error(409, "이미 다시 올렸거나 취소된 처방입니다. 화면을 새로고침해 주세요.")
    return jsonify(id=str(new["id"]), display_no=new["display_no"], status=new["status"]), 202


@bp.post("/prescriptions/<rx_id>/save-retry")
@api_login_required
def save_retry(rx_id: str):
    """조제 기록 저장 실패(E_DB) 처방의 저장만 다시 한다 (6장). 결과는 상세 조회로 본다 (완료 또는 다시 E_DB)."""
    row = _load(rx_id)
    if not workflow.retry_save(row["id"], g.user["pharmacy_id"]):
        return _error(409, "저장을 다시 시도할 수 없는 처방입니다. 화면을 새로고침해 주세요.")
    return jsonify(id=str(row["id"]), status="processing"), 202


MANUAL_REASON_MAX = 500


@bp.post("/prescriptions/<rx_id>/manual-inspection")
@api_login_required
def manual_inspection(rx_id: str):
    """육안 검수 후 수동 진행 (6장 E_CV_INFER). 트레이 검출 모델 오류로 트레이 대기에 멈춘 처방만, 사유 필수.
    승인 기록에 'inspection · manual'과 사유를, 조제 기록에 검수 결과 없이 남기고 완료한다."""
    row = _load(rx_id)
    body = request.get_json(silent=True) or {}
    reason = body.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return _error(400, "육안으로 확인한 내용(사유)을 적어 주세요.")
    reason = reason.strip()
    if len(reason) > MANUAL_REASON_MAX:
        return _error(400, f"사유는 {MANUAL_REASON_MAX}자까지 적을 수 있습니다.")
    if row["status"] != "waiting_tray" or not workflow.finish_manual_inspection(
            row["id"], g.user["pharmacy_id"], g.user["id"], reason):
        return _error(409, "검출 모델 오류로 멈춘 처방에서만 수동으로 진행할 수 있습니다. 화면을 새로고침해 주세요.")
    return jsonify(id=str(row["id"]), status="done")


@bp.post("/prescriptions/<rx_id>/inspection")
@api_login_required
def inspection_reply(rx_id: str):
    """HITL ② 검수 확인 (F-WEB-09, REQ-WEB-04). confirm이면 완료(done), redo(재조제)면 트레이 대기로 돌아가 다시 찍는다."""
    row = _load(rx_id)
    if row["status"] != "waiting_approval" or row["hitl_point"] != "inspection":
        return _error(409, "검수 확인을 기다리는 처방이 아닙니다. 화면을 새로고침해 주세요.")
    pause = gateway().pending(str(row["id"]))
    if pause is None or pause.hitl_point != "inspection":
        return _error(409, "확인할 검수 결과를 찾지 못했습니다. 화면을 새로고침해 주세요.")
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in ("confirm", "redo"):
        return _error(400, "확인 완료(confirm) 또는 재조제(redo)를 골라 주세요.")
    reply = {"action": action}
    try:
        gateway().check_reply(str(row["id"]), "inspection", reply)
    except ValueError as e:
        return _error(400, str(e))
    if not rx.respond(row["id"], g.user["pharmacy_id"], g.user["id"], "inspection", action,
                      before=pause.payload.get("inspection"), after=reply):
        return _error(409, "이미 처리된 검수 확인입니다. 화면을 새로고침해 주세요.")
    workflow.jobs.submit(workflow.run_resume, row["id"], "inspection", reply)
    return jsonify(id=str(row["id"]), status="processing"), 202
