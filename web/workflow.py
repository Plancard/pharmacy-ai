"""처방 처리 흐름: 업로드 → (백그라운드) Agent → HITL에서 멈춤 → 약사 응답 → 다시 Agent → 트레이 대기.

라우트(web/api)는 요청을 검사하고 이 함수들을 부른다. DB 상태 바꾸기는 web/rx.py, Agent 호출은
web/agent_gateway.py가 맡는다.
"""
from __future__ import annotations

import uuid
from typing import Any

import click
from flask import Flask, current_app
from flask.cli import with_appcontext

from web import jobs, rx, storage
from web.agent_gateway import Pause, Wait, gateway

ERROR_CODES = {"E_OCR_EMPTY", "E_TRAY_NOT_FOUND", "E_CV_INFER", "E_PLAN_RANGE",
               "E_BARCODE_FAIL", "E_LLM", "E_EXT_API", "E_DB", "E_INTERRUPTED"}

# 화면 안내 문구 (기능분석서 6장과 맞춘다)
ERROR_MESSAGES = {
    "E_OCR_EMPTY": "처방전에서 글자를 찾지 못했습니다. 처방전 전체가 보이게 다시 찍어 올려 주세요.",
    "E_CV_INFER": "이미지 분석 중 오류가 났습니다. 다시 올려 주세요.",
    "E_LLM": "AI 응답을 받지 못했습니다. 잠시 뒤 다시 올려 주세요.",
    "E_EXT_API": "의약품 정보 조회에 실패했습니다. 잠시 뒤 다시 올려 주세요.",
    "E_DB": "저장 중 오류가 났습니다. 잠시 뒤 다시 시도해 주세요.",
    "E_INTERRUPTED": "처리 중에 서버가 다시 시작되어 중단되었습니다. 처방전을 다시 올려 주세요.",
}
UNKNOWN_ERROR = "처리 중 알 수 없는 오류가 났습니다. 다시 올려 주세요."


STALE_MINUTES = 10   # 처방 1장 처리 목표는 1분(REQ-NF-01). 이보다 오래 '처리 중'이면 끊긴 것으로 본다


def init_app(app: Flask) -> None:
    app.cli.add_command(fail_stale_command)


@click.command("fail-stale")
@with_appcontext
@click.option("--minutes", default=STALE_MINUTES, show_default=True, help="이 시간(분)보다 오래 처리 중인 처방만")
def fail_stale_command(minutes: int) -> None:
    """서버 재시작으로 끊겨 '처리 중'에 남은 처방을 실패(E_INTERRUPTED)로 정리한다."""
    done = rx.fail_stale(minutes)
    click.echo(f"끊긴 처방 {len(done)}건 정리" + (f": {', '.join(done)}" if done else ""))


def receive(pharmacy_id: int, user_id: int, data: bytes, ext: str) -> dict:
    """처방전 이미지를 저장하고 처방을 만든 뒤 처리를 백그라운드로 넘긴다."""
    rx_id = uuid.uuid4()
    key = storage.save(str(rx_id), data, ext)
    try:
        row = rx.create(pharmacy_id, user_id, rx_id, key)
    except Exception:
        storage.path_of(key).unlink(missing_ok=True)   # DB에 못 넣었으면 파일도 남기지 않는다
        raise
    jobs.submit(run_start, rx_id, key)
    return row


def _apply(rx_id: uuid.UUID, outcome: Pause | Wait | None) -> None:
    """Agent 실행 결과를 처방 상태에 옮긴다. None이면 아직 처리 중이라 그대로 둔다."""
    if isinstance(outcome, Pause):
        rx.set_waiting(rx_id, outcome.hitl_point)
    elif isinstance(outcome, Wait):
        if outcome.status == "waiting_tray":
            rx.set_waiting_tray(rx_id)


def _fail(rx_id: uuid.UUID, e: Exception) -> None:
    code = getattr(e, "code", None)
    if code not in ERROR_CODES:
        current_app.logger.exception("처방 %s 처리 실패 (원인 미상)", rx_id)
        code = None
    else:
        current_app.logger.warning("처방 %s 처리 실패: %s", rx_id, e)
    rx.set_failed(rx_id, code)


def run_start(rx_id: uuid.UUID, key: str) -> None:
    try:
        pause = gateway().start(str(rx_id), storage.path_of(key))
    except Exception as e:
        _fail(rx_id, e)
        return
    _apply(rx_id, pause)


def run_resume(rx_id: uuid.UUID, hitl_point: str, reply: dict[str, Any]) -> None:
    try:
        pause = gateway().resume(str(rx_id), hitl_point, reply)
    except Exception as e:
        _fail(rx_id, e)
        return
    _apply(rx_id, pause)


# ------------------------------------------------------------ HITL ① 승인 규칙
def approval_blockers(order: dict[str, Any]) -> list[str]:
    """승인할 수 없는 이유. 비어 있으면 승인할 수 있다."""
    missing = [it["name"] for it in order.get("items", []) if not it.get("item_code")]
    if missing:
        return [f"품목을 찾지 못한 약품이 있습니다: {', '.join(missing)}. 항목을 수정한 뒤 승인하세요."]
    return []


def needs_confirm(order: dict[str, Any]) -> bool:
    """위험(danger) 단계 항목이나 확인 필요 표시가 있으면 약사가 확인란을 체크해야 승인된다 (명세서 2.8절)."""
    risk = (order.get("risk_check") or {}).get("level")
    flagged = any(it.get("needs_pharmacist_check") for it in order.get("items", []))
    return risk == "danger" or flagged
