"""처방 처리 흐름: 업로드 → (백그라운드) Agent → HITL에서 멈춤 → 약사 응답 → 다시 Agent → 트레이 대기
→ 트레이 사진 → 검수 → 완료 또는 HITL ②(불일치·확인 필요).

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

# 일시 오류: 같은 처방전 이미지로 처음부터 다시 시도할 수 있다 (11/17 회의 Q6 [재시도]). 그 밖의 실패는 다시 찍어 올려야 한다
# [다시 시도](같은 처방전 사진으로 새 처방): 일시 오류. E_CV_INFER(OCR 모델 추론 오류)는 11/23 추가 (6장: 재시도)
RETRY_CODES = {"E_LLM", "E_EXT_API", "E_INTERRUPTED", "E_CV_INFER"}
# [저장 다시 시도](같은 처방): 조제 기록 저장 실패. 검수까지 끝났으므로 처음부터 다시 하지 않는다 (6장: E_DB 재시도)
SAVE_RETRY_CODES = {"E_DB"}
# 트레이 대기에서 [육안 검수 후 수동 진행]을 열어 주는 오류: 검출 모델 오류 (6장). 트레이를 못 찾음은 다시 찍으면 된다
MANUAL_TRAY_CODES = {"E_CV_INFER"}

# 화면 안내 문구 (기능분석서 6장과 맞춘다)
ERROR_MESSAGES = {
    "E_OCR_EMPTY": "처방전에서 글자를 찾지 못했습니다. 처방전 전체가 보이게 다시 찍어 올려 주세요.",
    "E_TRAY_NOT_FOUND": "트레이를 찾지 못했습니다. 트레이 전체가 보이고 아침 줄이 위로 오게 다시 찍어 올려 주세요.",
    "E_CV_INFER": "이미지 분석 중 오류가 났습니다. 다시 올려 주세요.",
    "E_LLM": "AI 응답을 받지 못했습니다. 잠시 뒤 다시 올려 주세요.",
    "E_EXT_API": "의약품 정보 조회에 실패했습니다. 잠시 뒤 다시 올려 주세요.",
    "E_DB": "저장 중 오류가 났습니다. 잠시 뒤 다시 시도해 주세요.",
    "E_INTERRUPTED": "처리 중에 서버가 다시 시작되어 중단되었습니다. 처방전을 다시 올려 주세요.",
}
UNKNOWN_ERROR = "처리 중 알 수 없는 오류가 났습니다. 다시 올려 주세요."
# 트레이 검수 단계에서는 같은 코드라도 할 수 있는 일이 달라 문구를 따로 둔다
TRAY_ERROR_MESSAGES = {
    "E_CV_INFER": "트레이 사진을 분석하는 중 오류가 났습니다. 다시 올려 주세요. 계속 실패하면 트레이를 육안으로 확인하고 "
                  "사유를 적어 수동으로 진행할 수 있습니다.",
}

# 트레이 검수가 이 코드로 실패하면 처방을 실패로 두지 않고 트레이 대기로 돌려 다시 찍게 한다 (Agent도 planned에 머문다)
TRAY_RETRY_CODES = {"E_TRAY_NOT_FOUND", "E_CV_INFER"}


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


def retry(row: dict, pharmacy_id: int, user_id: int) -> dict | None:
    """일시 오류로 실패한 처방을 같은 처방전 이미지로 처음부터 다시 돌린다 (Q6 [재시도]). → 새 처방 행, 처리할 수 없으면 None.

    같은 처방 ID를 되살리지 않고 새 처방을 만든다: Agent 실행 기록(thread)이 처방 ID에 묶여 있어 실패한 기록 위에
    다시 시작하지 않는다. 실패한 처방은 [삭제]와 같이 '취소됨'으로 남긴다. 취소가 먼저라 두 번 눌러도 새 처방은 하나만 생긴다.
    """
    img = rx.first_image(row["id"])
    if img is None:
        return None
    path = storage.path_of(img["storage_key"])
    if not path.is_file() or not rx.set_cancelled(row["id"]):
        return None
    return receive(pharmacy_id, user_id, path.read_bytes(), path.suffix)


def retake(row: dict, pharmacy_id: int, user_id: int, data: bytes, ext: str) -> dict | None:
    """실패한 처방을 새로 찍은 처방전 사진으로 다시 접수한다 (6장 재촬영: E_OCR_EMPTY 등). → 새 처방 행, 못 하면 None.

    [다시 시도]와 같이 새 처방을 만들고 실패한 처방은 취소됨으로 남긴다. 취소가 먼저라 두 번 눌러도 새 처방은 하나다.
    조제 기록 저장 실패(E_DB)는 검수까지 끝난 처방이라 여기로 오지 않는다 ([저장 다시 시도]).
    """
    if row["error_code"] in SAVE_RETRY_CODES or not rx.set_cancelled(row["id"]):
        return None
    return receive(pharmacy_id, user_id, data, ext)


def retry_save(rx_id: uuid.UUID, pharmacy_id: int) -> bool:
    """조제 기록 저장 실패(E_DB)한 처방의 저장만 다시 한다 (6장). 실패 → 처리 중 → (백그라운드) 완료 또는 다시 E_DB."""
    if not rx.reopen_for_save(rx_id, pharmacy_id):
        return False
    jobs.submit(_finish, rx_id)
    return True


def finish_manual_inspection(rx_id: uuid.UUID, pharmacy_id: int, user_id: int, reason: str) -> bool:
    """육안 검수 후 수동 진행 (6장 E_CV_INFER). 트레이 대기에서 검출 모델 오류로 멈춘 처방만. 승인된 조제지시서 · 조제 계획으로
    조제 기록을 남기고 완료한다 (검수 결과는 없음, 승인 기록에 'inspection · manual'과 사유)."""
    sid = str(rx_id)
    err = gateway().tray_error(sid) or {}
    if err.get("code") not in MANUAL_TRAY_CODES:
        return False
    order, plan = gateway().approved_order(sid), gateway().approved_plan(sid)
    if not (order and plan):
        return False
    return rx.finish_manual_inspection(rx_id, pharmacy_id, user_id, reason, order, plan)


def receive_tray(rx_id: uuid.UUID, pharmacy_id: int, data: bytes, ext: str) -> uuid.UUID | None:
    """트레이 사진을 저장하고 승인된(트레이 대기) 처방에 붙인 뒤 검수를 백그라운드로 넘긴다. → 이미지 ID

    처방이 트레이 대기가 아니면(이미 다른 사진이 처리 중 등) 파일을 지우고 None.
    """
    key = storage.save(str(rx_id), data, ext)
    try:
        image_id = rx.attach_tray(rx_id, pharmacy_id, key)
    except Exception:
        storage.path_of(key).unlink(missing_ok=True)
        raise
    if image_id is None:
        storage.path_of(key).unlink(missing_ok=True)
        return None
    jobs.submit(run_tray, rx_id, image_id, key)
    return image_id


def _apply(rx_id: uuid.UUID, outcome: Pause | Wait | None) -> None:
    """Agent 실행 결과를 처방 상태에 옮긴다. None이면 아직 처리 중이라 그대로 둔다."""
    if isinstance(outcome, Pause):
        rx.set_waiting(rx_id, outcome.hitl_point)
    elif isinstance(outcome, Wait):
        if outcome.status == "waiting_tray":
            rx.set_waiting_tray(rx_id)
        elif outcome.status == "done":
            _finish(rx_id)


def _finish(rx_id: uuid.UUID) -> None:
    """트레이 검수가 끝난 처방을 완료로 옮기고 조제 기록을 저장한다 (F-WEB-12 ②).

    최종 조제지시서·조제 계획·검수 결과 중 하나라도 없거나 저장이 실패하면 완료로 두지 않고 E_DB 실패로 둔다
    (한 트랜잭션이라 조제 기록이 반쯤 남지 않는다).
    """
    sid = str(rx_id)
    try:
        order, plan, a3 = gateway().approved_order(sid), gateway().approved_plan(sid), gateway().inspection(sid)
        if not (order and plan and a3):
            raise ValueError("최종 조제지시서·조제 계획·검수 결과 중 없는 것이 있습니다")
        rx.set_done(rx_id, order, plan, a3)
    except Exception:
        current_app.logger.exception("처방 %s 조제 기록 저장 실패", rx_id)
        rx.set_failed(rx_id, "E_DB")


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


def run_tray(rx_id: uuid.UUID, image_id: uuid.UUID, key: str) -> None:
    """트레이 검수 (F-WEB-09). 검수 결과(A.3)는 inspections에 남기고, 전 칸 일치면 완료, 아니면 HITL ②에서 멈춘다.

    트레이를 못 찾는 등 다시 찍으면 되는 실패는 트레이 대기로 돌린다 (이유는 gateway().tray_error로 화면에 보인다).
    """
    try:
        outcome = gateway().tray(str(rx_id), str(image_id), storage.path_of(key))
    except Exception as e:
        if getattr(e, "code", None) in TRAY_RETRY_CODES:
            current_app.logger.warning("처방 %s 트레이 검수 실패, 다시 찍기: %s", rx_id, e)
            rx.set_waiting_tray(rx_id)
            return
        _fail(rx_id, e)
        return
    a3 = gateway().inspection(str(rx_id))
    if a3 and str(a3.get("image_id")) == str(image_id):
        rx.add_inspection(rx_id, image_id, a3)
    _apply(rx_id, outcome)


def run_resume(rx_id: uuid.UUID, hitl_point: str, reply: dict[str, Any]) -> None:
    try:
        pause = gateway().resume(str(rx_id), hitl_point, reply)
    except Exception as e:
        _fail(rx_id, e)
        return
    _apply(rx_id, pause)


# ------------------------------------------------------------ HITL ① 승인 규칙
def approval_blockers(order: dict[str, Any], plan: dict[str, Any] | None = ...) -> list[str]:
    """승인할 수 없는 이유. 비어 있으면 승인할 수 있다.

    plan을 넘겼는데 None이면(조제 계획을 못 만든 경우: 3일 초과·약을 정하지 못한 줄) 승인·직접 처리는
    거절하고 수정(edit)만 받는다 (11/17). plan을 안 넘기면 계획 검사는 건너뛴다.
    """
    missing = [it["name"] for it in order.get("items", []) if not it.get("item_code")]
    if missing:
        return [f"품목을 찾지 못한 약품이 있습니다: {', '.join(missing)}. 항목을 수정한 뒤 승인하세요."]
    if plan is None:
        return ["조제 계획을 만들지 못해 승인할 수 없습니다. 항목을 수정한 뒤 다시 확인하세요."]
    return []


TIME_VALUES = ("morning", "noon", "evening", "bedtime")
MAX_DAYS_INPUT = 30   # 3일을 넘는 값도 받는다: 조제 계획을 못 만든 이유(E_PLAN_RANGE)를 화면에 보이기 위해


def _int_in(value: Any, low: int, high: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label}은(는) {low}~{high} 사이 정수여야 합니다.")
    return value


def build_edited_order(order: dict[str, Any], edits: Any) -> dict[str, Any]:
    """약사가 고친 항목 목록으로 새 조제지시서를 만든다 (HITL ① edit). 잘못된 입력이면 ValueError(화면용 문장).

    edits: [{index: 원래 항목 번호 | null(새 항목), name, count_per_dose, times, days}] — 목록에 없는 원래 항목은 삭제.
    원래 줄 정보(source_line_ids)는 그대로 둔다. 이름을 바꾼 항목은 item_code를 비워 Agent가 다시 찾게 한다.
    """
    if not isinstance(edits, list) or not edits:
        raise ValueError("약품이 하나 이상 있어야 합니다. 모두 지우려면 새 처방전을 올려 주세요.")
    if len(edits) > 20:
        raise ValueError("약품은 20개까지 입력할 수 있습니다.")
    original = order.get("items", [])
    items, used = [], set()
    for n, e in enumerate(edits, 1):
        if not isinstance(e, dict):
            raise ValueError(f"{n}번째 항목 형식이 올바르지 않습니다.")
        idx = e.get("index")
        base = original[idx] if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(original) else None
        if base is not None:
            if idx in used:
                raise ValueError(f"{n}번째 항목이 중복되었습니다.")
            used.add(idx)
        name = str(e.get("name") or "").strip()
        if not name or len(name) > 100:
            raise ValueError(f"{n}번째 약품 이름을 1~100자로 입력해 주세요.")
        times = e.get("times")
        if (not isinstance(times, list) or not times or len(set(times)) != len(times)
                or any(t not in TIME_VALUES for t in times)):
            raise ValueError(f"{name}: 복용 시간을 하나 이상 고르세요.")
        item = dict(base) if base else {"item_code": None, "source_line_ids": []}
        if base is None or name != base.get("name"):
            # 이름을 바꾸면 약을 새로 찾는다. Agent 조제지시서 규칙(match_method가 있으면 item_code도 있어야 함)에 맞게
            # 매칭 결과(방법·후보)와 인쇄 코드(printed_code: 남기면 Agent가 예전 약을 코드로 다시 찾음)도 함께 비운다
            item["item_code"] = None
            for key in ("needs_pharmacist_check", "check_reasons", "match_method", "match_candidates", "printed_code"):
                item.pop(key, None)
        item.update(name=name,
                    count_per_dose=_int_in(e.get("count_per_dose"), 1, 10, f"{name}의 1회 알 수"),
                    days=_int_in(e.get("days"), 1, MAX_DAYS_INPUT, f"{name}의 복용 일수"),
                    times=[t for t in TIME_VALUES if t in times])
        items.append(item)
    return {**order, "items": items}


def is_danger(order: dict[str, Any]) -> bool:
    """위험(danger) 단계면 그대로 승인할 수 없다. 수정하거나 사유를 적어(manual) 진행한다 (기능분석서 F-WEB-07)."""
    return (order.get("risk_check") or {}).get("level") == "danger"


def needs_confirm(order: dict[str, Any]) -> bool:
    """경고 이하 위험이나 확인 필요 표시가 있으면 약사가 확인란을 체크해야 승인된다 (명세서 2.8절)."""
    risk = (order.get("risk_check") or {}).get("level")
    flagged = any(it.get("needs_pharmacist_check") for it in order.get("items", []))
    return risk in ("warning", "danger") or flagged
