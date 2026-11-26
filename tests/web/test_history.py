"""조제 이력 목록·상세 (11/23, 기능분석서 F-WEB-13, REQ-WEB-10).

DB 질의(web.rx의 이력 함수)는 가짜로 바꾸고 화면이 무엇을 어떻게 보여 주는지만 본다. SQL은 PG16에서 따로 확인한다.
실행: python -m pytest -q tests/web/test_history.py
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

import web.rx as rx
from test_app import app, client, fake, login  # noqa: F401  (fixture)

TYL, GEB = "202106092", "197900277"
RX_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
RX_IMG = uuid.UUID("22222222-2222-4222-8222-222222222222")
TRAY_IMG = uuid.UUID("33333333-3333-4333-8333-333333333333")
T0 = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)      # KST 10:00


def row(n=1, **kw):
    return {"id": RX_ID, "display_no": f"P{n:04d}", "created_at": T0, "risk_level": "warning", "completed_at": T0,
            "username": "ph1", "hitl_required": False, "attempts": 1, **kw}


def order():
    return {"prescription_id": str(RX_ID), "warnings": ["타이레놀정500mg: 복용 시점이 처방전에 없어 하루 3회 기준으로 배정했습니다."],
            "items": [{"item_code": TYL, "name": "타이레놀정500mg", "printed_name": "타이레놀정 50Orng", "count_per_dose": 1,
                       "times": ["morning", "evening"],
                       "days": 3, "needs_pharmacist_check": True, "check_reasons": ["1회량 칸 OCR 신뢰도 낮음"]},
                      {"item_code": GEB, "name": "게보린정", "count_per_dose": 1, "times": ["noon"], "days": 3}],
            "risk_check": {"scope": "rx_only", "level": "warning", "findings": [
                {"type": "ingredient_dup", "item_codes": [TYL, GEB], "level": "warning", "message": "아세트아미노펜 성분 중복",
                 "evidence": [{"title": "타이레놀정 복약 정보", "source": "e약은요", "excerpt": "다른 해열진통제와 함께 먹지 마세요"}]},
                {"type": "dose", "item_codes": [GEB], "level": "caution", "message": "용량 주의", "evidence": []}]}}


def inspection():
    cells = [{"day": d, "time": t, "expected": {}, "actual": {}, "status": "ok", "reasons": []}
             for d in (1, 2, 3) for t in ("morning", "noon", "evening", "bedtime")]
    cells[0] = {"day": 1, "time": "morning", "expected": {TYL: 1}, "actual": {TYL: 1}, "status": "ok", "reasons": []}
    cells[5] = {"day": 2, "time": "noon", "expected": {GEB: 1}, "actual": {}, "status": "mismatch", "reasons": []}
    cells[10] = {"day": 3, "time": "evening", "expected": {TYL: 1}, "actual": {TYL: 1}, "status": "need_review",
                 "reasons": ["boundary"]}
    return {"prescription_id": str(RX_ID), "image_id": str(TRAY_IMG), "cells": cells,
            "summary": {"ok": 10, "mismatch": 1, "need_review": 1, "hitl_required": True}}


@pytest.fixture
def db_history(monkeypatch):
    calls = {}

    def dispense_list(pharmacy_id, **kw):
        calls["list"] = (pharmacy_id, kw)
        return calls.get("rows", [row()]), calls.get("total", 1)

    def dispense_record(rx_id, pharmacy_id):
        calls["record"] = (rx_id, pharmacy_id)
        if rx_id != RX_ID or pharmacy_id != 1:
            return None
        return {**row(), "rx_json": order(), "plan_json": {}, "inspection_json": inspection()}

    monkeypatch.setattr(rx, "dispense_list", dispense_list)
    monkeypatch.setattr(rx, "dispense_record", dispense_record)
    monkeypatch.setattr(rx, "approval_log", lambda rx_id: [
        {"hitl_point": "rx_approval", "action": "approve", "reason": None, "warning_ack": True, "created_at": T0, "username": "ph1"},
        {"hitl_point": "inspection", "action": "redo", "reason": None, "warning_ack": None, "created_at": T0, "username": "ph1"},
        {"hitl_point": "inspection", "action": "confirm", "reason": None, "warning_ack": None, "created_at": T0, "username": "ph1"}])
    monkeypatch.setattr(rx, "inspections", lambda rx_id: [{}, {}])
    monkeypatch.setattr(rx, "first_image", lambda rx_id, kind="prescription": {"id": RX_IMG, "kind": kind, "storage_key": "x"})
    monkeypatch.setattr(rx, "tray_image_key", lambda image_id: f"uploads/{RX_ID}/tray.jpg" if image_id == TRAY_IMG else None)
    return calls


def page(client, path):
    r = client.get(path)
    return r.status_code, r.get_data(as_text=True)


# ------------------------------------------------------------ 목록 (F-WEB-13 ①)
def test_list_shows_records_with_a_link_to_the_detail(client, db_history):
    login(client)
    code, html = page(client, "/history")
    assert code == 200
    assert f'href="/history/{RX_ID}"' in html and "P0001" in html
    assert "2026-11-20 10:00" in html                                   # 화면 시간대(KST)
    assert "경고" in html and "전 칸 일치" in html and "ph1" in html
    pharmacy_id, kw = db_history["list"]
    assert pharmacy_id == 1 and kw == {"since": None, "until": None, "risk": None, "review": None, "page": 1}


def test_filters_become_kst_day_bounds_and_are_kept_in_the_form(client, db_history):
    login(client)
    code, html = page(client, "/history?from=2026-11-20&to=2026-11-21&risk=warning&review=checked")
    _, kw = db_history["list"]
    assert kw["since"].isoformat() == "2026-11-20T00:00:00+09:00"
    assert kw["until"].isoformat() == "2026-11-22T00:00:00+09:00"     # 끝일 다음날 0시 전까지
    assert (kw["risk"], kw["review"]) == ("warning", "checked")
    assert 'value="2026-11-20"' in html and '<option value="warning" selected>' in html
    assert '<option value="checked" selected>' in html and "조건 지우기" in html


def test_bad_filter_values_are_ignored_and_reversed_dates_are_swapped(client, db_history):
    login(client)
    page(client, "/history?from=nope&risk=fatal&review=x&page=-3")
    assert db_history["list"][1] == {"since": None, "until": None, "risk": None, "review": None, "page": 1}
    page(client, "/history?from=2026-11-22&to=2026-11-20")
    kw = db_history["list"][1]
    assert kw["since"].day == 20 and kw["until"].day == 23


def test_pager_keeps_the_filters(client, db_history):
    db_history.update(rows=[row(n) for n in range(1, 21)], total=45)
    login(client)
    _, html = page(client, "/history?risk=warning&page=2")
    assert db_history["list"][1]["page"] == 2
    assert "45건" in html and "2 / 3" in html
    assert "/history?page=1&amp;risk=warning" in html and "/history?page=3&amp;risk=warning" in html


def test_empty_results_say_why(client, db_history):
    db_history.update(rows=[], total=0)
    login(client)
    assert "조건에 맞는 조제 기록이 없습니다" in page(client, "/history?risk=danger")[1]
    assert "저장된 조제 기록이 없습니다" in page(client, "/history")[1]


def test_checked_record_shows_photo_attempts(client, db_history):
    db_history.update(rows=[row(hitl_required=True, attempts=2)])
    login(client)
    html = page(client, "/history")[1]
    assert "약사 확인 후 완료" in html and "촬영 2회" in html


# ------------------------------------------------------------ 상세 (F-WEB-13 ②)
def test_detail_shows_every_part_of_the_record(client, db_history):
    login(client)
    code, html = page(client, f"/history/{RX_ID}")
    assert code == 200 and db_history["record"] == (RX_ID, 1)
    # 처방전 원본
    assert f'src="/api/prescriptions/{RX_ID}/images/{RX_IMG}"' in html
    # 최종 조제지시서 (복용 시점 문구 · 확인 이유 · 경고)
    assert "타이레놀정500mg" in html and "아침 · 저녁" in html and "1회량 칸 OCR 신뢰도 낮음" in html
    assert "복용 시점이 처방전에 없어" in html
    assert "처방전 표기: 타이레놀정 50Orng" in html                    # 표준 이름으로 바꾼 줄은 원래 글자도 (Agent printed_name)
    assert html.count("처방전 표기") == 1                               # 이름이 같았던 줄(게보린)에는 없다
    # 위험 단계와 대조 근거 (근거가 없으면 '근거 없음')
    assert "아세트아미노펜 성분 중복" in html and "타이레놀정 복약 정보" in html and "e약은요" in html
    assert "타이레놀정500mg, 게보린정" in html and "근거 없음" in html
    # 검수 결과: 그림이 없으면 원본 트레이 사진, 칸별 판정, 문제 칸, 촬영 횟수
    assert f'src="/api/prescriptions/{RX_ID}/images/{TRAY_IMG}"' in html and "?view=overlay" not in html
    assert "불일치" in html and "확인 필요" in html and "빈 칸" in html
    assert "2일 점심 · 불일치" in html and "예상 게보린정 1개 / 실제 없음" in html
    assert "사유 칸 경계에 걸침" in html
    assert "트레이 촬영 2회 (마지막 검수 결과)" in html
    # 승인 이력 (시간순)
    assert html.index("① 조제지시서 승인") < html.index("재조제") < html.index("확인 완료")
    assert "확인함" in html


def test_detail_shows_the_overlay_when_it_exists(client, db_history, app):
    login(client)
    tray = app.config["VAR_DIR"] / "uploads" / str(RX_ID) / "tray.jpg"
    tray.parent.mkdir(parents=True)
    tray.write_bytes(b"x")
    tray.with_name("tray.jpg.overlay.jpg").write_bytes(b"x")
    html = page(client, f"/history/{RX_ID}")[1]
    assert f'src="/api/prescriptions/{RX_ID}/images/{TRAY_IMG}?view=overlay"' in html
    assert "원본 사진" in html


@pytest.mark.parametrize("path", ["/history/not-a-uuid", f"/history/{uuid.uuid4()}"])
def test_unknown_or_other_pharmacy_record_is_404(client, db_history, path):
    login(client)
    assert client.get(path).status_code == 404


def test_history_needs_login(client, db_history):
    assert client.get(f"/history/{RX_ID}").status_code == 302
