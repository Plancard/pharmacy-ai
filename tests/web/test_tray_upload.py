"""트레이 사진 업로드 → 검수 → 완료 · HITL ② · 다시 찍기 (11/18, F-WEB-02 ④ · F-WEB-09, REQ-WEB-04).

웹 DB는 test_prescriptions.py의 가짜(FakeRx), Agent는 가짜 연결부(FakeGateway) + cv/stub.py 검수(CV_STUB_SCENARIO).
진짜 Agent(LangGraph)와의 연결은 아래 '진짜 Agent' 절에서 가짜 LangGraph · 메모리 체크포인터로 본다.
실행: python -m pytest tests/web -q
"""
from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path

import pytest

from test_prescriptions import JPG, PNG, approve, client, fake_rx, make_client, upload  # noqa: F401  (client · fake_rx는 fixture)
from web.agent_gateway import FakeGateway, Pause, Wait


def tray(client, rx_id, data=JPG, name="tray.jpg"):
    return client.post(f"/api/prescriptions/{rx_id}/tray", data={"image": (io.BytesIO(data), name)},
                       headers={"X-CSRF-Token": client.token}, content_type="multipart/form-data")


def detail(client, rx_id):
    return client.get(f"/api/prescriptions/{rx_id}").get_json()


def waiting_tray(client):
    """처방전 업로드 → 승인까지 해서 트레이 대기인 처방 ID."""
    rx_id = upload(client).get_json()["id"]
    assert approve(client, rx_id, confirmed=True).status_code == 202
    assert detail(client, rx_id)["status"] == "waiting_tray"
    return rx_id


# ------------------------------------------------------------ 가짜 Agent
def test_all_ok_goes_straight_to_done_and_keeps_the_inspection(client, fake_rx, tmp_path):
    rx_id = waiting_tray(client)
    r = tray(client, rx_id)
    assert r.status_code == 202 and r.get_json()["status"] == "processing"
    image_id = r.get_json()["image_id"]

    d = detail(client, rx_id)
    assert d["status"] == "done" and d["hitl_point"] is None            # 전 칸 일치: HITL ②를 건너뜀 (E-1)
    a3 = d["card"]["inspection"]
    assert a3["image_id"] == image_id and a3["summary"]["hitl_required"] is False and len(a3["cells"]) == 12
    assert d["card"]["tray_image_url"].endswith(f"/images/{image_id}")
    assert client.get(d["card"]["tray_image_url"]).data == JPG           # 트레이 사진도 같은 약국만 받는다
    assert [e["event"] for e in d["timeline"]] == ["received", "response", "tray_uploaded", "inspected"]
    assert d["card"]["order"]["prescription_id"] == rx_id                # 완료 카드도 약 이름을 보일 수 있다
    assert len(fake_rx.inspection_rows) == 1
    assert len(list((tmp_path / "uploads").rglob("*.jpg"))) == 2         # 처방전 + 트레이


@pytest.mark.parametrize("scenario, status", [("mismatch", "mismatch"), ("review", "need_review")])
def test_a_cell_that_is_not_ok_stops_at_hitl_2(client, fake_rx, monkeypatch, scenario, status):
    rx_id = waiting_tray(client)
    monkeypatch.setenv("CV_STUB_SCENARIO", scenario)
    image_id = tray(client, rx_id).get_json()["image_id"]

    d = detail(client, rx_id)
    assert (d["status"], d["hitl_point"]) == ("waiting_approval", "inspection")
    card = d["card"]
    assert card["choices"] == ["confirm", "redo"] and card["inspection"]["summary"]["hitl_required"] is True
    bad = [c for c in card["inspection"]["cells"] if c["status"] != "ok"]
    assert [c["status"] for c in bad] == [status]
    assert len(card["cells"]) == 1 and "예상" in card["cells"][0] and "타이레놀정500mg" in card["cells"][0]
    assert card["tray_image_url"].endswith(f"/images/{image_id}")
    assert fake_rx.inspection_rows[0]["hitl_required"] is True


def test_tray_not_found_goes_back_to_waiting_tray_and_a_retake_works(client, fake_rx, monkeypatch):
    rx_id = waiting_tray(client)
    monkeypatch.setenv("CV_STUB_SCENARIO", "no_tray")
    assert tray(client, rx_id).status_code == 202

    d = detail(client, rx_id)
    assert d["status"] == "waiting_tray" and d["error_code"] is None     # 실패가 아니라 다시 찍기
    assert d["card"]["retake"]["code"] == "E_TRAY_NOT_FOUND" and "다시 찍어" in d["card"]["retake"]["message"]
    assert d["card"]["order"]["prescription_id"] == rx_id and fake_rx.inspection_rows == []
    assert [e["event"] for e in d["timeline"]][-1] == "tray_uploaded"     # 올렸지만 검수 결과가 없는 사진

    monkeypatch.setenv("CV_STUB_SCENARIO", "ok")
    assert tray(client, rx_id).status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "done" and len(fake_rx.inspection_rows) == 1


def test_waiting_tray_card_has_no_retake_after_approval(client):
    rx_id = waiting_tray(client)
    assert detail(client, rx_id)["card"]["retake"] is None


@pytest.mark.parametrize("state", ["waiting_approval", "processing", "done", "failed", "cancelled"])
def test_tray_is_refused_unless_waiting_tray(client, fake_rx, tmp_path, state):
    rx_id = upload(client).get_json()["id"]
    fake_rx.rows[uuid.UUID(rx_id)]["status"] = state
    r = tray(client, rx_id)
    assert r.status_code == 409 and "승인이 끝난" in r.get_json()["error"]
    assert len(list((tmp_path / "uploads").rglob("*.jpg"))) == 1         # 트레이 사진은 저장하지 않았다
    assert not any(i["kind"] == "tray" for i in fake_rx.images.values())


def test_a_second_tray_while_waiting_hitl_2_is_refused(client, monkeypatch):
    rx_id = waiting_tray(client)
    monkeypatch.setenv("CV_STUB_SCENARIO", "mismatch")
    assert tray(client, rx_id).status_code == 202
    assert tray(client, rx_id).status_code == 409


def test_lost_race_removes_the_saved_file(client, fake_rx, monkeypatch, tmp_path):
    """상태 검사 뒤 다른 요청이 먼저 붙였다면(attach_tray None) 409이고 파일을 남기지 않는다."""
    rx_id = waiting_tray(client)
    monkeypatch.setattr("web.rx.attach_tray", lambda *a: None)
    r = tray(client, rx_id)
    assert r.status_code == 409
    assert len(list((tmp_path / "uploads").rglob("*.jpg"))) == 1


@pytest.mark.parametrize("data,name", [(b"GIF89a....", "a.gif"), (b"not an image", "a.jpg")])
def test_tray_rejects_non_images(client, fake_rx, data, name):
    rx_id = waiting_tray(client)
    r = tray(client, rx_id, data, name)
    assert r.status_code == 400 and "JPG 또는 PNG" in r.get_json()["error"]
    assert detail(client, rx_id)["status"] == "waiting_tray"


def test_tray_needs_file_and_png_works(client):
    rx_id = waiting_tray(client)
    r = client.post(f"/api/prescriptions/{rx_id}/tray", data={}, headers={"X-CSRF-Token": client.token})
    assert r.status_code == 400 and "트레이 이미지를 선택" in r.get_json()["error"]
    assert tray(client, rx_id, PNG, "t.png").status_code == 202


def test_tray_of_other_pharmacy_is_404(client, fake_rx):
    rx_id = waiting_tray(client)
    fake_rx.rows[uuid.UUID(rx_id)]["pharmacy_id"] = 99
    assert tray(client, rx_id).status_code == 404


def inspect_reply(client, rx_id, **body):
    return client.post(f"/api/prescriptions/{rx_id}/inspection", json=body, headers={"X-CSRF-Token": client.token})


def at_hitl_2(client, monkeypatch):
    rx_id = waiting_tray(client)
    monkeypatch.setenv("CV_STUB_SCENARIO", "mismatch")
    tray(client, rx_id)
    assert detail(client, rx_id)["hitl_point"] == "inspection"
    return rx_id


def test_confirm_finishes_and_is_recorded(client, fake_rx, monkeypatch):
    rx_id = at_hitl_2(client, monkeypatch)
    r = inspect_reply(client, rx_id, action="confirm")
    assert r.status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "done" and d["card"]["inspection"]["summary"]["hitl_required"] is True
    last = fake_rx.approvals[-1]
    assert (last["hitl_point"], last["action"], last["after"]) == ("inspection", "confirm", {"action": "confirm"})
    assert last["before"]["summary"]["mismatch"] == 1 and last["warning_ack"] is None
    assert [e["action"] for e in d["timeline"] if e["event"] == "response"] == ["approve", "confirm"]


def test_redo_goes_back_to_waiting_tray_and_a_new_photo_is_checked(client, fake_rx, monkeypatch):
    rx_id = at_hitl_2(client, monkeypatch)
    assert inspect_reply(client, rx_id, action="redo").status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "waiting_tray" and d["card"]["retake"] is None
    monkeypatch.setenv("CV_STUB_SCENARIO", "ok")
    assert tray(client, rx_id).status_code == 202
    d = detail(client, rx_id)
    assert d["status"] == "done" and len(fake_rx.inspection_rows) == 2
    assert [e["event"] for e in d["timeline"]].count("tray_uploaded") == 2


@pytest.mark.parametrize("body", [{}, {"action": "approve"}, {"action": "fly"}])
def test_inspection_reply_needs_confirm_or_redo(client, fake_rx, monkeypatch, body):
    rx_id = at_hitl_2(client, monkeypatch)
    r = inspect_reply(client, rx_id, **body)
    assert r.status_code == 400 and len(fake_rx.approvals) == 1           # 승인 기록만 있다
    assert detail(client, rx_id)["hitl_point"] == "inspection"


def test_inspection_reply_only_while_waiting_hitl_2(client, fake_rx, monkeypatch):
    rx_id = waiting_tray(client)
    assert inspect_reply(client, rx_id, action="confirm").status_code == 409
    rx_id = at_hitl_2(client, monkeypatch)
    assert inspect_reply(client, rx_id, action="confirm").status_code == 202
    assert inspect_reply(client, rx_id, action="confirm").status_code == 409   # 두 번 누름


def test_approve_records_the_warning_check(client, fake_rx):
    """Q11: 경고·확인 필요가 있는 승인은 체크 여부를 승인 기록에 남긴다 (warning 시나리오: 성분 중복 경고)."""
    waiting_tray(client)
    assert fake_rx.approvals[0]["action"] == "approve" and fake_rx.approvals[0]["warning_ack"] is True


def test_approve_without_warnings_records_false(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="none")
    rx_id = upload(c).get_json()["id"]
    # none 시나리오도 샘플의 프론크논에 '확인 필요' 표시가 있어 체크가 필요하다 → 표시를 지운 조제지시서로 본다
    gw = c.application.extensions["agent"]
    data = gw._load(rx_id)
    for it in data["pause"]["payload"]["order"]["items"]:
        it.pop("needs_pharmacist_check", None)
    gw._save(rx_id, data)
    assert approve(c, rx_id).status_code == 202
    assert fake_rx.approvals[0]["warning_ack"] is False


def test_manual_records_no_warning_check(monkeypatch, tmp_path, fake_rx):
    c = make_client(monkeypatch, tmp_path, scenario="danger")
    rx_id = upload(c).get_json()["id"]
    assert approve(c, rx_id, action="manual", reason="의사 확인").status_code == 202
    assert fake_rx.approvals[0]["warning_ack"] is None


def test_fake_gateway_hitl_2_replies(tmp_path, monkeypatch):
    """HITL ② 응답(라우트는 다음 작업): confirm → 완료, redo → 트레이 대기. 다른 action은 check_reply가 거절한다."""
    gw = FakeGateway(tmp_path)
    pause = gw.start("r1", tmp_path / "x.jpg")
    assert gw.resume("r1", "rx_approval", {"action": "approve", "order": pause.payload["order"]}) == Wait("waiting_tray")
    monkeypatch.setenv("CV_STUB_SCENARIO", "mismatch")
    got = gw.tray("r1", "img1", tmp_path / "t.jpg")
    assert isinstance(got, Pause) and got.hitl_point == "inspection" and gw.pending("r1") == got
    with pytest.raises(ValueError):
        gw.check_reply("r1", "inspection", {"action": "fly"})
    gw.check_reply("r1", "inspection", {"action": "redo"})
    assert gw.resume("r1", "inspection", {"action": "redo"}) == Wait("waiting_tray")
    monkeypatch.setenv("CV_STUB_SCENARIO", "mismatch")
    gw.tray("r1", "img2", tmp_path / "t.jpg")
    assert gw.resume("r1", "inspection", {"action": "confirm"}) == Wait("done")
    assert gw.inspection("r1")["image_id"] == "img2"


def test_fake_gateway_refuses_tray_before_approval(tmp_path):
    from web.agent_gateway import AgentError

    gw = FakeGateway(tmp_path)
    gw.start("r1", tmp_path / "x.jpg")
    with pytest.raises(AgentError) as e:
        gw.tray("r1", "img1", tmp_path / "t.jpg")
    assert e.value.code == "E_CV_INFER"


# ------------------------------------------------------------ 진짜 Agent (가짜 LangGraph · 진짜 build_graph)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    import fake_langgraph

    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    monkeypatch.setenv("CV_STUB_SAMPLE", "rx8_01")      # 대상 8종만이라 바로 승인할 수 있다
    return request.param


def agent_gateway(scenario):
    from agent.api import AgentApi
    from agent.checkpoint import memory_checkpointer
    from agent.cv_adapter import stub_inspect
    from agent.drug_dictionary import load_dictionary
    from agent.graph import build_graph
    from agent.pipeline import structure_prescription
    from web.langgraph_gateway import LangGraphGateway, stub_ocr

    def rule_only(pid, ocr):
        return structure_prescription(pid, ocr, llm_fn=lambda *a: pytest.fail("LLM을 부르면 안 됩니다"))

    dictionary = load_dictionary()
    graph = build_graph(dictionary=dictionary, structure=rule_only, api=None, checkpointer=memory_checkpointer(),
                        inspect=stub_inspect(scenario))
    return LangGraphGateway(AgentApi(graph, dictionary), stub_ocr)


IMAGE = Path("var/uploads/upload.jpg")


def approved(gw, rx_id="P0001"):
    pause = gw.start(rx_id, IMAGE)
    assert gw.resume(rx_id, "rx_approval", {"action": "approve", "order": pause.payload["order"]}) == Wait("waiting_tray")
    return rx_id


def test_agent_all_ok_is_done(backend):
    gw = agent_gateway("ok")
    rx_id = approved(gw)
    assert gw.tray(rx_id, "img1", IMAGE) == Wait("done")
    assert gw.inspection(rx_id)["image_id"] == "img1" and gw.tray_error(rx_id) is None


def test_agent_mismatch_stops_at_inspection_and_replies_work(backend):
    gw = agent_gateway("mismatch")
    rx_id = approved(gw)
    pause = gw.tray(rx_id, "img1", IMAGE)
    assert isinstance(pause, Pause) and pause.hitl_point == "inspection"
    assert {"inspection", "plan", "names", "cells", "notice", "choices"} <= set(pause.payload)   # FakeGateway와 같은 칸
    assert gw.resume(rx_id, "inspection", {"action": "redo"}) == Wait("waiting_tray")
    assert isinstance(gw.tray(rx_id, "img2", IMAGE), Pause)
    assert gw.resume(rx_id, "inspection", {"action": "confirm"}) == Wait("done")


def test_agent_tray_not_found_keeps_waiting_tray(backend):
    from web.agent_gateway import AgentError

    gw = agent_gateway("no_tray")
    rx_id = approved(gw)
    with pytest.raises(AgentError) as e:
        gw.tray(rx_id, "img1", IMAGE)
    assert e.value.code == "E_TRAY_NOT_FOUND"
    assert gw.tray_error(rx_id)["code"] == "E_TRAY_NOT_FOUND"


def test_web_tray_flow_on_the_agent(backend, monkeypatch, tmp_path, fake_rx):
    """웹 전체: 승인 → 트레이 못 찾음(트레이 대기 + 다시 찍기 안내) → 다시 찍음 → 불일치(HITL ②)."""
    c = make_client(monkeypatch, tmp_path)
    c.application.extensions["agent"] = agent_gateway(None)    # 검수 시나리오는 CV_STUB_SCENARIO를 매번 읽는다
    rx_id = upload(c).get_json()["id"]
    assert approve(c, rx_id, confirmed=True).status_code == 202

    monkeypatch.setenv("CV_STUB_SCENARIO", "no_tray")
    assert tray(c, rx_id).status_code == 202
    d = detail(c, rx_id)
    assert d["status"] == "waiting_tray" and d["card"]["retake"]["code"] == "E_TRAY_NOT_FOUND"

    monkeypatch.setenv("CV_STUB_SCENARIO", "mismatch")
    image_id = tray(c, rx_id).get_json()["image_id"]
    d = detail(c, rx_id)
    assert (d["status"], d["hitl_point"]) == ("waiting_approval", "inspection")
    assert d["card"]["inspection"]["image_id"] == image_id and d["card"]["inspection"]["prescription_id"] == rx_id
    assert len(fake_rx.inspection_rows) == 1
