"""AI 조사 (agent/investigate.py): 위험 신호 근거 조사. 규칙의 결과는 바뀌지 않는다. LLM은 가짜.

매칭 실패 줄의 AI 후보 조사는 11/18에 뺐다 (대상 8종 밖의 약은 승인할 수 없다, REQ-AG-03).
"""
import json
from types import SimpleNamespace

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.drug_match import match_order
from agent.investigate import investigate_risks, local_evidence
from agent.nodes import Deps, match_node, risk_node
from agent.rag import RagIndex, chunk_docs
from agent.schema import DispensingOrder, RiskFinding
from test_react import Scripted, answer, call, uses

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL, GEVORIN = "202106092", "197900277"
GEVORIN_SOFT, TYLENOL_ER = "201800001", "201800002"
NAMES = {TYLENOL: "타이레놀정500mg", GEVORIN: "게보린정"}
PROMPT = "지시"


def product(code, name):
    return SimpleNamespace(item_code=code, name=name, company="시험제약")


PRODUCTS = [product(GEVORIN, "게보린정"), product(GEVORIN_SOFT, "게보린소프트연질캡슐"),
            product(TYLENOL, "타이레놀정500밀리그램"), product(TYLENOL_ER, "타이레놀8시간이알서방정650밀리그램")]


def fake_api(term):
    """첫 글자가 같은 제품 (게 → 게보린 둘, 타 → 타이레놀 둘)."""
    return [p for p in PRODUCTS if term and p.name[0] == term[0]]


def order_with(*names):
    items = [{"item_code": None, "name": n, "count_per_dose": 1, "times": ["morning"], "days": 1, "source_line_ids": [i]}
             for i, n in enumerate(names)]
    return match_order(DispensingOrder.model_validate({"prescription_id": "P0001", "items": items}), DICT)


# ------------------------------------------------------------ 매칭 실패 줄: AI 조사 없음 (11/18)


def test_an_unmatched_line_keeps_the_rule_candidates_and_the_llm_is_not_called():
    raw = order_with("게부린").model_dump(mode="json")
    update = match_node({"order": raw}, deps(react=Scripted()))                    # 부르면 Scripted가 빈 목록에서 실패
    line = update["order"]["items"][0]
    assert line["item_code"] is None and [c["item_code"] for c in line["match_candidates"]] == [GEVORIN]
    assert "notes" not in update


# ------------------------------------------------------------ 위험 신호 → 근거 조사


DUP = RiskFinding(type="ingredient_dup", item_codes=[TYLENOL, GEVORIN], level="warning",
                  message="타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요.")
GOOD = "타이레놀과 게보린 모두 아세트아미노펜이 들어 있어 하루 총량이 늘어납니다. 함께 처방한 이유와 총량을 확인하세요."


def doc(code, name, section, text):
    return {"doc_id": f"{code}:t:{section}", "item_code": code, "item_name": name, "source": "시험용", "section": section,
            "url": "", "text": text, "updated": None}


INDEX = RagIndex(chunk_docs([
    doc(TYLENOL, "타이레놀정500mg", "상호작용", "(시험용) 다른 해열진통제와 함께 복용하면 아세트아미노펜 과량으로 간 손상이 생길 수 있습니다."),
    doc(GEVORIN, "게보린정", "주의사항", "(시험용) 아세트아미노펜이 들어 있는 다른 약과 함께 복용하지 마십시오."),
    doc("999999999", "다른약", "상호작용", "(시험용) 아세트아미노펜 함께 복용 주의."),
]))


def test_the_evidence_tool_searches_only_the_flagged_drugs_and_numbers_the_results():
    chat = Scripted(uses(call("search_evidence", {"query": "아세트아미노펜 함께 복용", "item_codes": [TYLENOL, "999999999"]})),
                    answer({"items": [{"index": 1, "text": GOOD, "evidence": ["E1", "E9"]}]}))
    (found, note) = investigate_risks([DUP], NAMES, local_evidence(INDEX), chat, PROMPT)
    results = json.loads(chat.seen[1][0][-1]["content"])["results"]
    assert [r["drug"] for r in results] == ["타이레놀정500mg"] and results[0]["id"] == "E1"   # 걸리지 않은 약(999999999)은 찾지 않는다
    assert found[0].explanation == GOOD                                                       # 근거는 설명 뒤가 아니라 evidence 칸에 (A.5)
    assert [e.title for e in found[0].evidence] == ["타이레놀정500mg · 상호작용"]                # 없는 번호(E9)는 버린다
    assert "아세트아미노펜 과량" in found[0].evidence[0].excerpt
    assert (found[0].level, found[0].message) == (DUP.level, DUP.message)
    assert note == "AI 위험 신호 조사: search_evidence(성공)"
    assert "[1] 단계: 경고" in chat.seen[0][0][1]["content"]


def test_an_explanation_without_evidence_is_kept_without_sources():
    chat = Scripted(answer({"items": [{"index": 1, "text": GOOD, "evidence": []}]}))
    assert investigate_risks([DUP], NAMES, local_evidence(INDEX), chat, PROMPT)[0][0].explanation == GOOD


def test_a_misleading_explanation_is_dropped_even_with_evidence():
    chat = Scripted(uses(call("search_evidence", {"query": "함께"})),
                    answer({"items": [{"index": 1, "text": "타이레놀과 게보린은 같이 먹어도 괜찮습니다. 걱정하지 마세요.", "evidence": ["E1"]}]}))
    assert investigate_risks([DUP], NAMES, local_evidence(INDEX), chat, PROMPT)[0][0].explanation is None


def test_without_an_index_or_a_finished_answer_the_one_shot_explanation_takes_over():
    assert investigate_risks([DUP], NAMES, None, Scripted(), PROMPT) is None
    assert investigate_risks([], NAMES, local_evidence(INDEX), Scripted(), PROMPT) is None
    assert investigate_risks([DUP], NAMES, local_evidence(INDEX), Scripted({"content": "", "tool_calls": []}), PROMPT) is None


def test_the_query_is_embedded_when_the_index_has_vectors():
    chunks = INDEX.chunks
    index = RagIndex(chunks, {c.chunk_id: [1.0, 0.0] for c in chunks})
    seen = []
    chat = Scripted(uses(call("search_evidence", {"query": "간 손상"})), answer({"items": []}))
    investigate_risks([DUP], NAMES, local_evidence(index, lambda texts: seen.append(texts) or [[1.0, 0.0]]), chat, PROMPT)
    assert seen == [["간 손상"]]


# ------------------------------------------------------------ 노드


def deps(**over):
    return Deps(dictionary=DICT, structure=lambda *a: pytest.fail("구조화"), **over)


def test_a_drug_outside_the_eight_cannot_be_approved():
    """의약품 API로 찾은 약(8종 밖)이나 약사가 넣은 8종 밖 코드: 확인 표시 → 조제 계획 없음 → 승인 · 사유 진행 불가 (REQ-AG-03)."""
    from agent.nodes import NOT_TARGET_REASON, plan_node, read_reply

    items = [{"item_code": None, "name": "게보린소프트연질캡슐", "count_per_dose": 1, "times": ["morning"], "days": 1,
              "source_line_ids": [0]}]
    raw = DispensingOrder.model_validate({"prescription_id": "P0001", "items": items}).model_dump(mode="json")
    matched = match_node({"order": raw}, deps(api=fake_api))["order"]
    line = matched["items"][0]
    assert line["item_code"] == GEVORIN_SOFT and line["match_method"] == "api"
    assert line["needs_pharmacist_check"] and NOT_TARGET_REASON in line["check_reasons"]

    checked = risk_node({"order": matched}, deps())["order"]
    update = plan_node({"order": checked}, deps())
    assert update["plan"] is None and "대상 8종이 아닌 약" in update["notice"] and update["error_code"] is None
    for action in ("approve", "manual"):
        with pytest.raises(ValueError, match="조제 계획을 만들지 못해"):
            read_reply({"action": action, "order": checked, "reason": "확인함"}, DispensingOrder.model_validate(checked),
                       plan_ready=False)

    again = match_node({"order": matched}, deps(api=fake_api))["order"]["items"][0]          # 다시 매칭해도 이유는 한 번만
    assert again["check_reasons"].count(NOT_TARGET_REASON) == 1


def test_the_match_node_does_not_investigate_when_everything_matched():
    raw = order_with("타이레놀정500mg").model_dump(mode="json")
    update = match_node({"order": raw}, deps(react=Scripted()))
    assert "notes" not in update and update["order"]["items"][0]["item_code"] == TYLENOL


def test_the_risk_node_uses_the_investigation_and_falls_back_to_the_one_shot_explanation():
    order = order_with("타이레놀정500mg", "게보린정").model_dump(mode="json")
    chat = Scripted(uses(call("search_evidence", {"query": "아세트아미노펜"})),
                    answer({"items": [{"index": 1, "text": GOOD, "evidence": ["E1"]}]}))
    update = risk_node({"order": order}, deps(rag=INDEX, react=chat, explain=lambda t: pytest.fail("조사가 끝났으면 부르지 않는다")))
    finding = update["order"]["risk_check"]["findings"][0]
    assert finding["explanation"] == GOOD and finding["evidence"] and set(finding["evidence"][0]) == {"title", "source", "excerpt"}
    assert finding["level"] == "warning" and update["notes"] == ["AI 위험 신호 조사: search_evidence(성공)"]

    broken = Scripted({"content": None, "tool_calls": []})
    update = risk_node({"order": order}, deps(rag=INDEX, react=broken, explain=lambda t: {"items": [{"index": 1, "text": GOOD}]}))
    assert update["order"]["risk_check"]["findings"][0]["explanation"] == GOOD   # 한 번 부르는 설명으로 대신


def test_the_test_graph_never_uses_the_react_llm_and_it_can_be_switched_off(monkeypatch):
    from agent import config
    from agent.graph import _FROM_SETTINGS, _make_deps
    from agent.llm import tool_chat

    assert _make_deps(DICT, lambda *a: None, None).react is None
    monkeypatch.setattr("agent.graph.default_api", lambda: None)
    monkeypatch.setattr("agent.graph.default_dur_api", lambda: None)
    monkeypatch.setattr("agent.graph.load_index", lambda folder, search=None: None)
    monkeypatch.setattr(config, "AGENT_REACT", False)
    assert _make_deps(DICT, lambda *a: None, _FROM_SETTINGS).react is None
    monkeypatch.setattr(config, "AGENT_REACT", True)
    assert _make_deps(DICT, lambda *a: None, _FROM_SETTINGS).react is tool_chat


def test_the_prompts_ask_for_json_and_forbid_made_up_codes():
    from agent.llm import load_prompt

    risk = load_prompt("react_risk")
    assert "JSON" in risk and "evidence" in risk                  # response_format json_object는 지시문에 JSON이 있어야 한다


# ------------------------------------------------------------ MCP 서버의 도구로 (11/16: 도구 목록은 MCP list_tools에서)


def test_the_risk_node_searches_evidence_on_the_mcp_server():
    pytest.importorskip("mcp")
    from agent.mcp_client import in_process

    order = order_with("타이레놀정500mg", "게보린정").model_dump(mode="json")
    chat = Scripted(uses(call("search_evidence", {"query": "아세트아미노펜 함께 복용"})),
                    answer({"items": [{"index": 1, "text": GOOD, "evidence": ["E1"]}]}))
    with in_process(DICT, rag=INDEX) as tools:
        update = risk_node({"order": order}, deps(tools=tools, react=chat))      # deps.rag 없이 서버의 색인으로
    results = json.loads(chat.seen[1][0][-1]["content"])["results"]
    assert results and {r["drug"] for r in results} <= {"타이레놀정500mg", "게보린정"} and results[0]["id"] == "E1"
    assert update["order"]["risk_check"]["findings"][0]["evidence"]                    # 서버에서 찾은 근거가 A.5 evidence에 담긴다


def test_an_old_mcp_server_without_the_tools_falls_back_to_direct_calls():
    from agent.nodes import _evidence_search

    class OldServer:
        def tool_names(self):
            return ["search_drug"]

    d = deps(tools=OldServer(), rag=INDEX)
    assert _evidence_search(d)("간", [TYLENOL], 2)                                       # 근거 색인을 직접
    assert _evidence_search(deps(tools=OldServer())) is None
