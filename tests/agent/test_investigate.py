"""AI 조사 (agent/investigate.py): 매칭 실패 줄의 후보 조사 · 위험 신호 근거 조사. 규칙의 결과는 바뀌지 않는다. LLM은 가짜."""
import json
from types import SimpleNamespace

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.drug_match import match_order
from agent.investigate import AI_CHECK, NO_SUGGESTION, drug_tools, investigate_risks, local_evidence, suggest_candidates
from agent.nodes import Deps, match_node, recheck_node, risk_node
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


# ------------------------------------------------------------ 1) 매칭 실패 → 후보 조사


def test_the_tools_are_search_drug_and_check_item_code():
    assert [t.name for t in drug_tools(DICT, fake_api)] == ["search_drug", "check_item_code"]
    assert [t.name for t in drug_tools(DICT, None)] == ["search_drug"]                    # API가 없으면 코드 확인 도구도 없다
    found = drug_tools(DICT, fake_api)[0].run({"name": "게부린"})
    assert found["dictionary"][0]["item_code"] == GEVORIN and [p["item_code"] for p in found["api"]] == [GEVORIN, GEVORIN_SOFT]
    checked = drug_tools(DICT, fake_api)[1].run({"item_code": GEVORIN_SOFT, "name": "게보린"})
    assert checked["verdict"] == "일치"


def test_only_codes_the_tools_returned_are_added_and_the_drug_stays_undecided():
    order = order_with("게부린")
    assert order.items[0].item_code is None and [c.item_code for c in order.items[0].match_candidates] == [GEVORIN]
    chat = Scripted(uses(call("search_drug", {"name": "게보린"})),
                    answer({"candidates": [{"item_code": GEVORIN, "reason": "이미 있는 후보"},           # 이미 있다 → 더하지 않는다
                                           {"item_code": GEVORIN_SOFT, "reason": "게보린 계열 제품"},
                                           {"item_code": "123456789", "reason": "지어낸 코드"}]}))       # 도구가 준 적 없다 → 버린다
    new, notes = suggest_candidates(order, drug_tools(DICT, fake_api), chat, PROMPT)
    line = new.items[0]
    assert line.item_code is None and line.match_method == "none"                         # 약은 정하지 않는다
    assert [(c.item_code, c.name, c.reason) for c in line.match_candidates] == [
        (GEVORIN, "게보린정", "이름이 비슷함"), (GEVORIN_SOFT, "게보린소프트연질캡슐", "AI 조사: 게보린 계열 제품")]
    assert line.needs_pharmacist_check and AI_CHECK in line.check_reasons
    assert notes == ["AI 후보 조사 (게부린): 후보 1개 | search_drug(성공)"]
    assert "처방전에 적힌 약 이름: 게부린" in chat.seen[0][0][1]["content"]
    assert "이미 있는 후보: 게보린정(197900277)" in chat.seen[0][0][1]["content"]
    assert order.items[0].match_candidates[0].item_code == GEVORIN and len(order.items[0].match_candidates) == 1   # 원래 것은 그대로


def test_a_product_of_another_strength_is_not_a_candidate():
    order = order_with("타이레놀정650mg")
    chat = Scripted(uses(call("search_drug", {"name": "타이레놀"})),
                    answer({"candidates": [{"item_code": TYLENOL_ER, "reason": "650mg 제품"},
                                           {"item_code": TYLENOL, "reason": "500mg"}]}))
    line = suggest_candidates(order, drug_tools(DICT, fake_api), chat, PROMPT)[0].items[0]
    assert [c.item_code for c in line.match_candidates] == [TYLENOL, TYLENOL_ER][:1] + [TYLENOL_ER]   # 규칙 후보 + 650mg만
    assert line.match_candidates[-1].reason == "AI 조사: 650mg 제품"


def test_matched_lines_and_full_candidate_lists_are_not_investigated():
    order = order_with("타이레놀정500mg", "게보린정")
    new, notes = suggest_candidates(order, drug_tools(DICT, fake_api), Scripted(), PROMPT)                # 부르면 Scripted가 빈 목록에서 실패
    assert new is order and notes == []


def test_at_most_three_lines_are_investigated():
    order = order_with("게부린", "게부린정", "게부린정2", "게부린정3")
    chat = Scripted(*[answer({"candidates": []}) for _ in range(3)])
    _, notes = suggest_candidates(order, drug_tools(DICT, fake_api), chat, PROMPT)
    assert len(notes) == 3 and chat.replies == []


def test_a_failed_investigation_leaves_the_line_and_says_so():
    order = order_with("게부린")
    new, notes = suggest_candidates(order, drug_tools(DICT, fake_api), Scripted({"content": "모르겠어요", "tool_calls": []}), PROMPT)
    assert new.items[0] == order.items[0]
    assert notes[0].startswith(NO_SUGGESTION) and "최종 답이 JSON 객체가 아닙니다" in notes[0]


def test_an_api_failure_inside_a_tool_is_reported_to_the_llm():
    from agent.errors import DrugApiError

    def broken(term):
        raise DrugApiError("HTTP 500")

    chat = Scripted(uses(call("search_drug", {"name": "게보린"})), answer({"candidates": []}))
    _, notes = suggest_candidates(order_with("게부린"), drug_tools(DICT, broken), chat, PROMPT)
    assert "도구가 실패했습니다 (DrugApiError)" in chat.seen[1][0][-1]["content"] and "search_drug(실패)" in notes[0]


# ------------------------------------------------------------ 2) 위험 신호 → 근거 조사


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
    assert found[0].explanation == f"{GOOD} (근거: 타이레놀정500mg · 상호작용)"                   # 없는 번호(E9)는 버린다
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


def test_the_match_node_adds_ai_candidates_and_notes():
    raw = order_with("게부린").model_dump(mode="json")
    for item in raw["items"]:
        item.update(item_code=None, match_method=None, match_candidates=[], needs_pharmacist_check=False, check_reasons=[])
    chat = Scripted(uses(call("search_drug", {"name": "게보린"})),
                    answer({"candidates": [{"item_code": GEVORIN_SOFT, "reason": "게보린 계열"}]}))
    update = match_node({"order": raw}, deps(api=fake_api, react=chat))
    line = update["order"]["items"][0]
    assert line["item_code"] is None and GEVORIN_SOFT in [c["item_code"] for c in line["match_candidates"]]
    assert update["status"] == "matched" and update["notes"][0].startswith("AI 후보 조사 (게부린)")


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
    assert finding["explanation"].startswith(GOOD) and "(근거: " in finding["explanation"]
    assert finding["level"] == "warning" and update["notes"] == ["AI 위험 신호 조사: search_evidence(성공)"]

    broken = Scripted({"content": None, "tool_calls": []})
    update = risk_node({"order": order}, deps(rag=INDEX, react=broken, explain=lambda t: {"items": [{"index": 1, "text": GOOD}]}))
    assert update["order"]["risk_check"]["findings"][0]["explanation"] == GOOD   # 한 번 부르는 설명으로 대신


def test_the_recheck_node_investigates_new_findings_too():
    from agent.risk import check_order
    from fake_dur import KETOROLAC, KETOROLAC_NAME, TRISPEN, FakeDur

    order = check_order(order_with("트리스펜연질캡슐"), DICT, FakeDur()).model_dump(mode="json")
    meds = [{"item_code": KETOROLAC, "name": KETOROLAC_NAME}]
    state = {"conversation_id": "P0001", "prescription_id": "P0001", "status": "planned", "order": order,
             "meds_input": meds, "current_meds": meds}
    index = RagIndex(chunk_docs([doc(TRISPEN, "트리스펜연질캡슐", "상호작용", "(시험용) 케토롤락과 함께 투여하지 마십시오.")]))
    text = "트리스펜과 가상케토롤락정은 병용금기 조합입니다. 처방 의도를 확인하세요."
    chat = Scripted(uses(call("search_evidence", {"query": "케토롤락"})),
                    answer({"items": [{"index": 1, "text": text, "evidence": ["E1"]}]}))
    update = recheck_node(state, deps(dur=FakeDur(), rag=index, react=chat))
    finding = update["consult_result"]["new_findings"][0]
    assert finding["level"] == "danger" and finding["explanation"] == f"{text} (근거: 트리스펜연질캡슐 · 상호작용)"


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

    match, risk = load_prompt("react_match"), load_prompt("react_risk")
    assert "JSON" in match and "JSON" in risk                     # response_format json_object는 지시문에 JSON이 있어야 한다
    assert "지어내지 마세요" in match and "evidence" in risk


# ------------------------------------------------------------ MCP 서버의 도구로 (11/16: 도구 목록은 MCP list_tools에서)


def test_the_match_node_runs_the_ai_tools_on_the_mcp_server():
    pytest.importorskip("mcp")
    from agent.mcp_client import in_process

    raw = order_with("게부린").model_dump(mode="json")
    chat = Scripted(uses(call("search_drug", {"name": "게보린"})),
                    answer({"candidates": [{"item_code": GEVORIN_SOFT, "reason": "게보린 계열"}]}))
    with in_process(DICT, fake_api) as tools:
        update = match_node({"order": raw}, deps(api=None, tools=tools, react=chat))
    assert chat.seen[0][1] == ["search_drug", "check_item_code"]              # 서버가 알려 준 도구 목록
    found = json.loads(chat.seen[1][0][-1]["content"])                        # 서버가 실행한 결과
    assert [p["item_code"] for p in found["api"]] == [GEVORIN, GEVORIN_SOFT]
    assert GEVORIN_SOFT in [c["item_code"] for c in update["order"]["items"][0]["match_candidates"]]


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
    assert "(근거: " in update["order"]["risk_check"]["findings"][0]["explanation"]


def test_an_old_mcp_server_without_the_tools_falls_back_to_direct_calls():
    from agent.nodes import _evidence_search, _react_drug_tools

    class OldServer:
        def react_tools(self, names):
            from agent.mcp_client import McpToolError
            raise McpToolError("MCP 서버에 도구가 없습니다")

        def tool_names(self):
            return ["search_drug"]

    d = deps(tools=OldServer(), rag=INDEX)
    assert [t.name for t in _react_drug_tools(d)] == ["search_drug"]                     # 직접 호출 (API 없음)
    assert _evidence_search(d)("간", [TYLENOL], 2)                                       # 근거 색인을 직접
    assert _evidence_search(deps(tools=OldServer())) is None
