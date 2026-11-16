"""진짜 OpenAI로 도는 시험 (켰을 때만). 평소 `pytest`에서는 건너뛴다 (비용 · 네트워크).

    $env:AGENT_LIVE_OPENAI = "1"; python -m pytest tests/agent/test_live_openai.py -v     # PowerShell
    AGENT_LIVE_OPENAI=1 python -m pytest tests/agent/test_live_openai.py -v               # bash

.env의 OPENAI_API_KEY · OPENAI_MODEL · LLM_REASONING_EFFORT를 그대로 쓴다. 한 번 돌리면 호출 10번 안팎 (구조화 1 · 임베딩 2 · 위험 신호 설명 1 · AI 조사(ReAct) 2번 × 많아야 4).
LLM 답은 매번 조금씩 달라서 글자를 그대로 견주지 않고, 지켜야 할 성질(형식 · 근거 번호 · 시간 · 정답 값)만 본다.
근거 문서는 시험용으로 지어낸 글이다 (실제 허가정보가 아니다).
"""
import json
import time

import pytest

from agent import config
from agent.config import RESOURCES_DIR
from agent.rag import RagIndex, chunk_docs

pytestmark = pytest.mark.live_openai

SAMPLES = RESOURCES_DIR / "samples"
TYLENOL = "202106092"


def doc(section, text):
    return {"doc_id": f"{TYLENOL}:t:{section}", "item_code": TYLENOL, "item_name": "타이레놀정500mg",
            "source": "시험용 문서", "section": section, "url": "", "text": text, "updated": None}


DOCS = [
    doc("상호작용", "(시험용) 이 약을 복용하는 동안 술을 마시면 간 손상이 생길 수 있습니다. 다른 해열진통제와 함께 먹지 마십시오."),
    doc("보관법", "(시험용) 습기와 빛을 피해 실온에 보관하십시오."),
    doc("부작용", "(시험용) 드물게 발진, 가려움이 나타날 수 있습니다."),
]


def timed(fn, *args):
    started = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - started


# ------------------------------------------------------------ 처방 구조화 (LLM 경로)


def test_the_llm_reads_a_prescription_that_the_rules_cannot():
    from agent.drug_names import match_key
    from agent.llm import extract_llm
    from agent.pipeline import structure_prescription

    ocr = json.loads((SAMPLES / "a1_ocr" / "s05_no_code.json").read_text(encoding="utf-8"))     # 코드가 인쇄되지 않은 처방전
    want = json.loads((SAMPLES / "a5_order" / "s05_no_code.json").read_text(encoding="utf-8"))
    order, seconds = timed(lambda: structure_prescription("P_LIVE", ocr, llm_fn=extract_llm))
    assert order.source == "llm"
    got = [(match_key(d.name)[:4], d.count_per_dose, d.days, sorted(d.times)) for d in order.items]
    expected = [(match_key(d["name"])[:4], d["count_per_dose"], d["days"], sorted(d["times"])) for d in want["items"]]
    assert got == expected, f"LLM이 읽은 값이 정답과 다릅니다: {got}"
    assert seconds < 2 * config.LLM_TIMEOUT_SECONDS
    print(f"\n구조화 {seconds:.1f}초 (모델 {config.OPENAI_MODEL}, 추론 강도 {config.LLM_REASONING_EFFORT or '기본'})")


# ------------------------------------------------------------ 임베딩


def test_embeddings_find_the_section_by_meaning():
    from agent.embed import embed_texts

    chunks = chunk_docs(DOCS)
    vectors, seconds = timed(embed_texts, [f"{c.item_name} {c.section}: {c.text}" for c in chunks])
    assert len(vectors) == len(chunks) and len(vectors[0]) >= 256
    index = RagIndex(chunks, {c.chunk_id: v for c, v in zip(chunks, vectors)})
    query = "약 먹고 맥주 한잔 해도 괜찮을까요?"                            # '술'이라는 낱말 없이 뜻으로
    found = index.search(query, focus=[TYLENOL], query_vector=embed_texts([query])[0], k=1)
    print(f"\n임베딩 {seconds:.1f}초, 찾은 항목: {found[0].chunk.section}")
    assert found[0].chunk.section == "상호작용"


# ------------------------------------------------------------ 위험 신호 설명 (18단계)


def test_the_risk_explanation_keeps_the_level_and_names_the_drugs():
    from agent.explain import explain_findings
    from agent.llm import explain_llm
    from agent.schema import RiskFinding

    names = {"202106092": "타이레놀정500mg", "197900277": "게보린정"}
    finding = RiskFinding(type="ingredient_dup", item_codes=["202106092", "197900277"], level="warning",
                          message="타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요.")
    (out, note), seconds = timed(explain_findings, [finding], names, explain_llm)
    print(f"\n설명 ({seconds:.1f}초): {out[0].explanation}")
    assert note is None and out[0].explanation, "설명이 걸러졌거나 비었습니다 (위 출력 참고)"
    assert out[0].level == "warning" and out[0].message == finding.message
    assert seconds < config.LLM_TIMEOUT_SECONDS


# ------------------------------------------------------------ AI 조사 (ReAct + Function Calling, 11/16)


def test_react_finds_a_candidate_for_a_misread_drug_name_with_the_tools():
    from agent.drug_dictionary import load_dictionary
    from agent.drug_match import match_order
    from agent.investigate import drug_tools, suggest_candidates
    from agent.llm import load_prompt, tool_chat
    from agent.schema import DispensingOrder

    dictionary = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
    item = {"item_code": None, "name": "뮤코다인캡술", "count_per_dose": 1, "times": ["morning"], "days": 1}   # OCR이 '슐'을 '술'로
    order = match_order(DispensingOrder.model_validate({"prescription_id": "P_LIVE", "items": [item]}), dictionary)
    order = order.model_copy(update={"items": [order.items[0].model_copy(update={"match_candidates": []})]})  # 규칙 후보를 지우고 AI만
    (new, notes), seconds = timed(lambda: suggest_candidates(order, drug_tools(dictionary, None), tool_chat, load_prompt("react_match")))
    print(f"\nAI 후보 조사 ({seconds:.1f}초): {notes} | {[(c.item_code, c.reason) for c in new.items[0].match_candidates]}")
    line = new.items[0]
    assert line.item_code is None                                       # 약은 정하지 않는다
    assert "200308774" in [c.item_code for c in line.match_candidates]   # 뮤코다인캡슐200mg
    assert "search_drug(성공)" in notes[0]
    assert seconds < 4 * config.LLM_TIMEOUT_SECONDS


def test_react_investigates_a_risk_with_the_evidence_tool():
    from agent.investigate import investigate_risks, local_evidence
    from agent.llm import load_prompt, tool_chat
    from agent.schema import RiskFinding

    names = {TYLENOL: "타이레놀정500mg", "197900277": "게보린정"}
    docs = [*DOCS, {**doc("주의사항", "(시험용) 아세트아미노펜이 들어 있는 다른 약과 함께 복용하지 마십시오."),
                    "doc_id": "197900277:t:주의사항", "item_code": "197900277", "item_name": "게보린정"}]
    finding = RiskFinding(type="ingredient_dup", item_codes=[TYLENOL, "197900277"], level="warning",
                          message="타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요.")
    got, seconds = timed(lambda: investigate_risks([finding], names, local_evidence(RagIndex(chunk_docs(docs))), tool_chat, load_prompt("react_risk")))
    assert got is not None, "조사를 끝내지 못했습니다"
    (out, note) = got
    print(f"\n조사 ({seconds:.1f}초): {note}\n설명: {out[0].explanation}")
    assert out[0].level == "warning" and out[0].message == finding.message
    assert out[0].explanation, "설명이 걸러졌거나 비었습니다 (위 출력 참고)"
    assert "search_evidence" in note
    assert seconds < 4 * config.LLM_TIMEOUT_SECONDS
