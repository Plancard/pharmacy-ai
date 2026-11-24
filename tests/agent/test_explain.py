"""위험 신호 AI 설명 (18단계, agent/explain.py) — 단계는 규칙 그대로, 설명만 덧붙는다. LLM은 가짜로."""
import json

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import ConfigError, StructureError
from agent.explain import (FAILED_NOTE, acceptable, apply_explanations, attach_evidence, explain_findings, explain_input,
                           finding_terms, focused_excerpt, passages_for, repeats)
from agent.nodes import Deps, risk_node
from agent.rag import RagIndex, chunk_docs
from agent.schema import RiskFinding
from fake_dur import KETOROLAC, KETOROLAC_NAME, TRISPEN

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL, GEVORIN = "202106092", "197900277"
NAMES = {TYLENOL: "타이레놀정500mg", GEVORIN: "게보린정", TRISPEN: "트리스펜연질캡슐", KETOROLAC: KETOROLAC_NAME}
DUP = RiskFinding(type="ingredient_dup", item_codes=[TYLENOL, GEVORIN], level="warning",
                  message="타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요.")
TABOO = RiskFinding(type="combination_taboo", item_codes=[TRISPEN, KETOROLAC], level="danger", message="병용금기입니다.")
GOOD_DUP = "타이레놀과 게보린 모두 아세트아미노펜이 들어 있어 하루 총량이 늘어납니다. 함께 처방한 이유와 총량을 확인하세요."
GOOD_TABOO = "트리스펜과 가상케토롤락정은 함께 쓰면 위장관 출혈 위험이 커지는 병용금기 조합입니다. 처방 의도를 확인하세요."


def test_the_input_carries_level_type_drugs_and_the_rule_sentence():
    text = explain_input([DUP, TABOO], NAMES)
    assert "[1] 단계: 경고 | 유형: 성분 중복 | 약: 타이레놀정500mg, 게보린정 | 규칙 문장: 타이레놀정500mg" in text
    assert "[2] 단계: 위험 | 유형: 병용금기 | 약: 트리스펜연질캡슐, 가상케토롤락정" in text and "근거:" not in text


def test_explanations_are_added_without_touching_level_type_or_message():
    got = {"items": [{"index": 2, "text": GOOD_TABOO}, {"index": 1, "text": GOOD_DUP}]}
    out = apply_explanations([DUP, TABOO], got, NAMES)
    assert [f.explanation for f in out] == [GOOD_DUP, GOOD_TABOO]
    assert [(f.type, f.level, f.item_codes, f.message) for f in out] == [(f.type, f.level, f.item_codes, f.message) for f in (DUP, TABOO)]


@pytest.mark.parametrize("text", [
    "짧음",                                                                                  # 너무 짧다
    "가" * 400,                                                                               # 너무 길다
    "두 약 모두 아세트아미노펜이 들어 있습니다. 하루 총량을 확인하세요.",                          # 약 이름이 없다
    "타이레놀과 게보린은 함께 먹어도 안전합니다. 걱정하지 않아도 됩니다.",                         # 위험을 낮춘다
    "타이레놀과 게보린은 주의 단계입니다. 하루 총량을 확인하세요.",                                # 다른 단계를 말한다 (경고인데 주의)
])
def test_explanations_that_could_mislead_are_dropped(text):
    assert not acceptable(text, DUP, NAMES)
    assert apply_explanations([DUP], {"items": [{"index": 1, "text": text}]}, NAMES)[0].explanation is None


def test_the_right_level_may_be_named():
    assert acceptable("타이레놀과 게보린은 경고 단계의 성분 중복입니다. 하루 총량을 확인하세요.", DUP, NAMES)


@pytest.mark.parametrize("got", [{}, {"items": [{"index": 9, "text": GOOD_DUP}]}, {"items": [{"text": GOOD_DUP}]}, None])
def test_missing_or_wrong_numbers_leave_no_explanation(got):
    assert apply_explanations([DUP], got, NAMES)[0].explanation is None


def test_without_an_llm_or_findings_nothing_changes():
    assert explain_findings([DUP], NAMES, None) == ([DUP], None)
    assert explain_findings([], NAMES, lambda t: pytest.fail("항목 없이 부르지 않는다")) == ([], None)


def test_a_failing_llm_keeps_the_findings_and_says_so():
    def broken(text):
        raise StructureError("LLM 호출에 실패했습니다 (APITimeoutError).")

    out, note = explain_findings([DUP], NAMES, broken)
    assert out == [DUP] and note.startswith(FAILED_NOTE)


def test_evidence_of_the_involved_drugs_is_added_to_the_input():
    docs = [{"doc_id": f"{TYLENOL}:t:상호작용", "item_code": TYLENOL, "item_name": "타이레놀정500mg", "source": "시험용",
             "section": "상호작용", "url": "", "text": "(시험용) 다른 해열진통제와 함께 복용하지 마십시오."},
            {"doc_id": "999999999:t:x", "item_code": "999999999", "item_name": "다른약", "source": "시험용",
             "section": "상호작용", "url": "", "text": "(시험용) 함께 복용 주의."}]
    passages = passages_for(RagIndex(chunk_docs(docs)), [DUP], NAMES)
    assert [p.chunk.item_code for p in passages] == [TYLENOL]                   # 관련 약의 문서만
    assert "근거:\n- (타이레놀정500mg · 상호작용) (시험용) 다른 해열진통제와" in explain_input([DUP], NAMES, passages)
    assert passages_for(None, [DUP], NAMES) == []


def _doc(code, name, section, text):
    return {"doc_id": f"{code}:{section}", "item_code": code, "item_name": name, "source": "시험용", "section": section, "url": "", "text": text}


KEY = "아세트아미노펜을 포함하는 다른 제품과 함께 복용하지 마십시오."
DUP_AAP = DUP.model_copy(update={"message": "타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요."})


def test_the_excerpt_keeps_only_the_sentences_that_match_the_risk():
    text = f"음주자는 상의하십시오. 간손상을 일으킬 수 있습니다. 2. 이 약에는 아세트아미노펜이 함유되어 있다. {KEY} 3. 보관법은 따로 있다."
    out = focused_excerpt(text, finding_terms(DUP_AAP), limit=120)
    assert out == f"이 약에는 아세트아미노펜이 함유되어 있다. {KEY}"        # 앞뒤 관련 없는 문장 · 번호 제거, …도 없다
    assert "…" not in out


def test_a_matching_sentence_longer_than_the_limit_is_shown_whole():
    long_key = "아세트아미노펜을 포함하는 다른 제품과 함께 복용하여서는 안 되며 " + "매우 " * 60 + "주의해야 한다."
    assert focused_excerpt(long_key, finding_terms(DUP_AAP), limit=50) == long_key


def test_there_is_no_excerpt_without_a_sentence_that_matches():
    assert focused_excerpt("매일 세 잔 이상 정기적 음주자는 간손상을 일으킬 수 있습니다. " * 5, finding_terms(DUP_AAP)) is None
    assert focused_excerpt(KEY, ()) is None


def test_the_terms_of_a_duplicate_finding_include_the_shared_ingredient():
    assert ("아세트아미노펜", 1) in finding_terms(DUP_AAP) and ("다른 제품", 3) in finding_terms(DUP_AAP)
    assert finding_terms(DUP.model_copy(update={"type": "unknown"})) == ()


def _two_drug_index(with_key=(True, True)):
    docs = []
    for (code, name), has_key in zip(((TYLENOL, "타이레놀정500mg"), (GEVORIN, "게보린정")), with_key):
        docs.append(_doc(code, name, "상호작용", "바르비탈계 약물과 알코올을 복용하는 환자는 상의하십시오."))
        docs.append(_doc(code, name, "사용상의 주의사항", f"음주자는 상의하십시오. {KEY}" if has_key else "음주자는 상의하십시오. 피부 반응에 주의하십시오."))
    return RagIndex(chunk_docs(docs))


def test_each_drug_gets_the_passage_that_says_not_to_take_with_other_products():
    passages = passages_for(_two_drug_index(), [DUP_AAP], NAMES)
    assert {p.chunk.item_code for p in passages} == {TYLENOL, GEVORIN}
    (out,) = attach_evidence([DUP_AAP], passages)
    assert [e.title.split(" · ")[1] for e in out.evidence] == ["사용상의 주의사항", "사용상의 주의사항"]   # 상호작용 항목이 아니라 중복을 말하는 항목
    assert {e.title.split(" · ")[0] for e in out.evidence} == {"타이레놀정500mg", "게보린정"}
    assert all(e.excerpt == KEY and "…" not in e.excerpt for e in out.evidence)


def test_evidence_is_left_out_for_a_drug_without_a_matching_sentence():
    passages = passages_for(_two_drug_index(with_key=(False, True)), [DUP_AAP], NAMES)
    (out,) = attach_evidence([DUP_AAP], passages)
    assert [e.title.split(" · ")[0] for e in out.evidence] == ["게보린정"]        # 근거가 없는 약은 억지로 채우지 않는다
    (none,) = attach_evidence([DUP_AAP], passages_for(_two_drug_index(with_key=(False, False)), [DUP_AAP], NAMES))
    assert none.evidence == []                                                   # 둘 다 없으면 '근거 없음'


def test_cited_evidence_of_one_drug_gets_the_other_drug_added():
    from agent.schema import Evidence

    cited = DUP_AAP.model_copy(update={"evidence": [Evidence(title="게보린정 · 상호작용", source="시험용", excerpt="인용한 근거")]})
    (out,) = attach_evidence([cited], passages_for(_two_drug_index(), [DUP_AAP], NAMES))
    assert [e.title.split(" · ")[0] for e in out.evidence] == ["게보린정", "타이레놀정500mg"]   # 인용은 그대로, 뒤에 빠진 약
    assert out.evidence[0].excerpt == "인용한 근거"
    (again,) = attach_evidence([out], passages_for(_two_drug_index(), [DUP_AAP], NAMES))
    assert again.evidence == out.evidence                                          # 다시 해도 그대로


REPEAT = "타이레놀정500mg과 게보린정은 아세트아미노펜 성분이 겹칩니다."
NEW_INFO = "타이레놀정500mg과 게보린정은 모두 아세트아미노펜을 포함합니다. 환자의 아세트아미노펜 하루 총량(최대 4,000mg)을 확인하세요."
DUP_MSG = "타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량이 넘지 않는지 확인해 주세요."


def test_an_explanation_that_only_repeats_the_rule_sentence_is_dropped():
    assert repeats(REPEAT, DUP_MSG) and not repeats(NEW_INFO, DUP_MSG)
    dup = DUP.model_copy(update={"message": DUP_MSG})
    got = {"items": [{"index": 1, "text": REPEAT}]}
    assert apply_explanations([dup], got, NAMES)[0].explanation is None            # 되풀이는 버린다 (화면에는 규칙 문장만)
    got = {"items": [{"index": 1, "text": NEW_INFO}]}
    assert apply_explanations([dup], got, NAMES)[0].explanation == NEW_INFO         # 새 내용(한도)이 있으면 둔다


# ------------------------------------------------------------ 노드


def order_with(*names):
    from agent.drug_match import match_order
    from agent.schema import DispensingOrder

    items = [{"item_code": None, "name": n, "count_per_dose": 1, "times": ["morning"], "days": 1, "source_line_ids": [i]}
             for i, n in enumerate(names)]
    return match_order(DispensingOrder.model_validate({"prescription_id": "P0001", "items": items}), DICT).model_dump(mode="json")


def deps(**over):
    return Deps(dictionary=DICT, structure=lambda *a: pytest.fail("구조화"), **over)


def test_the_prescription_check_gets_explanations_and_keeps_its_level():
    seen = []
    update = risk_node({"order": order_with("타이레놀정500mg", "게보린정")},
                       deps(explain=lambda text: seen.append(text) or {"items": [{"index": 1, "text": GOOD_DUP}]}))
    risk = update["order"]["risk_check"]
    assert risk["level"] == "warning" and risk["findings"][0]["explanation"] == GOOD_DUP
    assert "단계: 경고" in seen[0] and "notes" not in update


def test_no_findings_means_no_llm_call():
    update = risk_node({"order": order_with("타이레놀정500mg")}, deps(explain=lambda t: pytest.fail("항목이 없으면 부르지 않는다")))
    assert update["order"]["risk_check"]["findings"] == []


def test_a_missing_key_leaves_the_rule_sentence_and_a_note():
    def no_key(text):
        raise ConfigError("OPENAI_API_KEY가 설정되지 않았습니다.")

    update = risk_node({"order": order_with("타이레놀정500mg", "게보린정")}, deps(explain=no_key))
    assert update["order"]["risk_check"]["findings"][0]["explanation"] is None
    assert update["notes"][0].startswith(FAILED_NOTE)


def test_the_test_graph_never_uses_the_llm_for_explanations():
    from agent.graph import _make_deps

    assert _make_deps(DICT, lambda *a: None, None).explain is None


def test_web_findings_without_an_explanation_still_validate():
    raw = json.loads(json.dumps({"type": "ingredient_dup", "item_codes": [TYLENOL, GEVORIN], "level": "warning", "message": "중복"}))
    assert RiskFinding.model_validate(raw).explanation is None


def test_explanations_can_be_switched_off(monkeypatch):
    from agent import config
    from agent.graph import _FROM_SETTINGS, _make_deps

    monkeypatch.setattr("agent.graph.default_api", lambda: None)
    monkeypatch.setattr("agent.graph.default_dur_api", lambda: None)
    monkeypatch.setattr("agent.graph.load_index", lambda folder, search=None: None)
    monkeypatch.setattr(config, "RISK_EXPLAIN", False)
    assert _make_deps(DICT, lambda *a: None, _FROM_SETTINGS).explain is None
    monkeypatch.setattr(config, "RISK_EXPLAIN", True)
    assert _make_deps(DICT, lambda *a: None, _FROM_SETTINGS).explain is not None
