"""그래프 상태에 넣는 OCR 줄이기 (structure.agent_view, 11/13).

그래프 상태는 체크포인터(PostgreSQL)에 단계마다 쌓인다. 그래서 그래프에 들어가기 전에 약품 표 밖 칸(환자 · 의료인 · 의료기관)의
글자를 비운다. Agent가 읽는 결과(규칙 경로 · LLM 입력 · 승인 화면의 원문 줄)는 같아야 한다.
"""
import copy
import json

import pytest

import fake_langgraph
from agent.api import AgentApi
from agent.checkpoint import memory_checkpointer
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import StructureError
from agent.graph import build_graph, thread_config
from agent.pipeline import structure_prescription
from agent.state import initial_state, message_input
from agent.structure import agent_view, extract_rule, llm_input
from test_skew import tilt

SAMPLES = RESOURCES_DIR / "samples"
ALL = [p for f in ("rx8_a1", "cv_a1", "a1_ocr") for p in sorted((SAMPLES / f).glob("*.json"))
       if "lines" in json.loads(p.read_text(encoding="utf-8"))]
PERSONAL = ("김가상", "900101", "02-000-0000", "02-000-0001", "가상의료재단", "99990001")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read(ocr):
    try:
        return extract_rule(ocr)
    except StructureError as e:
        return str(e)


@pytest.mark.parametrize("deg", [0.0, 3.0, -6.0])
@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_the_agent_reads_the_same_from_the_trimmed_ocr(path, deg):
    ocr = tilt(load(path), deg) if deg else load(path)
    view = agent_view(ocr)
    assert read(view) == read(ocr) and llm_input(view) == llm_input(ocr)


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_personal_and_institution_text_is_gone(path):
    text = " ".join(l["text"] for l in agent_view(load(path))["lines"])
    assert not [p for p in PERSONAL if p in text]


def test_every_cell_keeps_its_place_and_id_and_only_its_text_may_be_emptied():
    ocr = load(SAMPLES / "rx8_a1" / "rx8_01.json")
    view = agent_view(ocr)
    assert [(l["line_id"], l["bbox"], l["conf"]) for l in view["lines"]] == [(l["line_id"], l["bbox"], l["conf"]) for l in ocr["lines"]]
    assert all(v["text"] in ("", o["text"]) for v, o in zip(view["lines"], ocr["lines"]))
    assert {k: v for k, v in view.items() if k != "lines"} == {k: v for k, v in ocr.items() if k != "lines"}
    assert ocr == load(SAMPLES / "rx8_a1" / "rx8_01.json")                  # 받은 OCR은 바꾸지 않는다


def test_the_drug_rows_and_the_issue_number_stay():
    text = [l["text"] for l in agent_view(load(SAMPLES / "rx8_a1" / "rx8_01.json"))["lines"] if l["text"]]
    assert "2026년11월11일 00101호" in text and any(t.startswith("(1)202106092") for t in text)


def test_something_that_is_not_a1_is_left_as_it_is():
    for odd in ({"lines": []}, {"doc_type": "med_bag"}, {"lines": [{"text": 1}]}):
        assert agent_view(odd) is odd


def test_a_personal_number_inside_the_table_is_masked_too():
    ocr = copy.deepcopy(load(SAMPLES / "rx8_a1" / "rx8_01.json"))
    row = next(l for l in ocr["lines"] if l["text"].startswith("(1)202106092"))
    ocr["lines"].append({**row, "line_id": 999, "text": "보호자 010-1234-5678", "bbox": [1500, row["bbox"][1], 1900, row["bbox"][3]]})
    texts = [l["text"] for l in agent_view(ocr)["lines"]]
    assert "010-1234-5678" not in " ".join(texts) and "보호자 [가림]" in texts


def test_the_graph_input_is_trimmed_for_the_first_and_the_next_messages():
    ocr = load(SAMPLES / "cv_a1" / "rx_001.json")
    assert initial_state("P1", {"kind": "prescription"}, "P1", ocr)["ocr"] == agent_view(ocr)
    assert message_input({"kind": "prescription"}, ocr)["ocr"] == agent_view(ocr)


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


def test_no_checkpoint_of_the_conversation_holds_personal_text(backend):
    dictionary = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
    graph = build_graph(dictionary=dictionary, api=None, checkpointer=memory_checkpointer(),
                        structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=lambda *a: pytest.fail("LLM")))
    api = AgentApi(graph, dictionary)
    pause = api.start("P0001", load(SAMPLES / "rx8_a1" / "rx8_01.json"))
    api.resume("P0001", "rx_approval", {"action": "approve", "order": pause.payload["order"]})
    if backend == "real":                                                 # 진짜 LangGraph: 단계마다 쌓인 체크포인트 전부
        history = [h.values for h in graph.get_state_history(thread_config("P0001"))]
        assert len(history) > 3
    else:                                                                 # 시험용 가짜에는 이력이 없다: 지금 상태만
        history = [graph.get_state(thread_config("P0001")).values]
    stored = json.dumps(history, ensure_ascii=False, default=str)
    assert not [p for p in PERSONAL if p in stored]
    assert pause.payload["ocr_lines"]                                      # 승인 화면의 원문 줄은 그대로 보인다
