"""LangGraph로 조립한 그래프 (app/graph.py)와 실행 스크립트(main.py).

langgraph가 설치되어 있어야 돈다. 없으면 건너뛴다(skip). 설치: pip install langgraph
"""
import json

import pytest

pytest.importorskip("langgraph")

import main
from agent.config import DATA_DIR
from agent.drug_dictionary import dictionary_from_data, load_dictionary
from agent.errors import ConfigError, StructureError
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder
from app.graph import build_graph, run_prescription

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
FLAGGED = {"rx_001": [0, 1], "rx_002": [2], "rx_003": [0, 1], "rx_004": [0, 2, 3], "rx_005": [0, 1, 2, 3]}


def load(kind, sample_id):
    return json.loads((DATA_DIR / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback=None):
    raise AssertionError("규칙 경로가 성공하면 LLM을 부르지 않아야 한다")


def rule_only(prescription_id, ocr):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


@pytest.fixture(scope="module")
def graph():
    # 파일 전체가 함께 쓰는 그래프라 테스트마다 키를 지우는 conftest보다 먼저 만들어진다.
    # 그래서 여기서는 API를 직접 끈다 (.env에 DRUG_API_KEY가 있어도 진짜 API를 부르지 않게).
    return build_graph(structure=rule_only, api=None)


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_every_sample_reaches_the_approval_step(graph, sample_id):
    final = run_prescription(graph, f"P-{sample_id}", load("cv_samples", sample_id))
    assert final["status"] == "awaiting_approval" and final["errors"] == [] and final["error_code"] is None
    order = DispensingOrder.model_validate(final["order"])
    assert order.prescription_id == f"P-{sample_id}"
    assert [d.item_code for d in order.items] == [d["item_code"] for d in load("cv_expected", sample_id)["items"]]
    assert order.flagged_indices == FLAGGED[sample_id]
    assert final["notice"].startswith("약사 승인을 기다립니다.")


def test_a_failed_structure_ends_the_graph_without_matching():
    calls = []

    def fail(prescription_id, ocr):
        calls.append(prescription_id)
        raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code="E_LLM")

    final = run_prescription(build_graph(structure=fail), "P1", load("cv_samples", "rx_001"))
    assert calls == ["P1"]
    assert final["status"] == "failed" and final["order"] is None
    assert final["errors"] == ["[structure] E_LLM 자동 구조화에 실패했습니다. 직접 입력해 주세요."]
    assert final["error_code"] == "E_LLM"
    assert final["notice"] == "자동 구조화에 실패했습니다. 직접 입력해 주세요."


def test_the_given_dictionary_is_used():
    only_gevorin = dictionary_from_data({"drugs": [{"item_code": "900000021", "name": "게보린정"}]})
    final = run_prescription(build_graph(dictionary=only_gevorin, structure=rule_only), "P1", load("cv_samples", "rx_002"))
    order = DispensingOrder.model_validate(final["order"])
    assert [d.item_code for d in order.items] == ["900000021", None, None]


def test_a_broken_dictionary_stops_the_build(monkeypatch):
    import app.graph

    def broken():
        raise ConfigError("약품 사전 파일이 없습니다: x.json")

    monkeypatch.setattr(app.graph, "load_dictionary", broken)
    with pytest.raises(ConfigError):
        build_graph(structure=rule_only)


def test_the_same_graph_runs_many_prescriptions_without_mixing_them(graph):
    first = run_prescription(graph, "P-A", load("cv_samples", "rx_001"))
    second = run_prescription(graph, "P-B", load("cv_samples", "rx_005"))
    assert first["prescription_id"] == "P-A" and len(first["order"]["items"]) == 2
    assert second["prescription_id"] == "P-B" and len(second["order"]["items"]) == 4
    assert first["errors"] == [] and second["errors"] == []


# ------------------------------------------------------------ main.py


def run_main(argv, graph):
    lines = []
    code = main.main(argv, graph=graph, out=lines.append)
    return code, "\n".join(lines)


def test_main_shows_the_final_state_of_one_sample(graph):
    code, text = run_main(["rx_001"], graph)
    assert code == 0
    assert "[rx_001] 상태: awaiting_approval" in text
    assert "안내: 약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다." in text
    assert "경로: rule | 교부번호: 2026년11월03일 00012호" in text
    assert "1. [확인] 타이레놀정 5oOmg | 코드 900000011 -> 품목 202106092 (code) | 1개 x 3회 x 3일 | morning, noon, evening" in text
    assert "2. [확인] 알모란정 | 코드 900000031 -> 품목 - (none)" in text
    assert "- 글자 인식 신뢰도가 낮습니다(0.43)." in text


def test_main_never_prints_patient_information(graph):
    _, text = run_main(["--all"], graph)
    for private in ("김가상", "000000101", "900101", "주민등록번호", "02-000-0000", "이테스트"):
        assert private not in text


def test_main_all(graph):
    code, text = run_main(["--all"], graph)
    assert code == 0 and "전체 5장 중 5장이 약사 승인 대기까지 왔습니다" in text
    assert "오류 코드" not in text                                   # 성공한 처방에는 오류 코드 줄이 없다
    assert "    1. [    ] 게보린정 | 코드 900000021 -> 품목 197900277 (code)" in text      # 확인이 필요 없는 줄


def test_main_reports_a_failure_and_exits_nonzero():
    def fail(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code="E_LLM")

    code, text = run_main(["rx_001"], build_graph(structure=fail))
    assert code == 1
    assert "[rx_001] 상태: failed" in text and "오류: [structure] E_LLM 자동 구조화에 실패했습니다." in text
    assert "    오류 코드: E_LLM" in text
    assert "경로:" not in text


def test_main_reports_a_broken_dictionary(monkeypatch):
    import app.graph

    def broken():
        raise ConfigError("약품 사전 파일이 없습니다: x.json")

    monkeypatch.setattr(app.graph, "load_dictionary", broken)
    code, text = run_main([], None)
    assert code == 1 and text == "약품 API: 사용 안 함 (.env에 DRUG_API_KEY가 없어 약품 사전 8종만 씁니다)\n설정 문제: 약품 사전 파일이 없습니다: x.json"


def test_main_default_sample_is_the_first():
    assert main.sample_ids() == SAMPLE_IDS


# ------------------------------------------------------------ 의약품 API 연결


class FakeApi:
    def __init__(self, products=()):
        self.products, self.calls = list(products), []

    def __call__(self, name):
        self.calls.append(name)
        return self.products


def test_the_given_api_is_used_by_the_match_node():
    from agent.drug_api import ApiDrug

    api = FakeApi([ApiDrug("199900031", "알모탄정", "가상제약")])
    final = run_prescription(build_graph(structure=rule_only, api=api), "P1", load("cv_samples", "rx_001"))
    order = DispensingOrder.model_validate(final["order"])
    assert api.calls == ["알모란정"]
    assert [(d.item_code, d.match_method) for d in order.items] == [("202106092", "code"), ("199900031", "api")]


def test_by_default_the_api_comes_from_the_settings(monkeypatch):
    import app.graph

    api = FakeApi()
    made = []
    monkeypatch.setattr(app.graph, "default_api", lambda: made.append(1) or api)
    run_prescription(build_graph(structure=rule_only), "P1", load("cv_samples", "rx_001"))
    assert made == [1] and api.calls == ["알모란정"]


def test_api_none_turns_the_api_off_even_with_a_key(monkeypatch):
    import app.graph

    monkeypatch.setattr(app.graph, "default_api", lambda: pytest.fail("API를 만들면 안 된다"))
    final = run_prescription(build_graph(structure=rule_only, api=None), "P1", load("cv_samples", "rx_001"))
    assert "의약품 API를 쓰지 않아" in " ".join(DispensingOrder.model_validate(final["order"]).warnings)


def test_main_says_whether_the_api_is_used(monkeypatch):
    import app.graph

    monkeypatch.setattr(app.graph, "default_api", lambda: FakeApi())
    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    _, text = run_main(["rx_001"], None)
    assert text.split("\n")[0] == "약품 API: 사용 (사전에 없는 약은 식약처 API에서 찾습니다)"


def test_main_no_api_option_turns_the_api_off_even_with_a_key(monkeypatch):
    import app.graph

    monkeypatch.setattr(app.graph, "default_api", lambda: pytest.fail("API를 만들면 안 된다"))
    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    code, text = run_main(["rx_001", "--no-api"], None)
    assert code == 0 and text.split("\n")[0] == "약품 API: 사용 안 함 (--no-api, 약품 사전 8종만 씁니다)"
