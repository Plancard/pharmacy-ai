"""run_graph.py(상위 그래프 실행 스크립트)의 테스트. 진짜 OpenAI도 진짜 LangGraph도 부르지 않는다 (시험용 가짜 LangGraph)."""
import importlib.util
from pathlib import Path

import pytest

import fake_langgraph
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import E_LLM, ConfigError, StructureError
from agent.graph import build_graph
from agent.pipeline import structure_prescription

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "run_graph.py"
_spec = importlib.util.spec_from_file_location("run_graph", SCRIPT)
run_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_graph)

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")


def no_llm(text, feedback):
    raise AssertionError("규칙 경로가 성공하면 LLM을 부르지 않아야 한다")


def rule_only(prescription_id, ocr):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


@pytest.fixture
def graph(monkeypatch):
    fake_langgraph.install(monkeypatch)
    return build_graph(dictionary=DICT, structure=rule_only, api=None)


def run(argv, graph):
    lines = []
    code = run_graph.main(argv, graph=graph, out=lines.append)
    return code, "\n".join(lines)


def test_the_samples_are_the_five_cv_prescriptions():
    assert run_graph.sample_ids() == SAMPLE_IDS


def test_it_shows_the_final_state_of_one_sample(graph):
    code, text = run([], graph)                                      # 기본은 첫 샘플(rx_001)
    assert code == 0
    assert "[rx_001] 상태: awaiting_approval | 흐름: prescription" in text
    assert "안내: 약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다." in text
    assert "경로: rule | 교부번호: 2026년11월03일 00012호" in text
    assert "1. [확인] 타이레놀정 5oOmg | 코드 900000011 -> 품목 202106092 (name) | 1개 x 3회 x 3일 | morning, noon, evening" in text
    assert "2. [확인] 알모란정 | 코드 900000031 -> 품목 - (none)" in text
    assert "- 글자 인식 신뢰도가 낮습니다(0.43)." in text
    assert "경고: 의약품 API를 쓰지 않아" in text                      # 사전에 없는 약이 있으면 API가 없었다고 알린다


def test_the_times_and_days_of_every_line_are_shown(graph):
    _, text = run(["rx_002"], graph)
    assert "1. [확인] 게보린정 | 코드 900000021 -> 품목 197900277 (name) | 1개 x 2회 x 2일 | morning, evening" in text
    assert "2. [확인] 프론크논캠술 | 코드 900000051 -> 품목 - (none) | 1개 x 3회 x 2일 | morning, noon, evening" in text


def test_a_sample_can_be_chosen(graph):
    code, text = run(["rx_005"], graph)
    assert code == 0 and text.startswith("[rx_005] 상태: awaiting_approval")
    assert "rx_001" not in text


def test_it_never_prints_patient_information(graph):
    _, text = run(["--all"], graph)
    for private in ("김가상", "000000101", "900101", "주민등록번호", "02-000-0000", "이테스트"):
        assert private not in text


def test_all_runs_every_sample_and_counts_them(graph):
    code, text = run(["--all"], graph)
    assert code == 0 and text.endswith("전체 5장 중 5장이 약사 승인 대기까지 왔습니다.")
    for sample_id in SAMPLE_IDS:
        assert f"[{sample_id}] 상태: awaiting_approval" in text
    assert "오류 코드" not in text                                    # 성공한 처방에는 오류 코드 줄이 없다


def test_one_failed_sample_does_not_stop_the_others_and_is_not_counted_as_reached(monkeypatch):
    fake_langgraph.install(monkeypatch)

    def fail_rx_003(prescription_id, ocr):
        if prescription_id == "RUN-rx_003":
            raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code=E_LLM)
        return rule_only(prescription_id, ocr)

    code, text = run(["--all"], build_graph(dictionary=DICT, structure=fail_rx_003, api=None))
    assert code == 1 and text.endswith("전체 5장 중 4장이 약사 승인 대기까지 왔습니다.")
    assert "[rx_003] 상태: failed" in text and "[rx_004] 상태: awaiting_approval" in text


def test_a_failed_structure_is_reported_and_exits_nonzero(monkeypatch):
    fake_langgraph.install(monkeypatch)

    def fail(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code=E_LLM)

    code, text = run(["rx_001"], build_graph(dictionary=DICT, structure=fail, api=None))
    assert code == 1
    assert "[rx_001] 상태: failed | 흐름: prescription" in text
    assert "오류: [structure] E_LLM 자동 구조화에 실패했습니다." in text and "    오류 코드: E_LLM" in text
    assert "경로:" not in text                                        # 조제지시서가 없다


# ------------------------------------------------------------ 슈퍼바이저의 분기


@pytest.mark.parametrize("argv, status, flow, notice", [
    (["--kind", "med_bag", "--no-ocr"], "not_ready", "recheck", "복용 약 재대조 흐름은 아직 준비 중입니다."),
    (["--kind", "tray", "--no-ocr"], "not_ready", "tray", "트레이 검수 흐름은 아직 준비 중입니다."),
    (["--kind", "prescription"], "awaiting_approval", "prescription", "약사 승인을 기다립니다."),
    (["--no-ocr"], "awaiting_type", "ask_type", "사진 종류 정보가 없습니다."),
    (["--kind", "receipt"], "awaiting_type", "ask_type", "알 수 없는 사진 종류입니다."),
    (["--kind", "med_bag"], "awaiting_type", "ask_type", "지정한 종류(약 포장)와 OCR이 읽은 종류(처방전)가 다릅니다."),
])
def test_the_photo_kind_decides_the_flow(graph, argv, status, flow, notice):
    code, text = run(argv, graph)
    assert code == 0                                                  # 되묻거나 자리에서 멈추는 것은 실패가 아니다
    assert f"[rx_001] 상태: {status} | 흐름: {flow}" in text
    assert notice in text
    if flow != "prescription":
        assert "경로:" not in text                                    # 이 흐름들은 조제지시서를 만들지 않는다


# ------------------------------------------------------------ 그래프 만들기


def test_the_screen_output_can_be_written_in_cp949(graph):
    """윈도우 터미널(cp949)에서 글자가 깨지거나 멈추지 않는다."""
    outputs = [run(argv, graph)[1] for argv in (["--all"], ["--no-ocr"], ["--kind", "tray", "--no-ocr"], ["--kind", "med_bag"])]
    for text in outputs:
        text.encode("cp949")                                          # 못 쓰는 글자가 있으면 여기서 UnicodeEncodeError


def test_a_broken_setting_is_reported_and_exits_nonzero(monkeypatch):
    def broken(**kwargs):
        raise ConfigError("약품 사전 파일이 없습니다: x.json")

    monkeypatch.setattr(run_graph, "build_graph", broken)
    lines = []
    assert run_graph.main([], out=lines.append) == 1
    assert lines == ["설정 문제: 약품 사전 파일이 없습니다: x.json"]


def test_a_missing_langgraph_is_explained(monkeypatch):
    def missing(**kwargs):
        raise ModuleNotFoundError("No module named 'langgraph'", name="langgraph")

    monkeypatch.setattr(run_graph, "build_graph", missing)
    lines = []
    assert run_graph.main([], out=lines.append) == 1
    assert lines == ["langgraph가 설치되어 있지 않습니다. pip install langgraph 로 설치한 뒤 다시 돌려 주세요."]


def test_another_missing_package_is_not_hidden(monkeypatch):
    def missing(**kwargs):
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    monkeypatch.setattr(run_graph, "build_graph", missing)
    with pytest.raises(ModuleNotFoundError):
        run_graph.main([], out=lambda line: None)


def test_a_config_problem_while_running_is_reported(monkeypatch):
    fake_langgraph.install(monkeypatch)

    def broken(prescription_id, ocr):
        raise ConfigError("OPENAI_API_KEY가 없습니다.")

    lines = []
    code = run_graph.main(["rx_001"], graph=build_graph(dictionary=DICT, structure=broken, api=None), out=lines.append)
    assert code == 1 and lines == ["설정 문제: OPENAI_API_KEY가 없습니다."]


class SpyBuild:
    """build_graph 대신 끼워 어떤 인자로 불렸는지 보고, 시험용 그래프를 돌려준다."""

    def __init__(self, monkeypatch):
        fake_langgraph.install(monkeypatch)
        self.kwargs = None
        monkeypatch.setattr(run_graph, "build_graph", self)

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return build_graph(dictionary=DICT, structure=rule_only, api=None)


def test_it_says_whether_the_api_is_used(monkeypatch):
    spy = SpyBuild(monkeypatch)
    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    lines = []
    run_graph.main(["rx_001"], out=lines.append)
    assert lines[0] == "약품 API: 사용 (사전에 없는 약은 식약처 API에서 찾습니다)" and spy.kwargs == {}


def test_it_says_when_there_is_no_api_key(monkeypatch):
    spy = SpyBuild(monkeypatch)
    lines = []
    run_graph.main(["rx_001"], out=lines.append)
    assert lines[0] == "약품 API: 사용 안 함 (.env에 DRUG_API_KEY가 없어 약품 사전 8종만 씁니다)" and spy.kwargs == {}


def test_no_api_turns_the_api_off_even_with_a_key(monkeypatch):
    spy = SpyBuild(monkeypatch)
    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    lines = []
    run_graph.main(["rx_001", "--no-api"], out=lines.append)
    assert lines[0] == "약품 API: 사용 안 함 (--no-api, 약품 사전 8종만 씁니다)" and spy.kwargs == {"api": None}
