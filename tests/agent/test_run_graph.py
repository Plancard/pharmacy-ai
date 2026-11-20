"""run_graph.py(그래프 실행 스크립트)의 테스트. 진짜 OpenAI도 진짜 LangGraph도 부르지 않는다 (시험용 가짜 LangGraph)."""
import importlib.util
from pathlib import Path

import pytest

import fake_langgraph
from agent.checkpoint import memory_checkpointer
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import E_LLM, ConfigError, StructureError
from agent.graph import build_graph, build_prescription_graph
from agent.pipeline import structure_prescription

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "run_graph.py"
_spec = importlib.util.spec_from_file_location("run_graph", SCRIPT)
run_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_graph)

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005", *(f"rx8_{i:02d}" for i in range(1, 21))]
WAITING = "승인 대기(rx_approval)"
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")


def no_llm(text, feedback):
    raise AssertionError("규칙 경로가 성공하면 LLM을 부르지 않아야 한다")


def rule_only(prescription_id, ocr):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


@pytest.fixture
def graph(monkeypatch):
    fake_langgraph.install(monkeypatch)
    return build_graph(dictionary=DICT, structure=rule_only, api=None, checkpointer=memory_checkpointer())


@pytest.fixture
def sub_graph(monkeypatch):
    fake_langgraph.install(monkeypatch)
    return build_prescription_graph(dictionary=DICT, structure=rule_only, api=None, checkpointer=memory_checkpointer())


def run(argv, graph):
    lines = []
    code = run_graph.main(argv, graph=graph, out=lines.append)
    return code, "\n".join(lines)


def test_the_samples_are_the_cv_prescriptions_and_the_demo_prescriptions():
    assert run_graph.sample_ids() == SAMPLE_IDS


def test_it_shows_the_final_state_of_one_sample(graph):
    code, text = run([], graph)                                      # 기본은 첫 샘플(rx_001)
    assert code == 0
    assert f"[rx_001] 상태: {WAITING} | 흐름: prescription" in text
    assert ("안내: 조제 계획을 만들 수 없습니다. 약을 정하지 못한 줄이 있습니다: 2번째 약품(알모란정)"
            " 고친 뒤 다시 승인해 주세요. 약사 승인을 기다립니다. 확인이 필요한 약품이 2줄 있습니다.") in text
    assert "조제 계획: 만들 수 없음. 고친 뒤(edit) 승인할 수 있습니다" in text
    assert "경로: rule | 교부번호: 2026년11월03일 00012호" in text
    assert "1. [확인] 타이레놀정 5oOmg | 코드 900000011 -> 품목 202106092 (name) | 1개 x 3회 x 3일 | morning, noon, evening" in text
    assert "2. [확인] 알모란정 | 코드 900000031 -> 품목 - (none)" in text
    assert "- 글자 인식 신뢰도가 낮습니다(0.43)." in text
    assert "경고: 의약품 API를 쓰지 않아" in text                      # 사전에 없는 약이 있으면 API가 없었다고 알린다


def test_it_shows_the_risk_check_with_the_ingredient_duplicate_of_combination_a(graph):
    _, text = run(["rx_004"], graph)                                 # 타이레놀 · 게보린 (시연 조합 A)
    assert "위험 신호: 경고 (처방약끼리 대조, 1건)" in text
    assert "- [경고] 타이레놀정500mg · 게보린정에 아세트아미노펜 성분이 함께 들어 있습니다. 하루 총량을 확인하세요." in text
    assert "위험 신호가 1건 있습니다(경고). 약사가 확인한 뒤에 승인할 수 있습니다." in text      # 안내에도 나온다


def test_it_shows_none_when_the_prescription_drugs_share_no_ingredient(graph):
    _, text = run(["rx_001"], graph)
    assert "위험 신호: 해당 없음 (처방약끼리 대조, 0건)" in text
    assert "- [" not in text.split("위험 신호:")[1].split("경고:")[0]


def test_it_tells_which_drugs_could_not_be_checked(graph):
    _, text = run(["rx_001"], graph)
    assert "경고: 알모란정: 품목기준코드를 정하지 못해 성분 중복을 대조하지 못했습니다." in text


def test_the_plan_is_shown_on_the_approval_card_before_approval(graph):
    _, text = run(["rx8_01"], graph)
    assert "    조제 계획: (승인 카드에 보임)\n      아침: 1일 타이레놀정 500mg 1 · 게보린정 1" in text


def test_approve_confirms_the_plan_shown_on_the_card(graph):
    code, text = run(["rx8_01", "--approve"], graph)
    assert code == 0
    first, second = text.split("    -> 약사가 화면의 조제지시서를 그대로 승인합니다\n")
    assert f"[rx8_01] 상태: {WAITING}" in first and "조제 계획: (승인 카드에 보임)" in first
    assert second.startswith("[rx8_01] 상태: planned | 흐름: prescription | 순서: prescription\n    안내: 약사가 승인했습니다.")
    assert "    조제 계획:\n      아침: 1일 타이레놀정 500mg 1 · 게보린정 1 | 2일 타이레놀정 500mg 1 · 게보린정 1 | 3일 타이레놀정 500mg 1 · 게보린정 1" in second


def test_a_prescription_whose_plan_cannot_be_made_is_not_approved(graph):
    code, text = run(["rx8_18", "--approve"], graph)                  # 투약 4일 (트레이는 3일까지)
    assert code == 0                                                  # 약사에게 돌아간 것은 실패가 아니다
    assert "조제 계획: 만들 수 없음 (E_PLAN_RANGE). 고친 뒤(edit) 승인할 수 있습니다" in text
    assert "-> 그대로 승인할 수 없습니다: 조제 계획을 만들지 못해 승인할 수 없습니다." in text
    assert "그대로 승인합니다" not in text


def test_a_line_without_an_item_code_is_not_approved(graph):
    _, text = run(["rx_001", "--approve"], graph)
    assert "-> 그대로 승인할 수 없습니다: 약을 정하지 못한 줄(품목기준코드 없음)이 있어 승인할 수 없습니다." in text


def test_the_demo_prescriptions_are_matched_by_their_printed_code(graph):
    _, text = run(["rx8_05"], graph)
    assert "1. [확인] 뮤코다인캡슐 200mg | 코드 200308774 -> 품목 200308774 (code) | 2개 x 2회 x 3일 | morning, evening" in text
    assert "2. [    ] 트립라인정 | 코드 200002714 -> 품목 200002714 (code) | 1개 x 1회 x 3일 | bedtime" in text


def test_the_times_and_days_of_every_line_are_shown(graph):
    _, text = run(["rx_002"], graph)
    assert "1. [확인] 게보린정 | 코드 900000021 -> 품목 197900277 (name) | 1개 x 2회 x 2일 | morning, evening" in text
    assert "2. [확인] 프론크논캠술 | 코드 900000051 -> 품목 200200099 (name) | 1개 x 3회 x 2일 | morning, noon, evening" in text


def test_a_sample_can_be_chosen(graph):
    code, text = run(["rx_005"], graph)
    assert code == 0 and text.startswith(f"[rx_005] 상태: {WAITING}")
    assert "rx_001" not in text


def test_it_never_prints_patient_information(graph):
    _, text = run(["--all"], graph)
    for private in ("김가상", "000000101", "900101", "주민등록번호", "02-000-0000", "이테스트"):
        assert private not in text


def test_all_runs_every_sample_and_counts_them(graph):
    code, text = run(["--all"], graph)
    assert code == 0 and text.endswith("전체 25장 중 25장이 약사 승인 대기까지 왔습니다.")
    for sample_id in SAMPLE_IDS:
        assert f"[{sample_id}] 상태: {WAITING}" in text
    assert "오류 코드" not in text                                    # 성공한 처방에는 오류 코드 줄이 없다


def test_one_failed_sample_does_not_stop_the_others_and_is_not_counted_as_reached(monkeypatch):
    fake_langgraph.install(monkeypatch)

    def fail_rx_003(prescription_id, ocr):
        if prescription_id == "RUN-rx_003":
            raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code=E_LLM)
        return rule_only(prescription_id, ocr)

    code, text = run(["--all"], build_graph(dictionary=DICT, structure=fail_rx_003, api=None, checkpointer=memory_checkpointer()))
    assert code == 1 and text.endswith("전체 25장 중 24장이 약사 승인 대기까지 왔습니다.")
    assert "[rx_003] 상태: failed" in text and f"[rx_004] 상태: {WAITING}" in text


def test_a_failed_structure_is_reported_and_exits_nonzero(monkeypatch):
    fake_langgraph.install(monkeypatch)

    def fail(prescription_id, ocr):
        raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요.", code=E_LLM)

    code, text = run(["rx_001"], build_graph(dictionary=DICT, structure=fail, api=None, checkpointer=memory_checkpointer()))
    assert code == 1
    assert "[rx_001] 상태: failed | 흐름: prescription" in text
    assert "오류: [structure] E_LLM 자동 구조화에 실패했습니다." in text and "    오류 코드: E_LLM" in text
    assert "경로:" not in text                                        # 조제지시서가 없다


# ------------------------------------------------------------ 처방 서브그래프만 단독으로


def test_the_subgraph_alone_shows_the_same_result_without_the_flow(sub_graph, graph):
    code, alone = run(["rx_004", "--subgraph"], sub_graph)
    assert code == 0
    assert f"[rx_004] 상태: {WAITING} | 흐름: -" in alone         # 슈퍼바이저를 거치지 않아 흐름이 비어 있다
    _, top = run(["rx_004"], graph)
    assert alone.replace("흐름: - | 순서: -", "흐름: prescription | 순서: prescription") == top   # 그 밖에는 한 글자도 다르지 않다


def test_the_subgraph_alone_gives_combination_a_in_its_risk_check(sub_graph):
    _, text = run(["rx_004", "--subgraph"], sub_graph)
    assert "위험 신호: 경고 (처방약끼리 대조, 1건)" in text and "아세트아미노펜 성분이 함께 들어 있습니다" in text


def test_the_subgraph_alone_runs_every_sample_like_the_top_graph(sub_graph, graph):
    code, alone = run(["--all", "--subgraph"], sub_graph)
    _, top = run(["--all"], graph)
    assert code == 0 and alone.endswith("전체 25장 중 25장이 약사 승인 대기까지 왔습니다.")
    assert alone.replace("흐름: - | 순서: -", "흐름: prescription | 순서: prescription") == top


def test_the_subgraph_alone_reports_a_failure(sub_graph):
    code, text = run(["--subgraph", "--no-ocr"], sub_graph)
    assert code == 1 and "[rx_001] 상태: failed | 흐름: -" in text and "오류 코드: E_OCR_EMPTY" in text


def test_the_subgraph_flag_runs_the_subgraph_and_nothing_else_does(monkeypatch, sub_graph, graph):
    calls = []
    real_prescription, real_message = run_graph.run_prescription, run_graph.run_message
    monkeypatch.setattr(run_graph, "run_prescription", lambda *a, **k: calls.append("prescription") or real_prescription(*a, **k))
    monkeypatch.setattr(run_graph, "run_message", lambda *a, **k: calls.append("message") or real_message(*a, **k))
    run_graph.main(["rx_004", "--subgraph"], graph=sub_graph, out=lambda line: None)
    run_graph.main(["rx_004"], graph=graph, out=lambda line: None)
    assert calls == ["prescription", "message"]


def test_the_subgraph_alone_cannot_be_given_a_photo_kind(sub_graph, capsys):
    with pytest.raises(SystemExit) as stop:
        run_graph.main(["--subgraph", "--kind", "tray"], graph=sub_graph, out=lambda line: None)
    assert stop.value.code == 2
    assert "--subgraph" in capsys.readouterr().err


# ------------------------------------------------------------ 슈퍼바이저의 분기


@pytest.mark.parametrize("argv, status, flow, notice", [
    (["--kind", "tray", "--no-ocr"], "noted", "tray", "조제 계획이 아직 없습니다. 조제지시서를 승인한 뒤에 트레이 사진을 올려 주세요."),
    (["--kind", "prescription"], WAITING, "prescription", "약사 승인을 기다립니다."),
    (["--no-ocr"], "noted", "-", "사진 종류 정보가 없습니다."),
    (["--kind", "receipt"], "noted", "-", "받을 수 없는 사진 종류입니다: receipt"),
    (["--kind", "med_bag"], "noted", "-", "받을 수 없는 사진 종류입니다: med_bag"),     # v0.8: 약봉투 제외
])
def test_the_photo_kind_decides_the_flow(graph, argv, status, flow, notice):
    code, text = run(argv, graph)
    assert code == 0                                                  # 안내만 남기고 끝나는 것은 실패가 아니다
    assert f"[rx_001] 상태: {status} | 흐름: {flow}" in text
    assert notice in text
    if flow != "prescription":
        assert "경로:" not in text                                    # 이 흐름들은 조제지시서를 만들지 않는다


# ------------------------------------------------------------ 그래프 만들기


def test_the_screen_output_can_be_written_in_cp949(graph):
    """윈도우 터미널(cp949)에서 글자가 깨지거나 멈추지 않는다."""
    outputs = [run(argv, graph)[1] for argv in (["--all"], ["--no-ocr"], ["--kind", "tray", "--no-ocr"], ["--kind", "receipt"])]
    outputs.append(run(["--all", "--subgraph"], build_prescription_graph(dictionary=DICT, structure=rule_only, api=None, checkpointer=memory_checkpointer()))[1])
    outputs.append(run(["rx8_18", "--approve"], build_graph(dictionary=DICT, structure=rule_only, api=None, checkpointer=memory_checkpointer()))[1])
    for text in outputs:
        text.encode("cp949")                                          # 못 쓰는 글자가 있으면 여기서 UnicodeEncodeError


@pytest.fixture
def no_saver(monkeypatch):
    """그래프를 만드는 함수를 바꿔 끼운 시험에서 체크포인터는 아무것이나 넘긴다 (langgraph 없이도 돌게)."""
    monkeypatch.setattr(run_graph, "memory_checkpointer", lambda: "saver")


def test_a_broken_setting_is_reported_and_exits_nonzero(monkeypatch, no_saver):
    def broken(**kwargs):
        raise ConfigError("약품 사전 파일이 없습니다: x.json")

    monkeypatch.setattr(run_graph, "build_graph", broken)
    lines = []
    assert run_graph.main([], out=lines.append) == 1
    assert lines == ["설정 문제: 약품 사전 파일이 없습니다: x.json"]


def test_a_missing_langgraph_is_explained(monkeypatch, no_saver):
    def missing(**kwargs):
        raise ModuleNotFoundError("No module named 'langgraph'", name="langgraph")

    monkeypatch.setattr(run_graph, "build_graph", missing)
    lines = []
    assert run_graph.main([], out=lines.append) == 1
    assert lines == ["langgraph가 설치되어 있지 않습니다. pip install -r agent/requirements.txt 로 설치한 뒤 다시 돌려 주세요."]


def test_a_missing_langgraph_for_the_checkpointer_is_explained_too(monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'langgraph.checkpoint'", name="langgraph.checkpoint")

    monkeypatch.setattr(run_graph, "memory_checkpointer", missing)
    lines = []
    assert run_graph.main([], out=lines.append) == 1 and lines[0].startswith("langgraph가 설치되어 있지 않습니다.")


def test_another_missing_package_is_not_hidden(monkeypatch, no_saver):
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
    code = run_graph.main(["rx_001"], graph=build_graph(dictionary=DICT, structure=broken, api=None, checkpointer=memory_checkpointer()), out=lines.append)
    assert code == 1 and lines == ["설정 문제: OPENAI_API_KEY가 없습니다."]


class SpyBuild:
    """그래프를 만드는 함수 대신 끼워 어떤 인자로 불렸는지 보고, 시험용 그래프를 돌려준다."""

    def __init__(self, monkeypatch, name="build_graph", builder=build_graph):
        fake_langgraph.install(monkeypatch)
        self.kwargs = None
        self.builder = builder
        monkeypatch.setattr(run_graph, name, self)

    def __call__(self, **kwargs):
        self.checkpointer = kwargs.pop("checkpointer")
        self.kwargs = kwargs
        return self.builder(dictionary=DICT, structure=rule_only, api=None, checkpointer=self.checkpointer)


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


def test_the_subgraph_flag_builds_only_the_subgraph(monkeypatch):
    top = SpyBuild(monkeypatch)
    sub = SpyBuild(monkeypatch, "build_prescription_graph", build_prescription_graph)
    lines = []
    assert run_graph.main(["rx_004", "--subgraph", "--no-api"], out=lines.append) == 0
    assert sub.kwargs == {"api": None} and top.kwargs is None          # 상위 그래프는 만들지도 않았다
    assert f"[rx_004] 상태: {WAITING} | 흐름: -" in "\n".join(lines)


def test_the_top_graph_is_built_without_the_subgraph_flag(monkeypatch):
    top = SpyBuild(monkeypatch)
    sub = SpyBuild(monkeypatch, "build_prescription_graph", build_prescription_graph)
    run_graph.main(["rx_004"], out=lambda line: None)
    assert top.kwargs == {} and sub.kwargs is None


def test_it_says_where_the_stop_is_kept_and_uses_memory_by_default(monkeypatch):
    spy = SpyBuild(monkeypatch)
    lines = []
    run_graph.main(["rx_001"], out=lines.append)
    assert lines[1] == "멈춘 곳 저장: 메모리 (프로세스가 끝나면 사라진다)"
    assert type(spy.checkpointer).__name__ == "InMemorySaver"


def test_the_db_flag_uses_the_checkpointer_of_the_settings(monkeypatch):
    spy = SpyBuild(monkeypatch)
    made = []
    monkeypatch.setattr(run_graph, "default_checkpointer", lambda: made.append(1) or memory_checkpointer())
    run_graph.main(["rx_001", "--db"], out=lambda line: None)
    assert made == [1] and type(spy.checkpointer).__name__ == "InMemorySaver"
