"""demo.py(뼈대 그래프 동작 확인)의 테스트. 시험용 가짜 LangGraph와 진짜 LangGraph(있을 때만)로 돌린다."""
import importlib.util
from pathlib import Path

import pytest

import fake_langgraph

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "demo.py"
_spec = importlib.util.spec_from_file_location("demo", SCRIPT)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


@pytest.fixture(params=["fake", "real"])
def backend(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    return request.param


def run(argv=(), **kwargs):
    lines = []
    code = demo.main(list(argv), out=lines.append, **kwargs)
    return code, "\n".join(lines)


def test_every_scenario_runs_as_expected(backend):
    code, text = run()
    assert code == 0 and text.endswith("전체 10개 중 10개가 기대대로 돌았습니다.")
    assert text.count(": 기대대로") == 10


def test_the_story_shows_the_proposal_the_plan_and_the_questions(backend):
    _, text = run()
    assert "제안(수정 제안): 게보린 빼 주세요  ← 약사가 확인해야 반영" in text
    assert "계획 아침: 1일 타이레놀정 500mg 1 · 게보린정 1" in text
    assert "멈춤: type_select" in text and "고를 수 있는 것: 처방전 · 트레이 · 약봉투 · 포장" in text
    assert "트레이 칸당 최대 2알을 넘습니다: 1~3일 아침·점심·저녁 3알" in text
    assert "메모: 게보린 빼 주세요" in text
    assert "승인된 조제지시서 · 조제 계획: 그대로" in text                    # 같은 대화의 두 번째 메시지 (11/12)
    assert "처방 ID: None | 의도: patient_info, question" in text             # 처방 없는 대화의 글 → 상담
    assert "검수: 트레이 검수: 전 칸 일치 (12칸)." in text                       # 트레이 검수 (14단계)
    assert "멈춤: inspection - 트레이 검수: 불일치 1칸, 확인 필요 0칸." in text
    assert "오류: E_TRAY_NOT_FOUND" in text and "단계: await_tray (조제 계획은 그대로" in text


def test_quiet_shows_one_line_per_scenario(backend):
    code, text = run(["--quiet"])
    assert code == 0 and len(text.splitlines()) == 11


def test_a_scenario_that_goes_wrong_is_shown_in_full_and_ends_with_one(backend, monkeypatch):
    monkeypatch.setattr(demo, "SCENARIOS", [*demo.SCENARIOS[:1], ("일부러 틀린 시나리오", lambda api, out: out("   자세한 줄") or False)])
    code, text = run(["--quiet"])
    assert code == 1 and "[2] 일부러 틀린 시나리오: 기대와 다름\n   자세한 줄" in text
    assert text.endswith("전체 2개 중 1개가 기대대로 돌았습니다.")


def test_a_missing_langgraph_is_explained():
    def missing():
        raise ModuleNotFoundError("No module named 'langgraph'", name="langgraph")

    code, text = run(api_factory=missing)
    assert code == 1 and text.startswith("langgraph가 설치되어 있지 않습니다.")


def test_the_screen_output_can_be_written_in_cp949(backend):
    run()[1].encode("cp949")
