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
    assert code == 0 and text.endswith("전체 7개 중 7개가 기대대로 돌았습니다.")
    assert text.count(": 기대대로") == 7


def test_the_story_shows_the_risk_the_plan_and_the_tray_checks(backend):
    _, text = run()
    assert "위험 신호(warning): 트리스펜연질캡슐 · 아르펜정에 이부프로펜 성분이 함께 들어 있습니다." in text
    assert "계획 아침: 1일 타이레놀정 500mg 1 · 게보린정 1" in text
    assert "계획(승인 카드) 아침: 1일 타이레놀정 500mg 1 · 게보린정 1" in text   # 계획은 승인 전에 보인다 (v0.8)
    assert "근거: 아르펜정 · 상호작용 (식약처" in text                           # RAG 근거 (A.5 evidence)
    assert "계획: 만들 수 없음 (E_PLAN_RANGE) - 고치기(edit)만 할 수 있음" in text
    assert "거절: 조제 계획을 만들지 못해 승인할 수 없습니다." in text
    assert "안내: 이 처방에는 이미 처방전이 있습니다. 새 처방전은 새 처방으로 올려 주세요." in text
    assert "승인된 조제지시서 · 조제 계획: 그대로" in text                    # 같은 처방의 두 번째 메시지 (11/12)
    assert "검수: 트레이 검수: 전 칸 일치 (12칸)." in text                       # 트레이 검수 (14단계)
    assert "멈춤: inspection - 트레이 검수: 불일치 1칸, 확인 필요 0칸." in text
    assert "오류: E_TRAY_NOT_FOUND" in text and "단계: await_tray (조제 계획은 그대로" in text
    for gone in ("type_select", "메모:", "제안(", "재검토 필요", "상담"):         # v0.8에서 뺀 기능은 보이지 않는다
        assert gone not in text


def test_quiet_shows_one_line_per_scenario(backend):
    code, text = run(["--quiet"])
    assert code == 0 and len(text.splitlines()) == 8


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
