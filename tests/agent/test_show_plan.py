"""show_plan.py(샘플 조제지시서로 조제 계획을 만들어 보는 스크립트)의 테스트."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "show_plan.py"
_spec = importlib.util.spec_from_file_location("show_plan", SCRIPT)
show_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(show_plan)


def run(argv):
    lines = []
    code = show_plan.main(argv, out=lines.append)
    return code, "\n".join(lines)


def test_the_samples_are_the_five_team_orders():
    assert show_plan.sample_ids() == ["s01_combo_a", "s02_combo_b", "s03_no_risk", "s04_flagged", "s05_no_code"]


def test_sample_a_is_shown_and_equals_the_a4_sample():
    code, text = run([])
    assert code == 0
    assert text.startswith("[s01_combo_a] 약품 2줄: 타이레놀정500mg · 게보린정\n    조제 계획: 3일, 약이 든 칸 9개")
    assert "      아침: 1일 타이레놀정500mg 1 · 게보린정 1 | 2일 타이레놀정500mg 1 · 게보린정 1 | 3일 타이레놀정500mg 1 · 게보린정 1" in text
    assert text.endswith("A.4 샘플(a4_plan/s01_combo_a.json)과 같습니다")


def test_all_samples_are_shown_with_their_answer():
    code, text = run(["--all"])
    assert code == 0
    for name in show_plan.sample_ids():
        assert f"[{name}]" in text
    assert text.count("계획을 만들 수 없음 (E_PLAN_RANGE)") == 2            # s03 · s04
    assert "약을 정하지 못한 줄이 있습니다: 4번째 약품(알모탄정)" in text
    assert ("계획을 만들 수 없음 (E_PLAN_RANGE): 조제 계획을 만들 수 없습니다. 트레이 칸당 최대 2알을 넘습니다: 1~3일 아침·저녁 3알"
            in text)                                                   # s03: 프론크논 1알 + 뮤코다인 2알


def test_a_plan_that_differs_from_the_a4_sample_ends_with_one(monkeypatch, tmp_path):
    plans = tmp_path / "a4_plan"
    plans.mkdir()
    sample = json.loads((show_plan.PLANS / "s01_combo_a.json").read_text(encoding="utf-8"))
    sample["plan"] = sample["plan"][:-1]
    (plans / "s01_combo_a.json").write_text(json.dumps(sample), encoding="utf-8")
    monkeypatch.setattr(show_plan, "PLANS", plans)
    code, text = run(["s01_combo_a"])
    assert code == 1 and text.endswith("A.4 샘플(a4_plan/s01_combo_a.json)과 다릅니다")


def test_the_screen_output_can_be_written_in_cp949():
    run(["--all"])[1].encode("cp949")
