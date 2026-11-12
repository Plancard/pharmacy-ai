"""measure_accuracy.py(정확도 측정 스크립트)의 테스트. 진짜 OpenAI는 부르지 않는다."""
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from agent.errors import ConfigError, StructureError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "measure_accuracy.py"
_spec = importlib.util.spec_from_file_location("measure_accuracy", SCRIPT)
measure_accuracy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure_accuracy)

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
TODAY = date(2026, 11, 10)


def answer(sample_id):
    expected = measure_accuracy.load("cv_expected", sample_id)
    return {
        "issue_no": expected["issue_no"],
        "items": [
            {k: d[k] for k in ("name", "printed_code", "count_per_dose", "times_per_day", "days")} | {"times": []}
            for d in expected["items"]
        ],
    }


def fake_llm(outcomes):
    """처방전마다 정해 둔 결과(dict 또는 예외)를 돌려준다. 보낸 텍스트를 기록한다."""
    calls = []

    def llm(text, feedback=None):
        calls.append(text)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    llm.calls = calls
    return llm


def run(argv, llm=None):
    lines = []
    kwargs = {"out": lines.append, "today": lambda: TODAY}
    if llm is not None:
        kwargs["llm_fn"] = llm
    code = measure_accuracy.main(argv, **kwargs)
    return code, "\n".join(lines)


# ------------------------------------------------------------ 규칙 경로


def test_rule_path_only_by_default_and_the_llm_is_not_called():
    llm = fake_llm([])
    code, text = run([], llm)
    assert code == 0 and llm.calls == []
    assert "== 규칙 경로 ==" in text and "LLM 경로" not in text
    assert "처방전 5장, 약품 15줄 (그중 OCR이 약품명을 틀리게 읽은 줄 11줄)" in text
    assert "칸 정확도 (OCR 기준) 100.0% (80/80)  (목표 90% 달성)" in text
    assert "약품명 100.0% (15/15) · 1회 투약량 100.0% (15/15) · 1일 횟수 100.0% (15/15) · 투약 일수 100.0% (15/15)" in text
    assert "교부번호 100.0% (5/5) · 인쇄 코드 100.0% (15/15)" in text
    assert "약품 줄 완전 일치 100.0% (15/15) | 처방전 완전 일치 5/5 | 구조화 실패 0장" in text
    assert "진짜 값 기준 칸 정확도 98.5% (64/65)  (약품명 제외)" in text
    assert "약사 확인 표시 15줄" in text
    assert "rx_004 교부번호: 정답 '2026년11월03일 00058호', 결과 '2026년11월03일 000583'" in text
    assert "OCR 기준으로 틀린 칸" not in text


def test_each_sample_has_its_own_line():
    _, text = run([])
    assert "  rx_004  칸 21/21  약품 줄 4/4  약사 확인 4줄" in text


def test_the_matching_result_is_counted_and_the_candidate_lines_are_listed():
    _, text = run([])
    assert (
        "약품 매칭 (15줄): 정답 코드를 정함 7 · 정답을 후보로 보임 0 · 대상 약 아님 8 · 못 찾음 0 · 잘못 정함 0 · 줄 못 읽음 0"
    ) in text
    # OCR이 캠술 · 램술 · 편으로 읽은 줄도 OCR 한글 보정으로 정답 코드를 정한다 (11/10에는 후보 3줄이었다)
    assert "약품 매칭 - 정답을 후보로 보임:" not in text
    # 문제가 아닌 줄은 하나씩 풀어 쓰지 않는다
    assert "약품 매칭 - 정답 코드를 정함" not in text and "약품 매칭 - 대상 약 아님" not in text
    assert "약품 매칭 - 잘못 정함" not in text


def test_the_schedule_record_line_can_be_pasted_as_it_is():
    _, text = run([])
    after = text.split("== 일정표 정확도 기록 (그대로 붙여 넣으세요) ==\n")[1].split("\n")
    assert after[0] == (
        "2026-11-10 | 규칙 경로 | 처방전 5장 · 약품 15줄"
        " | 약품명 100.0% (15/15) · 1회 투약량 100.0% (15/15) · 1일 횟수 100.0% (15/15) · 투약 일수 100.0% (15/15)"
        " | 칸 정확도 100.0% (80/80) · 목표 90% 달성"
        " | 약품 매칭: 정답 코드 7줄 · 후보로 정답 0줄 · 잘못 정함 0줄"
    )


def test_sample_ids_are_the_five_real_samples():
    assert measure_accuracy.sample_ids() == SAMPLE_IDS


def test_time_per_sample_with_average_and_maximum(monkeypatch):
    # 처방전마다 (시작, 끝) 시각: 걸린 시간은 1, 2, 3, 4, 10초
    ticks = iter([0, 1, 10, 12, 20, 23, 30, 34, 40, 50])
    monkeypatch.setattr(measure_accuracy.time, "perf_counter", lambda: next(ticks))
    _, text = run([])
    assert "  rx_001  칸 11/11  약품 줄 2/2  약사 확인 2줄  1.0초" in text
    assert "  rx_005  칸 21/21  약품 줄 4/4  약사 확인 4줄  10.0초" in text
    assert "시간 평균 4.0초, 최대 10.0초" in text


# ------------------------------------------------------------ LLM 경로


def test_the_llm_path_is_measured_separately_with_only_the_drug_table_sent():
    llm = fake_llm([answer(s) for s in SAMPLE_IDS])
    code, text = run(["--llm"], llm)
    assert code == 0 and len(llm.calls) == 5
    assert text.count("칸 정확도 (OCR 기준) 100.0% (80/80)") == 2       # 두 경로 모두
    assert "== LLM 경로 ==" in text
    for sent in llm.calls:
        assert sent.startswith("교부번호: ") and "주민등록번호" not in sent and "김가상" not in sent


def test_llm_mistakes_and_failures_are_counted_and_listed():
    answers = [answer(s) for s in SAMPLE_IDS]
    answers[0]["items"][1]["name"] = "알모탄정"           # OCR 오류를 고쳐 옴: OCR 기준으로 틀림
    answers[1]["items"][0]["name"] = "타이레놀정 500mg"    # 게보린을 타이레놀로 읽어 옴: 틀린 약이 정해진다
    answers[3]["items"][2]["days"] = 4
    answers[4] = StructureError("LLM 호출에 실패했습니다 (APITimeoutError).")
    code, text = run(["--llm"], fake_llm(answers))
    llm_part = text.split("== LLM 경로 ==")[1]
    assert code == 0
    assert "rx_005  칸 0/21  약품 줄 0/4  약사 확인 0줄" in llm_part
    assert "[실패] LLM 호출에 실패했습니다 (APITimeoutError)." in llm_part
    assert "처방전 완전 일치 1/5 | 구조화 실패 1장" in llm_part
    assert "rx_001 2번째 약품 약품명: 정답 '알모란정', 결과 '알모탄정'" in llm_part
    assert "rx_004 3번째 약품 투약 일수: 정답 3, 결과 4" in llm_part
    assert "칸 정확도 (OCR 기준) 70.0% (56/80)  (목표 90% 미달)" in llm_part
    assert "약품명 60.0% (9/15)" in llm_part                   # 틀린 이름 2개 + 읽지 못한 처방전(rx_005)의 이름 4개
    # 틀린 약이 정해진 줄은 따로 알려 준다
    assert "약품 매칭 - 잘못 정함:" in llm_part
    assert "rx_002 1번째 약품 '타이레놀정 500mg': 정답 197900277, 결과 202106092 (잘못 정함)" in llm_part
    assert "rx_005 1번째 약품 '필루비서방정 145mg': 정답 없음(대상 약 아님), 결과 정하지 않음 (줄 못 읽음)" in llm_part
    record = text.split("== 일정표 정확도 기록 (그대로 붙여 넣으세요) ==\n")[1].strip().split("\n")[1]
    assert record.endswith("약품 매칭: 정답 코드 6줄 · 후보로 정답 0줄 · 잘못 정함 1줄")


def test_the_llm_record_line_carries_the_model_name(monkeypatch):
    monkeypatch.setattr(measure_accuracy.config, "OPENAI_MODEL", "gpt-5-mini-2025-08-07")
    _, text = run(["--llm"], fake_llm([answer(s) for s in SAMPLE_IDS]))
    records = text.split("== 일정표 정확도 기록 (그대로 붙여 넣으세요) ==\n")[1].strip().split("\n")
    assert len(records) == 2
    assert records[0].startswith("2026-11-10 | 규칙 경로 | ")
    assert records[1].startswith("2026-11-10 | LLM 경로 (gpt-5-mini-2025-08-07) | ")


def test_a_validation_failure_counts_as_a_failed_sample():
    answers = [answer(s) for s in SAMPLE_IDS]
    answers[1]["items"][0]["days"] = 0                     # 검증에서 거절된다
    _, text = run(["--llm"], fake_llm(answers))
    assert "구조화 실패 1장" in text.split("== LLM 경로 ==")[1]


def test_a_missing_key_skips_the_llm_path_and_exits_nonzero():
    def no_key(text, feedback=None):
        raise ConfigError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    code, text = run(["--llm"], no_key)
    assert code == 1
    assert "== LLM 경로 == 설정 문제로 재지 못했습니다: OPENAI_API_KEY" in text
    assert "== 규칙 경로 ==" in text
    records = text.split("== 일정표 정확도 기록 (그대로 붙여 넣으세요) ==\n")[1].strip().split("\n")
    assert len(records) == 1 and "규칙 경로" in records[0]       # 재지 못한 경로는 기록에 없다


def test_a_broken_dictionary_stops_before_measuring(monkeypatch):
    def broken():
        raise ConfigError("약품 사전 파일이 없습니다: x.json")

    monkeypatch.setattr(measure_accuracy, "load_dictionary", broken)
    llm = fake_llm([])
    code, text = run(["--llm"], llm)
    assert code == 1 and "설정 문제: 약품 사전 파일이 없습니다" in text and llm.calls == [] and "==" not in text


# ------------------------------------------------------------ 결과 파일


def test_results_can_be_saved_as_json_and_the_folder_is_created(tmp_path):
    path = tmp_path / "var" / "accuracy" / "acc.json"
    code, text = run(["--out", str(path)])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert code == 0 and f"결과 파일: {path}" in text
    assert data["date"] == "2026-11-10" and data["target_rate"] == 0.9
    assert set(data["paths"]) == {"rule"}
    rule = data["paths"]["rule"]
    assert rule["model"] is None and rule["meets_target"] is True
    assert (rule["ocr_correct"], rule["ocr_total"], rule["true_correct"], rule["true_total"]) == (80, 80, 64, 65)
    assert (rule["all_correct"], rule["samples"], rule["failed"], rule["flagged"]) == (5, 5, 0, 15)
    assert rule["fields"]["name"] == {"correct": 15, "total": 15} and rule["fields"]["issue_no"] == {"correct": 5, "total": 5}
    assert rule["matches"] == {
        "confirmed": 7, "candidate": 0, "not_target": 8, "not_found": 0, "wrong": 0, "missing": 0,
    }
    assert [r["sample_id"] for r in rule["results"]] == SAMPLE_IDS
    bad = [c for r in rule["results"] for c in r["true_checks"] if not c["ok"]]
    assert bad == [{"line": None, "field": "issue_no", "want": "2026년11월03일 00058호", "got": "2026년11월03일 000583", "ok": False}]
    first = rule["results"][1]["match_checks"][1]
    assert first == {"line": 1, "name": "프론크논캠술", "want": "200200099", "got": "200200099", "outcome": "confirmed"}


def test_the_json_of_the_llm_path_has_the_model_name(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_accuracy.config, "OPENAI_MODEL", "gpt-5-mini-2025-08-07")
    path = tmp_path / "acc.json"
    run(["--llm", "--out", str(path)], fake_llm([answer(s) for s in SAMPLE_IDS]))
    paths = json.loads(path.read_text(encoding="utf-8"))["paths"]
    assert set(paths) == {"rule", "llm"} and paths["llm"]["model"] == "gpt-5-mini-2025-08-07"


# ------------------------------------------------------------ 스크립트로 직접 돌릴 때


def _run_script(args, cwd):
    """스크립트를 다른 프로세스로 돌린다. 출력 인코딩을 cp949로 두고(윈도우 기본) 깨지거나 멈추지 않는지 본다."""
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "DRUG_API_KEY")}
    env.update(PYTHONIOENCODING="cp949", PYTHONUTF8="0")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True,
        encoding="cp949", errors="strict", timeout=60,
    )


def test_the_script_runs_from_any_folder_with_cp949_output_and_needs_no_key(tmp_path):
    # `python scripts/agent/measure_accuracy.py` 가 저장소 루트가 아닌 곳에서 불려도 agent 패키지를 찾아야 한다.
    # 출력은 cp949로 인코딩할 수 있는 글자만 쓴다 (윈도우에서 파일로 받아도 물음표가 섞이지 않는다)
    done = _run_script([], tmp_path)
    assert done.returncode == 0 and "Traceback" not in done.stderr
    assert "목표 90% 달성" in done.stdout and "?" not in done.stdout
    assert "OPENAI" not in done.stdout


def test_the_script_help_does_not_call_anything(tmp_path):
    done = _run_script(["--help"], tmp_path)
    assert done.returncode == 0 and "--llm" in done.stdout and "--out" in done.stdout
