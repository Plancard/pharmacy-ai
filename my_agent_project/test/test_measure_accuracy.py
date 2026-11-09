"""measure_accuracy.py(정확도 측정 스크립트)의 테스트. 진짜 OpenAI는 부르지 않는다."""
import json

import pytest

import measure_accuracy
from agent.errors import ConfigError, StructureError

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]


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
    kwargs = {"out": lines.append}
    if llm is not None:
        kwargs["llm_fn"] = llm
    code = measure_accuracy.main(argv, **kwargs)
    return code, "\n".join(lines)


def test_rule_path_only_by_default_and_the_llm_is_not_called():
    llm = fake_llm([])
    code, text = run([], llm)
    assert code == 0 and llm.calls == []
    assert "== 규칙 경로 ==" in text and "LLM 경로" not in text
    assert "처방전 5장, 약품 15줄 (그중 OCR이 약품명을 틀리게 읽은 줄 11줄)" in text
    assert "OCR 기준 칸 정확도 100.0% (95/95)" in text
    assert "약품 줄 완전 일치 100.0% (15/15) | 처방전 완전 일치 5/5 | 구조화 실패 0장" in text
    assert "진짜 값 기준 칸 정확도 98.8% (79/80)" in text
    assert "약사 확인 표시 12줄" in text
    assert "rx_004 교부번호: 정답 '2026년11월03일 00058호', 결과 '2026년11월03일 000583'" in text
    assert "OCR 기준으로 틀린 칸" not in text


def test_each_sample_has_its_own_line():
    _, text = run([])
    assert "  rx_004  칸 25/25  약품 줄 4/4  약사 확인 3줄" in text


def test_the_llm_path_is_measured_separately_with_only_the_drug_table_sent():
    llm = fake_llm([answer(s) for s in SAMPLE_IDS])
    code, text = run(["--llm"], llm)
    assert code == 0 and len(llm.calls) == 5
    assert text.count("OCR 기준 칸 정확도 100.0% (95/95)") == 2       # 두 경로 모두
    assert "== LLM 경로 ==" in text
    for sent in llm.calls:
        assert sent.startswith("교부번호: ") and "주민등록번호" not in sent and "김가상" not in sent


def test_llm_mistakes_and_failures_are_counted_and_listed():
    answers = [answer(s) for s in SAMPLE_IDS]
    answers[0]["items"][1]["name"] = "알모탄정"           # OCR 오류를 고쳐 옴: OCR 기준으로 틀림
    answers[3]["items"][2]["days"] = 4
    answers[4] = StructureError("LLM 호출에 실패했습니다 (APITimeoutError).")
    code, text = run(["--llm"], fake_llm(answers))
    llm_part = text.split("== LLM 경로 ==")[1]
    assert code == 0
    assert "rx_005  칸 0/25  약품 줄 0/4  약사 확인 0줄" in llm_part
    assert "[실패] LLM 호출에 실패했습니다 (APITimeoutError)." in llm_part
    assert "처방전 완전 일치 2/5 | 구조화 실패 1장" in llm_part
    assert "rx_001 2번째 약품 약품명: 정답 '알모란정', 결과 '알모탄정'" in llm_part
    assert "rx_004 3번째 약품 투약 일수: 정답 3, 결과 4" in llm_part
    assert "OCR 기준 칸 정확도 71.6% (68/95)" in llm_part


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


def test_a_broken_dictionary_stops_before_measuring(monkeypatch):
    def broken():
        raise ConfigError("약품 사전 파일이 없습니다: x.json")

    monkeypatch.setattr(measure_accuracy, "load_dictionary", broken)
    llm = fake_llm([])
    code, text = run(["--llm"], llm)
    assert code == 1 and "설정 문제: 약품 사전 파일이 없습니다" in text and llm.calls == [] and "==" not in text


def test_results_can_be_saved_as_json(tmp_path):
    path = tmp_path / "acc.json"
    code, text = run(["--out", str(path)])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert code == 0 and f"결과 파일: {path}" in text
    assert set(data) == {"rule"}
    rule = data["rule"]
    assert (rule["ocr_correct"], rule["ocr_total"], rule["true_correct"], rule["true_total"]) == (95, 95, 79, 80)
    assert (rule["all_correct"], rule["samples"], rule["failed"], rule["flagged"]) == (5, 5, 0, 12)
    assert [r["sample_id"] for r in rule["results"]] == SAMPLE_IDS
    bad = [c for r in rule["results"] for c in r["true_checks"] if not c["ok"]]
    assert bad == [{"line": None, "field": "issue_no", "want": "2026년11월03일 00058호", "got": "2026년11월03일 000583", "ok": False}]


def test_sample_ids_are_the_five_real_samples():
    assert measure_accuracy.sample_ids() == SAMPLE_IDS


def test_time_per_sample_with_average_and_maximum(monkeypatch):
    # 처방전마다 (시작, 끝) 시각: 걸린 시간은 1, 2, 3, 4, 10초
    ticks = iter([0, 1, 10, 12, 20, 23, 30, 34, 40, 50])
    monkeypatch.setattr(measure_accuracy.time, "perf_counter", lambda: next(ticks))
    _, text = run([])
    assert "  rx_001  칸 13/13  약품 줄 2/2  약사 확인 2줄  1.0초" in text
    assert "  rx_005  칸 25/25  약품 줄 4/4  약사 확인 4줄  10.0초" in text
    assert "시간 평균 4.0초, 최대 10.0초" in text
