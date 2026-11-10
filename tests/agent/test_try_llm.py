"""try_llm.py(실제 호출 확인 스크립트)의 테스트. 진짜 OpenAI는 부르지 않는다."""
import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.errors import ConfigError, StructureError
from agent.validate import order_from_extracted

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "try_llm.py"
_spec = importlib.util.spec_from_file_location("try_llm", SCRIPT)
try_llm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(try_llm)

SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005", "s05_no_code"]
RX_IDS = SAMPLE_IDS[:5]
DRUG_KEYS = ["name", "printed_code", "count_per_dose", "times_per_day", "days"]


def expected_of(sample_id):
    return try_llm.load_sample(sample_id)[1]


def perfect(sample_id):
    """정답과 똑같이 답하는 가짜 LLM의 응답. 샘플의 용법은 모두 'PO 식후 30분'이라 시점은 빈 목록이다."""
    expected = expected_of(sample_id)
    return {
        "issue_no": expected["issue_no"],
        "items": [{**{k: d[k] for k in DRUG_KEYS}, "times": []} for d in expected["items"]],
    }


def fake_llm(*outcomes):
    """호출마다 결과(dict 또는 예외)를 차례로 돌려주고, 받은 인자를 기록한다."""
    calls = []

    def llm(text, feedback=None):
        calls.append((text, feedback))
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    llm.calls = calls
    return llm


def run(argv, llm):
    lines = []
    code = try_llm.main(argv, llm_fn=llm, out=lines.append)
    return code, "\n".join(lines)


def structure_error_from(cause):
    try:
        try:
            raise cause
        except Exception as e:
            raise StructureError(f"LLM 호출에 실패했습니다 ({type(e).__name__}).") from e
    except StructureError as error:
        return error


class StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


# ------------------------------------------------------------ 정상


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_matching_answer_for_every_sample_exits_zero(sample_id):
    code, text = run([sample_id], fake_llm(perfect(sample_id)))
    assert code == 0 and "정답과 일치" in text


def test_missing_times_are_filled_with_a_warning_and_still_match():
    # 샘플은 용법에 복용 시점이 없다. 빈 목록을 공통 검증이 채우고 경고를 남긴다 (약품마다 한 번).
    code, text = run(["rx_001"], fake_llm(perfect("rx_001")))
    assert code == 0 and text.count("경고:") == 2 and "복용 시점이 처방전에 없어" in text


def test_only_the_drug_table_is_sent_not_the_patient_information():
    llm = fake_llm(perfect("rx_001"))
    run(["rx_001"], llm)
    sent, _ = llm.calls[0]
    assert sent.split("\n")[:2] == ["교부번호: 2026년11월03일 00012호", "약품 표:"]
    for private in ("김가상", "000000101", "900101", "주민등록번호", "02-000-0000", "이테스트"):
        assert private not in sent


def test_names_that_differ_only_by_spaces_or_slashes_count_as_equal():
    answer = perfect("rx_001")
    answer["items"][0]["name"] = "타이레놀정/5oOmg"
    code, text = run(["rx_001"], fake_llm(answer))
    assert code == 0 and "정답과 일치" in text


def test_default_sample_is_the_first_one_and_one_call_is_made():
    llm = fake_llm(perfect("rx_001"))
    code, _ = run([], llm)
    assert code == 0 and len(llm.calls) == 1


def test_text_sent_to_the_llm_is_shown_and_has_no_feedback():
    llm = fake_llm(perfect("rx_005"))
    _, text = run(["rx_005"], llm)
    sent, feedback = llm.calls[0]
    assert feedback is None
    assert all(line in text for line in sent.split("\n"))


def test_all_runs_every_sample_and_summarises():
    answers = [perfect(s) for s in SAMPLE_IDS]
    llm = fake_llm(*answers)
    code, text = run(["--all"], llm)
    assert code == 0 and len(llm.calls) == len(SAMPLE_IDS)
    assert f"전체 {len(SAMPLE_IDS)}장 중 {len(SAMPLE_IDS)}장이 정답과 일치" in text


def test_all_exits_nonzero_when_only_one_sample_differs():
    answers = [perfect(s) for s in SAMPLE_IDS]
    answers[2]["items"][0]["days"] += 1
    code, text = run(["--all"], fake_llm(*answers))
    assert code == 1 and f"전체 {len(SAMPLE_IDS)}장 중 {len(SAMPLE_IDS) - 1}장이 정답과 일치" in text


def test_unknown_sample_is_refused_by_the_argument_parser():
    with pytest.raises(SystemExit):
        run(["rx_999"], fake_llm(perfect("rx_001")))


def test_header_line_shows_model_and_timeout_but_not_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-VERYSECRET123")
    _, text = run(["rx_001"], fake_llm(perfect("rx_001")))
    first = text.split("\n")[0]
    assert try_llm.config.OPENAI_MODEL in first and "설정됨" in first
    assert "VERYSECRET" not in text


def test_header_says_when_the_key_is_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    _, text = run(["rx_001"], fake_llm(perfect("rx_001")))
    assert "OPENAI_API_KEY: 없음" in text.split("\n")[0]


# ------------------------------------------------------------ 정답과 다른 경우


def test_each_difference_is_reported_with_both_values():
    answer = perfect("rx_002")
    answer["items"][0]["days"] = 7
    answer["items"][1]["name"] = "엉뚱한정"
    answer["issue_no"] = "000"
    code, text = run(["rx_002"], fake_llm(answer))
    expected = expected_of("rx_002")
    assert code == 1
    assert f"약품 1 days: 정답 {expected['items'][0]['days']!r}, LLM 7" in text
    assert f"약품 2 name: 정답 {expected['items'][1]['name']!r}, LLM '엉뚱한정'" in text
    assert "교부번호" in text and "3곳 다름" in text


def test_a_missing_drug_is_reported_as_a_count_difference():
    answer = perfect("rx_001")
    answer["items"].pop()
    code, text = run(["rx_001"], fake_llm(answer))
    assert code == 1 and "약품 수: 정답 2, LLM 1" in text


def test_times_are_not_compared():
    # 정답 파일에는 시점이 없다 (처방전에 적혀 있지 않다). 시점은 공통 검증이 채운다.
    answer = perfect("rx_004")
    answer["items"][1]["times"] = ["morning", "noon", "evening"]
    code, _ = run(["rx_004"], fake_llm(answer))
    assert code == 0


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_compare_returns_nothing_when_identical(sample_id):
    order = order_from_extracted("P1", perfect(sample_id), source="llm")
    assert try_llm.compare(order, expected_of(sample_id)) == []


# ------------------------------------------------------------ 호출·검증 실패


def test_response_rejected_by_validation_shows_the_raw_answer():
    answer = copy.deepcopy(perfect("rx_001"))
    answer["items"][0]["days"] = 0
    code, text = run(["rx_001"], fake_llm(answer))
    assert code == 1 and "검증에서 거절" in text and "투약 일수" in text
    assert '"days": 0' in text   # 원본을 보여 줘야 무엇이 잘못 왔는지 알 수 있다


def test_config_problem_is_reported_plainly():
    code, text = run(["rx_001"], fake_llm(ConfigError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")))
    assert code == 1 and "설정 문제" in text and "OPENAI_API_KEY" in text


def test_call_failure_shows_the_real_cause_and_a_hint_for_the_model_name():
    cause = StatusError("Error code: 404 - The model `gpt-5-mini-2025-08-07` does not exist", 404)
    code, text = run(["rx_001"], fake_llm(structure_error_from(cause)))
    assert code == 1
    assert "StatusError" in text and "does not exist" in text
    assert "OPENAI_MODEL" in text


@pytest.mark.parametrize(
    "status, said",
    [(400, "지원하지 않는 옵션"), (401, "OPENAI_API_KEY"), (403, "권한"), (429, "한도")],
)
def test_hint_matches_the_http_status(status, said):
    hint = try_llm.hint_for(StatusError("x", status))
    assert said in hint


def test_timeout_hint_names_the_setting_and_the_current_limit(monkeypatch):
    class APITimeoutError(Exception):
        pass

    monkeypatch.setattr(try_llm.config, "LLM_TIMEOUT_SECONDS", 45.0)
    hint = try_llm.hint_for(APITimeoutError("Request timed out."))
    assert "LLM_TIMEOUT_SECONDS" in hint and "45초 안에" in hint


def test_connection_hint():
    class APIConnectionError(Exception):
        pass

    assert "연결" in try_llm.hint_for(APIConnectionError("x"))


def test_unknown_error_has_no_hint():
    assert try_llm.hint_for(ValueError("x")) is None
    assert try_llm.hint_for(StatusError("x", 500)) is None


def test_key_fragments_in_error_text_are_masked():
    # OpenAI의 401 메시지는 키의 가운데를 별표로 가리고 앞뒤 조각을 보여 준다. 조각 전체를 가려야 한다.
    cause = StatusError("Incorrect API key provided: sk-proj-abcd****WXYZ. You can find it", 401)
    code, text = run(["rx_001"], fake_llm(structure_error_from(cause)))
    assert code == 1 and "Incorrect API key provided" in text
    assert "sk-***. You can find it" in text
    for leaked in ("sk-proj", "abcd", "WXYZ"):
        assert leaked not in text


def test_whole_key_is_masked_even_with_underscores_and_hyphens():
    # 프로젝트 키는 sk-proj-… 처럼 하이픈·밑줄이 섞여 있다
    result = try_llm.redact("key=sk-proj-Ab_cD-eF_gH-12 end")
    assert result == "key=sk-*** end"


def test_redacted_text_never_exceeds_the_limit_even_if_a_key_sits_on_the_boundary():
    text = "a" * (try_llm.MAX_DETAIL - 5) + "sk-proj-SECRETSECRET"
    result = try_llm.redact(text)
    assert len(result) <= try_llm.MAX_DETAIL and "SECRET" not in result


def test_long_error_text_is_truncated():
    cause = StatusError("가" * 5000, 500)
    _, text = run(["rx_001"], fake_llm(structure_error_from(cause)))
    assert "가" * try_llm.MAX_DETAIL in text and "가" * (try_llm.MAX_DETAIL + 1) not in text


def test_failure_without_a_cause_still_reports():
    code, text = run(["rx_001"], fake_llm(StructureError("LLM이 요청을 거절했습니다.")))
    assert code == 1 and "거절" in text and "원인" not in text


def test_a_failed_sample_does_not_stop_the_rest_with_all():
    outcomes = [perfect(s) for s in SAMPLE_IDS]
    outcomes[1] = StructureError("실패")
    llm = fake_llm(*outcomes)
    code, text = run(["--all"], llm)
    assert code == 1 and len(llm.calls) == len(SAMPLE_IDS)
    assert f"전체 {len(SAMPLE_IDS)}장 중 {len(SAMPLE_IDS) - 1}장이 정답과 일치" in text


# ------------------------------------------------------------ 약품 코드가 인쇄되지 않은 처방전 (s05)


def test_the_code_less_sample_expects_no_printed_code():
    expected = expected_of("s05_no_code")
    assert [d["printed_code"] for d in expected["items"]] == [None, None]
    assert expected["issue_no"] == "2026년11월09일 00065호"


def test_a_printed_code_the_llm_made_up_is_reported():
    answer = perfect("s05_no_code")
    answer["items"][0]["printed_code"] = "900000011"
    code, text = run(["s05_no_code"], fake_llm(answer))
    assert code == 1 and "약품 1 printed_code: 정답 None, LLM '900000011'" in text


def test_only_the_table_without_codes_is_sent_for_the_code_less_sample():
    llm = fake_llm(perfect("s05_no_code"))
    run(["s05_no_code"], llm)
    sent, _ = llm.calls[0]
    assert sent.split("\n") == [
        "교부번호: 2026년11월09일 00065호",
        "약품 표:",
        "타이레놀정/ 500mg 1 TAB 3회 3일 PO 식후 30분",
        "뮤코다인캡슐/ 200mg 1 CAP 3회 3일 PO 식후 30분",
    ]


# ------------------------------------------------------------ 스크립트로 직접 실행


def _run_script(args, cwd, key=None):
    """스크립트를 다른 프로세스로 돌린다. 윈도우의 기본 출력 인코딩은 cp949라서, 자식 프로세스가 UTF-8로 쓰도록 맞춘다."""
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if key is not None:
        env["OPENAI_API_KEY"] = key
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )


def test_the_script_runs_from_any_folder_and_does_not_need_a_key(tmp_path):
    # `python scripts/agent/try_llm.py` 가 저장소 루트가 아닌 곳에서 불려도 agent 패키지를 찾아야 한다 (--help는 OpenAI를 부르지 않는다)
    done = _run_script(["--help"], tmp_path)
    assert done.returncode == 0 and "--all" in done.stdout


def test_the_script_without_a_key_stops_with_a_plain_message(tmp_path):
    # 키를 빈 값으로 두면 .env에 진짜 키가 있어도 덮어쓰지 않는다 (OpenAI를 부르지 않는다)
    done = _run_script([], tmp_path, key="")
    assert done.returncode == 1 and "OPENAI_API_KEY: 없음" in done.stdout and "설정 문제" in done.stdout
    assert "Traceback" not in done.stderr
