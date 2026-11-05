"""구조화 함수 테스트. 규칙·LLM 함수를 가짜로 바꿔 끼워서 흐름을 확인한다."""
import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.errors import ConfigError, StructureError
from agent.llm_extract import LlmPrescription, extract_llm
from agent.pipeline import structure_prescription
from agent.structure import ocr_to_llm_text

DATA = Path(__file__).resolve().parent.parent / "data" / "agent"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]


def load(kind, sample_id):
    return json.loads((DATA / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


# 머리글이 없어서 규칙 경로가 읽을 수 없는 OCR (형식은 올바르다). 약품 코드 줄은 있어서 LLM에는 보낼 수 있다.
NO_HEADER_OCR = {
    "image_id": "t",
    "lines": [
        {"text": "900000031 알마겔정", "bbox": [10, 10, 220, 40], "conf": 0.9},
        {"text": "3회", "bbox": [300, 10, 340, 40], "conf": 0.9},
    ],
}
NO_HEADER_TEXT = ocr_to_llm_text(NO_HEADER_OCR)


def extracted(**drug_overrides):
    drug = dict(
        name="알마겔정",
        printed_code="699000031",
        dose_per_time=1,
        times_per_day=3,
        days=3,
        time_slots=[],
    )
    drug.update(drug_overrides)
    return {"issue_no": None, "drugs": [drug]}


class FakeFn:
    """호출을 기록하고, 미리 정한 결과(dict 또는 예외)를 차례로 돌려준다. 결과가 모자라면 마지막 것을 반복한다."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def never_called(*args):
    raise AssertionError("부르면 안 되는 함수가 불렸습니다")


def run(ocr=NO_HEADER_OCR, **kwargs):
    kwargs.setdefault("rule_fn", FakeFn(StructureError("규칙 실패")))
    return structure_prescription("P0001", ocr, **kwargs)


# ------------------------------------------------------------ 규칙 경로가 통과하는 경우


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_rule_path_handles_the_real_format_prescriptions_without_calling_the_llm(sample_id):
    order = structure_prescription("P0001", load("cv_samples", sample_id), llm_fn=never_called)
    expected = load("cv_expected", sample_id)
    assert order.source == "rule" and order.issue_no == expected["issue_no"]
    assert [(d.name, d.printed_code, d.dose_per_time, d.times_per_day, d.days) for d in order.drugs] == [
        (want["name"], want["printed_code"], want["dose_per_time"], want["times_per_day"], want["days"])
        for want in expected["drugs"]
    ]
    assert not any("LLM" in w for w in order.warnings)


def test_rule_function_receives_the_ocr():
    rule = FakeFn(extracted())
    structure_prescription("P0001", NO_HEADER_OCR, rule_fn=rule, llm_fn=never_called)
    assert rule.calls == [(NO_HEADER_OCR,)]


# ------------------------------------------------------------ 규칙 경로 실패 -> LLM


def test_llm_is_used_when_the_rule_path_cannot_read_the_prescription():
    llm = FakeFn(extracted())
    order = structure_prescription("P0001", NO_HEADER_OCR, llm_fn=llm)   # 진짜 규칙 경로: 머리글이 없어 실패
    assert order.source == "llm"
    assert llm.calls == [(NO_HEADER_TEXT, None)]            # LLM용 텍스트, 거절 사유 없음
    assert NO_HEADER_TEXT == "교부번호: (찾지 못함)\n약품 표:\n900000031 알마겔정 3회"
    assert "머리글" in order.warnings[0] and "원문과 꼭 대조" in order.warnings[0]


def without_header(sample_id):
    """실제 샘플에서 머리글 낱말을 지워 규칙 경로가 실패하게 만든다."""
    ocr = load("cv_samples", sample_id)
    for line in ocr["lines"]:
        if "".join(line["text"].split()) in ("횟수", "일수", "구분코드", "투약량", "투악량", "처방의약품의명칭"):
            line["text"] = "?"
    return ocr


# 처방전 위쪽의 환자·의료인 정보. LLM에 보내는 텍스트에 들어가면 안 된다.
PRIVATE_TEXTS = ["000000101", "김가상", "900101", "02-000-0000", "02-000-0001", "이테스트", "가상의료재단", "hosp", "R51"]


@pytest.mark.parametrize("ocr", [load("cv_samples", "rx_001"), without_header("rx_001")], ids=["with_header", "without_header"])
def test_patient_and_doctor_information_is_never_sent_to_the_llm(ocr):
    llm = FakeFn(extracted())
    structure_prescription("P0001", ocr, rule_fn=FakeFn(StructureError("규칙 실패")), llm_fn=llm)
    sent = llm.calls[0][0]
    assert "타이레놀정" in sent and "알모란정" in sent
    for private in PRIVATE_TEXTS:
        assert private not in sent


def test_the_real_rule_path_falls_back_to_the_llm_with_only_the_table():
    llm = FakeFn(extracted())
    order = structure_prescription("P0001", without_header("rx_005"), llm_fn=llm)
    assert order.source == "llm"
    assert llm.calls[0][0].split("\n")[:3] == ["교부번호: 2026년11월03일 00063호", "약품 표:", "(1)671805860 필루비서방정 145mg 1 TAB 2회 3일 PO 식후 30분"]
    assert len(llm.calls[0][0].split("\n")) == 6


def test_without_any_table_row_the_llm_is_not_called():
    ocr = {"image_id": "t", "lines": [{"text": "알마겔정 3회", "bbox": [10, 10, 120, 40], "conf": 0.9}]}
    with pytest.raises(StructureError) as error:
        structure_prescription("P0001", ocr, llm_fn=never_called)
    message = str(error.value)
    assert "직접 입력" in message and "[규칙]" in message and "LLM에 보내지 않았습니다" in message
    assert "알마겔정" not in message


def test_first_try_success_has_only_the_fallback_note_and_the_validator_warnings():
    order = run(llm_fn=FakeFn(extracted(time_slots=["morning", "noon", "evening"])))
    assert len(order.warnings) == 1 and "규칙 기반으로 읽지 못해" in order.warnings[0]
    assert not any("번째 시도" in w for w in order.warnings)


def test_llm_is_also_used_when_the_rule_result_is_rejected_by_validation():
    llm = FakeFn(extracted())
    order = run(rule_fn=FakeFn(extracted(days=0)), llm_fn=llm)
    assert order.source == "llm" and len(llm.calls) == 1
    assert "투약 일수" in order.warnings[0]


def test_unexpected_rule_error_falls_back_without_leaking_its_message():
    llm = FakeFn(extracted())
    order = run(rule_fn=FakeFn(KeyError("비밀약품")), llm_fn=llm)
    assert order.source == "llm"
    assert "KeyError" in order.warnings[0] and "비밀약품" not in order.warnings[0]


def test_llm_result_goes_through_the_common_validator():
    order = run(llm_fn=FakeFn(extracted(time_slots=[])))
    assert order.drugs[0].time_slots == ["morning", "noon", "evening"]   # 빈 시점은 횟수로 채워진다
    assert any("복용 시점이 처방전에 없어" in w for w in order.warnings)


# ------------------------------------------------------------ 재시도


def test_rejected_llm_result_is_retried_with_the_rejection_reason():
    llm = FakeFn(extracted(days=0), extracted())
    order = run(llm_fn=llm)
    assert order.source == "llm" and len(llm.calls) == 2
    assert llm.calls[0][1] is None
    assert llm.calls[1][1] == "1번째 약품의 투약 일수가 1 미만입니다."
    assert any("2번째 시도" in w for w in order.warnings)


def test_a_failed_call_is_retried_without_feedback():
    llm = FakeFn(StructureError("LLM 호출에 실패했습니다 (TimeoutError)."), extracted())
    order = run(llm_fn=llm)
    assert len(llm.calls) == 2 and llm.calls[1][1] is None
    assert any("2번째 시도" in w for w in order.warnings)


def test_feedback_survives_a_failed_call_in_between():
    llm = FakeFn(extracted(days=0), StructureError("LLM 호출에 실패했습니다 (TimeoutError)."), extracted())
    run(llm_fn=llm, llm_retries=2)
    assert llm.calls[2][1] == "1번째 약품의 투약 일수가 1 미만입니다."


def test_gives_up_after_the_allowed_number_of_attempts_and_lists_every_reason():
    llm = FakeFn(extracted(days=0))
    with pytest.raises(StructureError) as error:
        run(llm_fn=llm)
    message = str(error.value)
    assert len(llm.calls) == 2                                    # 기본값: 처음 1번 + 재시도 1번
    assert "직접 입력" in message
    assert "[규칙] 규칙 실패" in message and "[LLM 1회]" in message and "[LLM 2회]" in message


@pytest.mark.parametrize("retries, calls", [(0, 1), (1, 2), (2, 3), (-5, 1)])
def test_number_of_llm_calls_is_bounded_by_retries(retries, calls):
    llm = FakeFn(StructureError("실패"))
    with pytest.raises(StructureError):
        run(llm_fn=llm, llm_retries=retries)
    assert len(llm.calls) == calls


# ------------------------------------------------------------ 재시도하면 안 되는 경우


def test_config_error_from_the_llm_is_not_retried_or_swallowed():
    llm = FakeFn(ConfigError("OPENAI_API_KEY가 설정되지 않았습니다."))
    with pytest.raises(ConfigError):
        run(llm_fn=llm)
    assert len(llm.calls) == 1


def test_config_error_from_the_rule_path_is_not_swallowed():
    with pytest.raises(ConfigError):
        run(rule_fn=FakeFn(ConfigError("설정 문제")), llm_fn=never_called)


def test_missing_prescription_id_fails_before_anything_runs():
    for bad in ("", "   ", None):
        with pytest.raises(StructureError, match="처방 ID"):
            structure_prescription(bad, NO_HEADER_OCR, rule_fn=never_called, llm_fn=never_called)


@pytest.mark.parametrize("bad", [None, {}, {"lines": []}, {"lines": [{"text": "a"}]}])
def test_malformed_ocr_fails_before_anything_runs(bad):
    with pytest.raises(StructureError):
        structure_prescription("P0001", bad, rule_fn=never_called, llm_fn=never_called)


# ------------------------------------------------------------ 진짜 extract_llm과 이어 붙이기


def test_works_with_the_real_extract_llm_and_a_fake_client():
    parsed = LlmPrescription(
        issue_no=None,
        drugs=[
            {
                "name": "알마겔정",
                "printed_code": "699000031",
                "dose_per_time": 1,
                "times_per_day": 2,
                "days": 3,
                "time_slots": ["morning", "bedtime"],
            }
        ],
    )
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(parsed=parsed, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    order = structure_prescription(
        "P0001", NO_HEADER_OCR, llm_fn=functools.partial(extract_llm, client=client)
    )
    assert order.source == "llm" and order.drugs[0].time_slots == ["morning", "bedtime"]
    assert calls[0]["messages"][1]["content"] == NO_HEADER_TEXT
