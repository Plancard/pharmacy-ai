"""구조화 함수 테스트. 규칙·LLM 함수를 가짜로 바꿔 끼워서 흐름을 확인한다."""
import functools
import json
from types import SimpleNamespace

import pytest

from agent.config import RESOURCES_DIR
from agent.errors import ConfigError, StructureError
from agent.llm import LlmPrescription, extract_llm
from agent.pipeline import structure_prescription
from agent.structure import ocr_to_llm_text

DATA = RESOURCES_DIR / "samples"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]
A1_IDS = ["s01_combo_a", "s02_combo_b", "s03_no_risk", "s04_flagged", "s05_no_code"]


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
        count_per_dose=1,
        times_per_day=3,
        days=3,
        times=[],
    )
    drug.update(drug_overrides)
    return {"issue_no": None, "items": [drug]}


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
    order = structure_prescription("P0001", load("cv_a1", sample_id), llm_fn=never_called)
    expected = load("cv_expected", sample_id)   # 하루 횟수(times_per_day)는 복용 시점의 수(len(times))와 같다
    assert order.source == "rule" and order.issue_no == expected["issue_no"]
    assert [(d.name, d.printed_code, d.count_per_dose, len(d.times), d.days) for d in order.items] == [
        (want["name"], want["printed_code"], want["count_per_dose"], want["times_per_day"], want["days"])
        for want in expected["items"]
    ]
    assert not any("LLM" in w for w in order.warnings)


def test_rule_function_receives_the_ocr():
    rule = FakeFn(extracted())
    structure_prescription("P0001", NO_HEADER_OCR, rule_fn=rule, llm_fn=never_called)
    assert rule.calls == [(NO_HEADER_OCR,)]


# ------------------------------------------------------------ 샘플(a1_ocr)과 정답(a5_order)


def _read(order):
    return [(d.count_per_dose, d.times, d.days) for d in order.items]


def _answer(sample_id):
    return [(d["count_per_dose"], d["times"], d["days"]) for d in load("a5_order", sample_id)["items"]]


@pytest.mark.parametrize("sample_id", A1_IDS[:4])
def test_samples_with_printed_codes_are_read_by_rules_and_match_the_answer(sample_id):
    order = structure_prescription("P0001", load("a1_ocr", sample_id), llm_fn=never_called)   # LLM은 불리지 않는다
    assert order.source == "rule"
    assert _read(order) == _answer(sample_id)


def test_the_code_less_sample_goes_to_the_llm_and_matches_the_answer():
    ocr = load("a1_ocr", "s05_no_code")
    llm = FakeFn(
        {
            "issue_no": "2026년11월09일 00065호",
            "items": [
                dict(name="타이레놀정 500mg", printed_code=None, count_per_dose=1, times_per_day=3, days=3, times=[]),
                dict(name="뮤코다인캡슐 200mg", printed_code=None, count_per_dose=1, times_per_day=3, days=3, times=[]),
            ],
        }
    )
    order = structure_prescription("P0005", ocr, llm_fn=llm)
    assert order.source == "llm" and order.issue_no == "2026년11월09일 00065호"
    assert _read(order) == _answer("s05_no_code")                                     # 빈 복용 시점은 횟수로 채워진다
    assert [d.item_code for d in order.items] == [None, None]                          # 약품 매칭은 아직 안 한 단계
    assert any("규칙 기반으로 읽지 못해" in w for w in order.warnings)
    assert any("복용 시점이 처방전에 없어" in w for w in order.warnings)

    (text, feedback), = llm.calls
    assert feedback is None
    assert text.split("\n") == [
        "교부번호: 2026년11월09일 00065호",
        "약품 표:",
        "타이레놀정/ 500mg 1 TAB 3회 3일 PO 식후 30분",
        "뮤코다인캡슐/ 200mg 1 CAP 3회 3일 PO 식후 30분",
    ]
    assert "OCR 테스트용" not in text and "가상의료재단" not in text             # 처방전 위·아래의 글은 보내지 않는다


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
    ocr = load("cv_a1", sample_id)
    for line in ocr["lines"]:
        if "".join(line["text"].split()) in ("횟수", "일수", "구분코드", "투약량", "투악량", "처방의약품의명칭"):
            line["text"] = "?"
    return ocr


# 처방전 위쪽의 환자·의료인 정보. LLM에 보내는 텍스트에 들어가면 안 된다.
PRIVATE_TEXTS = ["000000101", "김가상", "900101", "02-000-0000", "02-000-0001", "이테스트", "가상의료재단", "hosp", "R51"]


@pytest.mark.parametrize("ocr", [load("cv_a1", "rx_001"), without_header("rx_001")], ids=["with_header", "without_header"])
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
    order = run(llm_fn=FakeFn(extracted(times=["morning", "noon", "evening"])))
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
    order = run(llm_fn=FakeFn(extracted(times=[])))
    assert order.items[0].times == ["morning", "noon", "evening"]   # 빈 시점은 횟수로 채워진다
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
    assert error.value.code == "E_LLM"                            # A.7: AI 응답 오류


@pytest.mark.parametrize("retries, calls", [(0, 1), (1, 2), (2, 3), (-5, 1)])
def test_number_of_llm_calls_is_bounded_by_retries(retries, calls):
    llm = FakeFn(StructureError("실패"))
    with pytest.raises(StructureError):
        run(llm_fn=llm, llm_retries=retries)
    assert len(llm.calls) == calls


def test_default_number_of_retries_is_the_configured_one():
    from agent import config

    llm = FakeFn(StructureError("실패"))
    with pytest.raises(StructureError):
        run(llm_fn=llm)
    assert len(llm.calls) == 1 + config.LLM_MAX_RETRIES


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
        with pytest.raises(StructureError, match="처방 ID") as error:
            structure_prescription(bad, NO_HEADER_OCR, rule_fn=never_called, llm_fn=never_called)
        assert error.value.code is None            # 처방전이 아니라 부르는 쪽의 잘못이라 A.7 코드가 없다


@pytest.mark.parametrize("bad", [None, {}, {"lines": []}, {"lines": [{"text": "a"}]}])
def test_malformed_ocr_fails_before_anything_runs(bad):
    with pytest.raises(StructureError) as error:
        structure_prescription("P0001", bad, rule_fn=never_called, llm_fn=never_called)
    assert error.value.code == "E_OCR_EMPTY"


def test_no_drug_row_to_send_is_an_ocr_empty_error_without_calling_the_llm():
    # 글자는 있지만 약품 코드 줄이 하나도 없다: 규칙 경로도 못 읽고 LLM에 보낼 것도 없다
    ocr = {"lines": [{"text": "처 방 전", "bbox": [0, 0, 100, 40], "conf": 0.9}]}
    with pytest.raises(StructureError, match="약품 표 줄을 찾지 못해") as error:
        structure_prescription("P0001", ocr, llm_fn=never_called)
    assert error.value.code == "E_OCR_EMPTY"


def test_the_last_error_is_e_llm_even_when_the_llm_calls_themselves_failed():
    with pytest.raises(StructureError) as error:
        run(llm_fn=FakeFn(StructureError("LLM 호출에 실패했습니다 (TimeoutError).")))
    assert error.value.code == "E_LLM"


# ------------------------------------------------------------ 진짜 extract_llm과 이어 붙이기


def test_works_with_the_real_extract_llm_and_a_fake_client():
    parsed = LlmPrescription(
        issue_no=None,
        items=[
            {
                "name": "알마겔정",
                "printed_code": "699000031",
                "count_per_dose": 1,
                "times_per_day": 2,
                "days": 3,
                "times": ["morning", "bedtime"],
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
    assert order.source == "llm" and order.items[0].times == ["morning", "bedtime"]
    assert calls[0]["messages"][1]["content"] == NO_HEADER_TEXT


def test_the_error_of_the_real_extract_llm_ends_as_e_llm_after_the_allowed_attempts():
    def parse(**kwargs):
        raise TimeoutError("처방전 내용 비밀약품")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    with pytest.raises(StructureError) as error:
        structure_prescription("P0001", NO_HEADER_OCR, llm_fn=functools.partial(extract_llm, client=client))
    assert error.value.code == "E_LLM" and "TimeoutError" in str(error.value)
    assert "비밀약품" not in str(error.value)
