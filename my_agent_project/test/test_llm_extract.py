"""LLM 경로 테스트. 실제 OpenAI는 부르지 않고 가짜 클라이언트로 확인한다."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import config
from agent.errors import ConfigError, StructureError
from agent.llm_extract import SYSTEM_PROMPT, LlmPrescription, extract_llm
from agent.structure import ISSUE_NOT_FOUND, extract_rule, ocr_to_text
from agent.validate import order_from_extracted

DATA = Path(__file__).resolve().parent.parent / "data" / "agent"

PARSED = LlmPrescription(
    issue_no="2026110400001",
    items=[
        {
            "name": "알마겔정",
            "printed_code": "699000031",
            "count_per_dose": 1,
            "times_per_day": 3,
            "days": 3,
            "times": [],
        }
    ],
)


class FakeClient:
    """client.chat.completions.parse(...)만 흉내 낸다."""

    def __init__(self, parsed=PARSED, refusal=None, error=None):
        self.calls = []
        self._parsed, self._refusal, self._error = parsed, refusal, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(parsed=self._parsed, refusal=self._refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def load_ocr(sample_id):
    return json.loads((DATA / "cv_samples" / f"{sample_id}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------ ocr_to_text (사람이 훑어보는 전체 텍스트)


def test_ocr_to_text_writes_one_line_per_row():
    rows = ocr_to_text(load_ocr("rx_001")).split("\n")
    assert rows[0] == "[별지 제9호 서식]"
    assert "(1)900000011 타이레놀정/ 5oOmg 1 TAB 3회 3일 PO 식후 30분" in rows
    assert "900000031 알모란정 1 TAB 3회 3일 PO 식후 30분" in rows
    assert any("김가상(000000101)" in r for r in rows)     # 전체 텍스트에는 환자 정보도 있다. 그래서 LLM에는 쓰지 않는다


def test_ocr_to_text_rejects_malformed_ocr():
    with pytest.raises(StructureError):
        ocr_to_text({"lines": []})


# ------------------------------------------------------------ 프롬프트


def test_prompt_describes_the_input_that_is_actually_sent():
    assert "교부번호: " in SYSTEM_PROMPT and "약품 표:" in SYSTEM_PROMPT
    assert ISSUE_NOT_FOUND in SYSTEM_PROMPT


def test_prompt_example_matches_what_the_rule_path_reads():
    # 프롬프트의 예시가 규칙 경로의 결과와 같아야 두 경로가 같은 이름을 낸다
    assert '"(1)900000011 타이레놀정/ 5oOmg 1 TAB 3회 3일 PO 식후 30분" -> name "타이레놀정 5oOmg"' in SYSTEM_PROMPT
    assert extract_rule(load_ocr("rx_001"))["items"][0]["name"] == "타이레놀정 5oOmg"


def test_prompt_forbids_correcting_and_guessing():
    assert "고치지 말고" in SYSTEM_PROMPT and "짐작해 채우지 마세요" in SYSTEM_PROMPT


# ------------------------------------------------------------ extract_llm: 정상


def test_returns_a_plain_dict_in_the_shared_shape():
    result = extract_llm("텍스트", client=FakeClient())
    assert set(result) == {"issue_no", "items"}
    assert set(result["items"][0]) == {
        "name", "printed_code", "count_per_dose", "times_per_day", "days", "times",
    }
    assert isinstance(result, dict) and isinstance(result["items"][0], dict)


def test_result_goes_through_the_common_validator():
    order = order_from_extracted("P0001", extract_llm("텍스트", client=FakeClient()), source="llm")
    assert order.source == "llm" and order.items[0].times == ["morning", "noon", "evening"]


def test_call_uses_the_configured_model_the_schema_and_the_prompt():
    client = FakeClient()
    extract_llm("약품명 코드\n알마겔정 699000031", client=client)
    (call,) = client.calls
    assert call["model"] == config.OPENAI_MODEL
    assert call["response_format"] is LlmPrescription
    assert call["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "약품명 코드\n알마겔정 699000031"},
    ]


def test_model_can_be_overridden():
    client = FakeClient()
    extract_llm("텍스트", client=client, model="gpt-4o")
    assert client.calls[0]["model"] == "gpt-4o"


def test_no_temperature_is_sent():
    # gpt-5 계열은 temperature를 기본값(1)만 받는다. 0 등을 보내면 400 오류가 난다.
    client = FakeClient()
    extract_llm("텍스트", client=client)
    assert "temperature" not in client.calls[0]


def test_feedback_is_added_as_a_last_message():
    client = FakeClient()
    extract_llm("텍스트", "1번째 약품의 투약 일수가 1 미만입니다.", client=client)
    messages = client.calls[0]["messages"]
    assert len(messages) == 3 and messages[2]["role"] == "user"
    assert "1번째 약품의 투약 일수가 1 미만입니다." in messages[2]["content"]


def test_no_feedback_message_when_there_is_no_feedback():
    client = FakeClient()
    extract_llm("텍스트", None, client=client)
    assert len(client.calls[0]["messages"]) == 2


# ------------------------------------------------------------ extract_llm: 실패


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_fails_without_calling_the_api(text):
    client = FakeClient()
    with pytest.raises(StructureError) as error:
        extract_llm(text, client=client)
    assert client.calls == [] and error.value.code == "E_LLM"


def test_api_error_becomes_structure_error_without_leaking_its_message():
    client = FakeClient(error=TimeoutError("처방전 내용 비밀약품 타이레놀"))
    with pytest.raises(StructureError) as error:
        extract_llm("텍스트", client=client)
    assert "TimeoutError" in str(error.value)
    assert "비밀약품" not in str(error.value)
    assert error.value.code == "E_LLM"


def test_missing_parsed_result_fails():
    with pytest.raises(StructureError, match="돌려주지 않았습니다") as error:
        extract_llm("텍스트", client=FakeClient(parsed=None))
    assert error.value.code == "E_LLM"


def test_refusal_is_reported_as_a_refusal():
    with pytest.raises(StructureError, match="거절") as error:
        extract_llm("텍스트", client=FakeClient(parsed=None, refusal="죄송합니다"))
    assert error.value.code == "E_LLM"


def test_missing_api_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        extract_llm("텍스트")   # client를 안 주면 기본 클라이언트를 만들려다 키가 없어서 실패


def test_default_client_uses_the_key_the_timeout_and_no_sdk_retries(monkeypatch):
    # SDK의 자동 재시도(기본 2번)가 켜져 있으면 최악의 경우 시간이 몇 배로 늘어난다.
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "테스트용-가짜-키")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    from agent.llm_extract import _default_client

    _default_client()
    assert captured == {
        "api_key": "테스트용-가짜-키",
        "timeout": config.LLM_TIMEOUT_SECONDS,
        "max_retries": 0,
    }


def test_missing_openai_package_is_a_config_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "테스트용-가짜-키")
    monkeypatch.setitem(sys.modules, "openai", None)   # import openai 가 ImportError가 되게 한다
    with pytest.raises(ConfigError, match="openai"):
        extract_llm("텍스트")


# ------------------------------------------------------------ 스키마 모양


def test_output_schema_has_no_optional_fields():
    # OpenAI 구조화 출력(strict)은 모든 칸이 필수여야 한다. 기본값이 생기면 호출이 거절된다.
    schema = LlmPrescription.model_json_schema()
    objects = [schema, *schema.get("$defs", {}).values()]
    for obj in objects:
        assert set(obj["required"]) == set(obj["properties"]), obj.get("title")


def test_times_are_limited_to_the_four_tray_rows():
    schema = LlmPrescription.model_json_schema()
    items = schema["$defs"]["LlmDrug"]["properties"]["times"]["items"]
    assert items["enum"] == ["morning", "noon", "evening", "bedtime"]


def test_prompt_tells_the_model_about_bedtime():
    assert "bedtime" in SYSTEM_PROMPT and "취침 전" in SYSTEM_PROMPT
