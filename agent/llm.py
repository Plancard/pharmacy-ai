"""LLM을 부르는 유일한 곳 (가장자리, 3단계). 모델 이름 · 시간 제한 · 오류 코드(E_LLM)가 여기에만 있다.

판단 층(pipeline.py 등)은 이 파일을 직접 import하지 않는다. 호출 함수(extract_llm)를 인자로 받아서,
키도 네트워크도 없이 가짜 함수로 시험한다. 이 파일을 이어 붙이는 일은 그래프(nodes.py)의 몫이다.

처방 구조화
    extract_llm(text, feedback=None)  →  {"issue_no": ..., "items": [{"name", "printed_code", "count_per_dose",
                                                                      "times_per_day", "days", "times"}, ...]}
    text는 structure.format_llm_text의 결과(교부번호 + 약품 표 본문)다. 환자·의료인 정보는 보내지 않는다.
    검증은 여기서 하지 않는다. 결과는 validate.order_from_extracted를 거쳐야 쓸 수 있다.
    재시도는 여기서 하지 않는다. 몇 번 다시 시도할지는 pipeline.py가 센다 (SDK 자동 재시도는 꺼 둔다).

지시문은 agent/prompts/*.md 파일이다 (structure.md: 시스템 지시, structure_retry.md: 거절 사유를 알려 주는 말).

주의
- gpt-5 계열은 temperature를 기본값(1)만 받는다. 0 등을 보내면 400 오류가 나므로 보내지 않는다.
- openai 패키지는 실제로 호출할 때만 불러온다. (키·패키지 없이도 테스트가 돈다)
- 오류 문장에는 처방전 내용이 섞여 들어올 수 있어, 예외는 종류(이름)만 남기고 문장은 버린다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from agent import config
from agent.errors import E_LLM, ConfigError, StructureError
from agent.schema import TimeSlot


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """agent/prompts/<name>.md 의 글. 앞뒤 빈 줄은 뗀다."""
    return (config.PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


# 구조화 출력용 모양. OpenAI의 엄격한 스키마 규칙에 맞게 모든 칸을 필수로 두고(기본값 없음),
# 값의 범위 검사는 하지 않는다. 범위 검사는 agent.validate가 맡는다.
class LlmDrug(BaseModel):
    name: str
    printed_code: str | None
    count_per_dose: int
    times_per_day: int
    days: int
    times: list[TimeSlot]


class LlmPrescription(BaseModel):
    issue_no: str | None
    items: list[LlmDrug]


def _default_client():
    api_key = config.require_openai_key()   # 키가 없으면 ConfigError
    try:
        from openai import OpenAI   # 실제로 호출할 때만 불러온다
    except ImportError as e:
        raise ConfigError("openai 패키지가 설치되어 있지 않습니다. `pip install openai`로 설치하세요.") from e

    # SDK의 자동 재시도는 끈다. 재시도는 구조화 함수가 횟수를 세어 가며 직접 한다.
    return OpenAI(api_key=api_key, timeout=config.LLM_TIMEOUT_SECONDS, max_retries=0)


def chat_parse(
    messages: list[dict[str, str]],
    response_format: type[BaseModel],
    *,
    client: Any = None,
    model: str | None = None,
) -> BaseModel:
    """메시지를 보내고 response_format 모양으로 받는다. 실패하면 StructureError(E_LLM), 키가 없으면 ConfigError.

    앞으로 다른 LLM 호출(의도 분류, 답변 생성 …)도 이 함수를 거치면 시간 제한과 오류 처리가 한 곳에 모인다.
    """
    client = client or _default_client()   # 키가 없으면 ConfigError (재시도해도 소용없다)
    try:
        completion = client.chat.completions.parse(
            model=model or config.OPENAI_MODEL,
            messages=messages,
            response_format=response_format,
        )
        message = completion.choices[0].message
    except Exception as e:
        raise StructureError(f"LLM 호출에 실패했습니다 ({type(e).__name__}).", code=E_LLM) from e

    parsed = getattr(message, "parsed", None)
    if parsed is None:
        if getattr(message, "refusal", None):
            raise StructureError("LLM이 요청을 거절했습니다.", code=E_LLM)
        raise StructureError("LLM이 구조화 결과를 돌려주지 않았습니다.", code=E_LLM)
    return parsed


def extract_llm(
    text: str,
    feedback: str | None = None,
    *,
    client: Any = None,
    model: str | None = None,
) -> dict[str, Any]:
    """처방 표 텍스트(structure.format_llm_text의 결과)를 LLM으로 구조화한다.

    feedback이 있으면 앞선 결과가 검증에서 거절된 사유로 보고 모델에게 다시 알려 준다.
    실패하면 StructureError(E_LLM). 키가 없으면 ConfigError (재시도해도 소용없다).
    """
    if not isinstance(text, str) or not text.strip():
        raise StructureError("OCR 텍스트가 비어 있습니다.", code=E_LLM)

    messages = [
        {"role": "system", "content": load_prompt("structure")},
        {"role": "user", "content": text},
    ]
    if feedback:
        messages.append({"role": "user", "content": load_prompt("structure_retry").format(reason=feedback)})
    return chat_parse(messages, LlmPrescription, client=client, model=model).model_dump()
