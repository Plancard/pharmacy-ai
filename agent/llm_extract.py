"""LLM 경로 구조화 (11/4, 11/5 실제 처방전 양식에 맞춤).

OCR 텍스트(agent.structure.ocr_to_llm_text: 교부번호 + 약품 표 본문만, 환자·의료인 정보는 빠짐)를
OpenAI에 보내 규칙 경로와 같은 모양의 dict로 받는다.
    {"issue_no": ..., "drugs": [{"name", "printed_code", "dose_per_time",
                                 "times_per_day", "days", "time_slots"}, ...]}
검증은 여기서 하지 않는다. 결과는 agent.validate.order_from_extracted를 거쳐야 쓸 수 있다.

주의
- gpt-5 계열은 temperature를 기본값(1)만 받는다. 0 등을 보내면 400 오류가 나므로 보내지 않는다.
- openai 패키지는 실제로 호출할 때만 불러온다. (키·패키지 없이도 테스트가 돈다)
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent import config
from agent.errors import ConfigError, StructureError
from agent.schema import TimeSlot
from agent.structure import ISSUE_NOT_FOUND

SYSTEM_PROMPT = f"""\
당신은 약국 처방전(별지 제9호 서식)의 OCR 텍스트에서 약품 정보를 뽑는 도우미입니다.

입력
- 첫 줄은 "교부번호: " 뒤에 처방전의 발급 연월일 및 번호가 적혀 있습니다. 못 찾았으면 "{ISSUE_NOT_FOUND}"입니다.
- "약품 표:" 아래는 처방 의약품 표의 본문입니다. 같은 높이의 글자를 이어 붙여 한 줄로 적었습니다.
- 약품 한 줄은 보통 이 순서입니다: (순번) 9자리 코드, 약품명, 1회 투약량(예: 1 TAB), 1일 투여 횟수(예: 3회),
  총 투약 일수(예: 3일), 용법(예: PO 식후 30분). 순번 (1)은 첫 줄에만 있을 수 있습니다.
- 약품명이 길면 함량(예: 145mg)이 떨어져 나와 같은 줄의 다른 위치에 있을 수 있습니다.
- OCR이라 글자가 틀릴 수 있습니다 (예: 숫자 0을 o나 O로 읽음, 캡슐을 캠술로 읽음).

규칙
- 적힌 값만 사용하세요. 적혀 있지 않은 값을 추측하거나 지어내지 마세요.
- 틀려 보이는 글자도 고치지 말고 적힌 그대로 두세요. 약품은 뒤 단계에서 코드로 확인하고 약사가 검토합니다.
- issue_no: 교부번호를 적힌 그대로. "{ISSUE_NOT_FOUND}"이면 null.
- printed_code: 약품명 앞의 숫자 9자리. 순번 (1)은 빼세요. 9자리 숫자가 없으면 null.
- name: 코드 뒤의 약품명. 떨어져 나온 함량 조각도 이름 뒤에 붙이고, '/'는 빼고 낱말 사이는 공백 하나로 쓰세요.
  예: "(1)900000011 타이레놀정/ 5oOmg 1 TAB 3회 3일 PO 식후 30분" -> name "타이레놀정 5oOmg"
- dose_per_time: 1회 투약량(정수). "1 TAB"이면 1. 0.5나 1/2처럼 정수가 아니거나 읽을 수 없으면 0.
- times_per_day: 1일 투여 횟수(정수). "3회"이면 3. 읽을 수 없으면 0.
- days: 총 투약 일수(정수). "3일"이면 3. 읽을 수 없으면 0.
- time_slots: 용법에 아침·점심·저녁·취침 전이 적혀 있을 때만 해당하는 것을 morning, noon, evening, bedtime으로 담으세요.
  "취침 전"과 "자기 전"은 bedtime입니다. "PO 식후 30분", "매 식후"처럼 시점이 적혀 있지 않으면 빈 목록입니다.
  횟수를 보고 시점을 짐작해 채우지 마세요.
- 약품 표의 한 줄이 약품 하나입니다. 약품이 하나도 없으면 drugs는 빈 목록으로 두세요.
"""

FEEDBACK_TEMPLATE = (
    "앞선 추출 결과가 검증에서 거절되었습니다. 사유: {reason}\n"
    "처방전 원문에 적힌 값만 사용해 다시 추출하세요. 확실하지 않은 값은 추측하지 말고 0 또는 null로 두세요."
)


# 구조화 출력용 모양. OpenAI의 엄격한 스키마 규칙에 맞게 모든 칸을 필수로 두고(기본값 없음),
# 값의 범위 검사는 하지 않는다. 범위 검사는 agent.validate가 맡는다.
class LlmDrug(BaseModel):
    name: str
    printed_code: str | None
    dose_per_time: int
    times_per_day: int
    days: int
    time_slots: list[TimeSlot]


class LlmPrescription(BaseModel):
    issue_no: str | None
    drugs: list[LlmDrug]


def _default_client():
    api_key = config.require_openai_key()   # 키가 없으면 ConfigError
    try:
        from openai import OpenAI   # 실제로 호출할 때만 불러온다
    except ImportError as e:
        raise ConfigError("openai 패키지가 설치되어 있지 않습니다. `pip install openai`로 설치하세요.") from e

    # SDK의 자동 재시도는 끈다. 재시도는 구조화 함수가 횟수를 세어 가며 직접 한다.
    return OpenAI(api_key=api_key, timeout=config.LLM_TIMEOUT_SECONDS, max_retries=0)


def extract_llm(
    text: str,
    feedback: str | None = None,
    *,
    client: Any = None,
    model: str | None = None,
) -> dict[str, Any]:
    """OCR 텍스트(agent.structure.ocr_to_llm_text의 결과)를 LLM으로 구조화한다.

    feedback이 있으면 앞선 결과가 검증에서 거절된 사유로 보고 모델에게 다시 알려 준다.
    실패하면 StructureError. 키가 없으면 ConfigError (재시도해도 소용없다).
    """
    if not isinstance(text, str) or not text.strip():
        raise StructureError("OCR 텍스트가 비어 있습니다.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if feedback:
        messages.append({"role": "user", "content": FEEDBACK_TEMPLATE.format(reason=feedback)})

    client = client or _default_client()   # 키가 없으면 ConfigError
    try:
        completion = client.chat.completions.parse(
            model=model or config.OPENAI_MODEL,
            messages=messages,
            response_format=LlmPrescription,
        )
        message = completion.choices[0].message
    except Exception as e:
        # 예외 문장에는 처방전 내용이 섞여 들어올 수 있어 종류만 남긴다.
        raise StructureError(f"LLM 호출에 실패했습니다 ({type(e).__name__}).") from e

    parsed = getattr(message, "parsed", None)
    if parsed is None:
        if getattr(message, "refusal", None):
            raise StructureError("LLM이 요청을 거절했습니다.")
        raise StructureError("LLM이 구조화 결과를 돌려주지 않았습니다.")
    return parsed.model_dump()
