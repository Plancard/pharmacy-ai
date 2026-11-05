"""gpt-4o 호출 래퍼.

LLM 호출은 전부 여기를 거친다. 제한 시간·재시도를 한곳에서 관리해야
전체 처리 시간(요구사항 1분)을 넘기지 않게 조절할 수 있다.

주의: 이 파일은 실제 OpenAI API로는 아직 실행해 보지 않았다. 키를 넣은 뒤
notebooks/test_prompt.ipynb 의 3번(키 확인)·5번(구조화) 셀로 먼저 확인하자.
"""
from __future__ import annotations

from typing import Any

from app import config
from app.schemas import ExtractedPrescription


def load_prompt(name: str) -> str:
    return (config.PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def get_chat_model():
    # langchain_openai는 무거워서 실제로 호출할 때만 불러온다 (키 없이 테스트가 돌도록).
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
        api_key=config.require_openai_key(),
    )


def extract_prescription(ocr_text: str) -> dict[str, Any]:
    """OCR 텍스트 → ExtractedPrescription 형태의 dict."""
    model = get_chat_model().with_structured_output(ExtractedPrescription)
    result = model.invoke([("system", load_prompt("structure")), ("human", ocr_text)])
    return result.model_dump()
