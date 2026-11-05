"""환경변수·경로·모델 설정.

다른 모듈은 os.environ을 직접 읽지 말고 여기서 가져온다.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

load_dotenv(BASE_DIR / ".env")  # .env가 없어도 오류 없이 지나간다


class ConfigError(RuntimeError):
    """필수 설정이 빠졌을 때.

    데이터 문제가 아니라 개발 환경 문제이므로, 화면 안내로 바꾸지 않고 그대로 올려 보낸다.
    """


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# 호출 1번의 제한 시간(초)과 재시도 횟수.
# 최악의 경우 (1 + 재시도) x 제한 시간만큼 걸린다. 기본값이면 최대 40초.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

DATA_GO_KR_SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY", "")


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            ".env에 OPENAI_API_KEY가 없습니다. .env.example을 복사해 .env를 만들고 키를 채워 주세요."
        )
    return key
