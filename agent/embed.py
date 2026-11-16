"""임베딩 만들기 (가장자리, 16단계). OpenAI 임베딩 API를 부르는 유일한 곳. 모델은 config.EMBED_MODEL.

    embed_texts(texts) -> list[list[float]]     색인을 만들 때 (build_rag_index.py --embed) · 질문 하나를 찾을 때

키가 없으면 ConfigError. 호출이 실패하면 EmbedError (상담은 멈추지 않고 BM25만으로 찾는다, REQ-NF-06).
오류 문장에는 질문이 섞일 수 있어 예외 종류만 남긴다 (llm.py와 같다).
"""
from __future__ import annotations

from typing import Any, Sequence

from agent import config
from agent.errors import ConfigError

BATCH = 64


class EmbedError(Exception):
    pass


def _client():
    api_key = config.require_openai_key()
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ConfigError("openai 패키지가 설치되어 있지 않습니다. `pip install openai`로 설치하세요.") from e
    return OpenAI(api_key=api_key, timeout=config.LLM_TIMEOUT_SECONDS, max_retries=0)


def embed_texts(texts: Sequence[str], *, client: Any = None, model: str | None = None) -> list[list[float]]:
    client = client or _client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        try:
            answer = client.embeddings.create(model=model or config.EMBED_MODEL, input=list(texts[start:start + BATCH]))
        except Exception as e:
            raise EmbedError(f"임베딩 호출에 실패했습니다 ({type(e).__name__}).") from e
        vectors.extend(list(d.embedding) for d in answer.data)
    if len(vectors) != len(texts):
        raise EmbedError("임베딩 결과의 개수가 맞지 않습니다.")
    return vectors
