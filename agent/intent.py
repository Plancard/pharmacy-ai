"""글의 의도를 가른다 (11단계, 일정표 11/13, REQ-AG-01) — 규칙으로만. 규칙이 못 가른 글은 19단계에서 LLM이 가른다.

메시지의 글 하나에서 의도를 찾는다 (명세서 2.2절 복합 입력). 한 글에 여러 의도가 함께 있을 수 있다.
의도 키는 기능분석서 v0.3 4.2절(F-AG-01)과 같다. 웹도 이 키를 쓴다.
    patient_info  복용 약 정보   "타이레놀 먹고 있어요", "복용 중인 약: 게보린"
    edit_request  수정 요청      "게보린 빼 주세요", "2일분으로 바꿔 주세요"
    question      질문          "같이 먹어도 되나요?", "부작용 알려 주세요"
    note          메모          "메모: 오후 수령", "참고로 …"
아무것도 찾지 못하면 빈 목록이다 (애매함). 그때 슈퍼바이저가 되묻는다.

글은 자동으로 반영하지 않는다. 의도는 슈퍼바이저가 흐름을 고르고 proposal.py가 제안을 만드는 데만 쓴다.
판단 층이다. 파일도 네트워크도 쓰지 않는다.
"""
from __future__ import annotations

import re
import unicodedata

PATIENT_INFO, EDIT_REQUEST, QUESTION, NOTE = "patient_info", "edit_request", "question", "note"
INTENTS = (PATIENT_INFO, EDIT_REQUEST, QUESTION, NOTE)        # 돌려주는 순서
INTENT_LABELS = {PATIENT_INFO: "복용 약 정보", EDIT_REQUEST: "수정 요청", QUESTION: "질문", NOTE: "메모"}

# 의도마다 찾을 말. 글에서 공백을 지운 뒤 찾는다 ("먹고 있" → "먹고있").
WORDS: dict[str, tuple[str, ...]] = {
    PATIENT_INFO: ("복용중", "복용하고", "복용하는", "복용약", "먹고있", "먹고계", "먹는약", "드시고", "드시는", "이미먹", "원래먹"),
    EDIT_REQUEST: ("바꿔", "바꾸", "변경", "수정", "고쳐", "고치", "빼주", "빼고", "빼줘", "빼세요", "제외", "추가해", "추가로", "넣어",
           "말고", "대신", "늘려", "줄여"),
    QUESTION: ("괜찮", "되나요", "될까", "되는지", "가능한", "가능할", "알려", "어떻게", "무엇", "뭐가", "뭔가", "왜", "인가요",
               "나요", "까요", "어떤가", "부작용", "주의할", "먹어도"),
    NOTE: ("메모", "참고", "기록", "전달"),
}
_SPACES = re.compile(r"\s+")


def compact(text: str) -> str:
    return _SPACES.sub("", unicodedata.normalize("NFKC", text or ""))


def classify(text: str | None) -> list[str]:
    """글의 의도 목록 (INTENTS 순서). 빈 글이면 빈 목록, 아무 말도 못 찾아도 빈 목록(애매함)."""
    key = compact(text or "")
    if not key:
        return []
    found = {intent for intent in INTENTS if any(word in key for word in WORDS[intent])}
    if key.endswith("?"):
        found.add(QUESTION)
    return [intent for intent in INTENTS if intent in found]
