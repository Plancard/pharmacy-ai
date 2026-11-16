"""LLM에 보내기 직전의 개인정보 가리기 (판단, 11/13). 두 번째 안전장치다.

첫 번째 안전장치는 structure.llm_input이다: 처방전에서 약품 표 범위만 골라 보내고, 위쪽의 환자 · 의료인 칸은 보내지 않는다.
그런데 흐린 사진에서 표 범위를 잘못 잡으면 위쪽 줄이 섞일 수 있다. 그래서 보내기 직전에 한 번 더 본다.

    redact_rows(rows) -> Redacted(rows, masked, dropped)

1. 개인정보 칸의 이름이 든 줄(성명 · 주민등록번호 · 면허번호 · 전화 · 의료기관 …)은 통째로 뺀다.
   약품 표 본문에는 이런 말이 없다. 이 말이 보이면 표 범위를 잘못 잡은 것이다.
2. 남은 줄에서 개인정보 모양의 글자를 가린다.
   - 주민등록번호 · 외국인등록번호: 6자리 + (-) + 7자리 (뒷자리 첫 숫자 1~8). OCR이 '-'를 빠뜨리거나 다른 기호로 읽어도,
     뒷자리를 '*'로 가려 인쇄해도 잡는다.
   - 전화번호: 0으로 시작하는 2~3자리 - 3~4자리 - 4자리 (휴대폰 · 지역번호)
   - 환자 이름 옆 번호: '김가상(000000101)'
   가린 자리는 '[가림]'으로 바꾼다. 약품 코드(9자리)와 함량 · 투약량 숫자는 이 모양에 걸리지 않는다.

가린 것이 있으면 pipeline이 조제지시서 경고에 남긴다 (약사가 원문과 대조하도록. 무엇을 가렸는지는 적지 않는다).
판단 층이다. 파일도 네트워크도 쓰지 않는다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MASK = "[가림]"

# 1) 이 말이 든 줄은 약품 표가 아니다 (공백을 뺀 글자로 찾는다). 처방전 위쪽 칸의 이름들이다.
PERSONAL_FIELD_WORDS = (
    "성명", "주민등록", "주민번호", "외국인등록", "생년월일", "면허", "전화", "팩스", "의료기관", "요양기관", "보험자",
    "피보험자", "환자", "주소",
)

# 2) 개인정보 모양
_RRN = re.compile(r"(?<!\d)\d{6}\s*[-‐‑–—~_.]?\s*[1-8][\d*×xX]{6}(?![\d*])")
_PHONE = re.compile(r"(?<!\d)0\d{1,2}\s*[-‐‑–—.)]\s*\d{3,4}\s*[-‐‑–—.]\s*\d{4}(?!\d)")
_PATIENT_NO = re.compile(r"[가-힣]{2,5}\s*\(\s*\d{9}\s*\)")
PATTERNS = (_RRN, _PHONE, _PATIENT_NO)


@dataclass(frozen=True)
class Redacted:
    rows: list[str]        # 보내도 되는 줄
    masked: int            # 가린 곳의 수 (줄 안의 글자)
    dropped: int           # 통째로 뺀 줄의 수

    @property
    def changed(self) -> bool:
        return bool(self.masked or self.dropped)


def _key(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def is_personal_row(text: str) -> bool:
    """개인정보 칸의 이름이 든 줄인가."""
    key = _key(text)
    return any(word in key for word in PERSONAL_FIELD_WORDS)


def mask_text(text: str) -> tuple[str, int]:
    """개인정보 모양의 글자를 '[가림]'으로 바꾼 글과 가린 곳의 수."""
    text = unicodedata.normalize("NFKC", text)
    count = 0
    for pattern in PATTERNS:
        text, n = pattern.subn(MASK, text)
        count += n
    return text, count


def redact_rows(rows: list[str]) -> Redacted:
    kept: list[str] = []
    masked = dropped = 0
    for row in rows:
        if is_personal_row(row):
            dropped += 1
            continue
        text, n = mask_text(row)
        masked += n
        kept.append(text)
    return Redacted(kept, masked, dropped)


def redact_issue_no(issue_no: str | None) -> tuple[str | None, int]:
    """교부번호 칸의 글자. 날짜 · 일련번호만 있어야 한다. 개인정보 모양이 섞여 있으면 가린다."""
    if issue_no is None:
        return None, 0
    if is_personal_row(issue_no):
        return None, 1
    return mask_text(issue_no)
