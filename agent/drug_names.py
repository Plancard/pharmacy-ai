"""약품명 정규화 (11/5, REQ-AG-03).

표기 차이(공백·괄호·전각 문자·단위 표기)만 없애고, 함량 숫자는 그대로 남긴다.
함량이 다르면 다른 약이다. 함량까지 지우면 서로 다른 약(부루펜정200mg / 400mg)이 같은 약으로 합쳐진다.

여기서 만든 이름 열쇠(normalize_name 결과)가 같다는 것은 "표기만 다른 같은 이름"이라는 뜻이다.
이름이 비슷하다는 것(similarity)은 한글 자모 단위로 잰다. 비슷하다고 같은 약은 아니다(알모탄정 / 알모란정 0.91).
그래서 코드가 가리키는 약과 이름이 어긋나지 않는지 보는 데만 쓰고, 이름만으로 약을 정하지 않는다. (agent/drug_match.py)
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# 숫자 바로 뒤에 올 때만 단위로 바꾼다. "프로그램정"의 "그램"은 단위가 아니고,
# "500밀리그램" 안의 "그램"도 숫자 뒤가 아니라서 "밀리g"로 잘못 바뀌지 않는다.
# 식약처 자료는 '밀리그람'처럼 받침을 달리 적기도 한다 (타이레놀정500밀리그람). 두 철자를 같은 단위로 본다.
_UNIT_WORDS = re.compile(r"(?<=[0-9])(마이크로그[램람]|밀리그[램람]|밀리리터|그[램람])")
_UNIT_WORD_TO_SYMBOL = {
    "마이크로그램": "mcg", "마이크로그람": "mcg", "밀리그램": "mg", "밀리그람": "mg", "밀리리터": "ml", "그램": "g", "그람": "g",
}

_WHITESPACE = re.compile(r"\s+")
_KEEP = re.compile(r"[^0-9a-z가-힣.%]")        # 소수점(0.5)과 퍼센트(0.5%)는 함량이라 남긴다
_STRENGTH = re.compile(r"(\d+(?:\.\d+)?)(mg|mcg|g|ml|iu|%)")


def normalize_name(name: str | None) -> str:
    """'타이레놀정 500mg', '타이레놀정500밀리그램', '타이레놀정(500 ㎎)' -> '타이레놀정500mg'

    - 전각을 반각으로, 영문은 소문자로 (NFKC: ㎎ -> mg, ㎍ -> μg, ＡＢＣ -> ABC)
    - 공백을 모두 없앤다. 단위를 바꾸기 전에 해야 OCR이 '밀리 그램'처럼 끊어 읽어도 같아진다.
    - 단위 표기를 mg / mcg / g / ml로 통일한다.
    - 글자·숫자·소수점·퍼센트만 남기고 괄호·하이픈·쉼표 등은 없앤다.
    """
    s = unicodedata.normalize("NFKC", name or "").lower()
    s = _WHITESPACE.sub("", s)
    s = _UNIT_WORDS.sub(lambda m: _UNIT_WORD_TO_SYMBOL[m.group(1)], s)
    s = s.replace("μg", "mcg")   # NFKC가 마이크로 기호(µ, ㎍)를 그리스 문자 μ로 통일해 둔다
    return _KEEP.sub("", s).strip(".")


def extract_strengths(name: str | None) -> set[str]:
    """이름에서 함량 표기를 뽑는다. '부루펜정 200 밀리그램' -> {'200mg'}"""
    return {f"{number}{unit}" for number, unit in _STRENGTH.findall(normalize_name(name))}


# OCR이 숫자 0을 알파벳 o/O로 읽는 일이 실제로 있다 (5oOmg, 1Omg, 25Omg). 함량(숫자+단위) 안에서만 0으로 되돌린다.
# 숫자가 하나는 있어야 하고(Omg는 그대로), 앞이 영문·숫자에 붙어 있으면(Cipro500mg의 o) 건드리지 않는다.
_UNITS = r"(?:mg|mcg|\u03bcg|g|ml|iu|%|밀리그[램람]|마이크로그[램람]|그[램람]|밀리리터)"     # \u03bc = NFKC가 마이크로 기호를 바꾼 그리스 문자
_OCR_ZERO_RUN = re.compile(rf"(?<![0-9A-Za-z])([0-9oO]*[0-9][0-9oO]*)(?=\s*{_UNITS})", re.IGNORECASE)


def fix_ocr_digits(name: str | None) -> str:
    """함량 안의 알파벳 o/O를 0으로 되돌린다. '타이레놀정 5oOmg' -> '타이레놀정 500mg', '에스코렌정 1Omg' -> '에스코렌정 10mg'

    숫자 자체를 잘못 읽은 것(45mg -> 145mg)은 고칠 수 없다. 그런 차이는 함량 비교에서 약사 확인으로 넘어간다.
    """
    text = unicodedata.normalize("NFKC", name or "")
    return _OCR_ZERO_RUN.sub(lambda m: m.group(1).replace("o", "0").replace("O", "0"), text)


def match_key(name: str | None) -> str:
    """약품 매칭에서 이름을 견줄 때 쓰는 열쇠: OCR 숫자 보정 + 표기 정규화."""
    return normalize_name(fix_ocr_digits(name))


# 의약품 API 검색에 쓰는 말 만들기. API의 item_name 검색은 이름의 일부로 찾는다. 그래서 함량·'/'·괄호를 뺀 이름 앞부분으로
# 찾고('타이레놀정 5oOmg' -> '타이레놀정'), 함량까지 맞는지는 돌아온 제품 이름과 견주어 본다 (drug_match.py).
# OCR이 이름 글자를 틀리게 읽었으면('필루비서방정') API도 찾지 못한다. 그런 약은 약사 확인으로 넘어간다.
_STRENGTH_ANY = re.compile(r"[0-9]+(?:\.[0-9]+)?\s*(?:mg|mcg|\u03bcg|g|ml|iu|%|밀리그[램람]|마이크로그[램람]|그[램람]|밀리리터)", re.IGNORECASE)
_PARENS = re.compile(r"\([^)]*\)")


def search_term(name: str | None) -> str:
    """API에 보낼 검색어: 함량·괄호·'/'·공백을 뺀 이름. '타이레놀정 5oOmg' -> '타이레놀정'"""
    text = unicodedata.normalize("NFKC", fix_ocr_digits(name))
    text = _PARENS.sub(" ", text)
    text = _STRENGTH_ANY.sub(" ", text)
    return "".join(text.replace("/", " ").split())


def product_key(item_name: str | None) -> str:
    """API 제품명을 견줄 열쇠. 괄호 안(성분명 등)은 뺀다. '타이레놀정500밀리그램(아세트아미노펜)' -> '타이레놀정500mg'"""
    return match_key(_PARENS.sub(" ", item_name or ""))


def jamo(text: str) -> str:
    """한글 음절을 자모로 풀어 쓴다. 한글이 아닌 글자는 그대로 둔다."""
    return unicodedata.normalize("NFD", text)


def similarity(a: str, b: str) -> float:
    """정규화된 두 이름이 비슷한 정도 (0~1). 한글 자모 단위로 잰다.

    음절 단위로 재면 받침이나 모음 하나만 다른 글자(탄/란)가 통째로 다른 글자가 되어
    OCR이 한 획을 잘못 읽은 이름과 정말 다른 이름이 구별되지 않는다.
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, jamo(a), jamo(b)).ratio()
