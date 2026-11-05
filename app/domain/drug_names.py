"""약품명 정규화.

표기 차이(공백·괄호·전각문자·단위 표기)만 없애고, 함량 숫자는 그대로 남긴다.
함량이 다르면 다른 약이라서, 함량까지 지우면 서로 다른 약이 같은 약으로 합쳐진다.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# 긴 표기를 먼저 바꿔야 한다 ("밀리그램"을 "그램"보다 먼저)
_UNIT_ALIASES = (
    ("마이크로그램", "mcg"),
    ("밀리그램", "mg"),
    ("밀리리터", "ml"),
    ("그램", "g"),
    ("μg", "mcg"),
)
_KEEP = re.compile(r"[^0-9a-z가-힣.%]")
_STRENGTH = re.compile(r"(\d+(?:\.\d+)?)(mg|mcg|g|ml|iu|%)")


def normalize_name(name: str | None) -> str:
    """'타이레놀정 500mg', '타이레놀정500밀리그램', '타이레놀정(500㎎)' → '타이레놀정500mg'"""
    s = unicodedata.normalize("NFKC", name or "").lower()   # 전각→반각, ㎎→mg
    for old, new in _UNIT_ALIASES:
        s = s.replace(old, new)
    return _KEEP.sub("", s).strip(".")


def extract_strengths(name: str | None) -> set[str]:
    """이름에서 함량 표기를 뽑는다. '부루펜정200mg' → {'200mg'}"""
    return {f"{num}{unit}" for num, unit in _STRENGTH.findall(normalize_name(name))}


def similarity(a: str, b: str) -> float:
    """정규화된 두 이름의 비슷한 정도(0~1).

    한쪽이 다른 쪽에 통째로 들어 있으면('부루펜정' ⊂ '부루펜정200mg') 0.85 이상으로 본다.
    너무 짧은 이름(2글자 이하)이 여기저기 들어맞는 것을 막기 위해 3글자 이상일 때만 적용한다.
    """
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 3 and shorter in longer:
        ratio = max(ratio, 0.85)
    return ratio
