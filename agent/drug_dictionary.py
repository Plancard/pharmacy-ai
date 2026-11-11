"""약품 사전 읽기 (4단계, 일정표 11/5, REQ-AG-03).

조제 대상 약품 8종의 사전을 JSON 파일(agent/resources/drug_dictionary.json)에서 읽는다. 항목마다
- item_code: 식약처 품목기준코드 (필수, 숫자 9자리). 모든 모듈이 약을 가리키는 공통 키다 (명세서 부록 A).
- name: 제품명 (필수). CV의 cv/classes.yaml과 같은 이름을 쓴다. 처방전의 약품명과 견줄 때 쓴다.
- class_id: CV 학습 클래스 ID (선택). cv/classes.yaml의 id와 같은 값이다.
- k_code: AI Hub의 약 번호 dl_mapping_code (선택, K-와 숫자 6자리). cv/classes.yaml의 k_code와 같은 값이다.
  트레이 검출 결과가 k_code로 오면 이 표로 품목기준코드를 찾는다 (12단계).
- ingredients: 주성분 이름 목록 (선택, 명세서 2.6절). 처방약끼리 성분 중복을 찾는 데 쓴다 (dur.py).
  비어 있으면 성분을 모르는 약이다. 중복 대조에서 빠지고 조제지시서의 경고에 그렇게 남는다.

처방전에 인쇄된 코드를 따로 적는 칸(rx_code)은 없다. 자동 확정은 처방전에 인쇄된 코드가 사전의 품목기준코드와
같을 때뿐이고, 그 밖에는 이름으로 찾아 약사 확인으로 넘긴다 (drug_match.py).
파일이 잘못되어 있으면 매칭 결과를 믿을 수 없으므로 읽는 순간 ConfigError로 멈춘다. 알 수 없는 키도 거부한다
(예전 사전에 있던 rx_code가 남아 있으면 조용히 무시하지 않고 알린다).

이 파일은 판단 층이다. 파일을 읽을 뿐 네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from agent import config
from agent.drug_names import match_key
from agent.errors import ConfigError

_ITEM_CODE = re.compile(r"[0-9]{9}")           # [0-9]: \d는 전각 숫자 같은 다른 나라 숫자도 받는다
_K_CODE = re.compile(r"K-[0-9]{6}")
_ENTRY_KEYS = {"item_code", "name", "class_id", "k_code", "ingredients"}


def ingredient_key(name: str) -> str:
    """성분 이름을 견주기 위한 글자. 공백을 없애고 전각·대소문자 차이를 맞춘다.

    다른 이름으로 불리는 같은 성분(파라세타몰 = 아세트아미노펜)은 맞추지 않는다. 사전에 한 가지 이름으로 적는다.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).casefold()


@dataclass(frozen=True)
class DrugEntry:
    item_code: str                # 품목기준코드
    name: str
    key: str                      # 비교용 이름 (match_key). 사전을 읽을 때 한 번만 만든다.
    class_id: int | None = None   # CV 클래스 ID
    k_code: str | None = None     # AI Hub 약 번호 (dl_mapping_code)
    ingredients: tuple[str, ...] = ()   # 주성분. 비어 있으면 모르는 약


class DrugDictionary:
    def __init__(self, entries: list[DrugEntry]):
        self.entries = tuple(entries)
        self._by_item_code = {e.item_code: e for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def by_item_code(self, code: str | None) -> DrugEntry | None:
        """품목기준코드로 찾는다. 없거나 글자가 아니면 None."""
        return self._by_item_code.get(code) if isinstance(code, str) else None

    def by_class_id(self, class_id: int | None) -> DrugEntry | None:
        """CV 클래스 ID로 찾는다. 없거나 정수가 아니면 None."""
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            return None
        return next((e for e in self.entries if e.class_id == class_id), None)

    def by_k_code(self, k_code: str | None) -> DrugEntry | None:
        """AI Hub 약 번호(k_code)로 찾는다. 없거나 글자가 아니면 None."""
        if not isinstance(k_code, str):
            return None
        return next((e for e in self.entries if e.k_code == k_code), None)


def _read_ingredients(value, number: int) -> tuple[str, ...]:
    """ingredients 칸을 읽는다. 칸이 없거나 빈 목록이면 성분을 모르는 약이다."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"약품 사전 {number}번째 항목의 ingredients는 성분 이름(글자) 목록이어야 합니다.")
    names = tuple(v.strip() for v in value)
    if len({ingredient_key(n) for n in names}) != len(names):
        raise ConfigError(f"약품 사전 {number}번째 항목의 ingredients에 같은 성분이 두 번 나옵니다.")
    return names


def dictionary_from_data(data) -> DrugDictionary:
    """JSON에서 읽은 값으로 사전을 만든다. 잘못된 곳이 있으면 몇 번째 약인지 말하고 멈춘다."""
    if not isinstance(data, dict) or not isinstance(data.get("drugs"), list) or not data["drugs"]:
        raise ConfigError("약품 사전에 drugs 목록이 없습니다.")
    entries: list[DrugEntry] = []
    codes: set[str] = set()
    keys: set[str] = set()
    class_ids: set[int] = set()
    k_codes: set[str] = set()
    for number, item in enumerate(data["drugs"], start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"약품 사전 {number}번째 항목이 객체가 아닙니다.")
        unknown = sorted(set(item) - _ENTRY_KEYS)
        if unknown:
            raise ConfigError(f"약품 사전 {number}번째 항목에 알 수 없는 칸이 있습니다: {', '.join(map(str, unknown))}")
        code, name = item.get("item_code"), item.get("name")
        if not isinstance(code, str) or not _ITEM_CODE.fullmatch(code):
            raise ConfigError(f"약품 사전 {number}번째 항목의 item_code는 숫자 9자리 글자여야 합니다.")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"약품 사전 {number}번째 항목에 이름이 없습니다.")
        class_id = item.get("class_id")
        if class_id is not None and (isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0):
            raise ConfigError(f"약품 사전 {number}번째 항목의 class_id는 0 이상의 정수여야 합니다.")
        k_code = item.get("k_code")
        if k_code is not None and (not isinstance(k_code, str) or not _K_CODE.fullmatch(k_code)):
            raise ConfigError(f"약품 사전 {number}번째 항목의 k_code는 K-와 숫자 6자리 글자여야 합니다.")
        ingredients = _read_ingredients(item.get("ingredients"), number)
        key = match_key(name)
        if not key:
            raise ConfigError(f"약품 사전 {number}번째 항목의 이름에 비교할 글자가 없습니다.")
        if code in codes:
            raise ConfigError(f"약품 사전 {number}번째 항목의 item_code가 앞 항목의 코드와 겹칩니다.")
        if key in keys:
            raise ConfigError(f"약품 사전 {number}번째 항목의 이름이 앞 항목과 같습니다.")
        if class_id is not None and class_id in class_ids:
            raise ConfigError(f"약품 사전 {number}번째 항목의 class_id가 앞 항목과 겹칩니다.")
        if k_code is not None and k_code in k_codes:
            raise ConfigError(f"약품 사전 {number}번째 항목의 k_code가 앞 항목과 겹칩니다.")
        codes.add(code)
        keys.add(key)
        if class_id is not None:
            class_ids.add(class_id)
        if k_code is not None:
            k_codes.add(k_code)
        entries.append(
            DrugEntry(item_code=code, name=name.strip(), key=key, class_id=class_id, k_code=k_code, ingredients=ingredients)
        )
    return DrugDictionary(entries)


def load_dictionary(path: str | Path | None = None) -> DrugDictionary:
    """사전 파일을 읽는다. path를 주지 않으면 설정(DRUG_DICTIONARY_PATH)의 파일."""
    path = Path(path if path is not None else config.DRUG_DICTIONARY_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"약품 사전 파일이 없습니다: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ConfigError(f"약품 사전 파일을 JSON(UTF-8)으로 읽을 수 없습니다: {path}") from None
    return dictionary_from_data(data)
