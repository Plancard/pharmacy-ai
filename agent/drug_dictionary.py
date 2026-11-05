"""약품 사전 (11/5-③, REQ-AG-03).

조제 대상 약품의 품목기준코드(item_code)와 이름을 담은 JSON 파일을 읽는다.
파일이 잘못되어 있으면 매칭 결과를 믿을 수 없으므로 읽는 순간 ConfigError로 멈춘다.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from agent.drug_names import match_key
from agent.errors import ConfigError
from agent.schema import _is_9_digits


@dataclass(frozen=True)
class DrugEntry:
    item_code: str
    name: str
    key: str          # 비교용 이름 (match_key). 사전을 읽을 때 한 번만 만든다.


class DrugDictionary:
    def __init__(self, entries: list[DrugEntry]):
        self.entries = tuple(entries)
        self._by_code = {e.item_code: e for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def by_code(self, code: str | None) -> DrugEntry | None:
        """코드로 찾는다. 없거나 코드가 아니면 None."""
        return self._by_code.get(code) if isinstance(code, str) else None


def dictionary_from_data(data) -> DrugDictionary:
    """JSON에서 읽은 값으로 사전을 만든다. 잘못된 곳이 있으면 몇 번째 약인지 말하고 멈춘다."""
    if not isinstance(data, dict) or not isinstance(data.get("drugs"), list) or not data["drugs"]:
        raise ConfigError("약품 사전에 drugs 목록이 없습니다.")
    entries: list[DrugEntry] = []
    codes: set[str] = set()
    keys: set[str] = set()
    for number, item in enumerate(data["drugs"], start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"약품 사전 {number}번째 항목이 객체가 아닙니다.")
        code, name = item.get("item_code"), item.get("name")
        if not isinstance(code, str) or not _is_9_digits(code):
            raise ConfigError(f"약품 사전 {number}번째 항목의 item_code는 숫자 9자리 글자여야 합니다.")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"약품 사전 {number}번째 항목에 이름이 없습니다.")
        key = match_key(name)
        if not key:
            raise ConfigError(f"약품 사전 {number}번째 항목의 이름에 비교할 글자가 없습니다.")
        if code in codes:
            raise ConfigError(f"약품 사전 {number}번째 항목의 item_code가 앞 항목과 겹칩니다.")
        if key in keys:
            raise ConfigError(f"약품 사전 {number}번째 항목의 이름이 앞 항목과 같습니다.")
        codes.add(code)
        keys.add(key)
        entries.append(DrugEntry(item_code=code, name=name.strip(), key=key))
    return DrugDictionary(entries)


def load_dictionary(path: str | Path | None = None) -> DrugDictionary:
    """사전 파일을 읽는다. path를 주지 않으면 설정(DRUG_DICTIONARY_PATH)의 파일."""
    if path is None:
        from agent.config import DRUG_DICTIONARY_PATH
        path = DRUG_DICTIONARY_PATH
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"약품 사전 파일이 없습니다: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ConfigError(f"약품 사전 파일을 JSON(UTF-8)으로 읽을 수 없습니다: {path}") from None
    return dictionary_from_data(data)
