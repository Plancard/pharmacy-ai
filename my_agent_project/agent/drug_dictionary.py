"""약품 사전 (11/5-③, 11/6 명세서 v0.3 2.6절에 맞춤, REQ-AG-03).

조제 대상 약품의 사전을 JSON 파일에서 읽는다. 항목마다
- item_code: 식약처 품목기준코드 (필수). 모든 모듈이 약을 가리키는 공통 키다 (명세서 부록 A).
- name: 제품명 (필수). 처방전의 약품명과 견줄 때 쓴다.
- rx_code: 처방전에 인쇄되는 9자리 코드 (선택). 가상 처방전의 9000000xx, 실제 처방전이면 보험코드 자리.
- class_id: CV 학습 클래스 ID (선택). pharmacy-ai/cv/classes.yaml과 같은 값.
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
    item_code: str                # 품목기준코드
    name: str
    key: str                      # 비교용 이름 (match_key). 사전을 읽을 때 한 번만 만든다.
    rx_code: str | None = None    # 처방전에 인쇄되는 코드
    class_id: int | None = None   # CV 클래스 ID


class DrugDictionary:
    def __init__(self, entries: list[DrugEntry]):
        self.entries = tuple(entries)
        self._by_item_code = {e.item_code: e for e in self.entries}
        self._by_rx_code = {e.rx_code: e for e in self.entries if e.rx_code}

    def __len__(self) -> int:
        return len(self.entries)

    def by_item_code(self, code: str | None) -> DrugEntry | None:
        """품목기준코드로 찾는다. 없거나 코드가 아니면 None."""
        return self._by_item_code.get(code) if isinstance(code, str) else None

    def by_printed_code(self, code: str | None) -> DrugEntry | None:
        """처방전에 인쇄된 코드로 찾는다. rx_code를 먼저 보고, 없으면 품목기준코드로 본다.
        (처방전에 품목기준코드가 그대로 인쇄된 경우도 같은 약이다.) 없거나 코드가 아니면 None."""
        if not isinstance(code, str):
            return None
        return self._by_rx_code.get(code) or self._by_item_code.get(code)


def _optional_code(item: dict, field: str, number: int) -> str | None:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not _is_9_digits(value):
        raise ConfigError(f"약품 사전 {number}번째 항목의 {field}는 숫자 9자리 글자여야 합니다.")
    return value


def dictionary_from_data(data) -> DrugDictionary:
    """JSON에서 읽은 값으로 사전을 만든다. 잘못된 곳이 있으면 몇 번째 약인지 말하고 멈춘다."""
    if not isinstance(data, dict) or not isinstance(data.get("drugs"), list) or not data["drugs"]:
        raise ConfigError("약품 사전에 drugs 목록이 없습니다.")
    entries: list[DrugEntry] = []
    codes: set[str] = set()        # item_code와 rx_code를 함께 모은다: 어느 코드든 약 하나만 가리켜야 한다
    keys: set[str] = set()
    class_ids: set[int] = set()
    for number, item in enumerate(data["drugs"], start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"약품 사전 {number}번째 항목이 객체가 아닙니다.")
        code, name = item.get("item_code"), item.get("name")
        if not isinstance(code, str) or not _is_9_digits(code):
            raise ConfigError(f"약품 사전 {number}번째 항목의 item_code는 숫자 9자리 글자여야 합니다.")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"약품 사전 {number}번째 항목에 이름이 없습니다.")
        rx_code = _optional_code(item, "rx_code", number)
        class_id = item.get("class_id")
        if class_id is not None and (isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0):
            raise ConfigError(f"약품 사전 {number}번째 항목의 class_id는 0 이상의 정수여야 합니다.")
        key = match_key(name)
        if not key:
            raise ConfigError(f"약품 사전 {number}번째 항목의 이름에 비교할 글자가 없습니다.")
        if code in codes:
            raise ConfigError(f"약품 사전 {number}번째 항목의 item_code가 앞 항목의 코드와 겹칩니다.")
        if rx_code is not None and (rx_code in codes or rx_code == code):
            raise ConfigError(f"약품 사전 {number}번째 항목의 rx_code가 다른 코드와 겹칩니다.")
        if key in keys:
            raise ConfigError(f"약품 사전 {number}번째 항목의 이름이 앞 항목과 같습니다.")
        if class_id is not None and class_id in class_ids:
            raise ConfigError(f"약품 사전 {number}번째 항목의 class_id가 앞 항목과 겹칩니다.")
        codes.add(code)
        if rx_code is not None:
            codes.add(rx_code)
        keys.add(key)
        if class_id is not None:
            class_ids.add(class_id)
        entries.append(DrugEntry(item_code=code, name=name.strip(), key=key, rx_code=rx_code, class_id=class_id))
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
