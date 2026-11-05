"""약품 카탈로그: 품목기준코드(item_code) ↔ 약품명·보험코드(edi_code).

지금은 data/drug_catalog.json(가상)을 읽는다. 나중에 식약처 API 조회나 MCP 서버(③)로
바꿔도 Catalog의 조회 메서드는 그대로 두면 매칭 코드는 안 바뀐다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from app import config
from app.domain.drug_names import normalize_name

_CODE_RE = re.compile(r"^\d{9}$")


@dataclass(frozen=True)
class CatalogItem:
    item_code: str                      # 품목기준코드 9자리
    name: str
    edi_code: Optional[str] = None      # 보험코드 9자리


class Catalog:
    def __init__(self, items: Iterable[CatalogItem]):
        self._items: list[CatalogItem] = []
        self._by_item_code: dict[str, CatalogItem] = {}
        self._by_edi_code: dict[str, CatalogItem] = {}
        self._by_name_key: dict[str, list[CatalogItem]] = {}

        for item in items:
            if not _CODE_RE.match(item.item_code):
                raise ValueError(f"품목기준코드가 숫자 9자리가 아닙니다: {item.item_code!r} ({item.name})")
            if item.edi_code is not None and not _CODE_RE.match(item.edi_code):
                raise ValueError(f"보험코드가 숫자 9자리가 아닙니다: {item.edi_code!r} ({item.name})")
            if item.item_code in self._by_item_code:
                raise ValueError(f"품목기준코드가 중복됩니다: {item.item_code}")
            if item.edi_code is not None and item.edi_code in self._by_edi_code:
                raise ValueError(f"보험코드가 중복됩니다: {item.edi_code}")

            self._items.append(item)
            self._by_item_code[item.item_code] = item
            if item.edi_code is not None:
                self._by_edi_code[item.edi_code] = item
            self._by_name_key.setdefault(normalize_name(item.name), []).append(item)

    @property
    def items(self) -> list[CatalogItem]:
        return list(self._items)

    def get_by_item_code(self, code: str) -> Optional[CatalogItem]:
        return self._by_item_code.get(code)

    def get_by_edi_code(self, code: str) -> Optional[CatalogItem]:
        return self._by_edi_code.get(code)

    def find_by_name_key(self, key: str) -> list[CatalogItem]:
        """정규화된 이름(normalize_name 결과)이 같은 품목들."""
        return list(self._by_name_key.get(key, []))


def load_catalog(path: str | Path | None = None) -> Catalog:
    path = Path(path) if path else config.DATA_DIR / "drug_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return Catalog(
        CatalogItem(item_code=row["item_code"], name=row["name"], edi_code=row.get("edi_code"))
        for row in data["items"]
    )


@lru_cache(maxsize=1)
def get_default_catalog() -> Catalog:
    """기본 카탈로그 (한 번만 읽고 재사용)."""
    return load_catalog()
