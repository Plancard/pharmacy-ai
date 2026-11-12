"""식약처 의약품안전사용서비스(DUR) 품목정보 API 조회 (13단계, 일정표 11/17, REQ-AG-05).

품목기준코드 하나로 그 약의 병용금기 상대 약 목록과 효능군을 받아 온다. 어느 처방약끼리 걸리는지는 dur.py가 판단한다.
가장자리 층이다: 바깥(네트워크)에 닿는 일만 한다.

- 서비스는 DURPrdlstInfoService03 (공공데이터포털 15059486)이다. 키는 제품 허가정보와 같은 .env의 DRUG_API_KEY
  (공공데이터포털 일반 인증키는 하나로 승인받은 여러 API를 쓴다). DUR 품목정보도 따로 활용신청 '승인'이어야 한다.
- 오퍼레이션 이름과 응답 필드는 아래 상수에 모았다. 실제 응답과 다르면 여기만 고친다.
  확인: python scripts/agent/show_dur.py  (응답 원본을 var/api/dur_*.json 으로 저장하고 필드 이름을 보여 준다)
- 실패하면 DrugApiError(E_EXT_API). 키는 화면 · 오류 문장 · 파일 어디에도 적지 않는다 (drug_api.http_get_json이 가린다).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent import config
from agent.drug_api import Fetch, http_get_json, raw_items

TABOO_OPERATION = "getUsjntTabooInfoList03"      # 병용금기 정보조회
EFFICACY_OPERATION = "getEfcyDplctInfoList03"    # 효능군중복 정보조회
PAGE_ROWS = 100                                  # 한 번에 받는 줄 수
MAX_PAGES = 5                                    # 한 약에 대해 받는 최대 쪽 수 (병용금기 상대가 수천 개인 약도 있다)

# 응답 필드 (병용금기): 상대 약의 품목기준코드 · 이름 · 성분, 금기 사유, 이 약의 성분
TABOO_PARTNER_CODE = "MIXTURE_ITEM_SEQ"
TABOO_PARTNER_NAME = "MIXTURE_ITEM_NAME"
TABOO_PARTNER_INGREDIENT = "MIXTURE_INGR_KOR_NAME"
TABOO_REASON = "PROHBT_CONTENT"
TABOO_INGREDIENT = "INGR_KOR_NAME"
# 응답 필드 (효능군중복): 효능군 이름 · 계열 이름
EFFICACY_GROUP = "EFFECT_NAME"
EFFICACY_SERIES = "SERS_NAME"


@dataclass(frozen=True)
class TabooPartner:
    item_code: str           # 함께 쓰면 안 되는 약의 품목기준코드
    name: str                # 그 약의 이름
    ingredient: str          # 그 약의 성분 (한글)
    reason: str              # 금기 사유 (PROHBT_CONTENT, 없으면 빈 문자열)


@dataclass(frozen=True)
class TabooList:
    partners: tuple[TabooPartner, ...]
    complete: bool           # 목록을 끝까지 받았는가 (MAX_PAGES에서 끊겼으면 False)


def _text(row: dict[str, Any], field: str) -> str:
    return " ".join(str(row.get(field) or "").split())


def _total(data: Any) -> int | None:
    root = data.get("response", data) if isinstance(data, dict) else {}
    body = root.get("body") if isinstance(root, dict) else None
    try:
        return int(body.get("totalCount")) if isinstance(body, dict) else None
    except (TypeError, ValueError):
        return None


def fetch_rows(operation: str, item_code: str, *, key: str | None = None, fetch: Fetch | None = None,
               max_pages: int = MAX_PAGES) -> tuple[list[dict[str, Any]], bool]:
    """한 오퍼레이션에서 이 약(itemSeq)의 줄을 모두 받는다. (줄들, 끝까지 받았는지)."""
    url = f"{config.DUR_API_URL.rstrip('/')}/{operation}"
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params = {
            "serviceKey": key or config.require_drug_api_key(),
            "type": "json",
            "pageNo": page,
            "numOfRows": PAGE_ROWS,
            "itemSeq": item_code,
        }
        data = (fetch or http_get_json)(url, params)
        got = raw_items(data)
        rows += got
        total = _total(data)
        if not got or len(got) < PAGE_ROWS or (total is not None and len(rows) >= total):
            return rows, True
    return rows, False


def parse_taboo(rows: list[dict[str, Any]], complete: bool = True) -> TabooList:
    partners: dict[str, TabooPartner] = {}
    for row in rows:
        code = _text(row, TABOO_PARTNER_CODE)
        if len(code) != 9 or not code.isdigit() or code in partners:
            continue
        partners[code] = TabooPartner(
            item_code=code,
            name=_text(row, TABOO_PARTNER_NAME),
            ingredient=_text(row, TABOO_PARTNER_INGREDIENT),
            reason=_text(row, TABOO_REASON),
        )
    return TabooList(partners=tuple(partners.values()), complete=complete)


def parse_efficacy(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """이 약의 효능군 이름들 (중복 없이, 응답 순서). 계열 이름이 있으면 '효능군(계열)'로 붙인다."""
    groups: list[str] = []
    for row in rows:
        group, series = _text(row, EFFICACY_GROUP), _text(row, EFFICACY_SERIES)
        if not group:
            continue
        label = f"{group}({series})" if series and series not in group else group
        if label not in groups:
            groups.append(label)
    return tuple(groups)


class DurApi:
    """dur.py에 넘기는 DUR 조회 (dur.DurLookup 모양). 같은 약은 한 번만 묻는다 (성공한 결과만 기억한다)."""

    def __init__(self, *, key: str | None = None, fetch: Fetch | None = None):
        self._key, self._fetch = key, fetch
        self._taboo: dict[str, TabooList] = {}
        self._efficacy: dict[str, tuple[str, ...]] = {}

    def taboo(self, item_code: str) -> TabooList:
        if item_code not in self._taboo:
            rows, complete = fetch_rows(TABOO_OPERATION, item_code, key=self._key, fetch=self._fetch)
            self._taboo[item_code] = parse_taboo(rows, complete)
        return self._taboo[item_code]

    def efficacy(self, item_code: str) -> tuple[str, ...]:
        if item_code not in self._efficacy:
            rows, _ = fetch_rows(EFFICACY_OPERATION, item_code, key=self._key, fetch=self._fetch)
            self._efficacy[item_code] = parse_efficacy(rows)
        return self._efficacy[item_code]


def default_dur_api(*, fetch: Fetch | None = None) -> DurApi | None:
    """.env에 DRUG_API_KEY가 있으면 DUR 조회, 없으면 None (병용금기 · 효능군 중복은 대조하지 않고 그렇다고 알린다)."""
    key = config.drug_api_key()
    return DurApi(key=key, fetch=fetch) if key else None

