"""식약처 의약품 제품 허가정보 API 조회 (5단계, 일정표 11/5, REQ-AG-03).

약품 사전에 없는 약을 이름으로 찾아 품목기준코드(ITEM_SEQ, 9자리) 후보를 돌려준다. 고르는 일은 drug_match.py가 한다.
이 파일은 가장자리 층이다: 바깥(네트워크)에 닿는 일만 하고, 어떤 제품이 맞는지는 판단하지 않는다.
검색어를 만드는 일(search_term)과 이름을 견주는 일(product_key)은 판단 층의 drug_names.py에 있다.

- 키는 .env의 DRUG_API_KEY (공공데이터포털 일반 인증키, Decoding). 환경변수로만 읽고
  화면·오류 문장·로그·파일 어디에도 적지 않는다 (REQ-NF-07). 서버가 돌려준 거절 사유에 섞여 있어도 가린다.
- 실패하면 DrugApiError(E_EXT_API)를 던진다. 서비스를 멈출지는 부른 쪽이 정한다 (drug_match는 사전만으로 계속한다).
- 서비스는 DrugPrdtPrmsnInfoService08 / getDrugPrdtPrmsnInq08 이다 (11/9 실제 키로 확인). 서비스 버전이 틀리면
  게이트웨이가 오류 문구 없이 HTTP 403을 돌려준다. 버전이 바뀌면 .env에 DRUG_API_URL=... 로 적는다.
- 응답 필드(ITEM_SEQ, ITEM_NAME, ENTP_NAME, CANCEL_NAME)와 모양은 확인한 것만 쓴다. 의심스러우면
  `python scripts/agent/lookup_drug.py 타이레놀정 --raw` 로 실제 응답과 맞는지 본다.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from agent import config
from agent.errors import DrugApiError

MAX_ROWS = 20   # 한 번에 받아 올 제품 수. 같은 이름의 회사별 제품이 여럿일 수 있다.

Fetch = Callable[[str, dict[str, Any]], Any]   # (url, 요청 인자) -> JSON으로 읽은 응답
ApiSearch = Callable[[str], list["ApiDrug"]]   # 검색어 -> 제품 목록 (drug_match에 넘기는 함수)

USER_AGENT = "pharmacy-dispensing-agent/1.0"   # 파이썬 기본 이름표(Python-urllib)를 막는 게이트웨이가 있어 바꾼다

_ITEM_CODE = re.compile(r"[0-9]{9}")           # [0-9]: \d는 전각 숫자 같은 다른 나라 숫자도 받는다

# 공공데이터포털이 거절할 때 돌려주는 사유 칸 (XML 또는 JSON)
_REASON_TAGS = re.compile(r"<(returnAuthMsg|errMsg|resultMsg|returnReasonCode)>\s*([^<]{1,120}?)\s*</\1>")
_TAG = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class ApiDrug:
    item_code: str           # 품목기준코드 (ITEM_SEQ)
    name: str                # 제품명 (ITEM_NAME)
    company: str | None      # 업체명 (ENTP_NAME)


# ------------------------------------------------------------ 서버의 거절 사유


def server_reason(raw: bytes | None, secret: str | None = None) -> str | None:
    """서버가 돌려준 거절 사유를 짧게. 키가 섞여 있으면 가린다. 읽을 게 없으면 None.

    예: b"<returnAuthMsg>SERVICE_ACCESS_DENIED_ERROR</returnAuthMsg>" -> "SERVICE_ACCESS_DENIED_ERROR"
    """
    text = (raw or b"")[:4000].decode("utf-8", errors="replace")
    found = [m.group(2) for m in _REASON_TAGS.finditer(text)]
    if not found:
        try:
            header = json.loads(text).get("response", {}).get("header", {})
            found = [str(header.get(k)) for k in ("resultCode", "resultMsg") if header.get(k)]
        except (ValueError, AttributeError):
            found = [" ".join(_TAG.sub(" ", text).split())[:120]]
    reason = " / ".join(dict.fromkeys(f for f in found if f.strip()))
    if secret:
        for form in {secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret)}:
            reason = reason.replace(form, "***")
    return reason or None


# ------------------------------------------------------------ 응답 읽기


def _items(data: Any) -> list[dict[str, Any]]:
    """공공데이터포털 응답에서 제품 목록을 꺼낸다. 0건·1건일 때 모양이 달라지는 경우까지 받는다."""
    if not isinstance(data, dict):
        raise DrugApiError("의약품 API 응답 형식이 올바르지 않습니다.")
    root = data.get("response", data)          # {"response": {"header", "body"}} 또는 {"header", "body"}
    header = root.get("header") if isinstance(root, dict) else None
    if isinstance(header, dict) and str(header.get("resultCode", "00")) not in ("00", "0"):
        raise DrugApiError(f"의약품 API가 오류를 돌려주었습니다 (resultCode {header.get('resultCode')}).")
    body = root.get("body") if isinstance(root, dict) else None
    if not isinstance(body, dict):
        raise DrugApiError("의약품 API 응답에 body가 없습니다.")
    items = body.get("items")
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    if items in (None, ""):
        items = []
    if not isinstance(items, list):
        raise DrugApiError("의약품 API 응답의 items 형식이 올바르지 않습니다.")
    return [i for i in items if isinstance(i, dict)]


def parse_products(data: Any) -> list[ApiDrug]:
    """응답에서 쓸 수 있는 제품만 골라 ApiDrug로. 코드가 9자리가 아니거나 이름이 없거나 취하·취소된 제품은 뺀다."""
    products: list[ApiDrug] = []
    seen: set[str] = set()
    for item in _items(data):
        code = str(item.get("ITEM_SEQ") or "").strip()
        name = str(item.get("ITEM_NAME") or "").strip()
        cancel = str(item.get("CANCEL_NAME") or "").strip()
        if not _ITEM_CODE.fullmatch(code) or not name or cancel not in ("", "정상") or code in seen:
            continue
        seen.add(code)
        company = str(item.get("ENTP_NAME") or "").strip() or None
        products.append(ApiDrug(item_code=code, name=name, company=company))
    return products


# ------------------------------------------------------------ 호출


def http_get_json(url: str, params: dict[str, Any]) -> Any:
    """GET 요청을 보내 JSON으로 읽는다. 실패하면 DrugApiError (키가 든 URL은 메시지에 넣지 않는다)."""
    secret = params.get("serviceKey")
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.DRUG_API_TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        try:
            reason = server_reason(e.read(), secret)
        except Exception:
            reason = None
        detail = f": {reason}" if reason else ""
        raise DrugApiError(f"의약품 API 호출에 실패했습니다 (HTTP {e.code}{detail}).", status=e.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DrugApiError(f"의약품 API에 연결하지 못했습니다 ({type(e).__name__}).") from None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # 공공데이터포털은 키 오류 등을 JSON이 아닌 XML로 돌려주기도 한다
        reason = server_reason(raw, secret)
        detail = f": {reason}" if reason else ""
        raise DrugApiError(f"의약품 API 응답을 JSON으로 읽을 수 없습니다{detail} (인증키·서비스 주소를 확인하세요).") from None


def search_drugs(term: str | None, *, fetch: Fetch | None = None, key: str | None = None) -> list[ApiDrug]:
    """검색어(함량 등을 뺀 이름, drug_names.search_term)로 API를 찾아 제품 목록을 돌려준다.

    검색어가 비면 부르지 않고 빈 목록. 키가 없으면 ConfigError. 호출·응답 문제는 DrugApiError.
    """
    term = (term or "").strip()
    if not term:
        return []
    params = {
        "serviceKey": key or config.require_drug_api_key(),
        "type": "json",
        "pageNo": 1,
        "numOfRows": MAX_ROWS,
        "item_name": term,
    }
    return parse_products((fetch or http_get_json)(config.DRUG_API_URL, params))


def default_api(*, fetch: Fetch | None = None) -> ApiSearch | None:
    """약품 매칭(drug_match.match_order)에 넘길 조회 함수. .env에 DRUG_API_KEY가 없으면 None (사전만 쓴다).

    키가 있을 때만 API를 쓰므로, 키가 없다고 매칭이 실패하지는 않는다.
    같은 검색어는 한 번만 묻는다: 성공한 결과를 이 함수가 살아 있는 동안 기억한다 (실패는 기억하지 않는다).
    여러 처방전에 같은 약이 나와도 API를 다시 부르지 않아 1분 기준(REQ-NF-01)과 호출 한도를 지킨다.
    """
    key = config.drug_api_key()
    if key is None:
        return None
    remembered: dict[str, list[ApiDrug]] = {}

    def search(term: str) -> list[ApiDrug]:
        if term not in remembered:
            remembered[term] = search_drugs(term, key=key, fetch=fetch)
        return list(remembered[term])

    return search
