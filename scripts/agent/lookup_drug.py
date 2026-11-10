"""의약품 정보 API 실제 조회 확인 (5단계) — 손으로 돌려 보는 스크립트 (pytest가 아니다).

    python scripts/agent/lookup_drug.py 타이레놀정 500mg       # 이름으로 찾아 품목기준코드 후보를 본다
    python scripts/agent/lookup_drug.py 아르펜정 --raw         # API 응답 원본도 본다 (필드 이름 확인용)

저장소 루트의 .env에 DRUG_API_KEY(공공데이터포털 일반 인증키, Decoding)가 있어야 한다. 키는 화면에 찍지 않는다.
확인하는 것: 서비스 주소가 맞는지 / 응답 필드(ITEM_SEQ, ITEM_NAME, ENTP_NAME, CANCEL_NAME)가 예상과 같은지 /
조제 대상 약품의 품목기준코드가 무엇인지 (약품 사전을 채울 때 쓴다).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/lookup_drug.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.drug_api import MAX_ROWS, http_get_json, parse_products  # noqa: E402
from agent.drug_names import match_key, product_key, search_term, similarity  # noqa: E402
from agent.errors import ConfigError, DrugApiError  # noqa: E402

RAW_ITEMS = 2   # --raw에서 원본을 보여 줄 제품 수


def main(
    argv: list[str] | None = None,
    *,
    fetch: Callable[[str, dict[str, Any]], Any] = http_get_json,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(description="의약품 제품 허가정보 API로 품목기준코드 찾기")
    parser.add_argument("name", nargs="+", help="약품명 (띄어 써도 된다)")
    parser.add_argument("--raw", action="store_true", help="API 응답 원본도 보여 준다")
    args = parser.parse_args(argv)
    name = " ".join(args.name)

    key_state = "설정됨" if os.getenv("DRUG_API_KEY", "").strip() else "없음"
    out(f"주소: {config.DRUG_API_URL} | DRUG_API_KEY: {key_state}")
    term = search_term(name)
    out(f"약품명: {name} -> 검색어: {term or '(없음)'}")
    if not term:
        out("검색할 글자가 없습니다.")
        return 1
    try:
        key = config.require_drug_api_key()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    params = {"serviceKey": key, "type": "json", "pageNo": 1, "numOfRows": MAX_ROWS, "item_name": term}
    try:
        data = fetch(config.DRUG_API_URL, params)
    except DrugApiError as e:
        out(f"조회 실패({e.code}): {e}")
        return 1

    if args.raw:
        out("응답 원본 (앞부분):")
        out(json.dumps(_shorten(data), ensure_ascii=False, indent=2))

    try:
        products = parse_products(data)
    except DrugApiError as e:
        out(f"응답을 읽지 못했습니다: {e}")
        return 1
    if not products:
        out("찾은 제품이 없습니다. (이름이 틀렸거나, 필드 이름이 예상과 다를 수 있습니다. --raw로 확인하세요)")
        return 1

    key_of_name = match_key(name)
    scored = sorted(((similarity(key_of_name, product_key(p.name)), p) for p in products), key=lambda t: -t[0])
    out(f"찾은 제품 {len(products)}개 (이름이 비슷한 순서)")
    for score, p in scored:
        out(f"    {p.item_code}  {p.name}  [{p.company or '업체 모름'}]  유사도 {score:.3f}")
    return 0


def _shorten(data: Any) -> Any:
    """원본에서 제품 목록은 앞의 몇 개만 남긴다 (나머지는 개수만)."""
    try:
        body = data.get("response", data)["body"]
        items = body.get("items")
        holder, field = (items, "item") if isinstance(items, dict) and "item" in items else (body, "items")
        listed = holder[field]
        if isinstance(listed, list) and len(listed) > RAW_ITEMS:
            data = json.loads(json.dumps(data))
            body = data.get("response", data)["body"]
            holder = body["items"] if field == "item" else body
            holder[field] = listed[:RAW_ITEMS] + [f"... 외 {len(listed) - RAW_ITEMS}개"]
    except (AttributeError, KeyError, TypeError):
        pass
    return data


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
