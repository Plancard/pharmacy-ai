"""약품 사전 8종의 품목기준코드를 식약처 의약품 API로 확인한다 (11/6) — 손으로 돌려 보는 스크립트.

    python verify_dictionary.py           # 8종 모두
    python verify_dictionary.py --raw     # 첫 약품의 API 응답 원본도 보여 준다 (필드 이름 확인용)

pytest가 아니다. .env의 DRUG_API_KEY로 실제 API를 부른다. 키는 화면에 찍지 않는다.
명세서 v0.3 2.6절의 품목기준코드 중 노란 칸(잠정)은 "의약품 정보 API로 확인해 채운다"고 되어 있다.
약마다 이름으로 API를 찾아, 사전의 item_code가 결과에 있는지(일치) / 없으면 비슷한 제품의 코드가 무엇인지 보여 준다.
사전은 고치지 않는다. 다른 코드가 나오면 팀이 확인한 뒤 drug_dictionary.json을 고친다.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from agent import config
from agent.drug_api import ApiDrug, DrugApiError, Fetch, http_get_json, product_key, search_drugs
from agent.drug_dictionary import DrugEntry, load_dictionary
from agent.drug_names import similarity
from agent.errors import ConfigError

MAX_SHOWN = 3   # 일치하지 않을 때 보여 줄 비슷한 제품 수


def check_entry(entry: DrugEntry, products: list[ApiDrug]) -> tuple[str, list[tuple[float, ApiDrug]]]:
    """('일치' | '불일치' | '없음', 이름이 비슷한 순서의 제품들)."""
    ranked = sorted(((similarity(entry.key, product_key(p.name)), p) for p in products), key=lambda t: -t[0])
    if not products:
        return "없음", []
    if any(p.item_code == entry.item_code for p in products):
        return "일치", ranked
    return "불일치", ranked


def failure_hints(statuses: set[int | None]) -> list[str]:
    """모두 실패했을 때 확인할 것. HTTP 상태 코드에 따라 다르다."""
    if 403 in statuses:
        return [
            "HTTP 403 = 이 키로 이 API를 쓸 권한이 없다는 뜻입니다. 공공데이터포털 마이페이지 → 데이터활용 → Open API →",
            "활용신청 현황에서 「식품의약품안전처_의약품 제품 허가정보」가 '승인'인지 확인하세요.",
            "다른 API(DUR, 낱알식별 등)만 신청했다면 이 API도 따로 활용신청해야 합니다. 승인 직후라면 1~2시간 뒤 다시 해 보세요.",
        ]
    if 401 in statuses:
        return [
            "HTTP 401 = 인증키를 받아 주지 않았다는 뜻입니다. .env의 DRUG_API_KEY가 'Decoding' 키인지(앞뒤 공백·따옴표 없이),",
            "발급 직후라면 1~2시간 뒤 다시 해 보세요.",
        ]
    return [
        "모두 실패했다면 키가 아직 활성화되지 않았거나(발급 후 1~2시간), 서비스 주소가 다를 수 있습니다.",
        ".env에 DRUG_API_URL=https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService06/getDrugPrdtPrmsnInq06 처럼 바꿔 보세요.",
    ]


def main(argv: list[str] | None = None, *, fetch: Fetch = http_get_json, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="약품 사전의 품목기준코드를 식약처 API로 확인")
    parser.add_argument("--raw", action="store_true", help="첫 약품의 API 응답 원본도 보여 준다")
    args = parser.parse_args(argv)

    out(f"주소: {config.DRUG_API_URL} | DRUG_API_KEY: {'설정됨' if config.drug_api_key() else '없음'}")
    try:
        key = config.require_drug_api_key()
        dictionary = load_dictionary()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    raw_shown = False

    def fetch_and_maybe_show(url: str, params: dict[str, Any]) -> Any:
        nonlocal raw_shown
        data = fetch(url, params)
        if args.raw and not raw_shown:
            raw_shown = True
            out("응답 원본 (앞부분 2000자):")
            out(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        return data

    results = []
    statuses: set[int | None] = set()
    for entry in dictionary.entries:
        try:
            products = search_drugs(entry.name, key=key, fetch=fetch_and_maybe_show)
        except DrugApiError as e:
            out(f"[실패] {entry.name} ({entry.item_code}): {e}")
            results.append("실패")
            statuses.add(getattr(e, "status", None))
            continue
        verdict, ranked = check_entry(entry, products)
        results.append(verdict)
        if verdict == "일치":
            name = next(p.name for p in products if p.item_code == entry.item_code)
            out(f"[일치] {entry.name} ({entry.item_code}) = API '{name}'")
        elif verdict == "없음":
            out(f"[없음] {entry.name} ({entry.item_code}): API에서 이 이름의 제품을 찾지 못했습니다")
        else:
            out(f"[불일치] {entry.name} ({entry.item_code}): API 결과에 이 코드가 없습니다. 비슷한 제품:")
            for score, p in ranked[:MAX_SHOWN]:
                out(f"    {p.item_code}  {p.name}  [{p.company or '업체 모름'}]  유사도 {score:.3f}")

    ok = results.count("일치")
    out(f"전체 {len(results)}종 중 {ok}종 일치" + ("" if ok == len(results) else " — 일치하지 않은 약은 팀이 확인한 뒤 사전을 고치세요"))
    if "실패" in results and ok == 0:
        for line in failure_hints(statuses):
            out("    " + line)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
