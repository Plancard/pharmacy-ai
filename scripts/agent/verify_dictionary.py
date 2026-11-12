"""약품 사전 8종의 품목기준코드를 식약처 의약품 API로 확인한다 (5단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/verify_dictionary.py           # 8종 모두
    python scripts/agent/verify_dictionary.py --raw     # 첫 약품의 API 응답 원본도 보여 준다 (필드 이름 확인용)

pytest가 아니다. 저장소 루트의 .env의 DRUG_API_KEY로 실제 API를 부른다. 키는 화면에 찍지 않는다.
사전을 고칠 때, 배포 전에 돌린다. 약마다 이름으로 API를 찾아, 사전의 item_code가 결과에 있는지(일치) /
없으면 비슷한 제품의 코드가 무엇인지 보여 준다. 사전은 고치지 않는다.
다른 코드가 나오면 팀이 확인한 뒤 agent/resources/drug_dictionary.json을 고친다.
(아이월드요비신정 416.7mg의 코드 200713170은 명세서 값이고 아직 이 스크립트로 확인하지 않았다.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/verify_dictionary.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.drug_api import Fetch, http_get_json, search_drugs  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.drug_match import CODE_DIFFERENT, CODE_FOUND, CODE_NOT_FOUND, check_item_code  # noqa: E402
from agent.drug_names import search_term  # noqa: E402
from agent.errors import ConfigError, DrugApiError  # noqa: E402

MAX_SHOWN = 3   # 일치하지 않을 때 보여 줄 비슷한 제품 수
FAILED = "실패"


def failure_hints(statuses: set[int | None]) -> list[str]:
    """모두 실패했을 때 확인할 것. HTTP 상태 코드에 따라 다르다."""
    if 403 in statuses:
        return [
            "HTTP 403 = 이 키로 이 주소를 쓸 권한이 없다는 뜻입니다. 두 가지를 확인하세요.",
            "1) 활용신청 상세의 End Point의 버전(…Service08 등)이 위 '주소'와 같은지. 다르면 .env에 DRUG_API_URL=<End Point> 로 적습니다.",
            "2) 공공데이터포털 마이페이지 → 데이터활용 → Open API → 활용신청 현황에서 「식품의약품안전처_의약품 제품 허가정보」의 상태가 '승인'인지.",
            "   다른 API(DUR, 낱알식별 등)만 신청했다면 이 API도 따로 활용신청해야 합니다. 승인 직후라면 1~2시간 뒤 다시 해 보세요.",
        ]
    if 401 in statuses:
        return [
            "HTTP 401 = 인증키를 받아 주지 않았다는 뜻입니다. .env의 DRUG_API_KEY가 'Decoding' 키인지(앞뒤 공백·따옴표 없이),",
            "발급 직후라면 1~2시간 뒤 다시 해 보세요.",
        ]
    return [
        "모두 실패했다면 키가 아직 활성화되지 않았거나(발급 후 1~2시간), 서비스 주소가 다를 수 있습니다.",
        "활용신청 상세의 End Point를 .env에 DRUG_API_URL=<End Point> 로 적어 보세요 (현재 주소는 위 '주소' 줄).",
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
            products = search_drugs(search_term(entry.name), key=key, fetch=fetch_and_maybe_show)
        except DrugApiError as e:
            out(f"[{FAILED}] {entry.name} ({entry.item_code}): {e}")
            results.append(FAILED)
            statuses.add(e.status)
            continue
        verdict, ranked = check_item_code(entry.item_code, entry.name, products)
        results.append(verdict)
        if verdict == CODE_FOUND:
            name = next(p.name for p in products if p.item_code == entry.item_code)
            out(f"[{CODE_FOUND}] {entry.name} ({entry.item_code}) = API '{name}'")
        elif verdict == CODE_NOT_FOUND:
            out(f"[{CODE_NOT_FOUND}] {entry.name} ({entry.item_code}): API에서 이 이름의 제품을 찾지 못했습니다")
        else:
            out(f"[{CODE_DIFFERENT}] {entry.name} ({entry.item_code}): API 결과에 이 코드가 없습니다. 비슷한 제품:")
            for score, p in ranked[:MAX_SHOWN]:
                out(f"    {p.item_code}  {p.name}  [{p.company or '업체 모름'}]  유사도 {score:.3f}")

    ok = results.count(CODE_FOUND)
    out(f"전체 {len(results)}종 중 {ok}종 일치" + ("" if ok == len(results) else ". 일치하지 않은 약은 팀이 확인한 뒤 사전을 고치세요"))
    if FAILED in results and ok == 0:
        for line in failure_hints(statuses):
            out("    " + line)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
