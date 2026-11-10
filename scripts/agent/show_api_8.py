"""사전 8종의 의약품 API 응답을 눈으로 보고 파일로 남긴다 (5단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/show_api_8.py                 # 8종 요약을 보여 주고 응답을 var/api/ 에 저장한다
    python scripts/agent/show_api_8.py --out 폴더       # 저장할 폴더를 바꾼다
    python scripts/agent/show_api_8.py --no-save       # 저장하지 않고 요약만 본다

pytest가 아니다. 저장소 루트의 .env의 DRUG_API_KEY로 실제 API를 부른다. 키는 화면에도 파일에도 적지 않는다.
verify_dictionary.py가 "코드가 맞는가"를 가리는 스크립트라면, 이것은 "API가 실제로 무엇을 돌려주는가"를 보는 스크립트다.
  - 약마다 API 제품 이름·업체와 사전 코드와의 일치 여부를 한 줄씩 보여 준다.
  - 응답 원본을 <폴더>/<번호>_<품목기준코드>.json 으로 저장한다. 기본 폴더 var/api/ 는 git에 올라가지 않는다 (.gitignore의 var/).
    저장한 파일로 응답 필드를 확인하고, 시연 때 API가 막혀도 실제 응답이 어떤 모양인지 보여 줄 수 있다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/show_api_8.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.drug_api import Fetch, http_get_json, search_drugs  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.drug_match import CODE_FOUND, check_item_code  # noqa: E402
from agent.drug_names import search_term  # noqa: E402
from agent.errors import ConfigError, DrugApiError  # noqa: E402

DEFAULT_OUT = ROOT / "var" / "api"


def mask(text: str, secret: str) -> str:
    """파일에 적기 전에 키가 섞여 있으면 가린다 (서버가 요청을 그대로 되돌려 주는 경우를 대비한 이중 장치)."""
    return text.replace(secret, "***") if secret else text


def main(
    argv: list[str] | None = None,
    *,
    fetch: Fetch = http_get_json,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(description="사전 8종의 의약품 API 응답을 보여 주고 저장")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="저장할 폴더 (기본 var/api)")
    parser.add_argument("--no-save", action="store_true", help="저장하지 않는다")
    args = parser.parse_args(argv)

    out(f"주소: {config.DRUG_API_URL} | DRUG_API_KEY: {'설정됨' if config.drug_api_key() else '없음'}")
    try:
        key = config.require_drug_api_key()
        dictionary = load_dictionary()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    if not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)

    captured: dict[str, Any] = {}

    def capture(url: str, params: dict[str, Any]) -> Any:
        data = fetch(url, params)
        captured["data"] = data
        return data

    failed = 0
    found = 0
    for number, entry in enumerate(dictionary.entries):
        term = search_term(entry.name)
        captured.clear()
        try:
            products = search_drugs(term, key=key, fetch=capture)
        except DrugApiError as e:
            out(f"[{number}] {entry.name} ({entry.item_code}): 조회 실패({e.code}): {e}")
            failed += 1
            continue
        verdict, ranked = check_item_code(entry.item_code, entry.name, products)
        same = next((p for p in products if p.item_code == entry.item_code), None)
        shown = same or (ranked[0][1] if ranked else None)
        if verdict == CODE_FOUND:
            found += 1
        label = f"API '{shown.name}' [{shown.company or '업체 모름'}]" if shown else "API에서 찾은 제품 없음"
        out(f"[{number}] {entry.name} ({entry.item_code}) 검색어 '{term}' → 제품 {len(products)}개, {verdict}: {label}")
        if not args.no_save:
            path = args.out / f"{number:02d}_{entry.item_code}.json"
            record = {"dictionary": {"item_code": entry.item_code, "name": entry.name}, "search_term": term, "response": captured["data"]}
            path.write_text(mask(json.dumps(record, ensure_ascii=False, indent=2), key), encoding="utf-8")
            out(f"      저장: {path}")

    out(f"전체 {len(dictionary)}종 중 {found}종 일치" + (f", {failed}종 조회 실패" if failed else ""))
    if found + failed < len(dictionary):
        out("      일치하지 않은 약은 scripts/agent/verify_dictionary.py 로 비슷한 제품을 확인하세요")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
