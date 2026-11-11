"""DUR 품목정보 API 응답을 눈으로 보고, 병용금기 시험 조합이 검출되는지 본다 (13단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/show_dur.py              # 사전 8종: 병용금기 상대 수 · 효능군, 응답 필드 이름. 응답을 var/api/ 에 저장
    python scripts/agent/show_dur.py --cases      # 공개 병용금기 조합 5건(agent/resources/dur_cases.json)이 검출되는지
    python scripts/agent/show_dur.py --no-save    # 저장하지 않는다

pytest가 아니다. 저장소 루트의 .env의 DRUG_API_KEY로 실제 DUR API(DURPrdlstInfoService03)를 부른다.
키는 화면에도 파일에도 적지 않는다. 오퍼레이션 이름이나 응답 필드가 다르면 agent/dur_api.py 위쪽 상수를 고친다.
  - 응답 원본을 <폴더>/dur_<번호>_<오퍼레이션>.json 으로 저장한다 (기본 var/api/, git에 올라가지 않는다).
  - '필드' 줄에 실제 응답의 필드 이름이 나온다. dur_api.py가 쓰는 필드가 없으면 '없음'으로 알린다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import config  # noqa: E402
from agent import dur_api  # noqa: E402
from agent.drug_api import Fetch, http_get_json, search_drugs  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.drug_names import search_term  # noqa: E402
from agent.dur import COMBINATION_TABOO, Ref, find_combination_taboos  # noqa: E402
from agent.errors import ConfigError, DrugApiError  # noqa: E402

DEFAULT_OUT = ROOT / "var" / "api"
CASES = config.RESOURCES_DIR / "dur_cases.json"
USED_FIELDS = {
    dur_api.TABOO_OPERATION: (dur_api.TABOO_PARTNER_CODE, dur_api.TABOO_PARTNER_NAME, dur_api.TABOO_PARTNER_INGREDIENT,
                              dur_api.TABOO_REASON),
    dur_api.EFFICACY_OPERATION: (dur_api.EFFICACY_GROUP, dur_api.EFFICACY_SERIES),
}
SHOWN_PARTNERS = 3


def mask(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def show_dictionary(key: str, fetch: Fetch, out: Callable[[str], None], save_to: Path | None) -> int:
    """사전 8종. 실패한 약의 수를 돌려준다."""
    failed = 0
    fields_shown: set[str] = set()
    for number, entry in enumerate(load_dictionary().entries):
        line = f"[{number}] {entry.name} ({entry.item_code})"
        for operation in (dur_api.TABOO_OPERATION, dur_api.EFFICACY_OPERATION):
            pages: list[Any] = []

            def capture(url: str, params: dict[str, Any]) -> Any:
                data = fetch(url, params)
                pages.append(data)
                return data

            try:
                rows, complete = dur_api.fetch_rows(operation, entry.item_code, key=key, fetch=capture)
            except DrugApiError as e:
                out(f"{line} {operation}: 조회 실패: {e}")
                failed += 1
                continue
            if operation == dur_api.TABOO_OPERATION:
                taboo = dur_api.parse_taboo(rows, complete)
                more = "" if complete else f" (앞 {len(rows)}줄만 받음)"
                examples = ", ".join(f"{p.name}[{p.ingredient}]" for p in taboo.partners[:SHOWN_PARTNERS])
                out(f"{line} 병용금기 상대 {len(taboo.partners)}개{more}" + (f": {examples}…" if examples else ""))
            else:
                groups = dur_api.parse_efficacy(rows)
                out(f"{line} 효능군: {', '.join(groups) if groups else '없음'} (줄 {len(rows)}개)")
            if rows and operation not in fields_shown:
                fields_shown.add(operation)
                names = sorted(rows[0])
                missing = [f for f in USED_FIELDS[operation] if f not in names]
                out(f"      필드({operation}): {', '.join(names)}")
                out(f"      dur_api.py가 쓰는 필드 중 없는 것: {', '.join(missing) if missing else '없음 (모두 있음)'}")
            if save_to is not None and pages:
                path = save_to / f"dur_{number:02d}_{operation}.json"
                record = {"item_code": entry.item_code, "name": entry.name, "operation": operation, "pages": pages}
                path.write_text(mask(json.dumps(record, ensure_ascii=False, indent=2), key), encoding="utf-8")
    return failed


def resolve(drug: str, key: str, fetch: Fetch) -> tuple[str, str] | None:
    """품목기준코드는 그대로, 이름이면 의약품 API에서 찾아 첫 제품. (코드, 이름)."""
    if drug.isdigit() and len(drug) == 9:
        entry = load_dictionary().by_item_code(drug)
        return drug, entry.name if entry else drug
    products = search_drugs(search_term(drug), key=key, fetch=fetch)
    return (products[0].item_code, products[0].name) if products else None


def check_cases(key: str, fetch: Fetch, out: Callable[[str], None]) -> int:
    """병용금기 시험 조합. 검출하지 못한 조합의 수를 돌려준다."""
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    lookup = dur_api.DurApi(key=key, fetch=fetch)
    missed = 0
    for case in cases:
        label = f"[{case['id']}] {case['drug_label']} + {case['partner_ingredient']}"
        try:
            found = resolve(case["drug"], key, fetch)
            if found is None:
                out(f"{label}: 의약품 API에서 '{case['drug']}' 제품을 찾지 못했습니다")
                missed += 1
                continue
            code, name = found
            partners = [p for p in lookup.taboo(code).partners if case["partner_ingredient"] in p.ingredient + p.name]
            if not partners:
                out(f"{label}: {name}({code})의 병용금기 목록에 '{case['partner_ingredient']}'가 없습니다")
                missed += 1
                continue
            partner = partners[0]
            report = find_combination_taboos([Ref(code, name), Ref(partner.item_code, partner.name)], lookup)
        except DrugApiError as e:
            out(f"{label}: 조회 실패: {e}")
            missed += 1
            continue
        hit = next((h for h in report.hits if h.type == COMBINATION_TABOO), None)
        if hit is None:
            out(f"{label}: 검출 못 함 ({name} + {partner.name})")
            missed += 1
        else:
            reason = f" 사유: {hit.detail}" if hit.detail else ""
            out(f"{label}: 검출 ({name}({code}) + {partner.name}({partner.item_code})){reason}")
    out(f"병용금기 시험 조합 {len(cases)}건 중 {len(cases) - missed}건 검출")
    return missed


def main(argv: list[str] | None = None, *, fetch: Fetch = http_get_json, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="DUR 품목정보 API 응답 보기 · 병용금기 시험 조합 확인")
    parser.add_argument("--cases", action="store_true", help="공개 병용금기 조합(dur_cases.json)이 검출되는지 본다")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="저장할 폴더 (기본 var/api)")
    parser.add_argument("--no-save", action="store_true", help="저장하지 않는다")
    args = parser.parse_args(argv)

    out(f"주소: {config.DUR_API_URL} | DRUG_API_KEY: {'설정됨' if config.drug_api_key() else '없음'}")
    try:
        key = config.require_drug_api_key()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1
    if args.cases:
        return 0 if check_cases(key, fetch, out) == 0 else 1
    save_to = None
    if not args.no_save:
        save_to = args.out
        save_to.mkdir(parents=True, exist_ok=True)
    failed = show_dictionary(key, fetch, out, save_to)
    if save_to is not None:
        out(f"응답 저장: {save_to}")
    if failed:
        out("조회가 실패했다면: 공공데이터포털에서 「의약품안전사용서비스(DUR)품목정보」 활용신청이 '승인'인지,")
        out("HTTP 403이면 활용신청 상세의 End Point 버전이 위 '주소'와 같은지 확인하세요 (다르면 .env에 DUR_API_URL=...).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
