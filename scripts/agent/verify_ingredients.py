"""약품 사전의 주성분을 저장해 둔 의약품 API 응답과 견준다 (8단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/show_api_8.py                  # 먼저: 사전 8종의 API 응답을 var/api/ 에 저장한다 (키 필요)
    python scripts/agent/verify_ingredients.py          # 저장한 응답으로 사전의 주성분을 확인한다
    python scripts/agent/verify_ingredients.py --dir 폴더 # 응답을 저장한 폴더를 바꾼다

pytest가 아니다. 네트워크도 키도 쓰지 않고, show_api_8.py가 저장한 파일만 읽는다. 사전은 고치지 않는다.
약마다 응답에서 사전의 품목기준코드와 같은 제품을 찾아 그 제품의 주성분 칸(ITEM_INGR_NAME)과 사전의 주성분을 견준다.
API는 성분을 영어로 '/'로 이어 준다 (예: Acetaminophen/Anhydrous Caffeine/Isopropylantipyrine, 11/11 실제 응답으로 확인).
사전은 약사에게 보일 한글 이름을 쓰므로, 아래 대응표(ENGLISH_NAMES)로 한글 이름을 영어로 바꿔 견준다.
  - 일치: 사전의 성분과 API의 성분이 빠짐없이 같다.
  - 일부 일치 / 불일치: 사전에만 있는 성분, API에만 있는 성분을 보여 준다. 팀이 확인한 뒤 drug_dictionary.json을 고친다.
  - 대응표에 없는 성분은 '영어 이름 모름'으로 나온다. 대응표에 더한 뒤 다시 돌린다.
  - 사전의 주성분이 비어 있으면 API 값을 보여 준다. 팀이 확인해 사전에 적는다.
응답에 ITEM_INGR_NAME 칸이 없으면(서비스 버전이 바뀐 경우) 성분으로 보이는 칸과 제품명에서 한글 이름을 찾아본다.
모든 약이 '일치'일 때만 0으로 끝난다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/verify_ingredients.py` 로 돌려도 agent 패키지를 찾게 한다

from agent.drug_api import raw_items  # noqa: E402
from agent.drug_dictionary import DrugEntry, ingredient_key, load_dictionary  # noqa: E402
from agent.errors import ConfigError, DrugApiError  # noqa: E402

DEFAULT_DIR = ROOT / "var" / "api"       # show_api_8.py의 기본 저장 폴더. git에 올라가지 않는다
API_FIELD = "ITEM_INGR_NAME"          # 주성분 칸. 영어 이름을 '/'로 이어 준다
INGREDIENT_FIELD_HINTS = ("INGR", "MATERIAL", "MAIN")

# 사전의 한글 성분 이름 -> API의 영어 이름 (11/11 실제 응답의 표기). 사전에 성분을 더하면 여기도 더한다.
ENGLISH_NAMES = {
    "아세트아미노펜": "Acetaminophen",
    "이소프로필안티피린": "Isopropylantipyrine",
    "카페인무수물": "Anhydrous Caffeine",
    "이부프로펜": "Ibuprofen",
    "L-아르기닌": "Arginine",
    "배농산급탕연조엑스": "Baenongsangeubtang Soft Extract",
    "브로멜라인": "Bromelain",
    "결정트립신": "Crystallized Trypsin",
    "아세틸시스테인": "Acetylcysteine",
    # 아이월드요비신정 (한약재 14종)
    "목통": "Akebia Stem",
    "택사": "Alisma Rhizome",
    "당귀": "Angelica Gigas Root",
    "육계": "Cinnamon Bark",
    "마황": "Ephedra Herb",
    "연교": "Forsythia Fruit",
    "치자": "Gardenia Fruit",
    "용담": "Gentian Root and Rhizome",
    "감초": "Licorice",
    "차전자": "Plantago Seed",
    "생강": "Raw Ginger",
    "지황": "Rehmannia Root",
    "대황": "Rhubarb",
    "황금": "Scutellaria Root",
}
_ENGLISH_BY_KEY = {ingredient_key(korean): english for korean, english in ENGLISH_NAMES.items()}
_BRACKETS = re.compile(r"\([^)]*\)")     # 'Soft Extract (4→1)' 같은 덧붙임은 빼고 견준다
UNKNOWN_ENGLISH = "(영어 이름 모름)"
SHOWN_CHARS = 400        # 성분 칸의 값을 이만큼만 보여 준다 (원료성분 칸은 길다. 한약재 14종 목록은 다 보이게)

MATCH, PARTIAL, DIFFERENT, EMPTY, NO_PRODUCT, NO_FILE, UNREADABLE = (
    "일치", "일부 일치", "불일치", "사전 주성분 비어 있음", "응답에 이 제품 없음", "저장된 응답 없음", "응답 파일을 읽을 수 없음",
)


def ingredient_fields(item: dict[str, Any]) -> dict[str, str]:
    """제품 한 건에서 성분으로 보이는 칸(이름에 INGR · MATERIAL · MAIN이 든 칸)의 값. 값이 빈 칸은 뺀다."""
    return {
        name: " ".join(str(value).split())
        for name, value in item.items()
        if any(hint in name.upper() for hint in INGREDIENT_FIELD_HINTS) and value not in (None, "") and str(value).strip()
    }


def english_key(name: str) -> str:
    return ingredient_key(_BRACKETS.sub("", name))


def api_ingredients(item: dict[str, Any]) -> list[str] | None:
    """제품 한 건의 주성분 칸(ITEM_INGR_NAME)을 성분 이름 목록으로. 칸이 없거나 비었으면 None. 같은 이름이 두 번 와도 한 번만."""
    value = item.get(API_FIELD)
    if not isinstance(value, str) or not value.strip():
        return None
    names: dict[str, str] = {}
    for part in value.split("/"):
        if part.strip():
            names.setdefault(english_key(part), " ".join(part.split()))
    return list(names.values())


def judge(entry: DrugEntry, item: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """(판정, 사전에만 있는 성분, API에만 있는 성분)."""
    if not entry.ingredients:
        return EMPTY, [], []
    api_names = api_ingredients(item)
    if api_names is None:
        # 주성분 칸이 없는 응답: 성분으로 보이는 칸과 제품명을 이어 붙인 글에서 사전의 한글 이름을 찾는다
        haystack = ingredient_key(" ".join([*ingredient_fields(item).values(), str(item.get("ITEM_NAME") or "")]))
        missing = [name for name in entry.ingredients if ingredient_key(name) not in haystack]
        extra: list[str] = []
    else:
        api_keys = {english_key(name) for name in api_names}
        wanted: set[str] = set()
        missing = []
        for name in entry.ingredients:
            english = _ENGLISH_BY_KEY.get(ingredient_key(name))
            if english is None:
                missing.append(f"{name} {UNKNOWN_ENGLISH}")
            elif english_key(english) not in api_keys:
                missing.append(name)
            else:
                wanted.add(english_key(english))
        extra = [name for name in api_names if english_key(name) not in wanted]
    if not missing and not extra:
        return MATCH, [], []
    return (DIFFERENT if len(missing) == len(entry.ingredients) else PARTIAL), missing, extra


def clip(text: str) -> str:
    return text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS] + "…"


def read_item(path: Path, entry: DrugEntry) -> tuple[str | None, dict[str, Any] | None]:
    """저장한 응답 파일에서 사전의 품목기준코드와 같은 제품 한 건을 꺼낸다. 꺼냈으면 (None, 제품), 못 꺼냈으면 (이유, None)."""
    if not path.exists():
        return NO_FILE, None
    try:
        items = raw_items(json.loads(path.read_text(encoding="utf-8")).get("response"))
    except (OSError, ValueError, AttributeError, DrugApiError):
        return UNREADABLE, None
    item = next((i for i in items if str(i.get("ITEM_SEQ") or "").strip() == entry.item_code), None)
    return (None, item) if item is not None else (NO_PRODUCT, None)


def main(argv: list[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="약품 사전의 주성분을 저장해 둔 의약품 API 응답과 견준다")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="show_api_8.py가 응답을 저장한 폴더 (기본 var/api)")
    args = parser.parse_args(argv)

    try:
        dictionary = load_dictionary()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    matched = 0
    fields_shown = False
    for number, entry in enumerate(dictionary.entries):
        out(f"[{number}] {entry.name} ({entry.item_code})")
        out(f"      사전 주성분: {', '.join(entry.ingredients) if entry.ingredients else '(비어 있음)'}")
        status, item = read_item(args.dir / f"{number:02d}_{entry.item_code}.json", entry)
        if item is None:
            out(f"      판정: {status}" + (f" → 먼저 python scripts/agent/show_api_8.py 를 돌리세요 ({args.dir})" if status == NO_FILE else ""))
            continue
        out(f"      API 제품명: {item.get('ITEM_NAME') or '-'}")
        found = ingredient_fields(item)
        for name, value in found.items():
            out(f"      API {name}: {clip(value)}")
        if not found:
            out("      API 성분으로 보이는 칸이 없습니다 (아래 응답 필드 이름을 확인하세요)")
        if not fields_shown:
            out(f"      응답 필드: {', '.join(item)}")
            fields_shown = True
        verdict, missing, extra = judge(entry, item)
        if verdict == EMPTY:
            out(f"      판정: {verdict} → 위 API 값을 팀이 확인한 뒤 drug_dictionary.json의 ingredients에 적으세요")
        elif verdict == MATCH:
            matched += 1
            out(f"      판정: {verdict}")
        else:
            notes = []
            if missing:
                notes.append(f"사전에만 있는 성분: {', '.join(missing)}")
            if extra:
                notes.append(f"API에만 있는 성분: {', '.join(extra)}")
            out(f"      판정: {verdict} → {' | '.join(notes)}")

    out(f"전체 {len(dictionary)}종 중 {matched}종 일치")
    return 0 if matched == len(dictionary) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
