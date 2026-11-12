"""처방약끼리 위험 신호(성분 중복 · 병용금기 · 효능군 중복)를 찾는다 (8 · 13단계, 일정표 11/10 · 11/17, REQ-AG-05).

    성분 중복    약품 사전의 '주성분' 칸으로 (8단계). 같은 성분이 서로 다른 처방약 둘 이상에 들어 있으면 중복이다.
                 시연 조합 A(타이레놀 · 게보린: 아세트아미노펜)와 B(트리스펜 · 아르펜: 이부프로펜)가 여기서 잡힌다.
    병용금기     DUR 품목정보의 병용금기 목록으로 (13단계). 한 약의 목록에 다른 처방약의 품목기준코드가 있으면 금기다.
                 어느 쪽 목록에서 찾아도 한 번만 센다.
    효능군 중복  DUR 품목정보의 효능군으로 (13단계). 서로 다른 처방약이 같은 효능군이면 중복이다.
                 이미 성분 중복으로 잡힌 약들끼리는 다시 알리지 않는다 (같은 약을 두 번 경고하지 않는다).

DUR 조회는 인자(DurLookup)로 받는다. 실제로는 dur_api.DurApi이고, 시험은 가짜를 넘긴다.
조회가 실패하면 멈추지 않고, 그 뒤로는 묻지 않으며, 대조하지 못했다는 사실을 unchecked에 남긴다.

이 파일은 사실만 찾는다. 그것이 어느 단계(주의 · 경고 · 위험)인지, 약사에게 어떤 문장으로 보일지는 risk.py가 정한다.
판단 층이다. 네트워크에 닿는 일은 하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from agent.drug_dictionary import DrugDictionary, ingredient_key
from agent.errors import ConfigError, DrugApiError

INGREDIENT_DUP = "ingredient_dup"        # DUR 유형 키: 성분 중복
COMBINATION_TABOO = "combination_taboo"  # DUR 유형 키: 병용금기
EFFICACY_DUP = "efficacy_dup"            # DUR 유형 키: 효능군 중복

# 대조하지 못한 이유
NO_ITEM_CODE = "no_item_code"            # 약을 정하지 못한 줄 (매칭 실패)
NO_INGREDIENTS = "no_ingredients"        # 사전에 없는 약이거나 주성분 칸이 비어 있다
DUR_FAILED = "dur_failed"                # DUR API를 쓰지 못했다 (이 약부터 끝까지)
TABOO_PARTIAL = "taboo_partial"          # 병용금기 목록이 길어 앞부분만 받았다


class DrugRef(Protocol):
    """대조에 쓰는 약품 줄. schema.DrugLine이 이 모양이다 (MCP 도구는 {item_code, name}만 받는다)."""

    @property
    def item_code(self) -> str | None: ...

    @property
    def name(self) -> str: ...


class TabooPartner(Protocol):
    """병용금기 상대 약 하나. dur_api.TabooPartner가 이 모양이다."""

    @property
    def item_code(self) -> str: ...

    @property
    def ingredient(self) -> str: ...

    @property
    def reason(self) -> str: ...


class TabooResult(Protocol):
    @property
    def partners(self) -> Sequence[TabooPartner]: ...

    @property
    def complete(self) -> bool: ...


class DurLookup(Protocol):
    """DUR 조회. 실패하면 DrugApiError(키가 없으면 ConfigError)."""

    def taboo(self, item_code: str) -> TabooResult: ...

    def efficacy(self, item_code: str) -> Sequence[str]: ...


@dataclass(frozen=True)
class DurHit:
    """찾아낸 위험 신호 하나 (단계는 아직 없다)."""

    type: str                            # DUR 유형 키 (INGREDIENT_DUP · COMBINATION_TABOO · EFFICACY_DUP)
    item_codes: tuple[str, ...]          # 관련된 약. 처방전에 적힌 순서
    names: tuple[str, ...]               # item_codes와 같은 순서의 약품 이름 (사전에 있으면 사전의 이름)
    ingredients: tuple[str, ...]         # 성분 중복: 겹치는 성분. 병용금기: 금기 성분. 효능군 중복: 겹치는 효능군
    detail: str = ""                     # 병용금기의 금기 사유 (DUR의 PROHBT_CONTENT). 없으면 빈 문자열


@dataclass(frozen=True)
class Unchecked:
    """성분 중복 대조에서 빠진 약. 빠졌다는 사실이 약사에게 보여야 '해당 없음'이 안전함으로 읽히지 않는다."""

    name: str                            # 약품 이름
    reason: str                          # NO_ITEM_CODE | NO_INGREDIENTS | DUR_FAILED | TABOO_PARTIAL
    detail: str = ""                     # DUR_FAILED: 실패 사유


@dataclass(frozen=True)
class DurReport:
    hits: tuple[DurHit, ...]
    unchecked: tuple[Unchecked, ...]
    dur_used: bool = False               # DUR 조회(병용금기 · 효능군 중복)를 했는가. 없었다면 risk.py가 그렇다고 알린다


def merge_reports(*reports: DurReport) -> DurReport:
    """여러 대조 결과를 하나로 (순서 유지, 같은 항목은 한 번만). DUR 조회를 하나라도 했으면 dur_used."""
    hits = tuple(dict.fromkeys(h for r in reports for h in r.hits))
    unchecked = tuple(dict.fromkeys(u for r in reports for u in r.unchecked))
    return DurReport(hits, unchecked, any(r.dur_used for r in reports))


def find_ingredient_duplicates(items: Sequence[DrugRef], dictionary: DrugDictionary) -> DurReport:
    """처방약 줄들에서 성분 중복을 찾는다.

    - 같은 품목이 두 줄에 나와도 한 약으로 센다 (같은 약끼리는 성분 중복이 아니다).
    - 품목기준코드가 없는 줄, 사전에 없는 약, 주성분 칸이 빈 약은 대조에서 빠지고 unchecked에 남는다.
    - 같은 약들이 둘 이상의 성분을 함께 겹치면 항목 하나에 성분을 모아 담는다.
    결과의 순서는 처방전에 적힌 순서(약) · 사전에 적힌 순서(성분)를 따라 늘 같다.
    """
    drugs: list[tuple[str, str, tuple[str, ...]]] = []     # (품목기준코드, 이름, 주성분) 중복 없이, 처방 순서
    unchecked: list[Unchecked] = []
    seen: set[str] = set()
    for line in items:
        code = line.item_code
        if code is None:
            unchecked.append(Unchecked(line.name, NO_ITEM_CODE))
            continue
        if code in seen:
            continue
        seen.add(code)
        entry = dictionary.by_item_code(code)
        if entry is None or not entry.ingredients:
            unchecked.append(Unchecked(entry.name if entry else line.name, NO_INGREDIENTS))
            continue
        drugs.append((code, entry.name, entry.ingredients))

    holders: dict[str, list[int]] = {}      # 성분 -> 그 성분이 든 약(drugs의 번호)
    spelled: dict[str, str] = {}            # 성분 -> 처음 본 표기
    for index, (_, _, ingredients) in enumerate(drugs):
        for ingredient in ingredients:
            key = ingredient_key(ingredient)
            holders.setdefault(key, []).append(index)
            spelled.setdefault(key, ingredient)

    groups: dict[tuple[int, ...], list[str]] = {}       # 같은 약들이 겹치는 성분을 모은다
    for key, indices in holders.items():
        if len(indices) >= 2:
            groups.setdefault(tuple(indices), []).append(spelled[key])

    hits = tuple(
        DurHit(
            type=INGREDIENT_DUP,
            item_codes=tuple(drugs[i][0] for i in indices),
            names=tuple(drugs[i][1] for i in indices),
            ingredients=tuple(ingredients),
        )
        for indices, ingredients in groups.items()
    )
    return DurReport(hits=hits, unchecked=tuple(unchecked))


# ------------------------------------------------------------ DUR 조회로 찾는 것 (13단계)


def _coded_drugs(items: Sequence[DrugRef], dictionary: DrugDictionary | None) -> list[tuple[str, str]]:
    """(품목기준코드, 이름) 중복 없이 처방 순서. 코드가 없는 줄은 빠진다 (성분 중복 대조가 이미 알린다)."""
    drugs: dict[str, str] = {}
    for line in items:
        if line.item_code and line.item_code not in drugs:
            entry = dictionary.by_item_code(line.item_code) if dictionary is not None else None
            drugs[line.item_code] = entry.name if entry else line.name
    return list(drugs.items())


class _Asking:
    """조회가 한 번 실패하면 그 뒤로는 묻지 않는다. 실패는 처음 실패한 약 이름과 함께 unchecked 하나로 남긴다."""

    def __init__(self, lookup: DurLookup):
        self.lookup, self.failed = lookup, None

    def ask(self, how, name: str, item_code: str):
        if self.failed is not None:
            return None
        try:
            return how(item_code)
        except (DrugApiError, ConfigError) as e:
            self.failed = Unchecked(name, DUR_FAILED, str(e))
            return None


def find_combination_taboos(items: Sequence[DrugRef], lookup: DurLookup,
                            dictionary: DrugDictionary | None = None) -> DurReport:
    """처방약끼리 병용금기를 찾는다. 약마다 병용금기 목록을 받아, 다른 처방약의 코드가 있으면 그 쌍이 금기다."""
    drugs = _coded_drugs(items, dictionary)
    asking = _Asking(lookup)
    partners: dict[str, dict[str, TabooPartner]] = {}
    unchecked: list[Unchecked] = []
    for code, name in drugs:
        found = asking.ask(lookup.taboo, name, code)
        if found is None:
            continue
        partners[code] = {p.item_code: p for p in found.partners}
        if not found.complete:
            unchecked.append(Unchecked(name, TABOO_PARTIAL))
    hits = []
    for i, (code_a, name_a) in enumerate(drugs):
        for code_b, name_b in drugs[i + 1:]:
            partner = partners.get(code_a, {}).get(code_b) or partners.get(code_b, {}).get(code_a)
            if partner is None:
                continue
            hits.append(DurHit(COMBINATION_TABOO, (code_a, code_b), (name_a, name_b),
                               tuple(x for x in (partner.ingredient,) if x), partner.reason))
    if asking.failed is not None:
        unchecked.append(asking.failed)
    return DurReport(tuple(hits), tuple(unchecked), dur_used=True)


def find_efficacy_duplicates(items: Sequence[DrugRef], lookup: DurLookup, dictionary: DrugDictionary | None = None,
                             *, already: Iterable[DurHit] = ()) -> DurReport:
    """처방약끼리 효능군 중복을 찾는다. already(성분 중복 등)로 이미 묶인 약들끼리는 다시 알리지 않는다."""
    drugs = _coded_drugs(items, dictionary)
    covered = [set(h.item_codes) for h in already]
    asking = _Asking(lookup)
    holders: dict[str, list[int]] = {}                   # 효능군 -> 그 효능군인 약(drugs의 번호)
    for index, (code, name) in enumerate(drugs):
        for group in asking.ask(lookup.efficacy, name, code) or ():
            if index not in holders.setdefault(group, []):
                holders[group].append(index)
    groups: dict[tuple[int, ...], list[str]] = {}
    for group, indices in holders.items():
        if len(indices) >= 2:
            groups.setdefault(tuple(indices), []).append(group)
    hits = []
    for indices, names in groups.items():
        codes = tuple(drugs[i][0] for i in indices)
        if any(set(codes) <= c for c in covered):
            continue
        hits.append(DurHit(EFFICACY_DUP, codes, tuple(drugs[i][1] for i in indices), tuple(names)))
    return DurReport(tuple(hits), (asking.failed,) if asking.failed else (), dur_used=True)


def check_interaction(items: Sequence[DrugRef], lookup: DurLookup, dictionary: DrugDictionary | None = None) -> DurReport:
    """MCP 도구 check_interaction이 감싸는 함수: 병용금기."""
    return find_combination_taboos(items, lookup, dictionary)


def check_duplicate(items: Sequence[DrugRef], dictionary: DrugDictionary, lookup: DurLookup | None = None) -> DurReport:
    """MCP 도구 check_duplicate가 감싸는 함수: 성분 중복(사전) + 효능군 중복(DUR, 조회가 있을 때)."""
    ingredient = find_ingredient_duplicates(items, dictionary)
    if lookup is None:
        return ingredient
    return merge_reports(ingredient, find_efficacy_duplicates(items, lookup, dictionary, already=ingredient.hits))


def find_all(items: Sequence[DrugRef], dictionary: DrugDictionary, lookup: DurLookup | None = None) -> DurReport:
    """처방약끼리 대조 전부. 조회가 없으면 성분 중복만 (dur_used=False). 결과 순서: 병용금기 → 성분 중복 → 효능군 중복."""
    duplicates = check_duplicate(items, dictionary, lookup)
    if lookup is None:
        return duplicates
    return combine_checks(check_interaction(items, lookup, dictionary), duplicates)


def combine_checks(taboos: DurReport, duplicates: DurReport) -> DurReport:
    """병용금기와 중복 대조를 합친다 (find_all과 MCP 클라이언트가 같은 규칙을 쓴다). DUR 실패는 한 번만 알린다."""
    report = merge_reports(taboos, duplicates)
    failed = [u for u in report.unchecked if u.reason == DUR_FAILED]
    unchecked = tuple(u for u in report.unchecked if u.reason != DUR_FAILED) + tuple(failed[:1])
    return DurReport(report.hits, unchecked, report.dur_used)


# ------------------------------------------------------------ MCP로 주고받는 모양 (JSON)


@dataclass(frozen=True)
class Ref:
    """{item_code, name}만 있는 약품 줄 (MCP 도구의 입력)."""

    item_code: str | None
    name: str


def report_to_json(report: DurReport) -> dict:
    return {
        "hits": [{"type": h.type, "item_codes": list(h.item_codes), "names": list(h.names),
                  "ingredients": list(h.ingredients), "detail": h.detail} for h in report.hits],
        "unchecked": [{"name": u.name, "reason": u.reason, "detail": u.detail} for u in report.unchecked],
        "dur_used": report.dur_used,
    }


def report_from_json(data: dict) -> DurReport:
    hits = tuple(DurHit(h["type"], tuple(h["item_codes"]), tuple(h["names"]), tuple(h["ingredients"]), h.get("detail", ""))
                 for h in data["hits"])
    unchecked = tuple(Unchecked(u["name"], u["reason"], u.get("detail", "")) for u in data["unchecked"])
    return DurReport(hits, unchecked, bool(data["dur_used"]))
