"""처방약끼리 성분 중복을 찾는다 (8단계, 일정표 11/10, REQ-AG-05).

약품 사전의 '주성분' 칸만으로 판정한다 (명세서 2.6절). 네트워크를 쓰지 않는다.
같은 성분이 서로 다른 처방약 둘 이상에 들어 있으면 성분 중복이다. 시연 조합 A(타이레놀 · 게보린: 아세트아미노펜)와
B(트리스펜 · 아르펜: 이부프로펜)가 여기서 잡힌다.

병용금기와 효능군 중복은 DUR API가 필요해 13단계에서 이 파일에 더한다. 그때도 결과는 DurHit 목록이다.

이 파일은 사실만 찾는다. 그것이 어느 단계(주의 · 경고 · 위험)인지, 약사에게 어떤 문장으로 보일지는 risk.py가 정한다.

판단 층이다. 사전을 읽을 뿐 네트워크를 쓰지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.drug_dictionary import DrugDictionary, ingredient_key
from agent.schema import DrugLine

INGREDIENT_DUP = "ingredient_dup"        # DUR 유형 키: 성분 중복

# 대조하지 못한 이유
NO_ITEM_CODE = "no_item_code"            # 약을 정하지 못한 줄 (매칭 실패)
NO_INGREDIENTS = "no_ingredients"        # 사전에 없는 약이거나 주성분 칸이 비어 있다


@dataclass(frozen=True)
class DurHit:
    """찾아낸 위험 신호 하나 (단계는 아직 없다)."""

    type: str                            # DUR 유형 키 (지금은 INGREDIENT_DUP뿐)
    item_codes: tuple[str, ...]          # 관련된 약. 처방전에 적힌 순서
    names: tuple[str, ...]               # item_codes와 같은 순서의 약품 이름 (사전의 이름)
    ingredients: tuple[str, ...]         # 겹치는 성분 (사전의 표기)


@dataclass(frozen=True)
class Unchecked:
    """성분 중복 대조에서 빠진 약. 빠졌다는 사실이 약사에게 보여야 '해당 없음'이 안전함으로 읽히지 않는다."""

    name: str                            # 처방전에 적힌 약품 이름
    reason: str                          # NO_ITEM_CODE | NO_INGREDIENTS


@dataclass(frozen=True)
class DurReport:
    hits: tuple[DurHit, ...]
    unchecked: tuple[Unchecked, ...]


def find_ingredient_duplicates(items: Sequence[DrugLine], dictionary: DrugDictionary) -> DurReport:
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
