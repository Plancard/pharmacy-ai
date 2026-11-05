"""약품 매칭: 처방전의 약품명·코드 → 품목기준코드(item_code).

원칙 — 확실할 때만 확정한다.
- 정규화한 약품명이 카탈로그 품목 1개와 정확히 같고, 처방전 코드가 있다면 같은 품목을 가리킬 때만 확정.
- 그 밖의 경우(이름이 다름·같은 이름이 여러 개·코드와 이름이 서로 다른 약을 가리킴)는 확정하지 않고
  후보와 사유를 남겨 약사 확인으로 넘긴다.
- 이름이 비슷한 정도(fuzzy)는 후보로만 쓰고, 자동 확정에는 절대 쓰지 않는다.
- 약사가 후보에서 고른 코드(match_method="manual")는 다시 매칭해도 유지한다.
  단, 카탈로그에 실제로 있는 품목기준코드일 때만 인정한다 (보험코드는 인정하지 않는다).
"""
from __future__ import annotations

from typing import Optional

from app.domain.drug_catalog import Catalog, CatalogItem
from app.domain.drug_names import extract_strengths, normalize_name, similarity
from app.schemas import DispensingOrder, DrugLine, MatchCandidate

CANDIDATE_MIN_SCORE = 0.6          # 이 점수 미만의 비슷한 이름은 후보로 보여주지 않는다
MAX_CANDIDATES = 5
STRENGTH_MISMATCH_PENALTY = 0.5    # 이름은 비슷한데 함량이 다르면 점수를 반으로


def _candidate(item: CatalogItem, score: float, reason: str) -> MatchCandidate:
    return MatchCandidate(item_code=item.item_code, name=item.name, score=round(score, 3), reason=reason)


def _lookup_printed_code(code: Optional[str], catalog: Catalog) -> Optional[CatalogItem]:
    """처방전 코드는 보험코드일 가능성이 높아서 보험코드를 먼저 찾는다."""
    if not code:
        return None
    code = code.strip()
    return catalog.get_by_edi_code(code) or catalog.get_by_item_code(code)


def _fuzzy_candidates(name: str, catalog: Catalog) -> list[MatchCandidate]:
    key = normalize_name(name)
    if not key:
        return []
    query_strengths = extract_strengths(name)
    found: list[MatchCandidate] = []
    for item in catalog.items:
        score = similarity(key, normalize_name(item.name))
        reason = "이름이 비슷함"
        item_strengths = extract_strengths(item.name)
        if query_strengths and item_strengths and query_strengths != item_strengths:
            score *= STRENGTH_MISMATCH_PENALTY
            reason = "이름은 비슷하지만 함량이 다름"
        if score >= CANDIDATE_MIN_SCORE:
            found.append(_candidate(item, score, reason))
    return found


def _merge_candidates(mandatory: list[MatchCandidate], fuzzy: list[MatchCandidate]) -> list[MatchCandidate]:
    """같은 품목은 하나로 합치고(점수는 큰 값, 사유는 이어 붙임), 필수 후보를 먼저 채운 뒤 나머지로 채운다."""
    merged: dict[str, MatchCandidate] = {}

    def add(c: MatchCandidate) -> None:
        old = merged.get(c.item_code)
        if old is None:
            merged[c.item_code] = c
            return
        reasons = list(dict.fromkeys([*old.reason.split("; "), *c.reason.split("; ")]))
        merged[c.item_code] = old.model_copy(
            update={"score": max(old.score, c.score), "reason": "; ".join(r for r in reasons if r)}
        )

    for c in mandatory:
        add(c)
    for c in fuzzy:
        add(c)

    must = {c.item_code for c in mandatory}
    ranked = sorted(merged.values(), key=lambda c: (-c.score, c.item_code))
    keep = [c for c in ranked if c.item_code in must]
    for c in ranked:
        if len(keep) >= MAX_CANDIDATES:
            break
        if c.item_code not in must:
            keep.append(c)
    return sorted(keep, key=lambda c: (-c.score, c.item_code))


def match_line(line: DrugLine, catalog: Catalog) -> DrugLine:
    """약품 한 줄을 매칭한 새 DrugLine을 돌려준다. (매칭 결과 필드는 매번 새로 계산)"""
    warnings: list[str] = []

    if line.match_method == "manual":
        if line.item_code and catalog.get_by_item_code(line.item_code):
            return line.model_copy(update={"needs_pharmacist_check": False, "warnings": []})
        warnings.append("약사가 고른 코드가 카탈로그에 없어 처음부터 다시 매칭했습니다.")

    key = normalize_name(line.name)
    name_hits = catalog.find_by_name_key(key) if key else []
    code_hit = _lookup_printed_code(line.printed_code, catalog)

    # --- 확정: 이름이 품목 1개와 정확히 같고, 코드가 없거나 같은 품목을 가리킬 때
    if len(name_hits) == 1 and (code_hit is None or code_hit.item_code == name_hits[0].item_code):
        item = name_hits[0]
        if line.printed_code and code_hit is None:
            warnings.append(f"처방전 코드({line.printed_code.strip()})를 카탈로그에서 찾지 못해 약품명으로만 확인했습니다.")
        return line.model_copy(
            update={
                "item_code": item.item_code,
                "match_method": "exact" if line.name.strip() == item.name else "normalized",
                "match_candidates": [],
                "needs_pharmacist_check": False,
                "warnings": warnings,
            }
        )

    # --- 미확정: 이유를 남기고 후보를 붙여 약사 확인으로 넘긴다
    if len(name_hits) > 1:
        warnings.append("같은 이름의 품목이 여러 개라 하나로 정하지 못했습니다.")
    elif len(name_hits) == 1:
        warnings.append(
            f"처방전 코드와 약품명이 서로 다른 약을 가리킵니다 (코드: {code_hit.name}, 이름: {name_hits[0].name})."
        )
    elif code_hit is not None:
        warnings.append("처방전 코드가 가리키는 품목은 찾았지만 약품명과 일치하는 품목이 없습니다.")
    else:
        warnings.append("카탈로그에서 일치하는 약품을 찾지 못했습니다.")

    mandatory = [_candidate(item, 1.0, "약품명이 일치함") for item in name_hits]
    if code_hit is not None:
        mandatory.append(
            _candidate(code_hit, similarity(key, normalize_name(code_hit.name)), "처방전 코드가 가리키는 품목")
        )
    candidates = _merge_candidates(mandatory, _fuzzy_candidates(line.name, catalog))

    return line.model_copy(
        update={
            "item_code": None,
            "match_method": "none",
            "match_candidates": candidates,
            "needs_pharmacist_check": True,
            "warnings": warnings,
        }
    )


def match_order(order: DispensingOrder, catalog: Catalog) -> DispensingOrder:
    """모든 약품 줄을 매칭한다. order.warnings(구조화 경고)는 그대로 둔다."""
    return order.model_copy(update={"drugs": [match_line(d, catalog) for d in order.drugs]})


def unconfirmed_indices(order: DispensingOrder) -> list[int]:
    """아직 확정되지 않은 약품 줄의 번호(0부터)."""
    return [i for i, d in enumerate(order.drugs) if not d.is_confirmed]
