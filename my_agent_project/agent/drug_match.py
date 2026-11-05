"""약품 매칭 (11/5-③, REQ-AG-03).

조제지시서의 약품 줄마다 약품 사전의 품목기준코드(item_code)를 정하고,
약사가 HITL ①에서 꼭 봐야 할 줄에 표시(needs_pharmacist_check)와 이유(check_reasons)를 붙인다.

순서 (팀 합의: 코드 먼저, 이름으로 확인)
1. 처방전에 인쇄된 코드가 사전에 있으면 그 약으로 정한다 (match_method="code").
   이름이 그 약과 충분히 비슷하지 않거나 함량이 다르면 확인 표시를 붙인다.
2. 코드가 없거나 사전에 없으면 이름으로 찾는다. 1등이 아주 비슷하고 2등과 차이가 커야만
   정하고(match_method="name"), 이름으로 정한 줄은 언제나 확인 표시를 붙인다.
3. 사전에서 못 정했고 의약품 API 조회 함수(api)를 받았으면, API에서 이름으로 찾아 같은 기준으로
   정한다(match_method="api", 품목기준코드 ITEM_SEQ). 이것도 언제나 확인 표시를 붙인다.
4. 그래도 못 정하면 item_code는 비워 두고(match_method="none") 비슷한 후보를 보여 준다.
그 밖에 OCR 신뢰도가 낮은 줄, 투약 일수가 트레이 일수보다 긴 줄에도 확인 표시를 붙인다.

기준값은 실제 샘플 5장에서 잰 값이다 (자모 단위 유사도):
OCR이 읽은 조제 대상 약 이름 vs 사전 0.81~1.0 / 대상이 아닌 약 vs 사전 최대 0.57 / 사전 약끼리 최대 0.54.
"""
from typing import Callable

from agent.drug_api import ApiDrug, DrugApiError, product_key
from agent.drug_dictionary import DrugDictionary, DrugEntry
from agent.drug_names import extract_strengths, match_key, similarity
from agent.schema import TRAY_DAYS, DispensingOrder, DrugLine, MatchCandidate

ApiSearch = Callable[[str], list[ApiDrug]]   # 약품명 -> API 제품 목록 (agent.drug_api.search_drugs)

NAME_MATCH_MIN = 0.7      # 코드로 찾은 약과 이름이 이 이상 비슷해야 확인 없이 넘어간다
NAME_ONLY_MIN = 0.9       # 코드 없이 이름만으로 약을 정하려면 1등이 이 이상이어야 한다
NAME_ONLY_MARGIN = 0.1    # ... 그리고 2등보다 이만큼 이상 앞서야 한다
CANDIDATE_MIN = 0.6       # 후보로 보여 줄 최소 유사도
MAX_CANDIDATES = 3        # 후보는 많아야 3개
LOW_CONFIDENCE = 0.5      # OCR 신뢰도가 이보다 낮으면 확인 표시


def strengths_conflict(a: str, b: str) -> bool:
    """두 이름에 함량이 모두 적혀 있는데 서로 다르면 True. 한쪽에만 있으면 다르다고 보지 않는다."""
    sa, sb = extract_strengths(a), extract_strengths(b)
    return bool(sa) and bool(sb) and sa != sb


def rank_by_name(key: str, dictionary: DrugDictionary) -> list[tuple[float, DrugEntry]]:
    """사전의 약을 이름이 비슷한 순서로. 점수가 같으면 사전 순서."""
    scored = [(similarity(key, e.key), i, e) for i, e in enumerate(dictionary.entries)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(score, e) for score, _, e in scored]


def _candidate(entry: DrugEntry, score: float, reason: str) -> MatchCandidate:
    return MatchCandidate(item_code=entry.item_code, name=entry.name, score=round(score, 3), reason=reason)


def _name_candidates(ranked, *, skip_code: str | None = None) -> list[MatchCandidate]:
    return [
        _candidate(e, score, "이름이 비슷함")
        for score, e in ranked
        if score >= CANDIDATE_MIN and e.item_code != skip_code
    ]


def pick_by_name(key: str, ranked: list[tuple[float, str, object]]):
    """이름만으로 하나를 고른다. ranked는 (점수, 비교 열쇠, 항목)을 점수 순으로. 못 고르면 None.

    함량이 처방전과 다른 항목은 다른 제품이라 겨루지 않는다 (아르펜정 400mg에게 200mg 제품은 경쟁자가 아니다).
    남은 것 중 1등이 아주 비슷하고(NAME_ONLY_MIN 이상) 2등보다 충분히 앞서야(NAME_ONLY_MARGIN 이상) 고른다.
    """
    same_strength = [(score, item) for score, item_key, item in ranked if not strengths_conflict(key, item_key)]
    if not same_strength:
        return None
    best_score, best = same_strength[0]
    second_score = same_strength[1][0] if len(same_strength) > 1 else 0.0
    if best_score >= NAME_ONLY_MIN and best_score - second_score >= NAME_ONLY_MARGIN:
        return best
    return None


def _match_by_api(key: str, name: str, why: str, api: ApiSearch):
    """API에서 이름으로 찾는다. (item_code, match_method, 이유, 후보)"""
    try:
        products = api(name)
    except DrugApiError as e:
        return None, "none", f"{why} 사전에 없어 의약품 API로 찾으려 했지만 실패했습니다: {e}", []
    scored = [(similarity(key, product_key(p.name)), i, p) for i, p in enumerate(products)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    best = pick_by_name(key, [(score, product_key(p.name), p) for score, _, p in scored])
    if best is not None:
        reason = f"{why} 사전에 없어 의약품 API에서 이름으로 '{best.name}'({best.item_code})을(를) 골랐습니다."
        return best.item_code, "api", reason, []
    candidates = [
        MatchCandidate(item_code=p.item_code, name=p.name, score=round(score, 3), reason="의약품 API 검색")
        for score, _, p in scored
        if score >= CANDIDATE_MIN
    ]
    return None, "none", f"{why} 사전과 의약품 API 어디에서도 약을 정하지 못했습니다.", candidates


def match_line(line: DrugLine, dictionary: DrugDictionary, *, api: ApiSearch | None = None) -> DrugLine:
    """약품 한 줄의 매칭 결과를 채운 새 DrugLine을 돌려준다 (원래 줄은 바꾸지 않는다).

    api를 주면 사전에서 정하지 못한 약을 의약품 API에서 찾는다. 주지 않으면 사전만 본다.
    """
    key = match_key(line.name)
    ranked = rank_by_name(key, dictionary)
    reasons: list[str] = []
    candidates: list[MatchCandidate] = []
    item_code, method = None, "none"

    entry = dictionary.by_code(line.printed_code)
    if entry is not None:
        item_code, method = entry.item_code, "code"
        score = similarity(key, entry.key)
        if score < NAME_MATCH_MIN:
            reasons.append(f"처방전의 약품명이 코드({entry.item_code})의 약 '{entry.name}'과 다릅니다.")
        if strengths_conflict(key, entry.key):
            reasons.append(f"처방전의 함량이 코드({entry.item_code})의 약 '{entry.name}'과 다릅니다.")
        if reasons:
            candidates = [_candidate(entry, score, "처방전 코드")] + _name_candidates(ranked, skip_code=entry.item_code)
    else:
        if line.printed_code:
            why = f"처방전 코드({line.printed_code})가 약품 사전에 없습니다."
        else:
            why = "처방전에서 코드를 읽지 못했습니다."
        best = pick_by_name(key, [(score, e.key, e) for score, e in ranked])
        if best is not None:
            item_code, method = best.item_code, "name"
            reasons.append(f"{why} 이름으로 '{best.name}'을(를) 골랐습니다.")
        elif api is not None:
            item_code, method, reason, api_candidates = _match_by_api(key, line.name, why, api)
            reasons.append(reason)
            if method == "none":     # 사전과 API의 후보를 점수 순으로 함께 보여 준다 (점수가 같으면 사전 먼저)
                candidates = sorted(_name_candidates(ranked) + api_candidates, key=lambda c: -c.score)
        else:
            reasons.append(f"{why} 이름으로도 사전의 약을 정하지 못했습니다.")
            candidates = _name_candidates(ranked)

    if line.ocr_conf is not None and line.ocr_conf < LOW_CONFIDENCE:
        reasons.append(f"글자 인식 신뢰도가 낮습니다({line.ocr_conf:.2f}).")
    if line.days > TRAY_DAYS:
        reasons.append(f"투약 일수 {line.days}일이 트레이 {TRAY_DAYS}일보다 깁니다.")

    data = line.model_dump()
    data.update(
        item_code=item_code,
        match_method=method,
        match_candidates=[c.model_dump() for c in candidates[:MAX_CANDIDATES]],
        needs_pharmacist_check=bool(reasons),
        check_reasons=reasons,
    )
    return DrugLine.model_validate(data)


def match_order(order: DispensingOrder, dictionary: DrugDictionary, *, api: ApiSearch | None = None) -> DispensingOrder:
    """조제지시서의 모든 약품 줄을 매칭한 새 조제지시서를 돌려준다. api는 match_line과 같다."""
    data = order.model_dump()
    data["drugs"] = [match_line(line, dictionary, api=api).model_dump() for line in order.drugs]
    return DispensingOrder.model_validate(data)
