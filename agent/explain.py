"""위험 신호 설명 (판단, 18단계, REQ-AG-07) — 규칙이 찾은 위험 신호 항목에 AI 설명을 덧붙인다. 단계는 바꾸지 않는다.

    explain_input(findings, names, passages)     LLM에 보낼 글 (항목마다 단계 · 유형 · 약 · 성분 · 금기 사유 · 규칙 문장)
    apply_explanations(findings, got, names)     LLM 결과를 거른 뒤 항목의 explanation에 담은 새 목록
    explain_findings(findings, names, explain_fn, passages) -> (새 목록, 안내)   위 둘을 합친 것. 실패해도 항목은 그대로
    attach_evidence(findings, passages)         항목마다 관련 약의 근거 조각을 evidence(A.5)에 담는다 (설명 성공과 상관없이)
    recheck(findings, names)                    근거를 다 붙인 뒤 설명을 같은 기준으로 다시 거른다 (AI 조사 경로, 11/27)

규칙 (명세서 2.8절 · REQ-AG-07: 분류는 규칙, LLM은 설명만)
- 단계(level) · 유형(type) · 관련 약(item_codes) · 규칙 문장(message)은 그대로다. 설명은 explanation 칸에만 들어간다.
- 거르는 것: 번호가 없는 항목, 너무 짧거나 긴 글, 관련 약 이름이 하나도 없는 글, 위험을 낮추는 말('안전합니다' · '괜찮' · '문제없'),
  다른 단계를 말하는 글('경고 단계'인데 '주의 단계'라고 함), 규칙 문장을 되풀이하는 글, 근거가 있는데 근거의 내용이 없는 글(uses_evidence).
  걸러진 항목은 explanation이 None이다 (규칙 문장만 보인다).
- LLM이 없거나 실패하면 설명 없이 그대로 (서비스는 멈추지 않는다, REQ-NF-06).
판단 층이다. LLM은 explain_fn(글) -> {"items": [{"index", "text"}]}로 받는다.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from agent.dur import COMBINATION_TABOO, EFFICACY_DUP, INGREDIENT_DUP
from agent.errors import StructureError
from agent.rag import Passage
from agent.risk import LEVEL_LABELS, TYPE_LABELS
from agent.schema import Evidence, RiskFinding

ExplainFn = Callable[[str], dict[str, Any]]

MIN_CHARS, MAX_CHARS = 15, 300
DOWNPLAY = ("안전합니다", "안전한", "괜찮", "문제없", "문제 없", "걱정하지", "무시해도", "해당 없음")
_LEVEL_SAID = re.compile(r"(주의|경고|위험)\s*단계")
REPEAT_RATIO = 0.7        # 설명의 두 글자 조각 중 규칙 문장에 이미 있는 비율이 이 이상이면 되풀이로 보고 버린다
NEW_FROM_EVIDENCE = 4     # 근거가 있는 항목의 설명은 '규칙 문장에는 없고 근거 발췌에는 있는' 두 글자 조각이 이 개수 이상 있어야 한다
                          # (실측 11/26: 규칙 문장을 말만 바꿔 되풀이한 설명 0개, 근거의 하루 최대 용량 · 병용 금지를 담은 설명 7~21개)
FAILED_NOTE = "위험 신호 AI 설명을 만들지 못해 규칙 문장만 보였습니다"
EXCERPT_CHARS = 200        # 근거 발췌(A.5 evidence.excerpt)로 보이는 글자 수
EVIDENCE_PER_FINDING = 2   # 항목 하나에 붙이는 근거 수
EXCERPT_FOCUS_CHARS = 300  # 근거 문장을 골라 보여 줄 때의 글자 수 (문장을 자르지 않으려고 EXCERPT_CHARS보다 넉넉하다)
CANDIDATES_PER_DRUG = 4    # 근거를 고르기 전에 약마다 찾아 보는 조각 수
# 위험 유형마다 근거 문장에 들어 있어야 할 말 (말, 무게). 무게 3(STRONG)은 '다른 약과 함께 쓰지 말라'는 뜻의 말이다.
# 근거는 이 말이 든 문장이 있는 조각만 쓴다 (확실한 근거만). 무게 1(WEAK)은 용량 · 성분 이름처럼 보조로만 센다.
STRONG, WEAK = 3, 1
TYPE_TERMS = {
    INGREDIENT_DUP: (("다른 제품", STRONG), ("함께 복용", STRONG), ("함께 이 약을", STRONG), ("중복", STRONG), ("같은 성분", STRONG), ("포함하는", STRONG),
                     ("최대 용량", WEAK), ("일일 최대", WEAK), ("초과", WEAK)),
    COMBINATION_TABOO: (("병용", STRONG), ("함께 복용", STRONG), ("투여하지", STRONG), ("금기", STRONG), ("동시에", STRONG)),
    EFFICACY_DUP: (("같은 효능", STRONG), ("함께 복용", STRONG), ("중복", STRONG), ("병용", STRONG), ("동시에", STRONG)),
}
Terms = Sequence[tuple[str, int]]       # (말, 무게)
_INGREDIENTS_IN_MESSAGE = re.compile(r"에 (.+?) 성분이 함께")
_LIST_MARK = re.compile(r"^\(?\d+[.)]\s*")           # 문장 앞의 '2.' '(5)' '3)' 같은 번호
_TAIL_MARK = re.compile(r"\s*\(?\d+[.)]$")            # 끝에 남은 다음 번호


def explain_input(findings: Sequence[RiskFinding], names: dict[str, str], passages: Sequence[Passage] = ()) -> str:
    lines = ["항목:"]
    for n, f in enumerate(findings, start=1):
        drugs = ", ".join(names.get(c, c) for c in f.item_codes)
        lines.append(f"[{n}] 단계: {LEVEL_LABELS[f.level]} | 유형: {TYPE_LABELS.get(f.type, f.type)} | 약: {drugs} | 규칙 문장: {f.message}")
        for e in f.evidence:        # 약사 화면에 보일 발췌와 같은 글 (attach_evidence로 먼저 붙인 항목)
            lines.append(f"    근거({e.title}): {e.excerpt}")
    if passages:
        lines.append("근거:")
        for p in passages:
            lines.append(f"- ({p.chunk.item_name or p.chunk.item_code} · {p.chunk.section}) {p.chunk.text}")
    return "\n".join(lines)


def _stems(f: RiskFinding, names: dict[str, str]) -> list[str]:
    """설명에 나와야 하는 약 이름의 앞부분 (3글자. '타이레놀정500mg' -> '타이레')."""
    out = []
    for code in f.item_codes:
        name = "".join(names.get(code, "").split())
        if name:
            out.append(name[:3])
    return out


def _grams(text: str) -> set[str]:
    compact = "".join(re.sub(r"[·.,()\[\]]", " ", text).split())
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def repeats(text: str, message: str) -> bool:
    """설명이 규칙 문장을 되풀이하기만 하는가 (새로 알려 주는 내용이 없다).

    두 글자 조각으로 나눠, 설명의 조각 중 규칙 문장에 이미 있는 것이 REPEAT_RATIO 이상이면 되풀이다.
    (실측: 되풀이 0.76~0.84, 근거의 한도를 더한 설명 0.46~0.50)
    """
    grams = _grams(text)
    return bool(grams) and len(grams & _grams(message)) / len(grams) >= REPEAT_RATIO


def uses_evidence(text: str, f: RiskFinding) -> bool:
    """근거가 있는 항목의 설명이 근거에서 온 새 내용을 담았는가 (근거가 없는 항목은 항상 True).

    말만 바꿔 규칙 문장을 되풀이한 설명(repeats로는 못 거른다)을 거른다. 규칙 문장에는 없고 근거 발췌에는 있는
    두 글자 조각이 NEW_FROM_EVIDENCE개 이상이어야 한다.
    """
    if not f.evidence:
        return True
    from_evidence = set().union(*(_grams(e.excerpt) for e in f.evidence))
    return len((_grams(text) - _grams(f.message)) & from_evidence) >= NEW_FROM_EVIDENCE


def acceptable(text: str, f: RiskFinding, names: dict[str, str]) -> bool:
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if repeats(text, f.message):             # 규칙 문장과 같은 말이면 설명을 붙이지 않는다 (화면에는 규칙 문장만 보인다)
        return False
    if not uses_evidence(text, f):           # 근거가 있는데 근거의 내용이 하나도 없는 설명도 붙이지 않는다
        return False
    compact = "".join(text.split())
    stems = _stems(f, names)
    if stems and not any(s in compact for s in stems):
        return False
    if any(word.replace(" ", "") in compact for word in DOWNPLAY):
        return False
    return all(said == LEVEL_LABELS[f.level] for said in _LEVEL_SAID.findall(text))


def apply_explanations(findings: Sequence[RiskFinding], got: dict[str, Any], names: dict[str, str]) -> list[RiskFinding]:
    texts: dict[int, str] = {}
    for item in (got or {}).get("items") or []:
        try:
            index, text = int(item["index"]), " ".join(str(item["text"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= index <= len(findings) and index not in texts:
            texts[index] = text
    out = []
    for n, f in enumerate(findings, start=1):
        text = texts.get(n)
        keep = text if text and acceptable(text, f, names) else None
        out.append(f.model_copy(update={"explanation": keep}))
    return out


def recheck(findings: Sequence[RiskFinding], names: dict[str, str]) -> list[RiskFinding]:
    """근거를 다 붙인 뒤 설명을 한 번 더 거른다 (11/27). 거르지 못한 설명은 지워 규칙 문장만 보이게 한다.

    AI 조사(ReAct)는 설명을 받을 때 항목에 근거가 아직 없어 uses_evidence가 언제나 통과였다.
    화면에 보일 근거가 정해진 다음 같은 기준(acceptable)으로 거른다. 근거는 그대로 둔다.
    """
    return [f.model_copy(update={"explanation": None})
            if f.explanation and not acceptable(f.explanation, f, names) else f for f in findings]


def finding_terms(f: RiskFinding) -> tuple[tuple[str, int], ...]:
    """이 항목의 근거에서 찾을 말(말, 무게): 유형의 말 + (성분 중복이면) 겹친 성분 이름(약한 말)."""
    terms = list(TYPE_TERMS.get(f.type, ()))
    if f.type == INGREDIENT_DUP:
        found = _INGREDIENTS_IN_MESSAGE.search(f.message or "")
        if found:
            terms += [(x.strip(), WEAK) for x in found.group(1).split(",") if x.strip()]
    return tuple(terms)


def _squash(text: str) -> str:
    return "".join(str(text).split())


def term_hits(text: str, terms: Terms) -> int:
    """글에 들어 있는 말의 무게 합 (띄어쓰기 차이는 무시)."""
    flat = _squash(text)
    return sum(weight for term, weight in terms if _squash(term) and _squash(term) in flat)


def has_strong(text: str, terms: Terms) -> bool:
    """'다른 약과 함께 쓰지 말라'는 뜻의 말(STRONG)이 하나라도 든 글인가."""
    flat = _squash(text)
    return any(weight >= STRONG and _squash(term) in flat for term, weight in terms)


def _sentences(text: str) -> list[str]:
    """문장으로 나눈다. '2.' 같은 번호의 마침표에서는 나누지 않는다."""
    parts = re.split(r"(?<![0-9])(?<=[.!?])\s+|\n+", str(text))
    return [" ".join(x.split()) for x in parts if x.strip()]


def focused_excerpt(text: str, terms: Terms, limit: int | None = None) -> str | None:
    """근거 글에서 질문(terms)과 맞는 문장들만 이어 붙인 발췌. 맞는 문장(STRONG이 든 문장)이 없으면 None.

    - 문장을 자르지 않는다 (…을 붙이지 않는다). 고른 문장 앞뒤의 말이 하나도 없는 문장은 뺀다.
    - 문장 앞의 번호('2.')와 끝에 남은 다음 번호는 지운다.
    - 고른 문장들이 limit(기본 EXCERPT_FOCUS_CHARS)보다 길면, limit 안에서 맞는 말이 가장 많은 연속한 문장들을 고른다.
      맞는 문장 하나가 limit보다 길어도 그 문장은 통째로 보인다.
    """
    limit = EXCERPT_FOCUS_CHARS if limit is None else limit
    sentences = _sentences(text)
    strong = [has_strong(x, terms) for x in sentences]
    if not any(strong):
        return None
    hits = [term_hits(x, terms) + (10 if st else 0) for x, st in zip(sentences, strong)]   # 맞는 문장이 든 묶음을 앞세운다
    best_score, best_start, best_end = -1, 0, 0
    for start in range(len(sentences)):
        end, used, score = start, 0, 0
        while end < len(sentences) and used + len(sentences[end]) + (1 if end > start else 0) <= limit:
            used += len(sentences[end]) + (1 if end > start else 0)
            score += hits[end]
            end += 1
        if end == start:                      # 문장 하나가 limit보다 길다: 그 문장을 통째로
            end, score = start + 1, hits[start]
        if score > best_score:
            best_score, best_start, best_end = score, start, end
    while best_end - best_start > 1 and term_hits(sentences[best_start], terms) == 0:     # 앞뒤의 말 없는 문장은 뺀다
        best_start += 1
    while best_end - best_start > 1 and term_hits(sentences[best_end - 1], terms) == 0:
        best_end -= 1
    first = _LIST_MARK.sub("", sentences[best_start])
    picked = " ".join([first, *sentences[best_start + 1:best_end]])
    return _TAIL_MARK.sub("", picked).strip()


def evidence_item(title: str, source: str, url: str | None, text: str, terms: Terms = ()) -> Evidence:
    """근거 하나 (명세서 A.5: 제목 · 출처 · 발췌). 출처에는 사람이 열어 볼 수 있는 주소를 함께 적는다.

    terms(finding_terms)를 주면 발췌는 질문과 맞는 문장들이다 (focused_excerpt). 맞는 문장이 없거나 terms가 없으면
    앞에서부터 EXCERPT_CHARS 글자다.
    """
    excerpt = focused_excerpt(text, terms) if terms else None
    if excerpt is None:
        excerpt = " ".join(str(text).split())
        if len(excerpt) > EXCERPT_CHARS:
            excerpt = excerpt[:EXCERPT_CHARS].rstrip() + "…"
    return Evidence(title=title, source=f"{source} ({url})" if url else source, excerpt=excerpt or "-")


def passage_evidence(p: Passage, terms: Terms = ()) -> Evidence:
    return evidence_item(f"{p.chunk.item_name or p.chunk.item_code} · {p.chunk.section}", p.chunk.source or "허가정보",
                         p.chunk.url or None, p.chunk.text, terms)


def _usable(passages: Sequence[Passage], terms: Terms) -> list[Passage]:
    """근거로 쓸 수 있는 조각: 물은 약에서 찾았고(focused), terms가 있으면 맞는 문장이 든 것만."""
    return [p for p in passages if p.focused and (not terms or has_strong(p.chunk.text, terms))]


def _best_for_drug(mine: Sequence[Passage], code: str, terms: Terms) -> Passage | None:
    """그 약의 조각 가운데 terms가 가장 많이 든 것 (같으면 순위가 앞선 것)."""
    candidates = [p for p in mine if p.chunk.item_code == code]
    return max(candidates, key=lambda p: term_hits(p.chunk.text, terms), default=None)


def _spread_by_drug(passages: Sequence[Passage], codes: Sequence[str], limit: int, terms: Terms = ()) -> list[Passage]:
    """항목의 약마다 질문과 가장 맞는 조각을 하나씩 먼저 고르고, 남는 자리는 순위대로 채운다 (한 약의 근거만 몰리지 않게)."""
    mine = [p for p in _usable(passages, terms) if p.chunk.item_code in codes]
    picked = [pick for code in codes if (pick := _best_for_drug(mine, code, terms)) is not None][:limit]
    for p in mine:
        if len(picked) >= limit:
            break
        if all(p is not q for q in picked):
            picked.append(p)
    return picked


def _cover_missing_drugs(f: RiskFinding, passages: Sequence[Passage]) -> RiskFinding:
    """이미 근거가 있는 항목(AI 조사가 인용한 것)에서 근거가 하나도 없는 약이 있으면, 그 약의 맞는 조각을 하나씩 덧붙인다.

    AI가 한 약의 문서만 인용해도 성분 중복처럼 약 둘을 함께 말하는 항목에는 두 약의 근거가 보이게 한다.
    인용한 근거는 그대로 두고 뒤에만 더한다 (제목은 '약 이름 · 항목'이라 앞부분의 약 이름으로 어느 약인지 안다).
    """
    covered = {e.title.split(" · ")[0] for e in f.evidence}
    terms = finding_terms(f)
    usable = _usable(passages, terms)
    extra: list[Evidence] = []
    for code in f.item_codes:
        pick = _best_for_drug(usable, code, terms)
        if pick is None:
            continue
        name = pick.chunk.item_name or pick.chunk.item_code
        if name not in covered:
            covered.add(name)
            extra.append(passage_evidence(pick, terms))
    return f.model_copy(update={"evidence": [*f.evidence, *extra]}) if extra else f


def attach_evidence(findings: Sequence[RiskFinding], passages: Sequence[Passage]) -> list[RiskFinding]:
    """근거가 아직 없는 항목에 그 항목의 약들에서 찾은 조각을 담는다 (항목당 최대 EVIDENCE_PER_FINDING개).

    약마다 담는 조각은 위험 유형의 말(TYPE_TERMS)이 든 문장이 있는 것 중 가장 맞는 것이고, 발췌는 그 문장들뿐이다.
    약마다 하나씩 먼저 담아 한 약의 근거만 나오지 않게 한다. 다른 약에서 찾은 조각(focused False)과 맞는 문장이 없는 조각은
    담지 않는다. 담을 것이 없으면 빈 목록 그대로 (화면은 '근거 없음').
    단계 · 유형 · 규칙 문장 · 설명은 그대로다 (근거는 덧붙이기만 한다).
    """
    out = []
    for f in findings:
        if f.evidence:
            out.append(_cover_missing_drugs(f, passages))
            continue
        terms = finding_terms(f)
        mine = _spread_by_drug(passages, list(f.item_codes), EVIDENCE_PER_FINDING, terms)
        out.append(f.model_copy(update={"evidence": [passage_evidence(p, terms) for p in mine]}) if mine else f)
    return out


def explain_findings(findings: Sequence[RiskFinding], names: dict[str, str], explain_fn: ExplainFn | None,
                     passages: Sequence[Passage] = ()) -> tuple[list[RiskFinding], str | None]:
    """(설명 · 근거를 붙인 항목들, 안내). explain_fn이 없거나 실패해도 근거(passages)는 붙인다. 실패하면 + 안내.

    근거를 설명보다 먼저 붙인다 (11/26): LLM에는 약사 화면에 보일 항목별 발췌를 주고, 설명이 그 내용을 쓰는지 uses_evidence로 거른다.
    """
    findings = attach_evidence(list(findings), passages)
    if not findings:
        return findings, None
    if explain_fn is None:
        return findings, None
    try:
        got = explain_fn(explain_input(findings, names))
    except StructureError as e:
        return findings, f"{FAILED_NOTE} ({e})."
    return apply_explanations(findings, got, names), None


def passages_for(rag, findings: Sequence[RiskFinding], names: dict[str, str], per_finding: int = 2, limit: int = 6):
    """항목의 약들에 대한 근거 조각 (색인이 있을 때). 같은 조각은 한 번만.

    질문은 항목의 약 이름과 위험 유형의 말(finding_terms)을 담고, 검색은 약마다 따로 한다. 한 번에 찾으면 점수가 비슷할 때
    한 약의 조각만 상위를 차지해 다른 약의 근거가 빠진다. 약마다 CANDIDATES_PER_DRUG개를 찾아 맞는 문장이 든 것 중
    유형의 말이 많이 든 것부터 per_finding개를 남긴다.
    """
    if rag is None or not len(rag):
        return []
    seen, out = set(), []
    for f in findings:
        terms = finding_terms(f)
        query = " ".join([*(names.get(c, "") for c in f.item_codes), *(t for t, _ in terms), "상호작용 주의사항"])
        for code in f.item_codes:
            found = _usable(rag.search(query, focus=[code], k=max(per_finding, CANDIDATES_PER_DRUG)), terms)
            found.sort(key=lambda p: -term_hits(p.chunk.text, terms))        # 같으면 검색 순위대로 (정렬은 안정적이다)
            for p in found[:per_finding]:
                if p.chunk.chunk_id not in seen:
                    seen.add(p.chunk.chunk_id)
                    out.append(p)
    return out[:limit]
