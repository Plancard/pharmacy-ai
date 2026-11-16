"""위험 신호 설명 (판단, 18단계, REQ-AG-07) — 규칙이 찾은 위험 신호 항목에 AI 설명을 덧붙인다. 단계는 바꾸지 않는다.

    explain_input(findings, names, passages)     LLM에 보낼 글 (항목마다 단계 · 유형 · 약 · 성분 · 금기 사유 · 규칙 문장)
    apply_explanations(findings, got, names)     LLM 결과를 거른 뒤 항목의 explanation에 담은 새 목록
    explain_findings(findings, names, explain_fn, passages) -> (새 목록, 안내)   위 둘을 합친 것. 실패해도 항목은 그대로

규칙 (명세서 2.8절 · REQ-AG-07: 분류는 규칙, LLM은 설명만)
- 단계(level) · 유형(type) · 관련 약(item_codes) · 규칙 문장(message)은 그대로다. 설명은 explanation 칸에만 들어간다.
- 거르는 것: 번호가 없는 항목, 너무 짧거나 긴 글, 관련 약 이름이 하나도 없는 글, 위험을 낮추는 말('안전합니다' · '괜찮' · '문제없'),
  다른 단계를 말하는 글('경고 단계'인데 '주의 단계'라고 함). 걸러진 항목은 explanation이 None이다 (규칙 문장만 보인다).
- LLM이 없거나 실패하면 설명 없이 그대로 (서비스는 멈추지 않는다, REQ-NF-06).
판단 층이다. LLM은 explain_fn(글) -> {"items": [{"index", "text"}]}로 받는다.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from agent.errors import StructureError
from agent.rag import Passage
from agent.risk import LEVEL_LABELS, TYPE_LABELS
from agent.schema import RiskFinding

ExplainFn = Callable[[str], dict[str, Any]]

MIN_CHARS, MAX_CHARS = 15, 300
DOWNPLAY = ("안전합니다", "안전한", "괜찮", "문제없", "문제 없", "걱정하지", "무시해도", "해당 없음")
_LEVEL_SAID = re.compile(r"(주의|경고|위험)\s*단계")
FAILED_NOTE = "위험 신호 AI 설명을 만들지 못해 규칙 문장만 보였습니다"


def explain_input(findings: Sequence[RiskFinding], names: dict[str, str], passages: Sequence[Passage] = ()) -> str:
    lines = ["항목:"]
    for n, f in enumerate(findings, start=1):
        drugs = ", ".join(names.get(c, c) for c in f.item_codes)
        lines.append(f"[{n}] 단계: {LEVEL_LABELS[f.level]} | 유형: {TYPE_LABELS.get(f.type, f.type)} | 약: {drugs} | 규칙 문장: {f.message}")
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


def acceptable(text: str, f: RiskFinding, names: dict[str, str]) -> bool:
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
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


def explain_findings(findings: Sequence[RiskFinding], names: dict[str, str], explain_fn: ExplainFn | None,
                     passages: Sequence[Passage] = ()) -> tuple[list[RiskFinding], str | None]:
    """(설명을 붙인 항목들, 안내). 항목이 없거나 explain_fn이 없으면 그대로. 실패하면 그대로 + 안내."""
    findings = list(findings)
    if not findings or explain_fn is None:
        return findings, None
    try:
        got = explain_fn(explain_input(findings, names, passages))
    except StructureError as e:
        return findings, f"{FAILED_NOTE} ({e})."
    return apply_explanations(findings, got, names), None


def passages_for(rag, findings: Sequence[RiskFinding], names: dict[str, str], per_finding: int = 2, limit: int = 6):
    """항목의 약들에 대한 상호작용 · 주의사항 근거 조각 (색인이 있을 때). 같은 조각은 한 번만."""
    if rag is None or not len(rag):
        return []
    seen, out = set(), []
    for f in findings:
        query = " ".join([*(names.get(c, "") for c in f.item_codes), "함께 복용 상호작용 주의사항 병용"])
        for p in rag.search(query, focus=list(f.item_codes), k=per_finding):
            if p.focused and p.chunk.chunk_id not in seen:
                seen.add(p.chunk.chunk_id)
                out.append(p)
    return out[:limit]
