"""복용 약 등록 · 재대조 (15단계, 일정표 11/16 ~ 11/17, REQ-AG-05 · 07, 명세서 2.2절 '복용 약 재대조').

약사가 환자의 기존 복용 약을 등록하면(수동 입력 F-WEB-10, 1차는 이것만), 처방약과 복용 약을 함께 다시 대조한다.

    read_meds(data)                          웹이 넘긴 복용 약 목록을 읽는다 [{item_code, name}]
    add_meds(current, new)                   등록된 목록에 더한다 (같은 약은 한 번만)
    recheck(order, meds, ...) -> Recheck     처방약 + 복용 약 대조. 처방 단계(A.5)에서 이미 보인 항목은 빼고 새 항목만
    consult_result(...) -> ConsultResult     A.6 상담 결과 (상담 카드 F-WEB-11에 보일 것)

규칙
- 복용 약은 품목기준코드(9자리)로만 받는다. 웹의 검색(F-AG-08 도구 search_drug)으로 고른 약이다. 이름만 있는 글(복용 약 후보 제안)은
  약사가 골라 확정해야 등록된다 (자동 반영 없음, REQ-AG-01).
- 새 항목: 대조 결과 중 복용 약이 하나라도 끼어 있는 것. 처방약끼리의 항목은 이미 승인 화면(A.5 risk_check)에서 보였다.
- 복용 중인 약과 같은 약이 처방되었으면 그것도 새 항목이다 (같은 약 중복 투여, 성분 중복 · 경고). 같은 품목은 성분 대조에서 한 약으로
  묶이므로 따로 찾는다.
- 위험 단계는 DUR 유형 규칙(risk.py, 명세서 2.8절)을 그대로 쓴다. risk_level = A.5 처방약 대조와 이번 새 항목 중 가장 높은 단계.
- 조제 계획까지 끝난 처방(조제 · 검수 중이거나 끝남)에 새 항목이 나오면 recheck_after_dispense = True ('조제 내용 재검토 필요').
- 새 항목 중 위험(danger)이 있으면 약사가 확인해야 진행한다 (HITL ③ risk, needs_ack).

판단 층이다. DUR 조회는 인자(lookup)로 받는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from agent.drug_dictionary import DrugDictionary
from agent.dur import INGREDIENT_DUP, DurLookup, DurReport, Ref, find_all
from agent.risk import LEVEL_LABELS, TYPE_LABELS, finding_from_hit, report_warnings
from agent.schema import ConsultResult, DispensingOrder, RiskFinding, max_risk

_ITEM_CODE = re.compile(r"^\d{9}$")
DISPENSED = ("planned", "inspecting", "inspected")       # 조제 계획이 나와 조제가 시작된(또는 끝난) 처방
SAME_DRUG_MESSAGE = "{name}: 복용 중인 약과 같은 약이 처방되었습니다. 중복 투여인지 확인하세요."
NO_NEW_ANSWER = "복용 약 {n}개를 처방약과 함께 대조했습니다. 새로 확인된 위험 신호는 없습니다."
REVIEW_TAIL = " 이미 조제가 끝난 처방입니다. 조제 내용 재검토가 필요합니다."


class MedsError(ValueError):
    """복용 약 목록이 잘못되었다. 메시지는 화면에 그대로 보여도 되는 문장이다."""


def read_meds(data: Any) -> list[dict[str, str]]:
    """웹이 넘긴 복용 약 [{item_code, name}]을 읽는다. 품목기준코드가 9자리 숫자가 아니거나 이름이 없으면 MedsError."""
    if not isinstance(data, list) or not data:
        raise MedsError("복용 약 목록이 비어 있습니다. [{item_code, name}] 모양으로 하나 이상 보내 주세요.")
    meds = []
    for number, item in enumerate(data, start=1):
        code = str(item.get("item_code") or "").strip() if isinstance(item, dict) else ""
        name = str(item.get("name") or "").strip() if isinstance(item, dict) else ""
        if not _ITEM_CODE.match(code):
            raise MedsError(f"{number}번째 복용 약의 품목기준코드가 9자리 숫자가 아닙니다. 검색해서 고른 약만 등록할 수 있습니다.")
        if not name:
            raise MedsError(f"{number}번째 복용 약의 이름이 없습니다.")
        meds.append({"item_code": code, "name": name})
    return meds


def add_meds(current: Iterable[dict[str, str]], new: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(등록된 목록 전체, 이번에 새로 더해진 약). 같은 품목기준코드는 한 번만."""
    merged = [dict(m) for m in current]
    codes = {m["item_code"] for m in merged}
    added = []
    for med in new:
        if med["item_code"] not in codes:
            codes.add(med["item_code"])
            merged.append(dict(med))
            added.append(dict(med))
    return merged, added


@dataclass(frozen=True)
class Recheck:
    new_findings: tuple[RiskFinding, ...]
    risk_level: str                         # A.5 처방약 대조와 새 항목 중 가장 높은 단계
    warnings: tuple[str, ...] = field(default_factory=tuple)   # 대조하지 못한 것 (DUR API 미사용 · 주성분 모름 등)

    @property
    def needs_ack(self) -> bool:
        """HITL ③: 새 항목 중 위험(danger)이 있으면 약사가 확인해야 진행한다 (명세서 2.2절)."""
        return any(f.level == "danger" for f in self.new_findings)


def _same_drug_findings(order: DispensingOrder, meds: list[dict[str, str]]) -> list[RiskFinding]:
    rx = {d.item_code: d.name for d in order.items if d.item_code}
    return [
        RiskFinding(type=INGREDIENT_DUP, item_codes=[m["item_code"]], level="warning",
                    message=SAME_DRUG_MESSAGE.format(name=rx[m["item_code"]]))
        for m in meds if m["item_code"] in rx
    ]


def new_hits(report: DurReport, meds: list[dict[str, str]]):
    """복용 약이 하나라도 끼어 있는 항목만 (처방약끼리는 A.5에서 이미 보였다)."""
    med_codes = {m["item_code"] for m in meds}
    return [h for h in report.hits if med_codes & set(h.item_codes)]


def recheck(order: DispensingOrder, meds: list[dict[str, str]], dictionary: DrugDictionary,
            lookup: DurLookup | None = None, *, report: DurReport | None = None) -> Recheck:
    """처방약과 복용 약을 함께 대조한다. report를 주면(MCP 도구로 대조한 결과) 그것을 쓴다."""
    rx_codes = {d.item_code for d in order.items if d.item_code}
    others = [m for m in meds if m["item_code"] not in rx_codes]     # 같은 약은 따로 알린다 (아래)
    items = [*order.items, *(Ref(m["item_code"], m["name"]) for m in others)]
    if report is None:
        report = find_all(items, dictionary, lookup)
    findings = [*_same_drug_findings(order, meds), *(finding_from_hit(h) for h in new_hits(report, others))]
    rx_level = order.risk_check.level if order.risk_check is not None else "none"
    level = max_risk([rx_level, *(f.level for f in findings)])
    return Recheck(tuple(findings), level, tuple(report_warnings(report)))


def recheck_items(order: DispensingOrder, meds: list[dict[str, str]]) -> list[Ref]:
    """대조할 약 목록 (MCP 도구 check_interaction · check_duplicate의 입력과 같다): 처방약 + 처방에 없는 복용 약."""
    rx_codes = {d.item_code for d in order.items if d.item_code}
    return [*(Ref(d.item_code, d.name) for d in order.items),
            *(Ref(m["item_code"], m["name"]) for m in meds if m["item_code"] not in rx_codes)]


def answer_text(result: Recheck, meds: list[dict[str, str]], dispensed: bool) -> str:
    """상담 카드에 보일 Agent 설명 (지금은 고정 문장. LLM 설명은 18단계)."""
    if not result.new_findings:
        return NO_NEW_ANSWER.format(n=len(meds))
    lines = [f"복용 약 {len(meds)}개를 처방약과 함께 대조해 새 위험 신호 {len(result.new_findings)}건을 찾았습니다"
             f" (가장 높은 단계: {LEVEL_LABELS[max_risk(f.level for f in result.new_findings)]})."]
    for f in result.new_findings:
        lines.append(f"- [{LEVEL_LABELS[f.level]} · {TYPE_LABELS.get(f.type, f.type)}] {f.message}")
    text = "\n".join(lines)
    return text + REVIEW_TAIL if dispensed else text


def consult_result(conversation_id: str, prescription_id: str, result: Recheck, meds: list[dict[str, str]],
                   status: str | None) -> ConsultResult:
    """A.6 상담 결과. 복용 약을 넣어 대조했으므로 risk_scope는 rx_with_current_meds다."""
    dispensed = status in DISPENSED
    return ConsultResult(
        conversation_id=conversation_id,
        prescription_id=prescription_id,
        risk_scope="rx_with_current_meds",
        new_findings=list(result.new_findings),
        recheck_after_dispense=dispensed and bool(result.new_findings),
        risk_level=result.risk_level,
        answer=answer_text(result, meds, dispensed),
        evidence=[],
    )
