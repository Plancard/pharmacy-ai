"""구조화 결과의 공통 검증 관문 (2단계).

규칙 경로와 LLM 경로는 같은 모양의 dict를 내고, 둘 다 여기서 스키마 검증을 통과해야
다음 노드로 넘어간다.
    {"issue_no": ..., "items": [{"name", "printed_code", "count_per_dose", "times_per_day", "days", "times",
                                 "source_line_ids"·"ocr_conf"(규칙 경로만)}, ...]}

믿기 어려운 값은 고쳐 쓰지 않고 StructureError로 알린다. 고치는 것은 복용 시점 채우기뿐이고,
채울 때는 반드시 warnings에 남겨 약사가 볼 수 있게 한다.

나온 조제지시서(A.5 형식)는 아직 약품 매칭 전이라 item_code가 비어 있고 risk_check도 없다. 웹에 넘기려면
그 두 가지를 채운 뒤 DispensingOrder.check_ready_for_web()을 통과해야 한다.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from agent.errors import StructureError
from agent.schema import TIME_SLOTS, DispensingOrder, DrugLine

# 처방전에 복용 시점이 없을 때 하루 횟수로 정하는 기본 시점. 4회면 트레이의 네 행을 모두 쓴다.
_DEFAULT_SLOTS = {
    1: ["morning"],
    2: ["morning", "evening"],
    3: ["morning", "noon", "evening"],
    4: ["morning", "noon", "evening", "bedtime"],
}


def default_times(times_per_day: int) -> list[str]:
    return list(_DEFAULT_SLOTS[times_per_day])


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_text(value: Any, owner: str, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StructureError(f"{owner}의 {field} 형식이 올바르지 않습니다.")
    return value.strip() or None


def order_from_extracted(
    prescription_id: str, extracted: Any, *, source: Literal["rule", "llm"]
) -> DispensingOrder:
    """구조화 결과 dict를 검증해 조제지시서로 만든다. 문제가 있으면 StructureError."""
    if not isinstance(prescription_id, str) or not prescription_id.strip():
        raise StructureError("처방 ID가 없습니다.")
    if not isinstance(extracted, dict):
        raise StructureError("구조화 결과의 형식이 올바르지 않습니다.")
    raw_drugs = extracted.get("items")
    if not isinstance(raw_drugs, list) or not raw_drugs:
        raise StructureError("처방전에서 약품을 찾지 못했습니다.")

    warnings: list[str] = []
    items: list[DrugLine] = []
    for number, raw in enumerate(raw_drugs, start=1):
        label = f"{number}번째 약품"
        if not isinstance(raw, dict):
            raise StructureError(f"{label}의 형식이 올바르지 않습니다.")

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise StructureError(f"{label}의 약품명이 비어 있습니다.")

        for field, title in (("count_per_dose", "1회량"), ("times_per_day", "하루 복용 횟수"), ("days", "투약 일수")):
            if not _is_int(raw.get(field)):
                raise StructureError(f"{label}의 {title} 값을 정수로 읽지 못했습니다.")
        dose, per_day, days = raw["count_per_dose"], raw["times_per_day"], raw["days"]
        if dose < 1:
            raise StructureError(f"{label}의 1회량이 1 미만입니다.")
        if not 1 <= per_day <= len(TIME_SLOTS):
            raise StructureError(f"{label}의 하루 복용 횟수({per_day}회)가 1~{len(TIME_SLOTS)}회 범위를 벗어납니다.")
        if days < 1:
            raise StructureError(f"{label}의 투약 일수가 1 미만입니다.")

        ocr_conf = raw.get("ocr_conf")
        if ocr_conf is not None and (isinstance(ocr_conf, bool) or not isinstance(ocr_conf, (int, float))):
            raise StructureError(f"{label}의 OCR 신뢰도 형식이 올바르지 않습니다.")

        source_line_ids = raw.get("source_line_ids")
        if source_line_ids is None:
            source_line_ids = []
        if not isinstance(source_line_ids, list) or not all(_is_int(v) and v >= 0 for v in source_line_ids):
            raise StructureError(f"{label}의 원문 줄 번호(source_line_ids) 형식이 올바르지 않습니다.")

        raw_slots = raw.get("times")
        if raw_slots is None:
            raw_slots = []
        if not isinstance(raw_slots, list) or any(s not in TIME_SLOTS for s in raw_slots):
            raise StructureError(f"{label}의 복용 시점 값이 올바르지 않습니다.")
        slots = [s for s in TIME_SLOTS if s in raw_slots]
        if not slots:
            slots = default_times(per_day)
            warnings.append(f"{label}: 복용 시점이 처방전에 없어 하루 {per_day}회 기준으로 배정했습니다.")
        elif len(slots) != per_day:
            slots = default_times(per_day)
            warnings.append(f"{label}: 복용 시점 수와 하루 횟수가 달라 횟수 기준으로 다시 배정했습니다. 확인이 필요합니다.")

        try:
            items.append(
                DrugLine(
                    item_code=None,   # 약품 매칭(drug_match.py)이 채운다
                    name=name,
                    printed_code=_optional_text(raw.get("printed_code"), label, "코드"),
                    count_per_dose=dose,
                    days=days,
                    times=slots,
                    ocr_conf=ocr_conf,
                    source_line_ids=source_line_ids,
                )
            )
        except ValidationError as e:
            raise StructureError(f"{label}의 값이 올바르지 않습니다.") from e

    try:
        return DispensingOrder(
            prescription_id=prescription_id,   # 앞뒤 공백은 스키마가 지운다
            source=source,
            issue_no=_optional_text(extracted.get("issue_no"), "처방전", "교부번호"),
            items=items,
            warnings=warnings,
        )
    except ValidationError as e:
        raise StructureError("조제지시서를 만들지 못했습니다.") from e
