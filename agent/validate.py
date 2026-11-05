"""구조화 결과의 공통 검증 관문 (11/4).

규칙 경로와 LLM 경로는 같은 모양의 dict를 내고, 둘 다 여기서 스키마 검증을 통과해야
다음 노드로 넘어간다.
    {"issue_no": ..., "drugs": [{"name", "printed_code", "dose_per_time",
                                 "times_per_day", "days", "time_slots", "ocr_conf"(규칙 경로만)}, ...]}

믿기 어려운 값은 고쳐 쓰지 않고 StructureError로 알린다. 고치는 것은 복용 시점 채우기뿐이고,
채울 때는 반드시 warnings에 남겨 약사가 볼 수 있게 한다.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from agent.errors import StructureError
from agent.schema import SLOT_ORDER, DispensingOrder, DrugLine

# 처방전에 복용 시점이 없을 때 하루 횟수로 정하는 기본 시점. 4회면 트레이의 네 행을 모두 쓴다.
_DEFAULT_SLOTS = {
    1: ["morning"],
    2: ["morning", "evening"],
    3: ["morning", "noon", "evening"],
    4: ["morning", "noon", "evening", "bedtime"],
}


def default_time_slots(times_per_day: int) -> list[str]:
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
    raw_drugs = extracted.get("drugs")
    if not isinstance(raw_drugs, list) or not raw_drugs:
        raise StructureError("처방전에서 약품을 찾지 못했습니다.")

    warnings: list[str] = []
    drugs: list[DrugLine] = []
    for number, raw in enumerate(raw_drugs, start=1):
        label = f"{number}번째 약품"
        if not isinstance(raw, dict):
            raise StructureError(f"{label}의 형식이 올바르지 않습니다.")

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise StructureError(f"{label}의 약품명이 비어 있습니다.")

        for field, title in (("dose_per_time", "1회량"), ("times_per_day", "하루 복용 횟수"), ("days", "투약 일수")):
            if not _is_int(raw.get(field)):
                raise StructureError(f"{label}의 {title} 값을 정수로 읽지 못했습니다.")
        dose, times, days = raw["dose_per_time"], raw["times_per_day"], raw["days"]
        if dose < 1:
            raise StructureError(f"{label}의 1회량이 1 미만입니다.")
        if not 1 <= times <= len(SLOT_ORDER):
            raise StructureError(f"{label}의 하루 복용 횟수({times}회)가 1~{len(SLOT_ORDER)}회 범위를 벗어납니다.")
        if days < 1:
            raise StructureError(f"{label}의 투약 일수가 1 미만입니다.")

        ocr_conf = raw.get("ocr_conf")
        if ocr_conf is not None and (isinstance(ocr_conf, bool) or not isinstance(ocr_conf, (int, float))):
            raise StructureError(f"{label}의 OCR 신뢰도 형식이 올바르지 않습니다.")

        raw_slots = raw.get("time_slots")
        if raw_slots is None:
            raw_slots = []
        if not isinstance(raw_slots, list) or any(s not in SLOT_ORDER for s in raw_slots):
            raise StructureError(f"{label}의 복용 시점 값이 올바르지 않습니다.")
        slots = [s for s in SLOT_ORDER if s in raw_slots]
        if not slots:
            slots = default_time_slots(times)
            warnings.append(f"{label}: 복용 시점이 처방전에 없어 하루 {times}회 기준으로 배정했습니다.")
        elif len(slots) != times:
            slots = default_time_slots(times)
            warnings.append(f"{label}: 복용 시점 수와 하루 횟수가 달라 횟수 기준으로 다시 배정했습니다. 확인이 필요합니다.")

        try:
            drugs.append(
                DrugLine(
                    name=name,
                    printed_code=_optional_text(raw.get("printed_code"), label, "코드"),
                    dose_per_time=dose,
                    times_per_day=times,
                    days=days,
                    time_slots=slots,
                    ocr_conf=ocr_conf,
                )
            )
        except ValidationError as e:
            raise StructureError(f"{label}의 값이 올바르지 않습니다.") from e

    try:
        return DispensingOrder(
            prescription_id=prescription_id,   # 앞뒤 공백은 스키마가 지운다
            source=source,
            issue_no=_optional_text(extracted.get("issue_no"), "처방전", "교부번호"),
            drugs=drugs,
            warnings=warnings,
        )
    except ValidationError as e:
        raise StructureError("조제지시서를 만들지 못했습니다.") from e
