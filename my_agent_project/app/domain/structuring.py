"""처방전 구조화: LLM이 읽어 온 값 → 검증된 조제지시서(DispensingOrder).

LLM 호출 자체는 extract_fn으로 받는다 (실제 서비스는 app.llm.extract_prescription,
테스트는 가짜 함수). 그래서 이 모듈은 키·네트워크 없이 테스트된다.

1차 범위: 정제·캡슐을 통째로 먹는 경우만 다룬다. 반 알 같은 값은 조용히 고치지 않고 실패로 알린다.
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from app.config import ConfigError
from app.schemas import (
    SLOT_ORDER,
    DispensingOrder,
    DrugLine,
    ExtractedPrescription,
    default_time_slots,
)

ExtractFn = Callable[[str], dict[str, Any]]


class StructureError(Exception):
    """구조화 실패. 메시지는 그대로 화면에 보여줘도 되는 한국어 문장이다 (처방전 내용은 담지 않는다)."""


def order_from_extracted(prescription_id: str, extracted: dict[str, Any], *, source: str = "llm") -> DispensingOrder:
    try:
        ex = ExtractedPrescription.model_validate(extracted)
    except ValidationError as e:
        raise StructureError("LLM이 읽어 온 결과의 형식이 올바르지 않습니다.") from e

    if not ex.drugs:
        raise StructureError("처방전에서 약품을 찾지 못했습니다.")

    warnings: list[str] = []
    lines: list[DrugLine] = []
    for i, d in enumerate(ex.drugs, start=1):
        label = f"{i}번째 약품 '{d.name.strip()}'"
        if d.dose_per_time < 1:
            raise StructureError(f"{label}의 1회 투약량을 정수로 읽지 못했습니다 (반 알 등은 1차 범위 밖입니다).")
        if not 1 <= d.times_per_day <= 3:
            raise StructureError(f"{label}의 하루 복용 횟수({d.times_per_day}회)가 1~3회 범위를 벗어납니다.")
        if d.days < 1:
            raise StructureError(f"{label}의 투약 일수를 읽지 못했습니다.")

        slots = [s for s in SLOT_ORDER if s in set(d.time_slots)]
        if not slots:
            slots = default_time_slots(d.times_per_day)
            warnings.append(f"{label}: 복용 시점이 처방전에 없어 하루 {d.times_per_day}회 기준으로 배정했습니다.")
        elif len(slots) != d.times_per_day:
            slots = default_time_slots(d.times_per_day)
            warnings.append(f"{label}: 복용 시점 수와 하루 횟수가 달라 횟수 기준으로 다시 배정했습니다. 확인이 필요합니다.")

        try:
            lines.append(
                DrugLine(
                    name=d.name.strip(),
                    printed_code=(d.printed_code or "").strip() or None,
                    dose_per_time=d.dose_per_time,
                    times_per_day=d.times_per_day,
                    days=d.days,
                    time_slots=slots,
                )
            )
        except ValidationError as e:
            raise StructureError(f"{label}의 값이 올바르지 않습니다.") from e

    try:
        return DispensingOrder(
            prescription_id=prescription_id,
            source=source,
            issue_no=(ex.issue_no or "").strip() or None,
            drugs=lines,
            warnings=warnings,
        )
    except ValidationError as e:
        raise StructureError("조제지시서를 만들지 못했습니다.") from e


def structure_prescription(
    prescription_id: str, ocr_text: str, extract_fn: ExtractFn, *, source: str = "llm"
) -> DispensingOrder:
    if not prescription_id or not prescription_id.strip():
        raise StructureError("처방 ID가 없습니다.")
    if not ocr_text or not ocr_text.strip():
        raise StructureError("OCR 텍스트가 비어 있습니다.")

    try:
        extracted = extract_fn(ocr_text)
    except (StructureError, ConfigError):
        raise
    except Exception as e:
        # 예외 메시지에는 처방전 내용이 섞여 들어올 수 있어 종류만 남긴다.
        raise StructureError(f"LLM 호출에 실패했습니다 ({type(e).__name__}).") from e

    return order_from_extracted(prescription_id.strip(), extracted, source=source)
