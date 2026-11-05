"""structure 노드: OCR 텍스트 → 조제지시서."""
from __future__ import annotations

from app.domain.structuring import ExtractFn, StructureError, structure_prescription
from app.llm import extract_prescription


def structure_node(state: dict, extract_fn: ExtractFn | None = None) -> dict:
    """state["prescription_id"], state["ocr"]를 읽어 조제지시서를 만든다.

    실패하면 예외 대신 status="failed"와 원인(notice)을 돌려준다. (화면에 그대로 보여줄 수 있게)
    단, ConfigError(키 없음 등 개발 환경 문제)는 그대로 올려 보낸다.
    """
    extract_fn = extract_fn or extract_prescription
    try:
        order = structure_prescription(state.get("prescription_id", ""), state.get("ocr", ""), extract_fn)
    except StructureError as e:
        return {"status": "failed", "notice": str(e), "errors": [f"structure: {e}"]}
    return {"order": order.model_dump(mode="json"), "status": "structured", "notice": ""}
