"""조건부 분기 (11/6 뼈대)."""
from __future__ import annotations

from typing import Any, Literal


def after_structure(state: dict[str, Any]) -> Literal["match", "end"]:
    """구조화에 성공했을 때만 약품 매칭으로 간다. 그 밖의 모든 경우(실패, 상태 이상)는 끝낸다."""
    if state.get("status") == "structured" and state.get("order") is not None:
        return "match"
    return "end"
