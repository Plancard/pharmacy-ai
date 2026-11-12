"""글로 받은 수정 요청 · 복용 약 정보를 제안으로 만든다 (11단계, 일정표 11/13, REQ-WEB-02).

글은 조제지시서에 자동으로 반영하지 않는다 (명세서 2.2절). 승인 카드에 '제안'으로 보이고, 약사가 확인한 것만
약사가 직접 고쳐서(edit) 반영한다. 제안 하나의 모양 (JSON으로 바꿀 수 있는 dict):
    {"kind": "edit_request" | "patient_info", "text": 받은 글, "drugs": [{"item_code", "name"}, ...]}
drugs는 글에서 이름이 보인 대상 약이다 (약품 사전). 없으면 빈 목록이다. 약을 고른 것이 아니라 글에 보인 것이다.

판단 층이다. 사전을 읽을 뿐 네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from agent.drug_dictionary import DrugDictionary
from agent.drug_names import search_term
from agent.intent import PATIENT_INFO, EDIT_REQUEST, compact

PROPOSAL_KINDS = (EDIT_REQUEST, PATIENT_INFO)
PROPOSAL_LABELS = {EDIT_REQUEST: "수정 제안", PATIENT_INFO: "복용 약 후보"}
_FORM = re.compile(r"(연질캡슐|캡슐|서방정|정)$")     # 이름 끝의 제형: 글에는 보통 빠진다 ('타이레놀 먹어요')
MIN_STEM = 3                                         # 이름 뒷부분이 이만큼 이상 같으면 그 약으로 본다 ('아이월드요비신' ← '요비신')


def stem(name: str) -> str:
    """글에서 약을 찾을 때 쓰는 이름 줄기: 함량 · 괄호 · 제형을 뺀다. '아이월드요비신정 416.7mg' -> '아이월드요비신'"""
    return _FORM.sub("", search_term(name))


def mentioned_drugs(text: str | None, dictionary: DrugDictionary) -> list[dict[str, str]]:
    """글에 이름이 보인 대상 약 (사전 순서). 이름 줄기나 그 뒷부분(3글자 이상)이 글에 있으면 보인 것으로 본다."""
    key = compact(text or "")
    found = []
    for entry in dictionary.entries:
        s = stem(entry.name)
        if any(s[i:] in key for i in range(0, max(1, len(s) - MIN_STEM + 1))):
            found.append({"item_code": entry.item_code, "name": entry.name})
    return found


def make_proposals(text: str | None, intents: Iterable[str], dictionary: DrugDictionary) -> list[dict[str, Any]]:
    """의도 중 수정 요청 · 복용 약 정보를 제안으로. 글이 비었거나 해당 의도가 없으면 빈 목록."""
    if not (text or "").strip():
        return []
    drugs = mentioned_drugs(text, dictionary)
    return [{"kind": kind, "text": text.strip(), "drugs": drugs} for kind in PROPOSAL_KINDS if kind in set(intents)]
