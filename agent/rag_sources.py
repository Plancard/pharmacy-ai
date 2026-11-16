"""복약 상담 RAG의 근거 문서 받기 (가장자리, 16단계, REQ-AG-06). 식약처 공공데이터 API 두 곳에서 약마다 문서를 받는다.

    easy_docs(item_code)     e약은요(의약품개요정보): 효능 · 사용법 · 주의사항 경고 · 주의사항 · 상호작용 · 부작용 · 보관법
    detail_docs(item_code)   의약품 제품 허가정보 상세: 효능효과 · 용법용량 · 사용상의 주의사항 (허가 문서 원문)

문서 하나 = 약 하나의 항목 하나 (dict):
    {"doc_id", "item_code", "item_name", "source", "section", "url", "text", "updated"}
    source는 출처 이름(화면의 근거 목록에 그대로 보인다), url은 사람이 열어 볼 수 있는 식약처 의약품안전나라 제품 페이지다.

색인은 scripts/agent/build_rag_index.py가 PC에서 이것으로 만들어 agent/resources/rag_corpus/에 둔다. 서비스 중에는 부르지 않는다
(답변할 때 API를 기다리지 않고, API가 죽어도 상담은 된다).
응답 원문의 HTML · XML 태그는 글자만 남긴다. 받은 글을 고치거나 요약하지 않는다 (근거는 원문 그대로).
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any

from agent import config
from agent.drug_api import Fetch, http_get_json, raw_items

EASY_SOURCE = "식약처 e약은요 (의약품개요정보)"
DETAIL_SOURCE = "식약처 의약품 제품 허가정보"
PRODUCT_PAGE = "https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={item_code}"

# e약은요 응답 필드 -> 항목 이름 (data.go.kr 15075057 문서)
EASY_FIELDS = {
    "efcyQesitm": "효능",
    "useMethodQesitm": "사용법",
    "atpnWarnQesitm": "주의사항 경고",
    "atpnQesitm": "주의사항",
    "intrcQesitm": "상호작용",
    "seQesitm": "부작용",
    "depositMethodQesitm": "보관법",
}
# 허가정보 상세 응답 필드 -> 항목 이름. 값은 XML 문서(<DOC><SECTION><ARTICLE title=…><PARAGRAPH>…)이거나 글이다
DETAIL_FIELDS = {
    "EE_DOC_DATA": "효능효과",
    "UD_DOC_DATA": "용법용량",
    "NB_DOC_DATA": "사용상의 주의사항",
}

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t ]+")


def clean_text(value: Any) -> str:
    """HTML 글자: 태그를 빼고, 엔티티를 풀고, 빈칸을 줄인다. 줄바꿈(<br>, </p>)은 줄로 남긴다."""
    text = str(value or "")
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", text)
    text = html.unescape(_TAG.sub(" ", text))
    lines = [_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _plain(text: str) -> str:
    """XML에서 꺼낸 글: 태그는 이미 풀렸으므로 빈칸만 줄인다 ('<간장애>'처럼 본문의 꺾쇠를 태그로 지우지 않는다)."""
    lines = [_SPACE.sub(" ", line).strip() for line in html.unescape(text).splitlines()]
    return "\n".join(line for line in lines if line)


def doc_xml_text(value: Any) -> str:
    """허가 문서 XML(<DOC><SECTION><ARTICLE title="…"><PARAGRAPH>…</PARAGRAPH>)을 글로. 소제목(title)은 줄 앞에 남긴다.

    XML이 아니거나 깨져 있으면 태그만 빼고 글로 읽는다.
    """
    text = str(value or "").strip()
    if not text.startswith("<"):
        return clean_text(text)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return clean_text(text)
    lines: list[str] = []
    for node in root.iter():
        title = (node.get("title") or "").strip()
        if node.tag in ("SECTION", "ARTICLE") and title:
            lines.append(_plain(title))
        if node.tag == "PARAGRAPH":
            body = _plain("".join(node.itertext()))
            if body:
                lines.append(body)
    return "\n".join(line for line in lines if line)


def _doc(item_code: str, item_name: str, source: str, section: str, text: str, updated: str | None, kind: str) -> dict[str, Any]:
    return {
        "doc_id": f"{item_code}:{kind}:{section}",
        "item_code": item_code,
        "item_name": item_name,
        "source": source,
        "section": section,
        "url": PRODUCT_PAGE.format(item_code=item_code),
        "text": text,
        "updated": updated,
    }


def easy_docs(item_code: str, *, key: str | None = None, fetch: Fetch | None = None) -> list[dict[str, Any]]:
    """e약은요에서 이 약의 문서들. 이 약이 e약은요에 없으면(전문의약품 등) 빈 목록. 호출 실패는 DrugApiError."""
    params = {"serviceKey": key or config.require_drug_api_key(), "type": "json", "itemSeq": item_code,
              "pageNo": 1, "numOfRows": 3}
    rows = [r for r in raw_items((fetch or http_get_json)(config.EASY_DRUG_URL, params))
            if str(r.get("itemSeq") or "").strip() == item_code]
    docs = []
    for row in rows[:1]:
        name = clean_text(row.get("itemName")) or item_code
        updated = str(row.get("updateDe") or row.get("openDe") or "").strip() or None
        for field, section in EASY_FIELDS.items():
            text = clean_text(row.get(field))
            if text:
                docs.append(_doc(item_code, name, EASY_SOURCE, section, text, updated, "easy"))
    return docs


def detail_docs(item_code: str, *, key: str | None = None, fetch: Fetch | None = None) -> list[dict[str, Any]]:
    """허가정보 상세에서 이 약의 문서들 (효능효과 · 용법용량 · 사용상의 주의사항). 없으면 빈 목록. 호출 실패는 DrugApiError."""
    params = {"serviceKey": key or config.require_drug_api_key(), "type": "json", "item_seq": item_code,
              "pageNo": 1, "numOfRows": 3}
    rows = [r for r in raw_items((fetch or http_get_json)(config.DRUG_DETAIL_URL, params))
            if str(r.get("ITEM_SEQ") or "").strip() == item_code]
    docs = []
    for row in rows[:1]:
        name = clean_text(row.get("ITEM_NAME")) or item_code
        updated = str(row.get("CHANGE_DATE") or row.get("ITEM_PERMIT_DATE") or "").strip() or None
        for field, section in DETAIL_FIELDS.items():
            text = doc_xml_text(row.get(field))
            if text:
                docs.append(_doc(item_code, name, DETAIL_SOURCE, section, text, updated, "detail"))
    return docs


def missing_fields(row: dict[str, Any], fields) -> list[str]:
    """응답 줄에 없는 필드 이름 (build_rag_index.py가 오퍼레이션 · 필드 이름이 맞는지 알릴 때 쓴다)."""
    return [f for f in fields if f not in row]
