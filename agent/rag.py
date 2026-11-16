"""위험 신호 근거 RAG 검색 (판단, 16단계, REQ-AG-06) — 근거 문서 조각을 찾는다. 새 패키지 없이 파이썬만으로.

    chunk_docs(docs)                       문서를 조각(Chunk)으로 나눈다 (문장 경계, 조각당 CHUNK_CHARS 글자 안팎)
    RagIndex(chunks, vectors=None)         색인. vectors는 조각 ID -> 임베딩 (있으면 섞어 쓴다)
    index.search(query, focus, k, query_vector)  -> list[Passage]
    load_index(folder, vector_search)      build_rag_index.py가 만든 docs.jsonl · embeddings.json 읽기
    RagIndex(chunks, vectors, vector_search)   vector_search가 있으면 임베딩 검색을 그 함수(pgvector, 11/16)로 한다.
                                           실패하면(VectorStoreError) 파일 임베딩(vectors)으로, 그것도 없으면 BM25만으로 찾는다

검색은 두 가지를 섞는다 (11/13 결정: BM25 + 임베딩)
  1. BM25: 낱말 단위. 한글은 띄어쓰기 · 조사가 제각각이라 낱말을 2글자씩 겹쳐 자른 조각(바이그램)도 낱말로 센다.
     약 이름 · 성분명이 든 질문에 강하고, 키 · 네트워크 없이 돈다.
  2. 임베딩(있을 때만): 질문과 조각의 뜻이 가까운 정도 (코사인). '같이 먹어도 돼요?' ↔ '상호작용'처럼 낱말이 달라도 찾는다.
  두 순위를 RRF(1/(60+순위)의 합)로 합친다. 점수 크기가 달라도 순위만 쓰므로 맞출 필요가 없다.
focus(품목기준코드 목록)를 주면 그 약들의 조각에서만 찾는다 (위험 신호에 걸린 처방약). 그 약들의 문서가
하나도 없으면 전체에서 찾는다 (그때 Passage.focused=False로 알려, 다른 약의 근거라는 것이 드러나게 한다).
판단 층이다. 네트워크를 쓰지 않는다 (질문 임베딩은 부르는 쪽이 만들어 넘긴다).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from agent.errors import VectorStoreError

# (질문 임베딩, 품목기준코드들(비면 전부), 개수) -> [(조각 ID, 유사도)] 가까운 순서. 실제로는 vector_store.PgVectorStore.searcher
VectorSearch = Callable[[Sequence[float], Sequence[str], int], list[tuple[str, float]]]
VECTOR_POOL = 50            # 임베딩 검색에서 가져오는 조각 수 (BM25와 순위를 섞을 후보)

CHUNK_CHARS = 400          # 조각 하나의 글자 수 (안팎). 답변 근거로 보이기에 너무 길지 않게
BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60
DOCS_FILE, VECTORS_FILE = "docs.jsonl", "embeddings.json"

_WORD = re.compile(r"[가-힣]+|[a-z0-9]+(?:\.[0-9]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)\s*|\n+")
# 질문에 흔하지만 근거를 가르지 못하는 말 (조사 · 어미 덩어리)
_STOP = {"요", "나요", "되나요", "있나요", "인가요", "까요", "어요", "해요", "이", "가", "을", "를", "은", "는", "에", "의", "도", "와", "과"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    item_code: str
    item_name: str
    source: str
    section: str
    url: str
    text: str


@dataclass(frozen=True)
class Passage:
    chunk: Chunk
    score: float
    focused: bool = True       # focus의 약에서 찾았는가 (False면 다른 약의 문서다)


def tokens(text: str) -> list[str]:
    """검색 낱말: 영문 · 숫자 낱말, 한글 낱말과 그 1 · 2글자 조각.

    한글 1글자 조각은 '술' ↔ '술을'처럼 조사가 붙은 한 글자 낱말을 잇는다 (흔한 글자는 BM25의 idf가 낮춘다).
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    out: list[str] = []
    for word in _WORD.findall(text):
        if word in _STOP:
            continue
        out.append(word)
        if "가" <= word[0] <= "힣" and len(word) > 1:
            out.extend(word)                                          # 1글자 조각
            if len(word) > 2:
                out.extend(word[i:i + 2] for i in range(len(word) - 1))   # 2글자 조각
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def chunk_docs(docs: Iterable[dict[str, Any]], max_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """문서를 문장 경계에서 잘라 조각으로. 한 문장이 max_chars보다 길면 그 문장만 한 조각이다 (문장을 자르지 않는다)."""
    chunks: list[Chunk] = []
    for doc in docs:
        parts: list[str] = []
        current = ""
        for sentence in _sentences(doc["text"]):
            if current and len(current) + 1 + len(sentence) > max_chars:
                parts.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            parts.append(current)
        for i, part in enumerate(parts):
            chunks.append(Chunk(f"{doc['doc_id']}#{i}", doc["doc_id"], doc["item_code"], doc.get("item_name") or "",
                                doc["source"], doc["section"], doc.get("url") or "", part))
    return chunks


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class RagIndex:
    def __init__(self, chunks: Sequence[Chunk], vectors: dict[str, Sequence[float]] | None = None,
                 vector_search: VectorSearch | None = None):
        self.chunks = list(chunks)
        self.vectors = dict(vectors or {})
        self.vector_search = vector_search
        self.vector_errors = 0              # pgvector 검색이 실패해 파일 임베딩 · BM25로 넘어간 횟수 (확인용)
        self._index_of = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        # 항목 이름을 붙여 센다 ('상호작용' 같은 질문이 그 항목을 찾게). 약 이름은 붙이지 않는다: 약은 focus로 고르고,
        # 이름을 붙이면 짧은 항목(보관법)이 이름 덕분에 긴 항목보다 앞선다.
        self._terms = [tokens(f"{c.section} {c.text}") for c in self.chunks]
        self._sets = [set(t) for t in self._terms]
        self._lengths = [len(t) for t in self._terms]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df: dict[str, int] = {}
        for terms in self._terms:
            for term in set(terms):
                self._df[term] = self._df.get(term, 0) + 1

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def has_vectors(self) -> bool:
        return bool(self.vectors) or self.vector_search is not None

    def _nearest(self, query_vector: Sequence[float], pool: list[int], codes: Sequence[str]) -> list[int]:
        """임베딩이 가까운 순서의 조각 번호 (pool 안에서). pgvector가 있으면 DB에서, 실패하거나 없으면 파일 임베딩으로."""
        allowed = set(pool)
        if self.vector_search is not None:
            try:
                hits = self.vector_search(query_vector, list(codes), VECTOR_POOL)
                return [i for chunk_id, _ in hits if (i := self._index_of.get(chunk_id)) is not None and i in allowed]
            except VectorStoreError:
                self.vector_errors += 1
        if not self.vectors:
            return []
        near = {i: _cosine(query_vector, self.vectors[c.chunk_id]) for i in pool
                if (c := self.chunks[i]).chunk_id in self.vectors}
        return sorted(near, key=lambda i: (-near[i], i))

    def item_codes(self) -> set[str]:
        return {c.item_code for c in self.chunks}

    def _bm25(self, query_terms: list[str], i: int) -> float:
        n = len(self.chunks)
        counts: dict[str, int] = {}
        for term in self._terms[i]:
            counts[term] = counts.get(term, 0) + 1
        score = 0.0
        for term in set(query_terms):
            f = counts.get(term, 0)
            if not f:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = f + BM25_K1 * (1 - BM25_B + BM25_B * self._lengths[i] / (self._avg or 1))
            score += idf * f * (BM25_K1 + 1) / norm
        return score

    def search(self, query: str, focus: Iterable[str] = (), k: int = 4,
               query_vector: Sequence[float] | None = None) -> list[Passage]:
        """질문에 맞는 조각 k개. BM25(와 임베딩이 있으면 RRF로 섞은) 순위. 맞는 것이 없으면 빈 목록."""
        focus = set(focus)
        pool = [i for i, c in enumerate(self.chunks) if c.item_code in focus] if focus else []
        focused = bool(pool) or not focus               # 약을 정하지 않고 찾았으면 '다른 약'이 아니다
        if not pool:
            pool = list(range(len(self.chunks)))
        q = tokens(query)
        # 물은 약의 문서가 없어 다른 약에서 찾을 때는 2글자 이상 낱말이 하나는 맞아야 근거로 본다 (한 글자만 겹치는 우연으로
        # 엉뚱한 약의 근거를 대지 않게). 물은 약의 문서 안에서는 '술' 같은 한 글자 낱말로도 찾는다.
        strong = {t for t in q if len(t) >= 2}
        lexical = {i: self._bm25(q, i) for i in pool}
        ranked = [i for i in sorted(pool, key=lambda i: (-lexical[i], i))
                  if lexical[i] > 0 and (focused or strong & self._sets[i])]
        scores = {i: 1 / (RRF_K + r) for r, i in enumerate(ranked, start=1)}
        if query_vector is not None and self.has_vectors:
            codes = sorted(focus) if focus and focused else []
            for r, i in enumerate(self._nearest(query_vector, pool, codes), start=1):
                scores[i] = scores.get(i, 0.0) + 1 / (RRF_K + r)
        if not focused and not ranked:
            return []                                    # 물은 약의 문서도 없고 낱말도 안 맞으면 엉뚱한 근거를 대지 않는다
        best = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [Passage(self.chunks[i], round(scores[i], 6), focused) for i in best]


EVIDENCE_TEXT = 400        # 근거 조각 하나를 도구 결과로 돌려주는 글자 수


def evidence_search(rag: RagIndex, query: str, item_codes: Sequence[str] = (), k: int = 5,
                    embed: Callable[[list[str]], list[list[float]]] | None = None) -> list[dict[str, Any]]:
    """근거 검색 도구(MCP search_evidence · AI 조사)가 돌려주는 줄들. 임베딩이 되면 섞고, 안 되면 BM25만.

    focused가 False인 줄은 물은 약의 문서가 없어 다른 약에서 찾은 것이다 (근거로 쓰지 않는 쪽이 안전하다).
    """
    vector = None
    if embed is not None and rag.has_vectors:
        try:
            vector = embed([query])[0]
        except Exception:                    # 키가 없거나 네트워크가 안 되면 BM25만
            vector = None
    return [{"chunk_id": p.chunk.chunk_id, "item_code": p.chunk.item_code, "drug": p.chunk.item_name,
             "section": p.chunk.section, "source": p.chunk.source, "url": p.chunk.url,
             "text": p.chunk.text[:EVIDENCE_TEXT], "focused": p.focused, "score": p.score}
            for p in rag.search(query, focus=item_codes, k=k, query_vector=vector)]


def load_index(folder: Path, vector_search: VectorSearch | None = None) -> RagIndex | None:
    """색인 폴더를 읽는다. 문서 파일이 없으면 None (색인을 아직 만들지 않았다). 임베딩 파일은 있으면 읽는다."""
    docs_path = Path(folder) / DOCS_FILE
    if not docs_path.exists():
        return None
    docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors_path = Path(folder) / VECTORS_FILE
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))["vectors"] if vectors_path.exists() else None
    return RagIndex(chunk_docs(docs), vectors, vector_search)
