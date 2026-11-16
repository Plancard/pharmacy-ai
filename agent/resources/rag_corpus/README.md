# RAG 근거 색인 (16단계, 11/16부터 위험 신호 설명의 근거)

`scripts/agent/build_rag_index.py`가 PC에서 실제 식약처 API로 만든다. 손으로 고치지 않는다.

| 파일 | 내용 |
|---|---|
| `docs.jsonl` | 근거 문서. 약 하나의 항목 하나가 한 줄 (`doc_id, item_code, item_name, source, section, url, text, updated`) |
| `embeddings.json` | 조각 임베딩 (`--embed`로 만들 때만). 모델은 `model` 칸. 문서를 다시 만들면 지워진다 |
| `manifest.json` | 만든 시각, 약 목록, 소스별 성공 수 |

출처: 식약처 e약은요(의약품개요정보), 식약처 의약품 제품 허가정보 상세. 공개 허가정보라 git에 올린다.
이 색인은 위험 신호 AI 설명의 근거로 쓴다 (복약 상담 답변은 11/16에 뺐다). `docs.jsonl`이 없으면 근거 없이 규칙 문장 + AI 설명만 보인다.

## pgvector (11/16)

임베딩은 PostgreSQL(pgvector)의 `agent_rag_vectors` 테이블에도 넣을 수 있다. 글과 BM25는 그대로 이 폴더의 `docs.jsonl`을 쓴다.

1. DB에 확장 (DB 관리자, 한 번): `CREATE EXTENSION IF NOT EXISTS vector;`
2. `python scripts/agent/build_rag_index.py --pgvector` (이미 만든 `embeddings.json`을 넣는다. 새로 만들며 넣으려면 `--embed --pgvector`)
3. `.env`에 `RAG_VECTOR_STORE=pgvector`

DB가 안 되면 검색은 멈추지 않고 `embeddings.json` → BM25 순서로 대신한다.
