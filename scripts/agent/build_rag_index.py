"""RAG 색인 만들기 (16단계, 11/16부터 위험 신호 설명의 근거) — PC에서 실제 식약처 API로 근거 문서를 받아 agent/resources/rag_corpus/에 둔다.

    python scripts/agent/build_rag_index.py                      # 사전 8종: e약은요 + 허가정보 상세 → docs.jsonl · manifest.json
    python scripts/agent/build_rag_index.py --extra 199900002    # 사전 밖 약도 (복용 약 시연용 등, 품목기준코드)
    python scripts/agent/build_rag_index.py --embed              # 임베딩도 만든다 (OPENAI_API_KEY, 모델 EMBED_MODEL) → embeddings.json
    python scripts/agent/build_rag_index.py --save-raw           # 응답 원본을 var/api/rag_*.json 으로 저장 (필드 이름 확인용)
    python scripts/agent/build_rag_index.py --probe              # 허가정보 상세의 오퍼레이션 이름 후보를 한 약으로 차례로 시험한다
    python scripts/agent/build_rag_index.py --embed --pgvector   # 만들고 임베딩을 PostgreSQL(pgvector)에도 넣는다 (DATABASE_URL)
    python scripts/agent/build_rag_index.py --pgvector           # 이미 만든 임베딩(embeddings.json)만 pgvector에 넣는다 (API를 부르지 않는다)
    pgvector를 쓰려면 .env에 RAG_VECTOR_STORE=pgvector. DB에 확장이 없으면 DB 관리자가 CREATE EXTENSION IF NOT EXISTS vector; 를 한 번.

pytest가 아니다. .env의 DRUG_API_KEY로 부른다 (e약은요 · 허가정보는 공공데이터포털에서 각각 활용신청해야 한다).
- 한 소스가 실패해도(403 등) 다른 소스로 계속 만든다. 끝에 소스별 성공 · 실패와 확인할 것을 알린다.
- 응답에 쓰는 필드가 없으면 이름을 알린다 (오퍼레이션 · 필드 이름이 바뀌었을 때 agent/rag_sources.py 위쪽 표를 고친다).
- 키는 화면에도 파일에도 적지 않는다. 만든 파일은 공개 허가정보라 git에 올려도 된다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import config, rag_sources  # noqa: E402
from agent.drug_api import Fetch, http_get_json, raw_items  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.errors import ConfigError, DrugApiError, VectorStoreError  # noqa: E402
from agent.rag import DOCS_FILE, VECTORS_FILE, chunk_docs  # noqa: E402
from agent.vector_store import PgVectorStore  # noqa: E402

RAW_DIR = ROOT / "var" / "api"
Out = Callable[[str], None]
SOURCES = (
    ("e약은요", rag_sources.easy_docs, lambda: config.EASY_DRUG_URL, rag_sources.EASY_FIELDS),
    ("허가정보 상세", rag_sources.detail_docs, lambda: config.DRUG_DETAIL_URL, rag_sources.DETAIL_FIELDS),
)


def mask(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def recording(fetch: Fetch, raw: dict[str, Any]) -> Fetch:
    """응답 원본을 모은다 (마지막 응답이 필드 확인에 쓰인다)."""
    def call(url: str, params: dict[str, Any]) -> Any:
        data = fetch(url, params)
        raw[url.rsplit("/", 1)[-1]] = data
        return data
    return call


def build(codes: list[str], key: str, fetch: Fetch, out: Out, raw_dir: Path | None) -> tuple[list[dict[str, Any]], dict]:
    docs: list[dict[str, Any]] = []
    report: dict[str, dict[str, Any]] = {name: {"drugs": 0, "docs": 0, "failed": [], "missing": set()} for name, *_ in SOURCES}
    for code in codes:
        for name, get_docs, url_of, fields in SOURCES:
            raw: dict[str, Any] = {}
            try:
                got = get_docs(code, key=key, fetch=recording(fetch, raw))
            except DrugApiError as e:
                report[name]["failed"].append(f"{code}: {mask(str(e), key)}")
                continue
            for data in raw.values():
                try:
                    rows = raw_items(data)
                except DrugApiError:
                    rows = []
                if rows:
                    report[name]["missing"].update(rag_sources.missing_fields(rows[0], fields))
                if raw_dir is not None:
                    (raw_dir / f"rag_{name.replace(' ', '_')}_{code}.json").write_text(
                        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            if got:
                report[name]["drugs"] += 1
                report[name]["docs"] += len(got)
                docs += got
        out(f"  {code}: 문서 {sum(1 for d in docs if d['item_code'] == code)}개")
    return docs, report


# 허가정보 상세 오퍼레이션 이름 후보 (공개 문서에서 확인하지 못해 PC에서 시험한다). 목록 오퍼레이션도 문서 필드를 주는지 본다.
PROBE_OPERATIONS = ("getDrugPrdtPrmsnDtlInq08", "getDrugPrdtPrmsnDtlInq07", "getDrugPrdtPrmsnDtlInq06",
                    "getDrugPrdtPrmsnDtlInq05", "getDrugPrdtPrmsnDtlInq04", "getDrugPrdtPrmsnInq08")
PROBE_DRUG = "202106092"      # 타이레놀정500mg


def probe(key: str, fetch: Fetch, out: Out) -> int:
    """오퍼레이션 이름 후보마다 한 번씩 부르고, 문서 필드(EE_DOC_DATA 등)를 주는 첫 이름을 .env 줄로 알려 준다."""
    base = config.DRUG_DETAIL_URL.rsplit("/", 1)[0]
    out(f"서비스 주소: {base} | 시험 약: {PROBE_DRUG}")
    found = None
    for operation in PROBE_OPERATIONS:
        url = f"{base}/{operation}"
        params = {"serviceKey": key, "type": "json", "item_seq": PROBE_DRUG, "pageNo": 1, "numOfRows": 3}
        try:
            rows = raw_items(fetch(url, params))
        except DrugApiError as e:
            out(f"  {operation}: 실패 - {mask(str(e), key)}")
            continue
        fields = [f for f in rag_sources.DETAIL_FIELDS if any(f in r for r in rows)]
        mine = [r for r in rows if str(r.get("ITEM_SEQ") or "").strip() == PROBE_DRUG]
        out(f"  {operation}: 응답 {len(rows)}줄 (이 약 {len(mine)}줄) · 문서 필드 {', '.join(fields) or '없음'}")
        if mine and fields and found is None:
            found = url
    if found:
        out(f"쓸 수 있는 주소를 찾았습니다. .env에 이 줄을 넣고 다시 만드세요:\nDRUG_DETAIL_URL={found}")
        return 0
    out("문서 필드를 주는 오퍼레이션을 찾지 못했습니다. 공공데이터포털 마이페이지 → 활용신청 현황 → 이 서비스의 상세 → "
        "'상세기능정보'에 있는 오퍼레이션 이름을 알려 주세요 (e약은요만으로도 색인은 만들어집니다).")
    return 1


def push_pgvector(folder: Path, out: Out, store_factory: Callable[[str], Any] = PgVectorStore) -> int:
    """색인 폴더의 임베딩(embeddings.json)을 pgvector 테이블에 통째로 넣는다. 성공하면 0."""
    url = config.database_url()
    if not url:
        out("DATABASE_URL이 없어 pgvector에 넣지 못했습니다 (.env에 웹과 같은 DB 주소를 넣으세요).")
        return 1
    docs_path, vectors_path = folder / DOCS_FILE, folder / VECTORS_FILE
    if not (docs_path.exists() and vectors_path.exists()):
        out(f"{folder}에 {DOCS_FILE} · {VECTORS_FILE}이 없습니다. 먼저 --embed로 만드세요.")
        return 1
    docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(vectors_path.read_text(encoding="utf-8"))
    vectors, model = data.get("vectors") or {}, data.get("model") or config.EMBED_MODEL
    rows = [(c.chunk_id, c.item_code, vectors[c.chunk_id]) for c in chunk_docs(docs) if c.chunk_id in vectors]
    if not rows:
        out("넣을 임베딩이 없습니다 (문서와 임베딩의 조각 번호가 맞지 않습니다. --embed로 다시 만드세요).")
        return 1
    if model != config.EMBED_MODEL:
        out(f"주의: 임베딩 모델({model})이 지금 설정(EMBED_MODEL={config.EMBED_MODEL})과 다릅니다. 검색은 설정의 모델로 찾습니다.")
    store = store_factory(url)
    try:
        store.setup(len(rows[0][2]))
        count = store.replace_all(model, rows)
    except (VectorStoreError, ConfigError) as e:
        out(f"pgvector에 넣지 못했습니다: {e}")
        return 1
    out(f"pgvector: {store.table} 테이블에 조각 {count}개를 넣었습니다 (모델 {model}, {len(rows[0][2])}차원). "
        f"쓰려면 .env에 RAG_VECTOR_STORE=pgvector")
    return 0


def main(argv: list[str] | None = None, *, fetch: Fetch = http_get_json, out: Out = print,
         embed: Callable[[list[str]], list[list[float]]] | None = None,
         store_factory: Callable[[str], Any] = PgVectorStore) -> int:
    parser = argparse.ArgumentParser(description="복약 상담 RAG 색인 만들기 (식약처 e약은요 + 허가정보 상세)")
    parser.add_argument("--extra", nargs="*", default=[], help="사전 밖 약의 품목기준코드 (9자리)")
    parser.add_argument("--embed", action="store_true", help="임베딩도 만든다 (OPENAI_API_KEY)")
    parser.add_argument("--save-raw", action="store_true", help="응답 원본을 var/api/에 저장")
    parser.add_argument("--out", type=Path, default=config.RAG_DIR, help="색인 폴더 (기본 agent/resources/rag_corpus)")
    parser.add_argument("--probe", action="store_true", help="허가정보 상세의 오퍼레이션 이름 후보를 시험한다 (색인은 만들지 않음)")
    parser.add_argument("--pgvector", action="store_true",
                        help="임베딩을 PostgreSQL(pgvector)에 넣는다. --embed 없이 쓰면 이미 만든 임베딩만 넣는다")
    args = parser.parse_args(argv)
    if args.pgvector and not args.embed:
        return push_pgvector(args.out, out, store_factory)

    bad = [c for c in args.extra if not (c.isdigit() and len(c) == 9)]
    if bad:
        out(f"품목기준코드는 9자리 숫자여야 합니다: {', '.join(bad)}")
        return 1
    out(f"e약은요: {config.EASY_DRUG_URL}\n허가정보 상세: {config.DRUG_DETAIL_URL}\n"
        f"DRUG_API_KEY: {'설정됨' if config.drug_api_key() else '없음'}")
    try:
        key = config.require_drug_api_key()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1
    if args.probe:
        return probe(key, fetch, out)
    codes = list(dict.fromkeys([e.item_code for e in load_dictionary().entries] + args.extra))
    raw_dir = RAW_DIR if args.save_raw else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    out(f"약 {len(codes)}개의 문서를 받습니다")
    docs, report = build(codes, key, fetch, out, raw_dir)

    for name, r in report.items():
        line = f"[{name}] 약 {r['drugs']}/{len(codes)}개 · 문서 {r['docs']}개"
        if r["failed"]:
            line += f" · 실패 {len(r['failed'])}건 (예: {r['failed'][0]})"
        out(line)
        if r["missing"]:
            out(f"    응답에 없는 필드: {', '.join(sorted(r['missing']))} (agent/rag_sources.py의 필드 이름을 확인하세요)")
        if r["failed"] and any("403" in f for f in r["failed"]):
            out("    HTTP 403이면 공공데이터포털에서 이 서비스의 활용신청이 '승인'인지, 활용신청 상세의 End Point · 오퍼레이션이"
                " 위 주소와 같은지 확인하세요 (다르면 .env에 EASY_DRUG_URL=… / DRUG_DETAIL_URL=…).")
    if not docs:
        out("받은 문서가 없어 색인을 만들지 않았습니다 (기존 색인은 그대로).")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    docs.sort(key=lambda d: (d["item_code"], d["doc_id"]))
    (args.out / DOCS_FILE).write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    chunks = chunk_docs(docs)
    manifest = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "drugs": codes, "documents": len(docs), "chunks": len(chunks),
                "sources": {name: {"drugs": r["drugs"], "documents": r["docs"], "failed": len(r["failed"])} for name, r in report.items()},
                "embedding_model": None}
    if args.embed:
        from agent.embed import embed_texts

        try:
            vectors = (embed or embed_texts)([f"{c.item_name} {c.section}: {c.text}" for c in chunks])
        except ConfigError as e:
            out(f"임베딩 설정 문제: {e} (문서 색인만 만들었습니다)")
            vectors = None
        except Exception as e:
            out(f"임베딩을 만들지 못했습니다 ({type(e).__name__}). 문서 색인만 만들었습니다 (상담은 키워드 검색으로 돈다).")
            vectors = None
        if vectors is not None:
            data = {"model": config.EMBED_MODEL,
                    "vectors": {c.chunk_id: [round(x, 5) for x in v] for c, v in zip(chunks, vectors)}}
            (args.out / VECTORS_FILE).write_text(json.dumps(data), encoding="utf-8")
            manifest["embedding_model"] = config.EMBED_MODEL
    elif (args.out / VECTORS_FILE).exists():
        (args.out / VECTORS_FILE).unlink()          # 문서가 바뀌었으니 예전 임베딩은 맞지 않는다
        out("문서가 바뀌어 예전 임베딩 파일을 지웠습니다 (--embed로 다시 만드세요).")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    out(f"색인: {args.out} | 문서 {len(docs)}개 · 조각 {len(chunks)}개 · 임베딩 {'있음' if manifest['embedding_model'] else '없음'}")
    if args.pgvector:
        return push_pgvector(args.out, out, store_factory)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
