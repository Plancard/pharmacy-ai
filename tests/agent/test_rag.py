"""근거 문서 RAG (16단계 → 11/16부터 위험 신호 설명의 근거): 문서 받기(rag_sources) · 검색(rag) · 색인 스크립트.

진짜 API · OpenAI는 부르지 않는다. 시험 문서의 글은 시험용으로 지어낸 것이다 (실제 허가정보가 아니다).
"""
import importlib.util
import json
from pathlib import Path

import pytest

from agent import config
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.errors import StructureError
from agent.rag import RagIndex, chunk_docs, load_index, tokens
from agent.rag_sources import DETAIL_SOURCE, EASY_SOURCE, clean_text, detail_docs, doc_xml_text, easy_docs

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
TYLENOL, GEVORIN, TRISPEN = "202106092", "197900277", "201110120"
KEY = "비밀키1234"


def doc(code, name, section, text, source=EASY_SOURCE):
    return {"doc_id": f"{code}:t:{section}", "item_code": code, "item_name": name, "source": source, "section": section,
            "url": f"https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={code}", "text": text, "updated": None}


DOCS = [
    doc(TYLENOL, "타이레놀정500mg", "상호작용", "(시험용) 술을 마시면 간 손상이 생길 수 있습니다. 다른 해열진통제와 함께 먹지 마십시오."),
    doc(TYLENOL, "타이레놀정500mg", "보관법", "(시험용) 실온에 보관하십시오."),
    doc(GEVORIN, "게보린정", "부작용", "(시험용) 발진, 구역이 나타날 수 있습니다."),
    doc(TRISPEN, "트리스펜연질캡슐", "주의사항", "(시험용) 위장 출혈이 있는 사람은 먹기 전에 상의하십시오."),
]
INDEX = RagIndex(chunk_docs(DOCS))


# ------------------------------------------------------------ 문서 받기 (응답 모양)


def easy_response(code=TYLENOL, **fields):
    row = {"itemSeq": code, "itemName": "타이레놀정500밀리그람", "updateDe": "2024-01-01",
           "efcyQesitm": "<p>이 약은 해열 및 진통에 사용합니다.</p>", "intrcQesitm": "술&amp;다른 진통제와 함께 주의<br/>하십시오.",
           "seQesitm": None, **fields}
    return {"header": {"resultCode": "00"}, "body": {"pageNo": 1, "totalCount": 1, "items": [row]}}


NB_XML = ('<DOC title="사용상의 주의사항"><SECTION title=""><ARTICLE title="1. 경고"><PARAGRAPH>매일 세 잔 이상 술을 마시는 사람은'
          ' 의사와 상의할 것</PARAGRAPH></ARTICLE><ARTICLE title="2. 다음 환자에는 투여하지 말 것"><PARAGRAPH>&lt;간장애&gt; 환자'
          '</PARAGRAPH></ARTICLE></SECTION></DOC>')


def detail_response(code=TYLENOL, **fields):
    row = {"ITEM_SEQ": code, "ITEM_NAME": "타이레놀정500밀리그람(아세트아미노펜)", "EE_DOC_DATA": "<DOC><SECTION><ARTICLE title=\"해열\"/>"
           "</SECTION></DOC>", "UD_DOC_DATA": "", "NB_DOC_DATA": NB_XML, **fields}
    return {"response": {"header": {"resultCode": "00"}, "body": {"items": [row]}}}


def test_easy_docs_become_one_document_per_section():
    seen = []
    docs = easy_docs(TYLENOL, key=KEY, fetch=lambda url, params: seen.append((url, params)) or easy_response())
    assert [d["section"] for d in docs] == ["효능", "상호작용"]                       # 빈 항목(부작용 None)은 없다
    assert docs[1]["text"] == "술&다른 진통제와 함께 주의\n하십시오."
    assert docs[0]["source"] == EASY_SOURCE and docs[0]["url"].endswith(f"itemSeq={TYLENOL}")
    assert seen[0][0] == config.EASY_DRUG_URL and seen[0][1]["itemSeq"] == TYLENOL and seen[0][1]["type"] == "json"


def test_a_different_product_in_the_answer_is_not_used():
    assert easy_docs(TYLENOL, key=KEY, fetch=lambda u, p: easy_response(code="200000000")) == []


def test_detail_docs_read_the_permission_document_xml():
    docs = detail_docs(TYLENOL, key=KEY, fetch=lambda u, p: detail_response())
    by = {d["section"]: d["text"] for d in docs}
    assert by["사용상의 주의사항"] == ("1. 경고\n매일 세 잔 이상 술을 마시는 사람은 의사와 상의할 것\n"
                                     "2. 다음 환자에는 투여하지 말 것\n<간장애> 환자")           # 본문의 꺾쇠는 지우지 않는다
    assert by["효능효과"] == "해열" and "용법용량" not in by and docs[0]["source"] == DETAIL_SOURCE


def test_text_that_is_not_xml_or_is_broken_is_still_read():
    assert doc_xml_text("그냥 글") == "그냥 글" and doc_xml_text("") == ""
    assert doc_xml_text("<DOC><ARTICLE>깨진 문서") == "깨진 문서"                 # 끝나지 않은 XML: 태그만 빼고 글로


# ------------------------------------------------------------ 검색


def test_korean_words_are_split_into_one_and_two_letter_pieces():
    assert tokens("술을 마셔도") == ["술을", "술", "을", "마셔도", "마", "셔", "도", "마셔", "셔도"]
    assert tokens("Ibuprofen 400mg") == ["ibuprofen", "400mg"]


def test_long_documents_are_cut_at_sentences():
    long = doc(TYLENOL, "타이레놀정500mg", "주의사항", " ".join(f"문장{i}입니다." for i in range(200)))
    chunks = chunk_docs([long], max_chars=100)
    assert len(chunks) > 5 and all(len(c.text) <= 100 for c in chunks)
    assert " ".join(c.text for c in chunks) == " ".join(f"문장{i}입니다." for i in range(200))
    assert chunks[1].chunk_id == f"{TYLENOL}:t:주의사항#1"


def test_the_matching_section_of_the_asked_drug_comes_first():
    found = INDEX.search("타이레놀 먹고 술 마셔도 되나요?", focus=[TYLENOL])
    assert found[0].chunk.section == "상호작용" and all(p.chunk.item_code == TYLENOL for p in found)


def test_focus_keeps_other_drugs_out():
    assert all(p.chunk.item_code == GEVORIN for p in INDEX.search("부작용", focus=[GEVORIN]))


def test_a_drug_without_documents_falls_back_to_all_and_says_so():
    found = INDEX.search("부작용", focus=["999999999"])
    assert found and not found[0].focused
    assert INDEX.search("우주선 발사", focus=["999999999"]) == []                     # 낱말도 안 맞으면 엉뚱한 근거를 대지 않는다


def test_embeddings_are_mixed_in_when_both_sides_have_them():
    chunks = chunk_docs(DOCS)
    vectors = {c.chunk_id: [1.0, 0.0] if c.section == "보관법" else [0.0, 1.0] for c in chunks}
    index = RagIndex(chunks, vectors)
    plain = index.search("술", focus=[TYLENOL])
    mixed = index.search("술", focus=[TYLENOL], query_vector=[1.0, 0.0])
    assert plain[0].chunk.section == "상호작용" and {p.chunk.section for p in mixed} == {"상호작용", "보관법"}
    assert index.has_vectors and not INDEX.has_vectors


def test_the_index_is_read_from_its_folder(tmp_path):
    assert load_index(tmp_path) is None
    (tmp_path / "docs.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in DOCS), encoding="utf-8")
    index = load_index(tmp_path)
    assert len(index) == len(DOCS) and not index.has_vectors and index.item_codes() == {TYLENOL, GEVORIN, TRISPEN}


# ------------------------------------------------------------ 색인 스크립트 (build_rag_index.py)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "build_rag_index.py"
_spec = importlib.util.spec_from_file_location("build_rag_index", SCRIPT)
build_rag_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_rag_index)


def fake_fetch(url, params):
    assert params["serviceKey"] == KEY
    code = params.get("itemSeq") or params.get("item_seq")
    if url == config.EASY_DRUG_URL:
        return easy_response(code=code) if code == TYLENOL else {"header": {"resultCode": "00"}, "body": {"items": ""}}
    return detail_response(code=code)


def run(argv, monkeypatch, fetch=fake_fetch, **kw):
    monkeypatch.setenv("DRUG_API_KEY", KEY)
    lines = []
    code = build_rag_index.main(argv, fetch=fetch, out=lines.append, **kw)
    return code, "\n".join(lines)


def test_the_index_is_built_from_both_sources(monkeypatch, tmp_path):
    code, text = run(["--out", str(tmp_path)], monkeypatch)
    assert code == 0 and "[e약은요] 약 1/8개 · 문서 2개" in text and "[허가정보 상세] 약 8/8개" in text
    docs = [json.loads(l) for l in (tmp_path / "docs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {d["item_code"] for d in docs} == {e.item_code for e in DICT.entries}
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["embedding_model"] is None
    assert len(load_index(tmp_path)) >= len(docs)
    assert KEY not in text


def test_a_failing_source_does_not_stop_the_other_and_403_is_explained(monkeypatch, tmp_path):
    from agent.errors import DrugApiError

    def forbidden(url, params):
        if url == config.DRUG_DETAIL_URL:
            raise DrugApiError("의약품 API가 HTTP 403을 돌려주었습니다.", status=403)
        return fake_fetch(url, params)

    code, text = run(["--out", str(tmp_path)], monkeypatch, forbidden)
    assert code == 0 and "[허가정보 상세] 약 0/8개 · 문서 0개 · 실패 8건" in text and "활용신청" in text


def test_missing_fields_are_named(monkeypatch, tmp_path):
    def renamed(url, params):
        data = fake_fetch(url, params)
        if url == config.DRUG_DETAIL_URL:
            row = data["response"]["body"]["items"][0]
            row["NB_DOC"] = row.pop("NB_DOC_DATA")
        return data

    _, text = run(["--out", str(tmp_path)], monkeypatch, renamed)
    assert "응답에 없는 필드: NB_DOC_DATA" in text


def test_embeddings_are_written_when_asked(monkeypatch, tmp_path):
    code, text = run(["--out", str(tmp_path), "--embed"], monkeypatch, embed=lambda texts: [[0.123456789, 1.0] for _ in texts])
    data = json.loads((tmp_path / "embeddings.json").read_text(encoding="utf-8"))
    assert code == 0 and data["model"] == config.EMBED_MODEL and next(iter(data["vectors"].values())) == [0.12346, 1.0]
    assert load_index(tmp_path).has_vectors
    code, text = run(["--out", str(tmp_path)], monkeypatch)                  # 다시 만들면 예전 임베딩은 지운다
    assert not (tmp_path / "embeddings.json").exists() and "예전 임베딩" in text


def test_nothing_received_keeps_the_old_index(monkeypatch, tmp_path):
    (tmp_path / "docs.jsonl").write_text("옛 색인", encoding="utf-8")
    empty = lambda url, params: {"header": {"resultCode": "00"}, "body": {"items": ""}}     # noqa: E731
    code, text = run(["--out", str(tmp_path)], monkeypatch, empty)
    assert code == 1 and (tmp_path / "docs.jsonl").read_text(encoding="utf-8") == "옛 색인"


@pytest.mark.parametrize("argv, words", [(["--extra", "12345"], "9자리"), ([], "설정 문제")])
def test_wrong_input_or_no_key_is_explained(monkeypatch, argv, words):
    if words == "설정 문제":
        monkeypatch.delenv("DRUG_API_KEY", raising=False)
        lines = []
        assert build_rag_index.main(argv, fetch=fake_fetch, out=lines.append) == 1 and words in "\n".join(lines)
        return
    code, text = run(argv, monkeypatch)
    assert code == 1 and words in text


def test_probe_finds_the_operation_that_gives_document_fields(monkeypatch):
    from agent.errors import DrugApiError

    def fetch(url, params):
        if url.endswith("getDrugPrdtPrmsnDtlInq08"):
            raise DrugApiError("의약품 API 호출에 실패했습니다 (HTTP 400).", status=400)
        if url.endswith("getDrugPrdtPrmsnDtlInq07"):
            return detail_response(code=params["item_seq"])
        return {"header": {"resultCode": "00"}, "body": {"items": ""}}

    code, text = run(["--probe"], monkeypatch, fetch)
    assert code == 0 and "getDrugPrdtPrmsnDtlInq08: 실패 - 의약품 API 호출에 실패했습니다 (HTTP 400)." in text
    assert "getDrugPrdtPrmsnDtlInq07: 응답 1줄 (이 약 1줄) · 문서 필드 EE_DOC_DATA, UD_DOC_DATA, NB_DOC_DATA" in text
    assert text.endswith("DRUG_DETAIL_URL=https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08/getDrugPrdtPrmsnDtlInq07")


def test_probe_says_where_to_look_when_nothing_works(monkeypatch):
    code, text = run(["--probe"], monkeypatch, lambda url, params: {"header": {"resultCode": "00"}, "body": {"items": ""}})
    assert code == 1 and "상세기능정보" in text and KEY not in text
