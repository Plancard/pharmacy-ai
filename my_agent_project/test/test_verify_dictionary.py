"""verify_dictionary.py(약품 사전 8종을 식약처 API로 확인하는 스크립트)의 테스트. 진짜 API는 부르지 않는다."""
import pytest

import verify_dictionary
from agent.drug_api import DrugApiError

# 사전 8종: 검색어 -> (품목기준코드, API 제품명)
TRUE_PRODUCTS = {
    "타이레놀정": ("202106092", "타이레놀정500밀리그람(아세트아미노펜)"),
    "게보린정": ("197900277", "게보린정"),
    "트리스펜연질캡슐": ("201110120", "트리스펜연질캡슐(이부프로펜)"),
    "프론크논캡슐": ("200200099", "프론크논캡슐"),
    "아르펜정": ("200201239", "아르펜정"),
    "네프신정": ("201405845", "네프신정"),
    "트립라인정": ("200002714", "트립라인정(브로멜라인)"),
    "뮤코다인캡슐": ("200308774", "뮤코다인캡슐200밀리그램(아세틸시스테인)"),
}


def response(*items):
    return {"header": {"resultCode": "00"}, "body": {"items": [
        {"ITEM_SEQ": code, "ITEM_NAME": name, "ENTP_NAME": "가상제약", "CANCEL_NAME": "정상"} for code, name in items
    ]}}


class FakeFetch:
    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, url, params):
        self.calls.append(params)
        result = self.table.get(params["item_name"], response())
        if isinstance(result, BaseException):
            raise result
        return result


def run(argv=(), table=None):
    lines = []
    fetch = FakeFetch(table if table is not None else {term: response(p) for term, p in TRUE_PRODUCTS.items()})
    code = verify_dictionary.main(list(argv), fetch=fetch, out=lines.append)
    return code, "\n".join(lines), fetch


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("DRUG_API_KEY", "secret-test-key")


def test_every_drug_matches():
    code, text, fetch = run()
    assert code == 0 and "전체 8종 중 8종 일치" in text
    assert "[일치] 타이레놀정 500mg (202106092) = API '타이레놀정500밀리그람(아세트아미노펜)'" in text
    assert [c["item_name"] for c in fetch.calls] == list(TRUE_PRODUCTS)          # 사전 순서대로, 함량을 뺀 이름으로
    assert all(c["serviceKey"] == "secret-test-key" for c in fetch.calls)
    assert "secret-test-key" not in text and "DRUG_API_KEY: 설정됨" in text


def test_a_different_code_is_reported_with_similar_products():
    table = {term: response(p) for term, p in TRUE_PRODUCTS.items()}
    table["아르펜정"] = response(("200299999", "아르펜정"), ("200211111", "아르펜정400밀리그램"))
    code, text, _ = run(table=table)
    assert code == 1 and "전체 8종 중 7종 일치" in text
    assert "[불일치] 아르펜정 (200201239): API 결과에 이 코드가 없습니다. 비슷한 제품:" in text
    assert "    200299999  아르펜정  [가상제약]  유사도 1.000" in text
    assert text.index("200299999") < text.index("200211111")            # 이름이 비슷한 순서


def test_no_product_found():
    table = {term: response(p) for term, p in TRUE_PRODUCTS.items()}
    table["네프신정"] = response()
    code, text, _ = run(table=table)
    assert code == 1 and "[없음] 네프신정 (201405845): API에서 이 이름의 제품을 찾지 못했습니다" in text


def test_a_failure_is_reported_and_the_rest_still_run():
    table = {term: response(p) for term, p in TRUE_PRODUCTS.items()}
    table["게보린정"] = DrugApiError("의약품 API 호출에 실패했습니다 (HTTP 500).")
    code, text, fetch = run(table=table)
    assert code == 1 and "[실패] 게보린정 (197900277): 의약품 API 호출에 실패했습니다 (HTTP 500)." in text
    assert len(fetch.calls) == 8 and "전체 8종 중 7종 일치" in text
    assert "DRUG_API_URL=" not in text                                    # 하나만 실패하면 주소 안내는 없다


def test_when_everything_fails_the_address_hint_is_shown():
    code, text, _ = run(table={term: DrugApiError("의약품 API 응답을 JSON으로 읽을 수 없습니다.") for term in TRUE_PRODUCTS})
    assert code == 1 and "전체 8종 중 0종 일치" in text
    assert "키가 아직 활성화되지 않았거나" in text and "DRUG_API_URL=<End Point>" in text


def test_raw_shows_the_first_response_only_once():
    code, text, _ = run(["--raw"])
    assert code == 0 and text.count("응답 원본") == 1 and "ITEM_SEQ" in text


def test_without_raw_no_response_is_shown():
    _, text, _ = run()
    assert "응답 원본" not in text and "ITEM_SEQ" not in text


def test_missing_key_stops_before_calling(monkeypatch):
    monkeypatch.delenv("DRUG_API_KEY")
    code, text, fetch = run()
    assert code == 1 and fetch.calls == [] and "설정 문제: DRUG_API_KEY가 설정되지 않았습니다" in text
    assert "DRUG_API_KEY: 없음" in text


@pytest.mark.parametrize(
    "status, said",
    [
        (403, "End Point의 버전(…Service08 등)이 위 '주소'와 같은지"),
        (401, "'Decoding' 키인지"),
        (None, "키가 아직 활성화되지 않았거나"),
        (500, "키가 아직 활성화되지 않았거나"),
    ],
)
def test_the_hint_depends_on_the_http_status(status, said):
    table = {term: DrugApiError("의약품 API 호출에 실패했습니다.", status=status) for term in TRUE_PRODUCTS}
    code, text, _ = run(table=table)
    assert code == 1 and said in text


def test_403_hint_checks_both_the_version_and_the_approval():
    table = {term: DrugApiError("x", status=403) for term in TRUE_PRODUCTS}
    _, text, _ = run(table=table)
    assert "End Point의 버전" in text and "상태가 '승인'인지" in text and "키가 아직 활성화되지 않았거나" not in text
