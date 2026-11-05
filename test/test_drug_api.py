"""의약품 정보 API 조회 (agent/drug_api.py). 진짜 API는 부르지 않고 가짜 응답으로 확인한다."""
import json
import urllib.error

import pytest

from agent import config, drug_api
from agent.drug_api import ApiDrug, DrugApiError, parse_products, product_key, search_drugs, search_term
from agent.errors import ConfigError


def product(code="202005623", name="타이레놀정500밀리그램(아세트아미노펜)", company="(주)한국얀센", cancel="정상"):
    return {"ITEM_SEQ": code, "ITEM_NAME": name, "ENTP_NAME": company, "CANCEL_NAME": cancel}


def response(*items, code="00"):
    return {"header": {"resultCode": code, "resultMsg": "NORMAL SERVICE."},
            "body": {"pageNo": 1, "totalCount": len(items), "numOfRows": 20, "items": list(items)}}


class FakeFetch:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, url, params):
        self.calls.append((url, params))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


# ------------------------------------------------------------ 검색어


@pytest.mark.parametrize(
    "name, term",
    [
        ("타이레놀정 5oOmg", "타이레놀정"),             # OCR의 o를 0으로 고친 뒤 함량을 뺀다
        ("타이레놀정/500mg", "타이레놀정"),
        ("타이레놀정 500MG", "타이레놀정"),             # 대문자 단위
        ("타이레놀정500밀리그램(아세트아미노펜)", "타이레놀정"),
        ("뮤코다인캡슐200mg", "뮤코다인캡슐"),
        ("에스코렌정 / 1Omg", "에스코렌정"),
        ("타세놀8시간이알서방정 650mg", "타세놀8시간이알서방정"),   # 이름 속 숫자(8시간)는 함량이 아니라 남는다
        ("게보린정", "게보린정"),
        ("아르펜 정", "아르펜정"),
        ("", ""),
        (None, ""),
        ("500mg", ""),
    ],
)
def test_search_term(name, term):
    assert search_term(name) == term


def test_product_key_drops_the_ingredient_in_parentheses():
    assert product_key("타이레놀정500밀리그램(아세트아미노펜)") == "타이레놀정500mg"
    assert product_key("게보린정") == "게보린정"
    assert product_key("") == ""


# ------------------------------------------------------------ 응답 읽기


def test_products_are_read_from_the_usual_response():
    got = parse_products(response(product(), product("201900001", "게보린정", "삼진제약(주)")))
    assert got == [
        ApiDrug("202005623", "타이레놀정500밀리그램(아세트아미노펜)", "(주)한국얀센"),
        ApiDrug("201900001", "게보린정", "삼진제약(주)"),
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"response": response(product())},                                               # response로 한 번 감싼 모양
        {"header": {"resultCode": "00"}, "body": {"items": {"item": [product()]}}},      # items 안에 item 목록
        {"header": {"resultCode": "00"}, "body": {"items": {"item": product()}}},        # 1건이면 목록이 아닐 수 있다
        {"header": {"resultCode": "00"}, "body": {"items": [product()]}},
        {"body": {"items": [product()]}},                                                # header가 없어도
    ],
)
def test_other_response_shapes_are_read_too(data):
    assert [p.item_code for p in parse_products(data)] == ["202005623"]


@pytest.mark.parametrize("items", [[], "", None, {"item": []}])
def test_no_items_means_no_products(items):
    assert parse_products({"header": {"resultCode": "00"}, "body": {"items": items}}) == []


def test_a_numeric_item_seq_is_read_as_text():
    assert parse_products(response(product(code=202005623)))[0].item_code == "202005623"


@pytest.mark.parametrize(
    "bad",
    [
        product(code="20200562"),         # 8자리
        product(code="2020056230123"),    # 13자리
        product(code=None),
        product(name=""),
        product(name=None),
        product(cancel="취하"),
        product(cancel="취소"),
        "타이레놀정",
    ],
)
def test_unusable_products_are_skipped(bad):
    assert parse_products(response(bad, product("201900001", "게보린정"))) == [ApiDrug("201900001", "게보린정", "(주)한국얀센")]


def test_a_product_without_cancel_or_company_is_kept():
    item = {"ITEM_SEQ": "201900001", "ITEM_NAME": " 게보린정 "}
    assert parse_products(response(item)) == [ApiDrug("201900001", "게보린정", None)]


def test_products_cannot_be_changed():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        parse_products(response(product()))[0].item_code = "999999999"


def test_the_same_code_twice_is_kept_once():
    assert len(parse_products(response(product(), product()))) == 1


@pytest.mark.parametrize(
    "data, said",
    [
        (response(product(), code="30"), "resultCode 30"),
        (None, "형식"),
        ([], "형식"),
        ({"header": {"resultCode": "00"}}, "body"),
        ({"body": {"items": "타이레놀"}}, "items 형식"),
        ({"body": {"items": 3}}, "items 형식"),
    ],
)
def test_bad_responses_are_api_errors(data, said):
    with pytest.raises(DrugApiError, match=said):
        parse_products(data)


# ------------------------------------------------------------ search_drugs


def test_search_sends_the_term_and_the_key_to_the_configured_url():
    fetch = FakeFetch(response(product()))
    got = search_drugs("타이레놀정 5oOmg", fetch=fetch, key="TESTKEY")
    assert [p.item_code for p in got] == ["202005623"]
    url, params = fetch.calls[0]
    assert url == config.DRUG_API_URL
    assert params == {"serviceKey": "TESTKEY", "type": "json", "pageNo": 1, "numOfRows": 20, "item_name": "타이레놀정"}


def test_the_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DRUG_API_KEY", " ENVKEY ")
    fetch = FakeFetch(response())
    search_drugs("게보린정", fetch=fetch)
    assert fetch.calls[0][1]["serviceKey"] == "ENVKEY"


def test_missing_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("DRUG_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="DRUG_API_KEY"):
        search_drugs("게보린정", fetch=FakeFetch(response()))


def test_an_empty_term_does_not_call_the_api():
    fetch = FakeFetch(AssertionError("부르면 안 된다"))
    assert search_drugs("500mg", fetch=fetch, key="K") == [] and fetch.calls == []


def test_fetch_errors_pass_through():
    with pytest.raises(DrugApiError):
        search_drugs("게보린정", fetch=FakeFetch(DrugApiError("x")), key="K")


def test_default_url_and_timeout():
    assert config.DRUG_API_URL.startswith("https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService")
    assert config.DRUG_API_TIMEOUT_SECONDS > 0


# ------------------------------------------------------------ http_get_json (네트워크는 가짜로)


class FakeResponse:
    def __init__(self, raw):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_get_json_reads_json_and_encodes_the_parameters(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["url"], seen["timeout"] = request.full_url, timeout
        return FakeResponse(json.dumps(response(product()), ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr(drug_api.urllib.request, "urlopen", urlopen)
    data = drug_api.http_get_json("https://example.test/api", {"serviceKey": "a+b/c=", "item_name": "게보린정"})
    assert parse_products(data)[0].item_code == "202005623"
    assert seen["url"] == "https://example.test/api?serviceKey=a%2Bb%2Fc%3D&item_name=%EA%B2%8C%EB%B3%B4%EB%A6%B0%EC%A0%95"
    assert seen["timeout"] == config.DRUG_API_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "error, said",
    [
        (urllib.error.HTTPError("https://x?serviceKey=SECRET", 500, "err", {}, None), "HTTP 500"),
        (urllib.error.URLError("no route"), "연결하지 못했습니다"),
        (TimeoutError(), "연결하지 못했습니다"),
    ],
)
def test_http_failures_become_api_errors_without_the_key(monkeypatch, error, said):
    def urlopen(request, timeout):
        raise error

    monkeypatch.setattr(drug_api.urllib.request, "urlopen", urlopen)
    with pytest.raises(DrugApiError, match=said) as caught:
        drug_api.http_get_json("https://x", {"serviceKey": "SECRET"})
    assert "SECRET" not in str(caught.value) and caught.value.__cause__ is None


@pytest.mark.parametrize("raw", [b"<OpenAPI_ServiceResponse><cmmMsgHeader>SERVICE_KEY_IS_NOT_REGISTERED_ERROR", b"\xff\xfe"])
def test_a_non_json_answer_is_an_api_error(monkeypatch, raw):
    monkeypatch.setattr(drug_api.urllib.request, "urlopen", lambda request, timeout: FakeResponse(raw))
    with pytest.raises(DrugApiError, match="인증키"):
        drug_api.http_get_json("https://x", {})
