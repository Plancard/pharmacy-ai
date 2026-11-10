"""의약품 정보 API 조회 (agent/drug_api.py). 진짜 API는 부르지 않고 가짜 응답으로 확인한다."""
import json
import traceback
import urllib.error

import pytest

from agent import config, drug_api
from agent.drug_api import ApiDrug, parse_products, search_drugs
from agent.errors import ConfigError, DrugApiError


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
    got = search_drugs("타이레놀정", fetch=fetch, key="TESTKEY")
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


@pytest.mark.parametrize("term", ["", "   ", None])
def test_an_empty_term_does_not_call_the_api(term):
    fetch = FakeFetch(AssertionError("부르면 안 된다"))
    assert search_drugs(term, fetch=fetch, key="K") == [] and fetch.calls == []


def test_the_term_is_sent_as_it_is_without_changing_it():
    # 검색어 만들기(함량 빼기)는 판단 층의 drug_names.search_term이 한다. 가장자리는 받은 말을 그대로 보낸다
    fetch = FakeFetch(response())
    search_drugs("타이레놀정 500mg", fetch=fetch, key="K")
    assert fetch.calls[0][1]["item_name"] == "타이레놀정 500mg"


def test_fetch_errors_pass_through():
    with pytest.raises(DrugApiError):
        search_drugs("게보린정", fetch=FakeFetch(DrugApiError("x")), key="K")


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
    # 오류가 끝내 화면에 트레이스백으로 나가도, 키가 든 원래 오류(URL)는 함께 찍히지 않는다
    shown = "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__ and "serviceKey=SECRET" not in shown


@pytest.mark.parametrize("raw", [b"<OpenAPI_ServiceResponse><cmmMsgHeader>SERVICE_KEY_IS_NOT_REGISTERED_ERROR", b"\xff\xfe"])
def test_a_non_json_answer_is_an_api_error(monkeypatch, raw):
    monkeypatch.setattr(drug_api.urllib.request, "urlopen", lambda request, timeout: FakeResponse(raw))
    with pytest.raises(DrugApiError, match="인증키"):
        drug_api.http_get_json("https://x", {})


# ------------------------------------------------------------ default_api: 매칭에 넘기는 조회 함수


def test_default_api_is_none_without_a_key():
    # conftest가 테스트 동안 DRUG_API_KEY를 지운다
    from agent.drug_api import default_api

    assert default_api() is None


def test_default_api_searches_with_the_key(monkeypatch):
    from agent.drug_api import default_api

    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    fetch = FakeFetch(response(product()))
    search = default_api(fetch=fetch)
    assert [p.item_code for p in search("타이레놀정")] == ["202005623"]
    assert fetch.calls[0][1]["serviceKey"] == "test-key" and fetch.calls[0][1]["item_name"] == "타이레놀정"


def test_default_api_asks_each_search_term_only_once(monkeypatch):
    from agent.drug_api import default_api

    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    fetch = FakeFetch(response(product()))
    search = default_api(fetch=fetch)
    search("타이레놀정")
    search("타이레놀정")                  # 같은 검색어
    search("게보린정")
    assert [c[1]["item_name"] for c in fetch.calls] == ["타이레놀정", "게보린정"]


def test_default_api_does_not_remember_failures(monkeypatch):
    from agent.drug_api import default_api

    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    fetch = FakeFetch(DrugApiError("의약품 API에 연결하지 못했습니다 (URLError)."))
    search = default_api(fetch=fetch)
    with pytest.raises(DrugApiError):
        search("게보린정")
    fetch.result = response(product("197900277", "게보린정"))
    assert [p.item_code for p in search("게보린정")] == ["197900277"]     # 다시 묻는다
    assert len(fetch.calls) == 2


def test_remembered_results_cannot_be_changed_by_the_caller(monkeypatch):
    from agent.drug_api import default_api

    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    search = default_api(fetch=FakeFetch(response(product())))
    search("타이레놀정").clear()
    assert len(search("타이레놀정")) == 1


def test_each_default_api_has_its_own_memory(monkeypatch):
    from agent.drug_api import default_api

    monkeypatch.setenv("DRUG_API_KEY", "test-key")
    fetch = FakeFetch(response(product()))
    default_api(fetch=fetch)("타이레놀정")
    default_api(fetch=fetch)("타이레놀정")
    assert len(fetch.calls) == 2


# ------------------------------------------------------------ 서버의 거절 사유 (11/9: 실제 키로 HTTP 403)


@pytest.mark.parametrize(
    "raw, reason",
    [
        (b"<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>SERVICE ERROR</errMsg>"
         b"<returnAuthMsg>SERVICE_ACCESS_DENIED_ERROR</returnAuthMsg><returnReasonCode>20</returnReasonCode>"
         b"</cmmMsgHeader></OpenAPI_ServiceResponse>", "SERVICE ERROR / SERVICE_ACCESS_DENIED_ERROR / 20"),
        (b"<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>", "SERVICE_KEY_IS_NOT_REGISTERED_ERROR"),
        ('{"response": {"header": {"resultCode": "30", "resultMsg": "등록되지 않은 서비스키"}}}'.encode(), "30 / 등록되지 않은 서비스키"),
        (b"Forbidden", "Forbidden"),
        (b"<html><body><h1>403  Forbidden</h1></body></html>", "403 Forbidden"),
        (b"", None),
        (b"   ", None),
        (None, None),
    ],
)
def test_server_reason(raw, reason):
    assert drug_api.server_reason(raw) == reason


def test_server_reason_is_short():
    assert len(drug_api.server_reason(b"x" * 5000)) == 120


def test_server_reason_never_shows_the_key():
    key = "ab+c/d=="
    raw = f"<errMsg>bad key {key} / ab%2Bc%2Fd%3D%3D / ab%2Bc%2Fd%3D%3D</errMsg>".encode()
    reason = drug_api.server_reason(raw, key)
    assert "ab+c" not in reason and "ab%2B" not in reason and "***" in reason


def http_error(code, body):
    import io

    return urllib.error.HTTPError("https://x?serviceKey=SECRET", code, "err", {}, io.BytesIO(body))


def test_an_http_error_shows_the_server_reason_and_keeps_the_status(monkeypatch):
    def urlopen(request, timeout):
        raise http_error(403, b"<returnAuthMsg>SERVICE_ACCESS_DENIED_ERROR</returnAuthMsg>")

    monkeypatch.setattr(drug_api.urllib.request, "urlopen", urlopen)
    with pytest.raises(DrugApiError) as caught:
        drug_api.http_get_json("https://x", {"serviceKey": "SECRET"})
    assert str(caught.value) == "의약품 API 호출에 실패했습니다 (HTTP 403: SERVICE_ACCESS_DENIED_ERROR)."
    assert caught.value.status == 403 and caught.value.code == "E_EXT_API"


def test_an_http_error_body_with_the_key_is_masked(monkeypatch):
    def urlopen(request, timeout):
        raise http_error(401, b"invalid serviceKey SECRET")

    monkeypatch.setattr(drug_api.urllib.request, "urlopen", urlopen)
    with pytest.raises(DrugApiError) as caught:
        drug_api.http_get_json("https://x", {"serviceKey": "SECRET"})
    assert "SECRET" not in str(caught.value) and "invalid serviceKey ***" in str(caught.value)


def test_an_unreadable_http_error_body_still_gives_the_status(monkeypatch):
    class Broken(urllib.error.HTTPError):
        def read(self, *a):
            raise OSError("closed")

    def urlopen(request, timeout):
        raise Broken("https://x", 502, "err", {}, None)

    monkeypatch.setattr(drug_api.urllib.request, "urlopen", urlopen)
    with pytest.raises(DrugApiError, match=r"^의약품 API 호출에 실패했습니다 \(HTTP 502\)\.$") as caught:
        drug_api.http_get_json("https://x", {"serviceKey": "SECRET"})
    assert caught.value.status == 502


def test_other_api_errors_have_no_status():
    assert DrugApiError("x").status is None


def test_a_non_json_answer_shows_the_server_reason(monkeypatch):
    raw = b"<OpenAPI_ServiceResponse><returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg></OpenAPI_ServiceResponse>"
    monkeypatch.setattr(drug_api.urllib.request, "urlopen", lambda request, timeout: FakeResponse(raw))
    with pytest.raises(DrugApiError, match="JSON으로 읽을 수 없습니다: SERVICE_KEY_IS_NOT_REGISTERED_ERROR"):
        drug_api.http_get_json("https://x", {"serviceKey": "SECRET"})


def test_requests_have_our_own_user_agent(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["agent"] = request.get_header("User-agent")
        return FakeResponse(json.dumps(response()).encode())

    monkeypatch.setattr(drug_api.urllib.request, "urlopen", urlopen)
    drug_api.http_get_json("https://x", {"serviceKey": "SECRET"})
    assert seen["agent"] == drug_api.USER_AGENT and "urllib" not in seen["agent"].lower()


def test_the_same_reason_twice_is_shown_once():
    raw = b"<errMsg>SERVICE ERROR</errMsg><resultMsg>SERVICE ERROR</resultMsg>"
    assert drug_api.server_reason(raw) == "SERVICE ERROR"


def test_the_api_error_class_is_the_one_in_errors():
    assert drug_api.DrugApiError is DrugApiError           # 판단 층(drug_match)도 같은 클래스를 잡는다


def test_the_key_is_read_from_the_configuration_only():
    from pathlib import Path

    text = Path(drug_api.__file__).read_text(encoding="utf-8")
    assert "config.require_drug_api_key" in text and "config.drug_api_key" in text      # 키는 설정(환경변수)에서만 읽는다
