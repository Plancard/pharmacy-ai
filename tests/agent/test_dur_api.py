"""DUR 품목정보 API 조회 (agent/dur_api.py) — 가짜 응답으로. 진짜 API는 scripts/agent/show_dur.py로 본다."""
import pytest

from agent import config, dur_api
from agent.dur_api import (
    EFFICACY_OPERATION,
    MAX_PAGES,
    PAGE_ROWS,
    TABOO_OPERATION,
    DurApi,
    TabooPartner,
    default_dur_api,
    fetch_rows,
    parse_efficacy,
    parse_taboo,
)
from agent.errors import DrugApiError
from fake_dur import efficacy_row, response, taboo_row


class Pages:
    """쪽마다 정해 둔 응답을 돌려주고, 받은 요청을 남긴다."""

    def __init__(self, *pages):
        self.pages, self.calls = list(pages), []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        return self.pages[len(self.calls) - 1]


def rows(n, start=0):
    return [taboo_row(f"{200000000 + start + i}", f"상대약{start + i}정", "케토롤락트로메타민") for i in range(n)]


# ------------------------------------------------------------ 부르는 모양
def test_fetch_rows_calls_the_operation_with_the_item_code():
    fetch = Pages(response(rows(2)))
    found, complete = fetch_rows(TABOO_OPERATION, "201110120", key="비밀키", fetch=fetch)
    assert len(found) == 2 and complete
    url, params = fetch.calls[0]
    assert url == f"{config.DUR_API_URL}/{TABOO_OPERATION}"
    assert url == "https://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03"
    assert params == {"serviceKey": "비밀키", "type": "json", "pageNo": 1, "numOfRows": PAGE_ROWS, "itemSeq": "201110120"}


def test_fetch_rows_reads_the_next_pages_until_the_total():
    fetch = Pages(response(rows(PAGE_ROWS), total=150), response(rows(50, 100), total=150))
    found, complete = fetch_rows(TABOO_OPERATION, "201110120", key="k", fetch=fetch)
    assert len(found) == 150 and complete and [c[1]["pageNo"] for c in fetch.calls] == [1, 2]


def test_fetch_rows_stops_after_the_last_page_it_may_read():
    fetch = Pages(*[response(rows(PAGE_ROWS, i * PAGE_ROWS), total=10_000) for i in range(MAX_PAGES)])
    found, complete = fetch_rows(TABOO_OPERATION, "201110120", key="k", fetch=fetch)
    assert len(found) == MAX_PAGES * PAGE_ROWS and not complete and len(fetch.calls) == MAX_PAGES


def test_fetch_rows_with_no_rows_is_complete_and_empty():
    assert fetch_rows(EFFICACY_OPERATION, "202106092", key="k", fetch=Pages(response([]))) == ([], True)


def test_an_error_answer_is_a_drug_api_error():
    bad = {"response": {"header": {"resultCode": "30", "resultMsg": "SERVICE KEY IS NOT REGISTERED ERROR."}}}
    with pytest.raises(DrugApiError):
        fetch_rows(TABOO_OPERATION, "201110120", key="k", fetch=Pages(bad))


def test_without_a_key_the_settings_key_is_required(monkeypatch):
    from agent.errors import ConfigError

    with pytest.raises(ConfigError):
        fetch_rows(TABOO_OPERATION, "201110120", fetch=Pages(response([])))


# ------------------------------------------------------------ 응답 읽기
def test_parse_taboo_keeps_one_partner_per_code_and_skips_bad_codes():
    found = parse_taboo([
        taboo_row("199900001", "가상케토롤락정", "케토롤락트로메타민"),
        taboo_row("199900001", "가상케토롤락정", "케토롤락트로메타민"),
        taboo_row("12345", "코드가짧은정", "무엇"),
        taboo_row("", "코드없는정", "무엇"),
        taboo_row("199900002", "  띄어쓰기   많은정 ", "케토롤락", reason=None),
    ], complete=False)
    assert found.partners == (
        TabooPartner("199900001", "가상케토롤락정", "케토롤락트로메타민", "위장관 출혈 등 이상반응 증가"),
        TabooPartner("199900002", "띄어쓰기 많은정", "케토롤락", ""),
    )
    assert found.complete is False


def test_parse_efficacy_names_each_group_once_with_its_series():
    assert parse_efficacy([
        efficacy_row("해열진통소염제", "NSAIDs"),
        efficacy_row("해열진통소염제", "NSAIDs"),
        efficacy_row("소화성궤양용제"),
        efficacy_row("", "계열만"),
        efficacy_row("해열진통소염제(NSAIDs)", "NSAIDs"),
    ]) == ("해열진통소염제(NSAIDs)", "소화성궤양용제")


# ------------------------------------------------------------ DurApi
def test_dur_api_asks_each_drug_once():
    calls = []

    def fetch(url, params):
        calls.append(url.rsplit("/", 1)[1])
        if url.endswith(TABOO_OPERATION):
            return response([taboo_row("199900001", "가상케토롤락정", "케토롤락트로메타민")])
        return response([efficacy_row("해열진통소염제")])

    dur = DurApi(key="k", fetch=fetch)
    assert dur.taboo("201110120").partners[0].item_code == "199900001"
    assert dur.taboo("201110120") is dur.taboo("201110120")
    assert dur.efficacy("201110120") == ("해열진통소염제",) == dur.efficacy("201110120")
    assert calls == [TABOO_OPERATION, EFFICACY_OPERATION]


def test_a_failure_is_not_remembered():
    state = {"fail": True}

    def fetch(url, params):
        if state["fail"]:
            raise DrugApiError("의약품 API에 연결하지 못했습니다 (URLError).")
        return response([])

    dur = DurApi(key="k", fetch=fetch)
    with pytest.raises(DrugApiError):
        dur.taboo("201110120")
    state["fail"] = False
    assert dur.taboo("201110120").partners == ()


def test_default_dur_api_needs_the_key(monkeypatch):
    assert default_dur_api() is None                               # 시험은 DRUG_API_KEY를 지운다 (conftest)
    monkeypatch.setenv("DRUG_API_KEY", "k")
    assert isinstance(default_dur_api(), DurApi)


def test_the_operation_names_are_the_dur_service_ones():
    assert (dur_api.TABOO_OPERATION, dur_api.EFFICACY_OPERATION) == ("getUsjntTabooInfoList03", "getEfcyDplctInfoList03")


def test_a_short_page_is_the_last_even_without_a_total():
    data = response(rows(3))
    del data["response"]["body"]["totalCount"]
    fetch = Pages(data)                                             # 두 번째 쪽을 물으면 IndexError
    assert fetch_rows(TABOO_OPERATION, "201110120", key="k", fetch=fetch) == (rows(3), True)
