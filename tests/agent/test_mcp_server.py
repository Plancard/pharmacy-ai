"""약품 도구 MCP 서버 (agent/mcp_server.py) — 도구 4개가 판단 함수와 같은 답을 내는지.

서버 객체에 프로세스 없이 붙어(mcp_client.in_process) 부른다. 주고받는 것은 stdio 때와 같은 MCP 요청이다.
진짜 하위 프로세스(stdio)로 띄우는 시험은 맨 아래 몇 개만 둔다. mcp 패키지가 없으면 이 파일은 건너뛴다.
"""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("mcp")

from agent import mcp_server  # noqa: E402
from agent.config import RESOURCES_DIR  # noqa: E402
from agent.drug_api import ApiDrug  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.drug_match import CANDIDATE_MIN, CODE_DIFFERENT, CODE_FOUND, CODE_NOT_FOUND, match_order  # noqa: E402
from agent.drug_names import search_term  # noqa: E402
from agent.errors import ConfigError, DrugApiError  # noqa: E402
from agent.mcp_client import McpToolError, in_process, stdio  # noqa: E402
from agent.mcp_server import MAX_RESULTS, NO_API, TOOL_NAMES  # noqa: E402
from agent.pipeline import structure_prescription  # noqa: E402

ROOT = RESOURCES_DIR.parents[1]
SAMPLES = RESOURCES_DIR / "samples"
DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
SAMPLE_IDS = [p.stem for folder in ("cv_a1", "rx8_a1") for p in sorted((SAMPLES / folder).glob("rx*.json"))]


def order_of(sample_id):
    folder = "cv_a1" if sample_id.startswith("rx_") else "rx8_a1"
    ocr = json.loads((SAMPLES / folder / f"{sample_id}.json").read_text(encoding="utf-8"))
    return structure_prescription(f"P-{sample_id}", ocr, llm_fn=lambda *a: pytest.fail("LLM을 부르면 안 됩니다"))


class FakeApi:
    """의약품 API 대신. 검색어별로 정한 제품을 돌려주고, 불린 검색어를 남긴다."""

    def __init__(self, products=None, fail=None):
        self.products, self.fail, self.asked = products or {}, fail or set(), []

    def __call__(self, term):
        self.asked.append(term)
        if term in self.fail:
            raise DrugApiError("의약품 API에 연결하지 못했습니다 (URLError).")
        return self.products.get(term, [])


@pytest.fixture
def tools():
    with in_process(DICT) as client:
        yield client


def with_api(api):
    return in_process(DICT, api)


# ------------------------------------------------------------ 도구 목록
def test_the_server_offers_the_drug_and_dur_tools_with_descriptions(tools):
    assert tools.tool_names() == list(TOOL_NAMES) == ["search_drug", "check_item_code", "match_drug", "verify_dictionary",
                                                      "check_interaction", "check_duplicate", "build_plan", "get_plan"]
    assert all(description for _, description in tools.tools())


# ------------------------------------------------------------ search_drug
def test_search_drug_finds_similar_names_in_the_dictionary(tools):
    found = tools.call("search_drug", name="타이레놀")
    assert found["query"] == "타이레놀" and found["api_used"] is False and found["api"] is None
    assert found["dictionary"][0] == {"item_code": "202106092", "name": "타이레놀정500mg", "score": found["dictionary"][0]["score"]}
    assert all(d["score"] >= CANDIDATE_MIN for d in found["dictionary"]) and len(found["dictionary"]) <= MAX_RESULTS


def test_search_drug_returns_nothing_from_the_dictionary_for_an_unknown_name(tools):
    assert tools.call("search_drug", name="전혀없는약")["dictionary"] == []


def test_search_drug_also_asks_the_api_with_the_search_term():
    api = FakeApi({search_term("펠루비서방정"): [ApiDrug("200000001", "펠루비서방정(펠루비프로펜)", "가상제약")]})
    with with_api(api) as tools:
        found = tools.call("search_drug", name="펠루비서방정")
    assert api.asked == [search_term("펠루비서방정")]
    assert found["api_used"] is True
    assert found["api"] == [{"item_code": "200000001", "name": "펠루비서방정(펠루비프로펜)", "company": "가상제약"}]


def test_search_drug_shows_at_most_five_api_products():
    many = [ApiDrug(f"20000000{i}", f"가나다정{i}", None) for i in range(8)]
    with with_api(FakeApi({"가나다정": many})) as tools:
        assert len(tools.call("search_drug", name="가나다정")["api"]) == MAX_RESULTS


def test_an_api_failure_is_a_tool_error_with_the_reason():
    with with_api(FakeApi(fail={"가나다정"})) as tools:
        with pytest.raises(McpToolError, match="의약품 API에 연결하지 못했습니다"):
            tools.call("search_drug", name="가나다정")
        assert tools.call("search_drug", name="타이레놀")["dictionary"]      # 실패 뒤에도 연결은 그대로 쓴다


# ------------------------------------------------------------ check_item_code
def test_check_item_code_needs_the_api(tools):
    with pytest.raises(McpToolError) as error:
        tools.call("check_item_code", item_code="202106092", name="타이레놀정500mg")
    assert str(error.value) == NO_API


@pytest.mark.parametrize("products, verdict", [
    ([ApiDrug("202106092", "타이레놀정500밀리그람", "한국얀센")], CODE_FOUND),
    ([ApiDrug("199999999", "타이레놀정500밀리그람", "다른회사")], CODE_DIFFERENT),
    ([], CODE_NOT_FOUND),
])
def test_check_item_code_gives_the_verdict_of_the_judgment_function(products, verdict):
    with with_api(FakeApi({search_term("타이레놀정500mg"): products})) as tools:
        checked = tools.call("check_item_code", item_code="202106092", name="타이레놀정500mg")
    assert checked["verdict"] == verdict
    assert checked["in_dictionary"] is True and checked["dictionary_name"] == "타이레놀정500mg"
    assert [p["item_code"] for p in checked["similar"]] == [p.item_code for p in products]


def test_check_item_code_says_when_the_code_is_not_in_the_dictionary():
    with with_api(FakeApi()) as tools:
        checked = tools.call("check_item_code", item_code="199999999", name="가나다정")
    assert checked["in_dictionary"] is False and checked["dictionary_name"] is None


# ------------------------------------------------------------ match_drug
@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_match_drug_gives_the_same_order_as_the_function(tools, sample_id):
    order = order_of(sample_id)
    found = tools.call("match_drug", order=order.model_dump(mode="json"))
    assert found == {"order": match_order(order, DICT).model_dump(mode="json"), "api_used": False}
    assert tools.match_drug(order) == (match_order(order, DICT), False)


def test_match_drug_uses_the_server_api_for_drugs_outside_the_dictionary():
    order = order_of("rx_005")                                  # 대상 8종이 아닌 약뿐이다
    api = FakeApi()
    with with_api(api) as tools:
        matched, api_used = tools.match_drug(order)
    assert api_used is True and api.asked                      # 서버가 API에 물었다
    assert matched == match_order(order, DICT, api=FakeApi())


def test_match_drug_rejects_something_that_is_not_an_order(tools):
    with pytest.raises(McpToolError, match="조제지시서\\(A.5\\) 형식이 아닙니다"):
        tools.call("match_drug", order={"prescription_id": "P1"})


# ------------------------------------------------------------ verify_dictionary
def test_verify_dictionary_needs_the_api(tools):
    with pytest.raises(McpToolError, match="의약품 API를 쓰지 않는 서버"):
        tools.call("verify_dictionary")


def test_verify_dictionary_checks_every_entry():
    products = {search_term(e.name): [ApiDrug(e.item_code, e.name, None)] for e in DICT.entries}
    first = DICT.entries[0]
    products[search_term(first.name)] = [ApiDrug("199999999", first.name, None)]
    failing = search_term(DICT.entries[1].name)
    with with_api(FakeApi(products, fail={failing})) as tools:
        report = tools.call("verify_dictionary")
    assert report["total"] == len(DICT.entries) == 8
    assert report["matched"] == 6
    verdicts = {r["item_code"]: r["verdict"] for r in report["results"]}
    assert verdicts[first.item_code] == CODE_DIFFERENT and verdicts[DICT.entries[1].item_code] == "실패"
    assert "연결하지 못했습니다" in report["results"][1]["error"]


# ------------------------------------------------------------ stdio (진짜 하위 프로세스)
def test_the_server_runs_as_a_process_and_answers_like_the_function():
    order = order_of("rx8_05")
    with stdio(no_api=True) as tools:
        assert tools.tool_names() == list(TOOL_NAMES)
        assert tools.match_drug(order) == (match_order(order, DICT), False)
        with pytest.raises(McpToolError, match="의약품 API를 쓰지 않는 서버"):
            tools.call("verify_dictionary")


def test_the_server_writes_nothing_to_stdout_when_it_cannot_start(monkeypatch, capsys):
    def broken(path=None):
        raise ConfigError("약품 사전 파일이 없습니다: 없음.json")

    monkeypatch.setattr(mcp_server, "load_dictionary", broken)
    assert mcp_server.main(["--no-api"]) == 1
    out, err = capsys.readouterr()
    assert out == "" and "약품 사전을 읽지 못했습니다" in err      # stdout은 MCP 통신선이다


def test_the_server_module_starts_as_python_dash_m():
    """--help만 확인한다 (서버를 띄우면 입력을 기다리며 끝나지 않는다)."""
    done = subprocess.run([sys.executable, "-m", "agent.mcp_server", "--help"], cwd=ROOT, capture_output=True,
                          timeout=60, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert done.returncode == 0 and "--no-api" in done.stdout.decode("utf-8")


# ------------------------------------------------------------ DUR 도구 (13단계)
from agent.dur import COMBINATION_TABOO, EFFICACY_DUP, INGREDIENT_DUP, find_all  # noqa: E402
from agent.mcp_server import NO_DUR  # noqa: E402
from fake_dur import ARPEN, GEVORIN, KETOROLAC, KETOROLAC_NAME, TRISPEN, TYLENOL, FakeDur  # noqa: E402


def refs(*pairs):
    return [{"item_code": code, "name": name} for code, name in pairs]


def test_check_interaction_needs_dur(tools):
    with pytest.raises(McpToolError) as error:
        tools.call("check_interaction", items=refs((TRISPEN, "트리스펜")))
    assert str(error.value) == NO_DUR


def test_check_interaction_finds_the_taboo_with_its_reason():
    with in_process(DICT, dur=FakeDur()) as tools:
        found = tools.call("check_interaction", items=refs((TRISPEN, "트리스펜"), (KETOROLAC, KETOROLAC_NAME)))
    assert found["dur_used"] is True
    assert [(h["type"], h["item_codes"], h["detail"] != "") for h in found["hits"]] == [(COMBINATION_TABOO, [TRISPEN, KETOROLAC], True)]


def test_check_duplicate_without_dur_gives_only_ingredient_duplicates(tools):
    found = tools.call("check_duplicate", items=refs((TYLENOL, "타이레놀"), (GEVORIN, "게보린"), (TRISPEN, "트리스펜")))
    assert [h["type"] for h in found["hits"]] == [INGREDIENT_DUP] and found["dur_used"] is False


def test_check_duplicate_with_dur_adds_efficacy_duplicates():
    with in_process(DICT, dur=FakeDur()) as tools:
        found = tools.call("check_duplicate", items=refs((TYLENOL, "타이레놀"), (TRISPEN, "트리스펜"), (ARPEN, "아르펜")))
    assert [h["type"] for h in found["hits"]] == [INGREDIENT_DUP, EFFICACY_DUP]


def test_the_dur_tools_reject_items_of_the_wrong_shape(tools):
    with pytest.raises(McpToolError, match="items는"):
        tools.call("check_duplicate", items=[{"item_code": TYLENOL}])


@pytest.mark.parametrize("with_dur", [False, True])
@pytest.mark.parametrize("sample_id", ["rx_004", "rx_005", "rx8_01", "rx8_02", "rx8_07"])
def test_check_dur_through_mcp_is_the_same_as_the_function(sample_id, with_dur):
    order = match_order(order_of(sample_id), DICT)
    dur = FakeDur() if with_dur else None
    with in_process(DICT, dur=dur) as tools:
        assert tools.check_dur(order.items) == find_all(order.items, DICT, dur)


def test_check_dur_with_a_taboo_through_mcp():
    from agent.schema import DrugLine

    items = [DrugLine(item_code=c, name=n, printed_code=None, count_per_dose=1, times=["morning"], days=3)
             for c, n in ((TRISPEN, "트리스펜"), (KETOROLAC, KETOROLAC_NAME), (TYLENOL, "타이레놀"))]
    with in_process(DICT, dur=FakeDur()) as tools:
        report = tools.check_dur(items)
    assert report == find_all(items, DICT, FakeDur())
    assert [h.type for h in report.hits] == [COMBINATION_TABOO, EFFICACY_DUP]


def test_the_server_process_uses_dur_when_the_key_is_set(monkeypatch):
    from agent.dur_api import DurApi

    built = []

    class NotRun:
        def run(self, transport):
            assert transport == "stdio"

    monkeypatch.setattr(mcp_server, "build_server", lambda *args: built.append(args) or NotRun())
    monkeypatch.setenv("DRUG_API_KEY", "k")
    assert mcp_server.main([]) == 0
    assert callable(built[0][1]) and isinstance(built[0][2], DurApi)
    assert mcp_server.main(["--no-api"]) == 0
    assert built[1][1:] == ()                                        # API도 DUR도 없이
