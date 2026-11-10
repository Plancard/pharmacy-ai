"""show_api_8.py(사전 8종의 의약품 API 응답을 보여 주고 저장하는 스크립트)의 테스트. 진짜 API는 부르지 않는다."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.errors import DrugApiError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "show_api_8.py"
_spec = importlib.util.spec_from_file_location("show_api_8", SCRIPT)
show_api_8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(show_api_8)

# 사전 8종: 검색어 -> (품목기준코드, API 제품명). 사전의 순서다
TRUE_PRODUCTS = {
    "타이레놀정": ("202106092", "타이레놀정500밀리그람(아세트아미노펜)"),
    "게보린정": ("197900277", "게보린정"),
    "트리스펜연질캡슐": ("201110120", "트리스펜연질캡슐(이부프로펜)"),
    "프론크논캡슐": ("200200099", "프론크논캡슐"),
    "아르펜정": ("200201239", "아르펜정"),
    "아이월드요비신정": ("200713170", "아이월드요비신정"),
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


def all_good():
    return {term: response(p) for term, p in TRUE_PRODUCTS.items()}


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("DRUG_API_KEY", "secret-test-key")

FILE_NAMES = [
    "00_202106092.json", "01_197900277.json", "02_201110120.json", "03_200200099.json",
    "04_200201239.json", "05_200713170.json", "06_200002714.json", "07_200308774.json",
]


def run(tmp_path, argv=(), table=None):
    lines = []
    fetch = FakeFetch(table if table is not None else all_good())
    code = show_api_8.main(["--out", str(tmp_path / "api"), *argv], fetch=fetch, out=lines.append)
    return code, "\n".join(lines), fetch


def test_every_drug_is_shown_and_its_response_saved(tmp_path):
    code, text, fetch = run(tmp_path)
    assert code == 0 and "전체 8종 중 8종 일치" in text
    assert "[0] 타이레놀정500mg (202106092) 검색어 '타이레놀정' → 제품 1개, 일치: API '타이레놀정500밀리그람(아세트아미노펜)' [가상제약]" in text
    assert sorted(p.name for p in (tmp_path / "api").iterdir()) == FILE_NAMES
    saved = json.loads((tmp_path / "api" / "05_200713170.json").read_text(encoding="utf-8"))
    assert saved["dictionary"] == {"item_code": "200713170", "name": "아이월드요비신정 416.7mg"}
    assert saved["search_term"] == "아이월드요비신정"
    assert saved["response"]["body"]["items"][0]["ITEM_SEQ"] == "200713170"
    assert len(fetch.calls) == 8


def test_the_key_is_in_neither_the_screen_nor_the_files(tmp_path):
    _, text, fetch = run(tmp_path)
    assert all(c["serviceKey"] == "secret-test-key" for c in fetch.calls)          # 요청에는 쓰지만
    assert "secret-test-key" not in text
    assert all("secret-test-key" not in p.read_text(encoding="utf-8") for p in (tmp_path / "api").iterdir())
    assert "serviceKey" not in (tmp_path / "api" / "00_202106092.json").read_text(encoding="utf-8")      # 요청 인자는 저장하지 않는다


def test_a_key_echoed_back_by_the_server_is_masked_in_the_file(tmp_path):
    table = all_good()
    table["게보린정"] = {"header": {"resultCode": "00", "resultMsg": "ok secret-test-key"},
                      "body": {"items": [{"ITEM_SEQ": "197900277", "ITEM_NAME": "게보린정", "CANCEL_NAME": "정상"}]}}
    run(tmp_path, table=table)
    text = (tmp_path / "api" / "01_197900277.json").read_text(encoding="utf-8")
    assert "secret-test-key" not in text and "ok ***" in text


def test_no_save_writes_nothing(tmp_path):
    code, text, _ = run(tmp_path, ["--no-save"])
    assert code == 0 and not (tmp_path / "api").exists() and "저장:" not in text


def test_a_different_code_is_labelled_and_still_saved(tmp_path):
    table = all_good()
    table["아르펜정"] = response(("200299999", "아르펜정"))
    code, text, _ = run(tmp_path, table=table)
    assert code == 0 and "전체 8종 중 7종 일치" in text
    assert "[4] 아르펜정 (200201239) 검색어 '아르펜정' → 제품 1개, 불일치: API '아르펜정' [가상제약]" in text
    assert "verify_dictionary.py" in text
    assert (tmp_path / "api" / "04_200201239.json").exists()


def test_a_failure_is_reported_and_the_rest_still_run(tmp_path):
    table = all_good()
    table["게보린정"] = DrugApiError("의약품 API 호출에 실패했습니다 (HTTP 500).")
    code, text, fetch = run(tmp_path, table=table)
    assert code == 1 and len(fetch.calls) == 8
    assert "[1] 게보린정 (197900277): 조회 실패(E_EXT_API): 의약품 API 호출에 실패했습니다 (HTTP 500)." in text
    assert "전체 8종 중 7종 일치, 1종 조회 실패" in text
    assert sorted(p.name for p in (tmp_path / "api").iterdir()) == [n for n in FILE_NAMES if not n.startswith("01_")]


def test_no_product_found_is_shown(tmp_path):
    table = all_good()
    table["트립라인정"] = response()
    code, text, _ = run(tmp_path, table=table)
    assert code == 0 and "제품 0개, 없음: API에서 찾은 제품 없음" in text


def test_missing_key_stops_before_calling(tmp_path, monkeypatch):
    monkeypatch.delenv("DRUG_API_KEY")
    code, text, fetch = run(tmp_path)
    assert code == 1 and fetch.calls == [] and "설정 문제: DRUG_API_KEY가 설정되지 않았습니다" in text
    assert not (tmp_path / "api").exists()


def test_the_default_folder_is_var_api_which_git_ignores():
    root = Path(__file__).resolve().parents[2]
    assert show_api_8.DEFAULT_OUT == root / "var" / "api"
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "var/" in [line.strip() for line in ignored]


def _run_script(args, cwd, key=None):
    """스크립트를 다른 프로세스로 돌린다. 윈도우의 기본 출력 인코딩은 cp949라서, 자식 프로세스가 UTF-8로 쓰도록 맞춘다."""
    env = {k: v for k, v in os.environ.items() if k != "DRUG_API_KEY"}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if key is not None:
        env["DRUG_API_KEY"] = key
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )


def test_the_script_runs_from_any_folder_and_does_not_need_a_key(tmp_path):
    done = _run_script(["--help"], tmp_path)
    assert done.returncode == 0 and "--no-save" in done.stdout


def test_the_script_without_a_key_stops_with_a_plain_message_and_saves_nothing(tmp_path):
    done = _run_script(["--out", str(tmp_path / "api")], tmp_path, key="")
    assert done.returncode == 1 and "DRUG_API_KEY: 없음" in done.stdout and "설정 문제" in done.stdout
    assert "Traceback" not in done.stderr and not (tmp_path / "api").exists()
