"""lookup_drug.py(의약품 API 실제 조회 스크립트)의 테스트. 진짜 API는 부르지 않는다."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.errors import DrugApiError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "lookup_drug.py"
_spec = importlib.util.spec_from_file_location("lookup_drug", SCRIPT)
lookup_drug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lookup_drug)


def response(*items):
    return {"header": {"resultCode": "00"}, "body": {"items": list(items)}}


def item(code, name, company="가상제약"):
    return {"ITEM_SEQ": code, "ITEM_NAME": name, "ENTP_NAME": company, "CANCEL_NAME": "정상"}


def run(argv, result, monkeypatch, key="SECRETKEY"):
    if key is None:
        monkeypatch.delenv("DRUG_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DRUG_API_KEY", key)
    calls, lines = [], []

    def fetch(url, params):
        calls.append((url, params))
        if isinstance(result, BaseException):
            raise result
        return result

    code = lookup_drug.main(argv, fetch=fetch, out=lines.append)
    return code, "\n".join(lines), calls


def test_products_are_listed_by_similarity(monkeypatch):
    result = response(item("199900002", "타이레놀정160밀리그램"), item("202005623", "타이레놀정500밀리그램(아세트아미노펜)"))
    code, text, calls = run(["타이레놀정", "5oOmg"], result, monkeypatch)
    assert code == 0
    assert "약품명: 타이레놀정 5oOmg -> 검색어: 타이레놀정" in text
    assert calls[0][1] == {"serviceKey": "SECRETKEY", "type": "json", "pageNo": 1, "numOfRows": 20, "item_name": "타이레놀정"}
    body = text.split("이름이 비슷한 순서)")[1]
    assert body.index("202005623") < body.index("199900002")          # 500mg가 더 비슷하다
    assert "유사도 1.000" in text and "가상제약" in text


def test_the_key_is_never_printed(monkeypatch):
    _, text, calls = run(["게보린정", "--raw"], response(item("201900001", "게보린정")), monkeypatch)
    assert calls[0][1]["serviceKey"] == "SECRETKEY"
    assert "SECRETKEY" not in text and "DRUG_API_KEY: 설정됨" in text


def test_missing_key_is_reported_without_calling(monkeypatch):
    code, text, calls = run(["게보린정"], response(), monkeypatch, key=None)
    assert code == 1 and "설정 문제" in text and "DRUG_API_KEY: 없음" in text and calls == []


def test_raw_shows_the_first_products_and_counts_the_rest(monkeypatch):
    result = response(*[item(f"20190000{i}", f"게보린정{i}") for i in range(1, 6)])
    code, text, _ = run(["게보린정", "--raw"], result, monkeypatch)
    assert code == 0 and "응답 원본" in text
    assert '"ITEM_SEQ": "201900001"' in text and '"ITEM_SEQ": "201900002"' in text
    raw_part = text.split("찾은 제품")[0]
    assert "201900003" not in raw_part and "... 외 3개" in raw_part


def test_raw_with_only_the_shown_number_of_products_is_not_cut(monkeypatch):
    _, text, _ = run(["게보린정", "--raw"], response(item("201900001", "게보린정"), item("201900002", "게보린정")), monkeypatch)
    assert "외" not in text.split("찾은 제품")[0]


def test_raw_also_works_with_items_item(monkeypatch):
    result = {"header": {"resultCode": "00"}, "body": {"items": {"item": [item(f"20190000{i}", "게보린정") for i in range(1, 4)]}}}
    _, text, _ = run(["게보린정", "--raw"], result, monkeypatch)
    assert "... 외 1개" in text


def test_no_product_suggests_raw(monkeypatch):
    code, text, _ = run(["알모란정"], response(), monkeypatch)
    assert code == 1 and "--raw" in text


def test_call_failure_is_reported(monkeypatch):
    code, text, _ = run(["게보린정"], DrugApiError("의약품 API 호출에 실패했습니다 (HTTP 500)."), monkeypatch)
    assert code == 1 and "조회 실패(E_EXT_API)" in text and "HTTP 500" in text


def test_a_bad_response_is_reported(monkeypatch):
    code, text, _ = run(["게보린정", "--raw"], {"header": {"resultCode": "30"}, "body": {}}, monkeypatch)
    assert code == 1 and "응답을 읽지 못했습니다" in text and "resultCode 30" in text
    assert "응답 원본" in text            # 원본은 먼저 보여 준다 (무엇이 왔는지 알 수 있게)


def test_nothing_to_search(monkeypatch):
    code, text, calls = run(["500mg"], response(), monkeypatch)
    assert code == 1 and "검색할 글자가 없습니다" in text and calls == []


def test_a_name_is_required():
    with pytest.raises(SystemExit):
        lookup_drug.main([], out=lambda _: None)


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
    # `python scripts/agent/lookup_drug.py` 가 저장소 루트가 아닌 곳에서 불려도 agent 패키지를 찾아야 한다 (--help는 API를 부르지 않는다)
    done = _run_script(["--help"], tmp_path)
    assert done.returncode == 0 and "--raw" in done.stdout


def test_the_script_without_a_key_stops_with_a_plain_message(tmp_path):
    # 키를 빈 값으로 두면 .env에 진짜 키가 있어도 덮어쓰지 않는다 (API를 부르지 않는다)
    done = _run_script(["게보린정"], tmp_path, key="")
    assert done.returncode == 1 and "DRUG_API_KEY: 없음" in done.stdout and "설정 문제" in done.stdout
    assert "Traceback" not in done.stderr
