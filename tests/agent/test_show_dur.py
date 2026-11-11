"""DUR 응답 보기 · 병용금기 시험 조합 스크립트 (scripts/agent/show_dur.py) — 가짜 응답으로."""
import importlib.util
import json
from pathlib import Path

from agent import dur_api
from agent.drug_dictionary import load_dictionary
from fake_dur import efficacy_row, response, taboo_row

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("show_dur", ROOT / "scripts" / "agent" / "show_dur.py")
show_dur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(show_dur)
DICT = load_dictionary()
KEY = "비밀키1234"


def fake_fetch(url, params):
    """병용금기: 이부프로펜 약에는 메토트렉세이트(11/11 실제 응답과 같은 성분) · 심바스타틴 짝을, 이름 검색에는 제품 하나를 돌려준다."""
    assert params["serviceKey"] == KEY
    if url.endswith(dur_api.TABOO_OPERATION):
        code = params["itemSeq"]
        partners = {
            "201110120": [taboo_row("199900001", "가상메토트렉세이트정", "메토트렉세이트")],
            "200201239": [taboo_row("199900001", "가상메토트렉세이트정", "메토트렉세이트")],
            "300000001": [taboo_row("300000009", "가상심바스타틴정", "심바스타틴")],
            "300000002": [taboo_row("300000008", "가상이소소르비드정", "이소소르비드질산염")],
        }.get(code, [])
        return response(partners)
    if url.endswith(dur_api.EFFICACY_OPERATION):
        return response([efficacy_row("해열진통소염제")] if params["itemSeq"] in ("202106092", "201110120") else [])
    term = params["item_name"]                                          # 의약품 제품 허가정보 (이름 검색)
    codes = {"스포라녹스캡슐": "300000001", "비아그라정": "300000002", "클래리시드": "300000001"}
    code = next((c for name, c in codes.items() if name.startswith(term) or term.startswith(name[:4])), None)
    return response([{"ITEM_SEQ": code, "ITEM_NAME": term + "가상", "ENTP_NAME": "가상"}] if code else [])


def run(argv, monkeypatch, fetch=fake_fetch):
    monkeypatch.setenv("DRUG_API_KEY", KEY)
    lines = []
    code = show_dur.main(argv, fetch=fetch, out=lines.append)
    return code, "\n".join(lines)


def test_the_dictionary_view_shows_partners_groups_and_fields(monkeypatch, tmp_path):
    code, text = run(["--out", str(tmp_path)], monkeypatch)
    assert code == 0, text
    assert "[2] 트리스펜연질캡슐 (201110120) 병용금기 상대 1개: 가상메토트렉세이트정[메토트렉세이트]…" in text
    assert "[0] 타이레놀정500mg (202106092) 효능군: 해열진통소염제 (줄 1개)" in text
    assert "dur_api.py가 쓰는 필드 중 없는 것: 없음 (모두 있음)" in text
    saved = sorted(p.name for p in tmp_path.iterdir())
    assert len(saved) == 2 * len(DICT.entries) and saved[0] == "dur_00_getEfcyDplctInfoList03.json"
    assert all(KEY not in p.read_text(encoding="utf-8") for p in tmp_path.iterdir())


def test_missing_fields_are_named(monkeypatch):
    def renamed(url, params):
        data = fake_fetch(url, params)
        for row in (data["response"]["body"]["items"] or {"item": []})["item"]:
            row.pop("MIXTURE_ITEM_SEQ", None)
        return data

    _, text = run(["--no-save"], monkeypatch, renamed)
    assert "dur_api.py가 쓰는 필드 중 없는 것: MIXTURE_ITEM_SEQ" in text


def test_the_cases_are_checked_end_to_end(monkeypatch):
    code, text = run(["--cases"], monkeypatch)
    assert "[T1] 트리스펜연질캡슐 (이부프로펜, 대상 8종) + 메토트렉세이트: 검출" in text
    assert "[T4] 비아그라정 (실데나필) + 이소소르비드: 검출" in text
    assert "병용금기 시험 조합 5건 중 5건 검출" in text and code == 0


def test_a_case_that_is_not_in_the_dur_list_fails(monkeypatch):
    cases = json.loads(show_dur.CASES.read_text(encoding="utf-8"))
    assert len(cases["cases"]) == 5

    def no_taboo(url, params):
        if url.endswith(dur_api.TABOO_OPERATION):
            return response([])
        return fake_fetch(url, params)

    code, text = run(["--cases"], monkeypatch, no_taboo)
    assert code == 1 and "병용금기 목록에 '메토트렉세이트'가 없습니다" in text and "5건 중 0건 검출" in text


def test_without_a_key_it_says_so(monkeypatch):
    lines = []
    monkeypatch.delenv("DRUG_API_KEY", raising=False)
    assert show_dur.main([], fetch=fake_fetch, out=lines.append) == 1
    assert lines[-1].startswith("설정 문제:")


def test_the_output_can_be_written_in_cp949(monkeypatch):
    _, text = run(["--no-save", "--cases"], monkeypatch)
    text.encode("cp949")
