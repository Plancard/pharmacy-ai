"""show_llm_text.py (LLM에 나가는 글 확인) — LLM을 부르지 않는다."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "show_llm_text.py"
_spec = importlib.util.spec_from_file_location("show_llm_text", SCRIPT)
show = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(show)


def run(argv):
    lines = []
    code = show.main(argv, out=lines.append)
    return code, "\n".join(lines)


def test_a_sample_shows_the_text_that_goes_to_the_llm():
    code, text = run(["rx8_01"])
    assert code == 0 and "(1)202106092 타이레놀정/500mg 1 TAB 3회 3일" in text and "가리기: 0곳" in text


def test_injected_personal_data_does_not_go_out():
    code, text = run(["rx8_01", "--inject"])
    body = text.split("---- LLM에 나가는 글 ----")[1]
    assert code == 0 and "1234567" not in body and "010-0000-0000" not in body and "김가상" not in body
    assert "가리기: 0곳" not in text


def test_every_sample_with_injected_data_is_clean():
    code, text = run(["--all", "--inject"])
    assert code == 0 and "남음" not in text


def test_leftover_personal_data_is_reported_and_ends_with_one(monkeypatch):
    monkeypatch.setattr(show, "prepare_llm_input", lambda ocr: (None, ["주민 900101-1234567"], 0))
    code, text = run(["rx8_01"])
    assert code == 1 and "남음 주민등록번호" in text
