"""measure_rx8_photos.py(rx8 사진 20장 정확도 측정 스크립트)의 테스트. 진짜 OCR · OpenAI는 부르지 않는다."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from agent.config import RESOURCES_DIR
from agent.errors import StructureError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent" / "measure_rx8_photos.py"
_spec = importlib.util.spec_from_file_location("measure_rx8_photos", SCRIPT)
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)

A1_DIR = RESOURCES_DIR / "samples" / "rx8_a1"


def a1(rx_id):
    return json.loads((A1_DIR / f"{rx_id}.json").read_text(encoding="utf-8"))


def misread(rx_id, old, new):
    """rx8 A.1에서 글자 일부를 OCR 오독처럼 바꾼다."""
    ocr = copy.deepcopy(a1(rx_id))
    for line in ocr["lines"]:
        line["text"] = line["text"].replace(old, new)
    return ocr


def run(argv, **kw):
    lines = []
    code = measure.main(argv, out=lines.append, **kw)
    return code, "\n".join(lines)


def photos(tmp_path, ids):
    """빈 사진 파일을 만든다 (OCR은 가짜 함수가 대신한다). prescription_001NN.jpg 이름."""
    for rx_id in ids:
        (tmp_path / f"prescription_001{rx_id[-2:]}.jpg").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize("name, rx_id", [
    ("prescription_00101.jpg", "rx8_01"), ("prescription_00120.jpg", "rx8_20"), ("rx8_07.png", "rx8_07"),
    ("IMG_0005.jpg", "rx8_05"), ("prescription_00121.jpg", None), ("photo.jpg", None),
])
def test_the_answer_is_chosen_by_the_last_number_in_the_file_name(name, rx_id):
    assert measure.rx_id_of(Path(name)) == rx_id


def test_the_generated_a1_scores_everything_right(tmp_path):
    out = tmp_path / "result.json"
    code, text = run(["--a1-dir", str(A1_DIR), "--out", str(out)])
    assert code == 0
    assert "rx8 처방전 20장 · 약품 41줄 (A.1 JSON, 규칙 경로만)" in text
    assert "4개 칸 합계 100.0% (164/164)" in text and "달성" in text
    assert "대상 약 매칭 39/39줄 · 8종 중 8종 확정" in text
    assert "출력 형식 검증 20/20 · 구조화 실패 0장 · 경로 규칙 20 / LLM 0" in text
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["passed"] is True and len(saved["rows"]) == 20


def test_a_misread_target_name_still_counts_because_the_code_decides_the_drug(tmp_path):
    ocr = {"rx8_01": misread("rx8_01", "타이레놀정", "타이레늘정")}
    code, text = run([str(photos(tmp_path, ["rx8_01"]))], ocr_fn=lambda path, rx_id: ocr[rx_id])
    assert code == 0
    assert "rx8_01  rule  약품 2/2줄  일치" in text
    assert "약품명 100.0% (2/2)" in text
    assert "(참고) OCR이 읽은 약품명 글자 그대로 일치 50.0% (1/2)" in text


def test_a_misread_name_of_a_drug_outside_the_eight_is_wrong(tmp_path):
    ocr = {"rx8_19": misread("rx8_19", "알모탄정", "알모란정")}
    code, text = run([str(photos(tmp_path, ["rx8_19"]))], ocr_fn=lambda path, rx_id: ocr[rx_id])
    assert code == 0
    assert "2번 약품명 정답 '알모탄정' 결과 '알모란정'" in text
    assert "약품명 50.0% (1/2)" in text and "처방전 완전 일치 0/1" in text


def test_a_misread_number_is_reported(tmp_path):
    ocr = {"rx8_18": misread("rx8_18", "4일", "3일")}
    code, text = run([str(photos(tmp_path, ["rx8_18"]))], ocr_fn=lambda path, rx_id: ocr[rx_id])
    assert "1번 투약 일수 정답 4 결과 3" in text
    assert "조제 계획 정답 E_PLAN_RANGE 결과 ok" in text


def test_a_photo_the_ocr_cannot_read_counts_as_a_failure_and_the_rest_go_on(tmp_path):
    def ocr(path, rx_id):
        if rx_id == "rx8_02":
            raise StructureError("OCR이 글자를 찾지 못했습니다", code="E_OCR_EMPTY")
        return a1(rx_id)

    code, text = run([str(photos(tmp_path, ["rx8_01", "rx8_02"]))], ocr_fn=ocr)
    assert code == 0
    assert "rx8_02  구조화 실패 (E_OCR_EMPTY" in text
    assert "구조화 실패 1장" in text and "약품명 50.0% (2/4)" in text and "미달" in text


def test_without_the_flag_a_rule_failure_is_a_failure_and_no_real_llm_is_used(tmp_path, monkeypatch):
    broken = misread("rx8_03", "200200099 프론크논캡슐", "프론")       # 품목코드·이름을 못 읽은 줄
    calls = []

    def recording_no_llm(text, feedback=None):
        calls.append(text)
        raise StructureError("규칙 경로만 재는 중")

    monkeypatch.setattr(measure, "no_llm", recording_no_llm)
    code, text = run([str(photos(tmp_path, ["rx8_03"]))], ocr_fn=lambda path, rx_id: broken)
    assert code == 0 and "rx8_03  구조화 실패" in text and "경로 규칙 0 / LLM 0" in text
    assert calls          # 서비스와 같이 LLM 자리로 넘어가지만 가짜(no_llm)라 OpenAI는 부르지 않는다


def test_unknown_files_are_skipped_and_an_empty_folder_is_an_error(tmp_path):
    (tmp_path / "notes.jpg").write_bytes(b"")
    code, text = run([str(tmp_path)], ocr_fn=lambda path, rx_id: pytest.fail("부르면 안 된다"))
    assert code == 1 and "건너뜀: notes.jpg" in text and "잴 처방전이 없습니다" in text


def test_photos_and_a1_dir_cannot_be_given_together(tmp_path):
    with pytest.raises(SystemExit):
        run([str(tmp_path), "--a1-dir", str(A1_DIR)])
