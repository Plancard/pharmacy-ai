"""오류 코드 (agent/errors.py, 명세서 v0.3 부록 A.7)."""
import pytest

from agent import errors
from agent.errors import ERROR_CODES, StructureError


def test_the_error_codes_of_the_spec():
    assert ERROR_CODES == (
        "E_OCR_EMPTY", "E_TRAY_NOT_FOUND", "E_CV_INFER", "E_PLAN_RANGE",
        "E_BARCODE_FAIL", "E_LLM", "E_EXT_API", "E_DB",
    )
    assert all(getattr(errors, code) == code for code in ERROR_CODES)     # 이름과 값이 같다


def test_a_structure_error_keeps_its_message_and_code():
    error = StructureError("자동 구조화에 실패했습니다.", code="E_LLM")
    assert (str(error), error.code) == ("자동 구조화에 실패했습니다.", "E_LLM")


def test_the_code_is_optional():
    assert StructureError("규칙 실패").code is None
    assert StructureError().code is None


@pytest.mark.parametrize("bad", ["E_UNKNOWN", "e_llm", ""])
def test_an_unknown_code_is_refused(bad):
    with pytest.raises(ValueError, match="알 수 없는 오류 코드"):
        StructureError("x", code=bad)


def test_the_api_error_always_has_the_external_api_code():
    from agent.drug_api import DrugApiError

    assert DrugApiError.code == "E_EXT_API" and DrugApiError("연결 실패").code == "E_EXT_API"
