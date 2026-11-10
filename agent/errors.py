"""Agent가 함께 쓰는 에러와 오류 코드 (명세서 v0.3 부록 A.7)."""

# A.7 오류 코드. 웹은 이 코드를 보고 화면 문구와 약사 선택지(재시도·수동 진행)를 정한다 (기능분석서 6장).
E_OCR_EMPTY = "E_OCR_EMPTY"            # 글자 없음: OCR 결과가 비었거나, 약품 표에서 읽을 글자를 찾지 못함
E_TRAY_NOT_FOUND = "E_TRAY_NOT_FOUND"  # (CV) 트레이 미검출
E_CV_INFER = "E_CV_INFER"              # (CV) 검출 모델 오류
E_PLAN_RANGE = "E_PLAN_RANGE"          # 조제 계획이 트레이 범위(1~3일, 4시점)를 넘음
E_BARCODE_FAIL = "E_BARCODE_FAIL"      # (CV) 바코드 판독 실패
E_LLM = "E_LLM"                        # AI 응답 오류: LLM이 끝내 읽지 못함
E_EXT_API = "E_EXT_API"                # 외부 API 실패 (의약품 정보·DUR)
E_DB = "E_DB"                          # (웹) 저장 실패
ERROR_CODES = (E_OCR_EMPTY, E_TRAY_NOT_FOUND, E_CV_INFER, E_PLAN_RANGE, E_BARCODE_FAIL, E_LLM, E_EXT_API, E_DB)


class StructureError(Exception):
    """구조화 실패. 메시지는 화면에 그대로 보여줘도 되는 한국어 문장이다 (처방전 내용은 담지 않는다).

    code는 A.7 오류 코드다. 구조화 함수(agent.pipeline)가 마지막에 내는 오류에는 언제나 붙어 있다.
    중간 단계(규칙 경로, 검증)의 오류에는 없을 수 있다 (None).
    """

    def __init__(self, message: str = "", code: str | None = None):
        super().__init__(message)
        if code is not None and code not in ERROR_CODES:
            raise ValueError(f"알 수 없는 오류 코드: {code}")
        self.code = code


class ConfigError(RuntimeError):
    """개발 환경 문제(키 없음 등). 재시도해도 소용없으므로 구조화 실패와 구분해 그대로 올려 보낸다."""
