"""구조화 단계에서 함께 쓰는 에러."""


class StructureError(Exception):
    """구조화 실패. 메시지는 화면에 그대로 보여줘도 되는 한국어 문장이다 (처방전 내용은 담지 않는다)."""


class ConfigError(RuntimeError):
    """개발 환경 문제(키 없음 등). 재시도해도 소용없으므로 구조화 실패와 구분해 그대로 올려 보낸다."""
