"""설정. 값은 .env(또는 환경변수)에서 읽는다. 키는 코드·문서·커밋에 쓰지 않는다 (REQ-NF-07)."""
import os
from pathlib import Path

from dotenv import load_dotenv

from agent.errors import ConfigError

load_dotenv()   # 프로젝트 맨 위의 .env를 찾아 읽는다

# 명세서: gpt-5-mini (날짜 버전 고정). 바꾸려면 .env에 OPENAI_MODEL=... 을 적는다.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")

# LLM 호출 한 번의 제한 시간(초). 처방전 1장 -> 조제지시서 1분 이내(REQ-NF-01)를 지키려고 짧게 둔다.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

# 데이터 폴더 (프로젝트 맨 위의 data/agent). 어디서 실행해도 같은 곳을 가리킨다.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "agent"

# 약품 사전 파일. 바꾸려면 .env에 DRUG_DICTIONARY_PATH=... 을 적는다.
DRUG_DICTIONARY_PATH = Path(os.getenv("DRUG_DICTIONARY_PATH") or DATA_DIR / "drug_dictionary.json")


# 식약처 의약품 제품 허가정보 API (공공데이터포털 15095677, '의약품 제품 허가 목록').
# 서비스 버전(…Service07/…Inq07)이 바뀌면 .env에 DRUG_API_URL=... 로 적는다.
DRUG_API_URL = os.getenv("DRUG_API_URL") or "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnInq07"
DRUG_API_TIMEOUT_SECONDS = float(os.getenv("DRUG_API_TIMEOUT_SECONDS", "5"))


def require_drug_api_key() -> str:
    """공공데이터포털 일반 인증키(Decoding). 키는 .env의 DRUG_API_KEY에만 둔다."""
    key = os.getenv("DRUG_API_KEY", "").strip()
    if not key:
        raise ConfigError("DRUG_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return key


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return key
