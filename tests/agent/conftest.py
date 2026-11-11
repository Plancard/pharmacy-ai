"""tests/agent 전체에 적용되는 설정."""
import pytest


@pytest.fixture(autouse=True)
def no_real_drug_api(monkeypatch):
    """테스트는 진짜 식약처 API를 부르지 않는다.

    .env에 DRUG_API_KEY가 있어도 테스트 동안에는 지운다. 그래야 결과가 네트워크·API 상태에 따라 달라지지 않는다.
    API가 있을 때의 동작은 가짜 조회 함수를 넘겨서 시험한다. 키가 필요한 테스트는 스스로 setenv 한다.
    """
    monkeypatch.delenv("DRUG_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    """테스트는 진짜 PostgreSQL에 체크포인트를 쓰지 않는다.

    .env에 DATABASE_URL(웹이 쓰는 DB)이 있어도 테스트 동안에는 지운다. 그래프는 메모리 체크포인터를 쓴다.
    PostgreSQL에 실제로 저장되는지는 AGENT_TEST_DATABASE_URL을 준 때만 test_checkpoint.py가 따로 본다.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
