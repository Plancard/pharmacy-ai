"""tests/agent 전체에 적용되는 설정."""
import os

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


@pytest.fixture(autouse=True)
def no_real_mcp_server(monkeypatch):
    """테스트는 .env의 AGENT_MCP_URL(따로 띄운 MCP 서버)에 붙지 않는다. HTTP 서버 시험은 test_mcp_http.py가 스스로 띄운다."""
    monkeypatch.delenv("AGENT_MCP_URL", raising=False)
    monkeypatch.setenv("AGENT_USE_MCP", "0")      # 기본은 MCP 켜짐이지만 시험은 함수로 직접 부른다 (MCP를 보는 시험은 스스로 setenv)


LIVE_FLAG = "AGENT_LIVE_OPENAI"     # 1이면 @pytest.mark.live_openai 시험이 진짜 OpenAI를 부른다 (비용이 든다)


@pytest.fixture(autouse=True)
def no_real_openai(request, monkeypatch):
    """테스트는 .env의 OPENAI_API_KEY로 진짜 OpenAI(구조화 · 상담 · 임베딩)를 부르지 않는다. 키가 필요한 시험은 스스로 setenv 한다.

    예외: live_openai 표시가 붙은 시험(test_live_openai.py)은 AGENT_LIVE_OPENAI=1일 때만 진짜 키를 그대로 쓴다.
    그 밖의 시험은 이 값을 켜도 키가 지워진다 (실수로 다른 시험이 OpenAI를 부르지 않게).
    """
    if request.node.get_closest_marker("live_openai") is not None:
        if os.environ.get(LIVE_FLAG) != "1":
            pytest.skip(f"진짜 OpenAI 시험은 {LIVE_FLAG}=1 일 때만 돈다 (비용이 든다)")
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            pytest.skip("OPENAI_API_KEY가 없다 (.env를 확인)")
        return
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
