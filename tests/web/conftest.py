"""tests/web 전체에 적용되는 설정."""
import pytest


@pytest.fixture(autouse=True)
def fake_agent_backend(monkeypatch):
    """웹 시험은 가짜 Agent(FakeGateway)로 돈다. .env에 AGENT_BACKEND=langgraph가 있어도 시험 동안에는 fake로 바꾼다.

    진짜 Agent로 띄우면 .env의 DATABASE_URL · OPENAI_API_KEY · DRUG_API_KEY로 실제 DB와 API에 붙는다.
    진짜 Agent와의 연결은 test_langgraph_gateway.py가 가짜 LangGraph · 메모리 체크포인터로 따로 본다.
    """
    monkeypatch.setenv("AGENT_BACKEND", "fake")
    monkeypatch.delenv("CV_STUB_SCENARIO", raising=False)
