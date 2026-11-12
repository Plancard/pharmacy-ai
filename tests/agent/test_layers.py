"""파일이 부르는 방향을 지키는지 검사한다 (설계: 파일은 같은 층이나 아래 층만 부른다).

- 바닥       schema · config · errors             → 바닥만 부른다
- 가장자리   llm · drug_api · dur_api · checkpoint · cv_adapter   → 바닥과 가장자리
- 판단       structure · validate · … · proposal  → 바닥과 판단. 가장자리를 직접 import하지 않는다
             (LLM·API는 호출 함수를 인자로 받아야 네트워크 없이 시험할 수 있다)
- MCP 포장   mcp_server · mcp_client             → 바닥, 가장자리, 판단, MCP 포장
             (mcp_server는 따로 도는 서버 프로세스의 입구라 의약품 API 연결을 스스로 만들어 판단 함수에 넘긴다)
- 그래프·입구 graph · nodes · state · supervisor · api   → 제한 없음

agent/ 에 새 파일을 만들면 아래 LAYERS에 이름과 층을 적는다. 일정표·설계 문서에 없는 이름이면 먼저 정한다.
"""
import ast
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"

LAYERS = {
    "schema": "바닥", "config": "바닥", "errors": "바닥",
    "llm": "가장자리", "drug_api": "가장자리", "dur_api": "가장자리", "checkpoint": "가장자리", "cv_adapter": "가장자리",
    **{n: "판단" for n in ("structure", "validate", "pipeline", "drug_names", "drug_dictionary", "drug_match", "accuracy",
                          "plan", "dur", "rag", "risk", "consult", "intent", "recheck", "proposal")},
    "mcp_server": "MCP 포장", "mcp_client": "MCP 포장",
    **{n: "그래프·입구" for n in ("graph", "nodes", "state", "supervisor", "api")},
}
ALLOWED = {
    "바닥": {"바닥"},
    "가장자리": {"바닥", "가장자리"},
    "판단": {"바닥", "판단"},
    "MCP 포장": {"바닥", "가장자리", "판단", "MCP 포장"},
    "그래프·입구": {"바닥", "가장자리", "판단", "MCP 포장", "그래프·입구"},
}


def agent_modules() -> list[Path]:
    return sorted(p for p in AGENT_DIR.glob("*.py") if p.name != "__init__.py")


def imported_agent_modules(path: Path) -> set[str]:
    """이 파일이 부르는 agent 안의 모듈 이름 (agent.X, from agent import X)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[1] for a in node.names if a.name.startswith("agent.")}
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "agent":
                found |= {a.name for a in node.names}
            elif node.module.startswith("agent."):
                found.add(node.module.split(".")[1])
    return found


def test_every_agent_file_has_a_layer():
    unknown = [p.name for p in agent_modules() if p.stem not in LAYERS]
    assert not unknown, f"{unknown}: tests/agent/test_layers.py 의 LAYERS에 층을 적어 주세요"


@pytest.mark.parametrize("path", agent_modules(), ids=lambda p: p.stem)
def test_file_calls_only_allowed_layers(path):
    layer = LAYERS.get(path.stem)
    if layer is None:
        pytest.skip("층이 없는 파일은 test_every_agent_file_has_a_layer가 잡는다")
    bad = sorted(m for m in imported_agent_modules(path) if LAYERS.get(m) not in ALLOWED[layer])
    assert not bad, f"{path.name}({layer})이 부를 수 없는 파일을 부릅니다: {bad}"
