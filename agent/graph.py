"""상위 그래프 (7단계, 일정표 11/9, REQ-AG-01) — 슈퍼바이저의 규칙 분기와 처방 흐름의 뼈대.

    START -> supervisor -+-> structure -+-> match -> approve -> END     (처방전)
                         |              +-> END                         (구조화 실패)
                         +-> recheck -> END                             (약 포장: 복용 약 재대조 자리)
                         +-> tray -> END                                (트레이: 검수 자리)
                         +-> ask_type -> END                            (종류가 애매함: type_select 자리)

- 메시지 하나는 슈퍼바이저를 한 번만 지난다. 슈퍼바이저로 돌아오는 화살표가 없고, 그래프 전체에 되돌아가는 길이 없다.
- 노드와 화살표는 아래 표(NODES · EDGES · BRANCHES)에 적혀 있고, build_graph는 그 표를 LangGraph에 옮기기만 한다.
  그래서 LangGraph 없이도 표만으로 연결을 시험할 수 있다 (tests/agent/test_graph.py).
- approve · ask_type 은 아직 멈추지 않고 상태만 바꾼다. 10단계에서 interrupt(HITL ① · type_select)로 바뀐다.
- 복용 약 재대조(recheck)와 트레이 검수(tray)는 자리만 있다. 15단계와 12단계에서 채운다.
"""
from __future__ import annotations

from typing import Any, Callable

from agent.drug_api import default_api
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import ApiSearch
from agent.llm import extract_llm
from agent.nodes import (
    Deps,
    StructureFn,
    approve_node,
    ask_type_node,
    match_node,
    recheck_node,
    structure_node,
    supervisor_node,
    tray_node,
)
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder
from agent.state import FLOWS, GraphState, initial_state

END = "END"              # 표에서 '끝'을 뜻한다. build_graph가 LangGraph의 END로 바꿔 끼운다
ENTRY = "supervisor"     # START 다음에 가는 노드. 이 노드로 들어오는 화살표는 없다

NodeFn = Callable[[dict[str, Any], Deps], dict[str, Any]]

# 노드 이름은 상태의 칸 이름(GraphState의 키)과 같으면 안 된다 (LangGraph가 거절한다)
NODES: dict[str, NodeFn] = {
    "supervisor": supervisor_node,
    "structure": structure_node,
    "match": match_node,
    "approve": approve_node,
    "ask_type": ask_type_node,
    "recheck": recheck_node,
    "tray": tray_node,
}


def after_supervisor(state: dict[str, Any]) -> str:
    """슈퍼바이저가 정한 흐름의 첫 노드로 간다. 흐름이 비었거나 알 수 없으면 사진 종류를 되묻는다."""
    flow = state.get("flow")
    return flow if isinstance(flow, str) and flow in FLOWS else "ask_type"


def after_structure(state: dict[str, Any]) -> str:
    """구조화에 성공했을 때만 약품 매칭으로 간다. 그 밖의 모든 경우(실패, 상태 이상)는 끝낸다."""
    if state.get("status") == "structured" and state.get("order") is not None:
        return "match"
    return "end"


# 고정 화살표: (어디서, 어디로)
EDGES: tuple[tuple[str, str], ...] = (
    ("match", "approve"),
    ("approve", END),
    ("ask_type", END),
    ("recheck", END),
    ("tray", END),
)

# 갈림길: 노드 -> (고르는 함수, {함수가 돌려준 이름: 다음 노드})
BRANCHES: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]] = {
    "supervisor": (after_supervisor, {
        "prescription": "structure", "recheck": "recheck", "tray": "tray", "ask_type": "ask_type",
    }),
    "structure": (after_structure, {"match": "match", "end": END}),
}


def successors() -> dict[str, set[str]]:
    """노드마다 다음에 갈 수 있는 곳 (END 포함). 화살표 표(EDGES · BRANCHES)에서 만든다."""
    out: dict[str, set[str]] = {name: set() for name in NODES}
    for source, target in EDGES:
        out[source].add(target)
    for source, (_, targets) in BRANCHES.items():
        out[source] |= set(targets.values())
    return out


def default_structure(prescription_id: str, ocr: dict[str, Any] | None) -> DispensingOrder:
    """실제 구조화: 규칙 경로 -> 실패하면 OpenAI 경로. 키가 없으면 LLM 경로에서 ConfigError."""
    return structure_prescription(prescription_id, ocr, llm_fn=extract_llm)


_FROM_SETTINGS = object()   # api를 주지 않았을 때: .env의 DRUG_API_KEY가 있으면 API를 쓴다


def _step(fn: NodeFn, deps: Deps):
    """노드 함수에 도구(deps)를 묶어 LangGraph가 부르는 모양(상태 하나를 받는 함수)으로 만든다."""
    def step(state: GraphState) -> dict[str, Any]:
        return fn(state, deps)
    return step


def build_graph(
    *,
    dictionary: DrugDictionary | None = None,
    structure: StructureFn | None = None,
    api: ApiSearch | None | object = _FROM_SETTINGS,
):
    """그래프를 만들어 compile한 결과를 돌려준다.

    dictionary를 주지 않으면 설정의 약품 사전을 읽는다. 사전이 잘못되었으면 여기서 바로 ConfigError
    (처방전을 받기 전에 알 수 있다).
    structure를 주지 않으면 실제 구조화(default_structure)를 쓴다. 테스트는 LLM 없는 가짜를 넘긴다.
    api를 주지 않으면 .env에 DRUG_API_KEY가 있을 때 식약처 API를 쓰고, 없으면 사전만 쓴다.
    api=None을 주면 키가 있어도 사전만 쓴다 (테스트·정확도 측정용).
    """
    from langgraph.graph import END as LG_END, START, StateGraph   # langgraph는 그래프를 만들 때만 필요하다

    deps = Deps(
        dictionary=dictionary if dictionary is not None else load_dictionary(),
        structure=structure if structure is not None else default_structure,
        api=default_api() if api is _FROM_SETTINGS else api,
    )

    def end_to_lg(target: str) -> str:
        return LG_END if target == END else target

    builder = StateGraph(GraphState)
    for name, fn in NODES.items():
        builder.add_node(name, _step(fn, deps))
    builder.add_edge(START, ENTRY)
    for source, target in EDGES:
        builder.add_edge(source, end_to_lg(target))
    for source, (choose, targets) in BRANCHES.items():
        builder.add_conditional_edges(source, choose, {name: end_to_lg(t) for name, t in targets.items()})
    return builder.compile()


def run_message(
    graph,
    prescription_id: str,
    ocr: dict[str, Any] | None = None,
    photo_kind: str | None = None,
) -> dict[str, Any]:
    """메시지 하나(사진 한 장)를 그래프에 넣어 끝까지 돌린 마지막 상태."""
    return graph.invoke(initial_state(prescription_id, ocr, photo_kind))
