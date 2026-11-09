"""StateGraph 빌드 및 compile (11/6 뼈대).

    START -> structure -+-> match -> approve -> END
                        +-> END   (구조화 실패)

다음 주: approve에서 interrupt로 멈추고(체크포인터 필요), 승인 뒤 make_plan(조제 계획)으로 간다.
"""
from __future__ import annotations

from typing import Any

from agent.drug_api import default_api
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import ApiSearch
from agent.pipeline import structure_prescription
from app.edges import after_structure
from app.nodes import Deps, StructureFn, approve_node, match_node, structure_node
from app.state import GraphState, initial_state


_FROM_SETTINGS = object()   # api를 주지 않았을 때: .env의 DRUG_API_KEY가 있으면 API를 쓴다


def build_graph(
    *,
    dictionary: DrugDictionary | None = None,
    structure: StructureFn = structure_prescription,
    api: ApiSearch | None | object = _FROM_SETTINGS,
):
    """그래프를 만들어 compile한 결과를 돌려준다.

    dictionary를 주지 않으면 설정의 약품 사전을 읽는다. 사전이 잘못되었으면 여기서 바로 ConfigError
    (처방전을 받기 전에 알 수 있다).
    api를 주지 않으면 .env에 DRUG_API_KEY가 있을 때 식약처 API를 쓰고, 없으면 사전만 쓴다.
    api=None을 주면 키가 있어도 사전만 쓴다 (테스트·정확도 측정용).
    """
    from langgraph.graph import END, START, StateGraph   # langgraph는 그래프를 만들 때만 필요하다

    deps = Deps(
        dictionary=dictionary if dictionary is not None else load_dictionary(),
        structure=structure,
        api=default_api() if api is _FROM_SETTINGS else api,
    )

    def structure_step(state: GraphState) -> dict[str, Any]:
        return structure_node(state, deps)

    def match_step(state: GraphState) -> dict[str, Any]:
        return match_node(state, deps)

    def approve_step(state: GraphState) -> dict[str, Any]:
        return approve_node(state, deps)

    builder = StateGraph(GraphState)
    builder.add_node("structure", structure_step)
    builder.add_node("match", match_step)
    builder.add_node("approve", approve_step)

    builder.add_edge(START, "structure")
    builder.add_conditional_edges("structure", after_structure, {"match": "match", "end": END})
    builder.add_edge("match", "approve")
    builder.add_edge("approve", END)
    return builder.compile()


def run_prescription(graph, prescription_id: str, ocr: dict[str, Any]) -> dict[str, Any]:
    """처방전 한 장을 그래프에 넣어 끝까지 돌린 마지막 상태."""
    return graph.invoke(initial_state(prescription_id, ocr))
