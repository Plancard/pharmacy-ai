"""그래프 두 장 (7 ~ 11단계, 일정표 11/9 ~ 11/13, REQ-AG-01 · 04 · 05, REQ-WEB-03) — 상위 그래프와 처방 서브그래프.

상위 그래프: 슈퍼바이저가 메시지 하나(사진 · 글)의 흐름과 순서(route)를 정하고, 흐름을 정해진 순서로 하나씩 실행한다.

    START -> supervisor -> [ask_type] -> [prescription] -> [register_meds] -> [recheck] -> [tray] -> [consult] -> END
    (대괄호 = route에 있을 때만 들른다. 각 노드는 route에서 자기보다 뒤에 있는 첫 흐름으로 가고, 없으면 끝난다)

    ask_type       되묻기 (HITL type_select에서 멈춤). 약사가 고르면 route를 다시 정한다
    prescription   처방 서브그래프 (아래)
    그 밖           자리만 있다 (복용 약 등록 · 재대조 15단계, 트레이 14단계, 상담 16단계)

처방 서브그래프: 처방전 한 장을 승인 대기까지 데려간다.

    START -> structure -+-> match -> risk -> approve -+-> make_plan -+-> END        (승인 → 조제 계획)
                        |      ^                      |              +-> approve    (계획을 못 만듦: 약사에게 돌아감)
                        |      +----------------------+ (edited: 약사가 고침 → 다시 매칭 · 대조 → 다시 승인 대기)
                        +-> END                                                     (구조화 실패)

- 메시지 하나는 슈퍼바이저를 한 번만 지난다. 상위 그래프의 화살표는 FLOW_ORDER의 앞에서 뒤로만 간다 (되돌아가는 길이 없다).
  처방 서브그래프에서 되돌아가는 길은 승인 대기를 거치는 두 갈래뿐이다(고치기 · 계획 실패). 승인 대기에서 멈추므로 끝없이 돌지 않는다.
- 노드와 화살표는 아래 표(PRESCRIPTION_CHART · TOP_CHART)에 적혀 있고, 만드는 함수는 그 표를 LangGraph에 옮기기만 한다.
  그래서 LangGraph 없이도 표만으로 연결을 시험할 수 있다 (tests/agent/test_graph.py).
- 처방 서브그래프는 혼자서도 돌릴 수 있다 (build_prescription_graph · run_prescription). 상위 그래프의 'prescription' 노드는
  같은 서브그래프를 불러 그 결과를 상태에 옮겨 담는다. 그래서 두 길의 결과가 같다.
- risk는 처방약끼리 위험 신호를 대조한다 (지금은 성분 중복). 13단계에서 병용금기와 효능군 중복이 같은 노드에 더해진다.
- approve는 interrupt로 멈춘다(HITL ①). 멈추려면 체크포인터가 있어야 하므로 그래프를 만들 때 체크포인터를 붙인다
  (checkpoint.py: DATABASE_URL이 있으면 PostgreSQL, 없으면 메모리). thread_id는 대화 ID다 (처방 대화에서는 처방 ID와 같다).
  웹은 api.py의 start · send · pending · resume · stage로 이것을 쓴다.
- 같은 대화의 두 번째 이후 메시지(트레이 사진, 질문 등)는 이번 메시지 칸만 넣는다 (state.message_input).
  처음 상태를 다시 넣으면 체크포인터가 앞서 승인된 조제지시서(order)와 조제 계획(plan)을 None으로 덮어쓴다.
- ask_type도 interrupt로 멈춘다 (HITL type_select). 응답은 {"choice": 값}이다.
- 조제 계획(make_plan)은 승인된 조제지시서로만 만든다 (정할 것 10).
- 복용 약 재대조(recheck)와 트레이 검수(tray)는 자리만 있다. 15단계와 12단계에서 채운다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.checkpoint import default_checkpointer
from agent.drug_api import default_api
from agent.dur_api import default_dur_api
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import ApiSearch
from agent.dur import DurLookup
from agent.llm import extract_llm
from agent.nodes import (
    Deps,
    StructureFn,
    approve_node,
    ask_type_node,
    consult_node,
    match_node,
    plan_node,
    recheck_node,
    register_meds_node,
    risk_node,
    structure_node,
    supervisor_node,
    tray_node,
)
from agent.pipeline import structure_prescription
from agent.schema import DispensingOrder
from agent.state import GraphState, initial_state, message_input
from agent.supervisor import FLOW_ORDER

END = "END"              # 표에서 '끝'을 뜻한다. 만드는 함수가 LangGraph의 END로 바꿔 끼운다

NodeFn = Callable[[dict[str, Any], Deps], dict[str, Any]]
Choose = Callable[[dict[str, Any]], str]

# 노드 함수. 이름은 상태의 칸 이름(GraphState의 키)과 같으면 안 된다 (LangGraph가 거절한다).
# 상위 그래프의 'prescription' 노드만 여기 없다: 그것은 함수가 아니라 처방 서브그래프다.
STEPS: dict[str, NodeFn] = {
    "supervisor": supervisor_node,
    "structure": structure_node,
    "match": match_node,
    "risk": risk_node,
    "approve": approve_node,
    "make_plan": plan_node,
    "ask_type": ask_type_node,
    "register_meds": register_meds_node,
    "recheck": recheck_node,
    "tray": tray_node,
    "consult": consult_node,
}


@dataclass(frozen=True)
class Chart:
    """그래프 한 장의 설계표. 어디서 시작해, 어떤 노드가 있고, 노드에서 어디로 가는가."""

    entry: str                                                   # START 다음에 가는 노드. 이 노드로 들어오는 화살표는 없다
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]                           # 고정 화살표: (어디서, 어디로)
    branches: dict[str, tuple[Choose, dict[str, str]]]           # 갈림길: 노드 -> (고르는 함수, {함수가 돌려준 이름: 다음 노드})

    def successors(self) -> dict[str, set[str]]:
        """노드마다 다음에 갈 수 있는 곳 (END 포함). 화살표 표에서 만든다."""
        out: dict[str, set[str]] = {name: set() for name in self.nodes}
        for source, target in self.edges:
            out[source].add(target)
        for source, (_, targets) in self.branches.items():
            out[source] |= set(targets.values())
        return out


def next_flow(after: str) -> Choose:
    """노드 after 다음에 갈 흐름을 고르는 함수: route에서 FLOW_ORDER로 after보다 뒤에 있는 첫 흐름. 없으면 'end'.

    route에 모르는 값이 섞여 있으면 건너뛴다 (그래프가 갈 수 없는 곳으로 가지 않는다).
    """
    start = FLOW_ORDER.index(after) + 1 if after in FLOW_ORDER else 0

    def choose(state: dict[str, Any]) -> str:
        route = state.get("route")
        wanted = set(route) if isinstance(route, list) else set()
        return next((flow for flow in FLOW_ORDER[start:] if flow in wanted), "end")

    choose.__name__ = f"after_{after}"
    return choose


after_supervisor = next_flow("supervisor")


def _forward(after: str) -> tuple[Choose, dict[str, str]]:
    """노드 after의 갈림길: 그보다 뒤의 흐름 전부와 끝."""
    later = FLOW_ORDER[FLOW_ORDER.index(after) + 1:] if after in FLOW_ORDER else FLOW_ORDER
    return next_flow(after), {**{flow: flow for flow in later}, "end": END}


def after_structure(state: dict[str, Any]) -> str:
    """구조화에 성공했을 때만 약품 매칭으로 간다. 그 밖의 모든 경우(실패, 상태 이상)는 끝낸다."""
    if state.get("status") == "structured" and state.get("order") is not None:
        return "match"
    return "end"


def after_approve(state: dict[str, Any]) -> str:
    """승인되면 조제 계획으로, 약사가 고쳤으면 다시 매칭으로 간다. 그 밖은 끝낸다."""
    if state.get("order") is None:
        return "end"
    return {"approved": "make_plan", "edited": "match"}.get(state.get("status"), "end")


def after_plan(state: dict[str, Any]) -> str:
    """계획을 만들었으면 끝, 못 만들었으면 약사에게 돌아간다 (승인 대기로)."""
    return "end" if state.get("status") == "planned" else "approve"


PRESCRIPTION_CHART = Chart(
    entry="structure",
    nodes=("structure", "match", "risk", "approve", "make_plan"),
    edges=(
        ("match", "risk"),
        ("risk", "approve"),
    ),
    branches={
        "structure": (after_structure, {"match": "match", "end": END}),
        "approve": (after_approve, {"make_plan": "make_plan", "match": "match", "end": END}),
        "make_plan": (after_plan, {"end": END, "approve": "approve"}),
    },
)

TOP_CHART = Chart(
    entry="supervisor",
    nodes=("supervisor", *FLOW_ORDER),
    edges=(),
    branches={"supervisor": (after_supervisor, _forward("supervisor")[1]),
              **{flow: _forward(flow) for flow in FLOW_ORDER}},
)

# 상위 그래프에서 서브그래프를 부르는 노드의 이름
SUBGRAPH_NODE = "prescription"


def default_structure(prescription_id: str, ocr: dict[str, Any] | None) -> DispensingOrder:
    """실제 구조화: 규칙 경로 -> 실패하면 OpenAI 경로. 키가 없으면 LLM 경로에서 ConfigError."""
    return structure_prescription(prescription_id, ocr, llm_fn=extract_llm)


_FROM_SETTINGS = object()   # api를 주지 않았을 때: .env의 DRUG_API_KEY가 있으면 API를 쓴다


def _make_deps(
    dictionary: DrugDictionary | None,
    structure: StructureFn | None,
    api: ApiSearch | None | object,
    tools: DrugTools | None = None,
    dur: DurLookup | None | object = _FROM_SETTINGS,
) -> Deps:
    if dur is _FROM_SETTINGS:              # api를 따로 정하지 않았을 때만 설정을 따른다 (api=None이면 DUR도 쓰지 않는다)
        dur = default_dur_api() if api is _FROM_SETTINGS else None
    return Deps(
        dictionary=dictionary if dictionary is not None else load_dictionary(),
        structure=structure if structure is not None else default_structure,
        api=default_api() if api is _FROM_SETTINGS else api,
        tools=tools,
        dur=dur,
    )


def _step(fn: NodeFn, deps: Deps):
    """노드 함수에 도구(deps)를 묶어 LangGraph가 부르는 모양(상태 하나를 받는 함수)으로 만든다."""
    def step(state: GraphState) -> dict[str, Any]:
        return fn(state, deps)
    return step


def _subgraph_step(subgraph):
    """서브그래프를 부르는 노드. 서브그래프가 바꾼 칸만 상위 상태에 돌려준다.

    서브그래프를 노드로 그대로 끼우면 서브그래프가 돌려준 errors · notes 전체가 상위의 합치기 규칙(덧붙이기)에 다시 더해져
    같은 기록이 두 번 쌓인다. 그래서 함수 안에서 invoke하고, 덧붙이는 칸은 이번에 새로 쌓인 것만 돌려준다.
    """
    def step(state: GraphState) -> dict[str, Any]:
        result = subgraph.invoke(state)
        update = {key: value for key, value in result.items()
                  if key not in APPENDED_KEYS and (key not in state or state[key] != value)}
        for key in APPENDED_KEYS:
            new = (result.get(key) or [])[len(state.get(key) or []):]
            if new:
                update[key] = new
        return update
    return step


APPENDED_KEYS = ("errors", "notes")     # 상태에서 합치기 규칙이 '덧붙이기'인 칸 (state.py)


def _compile(chart: Chart, make_node: Callable[[str], Any], checkpointer: Any = None):
    """설계표를 LangGraph에 옮겨 compile한다. make_node(이름)이 그 이름의 노드(상태 하나를 받는 함수)를 돌려준다.

    checkpointer가 None이면 붙이지 않는다. 상위 그래프 안에서 부르는 서브그래프는 상위의 체크포인터를 이어받는다.
    """
    from langgraph.graph import END as LG_END, START, StateGraph   # langgraph는 그래프를 만들 때만 필요하다

    def end_to_lg(target: str) -> str:
        return LG_END if target == END else target

    builder = StateGraph(GraphState)
    for name in chart.nodes:
        builder.add_node(name, make_node(name))
    builder.add_edge(START, chart.entry)
    for source, target in chart.edges:
        builder.add_edge(source, end_to_lg(target))
    for source, (choose, targets) in chart.branches.items():
        builder.add_conditional_edges(source, choose, {name: end_to_lg(t) for name, t in targets.items()})
    return builder.compile(checkpointer=checkpointer)


def _compile_prescription(deps: Deps, checkpointer: Any = None):
    return _compile(PRESCRIPTION_CHART, lambda name: _step(STEPS[name], deps), checkpointer)


def _checkpointer(given: Any) -> Any:
    return default_checkpointer() if given is _FROM_SETTINGS else given


def build_prescription_graph(
    *,
    dictionary: DrugDictionary | None = None,
    structure: StructureFn | None = None,
    api: ApiSearch | None | object = _FROM_SETTINGS,
    checkpointer: Any = _FROM_SETTINGS,
    tools: DrugTools | None = None,
    dur: DurLookup | None | object = _FROM_SETTINGS,
):
    """처방 서브그래프만 만들어 compile한 결과를 돌려준다. 인자의 뜻은 build_graph와 같다."""
    return _compile_prescription(_make_deps(dictionary, structure, api, tools, dur), _checkpointer(checkpointer))


def build_graph(
    *,
    dictionary: DrugDictionary | None = None,
    structure: StructureFn | None = None,
    api: ApiSearch | None | object = _FROM_SETTINGS,
    checkpointer: Any = _FROM_SETTINGS,
    tools: DrugTools | None = None,
    dur: DurLookup | None | object = _FROM_SETTINGS,
):
    """상위 그래프를 만들어 compile한 결과를 돌려준다 (처방 서브그래프가 그 안에 들어 있다).

    dictionary를 주지 않으면 설정의 약품 사전을 읽는다. 사전이 잘못되었으면 여기서 바로 ConfigError
    (처방전을 받기 전에 알 수 있다).
    structure를 주지 않으면 실제 구조화(default_structure)를 쓴다. 테스트는 LLM 없는 가짜를 넘긴다.
    api를 주지 않으면 .env에 DRUG_API_KEY가 있을 때 식약처 API를 쓰고, 없으면 사전만 쓴다.
    api=None을 주면 키가 있어도 사전만 쓴다 (테스트·정확도 측정용).
    checkpointer를 주지 않으면 설정에 맞는 것을 붙인다 (checkpoint.default_checkpointer). 승인 대기에서 멈추려면 꼭 있어야 한다.
    dur(DUR 조회)를 주지 않으면 api와 같이 정한다: api를 주지 않았으면 키가 있을 때 DUR API, api=None이면 쓰지 않는다.
    tools(mcp_client.DrugTools)를 주면 약품 매칭과 위험 신호 대조를 MCP 도구로 부른다. 그때 API는 서버가 쓴다.
    """
    deps = _make_deps(dictionary, structure, api, tools, dur)
    prescription = _compile_prescription(deps)

    def make_node(name: str):
        if name == SUBGRAPH_NODE:
            return _subgraph_step(prescription)
        return _step(STEPS[name], deps)

    return _compile(TOP_CHART, make_node, _checkpointer(checkpointer))


def thread_config(conversation_id: str) -> dict[str, Any]:
    """체크포인터가 대화 하나를 가리키는 설정. thread_id는 대화 ID다 (처방 대화에서는 처방 ID와 같다)."""
    return {"configurable": {"thread_id": conversation_id}}


def has_thread(graph, conversation_id: str) -> bool:
    """이 대화가 체크포인터에 이미 있는가. 체크포인터가 없는 그래프는 늘 새 대화다."""
    if getattr(graph, "checkpointer", None) is None:
        return False
    return bool(graph.get_state(thread_config(conversation_id)).values)


def run_message(
    graph,
    conversation_id: str,
    message: dict[str, Any] | None = None,
    ocr: dict[str, Any] | None = None,
    prescription_id: str | None = None,
) -> dict[str, Any]:
    """메시지 하나({message_id, text, image_id, kind})를 상위 그래프에 넣어 끝까지(또는 멈출 때까지) 돌린 상태.

    대화의 첫 메시지면 initial_state, 이미 있는 대화면 이번 메시지 칸만(message_input) 넣는다.
    멈췄으면 돌려준 값에 '__interrupt__'가 들어 있다. 멈춘 내용은 api.py가 checkpointer에서 다시 읽는다.
    """
    if has_thread(graph, conversation_id):
        given = message_input(message, ocr, prescription_id)
    else:
        given = initial_state(conversation_id, message, prescription_id, ocr)
    return graph.invoke(given, thread_config(conversation_id))


def run_prescription(graph, prescription_id: str, ocr: dict[str, Any] | None = None) -> dict[str, Any]:
    """처방전 한 장을 처방 서브그래프에만 넣어 돌린 상태 (슈퍼바이저를 거치지 않는다). 대화 ID = 처방 ID."""
    state = initial_state(prescription_id, {"kind": "prescription"}, prescription_id, ocr)
    return graph.invoke(state, thread_config(prescription_id))


def resume_run(graph, conversation_id: str, reply: Any) -> dict[str, Any]:
    """멈춘 대화를 약사 응답으로 이어 간다 (LangGraph Command(resume=응답))."""
    from langgraph.types import Command

    return graph.invoke(Command(resume=reply), thread_config(conversation_id))
