"""시험용 가짜 LangGraph. 진짜 langgraph가 없어도 build_graph가 만드는 연결을 끝까지 돌려 볼 수 있다.

진짜와 같게 흉내 내는 것 (상위 그래프가 쓰는 부분만)
- StateGraph(상태 모양) · add_node · add_edge · add_conditional_edges · compile() · invoke(상태)
- 상태 칸에 Annotated[..., 합치는 함수]가 있으면 노드가 돌려준 값을 그 함수로 합친다 (없으면 덮어쓴다)
- 상태에 없는 칸을 노드가 돌려주면 오류, 노드 이름이 상태 칸 이름과 같으면 오류

진짜보다 엄격하게 막는 것 (실수를 일찍 잡으려고)
- 갈래는 노드마다 하나(고정 화살표 하나 또는 갈림길 하나)만 허용한다
- 나가는 화살표가 없는 노드, 없는 노드를 가리키는 화살표, 돌아오는 길(무한 반복)은 오류

진짜 LangGraph와 같은 시험은 tests/agent/test_graph.py 의 'real' 쪽이 맡는다.
"""
from __future__ import annotations

import sys
import types
import typing
from typing import Any, Callable

START = "__start__"
END = "__end__"
MAX_STEPS = 50


class FakeGraphError(Exception):
    pass


class StateGraph:
    def __init__(self, state_schema):
        self.hints = typing.get_type_hints(state_schema, include_extras=True)
        self.reducers: dict[str, Callable[[Any, Any], Any]] = {
            key: hint.__metadata__[0] for key, hint in self.hints.items() if hasattr(hint, "__metadata__")
        }
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.edges: list[tuple[str, str]] = []
        self.branches: dict[str, tuple[Callable[[dict], str], dict[str, str]]] = {}

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None:
        if name in self.hints:
            raise FakeGraphError(f"'{name}' is already being used as a state key")
        if name in self.nodes or name in (START, END):
            raise FakeGraphError(f"노드 이름이 겹치거나 예약어입니다: {name}")
        self.nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        self.edges.append((source, target))

    def add_conditional_edges(self, source: str, path: Callable[[dict], str], path_map: dict[str, str]) -> None:
        if source in self.branches:
            raise FakeGraphError(f"{source}에 갈림길이 두 번 붙었습니다")
        self.branches[source] = (path, dict(path_map))

    def compile(self) -> "CompiledGraph":
        known = set(self.nodes) | {START, END}
        fixed: dict[str, str] = {}
        for source, target in self.edges:
            if source not in known - {END} or target not in known - {START}:
                raise FakeGraphError(f"없는 노드를 가리키는 화살표입니다: {source} -> {target}")
            if source in fixed:
                raise FakeGraphError(f"{source}에서 나가는 고정 화살표가 둘 이상입니다 (가짜는 갈래 하나만 지원)")
            fixed[source] = target
        for source, (_, path_map) in self.branches.items():
            if source not in self.nodes:
                raise FakeGraphError(f"갈림길이 없는 노드에 붙었습니다: {source}")
            if source in fixed:
                raise FakeGraphError(f"{source}에 고정 화살표와 갈림길이 함께 있습니다")
            bad = [t for t in path_map.values() if t not in known - {START}]
            if bad:
                raise FakeGraphError(f"갈림길이 없는 노드를 가리킵니다: {source} -> {bad}")
        if START not in fixed:
            raise FakeGraphError("START에서 나가는 화살표가 없습니다")
        dead_ends = [n for n in self.nodes if n not in fixed and n not in self.branches]
        if dead_ends:
            raise FakeGraphError(f"나가는 화살표가 없는 노드가 있습니다: {dead_ends}")
        return CompiledGraph(self, fixed)


class CompiledGraph:
    def __init__(self, graph: StateGraph, fixed: dict[str, str]):
        self.graph = graph
        self.fixed = fixed

    def _merge(self, state: dict, update: Any) -> None:
        if not isinstance(update, dict):
            raise FakeGraphError(f"노드는 dict를 돌려줘야 합니다: {type(update).__name__}")
        for key, value in update.items():
            if key not in self.graph.hints:
                raise FakeGraphError(f"상태에 없는 칸을 바꾸려 했습니다: {key}")
            reducer = self.graph.reducers.get(key)
            state[key] = reducer(state.get(key), value) if reducer else value

    def invoke(self, state: dict, config: Any = None) -> dict:
        state = dict(state)
        node = self.fixed[START]
        for _ in range(MAX_STEPS):
            if node == END:
                return state
            self._merge(state, self.graph.nodes[node](state))
            if node in self.graph.branches:
                path, path_map = self.graph.branches[node]
                key = path(state)
                if key not in path_map:
                    raise FakeGraphError(f"{node}의 갈림길이 모르는 이름을 돌려줬습니다: {key!r}")
                node = path_map[key]
            else:
                node = self.fixed[node]
        raise FakeGraphError(f"{MAX_STEPS}걸음 안에 끝나지 않았습니다 (돌아오는 길이 있는지 확인하세요)")


def install(monkeypatch) -> None:
    """import langgraph.graph 가 이 가짜를 가리키게 한다 (시험이 끝나면 원래대로)."""
    package = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")
    graph.START, graph.END, graph.StateGraph = START, END, StateGraph
    package.graph = graph
    monkeypatch.setitem(sys.modules, "langgraph", package)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph)
