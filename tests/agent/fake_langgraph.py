"""시험용 가짜 LangGraph. 진짜 langgraph가 없어도 build_graph가 만드는 연결을 끝까지 돌려 볼 수 있다.

진짜와 같게 흉내 내는 것 (우리 그래프가 쓰는 부분만)
- StateGraph(상태 모양) · add_node · add_edge · add_conditional_edges · compile(checkpointer=) · invoke(상태, config)
- 상태 칸에 Annotated[..., 합치는 함수]가 있으면 노드가 돌려준 값을 그 함수로 합친다 (없으면 덮어쓴다)
- 상태에 없는 칸을 노드가 돌려주면 오류, 노드 이름이 상태 칸 이름과 같으면 오류
- 멈추고 이어 가기 (10단계): langgraph.types의 interrupt · Command, langgraph.checkpoint.memory의 InMemorySaver,
  get_state(config)의 values · interrupts · next. 진짜 LangGraph의 규칙을 따른다.
  - interrupt(값)는 처음 불리면 그래프를 멈춘다. 체크포인터가 있는 그래프가 thread_id마다 멈춘 곳을 저장한다.
  - invoke(Command(resume=응답), config)로 이어 가면 멈춘 노드를 처음부터 다시 돌리고, 그 노드의 interrupt가 응답을 돌려준다.
  - 노드 함수 안에서 부른 서브그래프가 멈췄으면: 상위 그래프는 그 노드를 처음부터 다시 돌리고,
    서브그래프는 받은 입력을 무시하고 멈췄던 노드부터 이어 간다 (진짜 LangGraph의 '함수로 부른 서브그래프' 규칙).
  - 멈춘 동안 get_state의 values는 멈춘 노드에 들어가기 전의 상태다 (멈춘 노드가 쓰려던 값은 남지 않는다).
- 합치는 함수가 있는 칸은 아무도 쓰지 않아도 빈 값(list()처럼 그 타입의 기본값)으로 결과에 나온다 (진짜와 같다).

진짜보다 엄격하게 막는 것 (실수를 일찍 잡으려고)
- 갈래는 노드마다 하나(고정 화살표 하나 또는 갈림길 하나)만 허용한다
- 나가는 화살표가 없는 노드, 없는 노드를 가리키는 화살표, 돌아오는 길(무한 반복)은 오류

진짜 LangGraph와 같은 시험은 tests/agent/test_graph.py 의 'real' 쪽이 맡는다.
"""
from __future__ import annotations

import contextvars
import sys
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

START = "__start__"
END = "__end__"
MAX_STEPS = 50


class FakeGraphError(Exception):
    pass


# ------------------------------------------------------------ 멈추고 이어 가기


@dataclass(frozen=True)
class Interrupt:
    value: Any
    id: str = "fake-interrupt"


@dataclass(frozen=True)
class Command:
    resume: Any = None


@dataclass
class _Frame:
    state: dict           # 멈춘 노드에 들어가기 전의 상태
    node: str             # 멈춘 노드


class _Paused(Exception):
    """interrupt()가 그래프를 멈출 때 던진다. 바깥 그래프로 나가며 그래프마다 멈춘 곳(_Frame)을 앞에 쌓는다."""

    def __init__(self, value: Any):
        super().__init__("paused")
        self.value = value
        self.frames: list[_Frame] = []


_RESUME: contextvars.ContextVar[list | None] = contextvars.ContextVar("fake_resume", default=None)
_FRAMES: contextvars.ContextVar[list | None] = contextvars.ContextVar("fake_frames", default=None)


def interrupt(value: Any) -> Any:
    box = _RESUME.get()
    if box:
        return box.pop()          # 이어 가는 중이면 응답을 한 번만 돌려준다. 그 뒤의 interrupt는 다시 멈춘다
    raise _Paused(value)


@dataclass
class _Record:
    values: dict
    frames: list[_Frame] | None = None
    value: Any = None


class InMemorySaver:
    def __init__(self):
        self.threads: dict[str, _Record] = {}


@dataclass(frozen=True)
class StateSnapshot:
    values: dict
    next: tuple = ()
    interrupts: tuple = field(default_factory=tuple)


def _thread(config: Any) -> str:
    try:
        return config["configurable"]["thread_id"]
    except (TypeError, KeyError):
        raise FakeGraphError("체크포인터가 있는 그래프는 config의 configurable.thread_id가 필요합니다") from None


class StateGraph:
    def __init__(self, state_schema):
        self.hints = typing.get_type_hints(state_schema, include_extras=True)
        self.reducers: dict[str, Callable[[Any, Any], Any]] = {
            key: hint.__metadata__[0] for key, hint in self.hints.items() if hasattr(hint, "__metadata__")
        }
        # 합치는 함수가 있는 칸의 처음 값: Annotated[list[str], ...] -> list()
        self.starts: dict[str, Callable[[], Any]] = {
            key: typing.get_origin(hint.__origin__) or hint.__origin__
            for key, hint in self.hints.items() if hasattr(hint, "__metadata__")
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

    def compile(self, checkpointer: Any = None) -> "CompiledGraph":
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
        return CompiledGraph(self, fixed, checkpointer)


class CompiledGraph:
    def __init__(self, graph: StateGraph, fixed: dict[str, str], checkpointer: Any = None):
        self.graph = graph
        self.fixed = fixed
        self.checkpointer = checkpointer

    def _merge(self, state: dict, update: Any) -> None:
        if not isinstance(update, dict):
            raise FakeGraphError(f"노드는 dict를 돌려줘야 합니다: {type(update).__name__}")
        for key, value in update.items():
            if key not in self.graph.hints:
                raise FakeGraphError(f"상태에 없는 칸을 바꾸려 했습니다: {key}")
            reducer = self.graph.reducers.get(key)
            state[key] = reducer(state[key], value) if reducer and key in state else value   # 비어 있던 칸은 받은 값으로 시작한다 (진짜와 같다)

    def _fresh(self, given: dict) -> dict:
        """새로 시작하는 상태: 합치는 함수가 있는 칸을 빈 값으로 두고, 받은 값을 합친다."""
        state = {key: start() for key, start in self.graph.starts.items()}
        self._merge(state, dict(given))
        return state

    def _run(self, state: dict, node: str) -> dict:
        for _ in range(MAX_STEPS):
            if node == END:
                return state
            try:
                update = self.graph.nodes[node](state)
            except _Paused as paused:
                paused.frames.insert(0, _Frame(dict(state), node))
                raise
            self._merge(state, update)
            if node in self.graph.branches:
                path, path_map = self.graph.branches[node]
                key = path(state)
                if key not in path_map:
                    raise FakeGraphError(f"{node}의 갈림길이 모르는 이름을 돌려줬습니다: {key!r}")
                node = path_map[key]
            else:
                node = self.fixed[node]
        raise FakeGraphError(f"{MAX_STEPS}걸음 안에 끝나지 않았습니다 (돌아오는 길이 있는지 확인하세요)")

    def invoke(self, state: Any, config: Any = None) -> dict:
        pending = _FRAMES.get()
        if pending:                                   # 상위 그래프가 이어 가는 중: 받은 입력 대신 멈췄던 곳부터
            frame = pending.pop(0)
            return self._run(dict(frame.state), frame.node)
        if self.checkpointer is None:
            if isinstance(state, Command):
                raise FakeGraphError("체크포인터가 없는 그래프는 이어 갈 수 없습니다")
            return self._run(self._fresh(state), self.fixed[START])
        return self._invoke_saved(state, config)

    def _invoke_saved(self, state: Any, config: Any) -> dict:
        threads = self.checkpointer.threads
        thread = _thread(config)
        resume_token = frames_token = None
        if isinstance(state, Command):
            record = threads.get(thread)
            if record is None or not record.frames:
                raise FakeGraphError(f"이어 갈 멈춘 곳이 없습니다 (thread_id={thread})")
            first, *inner = record.frames
            current, node = dict(first.state), first.node
            resume_token, frames_token = _RESUME.set([state.resume]), _FRAMES.set(list(inner))
        else:
            if thread in threads:
                current = dict(threads[thread].values)
                self._merge(current, dict(state))
            else:
                current = self._fresh(state)
            node = self.fixed[START]
        try:
            final = self._run(current, node)
        except _Paused as paused:
            values = dict(paused.frames[0].state)
            threads[thread] = _Record(values=values, frames=paused.frames, value=paused.value)
            return {**values, "__interrupt__": [Interrupt(paused.value)]}
        finally:
            if resume_token is not None:
                _RESUME.reset(resume_token)
                _FRAMES.reset(frames_token)
        threads[thread] = _Record(values=dict(final))
        return final

    def get_state(self, config: Any) -> StateSnapshot:
        if self.checkpointer is None:
            raise FakeGraphError("체크포인터가 없는 그래프에는 저장된 상태가 없습니다")
        record = self.checkpointer.threads.get(_thread(config))
        if record is None:
            return StateSnapshot(values={})
        if not record.frames:
            return StateSnapshot(values=dict(record.values))
        return StateSnapshot(values=dict(record.values), next=(record.frames[0].node,), interrupts=(Interrupt(record.value),))


def install(monkeypatch) -> None:
    """import langgraph.graph 가 이 가짜를 가리키게 한다 (시험이 끝나면 원래대로)."""
    package = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")
    graph.START, graph.END, graph.StateGraph = START, END, StateGraph
    package.graph = graph
    kinds = types.ModuleType("langgraph.types")
    kinds.interrupt, kinds.Command, kinds.Interrupt = interrupt, Command, Interrupt
    package.types = kinds
    checkpoint = types.ModuleType("langgraph.checkpoint")
    memory = types.ModuleType("langgraph.checkpoint.memory")
    memory.InMemorySaver = InMemorySaver
    checkpoint.memory = memory
    package.checkpoint = checkpoint
    for name, module in (("langgraph", package), ("langgraph.graph", graph), ("langgraph.types", kinds),
                         ("langgraph.checkpoint", checkpoint), ("langgraph.checkpoint.memory", memory)):
        monkeypatch.setitem(sys.modules, name, module)
