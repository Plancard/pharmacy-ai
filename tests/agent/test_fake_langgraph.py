"""시험용 가짜 LangGraph(fake_langgraph.py)가 진짜와 같게 멈추고 이어 가는지.

같은 시험을 'fake'와 'real'(진짜 LangGraph, 설치되어 있을 때만)로 돌린다. 두 쪽이 모두 통과해야
가짜로 돌린 그래프 · api 시험을 믿을 수 있다. 우리 그래프가 기대는 규칙만 본다.
"""
import operator
from typing import Annotated, TypedDict

import pytest

import fake_langgraph


class S(TypedDict, total=False):
    n: int
    log: Annotated[list[str], operator.add]
    answer: str


@pytest.fixture(params=["fake", "real"])
def lg(request, monkeypatch):
    if request.param == "real":
        pytest.importorskip("langgraph")
    else:
        fake_langgraph.install(monkeypatch)
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    return type("LG", (), dict(StateGraph=StateGraph, START=START, END=END, Command=Command, interrupt=interrupt, Saver=InMemorySaver))


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def asking_graph(lg, calls, checkpointer):
    """one -> ask(멈춤) -> two -> END"""
    def one(state):
        calls.append("one")
        return {"n": state["n"] + 1, "log": ["one"]}

    def ask(state):
        calls.append("ask")
        reply = lg.interrupt({"question": "ok?", "n": state["n"]})
        return {"answer": reply, "log": ["ask"]}

    def two(state):
        calls.append("two")
        return {"n": state["n"] * 10, "log": ["two"]}

    b = lg.StateGraph(S)
    b.add_node("one", one)
    b.add_node("ask", ask)
    b.add_node("two", two)
    b.add_edge(lg.START, "one")
    b.add_edge("one", "ask")
    b.add_edge("ask", "two")
    b.add_edge("two", lg.END)
    return b.compile(checkpointer=checkpointer)


def test_an_interrupt_stops_the_run_and_the_state_is_from_before_the_node(lg):
    calls = []
    g = asking_graph(lg, calls, lg.Saver())
    out = g.invoke({"n": 1, "log": []}, config())
    assert out["__interrupt__"][0].value == {"question": "ok?", "n": 2}
    snapshot = g.get_state(config())
    assert snapshot.values == {"n": 2, "log": ["one"]}
    assert snapshot.next == ("ask",)
    assert [i.value for i in snapshot.interrupts] == [{"question": "ok?", "n": 2}]
    assert calls == ["one", "ask"]


def test_resuming_runs_the_node_again_and_interrupt_returns_the_reply(lg):
    calls = []
    g = asking_graph(lg, calls, lg.Saver())
    g.invoke({"n": 1, "log": []}, config())
    out = g.invoke(lg.Command(resume="yes"), config())
    assert out == {"n": 20, "log": ["one", "ask", "two"], "answer": "yes"}
    assert calls == ["one", "ask", "ask", "two"]                      # 멈춘 노드만 다시 돈다 (one은 다시 돌지 않는다)
    snapshot = g.get_state(config())
    assert snapshot.next == () and snapshot.interrupts == () and snapshot.values["answer"] == "yes"


def test_threads_are_kept_apart(lg):
    g = asking_graph(lg, [], lg.Saver())
    g.invoke({"n": 1, "log": []}, config("a"))
    g.invoke({"n": 5, "log": []}, config("b"))
    assert g.get_state(config("a")).interrupts[0].value["n"] == 2
    assert g.get_state(config("b")).interrupts[0].value["n"] == 6
    assert g.get_state(config("nobody")).values == {}


def test_the_same_checkpointer_in_a_new_graph_still_knows_where_it_stopped(lg):
    saver = lg.Saver()
    asking_graph(lg, [], saver).invoke({"n": 1, "log": []}, config())
    again = asking_graph(lg, [], saver)                              # 프로세스를 다시 켠 것처럼 그래프를 새로 만든다
    assert again.get_state(config()).interrupts[0].value["n"] == 2
    assert again.invoke(lg.Command(resume="later"), config())["answer"] == "later"


def subgraph_in_a_node(lg, calls, checkpointer):
    """상위: start -> wrap(안에서 서브그래프를 invoke) -> after.  서브: prep -> ask(멈춤) -> done."""
    def prep(state):
        calls.append("prep")
        return {"n": state["n"] + 1, "log": ["prep"]}

    def ask(state):
        calls.append("ask")
        return {"answer": lg.interrupt({"n": state["n"]}), "log": ["ask"]}

    def done(state):
        calls.append("done")
        return {"n": state["n"] * 10}

    sb = lg.StateGraph(S)
    for name, fn in (("prep", prep), ("ask", ask), ("done", done)):
        sb.add_node(name, fn)
    sb.add_edge(lg.START, "prep")
    sb.add_edge("prep", "ask")
    sb.add_edge("ask", "done")
    sb.add_edge("done", lg.END)
    sub = sb.compile()

    def start(state):
        calls.append("start")
        return {"log": ["start"]}

    def wrap(state):
        calls.append("wrap")
        result = sub.invoke(state)
        return {"n": result["n"], "answer": result["answer"], "log": result["log"][len(state["log"]):]}

    def after(state):
        calls.append("after")
        return {"log": ["after"]}

    b = lg.StateGraph(S)
    for name, fn in (("start", start), ("wrap", wrap), ("after", after)):
        b.add_node(name, fn)
    b.add_edge(lg.START, "start")
    b.add_edge("start", "wrap")
    b.add_edge("wrap", "after")
    b.add_edge("after", lg.END)
    return b.compile(checkpointer=checkpointer)


def test_a_subgraph_called_in_a_node_stops_the_whole_graph(lg):
    calls = []
    g = subgraph_in_a_node(lg, calls, lg.Saver())
    out = g.invoke({"n": 1, "log": []}, config())
    assert out["__interrupt__"][0].value == {"n": 2}
    assert g.get_state(config()).values == {"n": 1, "log": ["start"]}   # 상위 상태는 wrap에 들어가기 전
    assert [i.value for i in g.get_state(config()).interrupts] == [{"n": 2}]
    assert calls == ["start", "wrap", "prep", "ask"]


def test_resuming_a_subgraph_reruns_the_outer_node_but_not_the_inner_nodes_before_the_stop(lg):
    calls = []
    g = subgraph_in_a_node(lg, calls, lg.Saver())
    g.invoke({"n": 1, "log": []}, config())
    calls.clear()
    out = g.invoke(lg.Command(resume="go"), config())
    assert calls == ["wrap", "ask", "done", "after"]                   # prep(멈추기 전 노드)은 다시 돌지 않는다
    assert out == {"n": 20, "log": ["start", "prep", "ask", "after"], "answer": "go"}


def test_a_second_interrupt_after_resuming_stops_again(lg):
    """같은 노드에 다시 들어와 interrupt를 부르면 (고치기 → 다시 승인 대기) 다시 멈춘다."""
    calls = []

    def ask(state):
        calls.append("ask")
        return {"answer": lg.interrupt({"n": state["n"]}), "n": state["n"] + 1}

    def back(state):
        return "again" if state["answer"] == "again" else "end"

    b = lg.StateGraph(S)
    b.add_node("ask", ask)
    b.add_edge(lg.START, "ask")
    b.add_conditional_edges("ask", back, {"again": "ask", "end": lg.END})
    g = b.compile(checkpointer=lg.Saver())
    g.invoke({"n": 1}, config())
    out = g.invoke(lg.Command(resume="again"), config())
    assert out["__interrupt__"][0].value == {"n": 2}
    out = g.invoke(lg.Command(resume="stop"), config())
    assert out == {"n": 3, "log": [], "answer": "stop"} and calls == ["ask", "ask", "ask", "ask"]


def test_a_list_field_with_a_reducer_is_there_even_if_nobody_wrote_it(lg):
    """진짜 LangGraph는 합치는 함수가 있는 칸을 처음부터 빈 목록으로 둔다 (11/11 PC에서 확인)."""
    def one(state):
        return {"n": state["n"] + 1}

    b = lg.StateGraph(S)
    b.add_node("one", one)
    b.add_edge(lg.START, "one")
    b.add_edge("one", lg.END)
    assert b.compile().invoke({"n": 1}) == {"n": 2, "log": []}
    saved = b.compile(checkpointer=lg.Saver())
    saved.invoke({"n": 1}, config())
    assert saved.get_state(config()).values == {"n": 2, "log": []}
