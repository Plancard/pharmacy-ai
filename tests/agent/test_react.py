"""ReAct + Function Calling 루프 (agent/react.py)와 도구 호출 대화(llm.tool_chat). LLM은 정해진 답을 차례로 내는 가짜."""
import json
from types import SimpleNamespace

import pytest

from agent.errors import ConfigError, StructureError
from agent.react import FINAL_NUDGE, MAX_STEPS, Tool, ToolFailure, run_react, trace


def call(name, arguments, id_="c1"):
    return {"id": id_, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}


class Scripted:
    """정해진 답을 차례로 낸다. 받은 메시지와 도구 목록을 적어 둔다."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, specs):
        self.seen.append((json.loads(json.dumps(messages)), [s["function"]["name"] for s in specs]))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def answer(obj):
    return {"content": json.dumps(obj, ensure_ascii=False), "tool_calls": []}


def uses(*calls_):
    return {"content": None, "tool_calls": list(calls_)}


ECHO = Tool("echo", "받은 글을 돌려준다", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda args: {"echo": args["text"]})


def failing(error):
    def run(args):
        raise error
    return Tool("broken", "늘 실패한다", {"type": "object", "properties": {}}, run)


def test_a_tool_call_then_an_answer():
    chat = Scripted(uses(call("echo", {"text": "안녕"})), answer({"ok": True}))
    result = run_react("지시", "질문", [ECHO], chat)
    assert (result.final, result.stopped, result.llm_calls) == ({"ok": True}, None, 2)
    assert [(s.tool, s.output) for s in result.steps] == [("echo", {"echo": "안녕"})]
    messages, specs = chat.seen[1]
    assert specs == ["echo"]
    assert messages[2]["role"] == "assistant" and messages[2]["tool_calls"][0]["function"]["name"] == "echo"
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"echo": "안녕"}'}
    assert result.outputs("echo") == [{"echo": "안녕"}] and trace(result) == "echo(성공)"


def test_an_answer_without_tools_is_fine():
    result = run_react("지시", "질문", [ECHO], Scripted(answer({"items": []})))
    assert result.final == {"items": []} and result.steps == [] and trace(result) == "도구 호출 없음"


@pytest.mark.parametrize("name, args, error", [
    ("delete_everything", {}, "쓸 수 없는 도구입니다: delete_everything"),       # 허용 목록 밖
    ("echo", "not json", "인자가 JSON 객체가 아닙니다"),
    ("echo", {}, "인자가 맞지 않습니다 (KeyError)"),
])
def test_a_tool_outside_the_list_or_bad_arguments_is_not_run(name, args, error):
    raw = {"id": "c1", "name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}
    chat = Scripted(uses(raw), answer({"ok": True}))
    result = run_react("지시", "질문", [ECHO], chat)
    assert result.steps[0].error == error and not result.steps[0].ok
    assert json.loads(chat.seen[1][0][-1]["content"]) == {"error": error}       # LLM에게 오류로 돌려준다
    assert result.final == {"ok": True}


@pytest.mark.parametrize("exc, error", [
    (ToolFailure("item_code는 9자리 숫자입니다"), "item_code는 9자리 숫자입니다"),
    (RuntimeError("비밀 내용이 섞인 문장"), "도구가 실패했습니다 (RuntimeError)"),     # 모르는 예외는 종류만
])
def test_a_failing_tool_does_not_stop_the_loop(exc, error):
    result = run_react("지시", "질문", [failing(exc)], Scripted(uses(call("broken", {})), answer({"ok": 1})))
    assert result.steps[0].error == error and result.final == {"ok": 1} and trace(result) == "broken(실패)"


def test_the_last_turn_gets_no_tools_and_a_nudge():
    chat = Scripted(*[uses(call("echo", {"text": str(i)}, f"c{i}")) for i in range(MAX_STEPS - 1)], answer({"done": 1}))
    result = run_react("지시", "질문", [ECHO], chat)
    assert result.final == {"done": 1} and len(result.steps) == MAX_STEPS - 1
    last_messages, last_specs = chat.seen[-1]
    assert last_specs == [] and last_messages[-1] == {"role": "user", "content": FINAL_NUDGE}


def test_tool_calls_beyond_the_limit_are_refused_but_answered():
    many = uses(*[call("echo", {"text": str(i)}, f"c{i}") for i in range(4)])
    chat = Scripted(many, answer({"done": 1}))
    result = run_react("지시", "질문", [ECHO], chat, max_calls=2)
    assert len(result.steps) == 2                                              # 2번만 실행
    tool_messages = [m for m in chat.seen[1][0] if m["role"] == "tool"]
    assert len(tool_messages) == 4 and "도구 호출 횟수를 넘었습니다" in tool_messages[3]["content"]   # 호출마다 답은 돌려준다
    assert chat.seen[1][1] == []                                              # 횟수를 다 썼으니 다음은 답만


def test_long_tool_output_is_cut():
    big = Tool("big", "긴 결과", {"type": "object", "properties": {}}, lambda args: {"text": "가" * 5000})
    chat = Scripted(uses(call("big", {})), answer({}))
    run_react("지시", "질문", [big], chat, output_chars=100)
    content = chat.seen[1][0][-1]["content"]
    assert len(content) < 120 and content.endswith("…(잘림)")


@pytest.mark.parametrize("content", ["그냥 글", "[1, 2]", "", None])
def test_an_answer_that_is_not_a_json_object_is_not_used(content):
    result = run_react("지시", "질문", [ECHO], Scripted({"content": content, "tool_calls": []}))
    assert result.final is None and result.stopped == "최종 답이 JSON 객체가 아닙니다"


def test_a_json_answer_in_a_code_block_is_read():
    result = run_react("지시", "질문", [], Scripted({"content": '```json\n{"a": 1}\n```', "tool_calls": []}))
    assert result.final == {"a": 1}


@pytest.mark.parametrize("error", [StructureError("시간 초과"), ConfigError("OPENAI_API_KEY가 없습니다")])
def test_a_failing_llm_stops_without_raising(error):
    result = run_react("지시", "질문", [ECHO], Scripted(uses(call("echo", {"text": "a"})), error))
    assert result.final is None and result.stopped.startswith("LLM을 부르지 못했습니다") and len(result.steps) == 1


def test_a_model_that_never_answers_stops_after_the_steps():
    chat = Scripted(*[uses(call("echo", {"text": "a"}, f"c{i}")) for i in range(MAX_STEPS)])
    chat.replies[-1] = uses(call("echo", {"text": "a"}, "late"))               # 마지막에도 도구를 부르려 한다 (도구가 없는데도)
    result = run_react("지시", "질문", [ECHO], chat)
    assert result.final is None and result.llm_calls == MAX_STEPS


# ------------------------------------------------------------ llm.tool_chat


class FakeClient:
    def __init__(self, message=None, error=None):
        self.calls, self._message, self._error = [], message, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


def test_tool_chat_sends_the_tools_and_reads_the_calls(monkeypatch):
    from agent import config
    from agent.llm import tool_chat

    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", "low")
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name="echo", arguments='{"text": "a"}'))
    client = FakeClient(SimpleNamespace(content=None, tool_calls=[tc]))
    got = tool_chat([{"role": "user", "content": "q"}], [ECHO.spec()], client=client, model="m")
    assert got == {"content": None, "tool_calls": [{"id": "call_1", "name": "echo", "arguments": '{"text": "a"}'}]}
    sent = client.calls[0]
    assert sent["tools"] == [ECHO.spec()] and sent["response_format"] == {"type": "json_object"}
    assert sent["reasoning_effort"] == "low" and sent["model"] == "m"


def test_tool_chat_without_tools_sends_none():
    from agent.llm import tool_chat

    client = FakeClient(SimpleNamespace(content='{"a": 1}', tool_calls=None))
    assert tool_chat([], [], client=client) == {"content": '{"a": 1}', "tool_calls": []}
    assert "tools" not in client.calls[0]


def test_tool_chat_failure_is_an_llm_error_with_only_the_kind():
    from agent.errors import E_LLM
    from agent.llm import tool_chat

    with pytest.raises(StructureError) as e:
        tool_chat([], [], client=FakeClient(error=RuntimeError("처방 내용")))
    assert e.value.code == E_LLM and "처방 내용" not in str(e.value)
