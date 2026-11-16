"""ReAct + Function Calling 루프 (판단, 11/16 'AI Agent 프로젝트 핵심 내용 요약' 반영).

    run_react(system, user, tools, chat)  ->  ReactResult(final, steps, stopped)

LLM이 '생각 → 도구 호출 → 결과 보기'를 되풀이하다가 최종 답(JSON)을 낸다. 쓰는 곳은 두 군데다 (agent/investigate.py).
- 약품 매칭 실패: search_drug · check_item_code로 후보만 찾는다 (약은 정하지 않는다).
- 위험 신호 조사: 근거 검색 도구로 허가정보를 찾아 설명한다 (단계는 바꾸지 않는다).

지키는 것
- 도구는 허용 목록(tools)에 있는 것만 부른다. 없는 이름 · 틀린 인자는 실행하지 않고 오류를 LLM에게 돌려준다.
- 횟수 제한: LLM 호출은 많아야 max_steps번, 도구 호출은 많아야 max_calls번. 마지막 차례에는 도구를 주지 않아 답을 내게 한다.
- 도구 결과는 output_chars 글자까지만 LLM에 보낸다 (긴 문서가 대화를 채우지 않게).
- 최종 답은 JSON이어야 한다. 그 안의 값이 쓸 만한지는 쓰는 쪽(investigate.py)이 도구 결과와 견주어 거른다.
  LLM이 낸 값을 그대로 믿지 않는다: 약 코드는 도구가 돌려준 것만, 근거 번호는 도구가 준 것만 받는다.
- LLM이 실패하면 예외를 올리지 않고 stopped에 이유를 담는다. 부른 쪽은 규칙 결과만으로 계속한다 (REQ-NF-06).

판단 층이다. LLM은 chat(messages, tool_specs) -> {"content": str | None, "tool_calls": [{"id", "name", "arguments"}]}로 받는다
(실제로는 llm.tool_chat. 메시지는 OpenAI 대화 형식이다). 시험은 정해진 답을 차례로 내는 가짜 chat으로 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agent.errors import ConfigError, StructureError

ToolChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]

MAX_STEPS = 4             # LLM 호출 수 (마지막 1번은 도구 없이 답만)
MAX_CALLS = 6             # 도구 호출 수
OUTPUT_CHARS = 2000       # 도구 결과 한 번에 LLM에 보내는 글자 수
FINAL_NUDGE = "도구는 더 쓸 수 없습니다. 지금까지 찾은 것만으로 정해진 JSON 모양의 최종 답을 내세요."


class ToolFailure(Exception):
    """도구가 할 수 없는 요청 (LLM에게 오류로 돌려준다). 문장은 LLM과 기록에 그대로 보인다."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]                 # JSON 스키마 (type: object)
    run: Callable[[dict[str, Any]], Any]       # 인자 dict -> JSON으로 바꿀 수 있는 결과. 할 수 없으면 ToolFailure

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolStep:
    tool: str
    arguments: dict[str, Any]
    output: Any = None                         # 도구 결과 (실패하면 None)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ReactResult:
    final: dict[str, Any] | None               # 최종 답 (JSON 객체). 끝내지 못했으면 None
    steps: list[ToolStep] = field(default_factory=list)
    stopped: str | None = None                 # 끝내지 못한 이유 (끝냈으면 None)
    llm_calls: int = 0

    def outputs(self, tool: str) -> list[Any]:
        """이 도구가 성공해서 돌려준 결과들 (쓰는 쪽이 LLM의 답을 견줄 때 쓴다)."""
        return [s.output for s in self.steps if s.tool == tool and s.ok]


def _json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    if body.startswith("```"):                 # ```json ... ``` 으로 감싸 오는 경우
        body = body.strip("`")
        body = body[4:] if body.lower().startswith("json") else body
    try:
        got = json.loads(body)
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def _arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    got = _json_object(raw) if isinstance(raw, str) else None
    if got is None and isinstance(raw, str) and not raw.strip():
        return {}
    return got


def _run_tool(tool: Tool | None, name: str, args: dict[str, Any] | None) -> ToolStep:
    if tool is None:
        return ToolStep(name, args or {}, error=f"쓸 수 없는 도구입니다: {name}")
    if args is None:
        return ToolStep(name, {}, error="인자가 JSON 객체가 아닙니다")
    try:
        output = tool.run(args)
        json.dumps(output, ensure_ascii=False)
    except ToolFailure as e:
        return ToolStep(name, args, error=str(e))
    except (KeyError, TypeError, ValueError) as e:           # 빠진 인자 · 틀린 형식
        return ToolStep(name, args, error=f"인자가 맞지 않습니다 ({type(e).__name__})")
    except Exception as e:                                    # 외부 API 실패 등. 서비스는 멈추지 않는다
        return ToolStep(name, args, error=f"도구가 실패했습니다 ({type(e).__name__})")
    return ToolStep(name, args, output=output)


def _tool_message(call_id: str, step: ToolStep, limit: int) -> dict[str, Any]:
    body = {"error": step.error} if step.error else step.output
    text = json.dumps(body, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + " …(잘림)"
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def run_react(system: str, user: str, tools: Sequence[Tool], chat: ToolChatFn, *,
              max_steps: int = MAX_STEPS, max_calls: int = MAX_CALLS, output_chars: int = OUTPUT_CHARS) -> ReactResult:
    """도구를 쓰며 답을 찾는다. 최종 답은 JSON 객체여야 한다 (system 지시문에 모양을 적는다)."""
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result = ReactResult(final=None)
    for step in range(max_steps):
        last = step == max_steps - 1 or len(result.steps) >= max_calls
        if last and step > 0:
            messages.append({"role": "user", "content": FINAL_NUDGE})
        try:
            reply = chat(messages, [] if last else specs)
        except (StructureError, ConfigError) as e:
            result.stopped = f"LLM을 부르지 못했습니다 ({e})"
            return result
        result.llm_calls += 1
        calls = list((reply or {}).get("tool_calls") or [])
        if not calls or last:
            result.final = _json_object((reply or {}).get("content"))
            if result.final is None:
                result.stopped = "최종 답이 JSON 객체가 아닙니다"
            return result
        messages.append({"role": "assistant", "content": reply.get("content") or None, "tool_calls": [
            {"id": c.get("id"), "type": "function",
             "function": {"name": c.get("name"), "arguments": c.get("arguments") if isinstance(c.get("arguments"), str)
                          else json.dumps(c.get("arguments") or {}, ensure_ascii=False)}}
            for c in calls]})
        for c in calls:                                       # 호출마다 답을 돌려줘야 한다 (OpenAI 형식)
            name = str(c.get("name"))
            if len(result.steps) >= max_calls:
                done = ToolStep(name, {}, error="도구 호출 횟수를 넘었습니다")
            else:
                done = _run_tool(by_name.get(name), name, _arguments(c.get("arguments")))
                result.steps.append(done)
            messages.append(_tool_message(str(c.get("id")), done, output_chars))
    result.stopped = f"LLM 호출 {max_steps}번 안에 답을 내지 못했습니다"
    return result


def trace(result: ReactResult) -> str:
    """기록용 한 줄: 도구 호출 순서와 성공 여부 ('search_drug(성공) > check_item_code(실패)'). cp949 화면에도 찍힌다."""
    return " > ".join(f"{s.tool}({'성공' if s.ok else '실패'})" for s in result.steps) or "도구 호출 없음"
