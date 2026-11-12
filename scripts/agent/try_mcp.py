"""MCP 약품 도구를 직접 불러 보고, 그래프 결과가 함수 호출 때와 같은지 본다 (12단계) — 손으로 돌려 보는 스크립트.

    python scripts/agent/try_mcp.py               # 서버를 띄워 도구 목록 → 도구 호출 → 샘플 13장을 함수 · MCP로 돌려 비교
    python scripts/agent/try_mcp.py --no-api      # 의약품 API 없이 (약품 사전만). 함수 쪽도 API를 쓰지 않는다
    python scripts/agent/try_mcp.py --verify      # verify_dictionary 도구까지 부른다 (의약품 API를 8번 부른다)
    python scripts/agent/try_mcp.py --in-process  # 하위 프로세스 없이 서버 객체에 바로 붙는다 (stdio 문제를 가를 때)
    python scripts/agent/try_mcp.py --url http://127.0.0.1:8765/mcp   # 따로 띄운 서버(python -m agent.mcp_server --http)에 붙는다

pytest가 아니다. 서버는 python -m agent.mcp_server 로 띄운다 (stdio). 서버 쪽 의약품 API는 .env의 DRUG_API_KEY를 따른다.
비교는 처방전마다 그래프를 두 번 돌린다: 매칭 노드가 함수(drug_match.match_order)를 부를 때와 MCP 도구(match_drug)를 부를 때.
승인 대기 화면(조제지시서 · 안내)과, 그대로 승인한 뒤의 결과(조제 계획 또는 다시 승인 대기)가 둘 다 같아야 한다.
비교에서는 OpenAI를 부르지 않는다 (규칙 경로만). 규칙 경로로 못 읽는 처방전은 두 번 다 같은 실패여야 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import config  # noqa: E402
from agent.checkpoint import memory_checkpointer  # noqa: E402
from agent.drug_api import default_api  # noqa: E402
from agent.dur_api import default_dur_api  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402
from agent.errors import ConfigError, StructureError  # noqa: E402
from agent.graph import build_graph, resume_run, run_message  # noqa: E402
from agent.mcp_client import DrugTools, McpToolError, http, in_process, stdio  # noqa: E402
from agent.nodes import APPROVE, HITL_APPROVAL  # noqa: E402
from agent.pipeline import structure_prescription  # noqa: E402

SAMPLES = config.RESOURCES_DIR / "samples"
SAMPLE_DIRS = ("cv_a1", "rx8_a1")


def load_samples() -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for folder in SAMPLE_DIRS
        for path in sorted((SAMPLES / folder).glob("rx*.json"))   # 처방전 파일만 (폴더에 다른 json이 있어도 읽지 않는다)
    }


def no_llm(text: str, feedback: str | None) -> Any:
    raise StructureError("비교에서는 LLM을 부르지 않습니다 (규칙 경로로 읽지 못한 처방전).")


def rule_structure(prescription_id: str, ocr: dict[str, Any] | None):
    return structure_prescription(prescription_id, ocr, llm_fn=no_llm)


def _pause(state: dict[str, Any]) -> dict[str, Any] | None:
    found = state.get("__interrupt__") or ()
    return dict(found[0].value) if found else None


def _seen(state: dict[str, Any]) -> dict[str, Any]:
    """비교할 것: 상태 · 조제지시서 · 안내 · 조제 계획 · 멈춘 화면."""
    return {key: state.get(key) for key in ("status", "order", "notice", "plan", "error_code")} | {"pause": _pause(state)}


def run_once(graph, rx_id: str, ocr: dict[str, Any]) -> list[dict[str, Any]]:
    """처방전 하나: 승인 대기까지, 그리고 그대로 승인한 뒤."""
    state = run_message(graph, rx_id, None, ocr, prescription_id=rx_id)
    steps = [_seen(state)]
    pause = _pause(state)
    if pause is not None and pause.get("hitl_point") == HITL_APPROVAL:
        steps.append(_seen(resume_run(graph, rx_id, {"action": APPROVE, "order": pause["order"]})))
    return steps


def describe_end(steps: list[dict[str, Any]]) -> str:
    last = steps[-1]
    if last["pause"] is not None:
        if len(steps) > 1:
            return "승인 → 다시 승인 대기 (조제 계획을 만들 수 없음)"
        return f"멈춤: {last['pause'].get('hitl_point')}"
    if last["status"] == "planned":
        return "승인 → 조제 계획"
    return f"{last['status']}" + (f" ({last['error_code']})" if last.get("error_code") else "")


MAX_DIFFERENCES = 6     # 다를 때 보여 줄 곳의 수
SHOWN_CHARS = 80        # 값 하나를 보여 줄 길이


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS] + "…"


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """두 값이 다른 곳을 '경로: 함수 쪽 → MCP 쪽'으로 (안쪽까지 내려가 가장 작은 단위로)."""
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in list(a) + [k for k in b if k not in a]:
            if a.get(key) != b.get(key):
                found += differences(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))
        return found
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [line for i, (x, y) in enumerate(zip(a, b)) if x != y for line in differences(x, y, f"{path}[{i}]")]
    return [f"{path}: {_short(a)} → {_short(b)}"]


def compare(samples: dict[str, dict[str, Any]], tools: DrugTools, *, api: Any = None, dur: Any = None,
            build: Callable[..., Any] = build_graph) -> list[tuple[str, bool, str]]:
    """샘플마다 (이름, 같은지, 한 줄 설명). 함수 쪽 그래프와 MCP 쪽 그래프는 따로 만들고 같은 사전을 쓴다."""
    dictionary = load_dictionary()
    by_function = build(dictionary=dictionary, structure=rule_structure, api=api, dur=dur, checkpointer=memory_checkpointer())
    by_mcp = build(dictionary=dictionary, structure=rule_structure, api=api, dur=dur, checkpointer=memory_checkpointer(),
                   tools=tools)
    results = []
    for name, ocr in samples.items():
        a, b = run_once(by_function, f"RUN-{name}", ocr), run_once(by_mcp, f"RUN-{name}", ocr)   # 체크포인터가 따로라 같은 ID를 써도 된다
        summary = describe_end(a)
        if a == b:
            results.append((name, True, summary))
        elif len(a) != len(b):
            results.append((name, False, "한쪽만 승인 대기에서 멈췄습니다"))
        else:
            step = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            keys = [k for k in a[step] if a[step][k] != b[step][k]]
            where = "".join(f"\n        {line}" for line in differences(a[step], b[step])[:MAX_DIFFERENCES])
            results.append((name, False, f"{'승인 대기' if step == 0 else '승인 뒤'}에서 다름: {', '.join(keys)}{where}"))
    return results


def show_tool_calls(tools: DrugTools, out: Callable[[str], None], *, verify: bool) -> None:
    dictionary = load_dictionary()
    first = dictionary.entries[0]

    found = tools.call("search_drug", name="타이레놀")
    out("search_drug(name='타이레놀')")
    for d in found["dictionary"]:
        out(f"    사전  {d['item_code']}  {d['name']}  유사도 {d['score']}")
    if found["api_used"]:
        for d in found["api"]:
            out(f"    API   {d['item_code']}  {d['name']}  [{d.get('company') or '업체 모름'}]")
    else:
        out("    API   (서버가 의약품 API를 쓰지 않습니다)")

    out(f"check_item_code(item_code='{first.item_code}', name='{first.name}')")
    try:
        checked = tools.call("check_item_code", item_code=first.item_code, name=first.name)
        out(f"    판정: {checked['verdict']} (사전에 {'있음' if checked['in_dictionary'] else '없음'})")
    except McpToolError as e:
        out(f"    건너뜀: {e}")

    pair = [{"item_code": e.item_code, "name": e.name} for e in dictionary.entries[:2]]     # 시연 조합 A
    shown = " + ".join(d["name"] for d in pair)
    for tool in ("check_duplicate", "check_interaction"):
        out(f"{tool}({shown})")
        try:
            found = tools.call(tool, items=pair)
        except McpToolError as e:
            out(f"    건너뜀: {e}")
            continue
        for hit in found["hits"]:
            out(f"    [{hit['type']}] {' · '.join(hit['names'])}: {', '.join(hit['ingredients'])}")
        if not found["hits"]:
            out("    찾은 것 없음")
        for item in found["unchecked"]:
            out(f"    대조 못 함: {item['name']} ({item['reason']})")

    if verify:
        out("verify_dictionary()")
        try:
            report = tools.call("verify_dictionary")
            for r in report["results"]:
                out(f"    [{r['verdict']}] {r['name']} ({r['item_code']})" + (f": {r['error']}" if r.get("error") else ""))
            out(f"    전체 {report['total']}종 중 {report['matched']}종 일치")
        except McpToolError as e:
            out(f"    건너뜀: {e}")


def main(argv: list[str] | None = None, *, out: Callable[[str], None] = print,
         build: Callable[..., Any] = build_graph) -> int:
    parser = argparse.ArgumentParser(description="MCP 약품 도구 시험 · 함수와 MCP의 그래프 결과 비교")
    parser.add_argument("--no-api", action="store_true", help="의약품 API를 쓰지 않는다 (서버 · 함수 둘 다)")
    parser.add_argument("--verify", action="store_true", help="verify_dictionary 도구도 부른다 (API 8번)")
    parser.add_argument("--in-process", action="store_true", help="하위 프로세스 없이 서버 객체에 바로 붙는다")
    parser.add_argument("--url", help="따로 띄운 서버의 주소 (예: http://127.0.0.1:8765/mcp). 서버 쪽 API 사용은 서버를 띄울 때 정한다")
    args = parser.parse_args(argv)
    if args.url and args.in_process:
        parser.error("--url과 --in-process는 함께 쓸 수 없습니다")

    try:
        api = None if args.no_api else default_api()
        dur = None if args.no_api else default_dur_api()
        if args.url:
            tools = http(args.url)
        elif args.in_process:
            tools = in_process(load_dictionary(), api, dur)
        else:
            tools = stdio(no_api=args.no_api)
    except (McpToolError, ConfigError) as e:
        out(f"MCP 서버에 연결하지 못했습니다: {e}")
        if args.url:
            out("서버가 떠 있는지 보세요: 다른 터미널에서 python -m agent.mcp_server --http")
        out("mcp 패키지가 없다면 pip install -r agent/requirements.txt 로 설치해 주세요.")
        return 1
    try:
        if args.url:
            how = f"HTTP: {args.url}"
        elif args.in_process:
            how = "같은 프로세스"
        else:
            how = "stdio: python -m agent.mcp_server" + (" --no-api" if args.no_api else "")
        out(f"서버: {how} | 의약품 API: {'사용' if api is not None else '사용 안 함'}"
            f" | DUR API: {'사용' if dur is not None else '사용 안 함'}")
        out(f"도구 {len(tools.tool_names())}개")
        for name, description in tools.tools():
            out(f"    {name}: {description}")
        out("")
        show_tool_calls(tools, out, verify=args.verify)
        out("")
        try:
            results = compare(load_samples(), tools, api=api, dur=dur, build=build)
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] != "langgraph":
                raise
            out("langgraph가 설치되어 있지 않아 그래프 비교를 건너뜁니다. pip install -r agent/requirements.txt")
            return 1
        out("그래프 결과 비교 (함수 호출 vs MCP 도구)")
        for name, same, detail in results:
            out(f"    [{'같음' if same else '다름'}] {name}: {detail}")
        same_count = sum(same for _, same, _ in results)
        out(f"전체 {len(results)}장 중 {same_count}장이 같습니다.")
        return 0 if same_count == len(results) else 1
    finally:
        tools.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
