"""OpenAI 실제 호출 확인 (3단계) — 손으로 한 번 돌려 보는 스크립트.

    python scripts/agent/try_llm.py                # rx_001 한 장
    python scripts/agent/try_llm.py s05_no_code    # 약품 코드가 인쇄되지 않은 처방전 (LLM이 읽어야 하는 경우)
    python scripts/agent/try_llm.py --all          # 샘플 6장 전부

저장소 루트(agent/ 가 보이는 곳)에서 실행한다. pytest가 아니다. 실제로 OpenAI를 부르므로 `python -m pytest`에는 들어 있지 않다.
agent/resources/samples 의 가상 처방전에서 교부번호와 약품 표 본문만 보낸다 (환자·의료인 정보는 보내지 않는다).
키는 화면에 찍지 않는다. 키는 저장소 루트의 .env(OPENAI_API_KEY=...)에만 둔다.

확인하는 것
  1. 모델 이름이 이 계정에서 통하는지 (404/403이면 안내가 나온다)
  2. 응답이 우리 모양(LlmPrescription)으로 오는지
  3. 걸리는 시간이 제한 시간(LLM_TIMEOUT_SECONDS) 안인지
  4. 공통 검증(validate)을 통과하는지, 그리고 정답과 같은지
     (정답: rx_001~005는 cv_expected, s05_no_code는 a5_order. 약품 이름은 띄어쓰기·'/' 차이를 무시하고 견준다.
      약품 매칭은 4~5단계 일이라 아직 보지 않는다.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # `python scripts/agent/try_llm.py` 로 돌려도 agent 패키지를 찾게 한다

from agent import config  # noqa: E402
from agent.errors import ConfigError, StructureError  # noqa: E402
from agent.llm import extract_llm  # noqa: E402
from agent.structure import llm_input, ocr_to_llm_text  # noqa: E402
from agent.validate import order_from_extracted  # noqa: E402

SAMPLES = config.RESOURCES_DIR / "samples"
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*]+")   # 키 문자 + 가려진 부분(*). 문장 끝 마침표는 남긴다
MAX_DETAIL = 600

HINTS_BY_STATUS = {
    400: "요청 형식이 거절되었습니다. 모델이 지원하지 않는 옵션이 있을 수 있습니다. 위 메시지를 그대로 알려 주세요.",
    401: "키가 거절되었습니다. .env의 OPENAI_API_KEY를 확인하세요 (앞뒤 공백·따옴표 없이 키만).",
    403: "이 키로는 이 모델을 쓸 권한이 없습니다. OpenAI 프로젝트의 모델 허용 설정을 확인하세요.",
    404: "모델 이름을 찾지 못했습니다. .env에 OPENAI_MODEL=... 로 이 계정에서 쓸 수 있는 모델을 적어 보세요.",
    429: "사용 한도나 요청 속도 제한에 걸렸습니다. OpenAI의 결제·한도 설정을 확인하세요.",
}


def redact(text: str) -> str:
    """오류 문장에 키 조각이 섞여 있어도 화면에는 가려서 보인다."""
    return KEY_PATTERN.sub("sk-***", text)[:MAX_DETAIL]


def hint_for(cause: BaseException) -> str | None:
    status = getattr(cause, "status_code", None)
    if status in HINTS_BY_STATUS:
        return HINTS_BY_STATUS[status]
    name = type(cause).__name__
    if "Timeout" in name:
        return (
            f"{config.LLM_TIMEOUT_SECONDS:g}초 안에 응답이 오지 않았습니다. "
            "걸린 시간을 알려 주세요. .env에 LLM_TIMEOUT_SECONDS=40 처럼 적어 늘려 볼 수 있습니다."
        )
    if "Connection" in name:
        return "OpenAI에 연결하지 못했습니다. 인터넷·프록시·방화벽을 확인하세요."
    return None


# ------------------------------------------------------------ 샘플과 정답


def _read(kind: str, sample_id: str) -> dict[str, Any]:
    return json.loads((SAMPLES / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def sample_ids() -> list[str]:
    ids = sorted(p.stem for p in (SAMPLES / "cv_expected").glob("rx_*.json"))
    return [*ids, "s05_no_code"]


def normalize_name(name: Any) -> str:
    """띄어쓰기와 '/'를 없애고 소문자로. OCR 이름과 사전 이름을 견주기 위한 것이다."""
    return re.sub(r"[\s/]+", "", str(name)).lower()


def load_sample(sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(보낼 OCR, 정답). 정답은 {"issue_no", "items": [{"name", "printed_code", "count_per_dose", "times_per_day", "days"}]}."""
    if sample_id == "s05_no_code":
        ocr = _read("a1_ocr", sample_id)
        answer = _read("a5_order", sample_id)
        expected = {
            "issue_no": llm_input(ocr)[0],   # 이 처방전은 OCR이 정확히 읽었다. 보내는 글에 있는 교부번호가 정답이다
            "items": [
                {
                    "name": d["name"],
                    "printed_code": None,   # 약품 코드가 인쇄되어 있지 않다
                    "count_per_dose": d["count_per_dose"],
                    "times_per_day": len(d["times"]),
                    "days": d["days"],
                }
                for d in answer["items"]
            ],
        }
        return ocr, expected
    return _read("cv_a1", sample_id), _read("cv_expected", sample_id)


COMPARE_KEYS = ("name", "printed_code", "count_per_dose", "times_per_day", "days")


def _same(key: str, got: Any, want: Any) -> bool:
    if key == "name":
        return normalize_name(got) == normalize_name(want)
    return got == want


def compare(order, expected: dict[str, Any]) -> list[str]:
    """검증을 거친 결과와 정답의 차이를 사람이 읽는 문장으로 돌려준다. 같으면 빈 목록."""
    diffs = []
    if order.issue_no != expected["issue_no"]:
        diffs.append(f"교부번호: 정답 {expected['issue_no']!r}, LLM {order.issue_no!r}")
    want_drugs = expected["items"]
    if len(order.items) != len(want_drugs):
        diffs.append(f"약품 수: 정답 {len(want_drugs)}, LLM {len(order.items)}")
    for i, (got, want) in enumerate(zip(order.items, want_drugs), start=1):
        got_values = {**got.model_dump(), "times_per_day": len(got.times)}   # 하루 횟수 = 복용 시점의 수
        for key in COMPARE_KEYS:
            if not _same(key, got_values[key], want[key]):
                diffs.append(f"약품 {i} {key}: 정답 {want[key]!r}, LLM {got_values[key]!r}")
    return diffs


# ------------------------------------------------------------ 실행


def check_one(
    sample_id: str,
    *,
    llm_fn: Callable[..., dict[str, Any]] = extract_llm,
    out: Callable[[str], None] = print,
) -> bool:
    """처방전 한 장으로 LLM 경로를 한 번 돌려 본다. 정답과 같으면 True."""
    ocr, expected = load_sample(sample_id)
    text = ocr_to_llm_text(ocr)
    out(f"[{sample_id}] LLM에 보내는 텍스트")
    for line in text.split("\n"):
        out(f"    {line}")
    out(f"[{sample_id}] 호출 중... (최대 {config.LLM_TIMEOUT_SECONDS:g}초)")

    started = time.perf_counter()
    try:
        extracted = llm_fn(text, None)
    except ConfigError as e:
        out(f"[{sample_id}] 설정 문제: {e}")
        return False
    except StructureError as e:
        out(f"[{sample_id}] 호출 실패 ({time.perf_counter() - started:.1f}초 만에): {e}")
        cause = e.__cause__
        if cause is not None:
            out(f"    원인: {type(cause).__name__}: {redact(str(cause))}")
            hint = hint_for(cause)
            if hint:
                out(f"    확인: {hint}")
        return False
    elapsed = time.perf_counter() - started
    out(f"[{sample_id}] 응답 {elapsed:.1f}초")

    try:
        order = order_from_extracted(f"TRY-{sample_id}", extracted, source="llm")
    except StructureError as e:
        out(f"[{sample_id}] 응답은 왔지만 검증에서 거절되었습니다: {e}")
        out("    LLM 응답 원본:")
        for line in json.dumps(extracted, ensure_ascii=False, indent=2).split("\n"):
            out(f"    {line}")
        return False

    for warning in order.warnings:
        out(f"    경고: {warning}")
    diffs = compare(order, expected)
    for diff in diffs:
        out(f"    차이 - {diff}")
    out(f"[{sample_id}] 결과: " + ("정답과 일치" if not diffs else f"정답과 {len(diffs)}곳 다름"))
    return not diffs


def main(
    argv: list[str] | None = None,
    *,
    llm_fn: Callable[..., dict[str, Any]] = extract_llm,
    out: Callable[[str], None] = print,
) -> int:
    ids = sample_ids()
    parser = argparse.ArgumentParser(description="OpenAI 실제 호출 확인 (가상 처방전만 보낸다)")
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="처방전 이름 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="샘플 전부")
    args = parser.parse_args(argv)

    key_state = "설정됨" if os.getenv("OPENAI_API_KEY", "").strip() else "없음"
    out(f"모델: {config.OPENAI_MODEL} | 제한 시간: {config.LLM_TIMEOUT_SECONDS:g}초 | OPENAI_API_KEY: {key_state}")

    targets = ids if args.all else [args.sample]
    results = [check_one(sample_id, llm_fn=llm_fn, out=out) for sample_id in targets]
    if len(targets) > 1:
        out(f"전체 {len(targets)}장 중 {sum(results)}장이 정답과 일치")
    return 0 if all(results) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
