"""OpenAI 실제 호출 확인 (11/4, 11/5 실제 처방전 양식으로 바꿈) — 손으로 한 번 돌려 보는 스크립트.

    python try_llm.py            # rx_001 한 장
    python try_llm.py rx_005     # 지정한 한 장
    python try_llm.py --all      # 실제 양식 샘플 5장 전부

pytest가 아니다. 실제로 OpenAI를 부르므로 `python -m pytest`에는 들어 있지 않다.
data/agent/cv_samples의 가상 처방전에서 교부번호와 약품 표 본문만 보낸다 (환자·의료인 정보는 보내지 않는다).
키는 화면에 찍지 않는다.
확인하는 것: 모델 이름이 이 계정에서 통하는지 / 응답이 우리 모양(LlmPrescription)으로 오는지 /
걸리는 시간이 제한 시간 안인지 / 결과가 정답(data/agent/cv_expected/)과 같은지 /
약품 매칭 뒤 품목기준코드가 정답과 같은지.
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

from agent import config
from agent.errors import ConfigError, StructureError
from agent.drug_dictionary import DrugDictionary, load_dictionary
from agent.drug_match import match_order
from agent.drug_names import normalize_name
from agent.llm_extract import extract_llm
from agent.structure import ocr_to_llm_text
from agent.validate import order_from_extracted

DATA = Path(__file__).resolve().parent / "data" / "agent"
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


# 정답과 견주는 칸. 이름은 띄어쓰기·'/'만 다른 것은 같은 것으로 본다 (normalize_name).
COMPARE_KEYS = ("name", "printed_code", "dose_per_time", "times_per_day", "days")


def sample_ids() -> list[str]:
    return sorted(p.stem for p in (DATA / "cv_expected").glob("rx_*.json"))


def load_sample(sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    def read(kind: str) -> dict[str, Any]:
        return json.loads((DATA / kind / f"{sample_id}.json").read_text(encoding="utf-8"))

    return read("cv_samples"), read("cv_expected")


def _same(key: str, got: Any, want: Any) -> bool:
    if key == "name":
        return normalize_name(got) == normalize_name(want)
    return got == want


def compare(order, expected: dict[str, Any]) -> list[str]:
    """검증·매칭을 거친 결과와 정답의 차이를 사람이 읽는 문장으로 돌려준다. 같으면 빈 목록."""
    diffs = []
    if order.issue_no != expected["issue_no"]:
        diffs.append(f"교부번호: 정답 {expected['issue_no']!r}, LLM {order.issue_no!r}")
    want_drugs = expected["drugs"]
    if len(order.drugs) != len(want_drugs):
        diffs.append(f"약품 수: 정답 {len(want_drugs)}, LLM {len(order.drugs)}")
    for i, (got, want) in enumerate(zip(order.drugs, want_drugs), start=1):
        got_values = got.model_dump()
        for key in (*COMPARE_KEYS, "item_code"):
            if not _same(key, got_values[key], want[key]):
                diffs.append(f"약품 {i} {key}: 정답 {want[key]!r}, LLM {got_values[key]!r}")
    return diffs


def check_one(
    sample_id: str,
    *,
    dictionary: DrugDictionary,
    llm_fn: Callable[..., dict[str, Any]] = extract_llm,
    out: Callable[[str], None] = print,
) -> bool:
    """처방전 한 장으로 LLM 경로 + 약품 매칭을 한 번 돌려 본다. 정답과 같으면 True."""
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

    order = match_order(order, dictionary)
    for warning in order.warnings:
        out(f"    경고: {warning}")
    for i, drug in enumerate(order.drugs, start=1):
        if drug.needs_pharmacist_check:
            out(f"    약사 확인 {i}번째 약품: " + " / ".join(drug.check_reasons))
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
    parser.add_argument("sample", nargs="?", default=ids[0], choices=ids, help="처방전 번호 (기본 %(default)s)")
    parser.add_argument("--all", action="store_true", help="실제 양식 샘플 전부")
    args = parser.parse_args(argv)

    key_state = "설정됨" if os.getenv("OPENAI_API_KEY", "").strip() else "없음"
    out(f"모델: {config.OPENAI_MODEL} | 제한 시간: {config.LLM_TIMEOUT_SECONDS:g}초 | OPENAI_API_KEY: {key_state}")

    try:
        dictionary = load_dictionary()
    except ConfigError as e:
        out(f"설정 문제: {e}")
        return 1

    targets = ids if args.all else [args.sample]
    results = [check_one(sample_id, dictionary=dictionary, llm_fn=llm_fn, out=out) for sample_id in targets]
    if len(targets) > 1:
        out(f"전체 {len(targets)}장 중 {sum(results)}장이 정답과 일치")
    return 0 if all(results) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 파일로 돌릴 때 글자 하나 때문에 멈추지 않게
    sys.exit(main())
