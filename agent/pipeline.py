"""처방 구조화 함수 (3단계, REQ-AG-02). 규칙으로 읽고, 안 되면 LLM으로 읽는다.

규칙 경로 → 검증 → (실패하면) LLM 경로 → 검증 → (거절되면) 횟수를 제한해 재시도.
두 경로 모두 validate.order_from_extracted를 통과해야 결과로 인정한다.

판단 층이라 LLM을 직접 부르지 않는다. llm_fn(text, feedback) 호출 함수를 인자로 받는다.
실제 호출 함수는 llm.extract_llm이고, 그래프(nodes.py)가 이어 붙인다. 시험에서는 가짜 함수를 넘긴다.

처리 시간 (처방전 1장 → 조제지시서 1분 이내, REQ-NF-01)
- 규칙 경로는 거의 즉시 끝난다.
- LLM은 호출 한 번이 LLM_TIMEOUT_SECONDS(기본 20초)를 넘지 않고, SDK 자동 재시도는 꺼 두었다.
  규칙 경로가 실패해도 LLM 호출은 최대 (1 + llm_retries)번이라 기본값이면 최악 약 40초다.
  LLM_TIMEOUT_SECONDS나 llm_retries를 늘리면 이 계산이 달라지니 같이 보자.
"""
from __future__ import annotations

from typing import Any, Callable

from agent import config
from agent.errors import E_LLM, E_OCR_EMPTY, ConfigError, StructureError
from agent.schema import DispensingOrder
from agent.structure import extract_rule, format_llm_text, llm_input
from agent.validate import order_from_extracted

RuleFn = Callable[[dict[str, Any]], dict[str, Any]]
LlmFn = Callable[[str, "str | None"], dict[str, Any]]


def structure_prescription(
    prescription_id: str,
    ocr: dict[str, Any],
    *,
    llm_fn: LlmFn,
    rule_fn: RuleFn = extract_rule,
    llm_retries: int = config.LLM_MAX_RETRIES,
) -> DispensingOrder:
    """A.1 형식의 OCR 결과 하나를 조제지시서로 만든다.

    - 규칙 경로 결과가 검증을 통과하면 그대로 쓴다. LLM은 부르지 않는다.
    - 규칙 경로가 읽지 못했거나 검증에서 거절되면 LLM 경로로 넘어간다.
      이때 조제지시서의 warnings에 그 사실과 이유가 남는다. (약사가 원문과 대조하도록)
    - LLM 결과가 검증에서 거절되면 거절 사유를 모델에게 알려 주며 최대 llm_retries번 다시 시도한다.
    - 모두 실패하면 StructureError. 메시지에 경로별 이유가 들어 있다. (처방전 내용은 없다)
      서비스는 멈추지 않는다: 웹은 이 오류 코드를 보고 약사에게 재시도·직접 입력을 고르게 한다 (REQ-NF-06).
      오류 코드(A.7): OCR에 글자가 없거나 약품 표 줄을 못 찾으면 E_OCR_EMPTY, LLM이 끝내 못 읽으면 E_LLM.
      (처방 ID가 없는 것은 처방전이 아니라 부르는 쪽의 잘못이라 코드가 없다)
    - 키가 없는 등의 개발 환경 문제(ConfigError)는 재시도하지 않고 그대로 올려 보낸다.
    """
    if not isinstance(prescription_id, str) or not prescription_id.strip():
        raise StructureError("처방 ID가 없습니다.")
    # OCR 형식이 틀렸으면 어느 경로도 쓸 수 없다. 여기서 바로 StructureError
    issue_no, table_rows = llm_input(ocr)   # LLM에는 교부번호와 약품 표만 보낸다 (환자 정보는 보내지 않는다)

    failures: list[str] = []

    # 1) 규칙 경로
    try:
        return order_from_extracted(prescription_id, rule_fn(ocr), source="rule")
    except ConfigError:
        raise
    except StructureError as e:
        rule_error = str(e)
    except Exception as e:   # 규칙 경로의 예기치 못한 오류가 서비스를 멈추게 하지 않는다 (REQ-NF-06)
        rule_error = f"규칙 경로에서 예기치 못한 오류가 났습니다 ({type(e).__name__})."
    failures.append(f"[규칙] {rule_error}")

    # 2) LLM 경로: 처음 1번 + 재시도. 보낼 약품 표 줄이 없으면 LLM도 읽을 수 없으니 부르지 않는다.
    if not table_rows:
        failures.append("[LLM] 처방전에서 약품 표 줄을 찾지 못해 LLM에 보내지 않았습니다.")
        raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요. " + " ".join(failures), code=E_OCR_EMPTY)
    text = format_llm_text(issue_no, table_rows)
    feedback: str | None = None
    for attempt in range(1, 2 + max(0, llm_retries)):
        try:
            extracted = llm_fn(text, feedback)
        except StructureError as e:   # 호출 실패·응답 없음. 모델에게 알려 줄 거절 사유는 없다
            # (ConfigError는 StructureError가 아니라서 여기 걸리지 않고 그대로 올라간다)
            failures.append(f"[LLM {attempt}회] {e}")
            continue

        try:
            order = order_from_extracted(prescription_id, extracted, source="llm")
        except StructureError as e:   # 검증에서 거절됨. 사유를 다음 시도에 알려 준다
            failures.append(f"[LLM {attempt}회] {e}")
            feedback = str(e)
            continue

        notes = [f"규칙 기반으로 읽지 못해 LLM으로 읽었습니다 ({rule_error}) 원문과 꼭 대조하세요."]
        if attempt > 1:
            notes.append(f"LLM이 {attempt}번째 시도에서 읽었습니다. 원문과 꼭 대조하세요.")
        order.warnings = notes + order.warnings
        return order

    raise StructureError("자동 구조화에 실패했습니다. 직접 입력해 주세요. " + " ".join(failures), code=E_LLM)
