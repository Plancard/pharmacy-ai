"""웹 ↔ 진짜 Agent(LangGraph) 연결부 (11/16 초안, 11/23 웹 연동). .env AGENT_BACKEND=langgraph 일 때 쓴다.

FakeGateway(web/agent_gateway.py)와 메서드 · 돌려주는 값이 같다. 웹의 나머지(라우트 · workflow · 화면)는 바꾸지 않는다.

    start(rx_id, image_path)          CV OCR(A.1) → AgentApi.start → 승인 대기면 Pause
    pending(rx_id)                    AgentApi.pending → Pause | None
    resume(rx_id, hitl_point, reply)  AgentApi.resume → 다시 멈추면 Pause, 조제 계획까지 끝나면 Wait('waiting_tray')
    approved_order(rx_id)             AgentApi.approved_order
    tray(rx_id, image_id, image_path) AgentApi.send(…, "tray") → 불일치가 있으면 Pause(inspection), 전 칸 일치면 Wait('done')
    inspection(rx_id) · tray_error(rx_id)   AgentApi.result의 검수 결과(A.3) · 트레이 검수 실패 코드와 안내
    check_reply(rx_id, hitl_point, reply)   응답이 맞는지 미리 본다 (라우트가 백그라운드로 넘기기 전에 쓰면 400으로 거를 수 있다)
    close()                           MCP 연결 등을 닫는다

대응 (기능분석서 5.2절)
- Agent가 멈춤(Pause)                    → 처방 상태 waiting_approval (hitl_point는 Pause.hitl_point)
- Agent stage 'await_tray'(status planned) → Wait('waiting_tray') (HITL ② 재조제도 여기로)
- Agent stage 'inspected' → Wait('done') (전 칸 일치, 또는 HITL ② 확인 완료)
- 그 밖의 stage는 None (상태를 바꾸지 않는다)
- 트레이 검수 실패(E_TRAY_NOT_FOUND 등)는 Agent가 planned에 머물고 AgentRunError를 올린다 → AgentError(같은 코드).
  웹은 처방을 실패로 두지 않고 트레이 대기로 돌려 다시 찍게 한다 (web/workflow.run_tray)
- AgentRunError(A.7 코드) → AgentError(같은 코드). CV 오류(OcrEmpty 등)는 .code가 있어 그대로 올린다.

체크포인트는 AgentApi.from_settings가 .env DATABASE_URL(웹과 같은 PostgreSQL)에 붙인다 (PostgresSaver).
그래서 서버를 껐다 켜도 승인 대기 처방은 같은 곳에서 이어진다 (REQ-WEB-03). Agent는 웹 테이블을 직접 쓰지 않는다.
OCR은 .env CV_USE_STUB=1이면 cv/stub.py(시연용 고정 결과), 0이면 cv/ocr.py(EasyOCR)다.
트레이 검수도 같다: 1이면 agent.cv_adapter.stub_inspect(CV_STUB_SCENARIO), 0이면 cv/tray_inspect.py(보정 → 검출 → 판정).
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any, Callable

from web.agent_gateway import AgentError, Pause, Wait

OcrFn = Callable[[Path, str], dict[str, Any]]      # (이미지 경로, 사진 ID) -> A.1
STAGE_TO_WAIT = {"await_tray": "waiting_tray", "inspected": "done"}     # Agent stage -> 웹 처방 상태 (Wait)
DEFAULT_ERROR = "E_LLM"                            # 코드 없는 Agent 실패 (A.7에 가장 가까운 것)


RX8_SAMPLES = Path(__file__).resolve().parents[1] / "agent" / "resources" / "samples" / "rx8_a1"


def stub_ocr(image_path: Path, image_id: str) -> dict[str, Any]:
    """시연 · 시험용 고정 OCR 결과. 업로드한 사진은 읽지 않는다.

    .env CV_STUB_SAMPLE=rx8_01 처럼 주면 Agent의 가상 처방전 20장(agent/resources/samples/rx8_a1, 대상 8종만)에서 고른다.
    없으면 cv/stub.py가 고른다 (업로드 이름이 fixture에 없으면 rx_001: 대상 밖 약 '알모란정'이 있어 약사가 고쳐야 승인된다).
    """
    import json

    from cv import stub

    sample = os.getenv("CV_STUB_SAMPLE", "").strip()
    if sample.startswith("rx8_") and (RX8_SAMPLES / f"{sample}.json").exists():
        if os.getenv("CV_STUB_SCENARIO", "") == "ocr_empty":
            raise stub.OcrEmpty(image_id)
        a1 = json.loads((RX8_SAMPLES / f"{sample}.json").read_text(encoding="utf-8"))
    else:
        a1 = stub.ocr(Path(image_path).with_name(f"{sample}.jpg") if sample else Path(image_path))
    a1["image_id"] = image_id
    return a1


def real_ocr(image_path: Path, image_id: str) -> dict[str, Any]:
    """cv/ocr.py(처방전 펴기 + EasyOCR) → A.1. 처음 부를 때 모델을 읽어 몇 초 걸린다."""
    from cv.ocr import extract_text

    return extract_text(Path(image_path), "prescription", image_id)


def tray_path(image_id: str) -> Path:
    """트레이 사진 ID(웹 images.id) → 저장된 파일 경로. 백그라운드 작업(앱 컨텍스트 안)에서 부른다."""
    import uuid

    from cv.stub import CVError
    from web import rx, storage

    try:
        key = rx.tray_image_key(uuid.UUID(str(image_id)))
    except ValueError:
        key = None
    if key is None:
        raise CVError(f"트레이 사진을 찾지 못했습니다: {image_id}")
    return storage.path_of(key)


def real_inspect(image_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    """cv/tray_inspect.py로 검수한다 (Agent 그래프가 받는 모양: (사진 ID, A.4) -> A.3). A.3 image_id는 웹의 사진 ID다.

    agent.cv_adapter.file_inspect와 같은 일인데, A.3 image_id를 파일 이름이 아니라 웹의 사진 ID로 맞추기 위해 따로 둔다
    (웹이 검수 결과를 inspections에 남길 때 이 ID로 사진과 잇는다). 처음 부를 때 모델 두 개를 읽어 몇 초 걸린다.
    """
    from cv.tray_inspect import inspect_tray

    return inspect_tray(tray_path(image_id), plan, image_id=image_id)


class LangGraphGateway:
    def __init__(self, api: Any, ocr: OcrFn):
        self.api = api              # agent.api.AgentApi
        self.ocr = ocr

    def _outcome(self, rx_id: str, pause: Any) -> Pause | Wait | None:
        if pause is not None:
            return Pause(pause.hitl_point, dict(pause.payload))
        status = STAGE_TO_WAIT.get(self.api.stage(rx_id) or "")
        return Wait(status) if status else None

    def _run(self, fn: Callable[[], Any]) -> Any:
        from agent.api import AgentRunError

        try:
            return fn()
        except AgentRunError as e:
            raise AgentError(e.code or DEFAULT_ERROR, str(e)) from None

    def start(self, rx_id: str, image_path: Path) -> Pause | Wait | None:
        image_id = Path(image_path).stem
        a1 = self.ocr(Path(image_path), image_id)            # CV 오류(E_OCR_EMPTY 등)는 그대로 올라간다
        pause = self._run(lambda: self.api.start(rx_id, a1, "prescription", image_id=image_id,
                                                 prescription_id=rx_id))
        return self._outcome(rx_id, pause)

    def pending(self, rx_id: str) -> Pause | None:
        pause = self.api.pending(rx_id)
        return Pause(pause.hitl_point, dict(pause.payload)) if pause is not None else None

    def check_reply(self, rx_id: str, hitl_point: str, reply: dict[str, Any]) -> None:
        """맞지 않으면 ValueError (문장은 화면에 보여도 된다). 그래프는 그대로다."""
        self.api.check_reply(rx_id, hitl_point, reply)

    def resume(self, rx_id: str, hitl_point: str, reply: dict[str, Any]) -> Pause | Wait | None:
        pause = self._run(lambda: self.api.resume(rx_id, hitl_point, reply))
        return self._outcome(rx_id, pause)

    def approved_order(self, rx_id: str) -> dict[str, Any] | None:
        return self.api.approved_order(rx_id)

    def tray(self, rx_id: str, image_id: str, image_path: Path) -> Pause | Wait | None:
        """트레이 사진을 검수한다. Agent는 사진 ID만 받고, CV(build_graph(inspect=…))가 ID로 사진을 찾는다.

        지금은 stub_inspect라 사진을 읽지 않는다. 실제 CV를 붙일 때 file_inspect(cv 함수, 사진 ID → 경로)로 바꾼다.
        """
        pause = self._run(lambda: self.api.send(rx_id, None, "tray", image_id=image_id, prescription_id=rx_id))
        return self._outcome(rx_id, pause)

    def inspection(self, rx_id: str) -> dict[str, Any] | None:
        return self.api.result(rx_id).get("inspection")

    def tray_error(self, rx_id: str) -> dict[str, Any] | None:
        """트레이 대기에 머무는 이유. 승인 직후처럼 오류가 없으면 None."""
        result = self.api.result(rx_id)
        if not result.get("error_code") or self.api.stage(rx_id) != "await_tray":
            return None
        return {"code": result["error_code"], "notice": result.get("notice")}

    def close(self) -> None:
        self.api.close()


def from_settings() -> LangGraphGateway:
    """.env대로 만든다: AgentApi.from_settings(사전 · 의약품 API · DUR · LLM · 체크포인트 · MCP) + CV OCR · 트레이 검수."""
    from agent.api import AgentApi
    from agent.cv_adapter import stub_inspect

    use_stub = os.getenv("CV_USE_STUB", "1").strip() != "0"
    kwargs: dict[str, Any] = {"inspect": stub_inspect() if use_stub else real_inspect}
    gateway = LangGraphGateway(AgentApi.from_settings(**kwargs), stub_ocr if use_stub else real_ocr)
    atexit.register(gateway.close)
    return gateway
