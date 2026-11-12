"""Agent가 CV 트레이 검수를 부르는 연결 (가장자리, 14단계). 그래프는 InspectFn = (사진 ID, 조제 계획 A.4) -> 검수 결과 A.3 만 안다.

    stub_inspect(scenario)     CV 가짜 모듈(cv/stub.py)로 검수한다. 데모 · 시험 · 웹 연동 연습용. 사진 ID를 그대로 파일 이름으로 쓴다
    file_inspect(fn, resolve)  실제 CV 함수(cv.inspect.inspect_tray 등, (경로, plan) -> A.3)와 '사진 ID -> 파일 경로'를 묶는다

웹은 사진을 저장한 경로를 알고 있으므로 file_inspect(cv의 함수, 웹의 경로 찾기)로 만들어 build_graph(inspect=...)에 넘긴다.
CV 모듈이 바뀌어도(cv.stub -> cv.pipeline) Agent 쪽은 이 파일만 바꾼다. 판정 규칙은 CV에 있고 여기에는 없다.
"""
from __future__ import annotations

from typing import Any, Callable

InspectFn = Callable[[str, dict[str, Any]], dict[str, Any]]    # 그래프가 받는 모양 (nodes.InspectFn과 같다): (사진 ID, A.4) -> A.3
CvInspect = Callable[[str, dict[str, Any]], dict[str, Any]]     # CV 함수: (이미지 경로, A.4) -> A.3
Resolve = Callable[[str], str]                                  # 사진 ID -> 이미지 경로 (웹의 images 테이블)


def file_inspect(cv_inspect: CvInspect, resolve: Resolve) -> InspectFn:
    def inspect(image_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        return cv_inspect(resolve(image_id), plan)
    return inspect


def stub_inspect(scenario: str | None = None) -> InspectFn:
    """cv/stub.py의 inspect로 검수한다. scenario: ok · mismatch · review · rotated · no_tray (없으면 CV_STUB_SCENARIO, 기본 ok)."""
    from cv import stub          # 저장소 루트의 cv 패키지. 쓸 때만 읽는다 (Agent만 따로 시험할 때 필요 없다)

    def inspect(image_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        return stub.inspect(image_id, plan, scenario=scenario)
    return inspect
