"""오래 걸리는 처리(OCR·Agent)를 요청 밖에서 돌린다.

gunicorn 워커 1개 + 스레드 구성(명세서 2.5절)이라 프로세스 안 스레드 풀로 충분하다. 업로드 요청은
바로 응답하고, 화면은 2초마다 상태를 다시 읽는다.

주의: 서버를 재시작하면 돌던 작업은 사라지고 처방은 '처리 중'에 남는다. LangGraph 체크포인터가 붙으면
그래프 상태에서 이어 갈 수 있다. 그 전까지는 다시 업로드한다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from flask import Flask, current_app


def init_app(app: Flask) -> None:
    app.extensions["jobs"] = ThreadPoolExecutor(max_workers=app.config["JOB_WORKERS"],
                                                thread_name_prefix="job")


def submit(fn: Callable[..., Any], *args: Any) -> None:
    """fn(*args)를 앱 컨텍스트 안에서 실행한다. JOBS_SYNC(테스트)면 지금 바로 실행한다."""
    app = current_app._get_current_object()
    if app.config.get("JOBS_SYNC"):
        _run(app, fn, args)
    else:
        app.extensions["jobs"].submit(_run, app, fn, args)


def _run(app: Flask, fn: Callable[..., Any], args: tuple) -> None:
    with app.app_context():
        try:
            fn(*args)
        except Exception:   # 작업 함수가 처방 상태를 실패로 바꾼다. 여기는 놓친 예외를 로그로 남기는 마지막 그물
            app.logger.exception("백그라운드 작업 실패: %s%r", fn.__name__, args)
