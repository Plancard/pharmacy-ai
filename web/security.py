"""CSRF 방어. 폼은 hidden csrf_token, 이후 fetch 요청은 X-CSRF-Token 헤더로 보낸다."""
from __future__ import annotations

import hmac
import secrets

from flask import Flask, abort, request, session

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
EXEMPT_ENDPOINTS = {"api.health"}


def csrf_token() -> str:
    token = session.get("_csrf")
    if not token:
        token = session["_csrf"] = secrets.token_urlsafe(32)
    return token


def init_app(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def _check_csrf():
        if request.method in SAFE_METHODS or request.endpoint in EXEMPT_ENDPOINTS:
            return None
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        expected = session.get("_csrf", "")
        if not expected or not hmac.compare_digest(sent, expected):
            abort(400, description="요청이 만료되었습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.")
        return None
