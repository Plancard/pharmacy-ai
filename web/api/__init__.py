"""JSON API. 상태 조회(2초 폴링)·업로드·승인은 2주차에 이 blueprint에 붙인다."""
from __future__ import annotations

from flask import Blueprint, jsonify

from web import db

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.get("/health")
def health():
    """Docker healthcheck용. 로그인 없이 열려 있고 DB 연결만 확인한다."""
    ok = db.ping()
    return jsonify(status="ok" if ok else "db_error", db=ok), (200 if ok else 503)
