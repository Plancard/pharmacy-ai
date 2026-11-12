"""웹 앱 팩토리. CV·Agent·웹이 이 Flask 앱 하나 안에서 돈다 (명세서 2.1절).

실행
  개발:   flask --app wsgi run --debug
  Docker: docker compose up --build   (gunicorn 워커 1개 + 스레드, 명세서 2.5절)
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, render_template

from web.config import check_required, load_config

STATUS_LABELS = {
    "processing": "처리 중",
    "waiting_approval": "승인 대기",
    "waiting_tray": "트레이 대기",
    "done": "완료",
    "failed": "실패",
}


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(load_config())
    if test_config:
        app.config.update(test_config)
    check_required(app.config)

    if app.config["TRUST_PROXY"]:   # 2차: Caddy 뒤에서 실제 접속 주소·https 여부를 받는다
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    from web import agent_gateway, db, jobs, security
    db.init_app(app)
    security.init_app(app)
    jobs.init_app(app)
    agent_gateway.init_app(app)

    from web.api import bp as api_bp
    from web.auth import bp as auth_bp
    from web.main import bp as main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    tz = ZoneInfo(app.config["DISPLAY_TZ"])

    @app.template_filter("kst")
    def to_local(value: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
        """DB의 UTC 시각을 화면 시간대(기본 KST)로 바꿔 보여 준다."""
        return value.astimezone(tz).strftime(fmt) if value else ""

    app.jinja_env.globals["status_label"] = lambda s: STATUS_LABELS.get(s, s)
    app.jinja_env.globals["status_labels"] = STATUS_LABELS

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    def client_error(e):
        return render_template("error.html", code=e.code, message=e.description), e.code

    @app.errorhandler(500)
    def server_error(e):
        return render_template("error.html", code=500, message="서버 오류가 났습니다. 잠시 뒤 다시 시도해 주세요."), 500

    return app
