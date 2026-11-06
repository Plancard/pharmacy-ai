"""설정. 값은 루트 .env(또는 환경변수)에서 읽는다. 키·비밀번호는 코드에 쓰지 않는다 (REQ-NF-07)."""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent   # 저장소 맨 위
load_dotenv(ROOT / ".env")                      # .env가 없어도 오류 없이 지나간다


class ConfigError(RuntimeError):
    """필수 설정이 빠졌을 때. 개발 환경 문제이므로 화면 안내로 바꾸지 않고 그대로 올려 보낸다."""


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config() -> dict:
    return {
        "SECRET_KEY": os.getenv("FLASK_SECRET_KEY", ""),
        # psycopg(libpq) 형식: postgresql://user:password@host:5432/dbname
        "DATABASE_URL": os.getenv("DATABASE_URL", ""),
        "DB_POOL_MAX": int(os.getenv("DB_POOL_MAX", "5")),
        "VAR_DIR": _path(os.getenv("VAR_DIR", "var")),
        "MAX_CONTENT_LENGTH": int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024,
        "DISPLAY_TZ": os.getenv("DISPLAY_TZ", "Asia/Seoul"),
        # 세션: 1차는 로컬 http라 SECURE=0, 2차 HTTPS 배포 때 1
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": os.getenv("SESSION_COOKIE_SECURE", "0") == "1",
        "PERMANENT_SESSION_LIFETIME": timedelta(hours=int(os.getenv("SESSION_HOURS", "8"))),
        # 2차: 리버스 프록시(Caddy) 뒤에서 실행할 때 1
        "TRUST_PROXY": os.getenv("TRUST_PROXY", "0") == "1",
    }


def check_required(config: dict) -> None:
    missing = [k for k in ("SECRET_KEY", "DATABASE_URL") if not config.get(k)]
    if missing:
        names = {"SECRET_KEY": "FLASK_SECRET_KEY", "DATABASE_URL": "DATABASE_URL"}
        raise ConfigError(
            ".env에 " + ", ".join(names[m] for m in missing) + "가 없습니다. "
            ".env.example을 복사해 .env를 만들고 값을 채워 주세요."
        )
