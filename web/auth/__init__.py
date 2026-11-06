"""로그인 (F-WEB-01, REQ-WEB-08).

- 세션 쿠키에 user_id만 둔다. 요청마다 DB에서 사용자를 다시 읽어 g.user에 넣는다.
- 비밀번호는 werkzeug 해시로만 저장한다 (REQ-NF-07).
- 계정은 화면이 아니라 CLI로 만든다: flask --app wsgi create-user <아이디> --pharmacy <약국 이름>
"""
from __future__ import annotations

import functools
from urllib.parse import urlsplit

import click
from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from web import db

bp = Blueprint("auth", __name__, cli_group=None)   # CLI 명령을 `flask create-user`로 바로 쓰게 한다

# 아이디가 없을 때도 같은 시간이 걸리게 하는 더미 해시 (아이디 존재 여부가 응답 시간으로 드러나지 않게)
_DUMMY_HASH = generate_password_hash("dummy-password-for-timing")


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapped


@bp.before_app_request
def load_user():
    g.user = None
    user_id = session.get("user_id")
    if user_id is not None:
        g.user = db.query_one(
            "SELECT u.id, u.username, u.role, u.pharmacy_id, p.name AS pharmacy_name "
            "FROM users u JOIN pharmacies p ON p.id = u.pharmacy_id WHERE u.id = %s",
            (user_id,),
        )
        if g.user is None:      # 계정이 지워졌으면 세션도 비운다
            session.clear()


def _safe_next(target: str | None) -> str:
    """로그인 뒤 돌아갈 주소. 같은 사이트의 경로만 허용한다 (외부 주소로 보내는 공격 방지)."""
    if target:
        parts = urlsplit(target)
        if not parts.scheme and not parts.netloc and target.startswith("/") and not target.startswith("//"):
            return target
    return url_for("main.index")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("main.index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = db.query_one("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        ok = check_password_hash(row["password_hash"] if row else _DUMMY_HASH, password)
        if row and ok:
            session.clear()
            session.permanent = True
            session["user_id"] = row["id"]
            return redirect(_safe_next(request.form.get("next")))
        flash("아이디 또는 비밀번호가 올바르지 않습니다.")
    return render_template("auth/login.html", next=request.args.get("next", ""))


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


# ------------------------------------------------------------ 계정 만들기 (CLI)
@bp.cli.command("create-user")
@click.argument("username")
@click.option("--pharmacy", "pharmacy_name", required=True, help="소속 약국 이름 (없으면 새로 만든다)")
@click.option("--role", type=click.Choice(["pharmacist", "admin"]), default="pharmacist", show_default=True)
@click.password_option("--password", prompt="비밀번호", confirmation_prompt="비밀번호 확인")
def create_user(username: str, pharmacy_name: str, role: str, password: str) -> None:
    """약사 계정을 만든다."""
    if len(password) < 8:
        raise click.ClickException("비밀번호는 8자 이상이어야 합니다.")
    if db.query_one("SELECT 1 FROM users WHERE username = %s", (username,)):
        raise click.ClickException(f"이미 있는 아이디입니다: {username}")
    pharmacy = db.query_one("SELECT id FROM pharmacies WHERE name = %s ORDER BY id LIMIT 1", (pharmacy_name,))
    if pharmacy is None:
        pharmacy = db.execute("INSERT INTO pharmacies (name) VALUES (%s) RETURNING id", (pharmacy_name,))
        click.echo(f"약국을 새로 만들었습니다: {pharmacy_name}")
    db.execute(
        "INSERT INTO users (pharmacy_id, username, password_hash, role) VALUES (%s, %s, %s, %s)",
        (pharmacy["id"], username, generate_password_hash(password), role),
    )
    click.echo(f"계정을 만들었습니다: {username} ({pharmacy_name}, {role})")
