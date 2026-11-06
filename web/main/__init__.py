"""화면 3개 중 로그인 뒤 두 개: 메인(대화)·조제 이력. 지금은 뼈대이고 카드는 2주차에 채운다."""
from __future__ import annotations

from flask import Blueprint, g, render_template

from web import db
from web.auth import login_required

bp = Blueprint("main", __name__)


@bp.route("/")
@login_required
def index():
    """메인 대화 화면 (F-WEB-02~11). 11/11부터 업로드·카드를 붙인다."""
    recent = db.query_all(
        "SELECT id, display_no, status, hitl_point, created_at FROM prescriptions "
        "WHERE pharmacy_id = %s ORDER BY created_at DESC LIMIT 20",
        (g.user["pharmacy_id"],),
    )
    return render_template("main/index.html", recent=recent)


@bp.route("/history")
@login_required
def history():
    """조제 이력 (F-WEB-12·13, REQ-WEB-10). 필터는 11/23에 붙인다."""
    records = db.query_all(
        "SELECT p.display_no, d.risk_level, d.completed_at, u.username "
        "FROM dispense_records d "
        "JOIN prescriptions p ON p.id = d.prescription_id "
        "JOIN users u ON u.id = d.user_id "
        "WHERE p.pharmacy_id = %s ORDER BY d.completed_at DESC LIMIT 100",
        (g.user["pharmacy_id"],),
    )
    return render_template("main/history.html", records=records)
