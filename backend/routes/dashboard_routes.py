from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func, extract
from datetime import datetime, timedelta, timezone
import logging

from database.database import db
from database.models import User, Prediction
from auth.auth_utils import sanitize_string

logger = logging.getLogger(__name__)
dashboard_bp = Blueprint("dashboard", __name__)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_user(user_id: int) -> User:
    return User.query.filter_by(id=user_id, is_active=True).first()


# ─── Profile ──────────────────────────────────────────────────────────────────

@dashboard_bp.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    user = _get_user(int(get_jwt_identity()))
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    return jsonify({"success": True, "data": user.to_dict()}), 200


@dashboard_bp.route("/profile", methods=["PUT"])
@jwt_required()
def update_profile():
    user = _get_user(int(get_jwt_identity()))
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    body    = request.get_json(silent=True) or {}
    updated = []

    if "full_name" in body:
        user.full_name = sanitize_string(body["full_name"])
        updated.append("full_name")
    if "age" in body:
        user.age = int(body["age"])
        updated.append("age")
    if "gender" in body and body["gender"] in ("Male", "Female", "Other"):
        user.gender = body["gender"]
        updated.append("gender")

    if not updated:
        return jsonify({"success": False, "message": "No valid fields to update"}), 400

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.exception("Profile update failed: %s", exc)
        return jsonify({"success": False, "message": "Update failed"}), 500

    return jsonify({"success": True, "message": "Profile updated", "updated_fields": updated}), 200


# ─── Summary ──────────────────────────────────────────────────────────────────

@dashboard_bp.route("/summary", methods=["GET"])
@jwt_required()
def get_summary():
    user_id = int(get_jwt_identity())
    user    = _get_user(user_id)
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    total = Prediction.query.filter_by(user_id=user_id).count()

    if total == 0:
        return jsonify({"success": True, "data": {
            "total_predictions":  0,
            "risk_distribution":  {"Low": 0, "Medium": 0, "High": 0},
            "category_breakdown": {"Non-Diabetic": 0, "Pre-Diabetic": 0, "Diabetic": 0},
            "average_risk_score": None,
            "latest_prediction":  None,
            "trend_last_30_days": 0,
        }}), 200

    # Risk distribution
    risk_rows = (db.session.query(Prediction.risk_level, func.count(Prediction.id))
                 .filter_by(user_id=user_id).group_by(Prediction.risk_level).all())
    risk_dist = {"Low": 0, "Medium": 0, "High": 0}
    for level, count in risk_rows:
        if level in risk_dist:
            risk_dist[level] = count

    # Category breakdown
    cat_rows = (db.session.query(Prediction.diabetes_category, func.count(Prediction.id))
                .filter_by(user_id=user_id).group_by(Prediction.diabetes_category).all())
    cat_breakdown = {"Non-Diabetic": 0, "Pre-Diabetic": 0, "Diabetic": 0}
    for cat, count in cat_rows:
        if cat in cat_breakdown:
            cat_breakdown[cat] = count

    # Average risk score
    avg = db.session.query(func.avg(Prediction.risk_score)).filter_by(user_id=user_id).scalar()

    # Latest prediction
    latest = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).first()
    latest_data = None
    if latest:
        latest_data = {
            "id":                latest.id,
            "diabetes_category": latest.diabetes_category,
            "risk_level":        latest.risk_level,
            "risk_score":        latest.risk_score,
            "created_at":        latest.created_at.isoformat(),
        }

    # 30-day trend
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    trend  = Prediction.query.filter_by(user_id=user_id).filter(
        Prediction.created_at >= cutoff).count()

    return jsonify({"success": True, "data": {
        "total_predictions":  total,
        "risk_distribution":  risk_dist,
        "category_breakdown": cat_breakdown,
        "average_risk_score": round(float(avg), 4) if avg else None,
        "latest_prediction":  latest_data,
        "trend_last_30_days": trend,
    }}), 200


# ─── Prediction History ───────────────────────────────────────────────────────

@dashboard_bp.route("/predictions", methods=["GET"])
@jwt_required()
def get_history():
    user_id    = int(get_jwt_identity())
    page       = request.args.get("page",       1,    type=int)
    per_page   = min(request.args.get("per_page", 10, type=int), 100)
    risk_level = request.args.get("risk_level")
    category   = request.args.get("category")
    sort       = request.args.get("sort", "desc")

    query = Prediction.query.filter_by(user_id=user_id)

    if risk_level in ("Low", "Medium", "High"):
        query = query.filter_by(risk_level=risk_level)
    if category in ("Non-Diabetic", "Pre-Diabetic", "Diabetic"):
        query = query.filter_by(diabetes_category=category)

    query     = query.order_by(
        Prediction.created_at.asc() if sort == "asc" else Prediction.created_at.desc()
    )
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({"success": True,
        "data": [p.to_dict(include_features=False) for p in paginated.items],
        "pagination": {
            "total":    paginated.total,
            "pages":    paginated.pages,
            "page":     paginated.page,
            "per_page": paginated.per_page,
            "has_next": paginated.has_next,
            "has_prev": paginated.has_prev,
        }
    }), 200


# ─── Single Prediction ────────────────────────────────────────────────────────

@dashboard_bp.route("/predictions/<int:prediction_id>", methods=["GET"])
@jwt_required()
def get_prediction(prediction_id: int):
    pred = Prediction.query.filter_by(id=prediction_id, user_id=int(get_jwt_identity())).first()
    if not pred:
        return jsonify({"success": False, "message": "Prediction not found"}), 404
    return jsonify({"success": True, "data": pred.to_dict()}), 200


# ─── Delete Prediction ────────────────────────────────────────────────────────

@dashboard_bp.route("/predictions/<int:prediction_id>", methods=["DELETE"])
@jwt_required()
def delete_prediction(prediction_id: int):
    pred = Prediction.query.filter_by(id=prediction_id, user_id=int(get_jwt_identity())).first()
    if not pred:
        return jsonify({"success": False, "message": "Prediction not found"}), 404

    try:
        db.session.delete(pred)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.exception("Delete failed: %s", exc)
        return jsonify({"success": False, "message": "Delete failed"}), 500

    return jsonify({"success": True, "message": "Prediction deleted"}), 200


# ─── Monthly Trends ───────────────────────────────────────────────────────────

@dashboard_bp.route("/trends/monthly", methods=["GET"])
@jwt_required()
def monthly_trends():
    user_id = int(get_jwt_identity())
    months  = min(request.args.get("months", 6, type=int), 12)
    cutoff  = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows = (db.session.query(
                extract("year",  Prediction.created_at).label("year"),
                extract("month", Prediction.created_at).label("month"),
                Prediction.risk_level,
                func.count(Prediction.id).label("count"),
            )
            .filter(Prediction.user_id == user_id, Prediction.created_at >= cutoff)
            .group_by("year", "month", Prediction.risk_level)
            .order_by("year", "month")
            .all())

    trend_map = {}
    for row in rows:
        key   = f"{int(row.year)}-{int(row.month):02d}"
        entry = trend_map.setdefault(key, {"month": key, "Low": 0, "Medium": 0, "High": 0, "total": 0})
        if row.risk_level in entry:
            entry[row.risk_level] += row.count
        entry["total"] += row.count

    return jsonify({"success": True, "data": sorted(trend_map.values(), key=lambda x: x["month"])}), 200


# ─── Risk Score Timeline ──────────────────────────────────────────────────────

@dashboard_bp.route("/trends/risk-score", methods=["GET"])
@jwt_required()
def risk_score_timeline():
    user_id = int(get_jwt_identity())
    limit   = min(request.args.get("limit", 20, type=int), 100)

    records = (Prediction.query.filter_by(user_id=user_id)
               .order_by(Prediction.created_at.asc())
               .limit(limit).all())

    return jsonify({"success": True, "data": [
        {
            "date":       p.created_at.strftime("%Y-%m-%d"),
            "risk_score": p.risk_score,
            "risk_level": p.risk_level,
            "category":   p.diabetes_category,
        }
        for p in records
    ]}), 200