from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)
import logging

from database.database import db
from database.models import User, TokenBlocklist
from auth.auth_utils import (
    hash_password,
    verify_password,
    is_password_strong,
    sanitize_string,
    sanitize_email,
)

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)


def _generate_tokens(user_id: int):
    access_token = create_access_token(identity=str(user_id))
    refresh_token = create_refresh_token(identity=str(user_id))
    return access_token, refresh_token

def _user_payload(user: User):
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
    }


def _revoke_token(jti, token_type, user_id):
    db.session.add(
        TokenBlocklist(
            jti=jti,
            token_type=token_type,
            user_id=user_id,
        )
    )
    db.session.commit()


@auth_bp.route("/register", methods=["POST"])
def register():

    body = request.get_json(silent=True) or {}

    username = sanitize_string(body.get("username", "")).lower()
    email = sanitize_email(body.get("email", ""))
    password = body.get("password", "")
    full_name = sanitize_string(body.get("full_name", ""))
    age = body.get("age")
    gender = body.get("gender")

    if not username or not email or not password:
        return jsonify(
            {"success": False, "message": "Username, email and password are required"}
        ), 400

    error = is_password_strong(password)
    if error:
        return jsonify({"success": False, "message": error}), 422

    if User.query.filter_by(username=username).first():
        return jsonify({"success": False, "message": "Username already exists"}), 409

    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email already registered"}), 409

    try:

        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            full_name=full_name or None,
            age=int(age) if age else None,
            gender=gender or None,
        )

        db.session.add(user)
        db.session.commit()

        logger.info("User registered | id=%s", user.id)

    except Exception as e:

        db.session.rollback()
        logger.exception("Registration failed: %s", e)

        return jsonify(
            {"success": False, "message": "Registration failed"}
        ), 500

    access_token, refresh_token = _generate_tokens(user.id)

    return jsonify(
        {
            "success": True,
            "message": "Account created successfully",
            "user": _user_payload(user),
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    ), 201


@auth_bp.route("/login", methods=["POST"])
def login():

    body = request.get_json(silent=True) or {}

    identifier = sanitize_string(body.get("identifier", "")).lower()
    password = body.get("password", "")

    if not identifier or not password:
        return jsonify(
            {"success": False, "message": "Identifier and password are required"}
        ), 400

    user = (
        User.query.filter_by(email=identifier).first()
        or User.query.filter_by(username=identifier).first()
    )

    dummy_hash = "$2b$12$invalidhashXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

    if not verify_password(password, user.password_hash if user else dummy_hash):

        logger.warning("Failed login | identifier=%s", identifier)

        return jsonify(
            {"success": False, "message": "Invalid credentials"}
        ), 401

    if not user.is_active:

        return jsonify(
            {"success": False, "message": "Account is deactivated"}
        ), 403

    try:

        user.record_login()
        db.session.commit()

    except Exception:
        db.session.rollback()

    access_token, refresh_token = _generate_tokens(user.id)

    logger.info("User logged in | id=%s", user.id)

    return jsonify(
        {
            "success": True,
            "message": "Login successful",
            "user": _user_payload(user),
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    ), 200


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():

    user_id = int(get_jwt_identity())
    jti = get_jwt()["jti"]

    user = User.query.get(user_id)

    if not user or not user.is_active:
        return jsonify(
            {"success": False, "message": "User not found"}
        ), 401

    try:

        _revoke_token(jti, "refresh", user_id)

    except Exception as e:

        db.session.rollback()
        logger.exception("Token refresh failed: %s", e)

        return jsonify(
            {"success": False, "message": "Token refresh failed"}
        ), 500

    access_token, refresh_token = _generate_tokens(user_id)

    return jsonify(
        {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    ), 200


@auth_bp.route("/logout", methods=["DELETE"])
@jwt_required()
def logout():

    user_id = int(get_jwt_identity())
    jti = get_jwt()["jti"]

    try:

        _revoke_token(jti, "access", user_id)

        logger.info("User logout | id=%s", user_id)

    except Exception as e:

        db.session.rollback()
        logger.exception("Logout failed: %s", e)

        return jsonify(
            {"success": False, "message": "Logout failed"}
        ), 500

    return jsonify(
        {"success": True, "message": "Logged out successfully"}
    ), 200


@auth_bp.route("/verify", methods=["GET"])
@jwt_required()
def verify():

    user = User.query.get(int(get_jwt_identity()))

    if not user or not user.is_active:

        return jsonify(
            {"success": False, "message": "User not found"}
        ), 401

    return jsonify(
        {"success": True, "user": _user_payload(user)}
    ), 200