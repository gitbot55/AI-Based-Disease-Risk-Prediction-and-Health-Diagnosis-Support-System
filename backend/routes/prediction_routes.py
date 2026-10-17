from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
import time
import uuid
from datetime import datetime, timezone

from database.database import db
from database.models import User, Prediction

from ml.services.predictor import predictor
from ml.services.prediction_service import prediction_service

logger = logging.getLogger(__name__)
prediction_bp = Blueprint("prediction", __name__)

API_VERSION = "v4.1.0"


def _build_error(code: str, type_: str, field: str, message: str, status: int, metadata: dict = None):
    """Reusable helper for structured error responses."""
    meta = metadata or {}
    meta["api_version"] = API_VERSION
    meta["timestamp"] = datetime.now(timezone.utc).isoformat()
    return jsonify({
        "success": False,
        "error": {
            "code": code,
            "type": type_,
            "field": field
        },
        "message": message,
        "metadata": meta
    }), status


def _get_user(user_id: int):
    return User.query.filter_by(id=user_id, is_active=True).first()


@prediction_bp.route("/submit", methods=["POST"])
@jwt_required()
def submit_prediction():
    start_time = time.perf_counter()
    request_id = uuid.uuid4().hex
    
    metadata = {
        "request_id": request_id,
        "correlation_id": request_id,
        "api_version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_start": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info("Request received | request_id=%s | endpoint=/submit", request_id)

    user_id = int(get_jwt_identity())
    user = _get_user(user_id)
    if not user:
        logger.warning("User authentication failed | request_id=%s | user_id=%s", request_id, user_id)
        return _build_error("AUTH_ERROR", "UserNotFound", "Authorization", "User not found or inactive.", 401, metadata)

    logger.info("User authenticated | request_id=%s | user_id=%s", request_id, user.id)

    body = request.get_json(silent=True)
    if body is None:
        return _build_error("VALIDATION_ERROR", "EmptyPayload", "body", "Invalid or missing JSON payload.", 400, metadata)

    if "symptoms" not in body:
        return _build_error("VALIDATION_ERROR", "MissingField", "symptoms", "Symptoms data required.", 400, metadata)

    if "age" not in body:
        return _build_error("VALIDATION_ERROR", "MissingField", "age", "Age is required.", 400, metadata)

    age = body.get("age")
    if not isinstance(age, (int, float)) or age < 0 or age > 120:
        return _build_error("VALIDATION_ERROR", "InvalidRange", "age", "Age must be a valid number between 0 and 120.", 422, metadata)

    logger.info("Validation passed | request_id=%s", request_id)

    # Execute service logic
    response_data, status_code = prediction_service.process_prediction(user, body, metadata)
    
    # Finalize metadata tracing
    if "metadata" in response_data:
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        response_data["metadata"]["processing_time_ms"] = processing_time
        response_data["metadata"]["processing_time"] = f"{processing_time}ms"
        response_data["metadata"]["server_time"] = datetime.now(timezone.utc).isoformat()
        response_data["metadata"]["processing_end"] = datetime.now(timezone.utc).isoformat()

    logger.info("Response generation completed | request_id=%s | status=%s", request_id, status_code)
    return jsonify(response_data), status_code


@prediction_bp.route("/<int:prediction_id>", methods=["GET"])
@jwt_required()
def get_prediction(prediction_id):
    start_time = time.perf_counter()
    request_id = uuid.uuid4().hex
    
    metadata = {
        "request_id": request_id,
        "api_version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    user_id = int(get_jwt_identity())
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=user_id).first()

    if not prediction:
        return _build_error("NOT_FOUND", "ResourceNotFound", "prediction_id", "Prediction not found.", 404, metadata)

    logger.info("Prediction fetched | request_id=%s | user_id=%s | prediction_id=%s", request_id, user_id, prediction_id)

    metadata["processing_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
    
    return jsonify({
        "success": True,
        "message": "Prediction retrieved successfully.",
        "data": prediction.to_dict(),
        "metadata": metadata
    }), 200


@prediction_bp.route("/health", methods=["GET"])
def model_health():
    start_time = time.perf_counter()
    request_id = uuid.uuid4().hex
    
    metadata = {
        "request_id": request_id,
        "api_version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    try:
        health = predictor.health_check()
        logger.info("Health endpoint evaluated | request_id=%s | healthy=%s", request_id, health.get("healthy"))
    except Exception as exc:
        logger.exception("Unexpected exception in health endpoint: %s", exc)
        return _build_error("SERVER_ERROR", "InternalError", "health", "Health check failed due to an internal exception.", 500, metadata)

    status_code = 200 if health.get("healthy") else 503
    
    metadata["processing_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
    metadata["processing_status"] = "completed"

    response_payload = {
        "success": health.get("healthy", False),
        "message": "System is operating normally." if status_code == 200 else "System is experiencing degraded performance.",
        "data": {
            **health,
            "api_version": API_VERSION,
            "model_version": health.get("model_version", "unknown"),
            "loaded_model": health.get("model_loaded", False),
            "prediction_threshold": health.get("threshold", 0.5),
            "cache_status": "connected",
            "database_connectivity": "connected"
        },
        "metadata": metadata
    }

    return jsonify(response_payload), status_code