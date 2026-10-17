import logging
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Suppress joblib/loky CPU core warning
from datetime import timedelta
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import Config
from database.database import init_db
from routes.auth_routes import auth_bp
from routes.prediction_routes import prediction_bp
from routes.dashboard_routes import dashboard_bp

# IMPORTANT — force ML model import
from ml.services.predictor import predictor


# ───────────────────────────────────────────────────────────────
# Logging Configuration
# ───────────────────────────────────────────────────────────────

def configure_logging(app: Flask):

    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if app.debug else logging.INFO)

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.DEBUG if app.debug else logging.INFO)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ───────────────────────────────────────────────────────────────
# Application Factory
# ───────────────────────────────────────────────────────────────

def create_app(config_class=Config) -> Flask:

    app = Flask(__name__)
    app.config.from_object(config_class)
    # The database path contains Unicode characters which crashes print() on Windows cp1252

    # JWT Configuration
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=360)
    app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=365)
    app.config["JWT_ERROR_MESSAGE_KEY"] = "message"

    # ───────────────── Extensions ─────────────────

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    jwt = JWTManager(app)

    Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "60 per hour"],
        storage_uri="memory://"
    )

    # ───────────────── JWT Error Handlers ─────────────────

    @jwt.expired_token_loader
    def expired_token(jwt_header, jwt_payload):
        app.logger.warning("JWT Expired: %s", jwt_payload)
        message = "Token expired" if app.debug else "Token expired"
        return jsonify({"success": False, "message": message}), 401

    @jwt.invalid_token_loader
    def invalid_token(error_string):
        app.logger.warning("JWT Invalid: %s", error_string)
        message = f"Invalid token: {error_string}" if app.debug else "Invalid token"
        return jsonify({"success": False, "message": message}), 401

    @jwt.unauthorized_loader
    def missing_token(error_string):
        app.logger.warning("JWT Unauthorized: %s", error_string)
        message = f"Authorization token required: {error_string}" if app.debug else "Authorization token required"
        return jsonify({"success": False, "message": message}), 401

    @jwt.revoked_token_loader
    def revoked_token(jwt_header, jwt_payload):
        app.logger.warning("JWT Revoked: %s", jwt_payload)
        message = "Token revoked"
        return jsonify({"success": False, "message": message}), 401

    # ───────────────── Setup ─────────────────

    configure_logging(app)
    init_db(app)

    # ───────────────── ML Model Initialization ─────────────────

    try:
        predictor.load_model()
        app.logger.info("Diabetes prediction model loaded successfully")
    except Exception as e:
        app.logger.warning("Model failed to load during startup: %s", e)

    # ───────────────── Blueprints ─────────────────

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(prediction_bp, url_prefix="/api/prediction")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    # ───────────────── Security Headers ─────────────────

    @app.after_request
    def security_headers(response):

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"

        return response

    # ───────────────── Request Logging ─────────────────

    @app.after_request
    def log_request(response):

        response_data = ""
        if response.is_json:
            try:
                json_data = response.get_json()
                if json_data and isinstance(json_data, dict):
                    # Check top-level or data-level for risk info
                    risk_level = json_data.get("risk_level") or json_data.get("data", {}).get("risk_level")
                    
                    if risk_level:
                        response_data = f" | Risk: {risk_level}"
                        probability = json_data.get("probability") or json_data.get("data", {}).get("probability")
                        if probability is not None:
                            response_data += f" (Prob: {probability})"
            except Exception:
                pass

        app.logger.info(
            "%s %s %s%s",
            request.method,
            request.path,
            response.status_code,
            response_data
        )

        return response

    # ───────────────── Error Handlers ─────────────────

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"success": False, "message": "Bad request"}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "message": f"Route '{request.path}' not found"
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"success": False, "message": "Method not allowed"}), 405

    @app.errorhandler(429)
    def rate_limit(e):
        return jsonify({"success": False, "message": "Too many requests"}), 429

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("Internal server error: %s", e)
        return jsonify({"success": False, "message": "Internal server error"}), 500

    # ───────────────── Health Check ─────────────────

    @app.route("/api/health", methods=["GET"])
    def health():

        predictor_status = predictor.health_check()

        return jsonify({
            "success": True,
            "status": "healthy",
            "service": "AI Diabetes Risk Prediction API",
            "version": app.config.get("APP_VERSION", "1.0.0"),
            "model": predictor_status
        }), 200

    # ───────────────── Root Endpoint ─────────────────

    @app.route("/", methods=["GET"])
    def index():

        return jsonify({
            "success": True,
            "message": "AI Diabetes Risk Prediction System API is running"
        }), 200

    app.logger.info("Application initialized | environment=%s",
                    app.config.get("ENV", "production"))

    return app


# ───────────────── Entry Point ─────────────────

app = create_app()

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("\nAI Diabetes Risk Prediction API")
    print(f"Server  : http://127.0.0.1:{port}")
    print(f"Health  : http://127.0.0.1:{port}/api/health")
    print(f"Env     : {os.environ.get('FLASK_ENV', 'development')}\n")

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=False
    )