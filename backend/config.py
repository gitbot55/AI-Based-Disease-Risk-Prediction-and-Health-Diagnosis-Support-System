import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _get_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("true", "1", "yes")

def _get_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


# ─── Base Config ──────────────────────────────────────────────────────────────

class BaseConfig:

    # App
    APP_NAME    = os.environ.get("APP_NAME",    "AI Diabetes Risk Prediction System")
    APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
    ENV         = os.environ.get("FLASK_ENV",   "production")
    SECRET_KEY  = os.environ.get("SECRET_KEY",  "dev-secret-key-do-not-use-in-prod")

    # Database
    DATABASE_DIR  = os.path.join(BASE_DIR, "database")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "diabetes_system.db")
    DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)
    SQLALCHEMY_DATABASE_URI        = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO                = False

    # JWT
    JWT_SECRET_KEY              = os.environ.get("JWT_SECRET_KEY", "dev-jwt-secret-key-do-not-use-in-prod")
    JWT_ACCESS_TOKEN_EXPIRES    = timedelta(hours=_get_int("JWT_ACCESS_EXPIRES_HOURS", 8760))
    JWT_REFRESH_TOKEN_EXPIRES   = timedelta(days=_get_int("JWT_REFRESH_EXPIRES_DAYS",  365))

    # CORS
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

    # Rate Limiting
    RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
    RATELIMIT_DEFAULT     = os.environ.get("RATELIMIT_DEFAULT", "200 per day;60 per hour")

    # ML Model
    ML_MODEL_DIR      = os.path.join(BASE_DIR, "ml", "model")
    ML_MODEL_FILENAME = os.environ.get("ML_MODEL_FILENAME", "diabetes_model.pkl")
    ML_MODEL_PATH     = os.path.join(ML_MODEL_DIR, ML_MODEL_FILENAME)
    ML_PREDICTION_THRESHOLD = float(os.environ.get("ML_PREDICTION_THRESHOLD", 0.5))
    ML_RISK_BANDS = {
        "low":    (0.0,  0.35),
        "medium": (0.35, 0.65),
        "high":   (0.65, 1.01),
    }

    # Clinical Thresholds
    GLUCOSE_PREDIABETIC_MIN = _get_int("GLUCOSE_PREDIABETIC_MIN", 100)
    GLUCOSE_DIABETIC_MIN    = _get_int("GLUCOSE_DIABETIC_MIN",    126)

    # Centralized Age Groups
    AGE_GROUPS = {
        "Child":       (0, 17),
        "Young Adult": (18, 29),
        "Adult":       (30, 44),
        "Middle-aged": (45, 59),
        "Senior":      (60, 200),
    }
    
    # Centralized Model & Recommendation Confidence / Probability Thresholds
    CONFIDENCE_LOW = 0.60
    CONFIDENCE_HIGH = 0.80
    CONFIDENCE_CRITICAL = 0.85
    PROBABILITY_HIGH = 0.70
    PROBABILITY_CRITICAL = 0.85

    # Password Policy
    PASSWORD_MIN_LENGTH        = _get_int("PASSWORD_MIN_LENGTH", 8)
    PASSWORD_REQUIRE_UPPERCASE = _get_bool("PASSWORD_REQUIRE_UPPERCASE", True)
    PASSWORD_REQUIRE_DIGIT     = _get_bool("PASSWORD_REQUIRE_DIGIT",     True)
    PASSWORD_REQUIRE_SPECIAL   = _get_bool("PASSWORD_REQUIRE_SPECIAL",   False)
    BCRYPT_LOG_ROUNDS          = _get_int("BCRYPT_LOG_ROUNDS", 12)

    # Feature Columns (must match training order)
    MODEL_FEATURE_COLUMNS = [
       "HighBP", "HighChol", "BMI",
         "Smoker", "Fruits", "GenHlth",
           "MentHlth", "Sex", "Age",
    ]

    # Logging
    LOG_DIR    = os.path.join(BASE_DIR, "logs")
    LOG_PATH   = os.path.join(LOG_DIR, "app.log")


# ─── Environment Configs ──────────────────────────────────────────────────────

class DevelopmentConfig(BaseConfig):
    ENV             = "development"
    DEBUG           = True
    SQLALCHEMY_ECHO = False
    BCRYPT_LOG_ROUNDS = 4
    RATELIMIT_DEFAULT = "10000 per day;1000 per hour"
    ALLOWED_ORIGINS   = ["http://localhost:3000", "http://127.0.0.1:5000"]


class TestingConfig(BaseConfig):
    ENV                        = "testing"
    TESTING                    = True
    SQLALCHEMY_DATABASE_URI    = "sqlite:///:memory:"
    BCRYPT_LOG_ROUNDS          = 4
    RATELIMIT_DEFAULT          = "100000 per day"
    JWT_ACCESS_TOKEN_EXPIRES   = timedelta(minutes=5)


class ProductionConfig(BaseConfig):
    ENV   = "production"
    DEBUG = False

    SECRET_KEY = os.environ.get("SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for Flask application in production environment.")

    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
    if not JWT_SECRET_KEY:
        raise ValueError("No JWT_SECRET_KEY set for Flask application in production environment.")

    BCRYPT_LOG_ROUNDS = _get_int("BCRYPT_LOG_ROUNDS", 13)
    RATELIMIT_DEFAULT = os.environ.get("RATELIMIT_DEFAULT", "500 per day;100 per hour")
    ALLOWED_ORIGINS   = os.environ.get("ALLOWED_ORIGINS", "").split(",")


# ─── Config Selector ──────────────────────────────────────────────────────────

_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing":     TestingConfig,
    "production":  ProductionConfig,
}

Config = _CONFIG_MAP.get(
    os.environ.get("FLASK_ENV", "production").lower(),
    ProductionConfig
)