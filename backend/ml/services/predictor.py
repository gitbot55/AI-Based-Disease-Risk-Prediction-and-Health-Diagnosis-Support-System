import pickle
import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timezone
from config import Config

logger = logging.getLogger(__name__)


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class ModelLoadError(Exception):
    """Raised when the ML model bundle cannot be loaded or validated."""
    pass


class PredictionError(Exception):
    """Raised when inference fails due to malformed input or model fault."""
    pass


class FeatureValidationError(PredictionError):
    """Raised when input features fail range or type validation."""
    pass


# ─── Feature Schema ───────────────────────────────────────────────────────────

# Defines expected dtype, valid range, and whether zero is clinically valid
FEATURE_SCHEMA = {
    "HighBP":      {"min": 0, "max": 1, "zero_valid": True},
    "HighChol":    {"min": 0, "max": 1, "zero_valid": True},
    "BMI":         {"min": 10.0, "max": 80.0, "zero_valid": False},
    "Smoker":      {"min": 0, "max": 1, "zero_valid": True},
    "Fruits":      {"min": 0, "max": 1, "zero_valid": True},
    "GenHlth":     {"min": 1, "max": 5, "zero_valid": False},
    "MentHlth":    {"min": 0, "max": 30, "zero_valid": True},
    "Sex":         {"min": 0, "max": 1, "zero_valid": False},
    "Age":         {"min": 18, "max": 120, "zero_valid": False},
}

# Safe median fallbacks used when zero_valid=False and value is 0
FEATURE_FALLBACKS = {
    "BMI": 27.5,
    "GenHlth": 3,
    "Age": 40
}


# ─── Predictor ────────────────────────────────────────────────────────────────

class DiabetesPredictor:
    """
    Encapsulates the full inference lifecycle for the diabetes prediction model.

    Lifecycle
    ---------
    1. Lazy-load the pickled model bundle on first inference call.
    2. Validate and sanitise input features against the feature schema.
    3. Align the feature vector to the exact column order seen during training.
    4. Run predict_proba through the sklearn pipeline (scaler + classifier).
    5. Return a structured inference result dictionary.

    Design Decisions
    ----------------
    - Singleton pattern ensures the model is deserialised only once per process.
    - Feature validation raises FeatureValidationError with per-field messages
      so the API layer can return actionable 422 responses.
    - Zero-imputation uses population medians rather than model-training medians
      to keep the predictor stateless and dependency-free at runtime.
    - All public methods are safe to call concurrently (no mutable shared state
      beyond the cached pipeline object, which is read-only post-load).
    """

    _instance:        Optional["DiabetesPredictor"] = None
    _pipeline:        Any                           = None
    _feature_columns: List[str]                     = []
    _model_meta:      Dict[str, Any]                = {}
    _load_timestamp:  Optional[datetime]            = None

    def __new__(cls) -> "DiabetesPredictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ─── Model Lifecycle ──────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Deserialise the .pkl bundle from Config.ML_MODEL_PATH.
        """
        path = Config.ML_MODEL_PATH
        logger.info("Loading model from: %s", path)

        try:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
        except FileNotFoundError:
            logger.error("Model file not found: %s", path)
            raise ModelLoadError(
                f"Model file not found at '{path}'. "
                "Train the model first by running backend/ml/train/train_model.py."
            )
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.error("Model file corrupted: %s", exc)
            raise ModelLoadError(
                f"Model file at '{path}' is corrupted or unreadable: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error loading model: %s", exc)
            raise ModelLoadError(
                f"Unexpected error loading model from '{path}': {exc}"
            ) from exc

        self._validate_bundle(bundle, path)

        self._pipeline        = bundle["pipeline"]
        self._feature_columns = bundle["feature_columns"]
        
        self._model_meta = {
            "version":         bundle.get("model_version", "unknown"),
            "description":     bundle.get("description", ""),
            "path":            path,
            "feature_count":   len(self._feature_columns),
            "threshold":       bundle.get("threshold"),
            "metrics": {
                "cv_auc":      bundle.get("cv_auc"),
                "test_auc":    bundle.get("test_auc"),
                "accuracy":    bundle.get("accuracy"),
                "precision":   bundle.get("precision"),
                "recall":      bundle.get("recall"),
                "f1_score":    bundle.get("f1_score"),
            },
            "best_params":     bundle.get("best_params"),
            "training_time":   bundle.get("training_time"),
            "sklearn_version": bundle.get("sklearn_version"),
            "trained_at":      bundle.get("trained_at"),
        }
        self._load_timestamp = datetime.now(timezone.utc)

        logger.info(
            "Model loaded successfully | version=%s | features=%d | path=%s",
            self._model_meta["version"],
            self._model_meta["feature_count"],
            path,
        )

    def _validate_bundle(self, bundle: dict, path: str) -> None:
        """Assert required keys exist and validate their types."""
        required = {"pipeline", "feature_columns"}
        missing  = required - set(bundle.keys())
        if missing:
            raise ModelLoadError(
                f"Model bundle at '{path}' is missing required keys: {missing}"
            )

        if not hasattr(bundle["pipeline"], "predict_proba"):
            raise ModelLoadError(
                "Loaded pipeline does not expose predict_proba. "
                "Ensure the final estimator is a probabilistic classifier."
            )

        feat_cols = bundle["feature_columns"]
        if not isinstance(feat_cols, list) or not feat_cols or not all(isinstance(f, str) for f in feat_cols):
            raise ModelLoadError(
                "Bundle 'feature_columns' must be a non-empty list of strings."
            )

        if "threshold" in bundle and bundle["threshold"] is not None:
            if not isinstance(bundle["threshold"], (int, float)):
                raise ModelLoadError("Bundle 'threshold' must be numeric.")

        metric_keys = ["cv_auc", "test_auc", "accuracy", "precision", "recall", "f1_score", "training_time"]
        for key in metric_keys:
            if key in bundle and bundle[key] is not None:
                if not isinstance(bundle[key], (int, float)):
                    raise ModelLoadError(f"Bundle metric '{key}' must be numeric.")

        if "description" in bundle and bundle["description"] is not None:
            if not isinstance(bundle["description"], str):
                raise ModelLoadError("Bundle 'description' must be a string.")

        if "model_version" in bundle and bundle["model_version"] is not None:
            if not isinstance(bundle["model_version"], str):
                raise ModelLoadError("Bundle 'model_version' must be a string.")

    def _ensure_loaded(self) -> None:
        """Trigger lazy model load on first inference call."""
        if self._pipeline is None:
            logger.debug("Model not in memory — triggering lazy load.")
            self.load_model()

    def reload_model(self) -> None:
        """
        Force a hot-reload of the model from disk without restarting the server.
        Useful after retraining when the .pkl file has been updated in place.
        """
        logger.info("Initiating model hot-reload from disk.")
        self._pipeline        = None
        self._feature_columns = []
        self._model_meta      = {}
        self._load_timestamp  = None
        try:
            self.load_model()
            logger.info("Model hot-reload complete.")
        except Exception as exc:
            logger.error("Model hot-reload failed: %s", exc)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_info(self) -> dict:
        self._ensure_loaded()
        return {
            "version":          self._model_meta.get("version"),
            "description":      self._model_meta.get("description"),
            "threshold":        self._model_meta.get("threshold"),
            "metrics":          self._model_meta.get("metrics"),
            "feature_count":    self._model_meta.get("feature_count"),
            "feature_columns":  self._feature_columns,
            "sklearn_version":  self._model_meta.get("sklearn_version"),
            "trained_at":       self._model_meta.get("trained_at"),
            "loaded_at":        self._load_timestamp.isoformat() if self._load_timestamp else None,
        }

    # ─── Feature Validation ───────────────────────────────────────────────────

    def _validate_features(self, raw: dict) -> dict[str, list[str]]:
        """
        Check each expected feature for type and range violations.
        Also detects unknown features.
        
        Returns a dict of {field: [error messages]}.
        Empty dict means all expected features are valid.
        """
        errors: dict[str, list[str]] = {}

        for col in self._feature_columns:
            col_errors: list[str] = []
            schema = FEATURE_SCHEMA.get(col, {})
            raw_val = raw.get(col)

            if raw_val is None:
                col_errors.append(f"'{col}' is required but was not provided.")
                errors[col] = col_errors
                continue

            if isinstance(raw_val, bool):
                raw_val = 1.0 if raw_val else 0.0
            
            if isinstance(raw_val, str):
                stripped = raw_val.strip()
                if not stripped:
                    col_errors.append(f"'{col}' cannot be an empty or whitespace-only string.")
                    errors[col] = col_errors
                    continue
                raw_val = stripped

            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                col_errors.append(
                    f"'{col}' must be numeric, received '{raw_val}' ({type(raw_val).__name__})."
                )
                errors[col] = col_errors
                continue

            if np.isnan(value) or np.isinf(value):
                col_errors.append(f"'{col}' contains an invalid numeric value (NaN or Inf).")

            if schema:
                if value < schema["min"] or value > schema["max"]:
                    col_errors.append(
                        f"'{col}' value {value} is outside the valid range "
                        f"[{schema['min']}, {schema['max']}]."
                    )

            if col_errors:
                errors[col] = col_errors

        unknown_features = set(raw.keys()) - set(self._feature_columns)
        if unknown_features:
            logger.warning("Validation found unknown features: %s", sorted(unknown_features))

        return errors

    # ─── Feature Frame Construction ───────────────────────────────────────────

    def _build_feature_frame(self, raw: dict) -> pd.DataFrame:
        """
        Convert a raw input dict into a single-row DataFrame aligned to the
        exact column order the pipeline was trained on.
        """
        processed: dict[str, float] = {}

        for col in self._feature_columns:
            schema  = FEATURE_SCHEMA.get(col, {})
            raw_val = raw.get(col, 0.0)

            if isinstance(raw_val, bool):
                raw_val = 1.0 if raw_val else 0.0
            elif isinstance(raw_val, str):
                raw_val = raw_val.strip()
                if not raw_val:
                    raw_val = 0.0

            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                raise FeatureValidationError(
                    f"Cannot cast feature '{col}' value '{raw_val}' to float."
                )

            if value == 0.0 and schema.get("zero_valid") is False:
                fallback = FEATURE_FALLBACKS.get(col, 0.0)
                logger.debug(
                    "Zero-imputation | feature=%s | zero replaced with fallback=%.2f",
                    col, fallback
                )
                value = fallback

            if schema:
                value = float(np.clip(value, schema["min"], schema["max"]))

            processed[col] = value

        ignored = set(raw.keys()) - set(self._feature_columns)
        if ignored:
            logger.debug("Ignored unrecognised input keys: %s", sorted(ignored))

        return pd.DataFrame([processed], columns=self._feature_columns)

    # ─── Single Prediction ────────────────────────────────────────────────────

    def predict(self, input_data: dict, strict: bool = False) -> dict:
        """
        Run inference on a single patient record.
        """
        start_time = time.perf_counter()
        logger.info("Prediction started")
        
        self._ensure_loaded()

        validation_errors = self._validate_features(input_data)
        if validation_errors:
            logger.error("Validation failed: %s", validation_errors)
            raise FeatureValidationError(
                f"Feature validation failed for {len(validation_errors)} field(s): {validation_errors}"
            )

        feature_frame = self._build_feature_frame(input_data)

        try:
            proba_matrix = self._pipeline.predict_proba(feature_frame)
        except Exception as exc:
            logger.error("Prediction inference error: %s", exc)
            raise PredictionError(
                f"Model inference raised an unexpected error: {exc}"
            ) from exc

        if proba_matrix.ndim != 2 or proba_matrix.shape != (1, 2):
            err_msg = f"Unexpected predict_proba output shape {proba_matrix.shape}. Expected (1, 2)."
            logger.error(err_msg)
            raise PredictionError(err_msg)

        prob_non_diabetic: float = round(float(proba_matrix[0][0]), 6)
        prob_diabetic:     float = round(float(proba_matrix[0][1]), 6)
        
        threshold: float = self._model_meta.get("threshold")
        if threshold is None:
            threshold = Config.ML_PREDICTION_THRESHOLD
            
        prediction:        int   = int(prob_diabetic >= threshold)
        confidence:        float = round(float(np.max(proba_matrix[0])), 6)

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        result = {
            "prediction":            prediction,
            "probability":           prob_diabetic,
            "prob_non_diabetic":     prob_non_diabetic,
            "confidence":            confidence,
            "threshold":             threshold,
            "features_used":         feature_frame.iloc[0].to_dict(),
            "model_version":         self._model_meta.get("version", "unknown"),
            "predicted_at":          datetime.now(timezone.utc).isoformat(),
            "threshold_used":        threshold,
            "prediction_latency_ms": latency_ms,
        }

        metrics = self._model_meta.get("metrics", {})
        if metrics.get("accuracy") is not None:
            result["model_accuracy"] = metrics["accuracy"]
        if metrics.get("test_auc") is not None:
            result["model_auc"] = metrics["test_auc"]
        if metrics.get("precision") is not None:
            result["model_precision"] = metrics["precision"]
        if metrics.get("recall") is not None:
            result["model_recall"] = metrics["recall"]
        if metrics.get("f1_score") is not None:
            result["model_f1"] = metrics["f1_score"]

        logger.info(
            "Prediction completed | duration=%.2fms | prediction=%d | prob=%.4f | confidence=%.4f | threshold=%.2f",
            latency_ms, prediction, prob_diabetic, confidence, threshold,
        )

        return result

    # ─── Batch Prediction ─────────────────────────────────────────────────────

    def batch_predict(self, records: list[dict], strict: bool = False) -> dict:
        """
        Run inference over a list of patient records.
        """
        if not records:
            raise PredictionError("batch_predict received an empty list. Provide at least one record.")

        self._ensure_loaded()

        results:     list[dict] = []
        probs_ok:    list[float] = []
        diabetic_n:  int = 0

        for idx, record in enumerate(records):
            try:
                result          = self.predict(record, strict=strict)
                result["index"] = idx
                results.append(result)
                probs_ok.append(result["probability"])
                if result["prediction"] == 1:
                    diabetic_n += 1
            except (PredictionError, FeatureValidationError) as exc:
                logger.warning("Batch record %d failed: %s", idx, exc)
                results.append({"index": idx, "error": str(exc)})

        succeeded = sum(1 for r in results if "error" not in r)
        failed    = len(results) - succeeded
        avg_prob  = round(float(np.mean(probs_ok)), 4) if probs_ok else None

        logger.info(
            "Batch statistics | total=%d | succeeded=%d | failed=%d | diabetic=%d",
            len(records), succeeded, failed, diabetic_n,
        )

        return {
            "results": results,
            "summary": {
                "total":                len(records),
                "succeeded":            succeeded,
                "failed":               failed,
                "diabetic_count":       diabetic_n,
                "non_diabetic_count":   succeeded - diabetic_n,
                "average_probability":  avg_prob,
            },
        }

    # ─── Diagnostics ──────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Return a status snapshot of the predictor for the /api/health route.
        """
        logger.info("Executing health check.")
        
        status = {
            "model_loaded":      self.is_loaded,
            "model_version":     None,
            "feature_count":     None,
            "loaded_at":         None,
            "threshold":         None,
            "metrics":           None,
            "model_age_days":    None,
            "inference_latency_ms": None,
            "inference_test":    "not_run",
            "healthy":           False,
        }

        try:
            self._ensure_loaded()
            status["model_version"] = self._model_meta.get("version")
            status["feature_count"] = len(self._feature_columns)
            status["threshold"]     = self._model_meta.get("threshold")
            status["metrics"]       = self._model_meta.get("metrics")
            
            if self._load_timestamp:
                status["loaded_at"] = self._load_timestamp.isoformat()

            trained_at_str = self._model_meta.get("trained_at")
            if trained_at_str:
                try:
                    trained_at = datetime.fromisoformat(trained_at_str)
                    age_td = datetime.now(timezone.utc) - trained_at
                    status["model_age_days"] = age_td.days
                except (ValueError, TypeError):
                    pass

            # Synthetic mid-range record for smoke-test inference
            test_record = {
                 "HighBP": 0,
                 "HighChol": 0,
                 "BMI": 27.5,
                 "Smoker": 0,
                 "Fruits": 1,
                 "GenHlth": 2,
                 "MentHlth": 1,
                 "Sex": 1,
                 "Age": 35,
            }
            
            test_start = time.perf_counter()
            test_result = self.predict(test_record)
            test_end = time.perf_counter()
            
            status["inference_latency_ms"] = round((test_end - test_start) * 1000, 2)
            status["inference_test"] = "passed"
            status["healthy"]        = True

            logger.info(
                "Health check passed | test_prob=%.4f | latency=%.2fms",
                test_result["probability"], status["inference_latency_ms"]
            )

        except (ModelLoadError, PredictionError) as exc:
            status["inference_test"] = f"failed: {exc}"
            status["healthy"]        = False
            logger.error("Predictor health check failed: %s", exc)

        return status


# ─── Module-level singleton ───────────────────────────────────────────────────
# Import this wherever prediction is needed:
#   from ml.services.predictor import predictor
predictor = DiabetesPredictor()

# Load model immediately when module imports
try:
    predictor.load_model()
    logger.info("Diabetes prediction model loaded during startup.")
except Exception as e:
    logger.warning("Model could not load at startup: %s", e)