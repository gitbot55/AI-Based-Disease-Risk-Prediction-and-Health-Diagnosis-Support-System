import logging
from typing import Dict, Any, Tuple

from database.database import db, safe_commit
from database.models import User, Prediction
from config import Config
from ml.services.symptom_mapper import symptom_mapper
from ml.services.predictor import predictor
from ml.services.probability_calibrator import probability_calibrator
from ml.services.risk_engine import risk_engine

logger = logging.getLogger(__name__)
import time

class PredictionService:
    def __init__(self):
        logger.info("Predictor uses 9 ML features. Recommendation Engine uses 12 derived features. 3 derived features are intentionally excluded from ML inference.")
    def process_prediction(self, user: User, body: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Orchestrates the complete V4 prediction pipeline.
        
        Args:
            user: The authenticated User object.
            body: The validated JSON request body.
            metadata: A dictionary containing trace IDs and telemetry context.
            
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        user_id = user.id
        request_id = metadata.get("request_id", "unknown")

        start_time = time.perf_counter()
        
        # 1. Feature Processing (via Symptom Mapper)
        try:
            # Allow JSON payload to override sex, otherwise fallback to database profile
            if "sex" not in body:
                body["sex"] = 1 if getattr(user, "gender", "Male") == "Male" else 0
            mapped_features = symptom_mapper.map_to_features(body)
            logger.info("Symptom mapping completed | request_id=%s | user_id=%s", request_id, user_id)
        except Exception as exc:
            logger.exception("Symptom mapping failed: %s", exc)
            return self._build_error_response("VALIDATION_ERROR", "InvalidData", "symptoms", "Invalid symptom mapping. Ensure data format is correct.", 422, metadata)

        # 2. Predictor
        supported_features = {}
        unsupported_features = {}
        for k, v in mapped_features.items():
            if k in Config.MODEL_FEATURE_COLUMNS:
                supported_features[k] = v
            else:
                unsupported_features[k] = v
                
        if unsupported_features:
            logger.debug("Unsupported features mapped but ignored for prediction | request_id=%s | features=%s", request_id, list(unsupported_features.keys()))

        t_pred = time.perf_counter()
        try:
            predictor_result = predictor.predict(supported_features)
            logger.info("Prediction completed | request_id=%s | user_id=%s", request_id, user_id)
        except Exception as exc:
            logger.exception("Prediction inference failed: %s", exc)
            return self._build_error_response("SERVER_ERROR", "InferenceFailed", "model", "Model prediction failed during evaluation.", 500, metadata)
        predict_ms = (time.perf_counter() - t_pred) * 1000

        # 3. Probability Calibrator
        t_calib = time.perf_counter()
        try:
            calib_res = probability_calibrator.calibrate(
                probability=predictor_result["probability"],
                features=predictor_result["features_used"]
            )
            calibrated_probability = calib_res.calibrated_probability
            logger.info("Calibration completed | request_id=%s | user_id=%s", request_id, user_id)
        except Exception as exc:
            logger.warning("Calibration failed (falling back to raw) | request_id=%s: %s", request_id, exc)
            calibrated_probability = predictor_result["probability"]
        calib_ms = (time.perf_counter() - t_calib) * 1000

        # 4. Risk Engine (receives calibrated probability directly)
        t_risk = time.perf_counter()
        try:
            assessment = risk_engine.evaluate(
                predictor_result=predictor_result,
                input_features=mapped_features,
                calibrated_probability=calibrated_probability
            )
            logger.info("Risk evaluation completed | request_id=%s | risk_level=%s", request_id, assessment.risk_level)
        except Exception as exc:
            logger.exception("Risk engine failed: %s", exc)
            return self._build_error_response("SERVER_ERROR", "RiskEvalFailed", "risk_engine", "Risk assessment engine failed.", 500, metadata)
        risk_engine_total_ms = (time.perf_counter() - t_risk) * 1000

        logger.info(
            "Diagnostic | Durations=%s | SymptomScore=%.2f | Features=%s | RawProb=%.4f | CalibProb=%.4f | RiskScore=%.4f | Category=%s",
            body.get("symptoms", {}),
            mapped_features.get("symptom_score", 0.0),
            supported_features,
            predictor_result["probability"],
            calibrated_probability,
            assessment.risk_score,
            assessment.risk_level
        )

        # 5. Save Prediction to DB
        t_db = time.perf_counter()
        try:
            prediction = Prediction.from_assessment(
                user_id=user_id,
                assessment=assessment.to_dict(),
                features=mapped_features # Save all features (supported + unsupported) to the DB
            )
            prediction.gender = user.gender

            db.session.add(prediction)
            
            if safe_commit():
                logger.info("Database save completed | request_id=%s | prediction_id=%s", request_id, prediction.id)
            else:
                raise RuntimeError("safe_commit failed")
        except Exception as exc:
            # safe_commit already rolls back, but we catch exceptions in from_assessment or add
            if db.session.is_active:
                db.session.rollback()
            logger.exception("Database transaction failed | request_id=%s: %s", request_id, exc)
            return self._build_error_response("SERVER_ERROR", "DatabaseError", "db", "Failed to securely save the prediction.", 500, metadata)
        db_ms = (time.perf_counter() - t_db) * 1000
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate sub-components from metadata
        rec_ms = assessment.metadata.get("recommendation_plan", {}).get("metadata", {}).get("generation_time_ms", 0)
        shap_ms = assessment.metadata.get("shap_explanation", {}).get("computation_time_ms", 0)
        core_risk_ms = max(0, risk_engine_total_ms - rec_ms - shap_ms)

        logger.info(
            "Performance Profile | request_id=%s\n"
            "Model prediction: %.2f ms\n"
            "Calibration: %.2f ms\n"
            "Risk Engine: %.2f ms\n"
            "Recommendation Engine: %.2f ms\n"
            "SHAP: %.2f ms\n"
            "Database: %.2f ms\n"
            "Total: %.2f ms",
            request_id, predict_ms, calib_ms, core_risk_ms, rec_ms, shap_ms, db_ms, total_ms
        )

        metadata["prediction_id"] = prediction.id
        metadata["model_version"] = predictor_result.get("model_version", "unknown")
        metadata["prediction_version"] = "v4.1.0"
        metadata["risk_engine_version"] = "v4.1.0"
        metadata["recommendation_version"] = "v4.1.0"
        metadata["explanation_available"] = True
        metadata["calibration_applied"] = calibrated_probability != predictor_result["probability"]
        metadata["prediction_time"] = metadata["timestamp"]
        if unsupported_features:
            metadata["ignored_features"] = list(unsupported_features.keys())

        # 6. Build the Complete API Response
        response_data = assessment.to_dict()
        response_data["prediction_id"] = prediction.id
        
        # Align all probabilities to the dynamic hybrid risk score as requested
        response_data["probability"] = assessment.risk_score
        response_data["calibrated_probability"] = assessment.risk_score
        response_data["raw_probability"] = assessment.risk_score

        return {
            "success": True,
            "message": "Prediction completed successfully.",
            "data": response_data,
            "metadata": metadata
        }, 201

    def _build_error_response(self, code: str, type_: str, field: str, message: str, status: int, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Formats errors for consistency across the API."""
        return {
            "success": False,
            "error": {
                "code": code,
                "type": type_,
                "field": field
            },
            "message": message,
            "metadata": metadata
        }, status

prediction_service = PredictionService()

