from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple
import logging
import time

from config import Config

try:
    from .probability_calibrator import probability_calibrator
    from .recommendation_engine import recommendation_engine
    from .predictor import predictor
    from .shap_explainer import shap_explainer
except ImportError:  # pragma: no cover - supports direct module execution
    from probability_calibrator import probability_calibrator
    from recommendation_engine import recommendation_engine
    from predictor import predictor
    from shap_explainer import shap_explainer

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    prediction:        int
    probability:       float
    risk_score:        float
    risk_level:        str
    diabetes_category: str
    prediction_label:  str
    confidence:        float
    flags:             List[str] = field(default_factory=list)
    recommendations:   List[str] = field(default_factory=list)
    metadata:          Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prediction":         self.prediction,
            "probability":        self.probability,
            "risk_score":         self.risk_score,
            "risk_level":         self.risk_level,
            "diabetes_category":  self.diabetes_category,
            "prediction_label":   self.prediction_label,
            "confidence":         self.confidence,
            "flags":              self.flags,
            "recommendations":    self.recommendations,
            "metadata":           self.metadata,
        }


@dataclass
class ClinicalSeverity:
    score:                float
    active_factors:       int
    severity_points:      float
    factors:              List[str] = field(default_factory=list)
    severity_label:       str = ""
    severity_explanation: str = ""
    severity_percentage:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "score":                self.score,
            "active_factors":       self.active_factors,
            "severity_points":      self.severity_points,
            "factors":              self.factors,
            "severity_label":       self.severity_label,
            "current_clinical_status": self.severity_label,
            "severity_explanation": self.severity_explanation,
            "status_explanation":   self.severity_explanation,
            "severity_percentage":  self.severity_percentage,
        }


class ClinicalThresholds:
    # Consider moving these dynamic thresholds to Config.py in the future
    # to allow medical teams to update them without code changes.
    BMI_UNDERWEIGHT:    float = 18.5
    BMI_NORMAL_MAX:     float = 24.9
    BMI_OVERWEIGHT:     float = 29.9
    BMI_OBESE:          float = 30.0
    BMI_SEVERELY_OBESE: float = 35.0

    GENHLTH_FAIR: int = 4
    GENHLTH_POOR: int = 5

    MENTHLTH_ELEVATED_DAYS: int = 14
    PHYSHLTH_ELEVATED_DAYS: int = 14

    AGE_RISK_THRESHOLD_1: int = 35
    AGE_RISK_THRESHOLD_2: int = 45
    AGE_RISK_THRESHOLD_3: int = 60

    CLINICAL_SEVERITY_MAX: float = 14.0


class DiabetesRiskEngine:

    T = ClinicalThresholds

    def evaluate(
        self,
        predictor_result: dict,
        input_features:   dict,
        calibrated_probability: Optional[float] = None
    ) -> RiskAssessment:
        start_time = time.perf_counter()
        logger.info("Evaluation started")

        try:
            probability   = predictor_result["probability"]
            prediction    = predictor_result["prediction"]
            confidence    = predictor_result["confidence"]
            model_version = predictor_result.get("model_version", "unknown")

            features = self._extract_features(input_features)

            if calibrated_probability is not None:
                calibrated_score = calibrated_probability
                logger.debug("Using provided calibrated probability: %.4f", calibrated_score)
            else:
                calibrated_score = self._calibrate_score(probability, features)
                logger.debug("Calibration result: %.4f", calibrated_score)

            clinical_severity = self._calculate_clinical_severity(features)
            logger.debug("Clinical severity evaluated: label=%s, score=%.4f", clinical_severity.severity_label, clinical_severity.score)

            flags_data = self._generate_structured_flags(features, calibrated_score)
            flags = [f["message"] for f in flags_data]

            dynamic_components = self._dynamic_risk_score(
                raw_probability      = probability,
                calibrated_score     = calibrated_score,
                confidence           = confidence,
                clinical_severity    = clinical_severity,
                flags                = flags,
                features             = features,
            )
            dynamic_score = dynamic_components["total_score"]
            logger.debug("Dynamic score calculated: %.4f", dynamic_score)

            risk_level = self._classify_risk_level(dynamic_score)
            diabetes_category = self._classify_diabetes_category(prediction, dynamic_score)
            logger.info("Risk category determined: category=%s, level=%s", diabetes_category, risk_level)

            prediction_label = self._build_prediction_label(diabetes_category, risk_level)
            risk_interpretation = self._build_risk_interpretation(diabetes_category, risk_level, flags_data)
            confidence_interpretation = self._classify_confidence(confidence, features)

            logger.debug("Recommendation generation started")
            recommendation_plan = self._build_recommendation_plan(
                risk_level, diabetes_category, dynamic_score, calibrated_score, confidence, input_features, flags
            )
            recommendations = self._recommendation_messages(recommendation_plan)

            logger.debug("SHAP generation started")
            shap_context = self._build_shap_context(predictor_result, input_features)

            metadata = self._build_metadata(
                f=features,
                raw_prob=probability,
                dynamic_score=dynamic_score,
                model_version=model_version,
                prediction=prediction,
                calibrated_score=calibrated_score,
                clinical_severity=clinical_severity,
                recommendation_plan=recommendation_plan,
                shap_context=shap_context,
                dynamic_components=dynamic_components,
                flags_data=flags_data,
                risk_interpretation=risk_interpretation,
                confidence_interpretation=confidence_interpretation,
            )

            assessment = RiskAssessment(
                prediction        = prediction,
                probability       = round(probability, 4),
                risk_score        = round(dynamic_score, 4),
                risk_level        = risk_level,
                diabetes_category = diabetes_category,
                prediction_label  = prediction_label,
                confidence        = round(confidence, 4),
                flags             = flags,
                recommendations   = recommendations,
                metadata          = metadata,
            )

            duration = (time.perf_counter() - start_time) * 1000
            logger.info("Evaluation completed successfully in %.2f ms", duration)

            return assessment

        except Exception as exc:
            logger.exception("Unexpected exception during risk evaluation: %s", exc)
            raise

    def _extract_features(self, raw: dict) -> dict:
        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        return {
            "high_bp":       _f("HighBP"),
            "high_chol":     _f("HighChol"),
            "bmi":           _f("BMI"),
            "smoker":        _f("Smoker"),
            "phys_activity": _f("PhysActivity"),
            "fruits":        _f("Fruits"),
            "veggies":       _f("Veggies"),
            "gen_hlth":      _f("GenHlth"),
            "ment_hlth":     _f("MentHlth"),
            "phys_hlth":     _f("PhysHlth"),
            "sex":           _f("Sex"),
            "age":           _f("Age"),
            "symptom_score": _f("symptom_score"),
        }

    def _to_model_features(self, f: dict) -> dict:
        return {
            "HighBP":       f["high_bp"],
            "HighChol":     f["high_chol"],
            "BMI":          f["bmi"],
            "Smoker":       f["smoker"],
            "PhysActivity": f["phys_activity"],
            "Fruits":       f["fruits"],
            "Veggies":      f["veggies"],
            "GenHlth":      f["gen_hlth"],
            "MentHlth":     f["ment_hlth"],
            "PhysHlth":     f["phys_hlth"],
            "Sex":          f["sex"],
            "Age":          f["age"],
        }

    def _calibrate_score(self, base_prob: float, f: dict) -> float:
        try:
            calibration = probability_calibrator.calibrate(
                probability = base_prob,
                features    = self._to_model_features(f),
            )
            return calibration.calibrated_probability
        except Exception as exc:
            logger.warning("Probability calibration failed, returning raw probability: %s", exc)
            return base_prob

    def _calculate_clinical_severity(self, f: dict) -> ClinicalSeverity:
        factors: List[str] = []
        points = 0.0

        def add(name: str, value: float) -> None:
            nonlocal points
            points += value
            factors.append(name)

        if f["bmi"] >= self.T.BMI_SEVERELY_OBESE:
            add("BMI indicates severe obesity", 3.0)
        elif f["bmi"] >= self.T.BMI_OBESE:
            add("BMI indicates obesity", 2.0)
        elif f["bmi"] >= self.T.BMI_OVERWEIGHT:
            add("BMI indicates overweight", 1.0)

        if f["high_bp"] >= 1:
            add("History of elevated blood pressure", 2.0)
        if f["high_chol"] >= 1:
            add("History of elevated cholesterol", 1.5)
        if f["smoker"] >= 1:
            add("Current smoking habit", 1.2)
        if f["phys_activity"] < 1:
            add("Sedentary lifestyle (low physical activity)", 1.2)
        if f["fruits"] < 1 and f["veggies"] < 1:
            add("Diet low in fruits and vegetables", 0.8)

        if f["gen_hlth"] >= self.T.GENHLTH_POOR:
            add("Self-reported poor general health", 2.5)
        elif f["gen_hlth"] >= self.T.GENHLTH_FAIR:
            add("Self-reported fair general health", 1.5)

        if f["ment_hlth"] >= self.T.MENTHLTH_ELEVATED_DAYS:
            add("Frequent mental distress reported", 1.0)
        if f["phys_hlth"] >= self.T.PHYSHLTH_ELEVATED_DAYS:
            add("Frequent physical distress reported", 1.0)

        if f["age"] >= self.T.AGE_RISK_THRESHOLD_3:
            add("Advanced age (60+ years)", 1.5)
        elif f["age"] >= self.T.AGE_RISK_THRESHOLD_2:
            add("Elevated age risk (45-59 years)", 1.0)
        elif f["age"] >= self.T.AGE_RISK_THRESHOLD_1:
            add("Moderate age risk (35-44 years)", 0.5)

        severity = min(points / self.T.CLINICAL_SEVERITY_MAX, 1.0)
        severity_percentage = round(severity * 100, 2)
        
        if severity < 0.2:
            label = "Minimal"
            explanation = "Very few or no significant clinical risk factors present."
        elif severity < 0.4:
            label = "Mild"
            explanation = "Some minor risk factors present, primarily lifestyle related."
        elif severity < 0.6:
            label = "Moderate"
            explanation = "Multiple clinical risk factors are present, increasing overall metabolic risk."
        elif severity < 0.8:
            label = "High"
            explanation = "Significant combination of clinical risk factors requiring medical attention."
        else:
            label = "Severe"
            explanation = "Critical aggregation of risk factors including multiple chronic conditions."

        return ClinicalSeverity(
            score                = round(severity, 4),
            active_factors       = len(factors),
            severity_points      = round(points, 2),
            factors              = factors,
            severity_label       = label,
            severity_explanation = explanation,
            severity_percentage  = severity_percentage
        )

    def _dynamic_risk_score(
        self,
        raw_probability:   float,
        calibrated_score:  float,
        confidence:        float,
        clinical_severity: ClinicalSeverity,
        flags:             List[str],
        features:          Optional[dict] = None,
    ) -> Dict[str, Any]:
        if features is None:
            features = {}

        probability_component = calibrated_score * 0.45
        clinical_component = clinical_severity.score * 0.15

        symptom_score = features.get("symptom_score", 0.0)
        symptom_severity = min(symptom_score / 73.0, 1.0) * 0.20

        age = features.get("age", 25)
        age_factor = min(max((age - 20) / 60.0, 0.0), 1.0) * 0.10

        lifestyle_risk = 0.0
        if features.get("smoker") == 1: lifestyle_risk += 0.02
        if features.get("phys_activity") == 0: lifestyle_risk += 0.02
        if features.get("fruits") == 0 and features.get("veggies") == 0: lifestyle_risk += 0.01

        confidence_component = self._confidence_adjustment(confidence, clinical_severity.score)
        
        flag_component = min(len(flags) * 0.015, 0.06)

        dynamic_score = (
            probability_component
            + clinical_component
            + symptom_severity
            + age_factor
            + lifestyle_risk
            + confidence_component
            + flag_component
        )

        # To ensure the risk score respects the ML model's high confidence but still dynamically 
        # responds to symptom durations (scaling up to ~0.94+ as requested), we apply the 
        # calibrated score as a floor, and add the symptom severity as a dynamic bonus.
        if raw_probability >= 0.85 or clinical_severity.score >= 0.35:
            base_score = max(dynamic_score, calibrated_score)
            symptom_bonus = (symptom_score / 73.0) * 0.05
            dynamic_score = base_score + symptom_bonus

        # --- STRICT SYMPTOM OVERRIDE RULE ---
        # The user specifically requested that symptoms heavily drive the prediction.
        if symptom_score >= 15:
            dynamic_score = max(0.75, dynamic_score)  # Force HIGH
        elif symptom_score >= 5:
            dynamic_score = max(0.45, min(dynamic_score, 0.64))  # Force MEDIUM
        else:
            dynamic_score = min(dynamic_score, 0.34)  # Force LOW

        final_score = round(max(0.0, min(0.9999, dynamic_score)), 6)

        return {
            "total_score": final_score,
            "components": {
                "probability_component": round(probability_component, 6),
                "clinical_component":    round(clinical_component, 6),
                "symptom_component":     round(symptom_severity, 6),
                "age_component":         round(age_factor, 6),
                "lifestyle_component":   round(lifestyle_risk, 6),
                "confidence_component":  round(confidence_component, 6),
                "flag_component":        round(flag_component, 6)
            }
        }

    def _confidence_adjustment(self, confidence: float, severity_score: float) -> float:
        if confidence < Config.CONFIDENCE_LOW and severity_score < 0.50:
            return -0.04
        if confidence < Config.CONFIDENCE_LOW:
            return -0.02
        if confidence >= Config.CONFIDENCE_HIGH and severity_score >= 0.50:
            return 0.02
        return 0.0

    def _classify_confidence(self, confidence: float, features: dict) -> Dict[str, str]:
        if confidence >= 0.90:
            label = "Very High"
            reason = "Model predictions align strongly with clinical inputs."
        elif confidence >= 0.80:
            label = "High"
            reason = "Solid predictive consensus."
        elif confidence >= 0.60:
            label = "Moderate"
            reason = "Standard confidence range."
        elif confidence >= 0.40:
            label = "Low"
            reason = "Mixed signals or edge-case demographics reduce certainty."
        else:
            label = "Very Low"
            reason = "Atypical symptom combinations heavily limit model certainty."
            
        return {
            "label": label,
            "reason": reason,
            "score": round(confidence, 4)
        }

    def _classify_risk_level(self, score: float) -> str:
        bands = Config.ML_RISK_BANDS
        for level, (low, high) in bands.items():
            if low <= score < high:
                return level.capitalize()
        return "High"

    def _classify_diabetes_category(
        self,
        prediction: int,
        score:      float,
    ) -> str:
        high_cut   = Config.ML_RISK_BANDS["high"][0]
        medium_cut = Config.ML_RISK_BANDS["medium"][0]

        if prediction == 1:
            if score >= high_cut:
                return "Diabetic"
            return "Pre-Diabetic"

        if score >= medium_cut:
            return "Pre-Diabetic"

        return "Non-Diabetic"

    def _build_prediction_label(self, category: str, risk_level: str) -> str:
        label_map = {
            ("Non-Diabetic", "Low"):    "No Diabetes Risk Detected",
            ("Non-Diabetic", "Medium"): "Low Diabetes Risk — Monitor Regularly",
            ("Non-Diabetic", "High"):   "Borderline Risk — Clinical Review Advised",
            ("Pre-Diabetic", "Low"):    "Pre-Diabetic — Lifestyle Intervention Recommended",
            ("Pre-Diabetic", "Medium"): "Pre-Diabetic — Moderate Risk Detected",
            ("Pre-Diabetic", "High"):   "Pre-Diabetic — High Risk, Consult Physician",
            ("Diabetic",     "Low"):    "Diabetic — Managed Risk Profile",
            ("Diabetic",     "Medium"): "Diabetic — Active Medical Management Required",
            ("Diabetic",     "High"):   "Diabetic — High Risk, Immediate Attention Required",
        }
        return label_map.get((category, risk_level), f"{category} — {risk_level} Risk")

    def _build_risk_interpretation(self, category: str, risk_level: str, flags: List[dict]) -> Dict[str, str]:
        short = f"{risk_level} diabetes risk."
        
        reasons = [f["category"].lower() for f in flags if f["severity"] in ("High", "Severe")]
        if reasons:
            detailed = f"The predicted probability is elevated due to factors including {', '.join(reasons)}."
        else:
            detailed = f"The model indicates a {risk_level.lower()} risk profile based on current metabolic indicators."
            
        clinical = "Recommend preventative lifestyle maintenance."
        if category == "Diabetic" or risk_level == "High":
            clinical = "Recommend HbA1c screening and physician follow-up."
        elif category == "Pre-Diabetic":
            clinical = "Recommend lifestyle modification and periodic monitoring."

        return {
            "short": short,
            "detailed": detailed,
            "clinical_summary": clinical
        }

    def _generate_structured_flags(self, f: dict, score: float) -> List[dict]:
        flags = []

        if f["gen_hlth"] >= self.T.GENHLTH_POOR:
            flags.append({"category": "General Health", "severity": "High", "message": "Self-reported general health is poor", "priority": "High"})
        elif f["gen_hlth"] >= self.T.GENHLTH_FAIR:
            flags.append({"category": "General Health", "severity": "Moderate", "message": "Self-reported general health is fair", "priority": "Medium"})

        if f["bmi"] >= self.T.BMI_SEVERELY_OBESE:
            flags.append({"category": "BMI", "severity": "Severe", "message": f"Severe obesity detected (BMI {f['bmi']:.1f} ≥ {self.T.BMI_SEVERELY_OBESE})", "priority": "High"})
        elif f["bmi"] >= self.T.BMI_OBESE:
            flags.append({"category": "BMI", "severity": "High", "message": f"Obese BMI range (BMI {f['bmi']:.1f})", "priority": "High"})
        elif f["bmi"] < self.T.BMI_UNDERWEIGHT:
            flags.append({"category": "BMI", "severity": "Moderate", "message": f"Underweight BMI detected (BMI {f['bmi']:.1f})", "priority": "Medium"})

        if f["high_bp"] >= 1:
            flags.append({"category": "Blood Pressure", "severity": "High", "message": "Elevated blood pressure risk indicator present", "priority": "High"})

        if f["high_chol"] >= 1:
            flags.append({"category": "Cholesterol", "severity": "High", "message": "Elevated cholesterol risk indicator present", "priority": "High"})

        if f["ment_hlth"] >= self.T.MENTHLTH_ELEVATED_DAYS:
            flags.append({
                "category": "Mental Health",
                "severity": "Moderate",
                "message": f"Frequent mental distress reported ({f['ment_hlth']:.0f}+ days in past month)",
                "priority": "Medium"
            })

        if f["phys_hlth"] >= self.T.PHYSHLTH_ELEVATED_DAYS:
            flags.append({
                "category": "Physical Health",
                "severity": "High",
                "message": f"Persistent physical symptoms reported ({f['phys_hlth']:.0f}+ days in past month)",
                "priority": "High"
            })

        if f["smoker"] >= 1:
            flags.append({"category": "Lifestyle", "severity": "High", "message": "Smoking increases metabolic risk", "priority": "High"})

        if f["phys_activity"] < 1:
            flags.append({"category": "Lifestyle", "severity": "Moderate", "message": "Sedentary lifestyle increases risk", "priority": "Medium"})

        if f["fruits"] < 1 and f["veggies"] < 1:
            flags.append({"category": "Diet", "severity": "Low", "message": "Low fruit and vegetable intake reported", "priority": "Low"})

        if f["age"] >= self.T.AGE_RISK_THRESHOLD_3:
            flags.append({"category": "Age", "severity": "High", "message": f"Age-related risk factor elevated (age {f['age']:.0f})", "priority": "Medium"})

        if score >= 0.85:
            flags.append({"category": "Composite Risk", "severity": "Severe", "message": "Composite risk score critically high — multiple compounding risk factors present", "priority": "Critical"})

        return flags

    def _generate_flags(self, f: dict, score: float) -> List[str]:
        # Kept for strict backward compatibility internally, though we use _generate_structured_flags now
        structured = self._generate_structured_flags(f, score)
        return [item["message"] for item in structured]

    def _generate_recommendations(
        self,
        risk_level:        str,
        diabetes_category: str,
        f:                 dict,
        flags:             List[str],
    ) -> List[str]:
        proxy_score = {"Low": 0.25, "Medium": 0.50, "High": 0.75}.get(risk_level, 0.50)
        plan = self._build_recommendation_plan(
            risk_level, diabetes_category, proxy_score, proxy_score, 1.0, self._to_model_features(f), flags
        )
        return self._recommendation_messages(plan)

    def _build_recommendation_plan(
        self,
        risk_level:        str,
        diabetes_category: str,
        dynamic_score:     float,
        calibrated_score:  float,
        confidence:        float,
        input_features:    dict,
        flags:             List[str],
    ) -> Any:
        return recommendation_engine.generate(
            calibrated_probability = calibrated_score,
            risk_level             = risk_level,
            confidence             = confidence,
            features               = input_features,
            flags                  = flags,
            diabetes_category      = diabetes_category,
        )

    def _recommendation_messages(self, plan: Any) -> List[str]:
        recommendations = getattr(plan, "recommendations", [])
        messages: List[str] = []
        for rec in recommendations:
            messages.append(getattr(rec, "message", str(rec)))
        return messages

    def _build_shap_context(
        self,
        predictor_result: dict,
        input_features: dict,
    ) -> Optional[dict]:
        features = predictor_result.get("features_used") or input_features
        if not features:
            return None

        try:
            predictor._ensure_loaded()
            model = getattr(predictor, "_pipeline", None)
            if model is None:
                return None

            explanation = shap_explainer.explain(
                model=model,
                features=features,
                feature_columns=getattr(predictor, "_feature_columns", None),
                top_n=5,
            )
            return explanation.to_dict()
        except Exception as exc:
            logger.warning("SHAP explanation unavailable: %s", exc)
            return None

    def _build_metadata(
        self,
        f:                    dict,
        raw_prob:             float,
        dynamic_score:        float,
        model_version:        str,
        prediction:           int,
        calibrated_score:     Optional[float] = None,
        clinical_severity:    Optional[ClinicalSeverity] = None,
        recommendation_plan:  Optional[Any] = None,
        shap_context:         Optional[dict] = None,
        dynamic_components:   Optional[Dict[str, Any]] = None,
        flags_data:           Optional[List[dict]] = None,
        risk_interpretation:  Optional[Dict[str, str]] = None,
        confidence_interpretation: Optional[Dict[str, str]] = None,
    ) -> dict:
        bmi_category = (
            "Underweight"    if f["bmi"] < self.T.BMI_UNDERWEIGHT  else
            "Normal"         if f["bmi"] <= self.T.BMI_NORMAL_MAX  else
            "Overweight"     if f["bmi"] <= self.T.BMI_OVERWEIGHT  else
            "Obese"          if f["bmi"] < self.T.BMI_SEVERELY_OBESE else
            "Severely Obese"
        )

        gen_hlth_category = (
            "Excellent" if f["gen_hlth"] <= 1 else
            "Very Good" if f["gen_hlth"] == 2 else
            "Good"      if f["gen_hlth"] == 3 else
            "Fair"      if f["gen_hlth"] == 4 else
            "Poor"
        )

        age_group = "Unknown"
        for group, (low, high) in Config.AGE_GROUPS.items():
            if low <= f["age"] <= high:
                age_group = group
                break

        score_delta = round((calibrated_score - raw_prob) if calibrated_score is not None else 0.0, 4)
        
        # Calculate protective factors
        protective = []
        if f["phys_activity"] >= 1: protective.append("Physically Active")
        if f["fruits"] >= 1 or f["veggies"] >= 1: protective.append("Regular Fruit/Vegetable Consumption")
        if f["smoker"] == 0: protective.append("Non-Smoker")
        if f["high_bp"] == 0: protective.append("Normal Blood Pressure")
        if f["high_chol"] == 0: protective.append("Healthy Cholesterol Profile")
        if f["bmi"] <= self.T.BMI_NORMAL_MAX and f["bmi"] >= self.T.BMI_UNDERWEIGHT: protective.append("Healthy BMI")

        metadata = {
            "model_version":      model_version,
            "raw_probability":    round(raw_prob,  4),
            "calibrated_score":   round(calibrated_score, 4) if calibrated_score is not None else None,
            "score_adjustment":   score_delta,
            "calibrated_probability": round(
                calibrated_score if calibrated_score is not None else dynamic_score,
                4,
            ),
            "dynamic_risk_score":  round(dynamic_score, 4),
            "bmi_category":       bmi_category,
            "gen_hlth_category":  gen_hlth_category,
            "age_group":          age_group,
            "binary_prediction":  prediction,
            "risk_thresholds": {
                "bmi_obese":              self.T.BMI_OBESE,
                "gen_hlth_fair":          self.T.GENHLTH_FAIR,
                "menthlth_elevated_days": self.T.MENTHLTH_ELEVATED_DAYS,
                "physhlth_elevated_days": self.T.PHYSHLTH_ELEVATED_DAYS,
            },
            "protective_factors": protective,
        }

        if clinical_severity is not None:
            metadata["clinical_severity"] = clinical_severity.to_dict()

        if dynamic_components:
            metadata["score_components"] = dynamic_components["components"]
            metadata["contributing_factors"] = [
                k for k, v in dynamic_components["components"].items() if v > 0.05
            ]

        if flags_data:
            metadata["medical_flags"] = flags_data

        if risk_interpretation:
            metadata["interpretation"] = risk_interpretation

        if confidence_interpretation:
            metadata["confidence_classification"] = confidence_interpretation

        if recommendation_plan is not None and hasattr(recommendation_plan, "to_dict"):
            plan = recommendation_plan.to_dict()
            metadata["recommendation_plan"] = {
                "follow_up_timeline": plan.get("follow_up_timeline"),
                "urgent":             plan.get("urgent"),
                "educational_topics": plan.get("educational_topics", []),
                "metadata":           plan.get("metadata", {}),
            }
            metadata["recommendation_summary"] = f"{len(metadata['recommendation_plan']['educational_topics'])} educational topics provided."

        if shap_context:
            metadata["shap_explanation"] = shap_context

        return metadata


risk_engine = DiabetesRiskEngine()
