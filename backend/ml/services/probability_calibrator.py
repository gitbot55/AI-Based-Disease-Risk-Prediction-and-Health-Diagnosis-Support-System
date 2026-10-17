import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationRule:
    name: str
    adjustment: float
    description: str


@dataclass
class CalibrationResult:
    calibrated_probability: float
    adjustment: float
    confidence: float
    calibration_weight: float
    applied_rules: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class CalibrationThresholds:
    BMI_UNDERWEIGHT: float = 18.5
    BMI_OVERWEIGHT: float = 25.0
    BMI_OBESE: float = 30.0
    BMI_SEVERELY_OBESE: float = 35.0
    GENHLTH_GOOD: int = 3
    GENHLTH_FAIR: int = 4
    GENHLTH_POOR: int = 5
    MENTHLTH_ELEVATED_DAYS: int = 14
    PHYSHLTH_ELEVATED_DAYS: int = 14
    AGE_RISK_THRESHOLD_1: int = 35
    AGE_RISK_THRESHOLD_2: int = 45
    AGE_RISK_THRESHOLD_3: int = 60
    PROBABILITY_MIN: float = 0.0
    PROBABILITY_MAX: float = 0.9999
    MAX_RULE_DELTA: float = 0.08


FEATURE_MAP = {
    "high_bp": "HighBP", "high_chol": "HighChol", "bmi": "BMI",
    "smoker": "Smoker", "phys_activity": "PhysActivity",
    "fruits": "Fruits", "veggies": "Veggies",
    "gen_hlth": "GenHlth", "ment_hlth": "MentHlth",
    "phys_hlth": "PhysHlth", "age": "Age",
}


class ProbabilityCalibrator:
    """Converts raw ML probability into a bounded clinically calibrated probability."""
    T = CalibrationThresholds
    _instance: "ProbabilityCalibrator | None" = None

    def __new__(cls) -> "ProbabilityCalibrator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def calibrate(self, probability: float, features: dict[str, Any]) -> CalibrationResult:
        raw_probability = self._normalise_probability(probability)
        extracted = self._extract_features(features)
        rules = self._evaluate_rules(extracted)
        raw_delta = sum(rule.adjustment for rule in rules)
        bounded_delta = self._bound_delta(raw_delta)
        weight = self._calibration_weight(raw_probability)
        weighted_delta = round(bounded_delta * weight, 6)
        calibrated = self._clamp_probability(raw_probability + weighted_delta)
        result = CalibrationResult(
            calibrated_probability=calibrated,
            adjustment=round(calibrated - raw_probability, 6),
            confidence=self._confidence(raw_probability, weighted_delta, extracted),
            calibration_weight=weight,
            applied_rules=[rule.name for rule in rules],
            metadata=self._metadata(raw_probability, calibrated, raw_delta, bounded_delta, extracted, rules),
        )
        logger.info(
            "Probability calibrated | raw=%.4f | calibrated=%.4f | adjustment=%.4f | rules=%d",
            raw_probability, result.calibrated_probability, result.adjustment, len(result.applied_rules),
        )
        return result
    def _extract_features(self, raw: dict[str, Any]) -> dict[str, Optional[float]]:
        return {name: self._as_float(raw.get(source)) for name, source in FEATURE_MAP.items()}

    def _evaluate_rules(self, f: dict[str, Optional[float]]) -> list[CalibrationRule]:
        rules: list[CalibrationRule] = []
        self._add_bmi_rules(rules, f["bmi"])
        self._add_age_rules(rules, f["age"])
        self._add_cardiometabolic_rules(rules, f)
        self._add_health_rules(rules, f)
        self._add_lifestyle_rules(rules, f)
        return rules

    def _add_bmi_rules(self, rules: list[CalibrationRule], bmi: Optional[float]) -> None:
        if bmi is None:
            return
        if bmi >= self.T.BMI_SEVERELY_OBESE:
            self._add(rules, "bmi_severely_obese", 0.055, "BMI is severely obese")
        elif bmi >= self.T.BMI_OBESE:
            self._add(rules, "bmi_obese", 0.035, "BMI is obese")
        elif bmi >= self.T.BMI_OVERWEIGHT:
            self._add(rules, "bmi_overweight", 0.015, "BMI is overweight")
        elif 0 < bmi < self.T.BMI_UNDERWEIGHT:
            self._add(rules, "bmi_underweight", -0.015, "BMI is underweight")

    def _add_age_rules(self, rules: list[CalibrationRule], age: Optional[float]) -> None:
        if age is None:
            return
        if age >= self.T.AGE_RISK_THRESHOLD_3:
            self._add(rules, "age_60_plus", 0.035, "Age is 60 or older")
        elif age >= self.T.AGE_RISK_THRESHOLD_2:
            self._add(rules, "age_45_plus", 0.020, "Age is 45 or older")
        elif age >= self.T.AGE_RISK_THRESHOLD_1:
            self._add(rules, "age_35_plus", 0.010, "Age is 35 or older")

    def _add_cardiometabolic_rules(self, rules: list[CalibrationRule], f: dict[str, Optional[float]]) -> None:
        high_bp = self._enabled(f["high_bp"])
        high_chol = self._enabled(f["high_chol"])
        if high_bp:
            self._add(rules, "high_blood_pressure", 0.030, "HighBP is present")
        if high_chol:
            self._add(rules, "high_cholesterol", 0.025, "HighChol is present")
        if high_bp and high_chol:
            self._add(rules, "bp_cholesterol_combined", 0.015, "HighBP and HighChol co-occur")

    def _add_health_rules(self, rules: list[CalibrationRule], f: dict[str, Optional[float]]) -> None:
        gen_hlth = f["gen_hlth"]
        if gen_hlth is not None and gen_hlth >= self.T.GENHLTH_POOR:
            self._add(rules, "poor_general_health", 0.045, "General health is poor")
        elif gen_hlth is not None and gen_hlth >= self.T.GENHLTH_FAIR:
            self._add(rules, "fair_general_health", 0.030, "General health is fair")
        elif gen_hlth is not None and gen_hlth <= self.T.GENHLTH_GOOD:
            self._add(rules, "good_general_health", -0.008, "General health is good")
        if self._gte(f["ment_hlth"], self.T.MENTHLTH_ELEVATED_DAYS):
            self._add(rules, "elevated_mental_health_days", 0.015, "MentHlth is elevated")
        if self._gte(f["phys_hlth"], self.T.PHYSHLTH_ELEVATED_DAYS):
            self._add(rules, "elevated_physical_health_days", 0.020, "PhysHlth is elevated")

    def _add_lifestyle_rules(self, rules: list[CalibrationRule], f: dict[str, Optional[float]]) -> None:
        if self._enabled(f["smoker"]):
            self._add(rules, "smoking", 0.020, "Smoking is reported")
        if self._disabled(f["phys_activity"]):
            self._add(rules, "low_physical_activity", 0.020, "Physical activity is absent")
        elif self._enabled(f["phys_activity"]):
            self._add(rules, "physical_activity_protective", -0.010, "Physical activity is present")
        if self._disabled(f["fruits"]) and self._disabled(f["veggies"]):
            self._add(rules, "low_fruit_vegetable_intake", 0.012, "Fruits and Veggies are absent")
        elif self._enabled(f["fruits"]) and self._enabled(f["veggies"]):
            self._add(rules, "diet_protective", -0.008, "Fruits and Veggies are present")

    def _metadata(self, raw_probability: float, calibrated_probability: float, raw_delta: float,
                  bounded_delta: float, features: dict[str, Optional[float]],
                  rules: list[CalibrationRule]) -> dict:
        return {
            "raw_probability": round(raw_probability, 6),
            "calibrated_probability": round(calibrated_probability, 6),
            "raw_rule_delta": round(raw_delta, 6),
            "bounded_rule_delta": round(bounded_delta, 6),
            "max_rule_delta": self.T.MAX_RULE_DELTA,
            "known_feature_count": sum(value is not None for value in features.values()),
            "evaluated_feature_count": len(features),
            "calibration_method": "adaptive_clinical_rule_weighting",
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "rule_details": [rule.__dict__ for rule in rules],
        }

    def _normalise_probability(self, probability: float) -> float:
        try:
            value = float(probability)
        except (TypeError, ValueError):
            raise ValueError("Probability must be numeric")
        if not math.isfinite(value):
            raise ValueError("Probability must be finite")
        return self._clamp_probability(value)

    def _calibration_weight(self, probability: float) -> float:
        if probability >= 0.90:
            return 0.08
        if probability >= 0.75:
            return 0.15
        return 0.25

    def _confidence(self, probability: float, adjustment: float, features: dict[str, Optional[float]]) -> float:
        completeness = sum(value is not None for value in features.values()) / len(features)
        model_certainty = abs(probability - 0.5) * 2
        adjustment_penalty = min(abs(adjustment) / self.T.MAX_RULE_DELTA, 1.0) * 0.20
        confidence = (0.60 * model_certainty) + (0.40 * completeness) - adjustment_penalty
        return round(max(0.0, min(1.0, confidence)), 4)

    def _as_float(self, value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _bound_delta(self, delta: float) -> float:
        return round(max(-self.T.MAX_RULE_DELTA, min(self.T.MAX_RULE_DELTA, delta)), 6)

    def _clamp_probability(self, probability: float) -> float:
        return round(float(max(self.T.PROBABILITY_MIN, min(self.T.PROBABILITY_MAX, probability))), 6)

    def _add(self, rules: list[CalibrationRule], name: str, adjustment: float, description: str) -> None:
        rules.append(CalibrationRule(name, adjustment, description))
    def _enabled(self, value: Optional[float]) -> bool:
        return value is not None and value >= 1
    def _disabled(self, value: Optional[float]) -> bool:
        return value is not None and value < 1
    def _gte(self, value: Optional[float], threshold: float) -> bool:
        return value is not None and value >= threshold


probability_calibrator = ProbabilityCalibrator()
