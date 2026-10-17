import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, List, Dict

from config import Config

logger = logging.getLogger(__name__)


class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RecommendationCategory(str, Enum):
    CONSULTATION = "consultation"
    SCREENING = "screening"
    LIFESTYLE = "lifestyle"
    NUTRITION = "nutrition"
    EDUCATION = "education"
    MONITORING = "monitoring"
    WARNING = "warning"


@dataclass(frozen=True)
class Recommendation:
    title: str
    message: str
    category: RecommendationCategory
    priority: RecommendationPriority
    timeline: str
    rationale: str
    confidence: str = "Moderate"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "message": self.message,
            "category": self.category.value,
            "priority": self.priority.value,
            "timeline": self.timeline,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


@dataclass
class RecommendationPlan:
    recommendations: List[Recommendation] = field(default_factory=list)
    follow_up_timeline: str = "Annual preventive screening"
    urgent: bool = False
    educational_topics: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "follow_up_timeline": self.follow_up_timeline,
            "urgent": self.urgent,
            "educational_topics": self.educational_topics,
            "summary": self.summary,
            "metadata": self.metadata,
        }


class RecommendationThresholds:
    BMI_OVERWEIGHT: float = 25.0
    BMI_OBESE: float = 30.0
    BMI_SEVERELY_OBESE: float = 35.0
    AGE_SCREENING: int = 35
    AGE_HIGHER_RISK: int = 45
    GENHLTH_FAIR: int = 4
    GENHLTH_POOR: int = 5
    ELEVATED_HEALTH_DAYS: int = 14
    HIGH_PROBABILITY: float = Config.PROBABILITY_HIGH
    CRITICAL_PROBABILITY: float = Config.PROBABILITY_CRITICAL
    LOW_CONFIDENCE: float = Config.CONFIDENCE_LOW


FEATURE_MAP = {
    "high_bp": "HighBP",
    "high_chol": "HighChol",
    "bmi": "BMI",
    "smoker": "Smoker",
    "phys_activity": "PhysActivity",
    "fruits": "Fruits",
    "veggies": "Veggies",
    "gen_hlth": "GenHlth",
    "ment_hlth": "MentHlth",
    "phys_hlth": "PhysHlth",
    "age": "Age",
}


class RecommendationEngine:
    """
    Generates patient-facing recommendation plans from existing risk outputs.

    This service does not predict, calibrate probabilities, classify risk,
    perform SHAP analysis, access Flask, or read/write databases.
    """

    T = RecommendationThresholds
    _instance: "Optional[RecommendationEngine]" = None

    def __new__(cls) -> "RecommendationEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(
        self,
        calibrated_probability: float,
        risk_level: str,
        confidence: float,
        features: Dict[str, Any],
        flags: Optional[List[str]] = None,
        diabetes_category: Optional[str] = None,
    ) -> RecommendationPlan:
        """
        Build recommendations from already-computed clinical risk context.
        """
        start_time = time.perf_counter()
        logger.info("Recommendation generation started")

        try:
            logger.debug("Risk assessment received. Extracting patient profile.")
            probability = self._as_probability(calibrated_probability)
            model_confidence = self._as_probability(confidence)
            normalized_risk = self._normalise_label(risk_level)
            patient = self._extract_features(features)
            clinical_flags = flags or []

            # Determine base confidence for the recommendations generated in this session
            rec_confidence = self._determine_recommendation_confidence(probability, model_confidence)

            logger.debug("Lifestyle and risk analysis proceeding.")
            recs: List[Recommendation] = []
            
            # Core domain recommendations
            self._add_consultation_recs(recs, probability, normalized_risk, diabetes_category, clinical_flags, patient, rec_confidence)
            self._add_screening_recs(recs, probability, model_confidence, patient, rec_confidence)
            self._add_lifestyle_recs(recs, patient, rec_confidence)
            self._add_nutrition_recs(recs, patient, rec_confidence)
            self._add_monitoring_recs(recs, patient, clinical_flags, rec_confidence)
            self._add_education_recs(recs, normalized_risk, patient, rec_confidence)
            
            # Future Extensibility Hooks (Currently no-ops or basic defaults)
            self._add_sleep_recs(recs, patient, rec_confidence)
            self._add_stress_recs(recs, patient, rec_confidence)
            self._add_hydration_recs(recs, patient, rec_confidence)
            self._add_family_history_recs(recs, patient, rec_confidence)
            self._add_lab_recs(recs, patient, rec_confidence)

            logger.debug("Deduplication and priority ordering started.")
            unique = self._deduplicate(recs)
            
            follow_up = self._follow_up_timeline(probability, normalized_risk, model_confidence)
            urgent = any(rec.priority == RecommendationPriority.URGENT for rec in unique)
            
            logger.debug("Summary and metadata generation started.")
            educational = self._educational_topics(unique, patient)
            summary = self._generate_summary(unique)
            metadata = self._metadata(
                probability, normalized_risk, model_confidence, patient, clinical_flags, unique,
                duration_ms=(time.perf_counter() - start_time) * 1000
            )

            plan = RecommendationPlan(
                recommendations=unique,
                follow_up_timeline=follow_up,
                urgent=urgent,
                educational_topics=educational,
                summary=summary,
                metadata=metadata,
            )

            logger.info(
                "Recommendation plan generated | risk=%s | probability=%.4f | recs=%d | urgent=%s | duration=%.2fms",
                normalized_risk,
                probability,
                len(unique),
                urgent,
                (time.perf_counter() - start_time) * 1000
            )
            return plan

        except Exception as exc:
            logger.exception("Unexpected exception during recommendation generation: %s", exc)
            raise

    def generate_from_assessment(self, assessment: Any, features: Dict[str, Any]) -> RecommendationPlan:
        data = assessment.to_dict() if hasattr(assessment, "to_dict") else dict(assessment)
        return self.generate(
            calibrated_probability=data.get("risk_score", data.get("probability", 0.0)),
            risk_level=data.get("risk_level", "Unknown"),
            confidence=data.get("confidence", 0.0),
            features=features,
            flags=data.get("flags", []),
            diabetes_category=data.get("diabetes_category"),
        )

    def _determine_recommendation_confidence(self, probability: float, model_confidence: float) -> str:
        if model_confidence >= 0.85 and probability > 0.5:
            return "High"
        if model_confidence < self.T.LOW_CONFIDENCE:
            return "Low"
        return "Moderate"

    def _add_consultation_recs(
        self,
        recs: List[Recommendation],
        probability: float,
        risk_level: str,
        category: Optional[str],
        flags: List[str],
        f: Dict[str, float],
        confidence: str,
    ) -> None:
        if probability >= self.T.CRITICAL_PROBABILITY or risk_level == "high":
            self._add(
                recs,
                "Arrange prompt medical review",
                "Schedule a clinician visit for confirmatory glucose and HbA1c testing.",
                RecommendationCategory.CONSULTATION,
                RecommendationPriority.URGENT if probability >= self.T.CRITICAL_PROBABILITY else RecommendationPriority.HIGH,
                "Within 1 week",
                "High calibrated risk or high-risk clinical flags require timely confirmation.",
                confidence
            )
        elif risk_level == "medium" or category == "Pre-Diabetic":
            self._add(
                recs,
                "Book preventive follow-up",
                "Discuss this screening result with a healthcare professional and confirm with lab testing.",
                RecommendationCategory.CONSULTATION,
                RecommendationPriority.MEDIUM,
                "Within 2 to 4 weeks",
                "Moderate risk benefits from early clinical confirmation and lifestyle planning.",
                confidence
            )
        elif risk_level == "low" and f["age"] < self.T.AGE_SCREENING and f["gen_hlth"] <= 2:
            self._add(
                recs,
                "Maintain routine wellness",
                "Continue current healthy habits and keep standard annual check-ups.",
                RecommendationCategory.CONSULTATION,
                RecommendationPriority.LOW,
                "Annually",
                "Young, healthy patients benefit most from maintaining baseline health and routine monitoring.",
                confidence
            )

        if any("critically high" in flag.lower() or "immediate" in flag.lower() for flag in flags):
            self._add(
                recs,
                "Do not delay care",
                "Seek urgent medical advice if symptoms are worsening, severe, or persistent.",
                RecommendationCategory.WARNING,
                RecommendationPriority.URGENT,
                "Today",
                "High-risk clinical flags indicate possible need for expedited review.",
                confidence
            )
        elif f["gen_hlth"] >= self.T.GENHLTH_POOR:
            self._add(
                recs,
                "Discuss general health decline",
                "Schedule a comprehensive review to address underlying causes of poor general health.",
                RecommendationCategory.CONSULTATION,
                RecommendationPriority.HIGH,
                "Within 2 weeks",
                "Self-reported poor general health is a strong indicator of metabolic or systemic issues.",
                confidence
            )

    def _add_screening_recs(
        self,
        recs: List[Recommendation],
        probability: float,
        model_confidence: float,
        f: Dict[str, float],
        confidence: str,
    ) -> None:
        if probability >= self.T.HIGH_PROBABILITY or f["age"] >= self.T.AGE_SCREENING:
            self._add(
                recs,
                "Complete diabetes screening",
                "Request fasting glucose or HbA1c testing and review cardiovascular risk factors.",
                RecommendationCategory.SCREENING,
                RecommendationPriority.HIGH if probability >= self.T.HIGH_PROBABILITY else RecommendationPriority.MEDIUM,
                "At next clinical visit",
                "Age and elevated probability increase the value of confirmatory screening.",
                confidence
            )
        if model_confidence < self.T.LOW_CONFIDENCE:
            self._add(
                recs,
                "Confirm uncertain model signal",
                "Because prediction confidence is limited, rely on clinical testing before making decisions.",
                RecommendationCategory.SCREENING,
                RecommendationPriority.MEDIUM,
                "Before treatment decisions",
                "Low model confidence should be resolved through objective clinical measurements.",
                "High"
            )

    def _add_lifestyle_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        # Multi-factor analysis: Sedentary + Obesity
        if f["phys_activity"] < 1 and f["bmi"] >= self.T.BMI_OBESE:
            self._add(
                recs,
                "Combined weight and activity management",
                "Begin a supervised, gradual exercise program alongside a structured weight-loss plan.",
                RecommendationCategory.LIFESTYLE,
                RecommendationPriority.HIGH,
                "Begin within 2 weeks",
                "Physical inactivity combined with elevated BMI significantly multiplies diabetes risk.",
                confidence
            )
        elif f["phys_activity"] < 1:
            self._add(
                recs,
                "Increase weekly activity",
                "Aim for at least 150 minutes of moderate activity per week, starting gradually if inactive.",
                RecommendationCategory.LIFESTYLE,
                RecommendationPriority.MEDIUM,
                "Start this week",
                "Regular activity improves insulin sensitivity and metabolic health.",
                confidence
            )
        elif f["bmi"] >= self.T.BMI_OBESE:
            self._add(
                recs,
                "Plan sustainable weight management",
                "Set a gradual weight-loss goal with diet quality, activity, sleep, and clinical support.",
                RecommendationCategory.LIFESTYLE,
                RecommendationPriority.HIGH if f["bmi"] >= self.T.BMI_SEVERELY_OBESE else RecommendationPriority.MEDIUM,
                "Begin within 2 to 4 weeks",
                "Obesity is a major modifiable driver of diabetes risk.",
                confidence
            )

        # Multi-factor analysis: Smoking + High BP
        if f["smoker"] >= 1 and f["high_bp"] >= 1:
            self._add(
                recs,
                "Cardiovascular lifestyle intervention",
                "Prioritize smoking cessation to rapidly reduce acute cardiovascular stress, alongside BP management.",
                RecommendationCategory.LIFESTYLE,
                RecommendationPriority.URGENT,
                "Immediate",
                "Smoking compounds the vascular damage caused by high blood pressure, drastically increasing risk.",
                confidence
            )
        elif f["smoker"] >= 1:
            self._add(
                recs,
                "Start smoking cessation support",
                "Consider counselling, nicotine replacement, or a supervised cessation program.",
                RecommendationCategory.LIFESTYLE,
                RecommendationPriority.HIGH,
                "Within 2 weeks",
                "Smoking increases cardiometabolic risk and complicates diabetes prevention.",
                confidence
            )

    def _add_nutrition_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        poor_diet = f["fruits"] < 1 or f["veggies"] < 1
        
        # Multi-factor analysis: Poor Diet + High Cholesterol
        if poor_diet and f["high_chol"] >= 1:
            self._add(
                recs,
                "Heart-healthy nutrition plan",
                "Adopt a diet high in dietary fiber (vegetables/whole grains) to naturally reduce cholesterol.",
                RecommendationCategory.NUTRITION,
                RecommendationPriority.HIGH,
                "Start with the next meals",
                "A diet low in produce and high in cholesterol strongly drives metabolic syndrome.",
                confidence
            )
        else:
            if poor_diet:
                self._add(
                    recs,
                    "Improve diet quality",
                    "Increase vegetables, whole grains, legumes, lean protein, and unsweetened drinks.",
                    RecommendationCategory.NUTRITION,
                    RecommendationPriority.MEDIUM,
                    "Start with the next meals",
                    "Higher-quality dietary patterns support glucose control and weight management.",
                    confidence
                )
            if f["high_chol"] >= 1 or f["high_bp"] >= 1:
                self._add(
                    recs,
                    "Use a heart-healthy eating pattern",
                    "Limit sugary drinks, refined carbohydrates, excess sodium, and highly processed foods.",
                    RecommendationCategory.NUTRITION,
                    RecommendationPriority.MEDIUM,
                    "Start this week",
                    "Blood pressure and cholesterol risks often improve with dietary changes.",
                    confidence
                )

    def _add_monitoring_recs(self, recs: List[Recommendation], f: Dict[str, float], flags: List[str], confidence: str) -> None:
        if f["high_bp"] >= 1:
            self._add(
                recs,
                "Monitor blood pressure",
                "Track blood pressure readings and review them with a clinician.",
                RecommendationCategory.MONITORING,
                RecommendationPriority.MEDIUM,
                "Weekly until reviewed",
                "Blood pressure control lowers cardiovascular and metabolic complications.",
                confidence
            )
        if f["gen_hlth"] >= self.T.GENHLTH_FAIR or len(flags) >= 3:
            self._add(
                recs,
                "Track symptoms and function",
                "Record symptom changes, energy, sleep, activity, and any worsening physical symptoms.",
                RecommendationCategory.MONITORING,
                RecommendationPriority.MEDIUM,
                "Daily for 2 weeks",
                "Persistent or compounding symptoms should be visible during clinical review.",
                confidence
            )
        if f["ment_hlth"] >= self.T.ELEVATED_HEALTH_DAYS:
            self._add(
                recs,
                "Discuss mental distress",
                "Bring frequent stress, anxiety, low mood, or sleep disruption into the clinical visit.",
                RecommendationCategory.MONITORING,
                RecommendationPriority.MEDIUM,
                "At next appointment",
                "Mental distress can affect activity, nutrition, sleep, and chronic disease risk.",
                confidence
            )

    def _add_education_recs(self, recs: List[Recommendation], risk_level: str, f: Dict[str, float], confidence: str) -> None:
        if risk_level in {"medium", "high"}:
            self._add(
                recs,
                "Learn early warning signs",
                "Review symptoms such as excessive thirst, frequent urination, unexplained weight loss, and fatigue.",
                RecommendationCategory.EDUCATION,
                RecommendationPriority.LOW,
                "This week",
                "Symptom awareness helps patients seek timely clinical assessment.",
                confidence
            )
        if f["age"] >= self.T.AGE_HIGHER_RISK:
            self._add(
                recs,
                "Prioritize routine prevention",
                "Keep regular preventive visits for glucose, cholesterol, blood pressure, and weight review.",
                RecommendationCategory.EDUCATION,
                RecommendationPriority.LOW,
                "Ongoing",
                "Age increases baseline cardiometabolic screening needs.",
                confidence
            )

    # =========================================================================
    # Future Extensibility Methods
    # =========================================================================
    def _add_sleep_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        # Placeholder for future sleep-related recommendations.
        pass

    def _add_stress_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        # Placeholder for future stress management recommendations.
        pass

    def _add_hydration_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        # Placeholder for future hydration recommendations.
        pass

    def _add_family_history_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        # Placeholder for future family history risk adjustments.
        pass

    def _add_lab_recs(self, recs: List[Recommendation], f: Dict[str, float], confidence: str) -> None:
        # Placeholder for future laboratory-based recommendations.
        pass
    # =========================================================================

    def _extract_features(self, raw: Dict[str, Any]) -> Dict[str, float]:
        return {name: self._as_float(raw.get(source), 0.0) for name, source in FEATURE_MAP.items()}

    def _add(
        self,
        recs: List[Recommendation],
        title: str,
        message: str,
        category: RecommendationCategory,
        priority: RecommendationPriority,
        timeline: str,
        rationale: str,
        confidence: str = "Moderate",
    ) -> None:
        recs.append(Recommendation(title, message, category, priority, timeline, rationale, confidence))

    def _deduplicate(self, recs: List[Recommendation]) -> List[Recommendation]:
        seen: set[str] = set()
        unique: List[Recommendation] = []
        # Sort by priority first so highest priority variant is retained
        for rec in sorted(recs, key=lambda item: self._priority_rank(item.priority), reverse=True):
            # Key ensures we only get one recommendation per category/title
            key = f"{rec.category.value}:{rec.title}"
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        return unique

    def _follow_up_timeline(self, probability: float, risk_level: str, confidence: float) -> str:
        if probability >= self.T.CRITICAL_PROBABILITY or risk_level == "high":
            return "Clinical follow-up within 1 week"
        if probability >= self.T.HIGH_PROBABILITY or risk_level == "medium":
            return "Clinical follow-up within 2 to 4 weeks"
        if confidence < self.T.LOW_CONFIDENCE:
            return "Repeat or confirm screening before relying on the result"
        return "Routine preventive screening in 6 to 12 months"

    def _educational_topics(self, recs: List[Recommendation], f: Dict[str, float]) -> List[str]:
        topics = {"Diabetes prevention", "Glucose screening"}
        
        cats = {rec.category for rec in recs}
        
        if RecommendationCategory.NUTRITION in cats:
            topics.add("Healthy eating for glucose control")
            topics.add("Weight management")
            
        if f["phys_activity"] < 1:
            topics.add("Safe physical activity planning")
            
        if f["high_bp"] >= 1:
            topics.add("Blood pressure control")
            
        if f["high_chol"] >= 1:
            topics.add("Cholesterol management")
            
        if f["smoker"] >= 1:
            topics.add("Smoking cessation")
            
        if f["ment_hlth"] >= self.T.ELEVATED_HEALTH_DAYS:
            topics.add("Stress management")
            topics.add("Sleep")
            
        return sorted(list(topics))

    def _generate_summary(self, recs: List[Recommendation]) -> str:
        if not recs:
            return "No specific recommendations were generated for this profile."

        top_priorities = [rec.category.value.replace("_", " ") for rec in recs if rec.priority in (RecommendationPriority.URGENT, RecommendationPriority.HIGH)]
        if not top_priorities:
            top_priorities = [rec.category.value.replace("_", " ") for rec in recs[:3]]
            
        # Deduplicate and clean category names for natural language
        unique_cats = list(dict.fromkeys(top_priorities))[:4]
        
        if len(unique_cats) > 1:
            cats_text = ", ".join(unique_cats[:-1]) + ", and " + unique_cats[-1]
        elif len(unique_cats) == 1:
            cats_text = unique_cats[0]
        else:
            cats_text = "general wellness"

        return f"Based on the current health profile, the highest priorities involve {cats_text}."

    def _metadata(
        self,
        probability: float,
        risk_level: str,
        confidence: float,
        features: Dict[str, float],
        flags: List[str],
        recs: List[Recommendation],
        duration_ms: float
    ) -> dict:
        
        categories = {cat.value: 0 for cat in RecommendationCategory}
        urgent_count = 0
        high_count = 0
        
        groups: Dict[str, List[str]] = {}
        for rec in recs:
            categories[rec.category.value] += 1
            if rec.priority == RecommendationPriority.URGENT:
                urgent_count += 1
            elif rec.priority == RecommendationPriority.HIGH:
                high_count += 1
                
            group_name = rec.category.value.capitalize()
            if rec.priority in (RecommendationPriority.URGENT, RecommendationPriority.HIGH):
                group_name = "Immediate Actions"
                
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(rec.title)

        highest_priority = RecommendationPriority.LOW.value
        if recs:
            highest_priority = recs[0].priority.value
            
        clinical_risk_factors = sum([
            1 for k, v in features.items() if (k == 'high_bp' and v >= 1) or (k == 'high_chol' and v >= 1) or (k == 'bmi' and v >= 25.0) or (k == 'gen_hlth' and v >= 4) or (k == 'ment_hlth' and v >= 14) or (k == 'phys_hlth' and v >= 14)
        ])
        lifestyle_risk_factors = sum([
            1 for k, v in features.items() if (k == 'smoker' and v >= 1) or (k == 'phys_activity' and v < 1) or (k == 'fruits' and v < 1) or (k == 'veggies' and v < 1)
        ])
        demographic_risk_factors = sum([
            1 for k, v in features.items() if (k == 'age' and v >= 45)
        ])
        total_risk_factors = clinical_risk_factors + lifestyle_risk_factors + demographic_risk_factors

        patient_profile_summary = (
            f"Observed Clinical Factors: {clinical_risk_factors}, "
            f"Lifestyle Factors: {lifestyle_risk_factors}, "
            f"Demographic Factors: {demographic_risk_factors}, "
            f"Total Risk Factors: {total_risk_factors}"
        )
        risk_profile_summary = f"{risk_level.capitalize()} risk profile requiring {len(recs)} clinical interventions."

        return {
            "calibrated_probability": round(probability, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "feature_count": len(features),
            "flag_count": len(flags),
            "recommendation_count": len(recs),
            "urgent_count": urgent_count,
            "high_priority_count": high_count,
            "highest_priority": highest_priority,
            "consultation_count": categories.get(RecommendationCategory.CONSULTATION.value, 0),
            "screening_count": categories.get(RecommendationCategory.SCREENING.value, 0),
            "lifestyle_count": categories.get(RecommendationCategory.LIFESTYLE.value, 0),
            "nutrition_count": categories.get(RecommendationCategory.NUTRITION.value, 0),
            "education_count": categories.get(RecommendationCategory.EDUCATION.value, 0),
            "monitoring_count": categories.get(RecommendationCategory.MONITORING.value, 0),
            "recommendation_groups": groups,
            "patient_profile_summary": patient_profile_summary,
            "risk_profile_summary": risk_profile_summary,
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "engine": "recommendation_engine",
            "generation_time_ms": round(duration_ms, 2)
        }

    def _normalise_label(self, value: Optional[str]) -> str:
        return str(value or "unknown").strip().lower()

    def _as_probability(self, value: Any) -> float:
        numeric = self._as_float(value, 0.0)
        return max(0.0, min(0.9999, numeric))

    def _as_float(self, value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if math.isfinite(numeric) else default

    def _priority_rank(self, priority: RecommendationPriority) -> int:
        return {
            RecommendationPriority.URGENT: 4,
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 1,
        }[priority]


recommendation_engine = RecommendationEngine()
