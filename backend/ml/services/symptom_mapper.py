import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SymptomMapperError(Exception):
    pass


class SymptomMapper:

    MAX_DURATION = 30

    BASE_BMI = 22.5

    SYMPTOM_WEIGHTS = {
        "frequent_urination_days": 1.4,
        "excessive_thirst_days": 1.4,
        "fatigue_days": 1.0,
        "blurred_vision_days": 1.0,
        "slow_healing_wounds_days": 1.2,
        "weight_loss_days": 1.3,
    }

    def map_to_features(self, data: Dict) -> Dict:

        age = self._extract_age(data)

        symptoms = data.get("symptoms", {})
        lifestyle = data.get("lifestyle", {})

        symptom_score = self._calculate_symptom_score(symptoms)

        lifestyle_score = self._calculate_lifestyle_score(lifestyle)

        bmi_estimate = self._estimate_bmi(
            symptom_score,
            lifestyle_score,
            age
        )

        gen_hlth = self._estimate_general_health(symptom_score)

        phys_hlth = self._estimate_physical_health(symptom_score)

        ment_hlth = self._estimate_mental_health(
            symptom_score,
            lifestyle_score
        )

        high_bp = self._estimate_high_bp(
            symptom_score,
            bmi_estimate,
            age,
            lifestyle
        )

        high_chol = self._estimate_high_chol(
            bmi_estimate,
            age,
            lifestyle
        )

        smoker = 1 if lifestyle.get("smoking", False) else 0

        phys_activity = 1 if lifestyle.get("exercise", False) else 0

        if phys_activity:
            fruits = 1 if lifestyle.get("fruits", True) else 0
            veggies = 1 if lifestyle.get("veggies", True) else 0
        elif smoker:
            fruits = 1 if lifestyle.get("fruits", False) else 0
            veggies = 1 if lifestyle.get("veggies", False) else 0
        else:
            fruits = 1 if lifestyle.get("fruits", True) else 0
            veggies = 1 if lifestyle.get("veggies", True) else 0

        sex = data.get("sex", 1)

        features = {
            "HighBP": high_bp,
            "HighChol": high_chol,
            "BMI": round(bmi_estimate, 2),
            "Smoker": smoker,
            "PhysActivity": phys_activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "GenHlth": gen_hlth,
            "MentHlth": ment_hlth,
            "PhysHlth": phys_hlth,
            "Sex": sex,
            "Age": age,
            "symptom_score": symptom_score,
            "raw_symptoms": symptoms
        }

        logger.info(
            "Features generated | "
            "SymptomScore=%.2f | BMI=%.2f | "
            "BP=%d | Chol=%d | Age=%d",
            symptom_score,
            bmi_estimate,
            high_bp,
            high_chol,
            age
        )

        return features

    def _extract_age(self, data: Dict) -> int:

        try:
            age = int(data["age"])
        except Exception:
            raise SymptomMapperError(
                "Age must be provided and numeric."
            )

        if age < 1 or age > 120:
            raise SymptomMapperError(
                "Age must be between 1 and 120."
            )

        return age

    def _calculate_symptom_score(
        self,
        symptoms: Dict
    ) -> float:

        score = 0.0

        for symptom, weight in self.SYMPTOM_WEIGHTS.items():

            days = symptoms.get(symptom, 0)

            if not isinstance(days, (int, float)):
                raise SymptomMapperError(
                    f"{symptom} must be numeric."
                )

            days = max(
                0,
                min(days, self.MAX_DURATION)
            )

            normalized = days / self.MAX_DURATION

            progression = (
                normalized ** 0.8
            ) * 10

            score += progression * weight

        logger.debug(
            "Calculated Symptom Score = %.2f",
            score
        )

        return round(score, 2)

    def _calculate_lifestyle_score(
        self,
        lifestyle: Dict
    ) -> float:

        score = 0

        if lifestyle.get("smoking", False):
            score += 2.5

        if lifestyle.get("alcohol", False):
            score += 1.0

        if lifestyle.get("exercise", False):
            score -= 1.5

        return score
    
    def _estimate_bmi(
        self,
        symptom_score: float,
        lifestyle_score: float,
        age: int
    ) -> float:

        bmi = (
            self.BASE_BMI
            + (symptom_score * 0.15)
            + (lifestyle_score * 1.2)
            + max(age - 40, 0) * 0.03
        )

        return max(18.5, min(42.0, bmi))

    def _estimate_high_bp(
        self,
        symptom_score: float,
        bmi: float,
        age: int,
        lifestyle: Dict
    ) -> int:

        score = (
            symptom_score * 0.45
            + max(bmi - 25, 0) * 0.50
            + max(age - 35, 0) * 0.12
        )

        if lifestyle.get("smoking", False):
            score += 2

        if not lifestyle.get("exercise", False):
            score += 1

        return 1 if score >= 10 else 0

    def _estimate_high_chol(
        self,
        bmi: float,
        age: int,
        lifestyle: Dict
    ) -> int:

        score = (
            max(bmi - 24, 0) * 0.40
            + age * 0.03
        )

        if lifestyle.get("smoking", False):
            score += 2

        if lifestyle.get("alcohol", False):
            score += 1

        return 1 if score >= 5.5 else 0

    def _estimate_general_health(
        self,
        symptom_score: float
    ) -> int:

        if symptom_score < 5:
            return 1
        elif symptom_score < 10:
            return 2
        elif symptom_score < 16:
            return 3
        elif symptom_score < 23:
            return 4
        else:
            return 5

    def _estimate_physical_health(
        self,
        symptom_score: float
    ) -> int:

        value = int(symptom_score * 0.85)

        return max(0, min(30, value))

    def _estimate_mental_health(
        self,
        symptom_score: float,
        lifestyle_score: float
    ) -> int:

        value = (
            symptom_score * 0.25
            + max(lifestyle_score, 0) * 2
        )

        return max(0, min(30, int(value)))

    def get_feature_summary(
        self,
        data: Dict
    ) -> Dict:

        features = self.map_to_features(data)

        return {
            "feature_count": len(features),
            "features": features
        }


symptom_mapper = SymptomMapper()