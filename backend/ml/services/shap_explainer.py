import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShapExplainerError(Exception):
    """Raised when SHAP explanations cannot be generated safely."""
    pass


# Central registry of clinical feature metadata to ensure human-readable output
FEATURE_METADATA = {
    "HighBP": {
        "human_name": "High Blood Pressure",
        "negative_name": "Normal Blood Pressure",
        "description": "High blood pressure is associated with increased diabetes risk.",
        "positive_reason": "Because elevated blood pressure is strongly associated with diabetes risk.",
        "negative_reason": "Normal blood pressure reduces metabolic risk.",
        "category": "Blood Pressure",
        "importance": "Critical",
        "recommendation": "Maintaining healthy blood pressure through medical guidance and lifestyle changes may reduce long-term metabolic risk."
    },
    "BMI": {
        "human_name": "Elevated BMI",
        "negative_name": "Healthy BMI",
        "description": "Higher BMI often increases the likelihood of developing diabetes.",
        "positive_reason": "Because elevated BMI indicates excess body weight which increases metabolic strain.",
        "negative_reason": "Maintaining a healthy BMI supports metabolic function.",
        "category": "Body Composition",
        "importance": "Critical",
        "recommendation": "Weight management through a balanced diet and physical activity is highly recommended."
    },
    "GenHlth": {
        "human_name": "General Health",
        "description": "Self-reported general health is a strong predictor of chronic disease.",
        "positive_reason": "Because poor general health correlates strongly with chronic disease development.",
        "negative_reason": "Good general health indicates a lower baseline risk.",
        "category": "Physical Health",
        "importance": "High",
        "recommendation": "Regular medical check-ups and addressing general health concerns early is beneficial."
    },
    "MentHlth": {
        "human_name": "Mental Health",
        "description": "Mental distress can affect physical health and lifestyle habits.",
        "positive_reason": "Because frequent mental distress can impact metabolic health and self-care.",
        "negative_reason": "Good mental health supports overall well-being and healthy habits.",
        "category": "Mental Health",
        "importance": "Moderate",
        "recommendation": "Managing stress and seeking mental health support can improve overall health outcomes."
    },
    "PhysHlth": {
        "human_name": "Physical Health",
        "description": "Recent physical illness or injury indicates systemic health stress.",
        "positive_reason": "Because frequent physical illness suggests underlying health vulnerabilities.",
        "negative_reason": "Infrequent physical illness suggests good immune and systemic health.",
        "category": "Physical Health",
        "importance": "High",
        "recommendation": "Addressing physical symptoms with a healthcare provider helps maintain baseline health."
    },
    "PhysActivity": {
        "human_name": "Sedentary Lifestyle",
        "negative_name": "Physically Active",
        "description": "Regular physical activity improves insulin sensitivity.",
        "positive_reason": "Because a sedentary lifestyle significantly increases diabetes risk.",
        "negative_reason": "Regular exercise improves insulin sensitivity and metabolic health.",
        "category": "Lifestyle",
        "importance": "Critical",
        "recommendation": "Engaging in at least 150 minutes of moderate physical activity per week is advised."
    },
    "HighChol": {
        "human_name": "High Cholesterol",
        "negative_name": "Healthy Cholesterol Profile",
        "description": "High cholesterol often co-occurs with insulin resistance.",
        "positive_reason": "Because elevated cholesterol is a major risk factor for cardiovascular and metabolic diseases.",
        "negative_reason": "Healthy cholesterol levels indicate good metabolic function.",
        "category": "Metabolic Health",
        "importance": "High",
        "recommendation": "Dietary changes and medical management can help control cholesterol levels."
    },
    "Smoker": {
        "human_name": "Smoking",
        "negative_name": "Non-Smoker",
        "description": "Smoking increases inflammation and metabolic risks.",
        "positive_reason": "Because smoking introduces toxins that increase insulin resistance.",
        "negative_reason": "Not smoking significantly reduces cardiovascular and metabolic risks.",
        "category": "Lifestyle",
        "importance": "High",
        "recommendation": "Smoking cessation strongly reduces the risk of multiple chronic diseases."
    },
    "Fruits": {
        "human_name": "Low Fruit Consumption",
        "negative_name": "Regular Fruit Consumption",
        "description": "Fruits provide essential vitamins, minerals, and fiber.",
        "positive_reason": "Because low fruit intake indicates a diet lacking in essential nutrients.",
        "negative_reason": "A diet rich in fruits provides protective antioxidants and fiber.",
        "category": "Nutrition",
        "importance": "Moderate",
        "recommendation": "Incorporating whole fruits into the daily diet supports a healthy metabolism."
    },
    "Veggies": {
        "human_name": "Low Vegetable Consumption",
        "negative_name": "Regular Vegetable Consumption",
        "description": "Vegetables are crucial for fiber and micronutrients.",
        "positive_reason": "Because low vegetable intake reduces dietary fiber, which helps regulate blood sugar.",
        "negative_reason": "High vegetable intake is strongly protective against metabolic diseases.",
        "category": "Nutrition",
        "importance": "High",
        "recommendation": "Eating a variety of non-starchy vegetables daily helps maintain steady blood sugar levels."
    },
    "Sex": {
        "human_name": "Biological Sex",
        "description": "Biological sex can influence how and when diabetes develops.",
        "positive_reason": "Because certain metabolic risk factors present differently based on biological sex.",
        "negative_reason": "Biological sex is factored into the baseline risk calculation.",
        "category": "Demographics",
        "importance": "Informational",
        "recommendation": "Discuss sex-specific risk factors with a healthcare provider."
    },
    "Age": {
        "human_name": "Age",
        "description": "Diabetes risk naturally increases with age.",
        "positive_reason": "Because advancing age naturally decreases metabolic efficiency.",
        "negative_reason": "Younger age is generally protective against Type 2 diabetes.",
        "category": "Demographics",
        "importance": "Critical",
        "recommendation": "Regular screening for diabetes should begin or increase in frequency as you age."
    }
}


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    human_name: str
    description: str
    value: float
    shap_value: float
    direction: str
    impact: str
    reason: str
    category: str
    importance: str
    recommendation: Optional[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ShapExplanation:
    feature_importance: List[FeatureContribution]
    positive_contributors: List[FeatureContribution]
    negative_contributors: List[FeatureContribution]
    summary: str
    overall_explanation: str
    base_value: Optional[float] = None
    total_impact: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feature_importance": [item.to_dict() for item in self.feature_importance],
            "positive_contributors": [item.to_dict() for item in self.positive_contributors],
            "negative_contributors": [item.to_dict() for item in self.negative_contributors],
            "summary": self.summary,
            "overall_explanation": self.overall_explanation,
            "base_value": self.base_value,
            "total_impact": self.total_impact,
            "metadata": self.metadata,
        }


class ShapExplainer:
    """
    Generates model explanations from SHAP values.

    This service does not predict, calibrate probability, classify risk,
    generate recommendations, access Flask, or read application databases.
    """

    _instance: "Optional[ShapExplainer]" = None
    _shap: Any = None
    _explainer_cache: Dict[int, Any] = {}

    def __new__(cls) -> "ShapExplainer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def explain(
        self,
        model: Any,
        features: dict | pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> ShapExplanation:
        """
        Explain a single feature row for a trained model or sklearn pipeline.

        Parameters
        ----------
        model : Any
            A fitted estimator or sklearn Pipeline. TreeExplainer is preferred.
        features : dict | pd.DataFrame
            Single patient feature payload aligned to model feature names.
        feature_columns : list[str], optional
            Training-time feature order from the model bundle.
        top_n : int
            Maximum number of positive and negative contributors to return.
        """
        if model is None:
            raise ShapExplainerError("Cannot explain a missing model.")

        start_time = time.perf_counter()
        logger.info("SHAP explanation started.")

        try:
            frame = self._build_frame(features, feature_columns)
            
            t0 = time.perf_counter()
            estimator, explain_frame, names = self._prepare_model_input(model, frame)
            logger.debug("Pipeline transformation completed in %.2f ms", (time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            explainer = self._get_explainer(estimator)
            logger.debug("Explainer selection completed in %.2f ms", (time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            try:
                shap_output = explainer(explain_frame)
            except Exception:
                shap_output = self._legacy_shap_values(explainer, explain_frame)
            logger.debug("SHAP computation time: %.2f ms", (time.perf_counter() - t0) * 1000)

            shap_values = self._extract_values(shap_output)
            base_value = self._extract_base_value(shap_output)
            
            t0 = time.perf_counter()
            contributions = self._rank_contributions(shap_values, explain_frame.iloc[0], names)
            logger.debug("Ranking completed in %.2f ms", (time.perf_counter() - t0) * 1000)

            positives = [item for item in contributions if item.shap_value > 0][:top_n]
            negatives = [item for item in contributions if item.shap_value < 0][:top_n]
            top_features = contributions[:top_n]
            
            t0 = time.perf_counter()
            summary = self._summarize(positives, negatives)
            overall_explanation = self._build_overall_explanation(positives, negatives)
            logger.debug("Summary generation completed in %.2f ms", (time.perf_counter() - t0) * 1000)

            total_impact = round(float(np.sum(shap_values)), 6)
            
            t0 = time.perf_counter()
            metadata = self._build_metadata(
                estimator=estimator,
                explain_frame=explain_frame,
                top_n=top_n,
                contributions=contributions,
                positives=positives,
                negatives=negatives,
                total_impact=total_impact,
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )
            logger.debug("Metadata generation completed in %.2f ms", (time.perf_counter() - t0) * 1000)

            explanation = ShapExplanation(
                feature_importance=top_features,
                positive_contributors=positives,
                negative_contributors=negatives,
                summary=summary,
                overall_explanation=overall_explanation,
                base_value=base_value,
                total_impact=total_impact,
                metadata=metadata,
            )

            logger.info(
                "SHAP explanation completed | model=%s | features=%d | top_n=%d | duration=%.2fms",
                estimator.__class__.__name__,
                len(names),
                top_n,
                (time.perf_counter() - start_time) * 1000
            )
            return explanation
            
        except Exception as exc:
            logger.exception("Unexpected exception during SHAP explanation generation: %s", exc)
            raise ShapExplainerError(f"Explanation failed: {exc}") from exc

    def clear_cache(self) -> None:
        self._explainer_cache.clear()
        logger.info("SHAP explainer cache cleared.")

    @property
    def cache_size(self) -> int:
        return len(self._explainer_cache)

    def _load_shap(self) -> Any:
        if self._shap is not None:
            return self._shap
        try:
            import shap  # type: ignore
        except ImportError as exc:
            raise ShapExplainerError(
                "SHAP is not installed. Install the 'shap' package to enable explanations."
            ) from exc
        self._shap = shap
        return shap

    def _get_explainer(self, estimator: Any) -> Any:
        cache_key = id(estimator)
        if cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        shap = self._load_shap()
        try:
            explainer = shap.TreeExplainer(estimator)
            method = "TreeExplainer"
        except Exception as tree_exc:
            logger.warning("TreeExplainer unavailable, falling back to shap.Explainer: %s", tree_exc)
            try:
                explainer = shap.Explainer(estimator)
                method = "Explainer"
            except Exception as exc:
                raise ShapExplainerError(f"Unable to create SHAP explainer: {exc}") from exc

        self._explainer_cache[cache_key] = explainer
        logger.info("Cached SHAP %s for model=%s", method, estimator.__class__.__name__)
        return explainer

    def _build_frame(
        self,
        features: dict | pd.DataFrame,
        feature_columns: Optional[List[str]],
    ) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, dict):
            frame = pd.DataFrame([features])
        else:
            raise ShapExplainerError("Features must be provided as a dict or pandas DataFrame.")

        if frame.empty:
            raise ShapExplainerError("Cannot explain an empty feature frame.")
        if len(frame) != 1:
            raise ShapExplainerError("SHAP explanation expects exactly one feature row.")

        if feature_columns:
            missing = [col for col in feature_columns if col not in frame.columns]
            if missing:
                raise ShapExplainerError(f"Missing feature columns for explanation: {missing}")
            frame = frame[feature_columns]

        try:
            return frame.astype(float)
        except (TypeError, ValueError) as exc:
            raise ShapExplainerError(f"Features must be numeric for SHAP explanation: {exc}") from exc

    def _prepare_model_input(self, model: Any, frame: pd.DataFrame) -> Tuple[Any, pd.DataFrame, List[str]]:
        if not hasattr(model, "steps"):
            return model, frame, list(frame.columns)

        estimator = model.steps[-1][1]
        if len(model.steps) == 1:
            return estimator, frame, list(frame.columns)

        try:
            transformed = model[:-1].transform(frame)
        except Exception as exc:
            raise ShapExplainerError(f"Unable to transform features for SHAP explanation: {exc}") from exc

        transformed = np.asarray(transformed)
        names = self._transformed_feature_names(model, frame.columns, transformed.shape[1])
        return estimator, pd.DataFrame(transformed, columns=names), names

    def _transformed_feature_names(self, pipeline: Any, fallback: Any, width: int) -> List[str]:
        try:
            names = list(pipeline[:-1].get_feature_names_out(fallback))
        except Exception:
            names = list(fallback) if len(fallback) == width else []
        return names if len(names) == width else [f"feature_{idx}" for idx in range(width)]

    def _legacy_shap_values(self, explainer: Any, frame: pd.DataFrame) -> Any:
        try:
            values = explainer.shap_values(frame)
        except Exception as exc:
            raise ShapExplainerError(f"Unable to calculate SHAP values: {exc}") from exc

        class LegacyOutput:
            pass

        output = LegacyOutput()
        output.values = values
        output.base_values = getattr(explainer, "expected_value", None)
        return output

    def _extract_values(self, shap_output: Any) -> np.ndarray:
        values = np.asarray(shap_output.values)
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        elif values.ndim == 1:
            values = values
        else:
            raise ShapExplainerError(f"Unexpected SHAP value shape: {values.shape}")
        return values.astype(float)

    def _extract_base_value(self, shap_output: Any) -> Optional[float]:
        base_values = getattr(shap_output, "base_values", None)
        if base_values is None:
            return None
        values = np.asarray(base_values).reshape(-1)
        if values.size == 0:
            return None
        return round(float(values[-1]), 6)

    def _rank_contributions(
        self,
        shap_values: np.ndarray,
        row: pd.Series,
        feature_names: List[str],
    ) -> List[FeatureContribution]:
        if len(shap_values) != len(feature_names):
            raise ShapExplainerError(
                f"SHAP value count {len(shap_values)} does not match feature count {len(feature_names)}."
            )

        contributions = []
        for name, shap_val in zip(feature_names, shap_values):
            meta = FEATURE_METADATA.get(name, {})
            direction = "positive" if shap_val > 0 else "negative" if shap_val < 0 else "neutral"
            
            human_name = meta.get("human_name", name)
            if direction == "negative" and "negative_name" in meta:
                human_name = meta["negative_name"]

            contributions.append(
                FeatureContribution(
                    feature=name,
                    human_name=human_name,
                    description=meta.get("description", f"Factor related to {name.lower()}."),
                    value=round(float(row[name]), 6),
                    shap_value=round(float(shap_val), 6),
                    direction=direction,
                    impact=self._impact_label(shap_val),
                    reason=meta.get("positive_reason") if direction == "positive" else meta.get("negative_reason", "This factor slightly impacts the prediction."),
                    category=meta.get("category", "General"),
                    importance=meta.get("importance", "Informational"),
                    recommendation=meta.get("recommendation") if direction == "positive" else None
                )
            )

        return sorted(contributions, key=lambda item: abs(item.shap_value), reverse=True)

    def _impact_label(self, shap_value: float) -> str:
        magnitude = abs(float(shap_value))
        if magnitude >= 0.10:
            return "Strong Impact"
        if magnitude >= 0.03:
            return "Moderate Impact"
        if magnitude > 0:
            return "Small Impact"
        return "Neutral"

    def _summarize(
        self,
        positives: List[FeatureContribution],
        negatives: List[FeatureContribution],
    ) -> str:
        if not positives and not negatives:
            return "No meaningful feature impact was detected for this prediction."

        parts: List[str] = []
        if positives:
            names = ", ".join(item.human_name.lower() for item in positives[:3])
            # Replaces the last comma with 'and' for natural reading
            if ", " in names:
                names = " and ".join(names.rsplit(", ", 1))
            parts.append(f"The model predicts elevated diabetes risk primarily because of {names}.")
            
        if negatives:
            names = ", ".join(item.human_name.lower() for item in negatives[:3])
            if ", " in names:
                names = " and ".join(names.rsplit(", ", 1))
            if positives:
                parts.append(f"However, {names} slightly reduced the overall predicted risk.")
            else:
                parts.append(f"{names.capitalize()} reduced the overall predicted risk.")
                
        return " ".join(parts)

    def _build_overall_explanation(
        self, 
        positives: List[FeatureContribution], 
        negatives: List[FeatureContribution]
    ) -> str:
        if not positives and not negatives:
            return "The model found no significant factors influencing this prediction."

        lines = []
        if positives:
            lines.append("The prediction is mainly influenced by these major clinical factors:")
            for p in positives[:3]:
                lines.append(f"• {p.human_name}")
            lines.append("These factors contributed most to increasing the predicted diabetes risk.\n")
            
        if negatives:
            lines.append("Protective lifestyle and clinical factors slightly reduced the overall risk:")
            for n in negatives[:3]:
                lines.append(f"• {n.human_name}")

        return "\n".join(lines).strip()

    def _build_metadata(
        self, 
        estimator: Any, 
        explain_frame: pd.DataFrame, 
        top_n: int,
        contributions: List[FeatureContribution],
        positives: List[FeatureContribution],
        negatives: List[FeatureContribution],
        total_impact: float,
        computation_time_ms: float
    ) -> dict:
        
        avg_magnitude = float(np.mean([abs(c.shap_value) for c in contributions])) if contributions else 0.0
        
        explainer_type = "TreeExplainer" if "Tree" in estimator.__class__.__name__ else "Explainer"
        # Determine confidence of explanation based on the presence of dominant features
        explanation_confidence = "High" if len(positives) > 0 and positives[0].shap_value > 0.1 else "Moderate"

        return {
            "model_type": estimator.__class__.__name__,
            "explainer_type": explainer_type,
            "feature_count": int(explain_frame.shape[1]),
            "positive_contributor_count": len([c for c in contributions if c.direction == "positive"]),
            "negative_contributor_count": len([c for c in contributions if c.direction == "negative"]),
            "top_contributor": positives[0].human_name if positives else None,
            "top_protective_factor": negatives[0].human_name if negatives else None,
            "average_shap_magnitude": round(avg_magnitude, 6),
            "total_shap_impact": round(total_impact, 6),
            "explanation_confidence": explanation_confidence,
            "top_n_requested": top_n,
            "cached_explainers": self.cache_size,
            "explained_at": datetime.now(timezone.utc).isoformat(),
            "explanation_method": "shap",
            "computation_time_ms": round(computation_time_ms, 2)
        }


shap_explainer = ShapExplainer()
