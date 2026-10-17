import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Index, CheckConstraint
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
from .database import db


# ─── User Model ───────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"

    __table_args__ = (
        CheckConstraint("age >= 0 AND age <= 120", name="chk_user_age"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(30),  unique=True, nullable=False)
    email         = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name     = Column(String(120), nullable=True)
    age           = Column(Integer,     nullable=True)
    gender        = Column(String(10),  nullable=True)
    is_active     = Column(Boolean,     nullable=False, default=True)
    last_login    = Column(DateTime(timezone=True), nullable=True)
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    # ─── Relationships ────────────────────────────────────────────────────────
    # passive_deletes=True prevents loading all child records into memory when deleting a User, delegating to the DB's CASCADE
    predictions     = relationship("Prediction",     back_populates="user",
                                   cascade="all, delete-orphan", passive_deletes=True, lazy="dynamic")
    token_blocklist = relationship("TokenBlocklist", back_populates="user",
                                   cascade="all, delete-orphan", passive_deletes=True, lazy="dynamic")

    # ─── Validators ───────────────────────────────────────────────────────────
    @validates("email")
    def validate_email(self, key, value):
        if not value or "@" not in value:
            raise ValueError("Invalid email address.")
        return value.strip().lower()

    @validates("gender")
    def validate_gender(self, key, value):
        if value and value not in ("Male", "Female", "Other"):
            raise ValueError("Gender must be Male, Female, or Other.")
        return value

    @validates("age")
    def validate_age(self, key, value):
        if value is not None and not (1 <= int(value) <= 120):
            raise ValueError("Age must be between 1 and 120.")
        return value

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def record_login(self):
        self.last_login = datetime.now(timezone.utc)

    def to_dict(self, minimal: bool = False) -> dict:
        """
        Serializes the User object safely. Never exposes password_hash.
        """
        base = {
            "id":         self.id,
            "username":   self.username,
            "is_active":  self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if not minimal:
            base.update({
                "email":      self.email,
                "full_name":  self.full_name,
                "age":        self.age,
                "gender":     self.gender,
                "last_login": self.last_login.isoformat() if self.last_login else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            })
        return base

    def __repr__(self):
        return f"<User id={self.id} username={self.username} active={self.is_active}>"


# ─── Prediction Model ─────────────────────────────────────────────────────────

class Prediction(db.Model):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_pred_user_created", "user_id", "created_at"),
        Index("ix_pred_risk_level",   "risk_level"),
        Index("ix_pred_category",     "diabetes_category"),
        CheckConstraint("probability >= 0.0 AND probability <= 1.0", name="chk_pred_probability"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="chk_pred_confidence"),
        CheckConstraint("risk_score >= 0.0", name="chk_pred_risk_score"),
    )

    id                = Column(Integer, primary_key=True, autoincrement=True)
    user_id           = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    prediction        = Column(Integer, nullable=False)
    probability       = Column(Float,   nullable=False)
    confidence        = Column(Float,   nullable=False)
    risk_score        = Column(Float,   nullable=False)
    risk_level        = Column(String(10),  nullable=False)
    diabetes_category = Column(String(20),  nullable=False)
    prediction_label  = Column(String(120), nullable=False)
    
    # JSON Storage Optimization Note: 
    # For full JSON querying, these should migrate to db.JSON or db.JSONB. 
    # Kept as Text here to prevent breaking existing schema compatibility without a migration.
    flags             = Column(Text, nullable=True)
    recommendations   = Column(Text, nullable=True)
    
    high_bp       = Column(Integer, nullable=True)
    high_chol     = Column(Integer, nullable=True)
    bmi           = Column(Float, nullable=True)
    smoker        = Column(Integer, nullable=True)
    fruits        = Column(Integer, nullable=True)
    gen_hlth      = Column(Integer, nullable=True)
    ment_hlth     = Column(Integer, nullable=True)
    sex           = Column(Integer, nullable=True)
    age           = Column(Integer, nullable=True)
    
    # Model Versioning (Nullable for backward compatibility)
    model_version                 = Column(String(50), nullable=True)
    risk_engine_version           = Column(String(50), nullable=True)
    recommendation_engine_version = Column(String(50), nullable=True)
    
    created_at        = Column(DateTime(timezone=True),
                               default=lambda: datetime.now(timezone.utc))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ─── Relationship ─────────────────────────────────────────────────────────
    user = relationship("User", back_populates="predictions")

    # ─── Validators ───────────────────────────────────────────────────────────
    @validates("probability", "confidence")
    def validate_probabilities(self, key, value):
        if value is None or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{key} must be between 0.0 and 1.0")
        return float(value)

    @validates("risk_score")
    def validate_risk_score(self, key, value):
        if value is None or float(value) < 0.0:
            raise ValueError("risk_score must be non-negative")
        return float(value)

    # ─── Hybrid Properties ────────────────────────────────────────────────────
    @hybrid_property
    def flags_list(self) -> list:
        if not self.flags:
            return []
        # Support both JSON strings and legacy newline-separated text dynamically
        try:
            parsed = json.loads(self.flags)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [f.strip() for f in self.flags.split("\n") if f.strip()]

    @hybrid_property
    def recommendations_list(self) -> list:
        if not self.recommendations:
            return []
        try:
            parsed = json.loads(self.recommendations)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [r.strip() for r in self.recommendations.split("\n") if r.strip()]

    # ─── Factory Method ───────────────────────────────────────────────────────
    @classmethod
    def from_assessment(cls, user_id: int, assessment: dict, features: dict) -> "Prediction":
        meta = assessment.get("metadata", {})
        return cls(
            user_id           = user_id,
            prediction        = assessment["prediction"],
            probability       = assessment["probability"],
            confidence        = assessment["confidence"],
            risk_score        = assessment["risk_score"],
            risk_level        = assessment["risk_level"],
            diabetes_category = assessment["diabetes_category"],
            prediction_label  = assessment["prediction_label"],
            
            # Write as JSON string for future JSONB migration readiness, 
            # while falling back to text parsing if needed
            flags             = json.dumps(assessment.get("flags", [])),
            recommendations   = json.dumps(assessment.get("recommendations", [])),
            
            high_bp       = features.get("HighBP"),
            high_chol     = features.get("HighChol"),
            bmi           = features.get("BMI"),
            smoker        = features.get("Smoker"),
            fruits        = features.get("Fruits"),
            gen_hlth      = features.get("GenHlth"),
            ment_hlth     = features.get("MentHlth"),
            sex           = features.get("Sex"),
            age           = features.get("Age"),
            
            model_version                 = meta.get("model_version"),
            risk_engine_version           = meta.get("risk_engine_version"),
            recommendation_engine_version = meta.get("recommendation_version"),
        )

    def to_dict(self, minimal: bool = False, include_features: bool = True) -> dict:
        data = {
            "id":                self.id,
            "user_id":           self.user_id,
            "prediction":        self.prediction,
            "probability":       self.probability,
            "risk_level":        self.risk_level,
            "diabetes_category": self.diabetes_category,
            "created_at":        self.created_at.isoformat() if self.created_at else None,
        }
        
        if not minimal:
            data.update({
                "confidence":        self.confidence,
                "risk_score":        self.risk_score,
                "prediction_label":  self.prediction_label,
                "flags":             self.flags_list,
                "recommendations":   self.recommendations_list,
                "model_version":     self.model_version,
                "risk_engine_version": self.risk_engine_version,
                "recommendation_engine_version": self.recommendation_engine_version,
            })
            
        if include_features and not minimal:
            data["input_features"] = {
                 "high_bp": self.high_bp,
                 "high_chol": self.high_chol,
                 "bmi": self.bmi,
                 "smoker": self.smoker,
                 "fruits": self.fruits,
                 "gen_hlth": self.gen_hlth,
                 "ment_hlth": self.ment_hlth,
                 "sex": self.sex,
                 "age": self.age,
            }
        return data

    def __repr__(self):
        return f"<Prediction id={self.id} user_id={self.user_id} prob={self.probability:.2f}>"


# ─── Token Blocklist Model ────────────────────────────────────────────────────

class TokenBlocklist(db.Model):
    __tablename__ = "token_blocklist"

    __table_args__ = (
        Index("ix_blocklist_user_id", "user_id"),
    )

    id         = Column(Integer,    primary_key=True, autoincrement=True)
    jti        = Column(String(64), nullable=False, unique=True)
    token_type = Column(String(10), nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    revoked_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    # ─── Relationship ─────────────────────────────────────────────────────────
    user = relationship("User", back_populates="token_blocklist")

    # ─── Class Methods ────────────────────────────────────────────────────────
    @classmethod
    def is_revoked(cls, jti: str) -> bool:
        return db.session.query(cls.query.filter_by(jti=jti).exists()).scalar()

    @classmethod
    def revoke(cls, jti: str, token_type: str, user_id: int) -> "TokenBlocklist":
        entry = cls(jti=jti, token_type=token_type, user_id=user_id)  # type: ignore
        db.session.add(entry)
        db.session.commit()
        return entry

    def __repr__(self):
        return f"<TokenBlocklist jti={self.jti} type={self.token_type}>"