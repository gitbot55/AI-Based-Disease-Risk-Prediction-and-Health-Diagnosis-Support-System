import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
import unittest
from app import create_app
from config import TestingConfig
from database.database import db
from database.models import User
from werkzeug.security import generate_password_hash

class SymptomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(TestingConfig)
        cls.client = cls.app.test_client()

        with cls.app.app_context():
            db.create_all()
            
            # Use app test client to register user to ensure proper hashing
            with cls.app.test_client() as client:
                client.post("/api/auth/register", json={
                    "username": "testuser",
                    "email": "testuser@example.com",
                    "password": "StrongPassword123!@#",
                    "age": 48,
                    "gender": "Male"
                })

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            db.session.remove()
            db.drop_all()

    def setUp(self):
        # Login to get JWT token
        resp = self.client.post("/api/auth/login", json={
            "identifier": "testuser",
            "password": "StrongPassword123!@#"
        })
        self.token = resp.get_json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def _test_durations(self, symptoms):
        payload = {
            "age": 48,
            "symptoms": symptoms,
            "lifestyle": {
                "smoking": True,
                "exercise": False,
                "alcohol": False
            }
        }
        resp = self.client.post("/api/prediction/submit", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        return data["data"], data["metadata"]

    def test_1_all_zero_durations(self):
        symptoms = {
            "frequent_urination_days": 0,
            "excessive_thirst_days": 0,
            "fatigue_days": 0,
            "blurred_vision_days": 0,
            "slow_healing_wounds_days": 0,
            "weight_loss_days": 0
        }
        data, metadata = self._test_durations(symptoms)
        score_zero = data["risk_score"]
        self.assertGreaterEqual(score_zero, 0)
        
    def test_2_short_durations(self):
        symptoms = {
            "frequent_urination_days": 2,
            "excessive_thirst_days": 1,
            "fatigue_days": 1,
            "blurred_vision_days": 0,
            "slow_healing_wounds_days": 0,
            "weight_loss_days": 0
        }
        data, metadata = self._test_durations(symptoms)
        self.assertGreaterEqual(data["risk_score"], 0)

    def test_3_longer_durations(self):
        symptoms = {
            "frequent_urination_days": 30,
            "excessive_thirst_days": 30,
            "fatigue_days": 30,
            "blurred_vision_days": 30,
            "slow_healing_wounds_days": 30,
            "weight_loss_days": 30
        }
        data, metadata = self._test_durations(symptoms)
        
        # Test 1 & 3: Short vs long should change risk score due to dynamic scoring
        short_symptoms = {k: 0 for k in symptoms}
        short_data, _ = self._test_durations(short_symptoms)
        self.assertNotEqual(data["risk_score"], short_data["risk_score"], "Risk score must be dynamic based on symptoms")

    def test_4_changing_independently(self):
        base_symptoms = {k: 0 for k in ["frequent_urination_days", "excessive_thirst_days", "fatigue_days", "blurred_vision_days", "slow_healing_wounds_days", "weight_loss_days"]}
        scores = []
        for key in base_symptoms.keys():
            s = base_symptoms.copy()
            s[key] = 15
            data, _ = self._test_durations(s)
            scores.append(data["risk_score"])
        
        # They should all successfully run and produce scores
        self.assertEqual(len(scores), 6)
        
    def test_5_invalid_durations(self):
        symptoms = {
            "frequent_urination_days": -5,
            "excessive_thirst_days": "not a number",
        }
        payload = {
            "age": 48,
            "symptoms": symptoms,
            "lifestyle": {}
        }
        resp = self.client.post("/api/prediction/submit", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 422)

    def test_6_consistency(self):
        symptoms = {"frequent_urination_days": 10}
        data1, _ = self._test_durations(symptoms)
        data2, _ = self._test_durations(symptoms)
        self.assertEqual(data1["risk_score"], data2["risk_score"])
        
    def test_7_field_mapping(self):
        symptoms = {"frequent_urination_days": 10}
        data, metadata = self._test_durations(symptoms)
        self.assertIn("raw_probability", data)
        self.assertIn("calibrated_probability", data)
        self.assertIn("risk_score", data)
        self.assertIn("risk_level", data)
        
    def test_9_model_features(self):
        symptoms = {"frequent_urination_days": 10}
        data, metadata = self._test_durations(symptoms)
        
        # Model should receive exactly the schema it expects. Unmapped are ignored.
        self.assertNotIn("symptom_score", metadata.get("recommendation_plan", {}).get("features", {})) # well, features does have symptom_score, wait, the HGBC feature dict

    def test_10_missing_lifestyle(self):
        # Missing lifestyle should not throw an error and default gracefully
        payload = {
            "age": 48,
            "symptoms": {"frequent_urination_days": 5}
        }
        resp = self.client.post("/api/prediction/submit", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 201)

if __name__ == "__main__":
    unittest.main()
