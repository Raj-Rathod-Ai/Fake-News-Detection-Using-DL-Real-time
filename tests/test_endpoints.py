"""
TruthLens Endpoint Test Suite
Verifies all public routes, input validations, error states, and live API grounding.
"""

import unittest
import json
from app import app, init_db

class TestTruthLensEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = app.test_client()

    def test_01_home_ui(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("TruthLens", resp.data.decode("utf-8"))

    def test_02_health_checks(self):
        resp1 = self.client.get("/health")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp1.get_json().get("status"), "healthy")

        resp2 = self.client.get("/api/health")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.get_json().get("status"), "healthy")

    def test_03_api_news_headlines(self):
        resp = self.client.get("/api/news")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("articles", data)
        self.assertIsInstance(data["articles"], list)
        self.assertGreater(len(data["articles"]), 0)

    def test_04_api_news_category_filter(self):
        resp = self.client.get("/api/news?category=technology")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("articles", data)

    def test_05_api_markets(self):
        resp = self.client.get("/api/markets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(isinstance(data, (dict, list)))

    def test_06_api_cricket(self):
        resp = self.client.get("/api/cricket")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("typeMatches", data)

    def test_07_api_weather_default_and_params(self):
        resp_default = self.client.get("/api/weather")
        self.assertEqual(resp_default.status_code, 200)
        self.assertIn("current", resp_default.get_json())

        resp_city = self.client.get("/api/weather?city=Mumbai")
        self.assertEqual(resp_city.status_code, 200)
        self.assertIn("current", resp_city.get_json())

    def test_08_api_chat_valid_and_validation(self):
        # Valid message
        resp = self.client.post("/api/chat", json={"message": "What is fake news?"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("reply", resp.get_json())

        # Empty message (400 validation)
        resp_empty = self.client.post("/api/chat", json={"message": ""})
        self.assertEqual(resp_empty.status_code, 400)

        # Missing body (400 validation)
        resp_none = self.client.post("/api/chat", json={})
        self.assertEqual(resp_none.status_code, 400)

    def test_09_api_ai_scan_valid_and_validation(self):
        # Valid claim
        resp = self.client.post("/api/ai-scan", json={"text": "ISRO announces space research advancements in satellite communication."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("verdict", data)
        self.assertIn("confidence", data)
        self.assertIn("model", data)

        # Short text (400 validation)
        resp_short = self.client.post("/api/ai-scan", json={"text": "Hey"})
        self.assertEqual(resp_short.status_code, 400)

        # Missing body (400 validation)
        resp_empty = self.client.post("/api/ai-scan", json={})
        self.assertEqual(resp_empty.status_code, 400)

    def test_10_api_scan_history(self):
        resp = self.client.get("/api/scan-history")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("history", data)
        self.assertIsInstance(data["history"], list)

    def test_11_api_feedback(self):
        # Valid feedback
        resp = self.client.post("/api/feedback", json={"message": "Clean UI and great speed", "rating": 5})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get("success"))

        # Empty feedback (400 validation)
        resp_bad = self.client.post("/api/feedback", json={"message": ""})
        self.assertEqual(resp_bad.status_code, 400)

    def test_12_http_method_security(self):
        # POST to GET-only route -> 405
        self.assertEqual(self.client.post("/health").status_code, 405)
        # GET to POST-only route -> 405
        self.assertEqual(self.client.get("/api/ai-scan").status_code, 405)
        self.assertEqual(self.client.get("/api/feedback").status_code, 405)
        # Non-existent endpoint -> 404
        self.assertEqual(self.client.get("/api/undefined-endpoint-xyz").status_code, 404)

if __name__ == "__main__":
    unittest.main()
