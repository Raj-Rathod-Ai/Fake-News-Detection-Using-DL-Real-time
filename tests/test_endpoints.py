"""
TruthLens Endpoint Test Suite
Verifies all public routes, input validations, error states, and live API grounding.
"""

import unittest
import json
import uuid
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

    def test_13_auth_flow_and_quota_upgrade(self):
        # 1. Guest quota starts at 5
        resp_guest = self.client.get("/api/auth/me")
        self.assertEqual(resp_guest.status_code, 200)
        self.assertEqual(resp_guest.get_json()["limit"], 5)
        self.assertFalse(resp_guest.get_json()["is_authenticated"])

        # 2. Signup
        test_email = f"unit_tester_{uuid.uuid4().hex[:8]}@truthlens.ai"
        resp_signup = self.client.post("/api/auth/signup", json={"email": test_email, "password": "secure_password_123"})
        self.assertEqual(resp_signup.status_code, 200)
        otp = resp_signup.get_json().get("dev_otp")
        self.assertIsNotNone(otp)

        # 3. Verify OTP
        resp_verify = self.client.post("/api/auth/verify-otp", json={"email": test_email, "otp": otp})
        self.assertEqual(resp_verify.status_code, 200)
        token = resp_verify.get_json().get("token")
        self.assertIsNotNone(token)
        self.assertEqual(resp_verify.get_json()["user"]["limit"], 50)

        # 4. Authenticated user quota is 50
        resp_user = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp_user.status_code, 200)
        self.assertEqual(resp_user.get_json()["limit"], 50)
        self.assertTrue(resp_user.get_json()["is_authenticated"])

        # 5. Sync history
        resp_sync = self.client.post(
            "/api/auth/sync-history",
            headers={"Authorization": f"Bearer {token}"},
            json={"scans": [{"id": "guest_scan_1", "text": "Sample claim test", "title": "Test Claim", "verdict": "REAL", "confidence": 98.0}]}
        )
        self.assertEqual(resp_sync.status_code, 200)
        self.assertEqual(resp_sync.get_json()["synced_count"], 1)

        # 6. Logout
        resp_logout = self.client.post("/api/auth/logout")
        self.assertEqual(resp_logout.status_code, 200)

    def test_14_cricket_live_details(self):
        resp = self.client.get("/api/cricket")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        tm = data.get("typeMatches", [])
        self.assertGreater(len(tm), 0)
        first_match = tm[0]["seriesMatches"][0]["seriesAdWrapper"]["matches"][0]
        self.assertIn("liveDetails", first_match)
        ld = first_match["liveDetails"]
        self.assertIn("batters", ld)
        self.assertIn("bowler", ld)
        self.assertIn("recent_balls", ld)
        self.assertIn("currentBatters", ld)
        self.assertIn("currentBowler", ld)
        self.assertIn("lastWicket", ld)

    def test_15_admin_accounts_and_unlimited_quota(self):
        # Admin 1 Login
        resp1 = self.client.post("/api/auth/login", json={"email": "kevalpiparotar4@gmail.com", "password": "keval@2006"})
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.get_json()
        self.assertTrue(data1.get("success"))
        self.assertTrue(data1["user"]["is_admin"])
        self.assertEqual(data1["user"]["limit"], 999999)
        token1 = data1.get("token")

        # Admin 1 /me check
        resp1_me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token1}"})
        self.assertEqual(resp1_me.status_code, 200)
        self.assertEqual(resp1_me.get_json()["limit"], 999999)
        self.assertTrue(resp1_me.get_json()["unlimited"])

        # Admin 2 Login
        resp2 = self.client.post("/api/auth/login", json={"email": "rathodraj1504@gmail.com", "password": "raj@2006"})
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertTrue(data2.get("success"))
        self.assertTrue(data2["user"]["is_admin"])
        self.assertEqual(data2["user"]["limit"], 999999)
        token2 = data2.get("token")

        # Admin overview endpoint
        resp_overview = self.client.get("/api/admin/overview", headers={"Authorization": f"Bearer {token1}"})
        self.assertEqual(resp_overview.status_code, 200)
        self.assertIn("total_users", resp_overview.get_json())

        # Admin clear cache endpoint
        resp_cache = self.client.post("/api/admin/clear-cache", headers={"Authorization": f"Bearer {token1}"})
        self.assertEqual(resp_cache.status_code, 200)
        self.assertTrue(resp_cache.get_json().get("success"))

        # Admin reset quota endpoint
        resp_reset = self.client.post("/api/admin/reset-user-quota", headers={"Authorization": f"Bearer {token1}"}, json={"email": "test@example.com"})
        self.assertEqual(resp_reset.status_code, 200)
        self.assertTrue(resp_reset.get_json().get("success"))

    def test_16_security_canary_fuck_html(self):
        resp = self.client.get("/fuck.html")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("Security canary", resp.data.decode("utf-8"))

    def test_17_cricket_rich_player_details(self):
        resp = self.client.get("/api/cricket")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        tm = data.get("typeMatches", [])
        m = tm[0]["seriesMatches"][0]["seriesAdWrapper"]["matches"][0]
        ld = m["liveDetails"]
        # Batters check
        batters = ld["currentBatters"]
        self.assertGreater(len(batters), 0)
        self.assertIn("runs", batters[0])
        self.assertIn("balls", batters[0])
        self.assertIn("name", batters[0])
        # Bowler check
        bowler = ld["currentBowler"]
        self.assertIn("name", bowler)
        self.assertIn("wickets", bowler)
        self.assertIn("runs", bowler)

if __name__ == "__main__":
    unittest.main()

