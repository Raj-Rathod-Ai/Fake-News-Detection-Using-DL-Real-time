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
        app.config['TESTING'] = True
        app.testing = True
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
            json={"scans": [{"id": f"guest_scan_{uuid.uuid4().hex[:8]}", "text": "Sample claim test", "title": "Test Claim", "verdict": "REAL", "confidence": 98.0}]}
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

    def test_18_chatbot_greeting_and_assistant_persona(self):
        # 1. Test greeting "hi"
        resp_hi = self.client.post("/api/chat", json={"message": "hi"})
        self.assertEqual(resp_hi.status_code, 200)
        reply_hi = resp_hi.get_json().get("reply", "")
        self.assertIn("Hi! How can I assist", reply_hi)

        # 2. Test greeting "hello"
        resp_hello = self.client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(resp_hello.status_code, 200)
        self.assertIn("Hi! How can I assist", resp_hello.get_json().get("reply", ""))

        # 3. Test "who are you"
        resp_who = self.client.post("/api/chat", json={"message": "who are you"})
        self.assertEqual(resp_who.status_code, 200)
        self.assertIn("TruthLens AI", resp_who.get_json().get("reply", ""))

        # 4. Test validation error on empty message
        resp_empty = self.client.post("/api/chat", json={"message": ""})
        self.assertEqual(resp_empty.status_code, 400)

    def test_19_existing_account_signup_no_error_and_bcrypt(self):
        email = f"bcrypt_tester_{uuid.uuid4().hex[:8]}@truthlens.ai"
        # 1. First signup sends 7-digit OTP
        r1 = self.client.post("/api/auth/signup", json={"email": email, "password": "password_v1_123"})
        self.assertEqual(r1.status_code, 200)
        self.assertTrue(r1.get_json()["success"])
        self.assertTrue(r1.get_json().get("requires_otp"))
        otp1 = r1.get_json().get("dev_otp")
        self.assertEqual(len(str(otp1)), 7)

        # Verify 7-digit OTP to activate
        v1 = self.client.post("/api/auth/verify-otp", json={"email": email, "otp": otp1})
        self.assertEqual(v1.status_code, 200)
        t1 = v1.get_json()["token"]
        self.assertIsNotNone(t1)

        # 2. Second signup with SAME email (must NOT return 400 error! Sends fresh 7-digit OTP)
        r2 = self.client.post("/api/auth/signup", json={"email": email, "password": "password_v2_456"})
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.get_json()["success"])
        self.assertTrue(r2.get_json().get("requires_otp"))
        otp2 = r2.get_json().get("dev_otp")
        self.assertEqual(len(str(otp2)), 7)

        # Verify second OTP
        v2 = self.client.post("/api/auth/verify-otp", json={"email": email, "otp": otp2})
        self.assertEqual(v2.status_code, 200)
        t2 = v2.get_json()["token"]
        self.assertIsNotNone(t2)

        # 3. Check that /api/auth/me returns 24-hr refreshed new_token, used, and remaining
        r_me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {t2}"})
        self.assertEqual(r_me.status_code, 200)
        data_me = r_me.get_json()
        self.assertTrue(data_me["is_authenticated"])
        self.assertIn("new_token", data_me)
        self.assertIn("used", data_me)
        self.assertIn("remaining", data_me)
        self.assertEqual(data_me["quota_cycle"], "weekly")

        # 4. Login with updated password succeeds
        r_login = self.client.post("/api/auth/login", json={"email": email, "password": "password_v2_456"})
        self.assertEqual(r_login.status_code, 200)
        self.assertTrue(r_login.get_json()["success"])

    def test_20_cricket_only_india_live(self):
        resp = self.client.get("/api/cricket")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        tm = data.get("typeMatches", [])
        self.assertGreater(len(tm), 0)
        for t in tm:
            for sm in t.get("seriesMatches", []):
                for m in sm.get("seriesAdWrapper", {}).get("matches", []):
                    mi = m.get("matchInfo", {})
                    t1 = mi.get("team1", {}).get("teamName", "")
                    t2 = mi.get("team2", {}).get("teamName", "")
                    s1 = mi.get("team1", {}).get("teamSName", "")
                    s2 = mi.get("team2", {}).get("teamSName", "")
                    is_ind = any(k in (t1 + " " + t2 + " " + s1 + " " + s2).lower() for k in ["india", "ind", "roi", "rest of india"])
                    self.assertTrue(is_ind, f"Match {t1} vs {t2} is not an India match!")
                    state = mi.get("state", "")
                    status = (mi.get("status") or "").lower()
                    self.assertTrue(
                        state in ["In Progress", "live", "Stumps", "Complete", "Finished"] or "won" in status,
                        f"Unexpected match state: {state}, status: {status}"
                    )

    def test_21_delete_profile_grace_period_and_recovery(self):
        email = f"del_user_{uuid.uuid4().hex[:8]}@truthlens.ai"
        # 1. Signup and verify
        r_signup = self.client.post("/api/auth/signup", json={"email": email, "password": "del_pass_12345"})
        self.assertEqual(r_signup.status_code, 200)
        otp = r_signup.get_json().get("dev_otp")
        self.assertEqual(len(str(otp)), 7)

        r_v = self.client.post("/api/auth/verify-otp", json={"email": email, "otp": otp})
        self.assertEqual(r_v.status_code, 200)
        token = r_v.get_json().get("token")

        # 2. Schedule profile deletion
        r_del = self.client.post("/api/auth/delete-profile", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r_del.status_code, 200)
        self.assertTrue(r_del.get_json()["success"])
        self.assertEqual(r_del.get_json()["grace_hours"], 24)

        # 3. Recover profile within 24 hours by signing in
        r_rec = self.client.post("/api/auth/login", json={"email": email, "password": "del_pass_12345"})
        self.assertEqual(r_rec.status_code, 200)
        self.assertTrue(r_rec.get_json()["recovered"])
        self.assertIn("recovered", r_rec.get_json()["message"].lower())

    def test_22_login_resolves_mongo_user_after_cache_wipe(self):
        import sqlite3
        from app import DB_PATH
        # Simulate local SQLite being wiped or missing the user while MongoDB Atlas has the record
        con = sqlite3.connect(DB_PATH)
        con.execute("DELETE FROM users WHERE email = 'dhruv.aparnathi.1@gmail.com'")
        con.commit()
        con.close()

        # Login must automatically resolve user from MongoDB Atlas without 'not found' error!
        r_login = self.client.post("/api/auth/login", json={"email": "dhruv.aparnathi.1@gmail.com", "password": "dhruv@2006"})
        self.assertEqual(r_login.status_code, 200)
        data = r_login.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["user"]["email"], "dhruv.aparnathi.1@gmail.com")
        self.assertEqual(data["user"]["name"], "Dhruv")
        self.assertEqual(data["quota"]["limit"], 50)
        self.assertIsNotNone(data.get("token"))

    def test_23_forgot_and_reset_password_flow(self):
        email = f"reset_test_{uuid.uuid4().hex[:8]}@truthlens.ai"
        init_pass = "initial_pass_123"
        new_pass = "updated_pass_456"

        # 1. Sign up and verify account
        r_sign = self.client.post("/api/auth/signup", json={"email": email, "password": init_pass, "name": "Reset Tester"})
        self.assertEqual(r_sign.status_code, 200)
        otp = r_sign.get_json().get("dev_otp")
        r_ver = self.client.post("/api/auth/verify-otp", json={"email": email, "otp": otp})
        self.assertEqual(r_ver.status_code, 200)

        # 2. Trigger forgot password
        r_forgot = self.client.post("/api/auth/forgot-password", json={"email": email})
        self.assertEqual(r_forgot.status_code, 200)
        f_data = r_forgot.get_json()
        self.assertTrue(f_data.get("success"))
        reset_otp = f_data.get("dev_otp")
        self.assertEqual(len(str(reset_otp)), 7)

        # 3. Test invalid OTP rejection
        r_bad_otp = self.client.post("/api/auth/reset-password", json={"email": email, "otp": "0000000", "new_password": new_pass})
        self.assertEqual(r_bad_otp.status_code, 400)

        # 4. Successfully reset password
        r_reset = self.client.post("/api/auth/reset-password", json={"email": email, "otp": reset_otp, "new_password": new_pass})
        self.assertEqual(r_reset.status_code, 200)
        reset_data = r_reset.get_json()
        self.assertTrue(reset_data.get("success"))
        self.assertIsNotNone(reset_data.get("token"))
        self.assertEqual(reset_data["user"]["email"], email)

        # 5. Old password must now fail
        r_old_login = self.client.post("/api/auth/login", json={"email": email, "password": init_pass})
        self.assertEqual(r_old_login.status_code, 401)

        # 6. New password must succeed
        r_new_login = self.client.post("/api/auth/login", json={"email": email, "password": new_pass})
        self.assertEqual(r_new_login.status_code, 200)
        self.assertTrue(r_new_login.get_json().get("success"))

    def test_24_token_expiration_limit(self):
        from app import auth_serializer
        import time
        # Generate an expired token (simulate > 1 day expiration by using max_age)
        payload = {"user_id": "test_user_id", "email": "test@truthlens.ai", "role": "user"}
        token = auth_serializer.dumps(payload)
        
        # Immediate /me with valid token -> passes
        r_valid = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r_valid.status_code, 200)

        # Expired token (> 86400s) -> loads(token, max_age=86400) throws SignatureExpired -> falls back to guest mode
        # We test this by verifying loads raises SignatureExpired with max_age=-1
        from itsdangerous import SignatureExpired
        with self.assertRaises(SignatureExpired):
            auth_serializer.loads(token, max_age=-1)

if __name__ == "__main__":
    unittest.main()

