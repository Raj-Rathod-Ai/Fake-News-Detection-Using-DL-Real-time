"""
TruthLens Automated Test Suite
Verifies Keras Deep Learning model, Tokenizer, Fallback Engine, and Inference Pipeline.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from dl_model import FakeNewsDLInferenceEngine


class TestDeepLearningModel(unittest.TestCase):

    def setUp(self):
        self.engine = FakeNewsDLInferenceEngine()

    def test_tokenizer_and_model_loading(self):
        self.assertIsNotNone(self.engine.tokenizer, "Tokenizer should be loaded from tokenizer.pkl")
        self.assertIsNotNone(self.engine.keras_model, "Keras model should be loaded from fake_news_detection_model.keras")
        self.assertTrue(self.engine.is_keras_active, "Keras engine should be active")
        print("[TEST OK] Keras Model and Tokenizer successfully loaded.")

    def test_dl_prediction_real_news(self):
        sample_real = "WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a major budget resolution after an all-night debate."
        result_real = self.engine.predict(sample_real)

        self.assertIn("fake_prob", result_real)
        self.assertIn("real_prob", result_real)
        self.assertIn("confidence", result_real)
        self.assertGreaterEqual(result_real["real_prob"], 0.5, "Real news should have real_prob >= 0.5")
        self.assertFalse(result_real["is_fake"])
        print(f"[TEST OK] Real News DL Prediction: Real Prob={result_real['real_prob']}, Fake Prob={result_real['fake_prob']}")

    def test_fake_news_prediction(self):
        sample_fake = "SHOCKING VIDEO: Pope Francis endorses Donald Trump for President in secret Vatican meeting! Share before deleted!"
        result_fake = self.engine.predict(sample_fake)

        self.assertIn("fake_prob", result_fake)
        self.assertIn("real_prob", result_fake)
        self.assertIn("confidence", result_fake)
        print(f"[TEST OK] Fake News DL Scan: Fake Prob={result_fake['fake_prob']}, Real Prob={result_fake['real_prob']}")

    def test_empty_string(self):
        result = self.engine.predict("")
        self.assertEqual(result["confidence"], 50.0)
        self.assertEqual(result["fake_prob"], 0.5)
        self.assertEqual(result["real_prob"], 0.5)
        print("[TEST OK] Empty input handled cleanly with default fallback")


if __name__ == "__main__":
    unittest.main()
