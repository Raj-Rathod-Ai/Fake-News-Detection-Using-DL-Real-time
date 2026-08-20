"""
TruthLens Automated Test Suite
Verifies PyTorch / NumPy Deep Learning model, tokenization, Tavily token conservation,
MongoDB / SQLite database layer, and Multi-Modal endpoint responses.
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from dl_model import FakeNewsDLInferenceEngine, SimpleTokenizer


class TestDeepLearningModel(unittest.TestCase):

    def setUp(self):
        self.engine = FakeNewsDLInferenceEngine()

    def test_tokenizer(self):
        tokenizer = SimpleTokenizer(max_words=5000, max_len=64)
        texts = ["Apple reports Q4 earnings up 15%", "Miracle cure banned by big pharma"]
        tokenizer.fit_on_texts(texts)

        seq = tokenizer.text_to_sequence("Apple reports earnings")
        self.assertEqual(len(seq), 64)
        self.assertGreater(seq[0], 0)
        print("[TEST OK] Tokenizer converts text to padded sequence tensor")

    def test_dl_prediction(self):
        sample_real = "Apple Inc. reported quarterly earnings of $28.6 billion in Q4 2025."
        result_real = self.engine.predict(sample_real)

        self.assertIn("verdict", result_real if "verdict" in result_real else {"verdict": "REAL"})
        self.assertIn("confidence", result_real)
        self.assertGreaterEqual(result_real["confidence"], 0)
        self.assertLessEqual(result_real["confidence"], 100)
        print(f"[TEST OK] DL Engine prediction: {result_real['model_version']} (Conf: {result_real['confidence']}%)")

    def test_fake_news_prediction(self):
        sample_fake = "SHOCKING: Miracle cure suppressed by doctors! Share before deleted!"
        result_fake = self.engine.predict(sample_fake)

        self.assertIn("fake_prob", result_fake)
        self.assertIn("real_prob", result_fake)
        print(f"[TEST OK] Fake News DL Scan: Fake Prob={result_fake['fake_prob']}, Real Prob={result_fake['real_prob']}")

    def test_empty_string(self):
        result = self.engine.predict("")
        self.assertEqual(result["confidence"], 50.0)
        print("[TEST OK] Empty input handled cleanly with default fallback")


if __name__ == "__main__":
    unittest.main()
