"""
TruthLens Deep Learning Core Engine
Architecture: Uses pre-trained Keras model (fake_real_news_detection_model.keras)
with tokenizer.pkl for news classification.
Fallback: Lightweight heuristic NumPy engine for low-memory environments.
"""

import os
import re
import json
import pickle
import numpy as np
from typing import Dict, List, Any

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

KERAS_MODEL_PATH = os.path.join(MODEL_DIR, 'fake_real_news_detection_model.keras')
TOKENIZER_PKL_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')

# Max sequence length for tokenizer padding
MAX_SEQ_LEN = 300

# ─────────────────────────────────────────────────────────────────────────────
# Optional Keras/TF Import
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    # Suppress TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Fallback Heuristic Engine (NumPy only, <10MB RAM)
# ─────────────────────────────────────────────────────────────────────────────
class HeuristicFallbackEngine:
    """Pure rule-based heuristic classifier as fallback when Keras is unavailable."""

    FAKE_WORDS = [
        'shocking', 'bombshell', 'exposed', 'coverup', 'alert', 'forward this',
        'wake up', 'share before deleted', 'banned video', 'hidden truth',
        'secret plan', 'you wont believe', 'mainstream media hiding',
        'deep state', 'new world order', 'illuminati', 'microchip implant',
        'depopulation agenda', 'chemtrail', 'flat earth', 'moon landing faked',
        'big pharma hiding', 'wake up sheeple', 'soros funded',
        'miracle cure', 'cures overnight', 'cures cancer', 'doctors furious',
        'doctors dont want', 'one simple trick', 'viral truth',
        'urgent alert', 'share now', 'insider reveals', 'whistleblower reveals',
        'leaked document proves', 'anonymous source confirms'
    ]

    REAL_WORDS = [
        'according to', 'sources confirm', 'official statement', 'press release',
        'government of india', 'ministry of', 'supreme court', 'high court',
        'reuters', 'bbc', 'ndtv', 'pti', 'ani', 'times of india',
        'indian express', 'economic times', 'bloomberg', 'livemint',
        'rbi', 'sebi', 'isro', 'bcci', 'percent', 'crore', 'lakh', 'billion'
    ]

    def predict(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        fake_hits = sum(1 for w in self.FAKE_WORDS if w in t)
        real_hits = sum(1 for w in self.REAL_WORDS if w in t)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        fake_score = fake_hits * 0.15 + (0.2 if caps_ratio > 0.45 else 0)
        real_score = real_hits * 0.12
        net = real_score - fake_score

        if net > 0.2:
            fake_prob, real_prob = 0.12, 0.88
        elif net < -0.15:
            fake_prob, real_prob = 0.82, 0.18
        else:
            fake_prob, real_prob = 0.48, 0.52

        is_fake = fake_prob > 0.55
        confidence = float(max(fake_prob, real_prob) * 100)

        return {
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
            "is_fake": is_fake,
            "confidence": round(confidence, 1),
            "model_version": "Heuristic Signal Engine (Fallback)",
            "memory_efficient": True
        }


# ─────────────────────────────────────────────────────────────────────────────
# Keras Inference Engine
# ─────────────────────────────────────────────────────────────────────────────
class FakeNewsDLInferenceEngine:
    """
    Production Deep Learning Inference Pipeline using pre-trained .keras model.
    Falls back to lightweight HeuristicFallbackEngine if Keras/model unavailable.
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        self.keras_model = None
        self.tokenizer = None
        self.use_keras = False
        self.fallback_engine = HeuristicFallbackEngine()

        self.model_path = model_path or KERAS_MODEL_PATH
        self.tokenizer_path = tokenizer_path or TOKENIZER_PKL_PATH

        self._initialize_engine()

    def _initialize_engine(self):
        if not KERAS_AVAILABLE:
            print("[INFO] TensorFlow/Keras not available. Using Heuristic Fallback Engine.")
            return

        # Load tokenizer from .pkl
        tokenizer_loaded = False
        if os.path.exists(self.tokenizer_path):
            try:
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                tokenizer_loaded = True
                print(f"[OK] Tokenizer loaded from {self.tokenizer_path}")
            except Exception as e:
                print(f"[WARN] Tokenizer load error: {e}")
        else:
            print(f"[WARN] Tokenizer not found at {self.tokenizer_path}")

        # Load Keras model
        if os.path.exists(self.model_path):
            try:
                self.keras_model = keras.models.load_model(self.model_path, compile=False)
                self.use_keras = True
                print(f"[OK] Keras model loaded from {self.model_path}")
            except Exception as e:
                print(f"[WARN] Keras model load error: {e}")
        else:
            print(f"[WARN] Keras model not found at {self.model_path}")

        if not self.use_keras or not tokenizer_loaded:
            print("[INFO] Using Heuristic Fallback Engine.")
            self.use_keras = False
        else:
            try:
                # Pre-warm Keras graph to eliminate first-request compilation latency
                self.predict("prewarm deep learning inference graph")
                print("[OK] Keras inference engine warmed up.")
            except Exception as e:
                print(f"[WARN] Warmup notice: {e}")

    def _preprocess(self, text: str) -> np.ndarray:
        """Tokenize and pad text for Keras model input."""
        try:
            if hasattr(self.tokenizer, 'texts_to_sequences'):
                # Standard Keras Tokenizer
                sequences = self.tokenizer.texts_to_sequences([text])
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
                return padded
            elif hasattr(self.tokenizer, 'encode'):
                # HuggingFace-style tokenizer fallback
                ids = self.tokenizer.encode(text)
                ids = ids[:MAX_SEQ_LEN]
                ids = ids + [0] * max(0, MAX_SEQ_LEN - len(ids))
                return np.array([ids])
            else:
                # Generic: try word_index dict
                word_index = getattr(self.tokenizer, 'word_index', {})
                words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
                seq = [word_index.get(w, 0) for w in words[:MAX_SEQ_LEN]]
                seq = seq + [0] * max(0, MAX_SEQ_LEN - len(seq))
                return np.array([seq])
        except Exception as e:
            print(f"[WARN] Preprocessing error: {e}")
            return np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)

    def predict(self, text: str) -> Dict[str, Any]:
        """Run classification on text input. Returns fake/real probabilities."""
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            return {"fake_prob": 0.5, "real_prob": 0.5, "is_fake": False, "confidence": 50.0,
                    "model_version": "Default", "memory_efficient": True}

        try:
            if self.use_keras and self.keras_model is not None and self.tokenizer is not None:
                padded = self._preprocess(text)
                raw_output = self.keras_model.predict(padded, verbose=0)

                # Handle both binary sigmoid (shape [1,1]) and softmax (shape [1,2])
                if raw_output.shape[-1] == 1:
                    # Binary classification: output is P(FAKE)
                    fake_prob = float(raw_output[0][0])
                    real_prob = 1.0 - fake_prob
                else:
                    # Softmax: assume [P(REAL), P(FAKE)] or [P(FAKE), P(REAL)]
                    # Standard convention: index 0 = FAKE, index 1 = REAL
                    # Check which index is higher and use context
                    probs = raw_output[0]
                    if len(probs) >= 2:
                        fake_prob = float(probs[1])   # index 1 = FAKE
                        real_prob = float(probs[0])   # index 0 = REAL
                    else:
                        fake_prob = float(probs[0])
                        real_prob = 1.0 - fake_prob

                is_fake = fake_prob > 0.50
                confidence = float(max(fake_prob, real_prob) * 100)

                return {
                    "fake_prob": round(fake_prob, 4),
                    "real_prob": round(real_prob, 4),
                    "is_fake": is_fake,
                    "confidence": round(confidence, 1),
                    "model_version": "Keras Deep Learning (fake_real_news_detection_model.keras)",
                    "memory_efficient": False
                }

        except Exception as e:
            print(f"[WARN] Keras inference error: {e}")

        # Fallback to heuristic engine
        return self.fallback_engine.predict(text)
