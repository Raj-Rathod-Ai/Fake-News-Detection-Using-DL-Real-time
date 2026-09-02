"""
TruthLens Deep Learning & NLP Core Engine
Architecture: Keras Deep Learning Sequential Neural Network
Trained Model: models/fake_real_news_detection_model.keras / fake_real_news_detection_model.h5
Tokenizer: models/tokenizer.pkl (Keras Tokenizer)
"""

import os
import pickle
import numpy as np
from typing import Dict, Any

# Suppress debug logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import Keras / TensorFlow
try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    try:
        from tensorflow import keras
        KERAS_AVAILABLE = True
    except ImportError:
        KERAS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def _find_file(filename: str) -> str:
    in_models = os.path.join(MODELS_DIR, filename)
    if os.path.exists(in_models):
        return in_models
    in_base = os.path.join(BASE_DIR, filename)
    if os.path.exists(in_base):
        return in_base
    return in_models


class FakeNewsDLInferenceEngine:
    """Production Keras Deep Learning Inference Pipeline."""

    def __init__(self, model_path: str = None, tokenizer_path: str = None, max_len: int = 500):
        self.max_len = max_len
        self.tokenizer_path = tokenizer_path or _find_file('tokenizer.pkl')
        
        # Priority: fake_real_news_detection_model.keras -> .h5 -> fake_news_detection_model.keras
        self.model_path = model_path or (
            _find_file('fake_real_news_detection_model.keras') if os.path.exists(_find_file('fake_real_news_detection_model.keras'))
            else (_find_file('fake_real_news_detection_model.h5') if os.path.exists(_find_file('fake_real_news_detection_model.h5'))
            else _find_file('fake_news_detection_model.keras'))
        )

        self.tokenizer = None
        self.keras_model = None
        self.is_keras_active = False

        self._initialize_engine()

    def _initialize_engine(self):
        """Loads the official Keras Tokenizer and trained Keras model."""
        # 1. Load Tokenizer
        if os.path.exists(self.tokenizer_path):
            try:
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print(f"[OK] Keras Tokenizer loaded successfully from {self.tokenizer_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load tokenizer from {self.tokenizer_path}: {e}")
        else:
            print(f"[WARN] Tokenizer file not found at {self.tokenizer_path}")

        # 2. Load Keras Model
        if KERAS_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.keras_model = keras.models.load_model(self.model_path, compile=False)
                self.is_keras_active = True
                print(f"[OK] Keras Deep Learning Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load Keras model from {self.model_path}: {e}")
                self.is_keras_active = False
        else:
            if not KERAS_AVAILABLE:
                print("[ERROR] Keras / TensorFlow is not installed in the environment.")
            else:
                print(f"[WARN] Model file not found at {self.model_path}")

    def _preprocess_text(self, text: str) -> np.ndarray:
        """Converts text to integer sequence padded to max_len (500)."""
        padded = np.zeros((1, self.max_len), dtype=np.int32)
        if not text or not isinstance(text, str) or not self.tokenizer:
            return padded

        seqs = self.tokenizer.texts_to_sequences([text])
        if seqs and len(seqs[0]) > 0:
            s = seqs[0][:self.max_len]
            padded[0, -len(s):] = s  # Pre-padding / right aligned
        return padded

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs Deep Learning sequence classification on text input."""
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "fake_prob": 0.5,
                "real_prob": 0.5,
                "is_fake": False,
                "confidence": 50.0,
                "verdict": "REAL",
                "model_version": "Keras Deep Learning Neural Network",
            }

        if not self.is_keras_active or self.keras_model is None or self.tokenizer is None:
            return {
                "fake_prob": 0.5,
                "real_prob": 0.5,
                "is_fake": False,
                "confidence": 50.0,
                "verdict": "REAL",
                "model_version": "Keras Deep Learning Neural Network (Model Not Loaded)",
            }

        try:
            padded_input = self._preprocess_text(text)
            raw_pred = float(self.keras_model.predict(padded_input, verbose=0)[0][0])

            # Model output: probability of being Real in [0.0, 1.0]
            real_prob = float(np.clip(raw_pred, 0.0, 1.0))
            fake_prob = float(np.clip(1.0 - real_prob, 0.0, 1.0))
            is_fake = fake_prob > 0.5
            confidence = float(round(max(real_prob, fake_prob) * 100, 1))
            verdict = "FAKE" if is_fake else "REAL"

            return {
                "fake_prob": round(fake_prob, 4),
                "real_prob": round(real_prob, 4),
                "is_fake": is_fake,
                "confidence": confidence,
                "verdict": verdict,
                "model_version": "Keras Deep Learning Neural Network",
            }

        except Exception as e:
            print(f"[Inference Error] {e}")
            return {
                "fake_prob": 0.5,
                "real_prob": 0.5,
                "is_fake": False,
                "confidence": 50.0,
                "verdict": "REAL",
                "model_version": "Keras Deep Learning Neural Network",
                "error": str(e)
            }
