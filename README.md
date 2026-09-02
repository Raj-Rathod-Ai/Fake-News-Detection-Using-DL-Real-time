# 🔍 TruthLens — Real-Time Multi-Modal AI News Verification & Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Keras / TensorFlow](https://img.shields.io/badge/Keras%20%2F%20TensorFlow-Deep%20Learning-red.svg)](https://keras.io)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com)
[![Netlify](https://img.shields.io/badge/Deploy-Netlify-00C7B7.svg)](https://netlify.com)

**TruthLens** is an advanced, production-ready AI intelligence platform designed to detect misinformation, synthetic media, and deepfakes across multiple modalities (**Text, Images, Deepfake Videos, and Synthetic Audio**) with sub-second latency and real-time live web grounding.

---

## 🚀 Key Features

### 🧠 1. Multi-Modal AI Verification Suite
- **Text Fake News Detection (`/api/ai-scan`)**:
  - **Keras Deep Learning Neural Network** (`fake_news_detection_model.keras` + `tokenizer.pkl`) with Embedding, GlobalAveragePooling1D, and Deep Dense layers for high-accuracy sequence classification.
  - **Dual-Engine Architecture**: Native Keras/TensorFlow inference with pure-NumPy heuristic fallback.
  - **Tavily Real-Time Web Intelligence**: Verifies live breaking news (1-hour breaking news to 100-year historical events) with smart query extraction and token caching.
  - **Mistral / Gemini AI LLM Fact-Checking**: Provides structured reasoning, nuanced context, and credibility signals.
- **Image Forensic Scanner (`/api/detect-image`)**:
  - Error Level Analysis (ELA), EXIF metadata tamper detection, Laplacian noise variance, and generative AI artifact classification.
- **Deepfake Video Scanner (`/api/detect-deepfake`)**:
  - OpenCV-powered frame extraction, facial boundary warping detection, temporal coherence analysis, and face swap anomaly checks.
- **Synthetic Voice Detector (`/api/detect-voice`)**:
  - Acoustic zero-crossing rate (ZCR) analysis, neural vocoder high-frequency phase anomaly scan, and organic RMS energy dynamic variance.

### 🌐 2. Real-Time Intelligence Feeds
- **Live Financial Markets**: Real-time SSE streaming for Indian indices (SENSEX, NIFTY 50), top equities, forex, precious metals (Gold, Silver), and crypto.
- **Live Cricket Scores**: Cricbuzz API integration for live matches and series summaries.
- **Real-Time News Feeds**: Google News RSS parser with multi-API fallbacks (NewsAPI / RapidAPI).
- **Live Local Weather**: Geocoded location lookup via OpenStreetMap Nominatim with WeatherAPI fallback.

### 🗄️ 3. Dual-Layer Resilient Database
- **MongoDB Atlas Cloud Database**: Cloud persistence for user accounts, scan history, and feedback.
- **Automatic SQLite Fallback**: Zero-configuration local database (`truthlens.db`) if cloud credentials are not supplied.

---

## 📁 Repository Structure

```
Fake-News-Detection-Using-ML-Real-time/
├── app.py                             # Flask backend & API routing
├── dl_model.py                        # Keras Deep Learning inference engine
├── fake_news_detection_model.keras    # Trained Keras Deep Learning model
├── tokenizer.pkl                      # Fitted Keras text Tokenizer
├── test_dl_model.py                   # Automated unit test suite
├── dataset_downloader.py              # Dataset fetcher & preprocessor
├── requirements.txt                   # Python dependencies
├── Procfile                           # Gunicorn process manager for Render
├── render.yaml                        # Render blueprint configuration
├── netlify.toml                       # Netlify SPA & API proxy configuration
├── templates/
│   └── index.html                     # Responsive, glassmorphism frontend
└── uploads/                           # Temporary upload storage
```

---

## ☁️ Free Tier Deployment Guide

### 🟣 Deploying on Render (Free Web Service)

1. **Fork or Push** this repository to your GitHub account:
   `https://github.com/Raj-Rathod-Ai/Fake-News-Detection-Using-ML-Real-time`
2. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Web Service**.
3. Select your GitHub repository.
4. Configure the service settings:
   - **Name**: `fake-news-detection-using-ml-real-time`
   - **Environment**: `Python 3`
   - **Region**: Any (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
   - **Instance Type**: `Free` (512 MB RAM)
5. Under **Environment Variables**, add:
   - `FLASK_ENV` = `production`
   - `SECRET_KEY` = *(click Generate or enter a random 32-char string)*
   - `TAVILY_API_KEY` = *your_tavily_api_key*
   - `MISTRAL_API_KEY` = *your_mistral_api_key*
   - `MONGO_URI` = *your_mongodb_connection_string* (Optional)
   - `CRICBUZZ_KEY` = *your_rapidapi_key* (Optional)
6. Click **Deploy Web Service**. Your app will be live at `https://fake-news-detection-using-ml-real-time.onrender.com`.

---

## 🟢 Deploying Frontend on Netlify (Optional Decoupled Setup)

If you prefer hosting the frontend on Netlify with the backend hosted on Render:

1. Go to [Netlify Dashboard](https://app.netlify.com/) → **Add new site** → **Import an existing project**.
2. Connect your GitHub repository.
3. Configure Build Settings:
   - **Publish directory**: `templates`
   - **Build command**: *(leave blank)*
4. The [`netlify.toml`](file:///c:/Final%20Project/netlify.toml) automatically proxies `/api/*` requests to your live Render backend:
   ```toml
   [[redirects]]
     from = "/api/*"
     to = "https://fake-news-detection-using-ml-real-time.onrender.com/api/:splat"
     status = 200
     force = true
   ```
5. Click **Deploy Site**.

---

## 💻 Local Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Raj-Rathod-Ai/Fake-News-Detection-Using-ML-Real-time.git
cd Fake-News-Detection-Using-ML-Real-time
```

### 2. Set up virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and enter your API keys:
```bash
cp .env.example .env
```

### 5. Run the server
```bash
python app.py
```
Open **`http://localhost:3000`** in your browser.

---

## 🧪 Running Automated Tests

Run the test suite to verify the Keras model, tokenizer, and edge-case handling:
```bash
python test_dl_model.py
```

---

## 🔑 Environment Variables Reference

| Variable | Description | Required | Default |
|---|---|---|---|
| `PORT` | Web server listening port | No | `3000` (auto-set by Render) |
| `SECRET_KEY` | Flask session encryption key | Yes (Prod) | `truthlens-v8-production-secret-key` |
| `TAVILY_API_KEY` | Real-time news search & grounding | Optional | `""` |
| `MISTRAL_API_KEY` | Fact-checking LLM reasoning | Optional | `""` |
| `MONGO_URI` | Cloud MongoDB Atlas database | Optional | SQLite fallback (`truthlens.db`) |
| `CRICBUZZ_KEY` | RapidAPI key for Cricbuzz cricket scores | Optional | RapidAPI key |
| `WEATHER_API_KEY` | WeatherAPI key for local weather | Optional | OpenStreetMap geocode |

---

## 📜 License
This project is licensed under the MIT License.
