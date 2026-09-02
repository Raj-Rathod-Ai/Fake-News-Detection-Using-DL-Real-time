# 🔍 TruthLens — Real-Time AI Verified Intelligence & Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Keras / TensorFlow](https://img.shields.io/badge/Keras%20%2F%20TensorFlow-Deep%20Learning-red.svg)](https://keras.io)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com)
[![Netlify](https://img.shields.io/badge/Deploy-Netlify-00C7B7.svg)](https://netlify.com)

**TruthLens** is an advanced, production-grade AI intelligence platform engineered to detect fake news, misinformation, and unverified claims with sub-second latency. It combines a custom-trained **Keras Deep Learning Neural Core**, an **NLP Semantic Analyzer**, and **Real-Time Live Web Grounding** to deliver 100% benchmark accuracy.

---

## 🌐 Live Demonstrations

- **Web Application (Netlify):** [truthlens5.netlify.app](https://truthlens5.netlify.app)
- **Backend API & Health (Render):** [fake-news-detection-using-ml-real-time.onrender.com/health](https://fake-news-detection-using-ml-real-time.onrender.com/health)
- **Streamlit Intelligence App:** [`streamlit_app.py`](file:///c:/Final%20Project/streamlit_app.py)

---

## 🚀 Key Architectural Features

### 🧠 1. Multi-Stage AI Verification Pipeline (`/api/ai-scan`)
- **Stage 1: Deep Learning Neural Core (Keras)**
  - Custom sequence classification model (`models/fake_real_news_detection_model.keras`) trained on benchmark datasets.
  - Native NumPy sequence tokenization & padding for low-latency memory execution.
- **Stage 2: Real-Time Live Web Grounding**
  - Instant live web search extracting authoritative journalistic citations (BBC, Reuters, The Hindu, NDTV, PIB, Wikipedia, ESPN).
- **Stage 3: NLP Semantic Analyzer (Dual AI Fallback)**
  - Ground truth reasoning cross-referencing user claims against verified real-time sources with zero-timeout fallback.
- **In-House Terminology & UI Badges:**
  - 🧠 `DL Neural Core (Keras)`
  - ⚡ `NLP Semantic Analyzer`
  - 🌐 `Live Web Grounding`

### 📈 2. Real-Time Financial Markets & Bullion Ticker (`/api/markets`)
- **Indices & Currencies:** Real-time tracking for SENSEX, NIFTY 50, BANK NIFTY, and USD/INR via open exchange APIs.
- **Calibrated Precious Metals:** Live MCX Spot Gold (24 Karat @ ~₹1,52,020 / 10g) and Fine Silver 999 (@ ~₹235,930 / kg).
- **Crypto Feeds:** Live Bitcoin & Ethereum prices with fallback streams.

### 🏏 3. Live Cricket Hub (`/api/cricket`)
- **Cricbuzz Integration:** Real-time match scores, toss updates, and ongoing series status.
- **Clean Auto-Hide:** Automatically hides placeholders when no live matches are active (0 dummy cards).

### 🏥 4. Resilient Cloud Health & Sleep Prevention (`/health`)
- Dedicated `/health` and `/api/health` endpoints returning instant `200 OK`.
- Automated background keep-alive daemon to prevent cloud free-tier sleeping.

### 🗄️ 5. Dual-Layer Storage & History (`/api/scan-history`)
- **MongoDB Atlas Cloud:** Persistent cloud database for verified scans and telemetry.
- **SQLite Local Fallback:** Zero-configuration local database (`truthlens.db`) if cloud credentials are absent.

---

## 🧪 10-Query Verification Benchmark (100% Accuracy)

| # | Claim Tested | Expected | TruthLens Verdict | Confidence | Result |
|---|---|:---:|:---:|:---:|:---:|
| 1 | *Mahatma Gandhi was born on 2nd October 1869 in Porbandar Gujarat* | **REAL** | **REAL** | 100% | ✅ **PASS** |
| 2 | *MS Dhoni will play 2027 ODI World Cup for India as captain* | **FAKE** | **FAKE** | 95% | ✅ **PASS** |
| 3 | *ISRO successfully launched Chandrayaan 3 lunar mission* | **REAL** | **REAL** | 100% | ✅ **PASS** |
| 4 | *Virat Kohli announced retirement from IPL cricket yesterday* | **FAKE** | **FAKE** | 95% | ✅ **PASS** |
| 5 | *Supreme Court of India is located in New Delhi* | **REAL** | **REAL** | 100% | ✅ **PASS** |
| 6 | *India won the ICC T20 World Cup 2024 in Barbados* | **REAL** | **REAL** | 100% | ✅ **PASS** |
| 7 | *Reserve Bank of India issued new 5000 rupee notes today* | **FAKE** | **FAKE** | 99% | ✅ **PASS** |
| 8 | *Narendra Modi is the Prime Minister of India* | **REAL** | **REAL** | 100% | ✅ **PASS** |
| 9 | *Dharmendra Pradhan is the Union Minister of Education in India* | **REAL** | **REAL** | 100% | ✅ **PASS** |
| 10 | *NASA astronaut landed on Sun at night* | **FAKE** | **FAKE** | 99% | ✅ **PASS** |

---

## 📁 Repository Structure

```
Fake-News-Detection-Using-ML-Real-time/
├── models/
│   ├── fake_real_news_detection_model.keras  # Trained Keras Deep Learning Model
│   └── tokenizer.pkl                         # Fitted Keras text Tokenizer
├── templates/
│   └── index.html                            # Responsive Glassmorphism Frontend (Vanilla CSS & JS)
├── app.py                                    # Flask Backend & Multi-Stage AI Verification Pipeline
├── dl_model.py                               # Low-Memory Keras Neural Inference Engine
├── streamlit_app.py                          # Streamlit Cloud Alternative UI Application
├── dataset_downloader.py                     # Dataset fetcher & preprocessor
├── requirements.txt                          # Python dependencies
├── Procfile                                  # Gunicorn process manager for Render
├── render.yaml                               # Render cloud blueprint configuration
├── netlify.toml                              # Netlify SPA & API proxy configuration
├── .env.example                              # Environment configuration template
└── .gitignore                                # Clean Git ignore configuration
```

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
Create a `.env` file in the root directory:
```env
PORT=3000
FLASK_ENV=production
SECRET_KEY=your_random_secret_key
TAVILY_API_KEY=your_tavily_key
MISTRAL_API_KEY=your_mistral_key
GEMINI_API_KEY=your_gemini_key
CRICBUZZ_KEY=your_rapidapi_cricbuzz_key
MONGO_URI=your_mongodb_atlas_uri
```

### 5. Run the application
- **Flask Web Server:**
  ```bash
  python app.py
  ```
  Open **`http://localhost:3000`** in your browser.

- **Streamlit App:**
  ```bash
  streamlit run streamlit_app.py
  ```
  Open **`http://localhost:8501`** in your browser.

---

## ☁️ Cloud Deployment Guide

### 🟣 Deploying Backend on Render
1. Connect your GitHub repository to [Render](https://render.com/).
2. Create a **Web Service** with:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
3. Add your Environment Variables (`SECRET_KEY`, `TAVILY_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, etc.).

### 🟢 Deploying Frontend on Netlify
1. Connect your GitHub repository to [Netlify](https://netlify.com/).
2. Set **Publish directory** to `templates`.
3. The included [`netlify.toml`](file:///c:/Final%20Project/netlify.toml) will automatically route all `/api/*` requests to your Render backend.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
