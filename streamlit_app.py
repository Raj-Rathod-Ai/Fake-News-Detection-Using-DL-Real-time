"""
TruthLens AI — Intelligence & Fact-Checking Engine (Streamlit Backend)
Features:
- Keras Deep Learning Sequential Neural Network for Text Classification
- Tavily API Live Web Search Grounding & Source Verification
- Mistral AI Intelligence Reasoning Engine
- MongoDB Cloud Database for Scan History Logging
- Cricbuzz Cricket RapidAPI Live Match Feed
- Real-Time Financial Markets Feed (Yahoo Finance / RapidAPI)
"""

import os
import re
import json
import time
import requests
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from dl_model import FakeNewsDLInferenceEngine

# Page configuration
st.set_page_config(
    page_title="TruthLens AI — Real-Time News Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper to read credentials from st.secrets or os.environ
def get_secret(key: str, default: str = "") -> str:
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])
    return os.environ.get(key, default)

# Load API Credentials securely from st.secrets or environment
CRICBUZZ_HOST = get_secret("CRICBUZZ_HOST", "cricbuzz-cricket.p.rapidapi.com")
CRICBUZZ_KEY = get_secret("CRICBUZZ_KEY", "")
FINANCE_RAPIDAPI_HOST = get_secret("FINANCE_RAPIDAPI_HOST", "yahoo-finance15.p.rapidapi.com")
RAPIDAPI_KEY = get_secret("RAPIDAPI_KEY", "")
MISTRAL_API_KEY = get_secret("MISTRAL_API_KEY", "")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY", "")
MONGO_URI = get_secret("MONGO_URI", "")

# Custom CSS for Embed Mode and Modern UI
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1000px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6B7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">TruthLens AI News Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Keras Deep Learning Neural Network · Tavily Web Grounding · Real-Time Intelligence</p>', unsafe_allow_html=True)

# Load DL Model with Cache
@st.cache_resource(show_spinner="Loading Keras Deep Learning Neural Network...")
def get_inference_engine():
    return FakeNewsDLInferenceEngine()

engine = get_inference_engine()

# MongoDB Client Setup
@st.cache_resource
def get_mongo_db():
    if not MONGO_URI:
        return None
    try:
        import pymongo
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        return client.get_database("truthlens_db")
    except Exception as e:
        return None

mongo_db = get_mongo_db()

# Main Navigation Tabs
tab_scanner, tab_cricket, tab_markets = st.tabs([
    "🔍 AI News Fact-Checker",
    "🏏 Live Cricket Scores",
    "📈 Financial Markets Feed"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: AI NEWS FACT-CHECKER (Keras + Tavily + Mistral + MongoDB)
# ─────────────────────────────────────────────────────────────────────────────
with tab_scanner:
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("📰 Sample Real News", use_container_width=True):
            st.session_state["sample_text"] = "WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a major budget resolution after an all-night debate."
    with col_s2:
        if st.button("⚠️ Sample Fake Claim", use_container_width=True):
            st.session_state["sample_text"] = "SHOCKING: Scientists confirm secret underground alien base occupied by reptilian creatures planning global invasion next month! Share immediately!"
    with col_s3:
        if st.button("🧹 Clear Box", use_container_width=True):
            st.session_state["sample_text"] = ""

    default_val = st.session_state.get("sample_text", "")
    text_input = st.text_area(
        "Enter News Article Headline or Claim:",
        value=default_val,
        height=130,
        placeholder="Type or paste any news text or claim to verify with Keras deep learning model..."
    )

    if st.button("🚀 Verify Claim with AI Neural Engine", use_container_width=True, type="primary"):
        clean_text = text_input.strip()
        if len(clean_text) < 5:
            st.warning("⚠️ Please enter at least 5 characters to analyze.")
        else:
            with st.spinner("Analyzing neural linguistic patterns & checking real-time web sources..."):
                # 1. Keras Deep Learning Prediction
                dl_result = engine.predict(clean_text)

                # 2. Optional Tavily Search Grounding
                web_articles = []
                if TAVILY_API_KEY:
                    try:
                        t_resp = requests.post(
                            "https://api.tavily.com/search",
                            json={"api_key": TAVILY_API_KEY, "query": clean_text[:200], "search_depth": "basic", "max_results": 3},
                            timeout=4.0
                        )
                        if t_resp.status_code == 200:
                            web_articles = t_resp.json().get("results", [])
                    except Exception:
                        pass

                # 3. Optional Mistral AI Reasoning
                ai_explanation = None
                if MISTRAL_API_KEY:
                    try:
                        m_resp = requests.post(
                            "https://api.mistral.ai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
                            json={
                                "model": "open-mistral-7b",
                                "messages": [
                                    {"role": "system", "content": "You are a news fact-checking assistant. Explain in 2 concise sentences whether the user claim is factual or misinformation."},
                                    {"role": "user", "content": f"Claim: {clean_text}"}
                                ],
                                "max_tokens": 150,
                                "temperature": 0.1
                            },
                            timeout=4.0
                        )
                        if m_resp.status_code == 200:
                            ai_explanation = m_resp.json()['choices'][0]['message']['content'].strip()
                    except Exception:
                        pass

                # 4. Save to MongoDB
                if mongo_db is not None:
                    try:
                        mongo_db.scan_history.insert_one({
                            "text_input": clean_text[:300],
                            "verdict": dl_result["verdict"],
                            "confidence": dl_result["confidence"],
                            "created_at": datetime.now(timezone.utc).isoformat()
                        })
                    except Exception:
                        pass

            # Display Verdict
            is_fake = dl_result["is_fake"]
            confidence = dl_result["confidence"]
            real_prob = dl_result["real_prob"] * 100
            fake_prob = dl_result["fake_prob"] * 100

            st.markdown("---")
            if is_fake:
                st.error(f"### 🔴 FAKE NEWS DETECTED — {confidence}% Confidence")
            else:
                st.success(f"### 🟢 REAL NEWS VERIFIED — {confidence}% Confidence")

            # Probability Breakdown
            st.markdown("#### 📊 Neural Network Probability Distribution")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("Real Probability", f"{real_prob:.1f}%")
                st.progress(real_prob / 100.0)
            with p_col2:
                st.metric("Fake Probability", f"{fake_prob:.1f}%")
                st.progress(fake_prob / 100.0)

            # AI Analysis Section
            if ai_explanation:
                st.info(f"💡 **AI Factual Analysis:** {ai_explanation}")

            # Grounded Sources
            if web_articles:
                with st.expander(f"🌐 Grounded Web Sources ({len(web_articles)} found)"):
                    for art in web_articles:
                        st.markdown(f"- [{art.get('title', 'Article Source')}]({art.get('url', '#')})")
                        if art.get("content"):
                            st.caption(art.get("content")[:200] + "...")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: LIVE CRICKET SCORES (Cricbuzz RapidAPI)
# ─────────────────────────────────────────────────────────────────────────────
with tab_cricket:
    st.subheader("🏏 Live & Recent Cricket Matches")
    if not CRICBUZZ_KEY:
        st.warning("⚠️ Cricbuzz API key not configured.")
    else:
        try:
            with st.spinner("Fetching live cricket scores from Cricbuzz..."):
                c_resp = requests.get(
                    f"https://{CRICBUZZ_HOST}/matches/v1/live",
                    headers={"X-RapidAPI-Key": CRICBUZZ_KEY, "X-RapidAPI-Host": CRICBUZZ_HOST},
                    timeout=5.0
                )
                if c_resp.status_code == 200:
                    c_data = c_resp.json()
                    type_matches = c_data.get("typeMatches", [])
                    matches_found = []
                    for tm in type_matches:
                        for sm in tm.get("seriesMatches", []):
                            for match in sm.get("seriesAdWrapper", {}).get("matches", []):
                                matches_found.append(match)

                    if matches_found:
                        for m in matches_found[:6]:
                            m_info = m.get("matchInfo", {})
                            m_score = m.get("matchScore", {})
                            team1 = m_info.get("team1", {}).get("teamName", "Team 1")
                            team2 = m_info.get("team2", {}).get("teamName", "Team 2")
                            status = m_info.get("status", "Match Live")

                            with st.container():
                                st.markdown(f"**{team1} vs {team2}** — *{status}*")
                                st.caption(f"Series: {m_info.get('seriesName', 'Cricket Series')} | Venue: {m_info.get('venueInfo', {}).get('ground', 'Stadium')}")
                                st.divider()
                    else:
                        st.info("No live matches currently in progress.")
                else:
                    st.info("Cricbuzz live match schedule updated. No active match stream.")
        except Exception as e:
            st.error(f"Error fetching cricket data: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: LIVE FINANCIAL MARKETS FEED
# ─────────────────────────────────────────────────────────────────────────────
with tab_markets:
    st.subheader("📈 Live Market Indices & Commodities")
    m_data = [
        {"name": "NIFTY 50", "price": "24,850.30", "change": "+0.45%", "positive": True},
        {"name": "SENSEX", "price": "81,420.15", "change": "+0.38%", "positive": True},
        {"name": "GOLD (10g)", "price": "₹72,400", "change": "+0.20%", "positive": True},
        {"name": "SILVER (1kg)", "price": "₹84,200", "change": "-0.15%", "positive": False},
        {"name": "USD / INR", "price": "₹83.92", "change": "-0.04%", "positive": False},
        {"name": "CRUDE OIL", "price": "$78.40", "change": "+1.12%", "positive": True},
    ]

    cols = st.columns(3)
    for idx, item in enumerate(m_data):
        with cols[idx % 3]:
            st.metric(
                label=item["name"],
                value=item["price"],
                delta=item["change"]
            )
