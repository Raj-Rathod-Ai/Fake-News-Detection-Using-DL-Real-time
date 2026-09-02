"""
TruthLens — AI Fake News Detection (Streamlit Web Application)
Optimized for high-speed standalone and embedded iframe operation.
Powered by Keras Deep Learning Sequential Neural Network.
"""

import streamlit as st
import numpy as np
import time
from dl_model import FakeNewsDLInferenceEngine

# Page configuration
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Performance Styling (clean embeddable layout)
st.markdown("""
<style>
    /* Hide Streamlit branding & padding when embedded */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 900px;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 0.95rem;
        text-align: center;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        border-radius: 12px;
        font-size: 0.95rem;
    }
    .stButton button {
        border-radius: 12px;
        font-weight: 700;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-title">TruthLens AI News Scanner</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Keras Deep Learning Neural Network · Text Fake News Verification</p>', unsafe_allow_html=True)

# Load DL Model with Resource Cache
@st.cache_resource(show_spinner="Loading Keras Neural Model...")
def get_inference_engine():
    return FakeNewsDLInferenceEngine()

engine = get_inference_engine()

# Quick Sample Buttons
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if st.button("📰 Sample Real News", use_container_width=True):
        st.session_state["sample_text"] = "WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a major budget resolution after an all-night debate."
with col_s2:
    if st.button("⚠️ Sample Fake Claim", use_container_width=True):
        st.session_state["sample_text"] = "SHOCKING: Scientists confirm secret underground alien base occupied by reptilian creatures planning global invasion next month! Share immediately!"
with col_s3:
    if st.button("🧹 Clear Input", use_container_width=True):
        st.session_state["sample_text"] = ""

# Input Form
default_val = st.session_state.get("sample_text", "")
text_input = st.text_area(
    "Enter News Headline or Full Article Text:",
    value=default_val,
    height=140,
    placeholder="Paste news claim or article content here (min 5 characters)...",
    key="news_input"
)

scan_clicked = st.button("🚀 Verify Claim with Keras Model", use_container_width=True, type="primary")

if scan_clicked:
    clean_text = text_input.strip()
    if len(clean_text) < 5:
        st.warning("⚠️ Please enter at least 5 characters to analyze.")
    else:
        with st.spinner("Analyzing neural linguistic patterns..."):
            result = engine.predict(clean_text)

        is_fake = result["is_fake"]
        verdict = result["verdict"]
        confidence = result["confidence"]
        real_prob = result["real_prob"] * 100
        fake_prob = result["fake_prob"] * 100

        # Result Display Banner
        st.markdown("---")
        if is_fake:
            st.error(f"### 🔴 FAKE NEWS DETECTED — {confidence}% Confidence")
        else:
            st.success(f"### 🟢 REAL NEWS VERIFIED — {confidence}% Confidence")

        # Probability Metrics
        st.markdown("#### 📊 Neural Network Probability Breakdown")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Real Probability", f"{real_prob:.1f}%")
            st.progress(real_prob / 100.0)
            
        with m_col2:
            st.metric("Fake Probability", f"{fake_prob:.1f}%")
            st.progress(fake_prob / 100.0)

        st.caption(f"Engine: Keras Deep Learning Sequential Neural Network · Model Active: {engine.is_keras_active}")
