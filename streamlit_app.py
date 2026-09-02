"""
TruthLens — AI Fake News Detection (Streamlit Web Application)
Powered by Keras Deep Learning Sequential Neural Network
"""

import streamlit as st
import numpy as np
import time
from dl_model import FakeNewsDLInferenceEngine

# Page configuration
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1rem;
        text-align: center;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .result-card-real {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .result-card-fake {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-title">TruthLens AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-Time Fake News Detection · Keras Deep Learning Neural Network</p>', unsafe_allow_html=True)

# Load DL Model with Cache
@st.cache_resource(show_spinner="Loading Keras Deep Learning Neural Network...")
def get_inference_engine():
    return FakeNewsDLInferenceEngine()

engine = get_inference_engine()

# Sidebar Information
with st.sidebar:
    st.header("🧠 Model Architecture")
    st.info("""
    - **Engine**: Keras Deep Learning Sequential
    - **Layers**: Embedding + GlobalAveragePooling1D + Multi-Dense + Sigmoid
    - **Input**: Natural Language Article Text
    - **Max Tokens**: 500
    - **Vocabulary**: 400,000+ Words
    """)
    
    st.divider()
    st.subheader("💡 Quick Sample Claims")
    
    if st.button("📰 Sample Real News"):
        st.session_state["sample_text"] = "WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a major budget resolution after an all-night debate."
        
    if st.button("⚠️ Sample Fake News"):
        st.session_state["sample_text"] = "SHOCKING: Scientists confirm secret underground alien base occupied by reptilian creatures planning global invasion next month! Share immediately!"

# Main Input Form
default_val = st.session_state.get("sample_text", "")
text_input = st.text_area(
    "Enter News Headline or Full Article Text:",
    value=default_val,
    height=160,
    placeholder="Paste news claim or article content here (min 5 characters)..."
)

col1, col2 = st.columns([1, 4])
with col1:
    scan_clicked = st.button("🚀 Verify News", use_container_width=True, type="primary")

if scan_clicked:
    clean_text = text_input.strip()
    if len(clean_text) < 5:
        st.error("⚠️ Please enter at least 5 characters to analyze.")
    else:
        with st.spinner("Analyzing neural linguistic patterns..."):
            time.sleep(0.3)
            result = engine.predict(clean_text)

        is_fake = result["is_fake"]
        verdict = result["verdict"]
        confidence = result["confidence"]
        real_prob = result["real_prob"] * 100
        fake_prob = result["fake_prob"] * 100

        # Result Display
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

        st.caption(f"Evaluated by: {result.get('model_version', 'Keras Neural Engine')}")
