import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
import pickle

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Wine Quality Classification", page_icon="🍷", layout="wide")

# ============================================================
# FIGMA-INSPIRED CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Global */
.stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #0d0520 !important; color: #fff !important; }
section[data-testid="stSidebar"] .stRadio label { color: rgba(255,255,255,0.45) !important; font-size: 0.8125rem !important; }
section[data-testid="stSidebar"] .stRadio label:hover { color: rgba(255,255,255,0.8) !important; }
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: rgba(255,255,255,0.45) !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #fff !important; }
section[data-testid="stSidebar"] .stCaption p { color: rgba(255,255,255,0.5) !important; font-size: 0.625rem !important; }
section[data-testid="stSidebar"] hr { border-color: #ffffff !important; background-color: #ffffff !important; color: #ffffff !important; opacity: 1 !important; }
section[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] { background: transparent; border-radius: 0.5rem; }
/* Sidebar text + icons -> white on dark background */
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li { color: #fff !important; }
section[data-testid="stSidebar"] svg { fill: #fff !important; color: #fff !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #fff !important; }
/* Sidebar collapse/expand chevron icon -> white */
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarCollapsedControl"] svg, button[kind="header"] svg { fill: #fff !important; color: #fff !important; }

/* Sidebar collapse button visibility */
button[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
    position: fixed !important;
    left: 10px !important;
    top: 10px !important;
    z-index: 999 !important;
    width: 2.5rem !important;
    height: 2.5rem !important;
    background: linear-gradient(135deg, #a855f7, #7c3aed) !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.5rem !important;
    cursor: pointer !important;
}

button[data-testid="stSidebarCollapseButton"]:hover {
    background: linear-gradient(135deg, #9333ea, #7c3aed) !important;
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4) !important;
}

button[data-testid="stSidebarCollapseButton"] svg {
    fill: #fff !important;
    color: #fff !important;
}

/* Sidebar position and always show */
section[data-testid="stSidebar"] {
    z-index: 998 !important;
}

/* Panel / Card */
.panel { background: #fff; border-radius: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); overflow: hidden; padding: 1.5rem; margin-bottom: 1rem; }
.panel-sm { background: #fff; border-radius: 0.75rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); padding: 1rem; }

/* Hero Banner */
.hero-banner { background: linear-gradient(135deg, #0d0520 0%, #1e0a4e 50%, #2d1b69 100%); border-radius: 1rem; padding: 3rem 2.5rem; position: relative; overflow: hidden; margin-bottom: 1.5rem; }
.hero-tag { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(167,139,250,0.15); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.25); padding: 0.375rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; margin-bottom: 1rem; }
.hero-dot { width: 0.375rem; height: 0.375rem; border-radius: 9999px; background: #a78bfa; }
.hero-title { color: #fff !important; font-size: 2.25rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.75rem 0; }
.hero-desc { color: #c4b5fd; font-size: 1rem; line-height: 1.65; margin: 0; }

/* Page Header */
.page-header { background: #fff; padding: 1.5rem; border-bottom: 1px solid #f1f5f9; border-radius: 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.75rem; }
.page-icon { width: 2.25rem; height: 2.25rem; border-radius: 0.75rem; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 1.1rem; flex-shrink: 0; }
.page-icon-violet { background: linear-gradient(135deg, #7c3aed, #6d28d9); }
.page-icon-green { background: linear-gradient(135deg, #059669, #0d9488); }
.page-icon-rose { background: linear-gradient(135deg, #e11d48, #f43f5e); }
.page-icon-amber { background: linear-gradient(135deg, #f59e0b, #ef4444); }
.page-icon-blue { background: linear-gradient(135deg, #2563eb, #4f46e5); }
.page-header-title { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin: 0; }
.page-header-sub { font-size: 0.85rem; color: #64748b; margin: 0; }

/* Section Title with number badge */
.section-title { display: flex; align-items: center; gap: 0.625rem; margin-bottom: 1rem; }
.section-badge { width: 1.75rem; height: 1.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #fff; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.badge-violet { background: #7c3aed; }
.badge-green { background: #059669; }
.badge-amber { background: #f59e0b; }
.badge-blue { background: #2563eb; }
.badge-rose { background: #e11d48; }
.section-label { font-weight: 600; color: #1e293b; font-size: 0.9375rem; }

/* Nav Cards */
.nav-card { background: #fff; border-radius: 1rem; padding: 1.25rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); transition: transform 0.2s, box-shadow 0.2s; height: 100%; }
.nav-card:hover { transform: translateY(-4px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
.nav-card-icon { width: 2.5rem; height: 2.5rem; border-radius: 0.75rem; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 1.2rem; margin-bottom: 0.75rem; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.gradient-violet { background: linear-gradient(to bottom right, #8b5cf6, #9333ea); }
.gradient-blue { background: linear-gradient(to bottom right, #3b82f6, #4f46e5); }
.gradient-emerald { background: linear-gradient(to bottom right, #10b981, #0d9488); }
.gradient-rose { background: linear-gradient(to bottom right, #f43f5e, #ec4899); }
.nav-card-title { font-weight: 600; color: #1e293b; font-size: 0.875rem; margin-bottom: 0.25rem; }
.nav-card-desc { color: #64748b; font-size: 0.75rem; line-height: 1.6; }

/* Metric Cards (color-coded) */
.metric-card { border-radius: 1rem; padding: 1.25rem; text-align: center; }
.metric-card .value { font-weight: 700; font-size: 1.5rem; line-height: 1; }
.metric-card .label { font-size: 0.75rem; font-weight: 600; margin-top: 0.5rem; }
.metric-violet { background: #f5f3ff; border: 1px solid rgba(124,58,237,0.13); }
.metric-violet .value { color: #7c3aed; }
.metric-violet .label { color: #7c3aedaa; }
.metric-blue { background: #eff6ff; border: 1px solid rgba(37,99,235,0.13); }
.metric-blue .value { color: #2563eb; }
.metric-blue .label { color: #2563ebaa; }
.metric-green { background: #f0fdf4; border: 1px solid rgba(5,150,105,0.13); }
.metric-green .value { color: #059669; }
.metric-green .label { color: #059669aa; }
.metric-amber { background: #fffbeb; border: 1px solid rgba(217,119,6,0.13); }
.metric-amber .value { color: #d97706; }
.metric-amber .label { color: #d97706aa; }
.metric-rose { background: #fff1f2; border: 1px solid #fecdd3; }
.metric-rose .value { color: #e11d48; }
.metric-rose .label { color: #e11d48aa; }

/* Result Cards */
.result-good { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #86efac; border-radius: 1rem; padding: 2rem; text-align: center; }
.result-bad { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border: 2px solid #fda4af; border-radius: 1rem; padding: 2rem; text-align: center; }
.result-label-good { color: #059669; font-size: 2.5rem; font-weight: 900; margin: 0; }
.result-label-bad { color: #e11d48; font-size: 2.5rem; font-weight: 900; margin: 0; }
.result-subtitle { font-size: 0.85rem; color: #64748b; margin: 0.25rem 0 0 0; }

/* Confidence bars */
.confidence-wrap { margin-top: 1rem; }
.confidence-label { font-size: 0.75rem; font-weight: 600; margin-bottom: 0.25rem; }
.progress-track { height: 0.75rem; border-radius: 9999px; background: #f1f5f9; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 9999px; transition: width 0.7s; }
.progress-good { background: #059669; }
.progress-bad { background: #e11d48; }

/* Action Buttons */
.stButton > button { border-radius: 0.75rem !important; font-weight: 600 !important; font-size: 0.9375rem !important; padding: 0.75rem 1.5rem !important; transition: transform 0.2s !important; }
.stButton > button:hover { transform: translateY(-2px) !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important; }

/* Banner alerts */
.banner-success { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 1rem; padding: 1rem 1.5rem; display: flex; align-items: center; gap: 0.75rem; }
.banner-dot { width: 0.5rem; height: 0.5rem; border-radius: 9999px; background: #10b981; }
.banner-text { color: #059669; font-size: 0.875rem; font-weight: 600; }
.banner-warning { background: #fffbeb; border: 1px solid #fde68a; border-radius: 0.75rem; padding: 0.875rem 1.25rem; color: #92400e; font-size: 0.875rem; }

/* About Banner */
.about-banner { background: linear-gradient(135deg, #0d0520 0%, #2d1b69 60%, #4c1d95 100%); border-radius: 1rem; padding: 2.5rem; text-align: center; margin-bottom: 1.5rem; }
.about-title { color: #fff !important; font-size: 2rem; font-weight: 800; margin: 0 0 0.5rem 0; }
.about-sub { color: #c4b5fd; font-size: 1rem; margin: 0 0 0.25rem 0; }
.about-sub2 { color: #a78bfa; font-size: 0.875rem; margin: 0; }

/* Member cards */
.member-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 1.5rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.member-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.member-name { font-size: 1rem; font-weight: 600; color: #1e293b; }
.member-nim { font-size: 0.85rem; color: #7c3aed; font-weight: 500; margin-top: 0.25rem; }

/* Tech card */
.tech-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem; text-align: center; }
.tech-name { font-size: 0.85rem; font-weight: 600; color: #1e293b; }
.tech-desc { font-size: 0.75rem; color: #64748b; }

/* Footer */
.footer { text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1.5rem; }

/* Workflow */
.workflow-step { display: inline-flex; align-items: center; gap: 0.5rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 0.5rem 1rem; font-size: 0.8rem; font-weight: 500; color: #334155; }
.workflow-arrow { color: #cbd5e1; font-size: 1.2rem; margin: 0 0.25rem; }

/* Accordion fix */
.streamlit-expanderHeader { font-weight: 600 !important; }

/* Hide default Streamlit menu/footer/decoration, but KEEP the header itself so the
   "expand sidebar" button stays reachable when the sidebar is collapsed. */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stToolbar"] {visibility: hidden !important;}
header[data-testid="stHeader"] {background: transparent !important;}

/* Sidebar expand control (shown when the sidebar is collapsed) -> always visible & styled.
   Streamlit 1.57 uses data-testid="stExpandSidebarButton"; older versions used
   "stSidebarCollapsedControl"/"collapsedControl", so target all of them. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000 !important;
}
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 2.5rem !important;
    height: 2.5rem !important;
    background: linear-gradient(135deg, #a855f7, #7c3aed) !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.5rem !important;
    cursor: pointer !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: linear-gradient(135deg, #9333ea, #7c3aed) !important;
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4) !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    fill: #fff !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_data():
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=186)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = [
        'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
        'Chlorides', 'Free SO2', 'Total SO2', 'Density', 'pH',
        'Sulphates', 'Alcohol', 'Quality'
    ]
    df['Quality Label'] = df['Quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')
    return df


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="padding: 1rem 0.5rem 0.75rem; border-bottom: 1px solid #ffffff; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.625rem;">
    <div style="width: 2rem; height: 2rem; border-radius: 0.5rem; background: linear-gradient(135deg, #a855f7, #7c3aed); display: flex; align-items: center; justify-content: center; font-size: 1rem;">🍷</div>
    <span style="color: #fff; font-weight: 600; font-size: 0.875rem; letter-spacing: -0.01em;">Wine Quality ML</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p style="color: rgba(255,255,255,0.2); font-size: 0.625rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem;">Navigation</p>', unsafe_allow_html=True)

nav_pages = ["Home", "EDA", "Description Page", "Data Preprocessing",
     "Train Your Model", "Result / Prediction Demo", "Feature Importance", "About Us"]

default_index = 0
if 'nav_target' in st.session_state:
    target = st.session_state.pop('nav_target')
    if target in nav_pages:
        default_index = nav_pages.index(target)

page = st.sidebar.radio("Navigation", nav_pages, index=default_index, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("UCI ML Repository")
st.sidebar.caption("Wine Quality Dataset")

# Add custom JS to ensure hamburger button always works
st.markdown("""
<script>
    // Ensure sidebar collapse button is always accessible
    setTimeout(function() {
        const collapseBtn = document.querySelector('[data-testid="stSidebarCollapseButton"]');
        if (collapseBtn) {
            collapseBtn.style.visibility = 'visible';
            collapseBtn.style.opacity = '1';
            collapseBtn.style.display = 'flex';
            collapseBtn.style.zIndex = '999';
        }
    }, 100);
    
    // Watch for sidebar state changes and ensure button stays visible
    const observer = new MutationObserver(function() {
        const collapseBtn = document.querySelector('[data-testid="stSidebarCollapseButton"]');
        if (collapseBtn) {
            collapseBtn.style.visibility = 'visible';
            collapseBtn.style.opacity = '1';
            collapseBtn.style.zIndex = '999';
        }
    });
    
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
</script>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-tag"><span class="hero-dot"></span> Machine Learning · Binary Classification</div>
        <h1 class="hero-title">Wine Quality<br>Classification</h1>
        <p class="hero-desc">Classify wine quality as Good or Bad using physicochemical properties.<br>Compare 5 ML models with interactive hyperparameter tuning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-icon gradient-violet">📊</div>
            <div class="nav-card-title">Explore Data</div>
            <div class="nav-card-desc">Interactive EDA with histograms, scatter plots, and correlation analysis</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Data →", key="nav_eda", use_container_width=True):
            st.session_state['nav_target'] = 'EDA'
            st.rerun()

    with col2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-icon gradient-blue">📝</div>
            <div class="nav-card-title">Understand Dataset</div>
            <div class="nav-card-desc">Dataset description, feature dictionary, and classification workflow</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Description →", key="nav_desc", use_container_width=True):
            st.session_state['nav_target'] = 'Description Page'
            st.rerun()

    with col3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-icon gradient-emerald">🧠</div>
            <div class="nav-card-title">Train Model</div>
            <div class="nav-card-desc">Select models, tune hyperparameters, compare performance</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Train Model →", key="nav_train", use_container_width=True):
            st.session_state['nav_target'] = 'Train Your Model'
            st.rerun()

    with col4:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-icon gradient-rose">🎯</div>
            <div class="nav-card-title">Classify Wine</div>
            <div class="nav-card-desc">Input wine properties and classify quality as Good or Bad</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Classify →", key="nav_pred", use_container_width=True):
            st.session_state['nav_target'] = 'Result / Prediction Demo'
            st.rerun()

    # Problem Statement + Dataset Overview
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("""
        ### 📋 Problem Statement

        Menentukan kualitas wine secara objektif merupakan tantangan di industri wine.
        Penilaian tradisional bergantung pada sommelier (ahli wine) yang subjektif dan mahal.
        Dengan analisis kimia, kita bisa membangun model yang mengklasifikasikan wine secara otomatis.

        ### 🎯 Tujuan

        Membangun model **Machine Learning Classification** yang mampu mengklasifikasikan
        kualitas wine (**Good** / **Bad**) berdasarkan sifat fisikokimianya.

        ### 🔬 Pendekatan

        - Dataset **Wine Quality** dari UCI ML Repository
        - Binary Classification: Quality ≥ 7 = **Good**, < 7 = **Bad**
        - 5 Model: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, XGBoost
        - Evaluasi: Accuracy, Precision, Recall, F1-Score, AUC-ROC
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        df = load_data()
        good_pct = (df['Quality Label'] == 'Good').mean() * 100
        st.markdown(f"""
        <div class="metric-card metric-violet" style="margin-bottom:0.75rem;">
            <div class="value">{df.shape[0]:,}</div>
            <div class="label">Total Samples</div>
        </div>
        <div class="metric-card metric-blue" style="margin-bottom:0.75rem;">
            <div class="value">11</div>
            <div class="label">Features</div>
        </div>
        <div class="metric-card metric-green" style="margin-bottom:0.75rem;">
            <div class="value">2</div>
            <div class="label">Classes (Good / Bad)</div>
        </div>
        <div class="metric-card metric-amber">
            <div class="value">{good_pct:.1f}%</div>
            <div class="label">Good Wine</div>
        </div>
        """, unsafe_allow_html=True)

    # App Structure
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("""
    ### 🗂️ Application Structure

    | Page | Description |
    |------|-------------|
    | **Home** | Overview dan problem statement |
    | **EDA** | Interactive Exploratory Data Analysis |
    | **Description Page** | Dataset overview dan feature dictionary |
    | **Data Preprocessing** | Scaling, outlier handling, train-test split |
    | **Train Your Model** | Model training lab dengan hyperparameter tuning |
    | **Result / Prediction Demo** | Input data dan klasifikasi kualitas wine |
    | **Feature Importance** | Feature importance ranking dan visualisasi |
    | **About Us** | Team members dan project info |
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: EDA
# ============================================================
elif page == "EDA":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon page-icon-violet">📊</div>
        <div><div class="page-header-title">Exploratory Data Analysis</div>
        <div class="page-header-sub">Interactive visualizations to understand the dataset</div></div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ['Quality', 'Quality Label']]

    # 1. Descriptive Statistics
    st.markdown('<div class="section-title"><div class="section-badge badge-violet">1</div><span class="section-label">Descriptive Statistics</span></div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")

    # 2. Target Distribution
    st.markdown('<div class="section-title"><div class="section-badge badge-violet">2</div><span class="section-label">Target Distribution (Class Balance)</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['Quality Label'].value_counts()
        colors = ['#EF4444', '#10B981']
        ax.bar(counts.index, counts.values, color=colors, width=0.5)
        ax.set_xlabel('Quality Label')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution: Good vs Bad')
        for i, (label, val) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, val + 50, str(val), ha='center', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        quality_counts = df['Quality'].value_counts().sort_index()
        ax.bar(quality_counts.index, quality_counts.values, color='#7C3AED', width=0.6)
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Count')
        ax.set_title('Original Quality Score Distribution')
        ax.axvline(x=6.5, color='red', linestyle='--', label='Threshold (≥7 = Good)')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown(f'<div class="banner-warning">⚠️ <strong>Class Balance:</strong> Good = {counts.get("Good", 0)} ({counts.get("Good", 0)/len(df)*100:.1f}%) | Bad = {counts.get("Bad", 0)} ({counts.get("Bad", 0)/len(df)*100:.1f}%) — Dataset ini <strong>imbalanced</strong>.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 3. Histogram Generator
    st.markdown('<div class="section-title"><div class="section-badge badge-violet">3</div><span class="section-label">Histogram Generator</span></div>', unsafe_allow_html=True)
    hist_options = ["Show All Features"] + feature_cols
    hist_feature = st.selectbox("Select feature:", hist_options, index=11, key="hist_feature")
    hist_bins = st.slider("Number of bins:", 10, 50, 25, key="hist_bins")

    if hist_feature == "Show All Features":
        n_cols = 3
        n_rows = (len(feature_cols) + 2) // 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(feature_cols):
            sns.histplot(df[col], bins=hist_bins, kde=True, color='#7C3AED', ax=axes[i])
            axes[i].set_title(f'{col}', fontsize=10)
            axes[i].set_xlabel('')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df[hist_feature], bins=hist_bins, kde=True, color='#7C3AED', ax=ax)
        ax.set_xlabel(hist_feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {hist_feature}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 4. Correlation Heatmap
    st.markdown('<div class="section-title"><div class="section-badge badge-violet">4</div><span class="section-label">Correlation Heatmap</span></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[feature_cols + ['Quality']].corr()
    sns.heatmap(corr, annot=True, cmap='Purples', fmt='.2f', ax=ax, square=True)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # 5. Boxplot Generator
    st.markdown('<div class="section-title"><div class="section-badge badge-violet">5</div><span class="section-label">Boxplot Generator (by Quality Label)</span></div>', unsafe_allow_html=True)
    box_options = ["Show All Features"] + feature_cols
    box_feature = st.selectbox("Select feature:", box_options, index=11, key="box_feature")

    if box_feature == "Show All Features":
        n_cols = 3
        n_rows = (len(feature_cols) + 2) // 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(feature_cols):
            sns.boxplot(data=df, x='Quality Label', y=col, palette=['#EF4444', '#10B981'], ax=axes[i])
            axes[i].set_title(f'{col}', fontsize=10)
            axes[i].set_xlabel('')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x='Quality Label', y=box_feature, palette=['#EF4444', '#10B981'], ax=ax)
        ax.set_title(f'{box_feature} by Quality Label')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 6. Scatter Plot Generator
    st.markdown('<div class="section-title"><div class="section-badge badge-violet">6</div><span class="section-label">Scatter Plot Generator</span></div>', unsafe_allow_html=True)
    scatter_mode = st.radio("Mode:", ["Single Pair", "Show All vs Alcohol"], horizontal=True, key="scatter_mode")

    if scatter_mode == "Single Pair":
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox("X-axis:", feature_cols, index=10, key="scatter_x")
        with col2:
            scatter_y = st.selectbox("Y-axis:", feature_cols, index=7, key="scatter_y")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = df['Quality Label'].map({'Good': '#10B981', 'Bad': '#EF4444'})
        ax.scatter(df[scatter_x], df[scatter_y], alpha=0.4, c=colors, s=20)
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        ax.set_title(f'{scatter_x} vs {scatter_y} (Green=Good, Red=Bad)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()
    else:
        other_features = [c for c in feature_cols if c != 'Alcohol']
        n_cols = 2
        n_rows = (len(other_features) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()
        for i, feat in enumerate(other_features):
            colors = df['Quality Label'].map({'Good': '#10B981', 'Bad': '#EF4444'})
            axes[i].scatter(df[feat], df['Alcohol'], alpha=0.3, c=colors, s=15)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel('Alcohol')
            axes[i].set_title(f'{feat} vs Alcohol')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ============================================================
# PAGE: DESCRIPTION
# ============================================================
elif page == "Description Page":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon page-icon-blue">📝</div>
        <div><div class="page-header-title">Dataset Description</div>
        <div class="page-header-sub">Understanding the Wine Quality dataset</div></div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    st.markdown('<div class="section-title"><div class="section-badge badge-blue">1</div><span class="section-label">Dataset Overview</span></div>', unsafe_allow_html=True)
    st.markdown("""
    | Field | Detail |
    |-------|--------|
    | **Source** | UCI Machine Learning Repository |
    | **Name** | Wine Quality Data Set |
    | **URL** | https://archive.ics.uci.edu/dataset/186/wine+quality |
    | **Creators** | Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis |
    | **Region** | Minho, Portugal (Vinho Verde) |
    | **Year** | 2009 |
    | **License** | CC BY 4.0 |
    """)

    st.markdown("---")

    st.markdown('<div class="section-title"><div class="section-badge badge-blue">2</div><span class="section-label">Feature Dictionary</span></div>', unsafe_allow_html=True)

    features_dict = {
        "🧪 Fixed Acidity": "Continuous | g/dm³ | Non-volatile acids (tartaric acid). Range: 3.8–15.9",
        "💨 Volatile Acidity": "Continuous | g/dm³ | Acetic acid. Too high = vinegar taste. Range: 0.08–1.58",
        "🍋 Citric Acid": "Continuous | g/dm³ | Adds freshness and flavor. Range: 0.0–1.66",
        "🍬 Residual Sugar": "Continuous | g/dm³ | Sugar remaining after fermentation. Range: 0.6–65.8",
        "🧂 Chlorides": "Continuous | g/dm³ | Salt content. Range: 0.009–0.611",
        "💨 Free SO2": "Continuous | mg/dm³ | Free form of SO2 (preservative). Range: 1–289",
        "💨 Total SO2": "Continuous | mg/dm³ | Total SO2 (free + bound). Range: 6–440",
        "⚖️ Density": "Continuous | g/cm³ | Close to water (~1.0). Range: 0.987–1.039",
        "⚗️ pH": "Continuous | Acidity level. Most wines are 3–4 pH. Range: 2.72–4.01",
        "🧪 Sulphates": "Continuous | g/dm³ | Antimicrobial and antioxidant. Range: 0.22–2.0",
        "🍺 Alcohol": "Continuous | % vol | Alcohol content. Range: 8.0–14.9",
        "⭐ Quality (TARGET)": "Integer → Binary | Quality ≥ 7 = Good 🟢 | Quality < 7 = Bad 🔴",
    }
    for name, desc in features_dict.items():
        with st.expander(name):
            st.markdown(desc)

    st.markdown("---")

    st.markdown('<div class="section-title"><div class="section-badge badge-blue">3</div><span class="section-label">Raw Data Preview</span></div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="section-title"><div class="section-badge badge-blue">4</div><span class="section-label">Data Shape & Info</span></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card metric-violet"><div class="value">{df.shape[0]:,}</div><div class="label">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card metric-blue"><div class="value">{df.shape[1]}</div><div class="label">Columns</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card metric-green"><div class="value">{df.isnull().sum().sum()}</div><div class="label">Missing Values</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-title"><div class="section-badge badge-blue">5</div><span class="section-label">Classification Workflow</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; padding: 1rem 0;">
        <span class="workflow-step">📦 Raw Data</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🔍 EDA</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">⚙️ Preprocessing</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">✂️ Train/Test Split</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🧠 Model Training</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">📊 Evaluation</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🎯 Classification</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Models:** Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, XGBoost

    **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve / AUC
    """)


# ============================================================
# PAGE: DATA PREPROCESSING
# ============================================================
elif page == "Data Preprocessing":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon page-icon-amber">⚙️</div>
        <div><div class="page-header-title">Data Preprocessing</div>
        <div class="page-header-sub">Prepare data for model training</div></div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ['Quality', 'Quality Label']]

    # Step 1
    st.markdown('<div class="section-title"><div class="section-badge badge-amber">1</div><span class="section-label">Raw Data Viewer</span></div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Step 2
    st.markdown('<div class="section-title"><div class="section-badge badge-amber">2</div><span class="section-label">Missing Values Check</span></div>', unsafe_allow_html=True)
    missing = df[feature_cols].isnull().sum()
    missing_df = pd.DataFrame({'Feature': missing.index, 'Missing Count': missing.values})
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    st.markdown('<div class="banner-success"><span class="banner-dot"></span><span class="banner-text">No missing values found in the dataset!</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Step 3
    st.markdown('<div class="section-title"><div class="section-badge badge-amber">3</div><span class="section-label">Class Balance Check</span></div>', unsafe_allow_html=True)
    class_counts = df['Quality Label'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card metric-green"><div class="value">{class_counts.get("Good", 0)}</div><div class="label">Good Wine ({class_counts.get("Good", 0)/len(df)*100:.1f}%)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card metric-rose"><div class="value">{class_counts.get("Bad", 0)}</div><div class="label">Bad Wine ({class_counts.get("Bad", 0)/len(df)*100:.1f}%)</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="banner-warning" style="margin-top:0.75rem;">⚠️ Dataset is imbalanced. Use F1/Precision/Recall, not just Accuracy.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Step 4
    st.markdown('<div class="section-title"><div class="section-badge badge-amber">4</div><span class="section-label">Preprocessing Configurator</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Train/Test Split (test %):", 10, 40, 20, 5)
        scaler_choice = st.selectbox("Scaling Method:", ["StandardScaler", "MinMaxScaler"])
    with col2:
        outlier_method = st.selectbox("Outlier Handling:", ["None", "IQR Removal"])
        random_state = st.number_input("Random State:", value=42, min_value=0, max_value=999)
        class_balance = st.selectbox("Class Balancing:", ["Undersample (Equal Count)", "Balanced (Class Weight)", "None"],
                                     help="'Undersample' = random sample Bad agar sama dengan Good. 'Balanced' = beri bobot lebih ke minoritas.")

    st.markdown("---")

    # Step 5
    st.markdown('<div class="section-title"><div class="section-badge badge-amber">5</div><span class="section-label">Apply Preprocessing</span></div>', unsafe_allow_html=True)

    if st.button("▶️ Run Preprocessing", use_container_width=True):
        X = df[feature_cols]
        y = (df['Quality'] >= 7).astype(int)

        if outlier_method == "IQR Removal":
            mask = pd.Series([True] * len(df))
            for col in feature_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR_val = Q3 - Q1
                mask = mask & (X[col] >= Q1 - 1.5 * IQR_val) & (X[col] <= Q3 + 1.5 * IQR_val)
            X = X[mask]
            y = y[mask]
            st.info(f"IQR Outlier Removal: {len(df) - mask.sum()} removed. Remaining: {mask.sum()}.")

        if class_balance == "Undersample (Equal Count)":
            good_idx = y[y == 1].index
            bad_idx = y[y == 0].index
            n_good = len(good_idx)
            bad_sampled_idx = np.random.RandomState(random_state).choice(bad_idx, size=n_good, replace=False)
            balanced_idx = np.concatenate([good_idx.values, bad_sampled_idx])
            X = X.loc[balanced_idx]
            y = y.loc[balanced_idx]
            st.info(f"Undersample: Bad reduced from {len(bad_idx)} to {n_good}. Total: {len(balanced_idx)} (50:50).")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state, stratify=y)
        scaler = StandardScaler() if scaler_choice == "StandardScaler" else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state['X_train_scaled'] = X_train_scaled
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler
        st.session_state['scaler_name'] = scaler_choice
        st.session_state['feature_names'] = feature_cols
        st.session_state['preprocessing_done'] = True
        st.session_state['class_weight'] = 'balanced' if class_balance == "Balanced (Class Weight)" else None

        st.markdown("---")
        st.markdown('<div class="section-title"><div class="section-badge badge-amber">6</div><span class="section-label">Results</span></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card metric-violet"><div class="value">{X_train.shape[0]}</div><div class="label">Training Samples</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card metric-blue"><div class="value">{X_test.shape[0]}</div><div class="label">Testing Samples</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card metric-green"><div class="value">{scaler_choice[:8]}</div><div class="label">Scaler</div></div>', unsafe_allow_html=True)

        st.markdown("**Before Scaling (first 5 rows):**")
        st.dataframe(X_train.head(), use_container_width=True)
        st.markdown(f"**After {scaler_choice} (first 5 rows):**")
        st.dataframe(pd.DataFrame(X_train_scaled[:5], columns=feature_cols), use_container_width=True)

        st.markdown('<div class="banner-success" style="margin-top:1rem;"><span class="banner-dot"></span><span class="banner-text">Preprocessing completed!</span></div>', unsafe_allow_html=True)


# ============================================================
# PAGE: TRAIN YOUR MODEL
# ============================================================
elif page == "Train Your Model":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon page-icon-green">🧠</div>
        <div><div class="page-header-title">Train Your Model</div>
        <div class="page-header-sub">Select, tune, and compare classification models</div></div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ['Quality', 'Quality Label']]

    if 'preprocessing_done' not in st.session_state:
        st.markdown('<div class="banner-warning">⚠️ Preprocessing not yet applied. Using default settings.</div>', unsafe_allow_html=True)
        X = df[feature_cols]
        y = (df['Quality'] >= 7).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        feature_names = feature_cols
    else:
        X_train_scaled = st.session_state['X_train_scaled']
        X_test_scaled = st.session_state['X_test_scaled']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        st.markdown(f'<div class="banner-success"><span class="banner-dot"></span><span class="banner-text">Using preprocessed data ({st.session_state["scaler_name"]})</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    model_options = ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "Gradient Boosting Classifier"]
    if XGBOOST_AVAILABLE:
        model_options.append("XGBoost Classifier")
    model_choice = st.selectbox("Choose a model:", model_options)

    st.markdown("---")

    cw = st.session_state.get('class_weight', None)
    use_balanced = st.checkbox("Use class_weight='balanced'", value=(cw == 'balanced'),
                              help="Memberi bobot lebih ke kelas minoritas.")
    class_weight_param = 'balanced' if use_balanced else None

    if model_choice == "Logistic Regression":
        col1, col2 = st.columns(2)
        with col1:
            C_val = st.slider("C (Regularization):", 0.01, 10.0, 1.0, 0.01)
        with col2:
            max_iter = st.slider("max_iter:", 100, 1000, 200, 50)
        model = LogisticRegression(C=C_val, max_iter=max_iter, random_state=42, class_weight=class_weight_param)
    elif model_choice == "Random Forest Classifier":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("n_estimators:", 10, 300, 100, 10)
        with col2:
            max_depth = st.slider("max_depth:", 1, 30, 10, 1)
        with col3:
            min_samples_split = st.slider("min_samples_split:", 2, 20, 2, 1)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42, class_weight=class_weight_param)
    elif model_choice == "Decision Tree Classifier":
        col1, col2 = st.columns(2)
        with col1:
            max_depth_dt = st.slider("max_depth:", 1, 30, 5, 1)
        with col2:
            min_samples_split_dt = st.slider("min_samples_split:", 2, 20, 2, 1)
        model = DecisionTreeClassifier(max_depth=max_depth_dt, min_samples_split=min_samples_split_dt, random_state=42, class_weight=class_weight_param)
    elif model_choice == "Gradient Boosting Classifier":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators_gb = st.slider("n_estimators:", 50, 300, 100, 10)
        with col2:
            learning_rate_gb = st.slider("learning_rate:", 0.01, 0.5, 0.1, 0.01)
        with col3:
            max_depth_gb = st.slider("max_depth:", 1, 15, 3, 1)
        model = GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=learning_rate_gb, max_depth=max_depth_gb, random_state=42)
    elif model_choice == "XGBoost Classifier" and XGBOOST_AVAILABLE:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators_xgb = st.slider("n_estimators:", 50, 300, 100, 10)
        with col2:
            learning_rate_xgb = st.slider("learning_rate:", 0.01, 0.5, 0.1, 0.01)
        with col3:
            max_depth_xgb = st.slider("max_depth:", 1, 15, 3, 1)
        if use_balanced:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            spw = neg_count / pos_count if pos_count > 0 else 1
        else:
            spw = 1
        model = XGBClassifier(n_estimators=n_estimators_xgb, learning_rate=learning_rate_xgb, max_depth=max_depth_xgb, random_state=42, verbosity=0, eval_metric='logloss', scale_pos_weight=spw)

    st.markdown("---")
    model_name_input = st.text_input("Model name:", value=model_choice)

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

        st.markdown("---")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card metric-violet"><div class="value">{acc:.4f}</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card metric-blue"><div class="value">{prec:.4f}</div><div class="label">Precision</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card metric-green"><div class="value">{rec:.4f}</div><div class="label">Recall</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card metric-amber"><div class="value">{f1:.4f}</div><div class="label">F1-Score</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Confusion Matrix + ROC side by side
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax, xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_roc:
            if y_prob is not None:
                st.markdown("**ROC Curve**")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.plot(fpr, tpr, color='#7C3AED', lw=2, label=f'AUC = {roc_auc:.4f}')
                ax.plot([0, 1], [0, 1], color='#cbd5e1', linestyle='--', lw=1, label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc='lower right', fontsize=8)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1.05])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with st.expander("ℹ️ Cara baca Confusion Matrix & ROC"):
            st.markdown(f"""
            | | Predicted Bad | Predicted Good |
            |---|---|---|
            | **Actual Bad** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |
            | **Actual Good** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |

            **ROC:** Semakin mendekati sudut kiri atas = semakin baik.
            """)

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.markdown("**Feature Importance**")
            importance = model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#7C3AED')
            ax.set_xlabel('Importance')
            max_val = feat_imp['Importance'].max()
            ax.set_xlim(0, max_val * 1.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_name_input
        st.session_state['scaler'] = scaler
        st.session_state['metrics'] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        st.session_state['feature_names'] = feature_names

    st.markdown("---")

    # Compare All Models
    st.markdown("### 📊 Model Comparison Leaderboard")
    if st.button("🔄 Compare All Models", use_container_width=True):
        with st.spinner("Training all models..."):
            all_models = {
                'Logistic Regression': LogisticRegression(max_iter=200, random_state=42, class_weight=class_weight_param),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight=class_weight_param),
                'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42, class_weight=class_weight_param),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            }
            if XGBOOST_AVAILABLE:
                neg_c = (y_train == 0).sum()
                pos_c = (y_train == 1).sum()
                spw_cmp = neg_c / pos_c if (pos_c > 0 and use_balanced) else 1
                all_models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0, eval_metric='logloss', scale_pos_weight=spw_cmp)

            results = []
            for name, m in all_models.items():
                m.fit(X_train_scaled, y_train)
                pred = m.predict(X_test_scaled)
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy_score(y_test, pred), 4),
                    'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
                    'Recall': round(recall_score(y_test, pred, zero_division=0), 4),
                    'F1-Score': round(f1_score(y_test, pred, zero_division=0), 4)
                })
            results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#7C3AED' if i == 0 else '#C4B5FD' for i in range(len(results_df))]
        ax.bar(range(len(results_df)), results_df['F1-Score'], color=colors, width=0.5)
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df['Model'])
        ax.set_ylabel('F1-Score')
        ax.set_ylim(0, 1.15)
        for i, v in enumerate(results_df['F1-Score']):
            ax.text(i, v + 0.02, str(v), ha='center', fontweight='bold', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Export
    st.markdown("### 💾 Model Export")
    if 'trained_model' in st.session_state:
        model_bytes = pickle.dumps(st.session_state['trained_model'])
        st.download_button("⬇️ Download Trained Model (.pkl)", data=model_bytes, file_name=f"{st.session_state.get('model_name', 'model')}.pkl", mime="application/octet-stream", use_container_width=True)
    else:
        st.info("Train a model first to enable download.")


# ============================================================
# PAGE: PREDICTION DEMO
# ============================================================
elif page == "Result / Prediction Demo":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon page-icon-rose">🎯</div>
        <div><div class="page-header-title">Prediction Demo</div>
        <div class="page-header-sub">Input wine properties and classify quality</div></div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ['Quality', 'Quality Label']]

    if 'trained_model' not in st.session_state:
        st.markdown('<div class="banner-warning">⚠️ No model trained yet. Train a model first or use the default below.</div>', unsafe_allow_html=True)
        if st.button("Train Default Model (Random Forest)", use_container_width=True):
            X = df[feature_cols]
            y = (df['Quality'] >= 7).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['model_name'] = "Random Forest (Default)"
            st.session_state['feature_names'] = feature_cols
            y_pred = model.predict(scaler.transform(X_test))
            st.session_state['metrics'] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0)
            }
            st.rerun()
    else:
        # Active model banner
        st.markdown(f"""
        <div class="banner-success">
            <span class="banner-dot"></span>
            <span class="banner-text">Active Model: {st.session_state['model_name']}</span>
        </div>
        """, unsafe_allow_html=True)

        if 'metrics' in st.session_state:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card metric-violet"><div class="value">{st.session_state["metrics"]["Accuracy"]:.4f}</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card metric-blue"><div class="value">{st.session_state["metrics"]["Precision"]:.4f}</div><div class="label">Precision</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card metric-green"><div class="value">{st.session_state["metrics"]["Recall"]:.4f}</div><div class="label">Recall</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card metric-amber"><div class="value">{st.session_state["metrics"]["F1"]:.4f}</div><div class="label">F1-Score</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        with st.expander("ℹ️ Parameter Guide"):
            st.markdown("""
            | Parameter | Explanation |
            |-----------|-------------|
            | **Fixed Acidity** | Keasaman tetap. 4-12 g/dm³ |
            | **Volatile Acidity** | Keasaman menguap. <0.5 = baik |
            | **Citric Acid** | Asam sitrat. 0.0-0.8 |
            | **Residual Sugar** | Gula sisa. <4 = dry wine |
            | **Chlorides** | Kadar garam. 0.01-0.2 |
            | **Free SO2** | SO2 bebas. 1-70 |
            | **Total SO2** | Total SO2. 6-200 |
            | **Density** | ~0.99-1.00 g/cm³ |
            | **pH** | 2.8-3.8 |
            | **Sulphates** | 0.3-1.5 |
            | **Alcohol** | 8-15% vol |
            """)

        col1, col2, col3 = st.columns(3)
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", 3.0, 16.0, 7.0, 0.1)
            volatile_acidity = st.number_input("Volatile Acidity (g/dm³)", 0.0, 2.0, 0.3, 0.01)
            citric_acid = st.number_input("Citric Acid (g/dm³)", 0.0, 2.0, 0.3, 0.01)
            residual_sugar = st.number_input("Residual Sugar (g/dm³)", 0.0, 70.0, 5.0, 0.5)
        with col2:
            chlorides = st.number_input("Chlorides (g/dm³)", 0.0, 1.0, 0.05, 0.005, format="%.3f")
            free_so2 = st.number_input("Free SO2 (mg/dm³)", 0.0, 300.0, 30.0, 5.0)
            total_so2 = st.number_input("Total SO2 (mg/dm³)", 0.0, 450.0, 120.0, 10.0)
            density = st.number_input("Density (g/cm³)", 0.98, 1.05, 0.995, 0.001, format="%.4f")
        with col3:
            pH = st.number_input("pH", 2.5, 4.5, 3.2, 0.01)
            sulphates = st.number_input("Sulphates (g/dm³)", 0.0, 2.5, 0.5, 0.05)
            alcohol = st.number_input("Alcohol (% vol)", 7.0, 16.0, 10.5, 0.1)

        st.markdown("---")

        if st.button("🔮 Classify Wine Quality", use_container_width=True):
            input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_so2, total_so2, density, pH, sulphates, alcohol]])
            input_scaled = st.session_state['scaler'].transform(input_data)
            prediction = st.session_state['trained_model'].predict(input_scaled)[0]
            prob = st.session_state['trained_model'].predict_proba(input_scaled)[0] if hasattr(st.session_state['trained_model'], 'predict_proba') else None

            st.session_state['last_prediction'] = prediction
            st.session_state['last_prob'] = prob
            st.session_state['last_input'] = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                              chlorides, free_so2, total_so2, density, pH, sulphates, alcohol]
            st.session_state['last_input_scaled'] = input_scaled

        if 'last_prediction' in st.session_state:
            prediction = st.session_state['last_prediction']
            prob = st.session_state['last_prob']
            last_input = st.session_state['last_input']

            st.markdown("---")

            col1, col2 = st.columns([1, 1])
            with col1:
                if prediction == 1:
                    st.markdown('<div class="result-good"><p class="result-label-good">Good 🟢</p><p class="result-subtitle">Quality ≥ 7</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-bad"><p class="result-label-bad">Bad 🔴</p><p class="result-subtitle">Quality < 7</p></div>', unsafe_allow_html=True)

                if prob is not None:
                    st.markdown(f"""
                    <div class="confidence-wrap">
                        <div class="confidence-label" style="color: #e11d48;">Bad: {prob[0]*100:.1f}%</div>
                        <div class="progress-track"><div class="progress-fill progress-bad" style="width: {prob[0]*100}%;"></div></div>
                        <div class="confidence-label" style="color: #059669; margin-top: 0.5rem;">Good: {prob[1]*100:.1f}%</div>
                        <div class="progress-track"><div class="progress-fill progress-good" style="width: {prob[1]*100}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Input Summary**")
                input_summary = pd.DataFrame({
                    'Feature': ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
                               'Chlorides', 'Free SO2', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol'],
                    'Value': last_input
                })
                st.dataframe(input_summary, use_container_width=True, hide_index=True)

            with st.expander("🔧 Prediction Details"):
                st.markdown(f"**Model:** {st.session_state['model_name']}")
                if 'metrics' in st.session_state:
                    st.markdown(f"**Accuracy:** {st.session_state['metrics']['Accuracy']:.4f} | **F1:** {st.session_state['metrics']['F1']:.4f}")


# ============================================================
# PAGE: FEATURE IMPORTANCE
# ============================================================
elif page == "Feature Importance":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon page-icon-amber">📊</div>
        <div><div class="page-header-title">Feature Importance</div>
        <div class="page-header-sub">Which features matter most for wine quality?</div></div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ['Quality', 'Quality Label']]
    X = df[feature_cols]
    y = (df['Quality'] >= 7).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models_with_importance = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models_with_importance['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0, eval_metric='logloss')

    model_colors = {'Random Forest': '#7c3aed', 'Gradient Boosting': '#2563eb', 'XGBoost': '#059669', 'Decision Tree': '#d97706'}

    importance_data = {}
    for name, m in models_with_importance.items():
        m.fit(X_train_scaled, y_train)
        importance_data[name] = m.feature_importances_

    model_select = st.selectbox("Select model:", list(importance_data.keys()))
    bar_color = model_colors.get(model_select, '#7c3aed')

    imp = importance_data[model_select]
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': imp}).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=bar_color)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Feature Importance — {model_select}')
    max_val = feat_imp['Importance'].max()
    ax.set_xlim(0, max_val * 1.25)
    for bar, val in zip(ax.patches, feat_imp['Importance']):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    st.markdown("### Comparison Table")
    comparison_df = pd.DataFrame(importance_data, index=feature_cols).round(4)
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("---")

    st.markdown("### Key Insights")
    selected_imp = pd.Series(importance_data[model_select], index=feature_cols).sort_values(ascending=False)
    st.markdown(f"""
    Based on **{model_select}**:
    - 🥇 **Most Important:** {selected_imp.index[0]} ({selected_imp.values[0]:.4f})
    - 🥈 **Second:** {selected_imp.index[1]} ({selected_imp.values[1]:.4f})
    - 🥉 **Third:** {selected_imp.index[2]} ({selected_imp.values[2]:.4f})

    Alcohol content, volatile acidity, and sulphates tend to be the strongest predictors.
    """)


# ============================================================
# PAGE: ABOUT US
# ============================================================
elif page == "About Us":
    st.markdown("""
    <div class="about-banner">
        <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🍷</div>
        <h2 class="about-title">Group 4</h2>
        <p class="about-sub">Machine Learning Assignment — Binus University</p>
        <p class="about-sub2">Binusian 2028</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎓 Team Members")
    members = [
        {"nim": "2802451541", "name": "Michael Peterson", "icon": "👨‍💻"},
        {"nim": "2802451655", "name": "Jos Hwen Sim", "icon": "👨‍💻"},
        {"nim": "2802461832", "name": "Jonathan Kuniardi", "icon": "👨‍💻"},
    ]
    col1, col2, col3 = st.columns(3)
    for i, m in enumerate(members):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="member-card">
                <div class="member-icon">{m['icon']}</div>
                <div class="member-name">{m['name']}</div>
                <div class="member-nim">{m['nim']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div class="member-card">
            <div style="font-size: 2.5rem;">🎓</div>
            <div class="member-name" style="margin-top: 0.5rem;">Binus University</div>
            <div class="member-nim">Binusian 2028</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        | Detail | Info |
        |--------|------|
        | **University** | Binus University |
        | **Program** | Computer Science |
        | **Batch** | Binusian 2028 |
        | **Course** | Machine Learning |
        | **Assignment** | Group Project — Classification |
        | **Semester** | Even Semester 2025/2026 |
        """)

    st.markdown("---")

    st.markdown("### 📋 Project Information")
    st.markdown("""
    | Item | Detail |
    |------|--------|
    | **Project Title** | Wine Quality Classification using Machine Learning |
    | **Type** | Binary Classification (Supervised Learning) |
    | **Dataset** | UCI ML Repository — Wine Quality (ID: 186) |
    | **Models** | Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, XGBoost |
    | **Metrics** | Accuracy, Precision, Recall, F1-Score, AUC-ROC |
    | **Framework** | Streamlit (Python) |
    """)

    st.markdown("---")

    st.markdown("### 🛠️ Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    for col, (name, desc) in zip([col1, col2, col3, col4], [("🐍 Python", "Core Language"), ("📊 Streamlit", "Web Framework"), ("🧠 Scikit-learn", "ML Library"), ("🚀 XGBoost", "Boosting Library")]):
        with col:
            st.markdown(f'<div class="tech-card"><div class="tech-name">{name}</div><div class="tech-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="footer">© 2026 Group 4 — Binus University | Machine Learning Assignment<br>Built with ❤️ using Python & Streamlit</div>', unsafe_allow_html=True)
