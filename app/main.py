import sys
import os

# Ensure project root is on the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tools.dataset_profiler import profile_dataset
from tools.data_validator import validate_dataset
from tools.data_filter import filter_dataset
from tools.auto_eda import generate_eda_summary
from tools.model_trainer import run_baseline_ml
from agent.agent import AutoDSAgent

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Spidey DATA SCIENTIST — Autonomous Data Scientist",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Miles Morales Red & Black Theme & Styling ─────────────────────
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseVenomRed {
        0% { box-shadow: 0 0 12px rgba(255, 23, 68, 0.3); }
        50% { box-shadow: 0 0 32px rgba(255, 23, 68, 0.7); }
        100% { box-shadow: 0 0 12px rgba(255, 23, 68, 0.3); }
    }

    @keyframes pulseDotRed {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Base App Background - Deep Obsidian & Venom Red */
    .stApp {
        background: radial-gradient(circle at 50% -10%, #2A0913 0%, #13070E 40%, #090A0F 95%);
        font-family: 'Inter', sans-serif;
        color: #f5f5f7;
    }

    /* Launchpad Hero Badge */
    .launchpad-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(45, 10, 20, 0.75);
        border: 1px solid rgba(255, 23, 68, 0.5);
        padding: 0.45rem 1.2rem;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #FF1744;
        text-transform: uppercase;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.25);
        backdrop-filter: blur(12px);
        margin-bottom: 0.8rem;
    }

    /* Main Title & Glossy Header */
    .main-header {
        background: linear-gradient(135deg, #FF6B8B 0%, #FF1744 45%, #B70928 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.3rem;
        font-weight: 900;
        margin-bottom: 0;
        line-height: 1.15;
        letter-spacing: -1px;
        animation: fadeIn 0.5s ease-out;
    }

    .sub-header {
        color: #FFA4B6;
        font-size: 1.08rem;
        margin-top: -0.2rem;
        margin-bottom: 2.2rem;
        font-weight: 400;
        opacity: 0.9;
        animation: fadeIn 0.7s ease-out;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(35, 14, 22, 0.65) 0%, rgba(16, 10, 18, 0.85) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 23, 68, 0.35);
        border-top: 3px solid #FF1744;
        border-radius: 18px;
        padding: 1.4rem 1.1rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.65);
        animation: fadeIn 0.5s ease-out;
    }

    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: #FF5252;
        box-shadow: 0 14px 35px rgba(255, 23, 68, 0.4);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FF1744;
        letter-spacing: -0.5px;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #FFA4B6;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-top: 0.4rem;
        font-weight: 600;
        opacity: 0.85;
    }

    /* Quality Badges */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 1.3rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        animation: fadeIn 0.5s ease-out;
    }

    .badge-good {
        background: rgba(255, 23, 68, 0.18);
        color: #FF5252;
        border: 1px solid rgba(255, 23, 68, 0.5);
        animation: pulseVenomRed 3s infinite ease-in-out;
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }

    .badge-error {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.5);
    }

    /* Issue Items */
    .issue-item {
        padding: 0.85rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        font-size: 0.93rem;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.4s ease-out;
    }

    .issue-error {
        background: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        color: #fca5a5;
    }

    .issue-warning {
        background: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        color: #fef08a;
    }

    .issue-info {
        background: rgba(0, 229, 255, 0.12);
        border-left: 4px solid #00E5FF;
        color: #E0F7FA;
    }

    .changelog-item {
        padding: 0.75rem 1.2rem;
        background: rgba(255, 23, 68, 0.12);
        border-left: 4px solid #FF1744;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-size: 0.94rem;
        color: #FFA4B6;
        animation: fadeIn 0.4s ease-out;
    }

    /* AI Insight Box */
    .ai-box {
        background: linear-gradient(135deg, rgba(35, 14, 22, 0.7) 0%, rgba(16, 10, 18, 0.9) 100%);
        border: 1px solid rgba(255, 23, 68, 0.45);
        border-radius: 18px;
        padding: 1.8rem;
        margin: 1.4rem 0;
        box-shadow: 0 10px 35px rgba(255, 23, 68, 0.18);
        backdrop-filter: blur(18px);
        animation: fadeIn 0.6s ease-out;
    }

    .ai-label {
        color: #FF1744;
        font-weight: 800;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .pulse-dot-red {
        width: 10px;
        height: 10px;
        background-color: #FF1744;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #FF1744;
        animation: pulseDotRed 1.5s infinite;
    }

    .step-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 23, 68, 0.45), transparent);
        margin: 2.8rem 0;
    }

    /* Streamlit File Uploader Button Fix */
    [data-testid="stFileUploader"] section button {
        border-color: #FF1744 !important;
        color: #fafafa !important;
        background-color: rgba(45, 10, 20, 0.75) !important;
    }
    [data-testid="stFileUploader"] section button * {
        letter-spacing: normal !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: rgba(26, 15, 22, 0.5);
        border: 1px solid rgba(255, 23, 68, 0.28);
        border-radius: 14px;
        transition: border-color 0.3s ease;
    }

    div[data-testid="stExpander"]:hover {
        border-color: rgba(255, 23, 68, 0.55);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        background-color: #160B12;
        border: 1px solid rgba(255, 23, 68, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 23, 68, 0.25) !important;
        border-color: #FF1744 !important;
        color: #FF5252 !important;
    }

    /* Left Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0B10 0%, #120C14 50%, #1D0A13 100%) !important;
        border-right: 1px solid rgba(255, 23, 68, 0.35) !important;
        box-shadow: 10px 0 35px rgba(0, 0, 0, 0.85) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #FF1744 !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
    }

    /* Sidebar file uploader box */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(18, 14, 22, 0.8) !important;
        border: 1px solid rgba(255, 23, 68, 0.3) !important;
        border-radius: 14px !important;
        padding: 8px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Lucide SVG Action Icons Helper ────────────────────────────────
def get_svg_icon(name: str, size: int = 20, color: str = "#FF1744") -> str:
    icons = {
        "rows": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>',
        "columns": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="12" y1="3" x2="12" y2="21"></line></svg>',
        "missing": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
        "duplicates": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>',
        "memory": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>',
        "sparkles": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"></path></svg>',
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        "rocket": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path></svg>',
    }
    return icons.get(name, "")


# ─── Helper Functions ─────────────────────────────────────────────
def render_metric_card(value, label, color="#FF1744", icon_name=None):
    icon_svg = get_svg_icon(icon_name, 22, color) if icon_name else ""
    html_code = (
        f'<div class="metric-card">'
        f'<div style="display:flex;align-items:center;justify-content:center;gap:8px;">'
        f'{icon_svg}'
        f'<div class="metric-value" style="color: {color}">{value}</div>'
        f'</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_quality_badge(score):
    check_icon = get_svg_icon("check", 18, "#FF1744") if score >= 80 else ""
    if score >= 80:
        cls, text = "badge-good", f"{check_icon} Data Quality Score: {score}/100"
    elif score >= 50:
        cls, text = "badge-warning", f"<span class='pulse-dot-red' style='background:#FBBF24;'></span> Data Quality Score: {score}/100"
    else:
        cls, text = "badge-error", f"<span class='pulse-dot-red' style='background:#F87171;'></span> Data Quality Score: {score}/100"
    st.markdown(f'<span class="quality-badge {cls}">{text}</span>', unsafe_allow_html=True)


def render_issue(issue):
    sev = issue["severity"]
    icon = {"error": "🔴", "warning": "🟡", "info": "⚡"}[sev]
    st.markdown(
        f'<div class="issue-item issue-{sev}">{icon} <strong>[{issue["category"]}]</strong> {issue["message"]}</div>',
        unsafe_allow_html=True,
    )


# ─── Header & Launchpad Hero ──────────────────────────────────────
st.markdown('<div class="launchpad-badge"><span class="pulse-dot-red"></span> SPIDEY DATA SCIENTIST v2.0 · MILES MORALES LAUNCHPAD</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🕷️ Spidey DATA SCIENTIST</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Venom-Strike Intelligence Hub — Profile, Clean & Train AutoML Models</p>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Dataset Ingestion")
    uploaded = st.file_uploader(
        "Drag & drop a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: .csv, .xlsx, .xls",
    )

    if uploaded:
        st.success(f"🚀 {uploaded.name} ingested")
        st.markdown("---")
        st.markdown("### 🧭 Launchpad Modules")
        st.markdown("""
        1. **📋 Module 01 — Validation**
        2. **📊 Module 02 — Intelligence & Profiling**
        3. **🧹 Module 03 — Data Transformation**
        4. **🚀 Module 04 — AutoML Model Engine**
        """)


# ─── Main Content ────────────────────────────────────────────────
if not uploaded:
    # Landing page Launchpad Tiles
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Automated Profiling")
        st.markdown("Instantly profile datasets with full statistical summaries, missing value detection, and outlier analysis.")

    with col2:
        st.markdown("### ⚡ AI Decision Engine")
        st.markdown("Leverage Groq-powered Qwen LLM for deep dataset analysis and automated cleaning recommendations.")

    with col3:
        st.markdown("### 🚀 Baseline AutoML")
        st.markdown("Automatically evaluate classification or regression pipelines with instant model comparison.")

    st.info("👈 **Upload a CSV or Excel file in the sidebar to launch Mission Control.**")

else:
    # ── Load Data ─────────────────────────────────────────────────
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    # ── Initialize Agent (cached) ────────────────────────────────
    @st.cache_resource
    def get_agent():
        return AutoDSAgent()

    agent = get_agent()

    # ══════════════════════════════════════════════════════════════
    # MODULE 01: UPLOAD & VALIDATE
    # ══════════════════════════════════════════════════════════════
    st.markdown("## 📋 Module 01 — Data Validation & Health")

    validation = validate_dataset(df, uploaded.name)
    profile = profile_dataset(df)

    # Quality badge + metrics row
    render_quality_badge(validation["score"])
    st.markdown("")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card(f"{profile['rows']:,}", "Total Rows", "#FF1744", "rows")
    with m2:
        render_metric_card(f"{profile['columns']}", "Total Columns", "#FFA4B6", "columns")
    with m3:
        render_metric_card(f"{profile['missing_pct']}%", "Missing Data", "#00E5FF" if profile['missing_pct'] == 0 else "#FBBF24", "missing")
    with m4:
        render_metric_card(f"{profile['duplicates']}", "Duplicates", "#00E5FF" if profile['duplicates'] == 0 else "#FBBF24", "duplicates")
    with m5:
        render_metric_card(f"{profile['memory_mb']} MB", "Memory Size", "#FF5252", "memory")

    # Validation issues
    if validation["issues"]:
        with st.expander(f"🔍 Validation Report ({validation['error_count']} Errors, {validation['warning_count']} Warnings)", expanded=True):
            for issue in validation["issues"]:
                render_issue(issue)
    else:
        st.success("✅ Dataset passed all quality checks!")

    # Data preview
    with st.expander("📄 Raw Data Preview (First 10 Rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MODULE 02: PROFILING & AI ANALYSIS
    # ══════════════════════════════════════════════════════════════
    st.markdown("## 📊 Module 02 — Exploratory Profiling & AI Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Column Meta", "📈 Distributions", "🔗 Correlations", "⚡ AI Analysis"])

    with tab1:
        # Column details table
        col_data = []
        for cd in profile["column_details"]:
            row = {
                "Column": cd["name"],
                "Type": cd["dtype"],
                "Missing": cd["missing"],
                "Missing %": f"{cd['missing_pct']}%",
                "Unique": cd["unique"],
                "Sample Values": ", ".join(cd["sample_values"][:3]),
            }
            if "outliers" in cd:
                row["Outliers"] = cd["outliers"]
            if "skewness" in cd:
                row["Skewness"] = cd["skewness"]
            col_data.append(row)

        st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    with tab2:
        numeric_cols = profile["numeric_columns"]
        if numeric_cols:
            selected_col = st.selectbox("Select Numeric Column to Plot", numeric_cols, key="dist_col")
            
            fig = px.histogram(
                df, x=selected_col, nbins=40,
                title=f"Histogram — {selected_col}",
                color_discrete_sequence=["#FF1744"],
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#FAFAFA"),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_box = px.box(
                df, y=selected_col,
                title=f"Box Plot — Outliers in {selected_col}",
                color_discrete_sequence=["#00E5FF"],
                template="plotly_dark",
            )
            fig_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#FAFAFA"),
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numeric columns available for plotting.")

        # Missing values chart
        missing_data = df.isna().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            fig_missing = px.bar(
                x=missing_data.index, y=missing_data.values,
                title="Missing Values Count by Column",
                labels={"x": "Column", "y": "Missing Count"},
                color_discrete_sequence=["#FF5252"],
                template="plotly_dark",
            )
            fig_missing.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#FAFAFA"),
            )
            st.plotly_chart(fig_missing, use_container_width=True)

    with tab3:
        numeric_df = df.select_dtypes(include="number")
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                title="Correlation Heatmap",
                color_continuous_scale=["#1A0810", "#5C0E20", "#B70928", "#FF1744", "#FFA4B6"],
                template="plotly_dark",
                aspect="auto",
            )
            fig_corr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#FAFAFA"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("At least 2 numeric columns required for correlation matrix.")

    with tab4:
        if st.button("⚡ Generate AI Insight Report", key="ai_analyze"):
            with st.spinner("🧠 Groq AI is analyzing your dataset structure..."):
                light_profile = {k: v for k, v in profile.items() if k != "column_details"}
                light_profile["column_summary"] = [
                    {"name": cd["name"], "dtype": cd["dtype"], "missing": cd["missing"],
                     "unique": cd["unique"], "outliers": cd.get("outliers", 0),
                     "skewness": cd.get("skewness", None)}
                    for cd in profile["column_details"]
                ]
                analysis = agent.analyze(light_profile, validation)

            st.markdown('<div class="ai-box">', unsafe_allow_html=True)
            st.markdown('<div class="ai-label"><span class="pulse-dot-red"></span> AI Dataset Intelligence Report</div>', unsafe_allow_html=True)
            st.markdown(analysis)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Click the button above to generate an LLM dataset synthesis report.")

    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MODULE 03: DATA FILTERING & CLEANING
    # ══════════════════════════════════════════════════════════════
    st.markdown("## 🧹 Module 03 — Data Filtering & Preprocessing")

    col_filters, col_preview = st.columns([1, 2])

    with col_filters:
        st.markdown("### Preprocessing Pipeline Options")

        opt_drop_dupes = st.checkbox("🗑️ Drop duplicate rows", value=profile["duplicates"] > 0)
        opt_drop_null_rows = st.checkbox("🗑️ Drop fully-null rows", value=False)
        opt_drop_constant = st.checkbox("🗑️ Drop constant (zero variance) columns", value=len(profile["constant_columns"]) > 0)
        opt_drop_null_cols = st.checkbox("🗑️ Drop fully-null columns", value=False)
        opt_drop_id = st.checkbox("🗑️ Drop high-cardinality ID columns", value=False)
        opt_impute = st.checkbox("🔧 Impute missing values (median/mode)", value=profile["missing_cells"] > 0)
        opt_outliers = st.checkbox("📊 Remove numeric outliers (IQR)", value=False)

        st.markdown("---")

        if st.button("⚡ Get AI Cleaning Strategy", key="ai_suggest"):
            with st.spinner("🧠 AI is formulating data cleaning strategy..."):
                light_profile = {k: v for k, v in profile.items() if k != "column_details"}
                suggestions = agent.suggest_filters(light_profile, validation)

            st.markdown('<div class="ai-box">', unsafe_allow_html=True)
            st.markdown('<div class="ai-label"><span class="pulse-dot-red"></span> AI Strategy Recommendations</div>', unsafe_allow_html=True)
            st.markdown(suggestions)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_preview:
        filter_options = {
            "drop_duplicates": opt_drop_dupes,
            "drop_null_rows": opt_drop_null_rows,
            "drop_constant_cols": opt_drop_constant,
            "drop_null_cols": opt_drop_null_cols,
            "drop_id_cols": opt_drop_id,
            "impute_missing": opt_impute,
            "remove_outliers": opt_outliers,
        }

        if st.button("🚀 Apply Transformations", key="apply_filters", type="primary"):
            cleaned_df, changelog = filter_dataset(df, filter_options)
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["changelog"] = changelog

        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]
            changelog = st.session_state["changelog"]

            st.markdown("### Shape Transformation")
            ba1, ba2 = st.columns(2)
            with ba1:
                render_metric_card(f"{len(df):,} × {len(df.columns)}", "Original Dataset", "#A1A1AA")
            with ba2:
                render_metric_card(f"{len(cleaned_df):,} × {len(cleaned_df.columns)}", "Cleaned Dataset", "#FF1744")

            st.markdown("")

            st.markdown("### 📝 Execution Log")
            for entry in changelog:
                st.markdown(f'<div class="changelog-item">{entry}</div>', unsafe_allow_html=True)

            st.markdown("")
            csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )

            with st.expander("📄 Processed Data Preview", expanded=False):
                st.dataframe(cleaned_df.head(10), use_container_width=True)
        else:
            st.info("👈 Select cleaning transformations and click **Apply Transformations**.")

    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MODULE 04: BASELINE ML
    # ══════════════════════════════════════════════════════════════
    st.markdown("## 🚀 Module 04 — AutoML Model Engine")

    ml_df = st.session_state.get("cleaned_df", df)

    target = st.selectbox(
        "🎯 Select Target Variable",
        ["-- choose target --"] + list(ml_df.columns),
        key="target_select",
    )

    if target != "-- choose target --":
        # Target metadata & Task Detection
        is_categorical = ml_df[target].dtype == "object" or ml_df[target].nunique() <= 10
        task_type = "Classification" if is_categorical else "Regression"

        # Display target summary metrics
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            render_metric_card(str(ml_df[target].dtype), "Data Type", "#00E5FF")
        with tc2:
            render_metric_card(f"{ml_df[target].nunique():,}", "Unique Values", "#FFA4B6")
        with tc3:
            render_metric_card(task_type, "Detected Task", "#FF1744" if task_type == "Classification" else "#00E5FF")
        with tc4:
            render_metric_card(f"{ml_df[target].isna().sum()}", "Missing Target Rows", "#F87171" if ml_df[target].isna().sum() > 0 else "#00E5FF")

        st.markdown("")

        # User-selected Visualization Controls
        st.markdown("### 📊 Target Variable Visualization")
        
        # Determine default chart option based on type
        default_chart_idx = 1 if is_categorical else 2
        chart_type = st.radio(
            "Select Chart Type:",
            ["📊 Bar Chart", "🥧 Pie Chart", "📈 Histogram / Distribution", "📦 Box Plot", "📉 Line / Sequence"],
            index=default_chart_idx,
            horizontal=True,
            key="target_chart_type",
        )

        # Render selected chart
        if "Bar" in chart_type:
            val_counts = ml_df[target].value_counts().head(20)
            fig_target = px.bar(
                x=val_counts.index.astype(str), y=val_counts.values,
                title=f"Bar Chart — Value Counts of '{target}'",
                labels={"x": target, "y": "Count"},
                color_discrete_sequence=["#FF1744"],
                template="plotly_dark",
            )
        elif "Pie" in chart_type:
            val_counts = ml_df[target].value_counts().head(10)
            fig_target = px.pie(
                values=val_counts.values,
                names=val_counts.index.astype(str),
                title=f"Pie Chart — Proportional Share of '{target}'",
                color_discrete_sequence=["#FF1744", "#FF5252", "#FF7961", "#B70928", "#FFD600"],
                template="plotly_dark",
            )
        elif "Histogram" in chart_type:
            fig_target = px.histogram(
                ml_df, x=target, nbins=40,
                title=f"Histogram Distribution — '{target}'",
                color_discrete_sequence=["#FF1744"],
                template="plotly_dark",
            )
        elif "Box" in chart_type:
            fig_target = px.box(
                ml_df, y=target,
                title=f"Box Plot Spread — '{target}'",
                color_discrete_sequence=["#00E5FF"],
                template="plotly_dark",
            )
        else: # Line / Sequence
            fig_target = px.line(
                ml_df.reset_index(), y=target,
                title=f"Line Trend Across Records — '{target}'",
                color_discrete_sequence=["#FF1744"],
                template="plotly_dark",
            )

        fig_target.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#FAFAFA"),
        )
        st.plotly_chart(fig_target, use_container_width=True)

        st.markdown("---")

        # Run AutoML Pipeline
        if st.button("🚀 Launch AutoML Training Pipeline", type="primary", key="btn_run_ml"):
            with st.spinner("⚡ Training candidate ML models (Logistic/Forest/Trees)..."):
                result = run_baseline_ml(ml_df, target)

            if "error" in result:
                st.error(f"⚠️ {result['error']}")
            else:
                st.success(f"⚡ Detected Task Type: **{result['task'].upper()}**")

                r1, r2 = st.columns(2)
                with r1:
                    render_metric_card(result["best_model"], "Top Performing Model", "#FF1744", "rocket")
                with r2:
                    render_metric_card(f"{result['best_score']:.4f}", f"Score ({result['metric']})", "#FFA4B6", "sparkles")

                st.markdown("")
                st.markdown("### 📊 Model Benchmark Leaderboard")
                st.dataframe(result["model_results"], use_container_width=True, hide_index=True)
    else:
        st.info("👆 Select a target column to configure visualization and launch AutoML model training.")

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#FFA4B6; font-size:0.85rem; opacity:0.8;">'
        '🕷️ Spidey DATA SCIENTIST · Miles Morales Venom-Strike Edition · Powered by Groq AI'
        '</p>',
        unsafe_allow_html=True,
    )
