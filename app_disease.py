# ============================================================
# 🏥 samiq AI — Disease Risk Prediction Platform
# Author  : Syed Samiq Abbas Bukhari
# Project : CodeAlpha ML Internship — Task 4
# Version : 2.0 Enterprise
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings, datetime, io, base64
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from fpdf import FPDF

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Samiq AI | Disease Risk Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS — Premium Healthcare UI
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020817 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1526 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
}

[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Header Hero ── */
.hero-container {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 40%, #0c1a2e 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 20px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.30);
    color: #38bdf8 !important;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-size: clamp(26px, 4vw, 42px);
    font-weight: 900;
    line-height: 1.15;
    margin: 0 0 10px;
    background: linear-gradient(135deg, #f1f5f9 0%, #38bdf8 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 14px;
    color: #64748b !important;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.hero-author {
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.author-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 800; color: #fff !important;
    flex-shrink: 0;
}
.author-info { line-height: 1.3; }
.author-name {
    font-size: 13px; font-weight: 700;
    color: #e2e8f0 !important;
}
.author-role {
    font-size: 11px; color: #64748b !important;
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.metric-card {
    background: linear-gradient(135deg, #0d1526 0%, #0a1020 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(56,189,248,0.25); }
.metric-card-accent {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-value {
    font-size: 28px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin: 8px 0 4px;
}
.metric-label { font-size: 11px; color: #64748b !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-icon { font-size: 20px; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 32px 0 18px;
}
.section-icon {
    width: 34px; height: 34px;
    border-radius: 8px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.22);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.section-title {
    font-size: 16px; font-weight: 700;
    color: #f1f5f9 !important; margin: 0;
}
.section-subtitle { font-size: 12px; color: #475569 !important; margin: 0; }

/* ── Input Cards ── */
.input-card {
    background: #0d1526;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.input-card-title {
    font-size: 13px; font-weight: 700;
    color: #94a3b8 !important;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Streamlit Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: #0a1020 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: rgba(56,189,248,0.4) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.06) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: #38bdf8 !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 4px !important;
}

/* ── Primary Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 14px 28px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 24px rgba(14,165,233,0.25) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 32px rgba(14,165,233,0.35) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── Risk Result Cards ── */
.risk-card {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
}
.risk-card-low {
    background: linear-gradient(135deg, #052e16 0%, #0a1a0a 100%);
    border: 1.5px solid #22c55e;
    box-shadow: 0 0 24px rgba(34,197,94,0.12);
}
.risk-card-medium {
    background: linear-gradient(135deg, #1c1a03 0%, #1a1400 100%);
    border: 1.5px solid #f59e0b;
    box-shadow: 0 0 24px rgba(245,158,11,0.12);
}
.risk-card-high {
    background: linear-gradient(135deg, #1c0505 0%, #1a0000 100%);
    border: 1.5px solid #ef4444;
    box-shadow: 0 0 24px rgba(239,68,68,0.15);
}
.risk-label-low  { color: #4ade80 !important; }
.risk-label-med  { color: #fbbf24 !important; }
.risk-label-high { color: #f87171 !important; }
.risk-percent {
    font-family: 'JetBrains Mono', monospace;
    font-size: 38px; font-weight: 800;
    line-height: 1; margin: 6px 0 4px;
}
.risk-disease { font-size: 12px; color: #64748b !important; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── Alert Boxes ── */
.alert-high {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #fca5a5 !important;
    font-size: 13px; font-weight: 500;
}
.alert-medium {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #fcd34d !important;
    font-size: 13px; font-weight: 500;
}
.alert-low {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.22);
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #86efac !important;
    font-size: 13px; font-weight: 500;
}

/* ── Rec Cards ── */
.rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 12px;
}
.rec-card {
    background: #0d1526;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
}
.rec-icon { font-size: 22px; margin-bottom: 8px; }
.rec-title { font-size: 12px; font-weight: 700; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.6px; }
.rec-text  { font-size: 12px; color: #64748b !important; margin-top: 4px; line-height: 1.5; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 28px 0 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 48px;
}
.footer-name { font-size: 13px; font-weight: 700; color: #38bdf8 !important; }
.footer-meta { font-size: 11px; color: #334155 !important; margin-top: 4px; }

/* ── Divider ── */
hr[data-testid="stDivider"] { border-color: rgba(255,255,255,0.06) !important; }

/* ── Info/Warn/Error blocks ── */
[data-testid="stAlert"] {
    background: rgba(14,165,233,0.07) !important;
    border: 1px solid rgba(14,165,233,0.18) !important;
    border-radius: 10px !important;
    color: #7dd3fc !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #38bdf8 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def risk_level(prob):
    if prob < 0.35:   return "LOW",    "low",    "#4ade80"
    elif prob < 0.65: return "MEDIUM", "medium", "#fbbf24"
    else:             return "HIGH",   "high",   "#f87171"


def gauge_chart(prob: float, title: str, color: str):
    fig, ax = plt.subplots(figsize=(3.8, 2.6), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0d1526")
    ax.set_facecolor("#0d1526")

    theta = np.linspace(np.pi, 0, 200)
    # Track
    ax.plot(theta, [1]*200, color="#1e293b", linewidth=10, solid_capstyle="round")
    # Zones
    ax.plot(np.linspace(np.pi,   np.pi*0.67, 80), [1]*80, color="#22c55e", linewidth=10, alpha=0.35, solid_capstyle="round")
    ax.plot(np.linspace(np.pi*0.67, np.pi*0.33, 80), [1]*80, color="#f59e0b", linewidth=10, alpha=0.35, solid_capstyle="round")
    ax.plot(np.linspace(np.pi*0.33, 0, 80), [1]*80, color="#ef4444", linewidth=10, alpha=0.35, solid_capstyle="round")
    # Value arc
    fill_end = np.pi - prob * np.pi
    ax.plot(np.linspace(np.pi, fill_end, 120), [1]*120, color=color, linewidth=10, solid_capstyle="round")

    # Needle
    needle_angle = np.pi - prob * np.pi
    ax.annotate("", xy=(needle_angle, 0.82), xytext=(needle_angle, 0.0),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.2, mutation_scale=12))
    ax.plot([needle_angle], [0], "o", color="white", markersize=5, zorder=5)

    ax.set_ylim(0, 1.35)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.axis("off")

    pct_x, pct_y = np.pi/2, 1.3
    ax.text(pct_x, pct_y, f"{prob*100:.0f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color, transform=ax.transData,
            fontfamily="monospace")
    ax.text(pct_x, -0.38, title, ha="center", va="center",
            fontsize=9.5, color="#64748b", transform=ax.transData)

    plt.tight_layout(pad=0)
    return fig


def feature_importance_chart(model, feature_names: list, title: str, color: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    top_n = 8
    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor("#0d1526")
    ax.set_facecolor("#0d1526")
    bars = ax.barh(names, vals, color=color, alpha=0.85, height=0.55)
    for bar in bars:
        ax.barh(bar.get_y() + bar.get_height()/2, bar.get_width(),
                height=0.01, color=color, alpha=0.3)
    ax.set_xlabel("Importance", color="#475569", fontsize=9)
    ax.set_title(f"Top Features — {title}", color="#94a3b8", fontsize=10, pad=10)
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.spines["left"].set_color("#1e293b")
    ax.xaxis.set_tick_params(color="#1e293b")
    plt.tight_layout()
    return fig


def probability_bar_chart(results: dict):
    diseases = list(results.keys())
    probs    = [results[d][0] * 100 for d in diseases]
    colors   = []
    for d in diseases:
        _, cls, c = risk_level(results[d][0])
        colors.append(c)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#0d1526")
    ax.set_facecolor("#0d1526")

    bars = ax.bar(diseases, probs, color=colors, alpha=0.85, width=0.5)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{p:.1f}%", ha="center", va="bottom", color="#94a3b8",
                fontsize=10, fontweight="bold", fontfamily="monospace")

    ax.axhline(35, color="#22c55e", linewidth=0.7, linestyle="--", alpha=0.4, label="Low threshold")
    ax.axhline(65, color="#f59e0b", linewidth=0.7, linestyle="--", alpha=0.4, label="Med threshold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Risk %", color="#475569", fontsize=9)
    ax.set_title("Disease Risk Comparison", color="#94a3b8", fontsize=10, pad=10)
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#1e293b")
    ax.legend(fontsize=7, labelcolor="#64748b", framealpha=0, loc="upper right")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_all_models():
    models, scalers = {}, {}

    # ── 1. Heart Disease ──────────────────
    data = fetch_openml(name="heart-statlog", version=1, as_frame=True)
    df = data.frame.copy()
    df.columns = [*data.feature_names, "target"]
    df["target"] = df["target"].map({"present": 1, "absent": 0}).astype(int)
    X = df.drop("target", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    best_m, best_auc = None, 0
    for M in [LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=100, random_state=42)]:
        M.fit(sc.fit_transform(X_tr), y_tr)
        auc = roc_auc_score(y_te, M.predict_proba(sc.transform(X_te))[:,1])
        if auc > best_auc:
            best_auc, best_m = auc, M
    sc2 = StandardScaler(); sc2.fit(X_tr)
    best_m.fit(sc2.transform(X_tr), y_tr)
    models["heart"] = best_m;  scalers["heart"] = sc2
    models["heart_features"] = list(X.columns)
    models["heart_acc"] = round(accuracy_score(y_te, best_m.predict(sc2.transform(X_te))) * 100, 2)

    # ── 2. Diabetes ───────────────────────
    data2 = fetch_openml(name="diabetes", version=1, as_frame=True)
    df2 = data2.frame.copy()
    df2.columns = [*data2.feature_names, "target"]
    df2["target"] = df2["target"].map({"tested_positive": 1, "tested_negative": 0}).astype(int)
    X2 = df2.drop("target", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y2 = df2["target"]
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
    sc2b = StandardScaler(); m2 = RandomForestClassifier(n_estimators=100, random_state=42)
    m2.fit(sc2b.fit_transform(X_tr2), y_tr2)
    models["diabetes"] = m2;  scalers["diabetes"] = sc2b
    models["diabetes_features"] = list(X2.columns)
    models["diabetes_acc"] = round(accuracy_score(y_te2, m2.predict(sc2b.transform(X_te2))) * 100, 2)

    # ── 3. Breast Cancer ──────────────────
    bc = load_breast_cancer()
    X3 = pd.DataFrame(bc.data, columns=bc.feature_names)
    y3 = bc.target
    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
    sc3 = StandardScaler(); m3 = RandomForestClassifier(n_estimators=150, random_state=42)
    m3.fit(sc3.fit_transform(X_tr3), y_tr3)
    models["cancer"] = m3;  scalers["cancer"] = sc3
    models["cancer_features"] = list(X3.columns)
    models["cancer_acc"] = round(accuracy_score(y_te3, m3.predict(sc3.transform(X_te3))) * 100, 2)

    return models, scalers


# ─────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────
def generate_pdf(name: str, age: int, gender: str, results: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(5, 15, 35)
    pdf.rect(0, 0, 210, 42, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 18, "", ln=True)
    pdf.cell(0, 10, "Samiq AI — Health Risk Assessment", align="C", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, "Powered by Advanced ML | CodeAlpha ML Internship — Task 4", align="C", ln=True)

    # Patient Info
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 9, "  Patient Information", ln=True, fill=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(60, 7, f"Name: {name}")
    pdf.cell(60, 7, f"Age: {age} years")
    pdf.cell(60, 7, f"Gender: {gender}", ln=True)
    pdf.cell(60, 7, f"Report Date: {datetime.date.today().strftime('%B %d, %Y')}")
    pdf.cell(60, 7, f"Generated By: Syed Samiq Abbas Bukhari", ln=True)
    pdf.ln(6)

    # Risk Results
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 9, "  Disease Risk Summary", ln=True, fill=True)
    pdf.ln(3)

    color_map = {"LOW": (34,197,94), "MEDIUM": (245,158,11), "HIGH": (239,68,68)}
    for disease, (prob, level, _) in results.items():
        c = color_map[level]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(80, 9, disease + ":")
        pdf.set_text_color(*c)
        pdf.cell(40, 9, f"{level} RISK")
        pdf.set_text_color(71, 85, 105)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 9, f"({prob*100:.1f}% probability)", ln=True)

    # Recommendations
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 9, "  Clinical Recommendations", ln=True, fill=True)
    pdf.ln(3)

    high_risks = [d for d,(p,l,_) in results.items() if l=="HIGH"]
    med_risks  = [d for d,(p,l,_) in results.items() if l=="MEDIUM"]

    pdf.set_font("Helvetica", "", 10)
    if high_risks:
        pdf.set_text_color(180,0,0)
        pdf.multi_cell(0, 7, f"⚠ HIGH RISK: {', '.join(high_risks)} — Immediate medical consultation recommended.")
        pdf.ln(2)
    if med_risks:
        pdf.set_text_color(150,100,0)
        pdf.multi_cell(0, 7, f"• MEDIUM RISK: {', '.join(med_risks)} — Schedule a check-up within 30 days.")
        pdf.ln(2)

    recs = [
        "• Maintain a balanced diet rich in vegetables, fruits, and whole grains.",
        "• Exercise at least 30 minutes per day — walking, swimming, or cycling.",
        "• Monitor blood pressure, blood glucose, and cholesterol regularly.",
        "• Limit sodium, saturated fats, processed sugars, and ultra-processed foods.",
        "• Avoid smoking; limit alcohol to recommended safe limits.",
        "• Target 7–8 hours of quality sleep per night.",
        "• Manage stress via mindfulness, yoga, or therapy.",
        "• Stay hydrated — at least 8 glasses of water daily.",
    ]
    pdf.set_text_color(30,41,59)
    for r in recs:
        pdf.multi_cell(0, 7, r)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100,116,139)
    pdf.multi_cell(0, 5,
        "DISCLAIMER: This report is generated by an AI-powered tool for educational and informational "
        "purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult "
        "a qualified healthcare professional before making any medical decisions.")

    # Footer
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(56,189,248)
    pdf.cell(0, 7, "Samiq AI Platform | Developed by Syed Samiq Abbas Bukhari | CodeAlpha ML Internship", align="C", ln=True)

    return bytes(pdf.output())


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-size:36px'>🧬</div>
      <div style='font-size:16px;font-weight:800;color:#38bdf8;margin-top:6px'>Samiq AI</div>
      <div style='font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:2px'>Disease Risk Platform</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.06);margin:12px 0'>
    """, unsafe_allow_html=True)

    st.markdown("**🔬 Active Models**")
    st.markdown("""
    <div style='font-size:12px;color:#64748b;line-height:2'>
    ❤️ Heart Disease — Random Forest<br>
    🍬 Diabetes — Random Forest<br>
    🩺 Breast Cancer — Random Forest
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("**📋 How to Use**")
    st.markdown("""
    <div style='font-size:12px;color:#64748b;line-height:2'>
    1. Enter patient information<br>
    2. Fill clinical parameters<br>
    3. Click Predict Risk<br>
    4. Review results & download PDF
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#334155;text-align:center'>
    <div style='color:#38bdf8;font-weight:700;font-size:12px'>Syed Samiq Abbas Bukhari</div>
    CodeAlpha ML Internship<br>
    Task 4 — Disease Prediction
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class='hero-container'>
  <div class='hero-badge'>🧬 AI-Powered Healthcare Analytics</div>
  <h1 class='hero-title'>Disease Risk Prediction<br>Platform</h1>
  <p class='hero-sub'>Advanced ML models trained on clinical datasets — Heart Disease, Diabetes &amp; Breast Cancer risk assessment with explainable AI</p>
  <div class='hero-author'>
    <div class='author-avatar'>S</div>
    <div class='author-info'>
      <div class='author-name'>Syed Samiq Abbas Bukhari</div>
      <div class='author-role'>ML Engineer · CodeAlpha Internship · Task 4</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
with st.spinner("🔄 Initializing AI models — please wait..."):
    models, scalers = train_all_models()

# ── Accuracy Cards ───────────────────────
st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'>
    <div class='metric-card-accent' style='background:linear-gradient(90deg,#ef4444,#f97316)'></div>
    <div class='metric-icon'>❤️</div>
    <div class='metric-value' style='color:#f87171'>{models["heart_acc"]}%</div>
    <div class='metric-label'>Heart Disease Accuracy</div>
  </div>
  <div class='metric-card'>
    <div class='metric-card-accent' style='background:linear-gradient(90deg,#f59e0b,#84cc16)'></div>
    <div class='metric-icon'>🍬</div>
    <div class='metric-value' style='color:#fbbf24'>{models["diabetes_acc"]}%</div>
    <div class='metric-label'>Diabetes Accuracy</div>
  </div>
  <div class='metric-card'>
    <div class='metric-card-accent' style='background:linear-gradient(90deg,#38bdf8,#818cf8)'></div>
    <div class='metric-icon'>🩺</div>
    <div class='metric-value' style='color:#38bdf8'>{models["cancer_acc"]}%</div>
    <div class='metric-label'>Breast Cancer Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PATIENT INFORMATION
# ─────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <div class='section-icon'>👤</div>
  <div>
    <p class='section-title'>Patient Information</p>
    <p class='section-subtitle'>Basic demographic details</p>
  </div>
</div>
""", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
patient_name   = p1.text_input("Full Name", placeholder="e.g. Ahmed Ali", label_visibility="visible")
patient_age    = p2.number_input("Age (years)", min_value=1, max_value=120, value=45)
patient_gender = p3.selectbox("Gender", ["Male", "Female", "Other"])

st.divider()

# ─────────────────────────────────────────
# DISEASE TABS
# ─────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <div class='section-icon'>🔬</div>
  <div>
    <p class='section-title'>Clinical Parameters</p>
    <p class='section-subtitle'>Enter values from the patient's medical records</p>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["❤️  Heart Disease", "🍬  Diabetes", "🩺  Breast Cancer"])

with tab1:
    st.caption("Cardiovascular risk factors from clinical examination")
    features_h = models["heart_features"]
    h_vals = {}
    defaults_h = {"age":50,"sex":1,"chest":1,"resting_blood_pressure":120,
                  "serum_cholestoral":200,"fasting_blood_sugar":0,
                  "resting_electrocardiographic_results":0,
                  "maximum_heart_rate_achieved":150,"exercise_induced_angina":0,
                  "oldpeak":1.0,"slope":1,"number_of_major_vessels":0,"thal":2}
    labels_h = {
        "age":"Age","sex":"Sex (1=Male, 0=Female)","chest":"Chest Pain Type (0–3)",
        "resting_blood_pressure":"Resting BP (mmHg)","serum_cholestoral":"Serum Cholesterol (mg/dl)",
        "fasting_blood_sugar":"Fasting Blood Sugar >120 (1=Yes)","resting_electrocardiographic_results":"ECG Results (0–2)",
        "maximum_heart_rate_achieved":"Max Heart Rate","exercise_induced_angina":"Exercise Angina (1=Yes)",
        "oldpeak":"ST Depression (Oldpeak)","slope":"Slope (0–2)",
        "number_of_major_vessels":"Major Vessels (0–3)","thal":"Thal (0=Normal, 1=Fixed, 2=Reversible)",
    }
    cols_h = st.columns(3)
    for i, f in enumerate(features_h):
        lbl = labels_h.get(f, f)
        dflt = defaults_h.get(f, 0)
        c = cols_h[i % 3]
        if f in ["sex","fasting_blood_sugar","exercise_induced_angina"]:
            h_vals[f] = c.selectbox(lbl, [0,1], index=int(dflt), key="h_"+f)
        elif f in ["chest","resting_electrocardiographic_results","slope","number_of_major_vessels","thal"]:
            h_vals[f] = c.slider(lbl, 0, 3, int(dflt), key="h_"+f)
        elif f == "oldpeak":
            h_vals[f] = c.slider(lbl, 0.0, 6.0, float(dflt), 0.1, key="h_"+f)
        else:
            h_vals[f] = c.number_input(lbl, value=float(dflt), key="h_"+f)

with tab2:
    st.caption("Metabolic and glycemic indicators for diabetes screening")
    features_d = models["diabetes_features"]
    d_vals = {}
    defaults_d = {"preg":2,"plas":120,"pres":70,"skin":20,"insu":80,"mass":28.0,"pedi":0.5,"age":35}
    labels_d = {"preg":"Pregnancies","plas":"Plasma Glucose","pres":"Diastolic BP (mmHg)",
                "skin":"Skin Thickness (mm)","insu":"Insulin (μU/ml)","mass":"BMI (kg/m²)",
                "pedi":"Diabetes Pedigree Function","age":"Age"}
    cols_d = st.columns(3)
    for i, f in enumerate(features_d):
        d_vals[f] = cols_d[i%3].number_input(labels_d.get(f,f), value=float(defaults_d.get(f,0)), key="d_"+f)

with tab3:
    st.caption("Tumor cell nucleus features from biopsy — consult your pathologist")
    st.info("💡 Default values represent dataset mean. Enter actual biopsy values for accurate prediction.", icon="ℹ️")
    features_c = models["cancer_features"]
    bc_data = load_breast_cancer()
    bc_mean = bc_data.data.mean(axis=0)
    c_vals = {}
    cols_c = st.columns(3)
    for i, f in enumerate(features_c):
        c_vals[f] = cols_c[i%3].number_input(f, value=float(round(bc_mean[i],4)), format="%.4f", key="c_"+f)


# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.divider()
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict = st.button("🔍  Predict Disease Risk", type="primary", use_container_width=True)


# ─────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────
if predict:
    # ── Compute Probabilities ─────────────
    h_input = np.array([[h_vals[f] for f in models["heart_features"]]])
    h_prob  = models["heart"].predict_proba(scalers["heart"].transform(h_input))[0][1]
    h_level, h_cls, h_color = risk_level(h_prob)

    d_input = np.array([[d_vals[f] for f in models["diabetes_features"]]])
    d_prob  = models["diabetes"].predict_proba(scalers["diabetes"].transform(d_input))[0][1]
    d_level, d_cls, d_color = risk_level(d_prob)

    c_input = np.array([[c_vals[f] for f in models["cancer_features"]]])
    c_prob  = 1 - models["cancer"].predict_proba(scalers["cancer"].transform(c_input))[0][1]
    c_level, c_cls, c_color = risk_level(c_prob)

    results_pdf = {
        "Heart Disease": (h_prob, h_level, h_color),
        "Diabetes":      (d_prob, d_level, d_color),
        "Breast Cancer": (c_prob, c_level, c_color),
    }

    # ── Section Header ───────────────────
    st.markdown("""
    <div class='section-header' style='margin-top:32px'>
      <div class='section-icon' style='background:rgba(139,92,246,0.12);border-color:rgba(139,92,246,0.25)'>📊</div>
      <div>
        <p class='section-title'>Risk Assessment Results</p>
        <p class='section-subtitle'>AI-generated predictions — for educational use only</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge Row ────────────────────────
    g1, g2, g3 = st.columns(3)
    for col, title, prob, level, cls, color in [
        (g1, "Heart Disease", h_prob, h_level, h_cls, h_color),
        (g2, "Diabetes",      d_prob, d_level, d_cls, d_color),
        (g3, "Breast Cancer", c_prob, c_level, c_cls, c_color),
    ]:
        with col:
            st.pyplot(gauge_chart(prob, title, color), use_container_width=True)
            st.markdown(f"""
            <div class='risk-card risk-card-{cls}'>
              <div class='risk-disease'>{title}</div>
              <div class='risk-percent risk-label-{"low" if cls=="low" else "med" if cls=="medium" else "high"}'>{prob*100:.1f}%</div>
              <div style='font-size:13px;font-weight:700;color:{"#4ade80" if cls=="low" else "#fbbf24" if cls=="medium" else "#f87171"}'>
                {level} RISK
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparison Chart ──────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cc1, cc2 = st.columns([1.3, 1])
    with cc1:
        st.pyplot(probability_bar_chart({
            "Heart Disease": (h_prob, h_level, h_color),
            "Diabetes":      (d_prob, d_level, d_color),
            "Breast Cancer": (c_prob, c_level, c_color),
        }), use_container_width=True)
    with cc2:
        high = [d for d,(p,l,_) in results_pdf.items() if l=="HIGH"]
        med  = [d for d,(p,l,_) in results_pdf.items() if l=="MEDIUM"]
        low  = [d for d,(p,l,_) in results_pdf.items() if l=="LOW"]

        st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
        for d in high:
            st.markdown(f"<div class='alert-high'>⛔ <b>HIGH RISK</b> — {d}<br><span style='font-size:11px'>Consult a physician immediately</span></div>", unsafe_allow_html=True)
        for d in med:
            st.markdown(f"<div class='alert-medium'>⚠️ <b>MEDIUM RISK</b> — {d}<br><span style='font-size:11px'>Schedule a check-up within 30 days</span></div>", unsafe_allow_html=True)
        for d in low:
            st.markdown(f"<div class='alert-low'>✅ <b>LOW RISK</b> — {d}<br><span style='font-size:11px'>Continue healthy habits</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Importances ───────────────
    st.markdown("""
    <div class='section-header' style='margin-top:28px'>
      <div class='section-icon' style='background:rgba(56,189,248,0.10);border-color:rgba(56,189,248,0.22)'>🧠</div>
      <div>
        <p class='section-title'>AI Explainability — Feature Importance</p>
        <p class='section-subtitle'>Which factors most influenced the prediction</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fi1, fi2, fi3 = st.columns(3)
    for col, key, feat_key, title, color in [
        (fi1, "heart",    "heart_features",    "Heart Disease", "#f87171"),
        (fi2, "diabetes", "diabetes_features",  "Diabetes",      "#fbbf24"),
        (fi3, "cancer",   "cancer_features",    "Breast Cancer", "#38bdf8"),
    ]:
        fig = feature_importance_chart(models[key], models[feat_key], title, color)
        if fig:
            col.pyplot(fig, use_container_width=True)

    # ── Recommendations ───────────────────
    st.markdown("""
    <div class='section-header' style='margin-top:28px'>
      <div class='section-icon' style='background:rgba(34,197,94,0.10);border-color:rgba(34,197,94,0.22)'>💊</div>
      <div>
        <p class='section-title'>Personalized Health Recommendations</p>
        <p class='section-subtitle'>AI-curated lifestyle and clinical guidance</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    recs = [
        ("🥗","Diet","Focus on whole foods, reduce saturated fats & processed sugars."),
        ("🏃","Exercise","30 min of moderate activity 5× per week — walking, cycling, swimming."),
        ("💤","Sleep","Aim for 7–8 hours of quality, uninterrupted sleep nightly."),
        ("💧","Hydration","Drink 8–10 glasses of water daily; limit sugary drinks."),
        ("🧘","Stress","Practice mindfulness, deep breathing, or yoga regularly."),
        ("🚭","Smoking","Quit smoking — it doubles cardiovascular and cancer risk."),
        ("⚖️","Weight","Maintain healthy BMI (18.5–24.9) through diet and activity."),
        ("🩺","Screening","Regular check-ups: BP, glucose, cholesterol, mammography."),
    ]
    rec_html = "<div class='rec-grid'>"
    for icon, title, text in recs:
        rec_html += f"""
        <div class='rec-card'>
          <div class='rec-icon'>{icon}</div>
          <div class='rec-title'>{title}</div>
          <div class='rec-text'>{text}</div>
        </div>"""
    rec_html += "</div>"
    st.markdown(rec_html, unsafe_allow_html=True)

    # # ── PDF Download ──────────────────────
    # st.markdown("""
    # <div class='section-header' style='margin-top:32px'>
    #   <div class='section-icon' style='background:rgba(99,102,241,0.12);border-color:rgba(99,102,241,0.25)'>📄</div>
    #   <div>
    #     <p class='section-title'>Download Health Report</p>
    #     <p class='section-subtitle'>Professional PDF with all results and recommendations</p>
    #   </div>
    # </div>
    # """, unsafe_allow_html=True)

    # name = patient_name if patient_name.strip() else "Patient"
    # try:
    #     pdf_bytes = generate_pdf(name, patient_age, patient_gender, results_pdf)
    #     dl1, dl2, dl3 = st.columns([1,2,1])
    #     with dl2:
    #         st.download_button(
    #             label="📥  Download PDF Health Report",
    #             data=pdf_bytes,
    #             file_name=f"Samiq_Report_{name.replace(' ','_')}_{datetime.date.today()}.pdf",
    #             mime="application/pdf",
    #             use_container_width=True,
    #         )
    # except Exception as e:
    #     st.error(f"PDF generation error: {e}")


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class='footer'>
  <div class='footer-name'>🧬 SAMIQ AI — Disease Risk Prediction Platform</div>
  <div class='footer-meta'>
    Developed by <b style='color:#38bdf8'>Syed Samiq Abbas Bukhari</b> &nbsp;·&nbsp;
    CodeAlpha ML Internship &nbsp;·&nbsp; Task 4 &nbsp;·&nbsp;
    For educational purposes only — not a medical device
  </div>
</div>
""", unsafe_allow_html=True)