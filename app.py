import json
import os
from urllib.parse import quote

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# CardioPredict - product-style interface and synthetic scenario explorer.
# The saved model, original 13 features, preprocessing and training pipeline
# are intentionally unchanged. Synthetic profiles are for software testing
# only and are never added to the training dataset.

st.set_page_config(
    page_title="CardioPredict | Heart Health Intelligence",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/cvd_model.joblib"
METRICS_PATH = "models/metrics.json"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
DEFAULTS = {
    "age": 50, "sex": 0, "cp": 2, "trestbps": 120, "chol": 200,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault("field_" + key, value)

# ----------------------------- VISUAL SYSTEM -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
      --ink:#F5F7FF; --muted:#B7C1D8; --panel:rgba(24,31,62,.78);
      --line:rgba(255,255,255,.12); --cyan:#4DEBFF; --pink:#FF5DB1;
      --violet:#9B7BFF; --lime:#B8FF73; --orange:#FFB45E;
    }
    html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp {
      color:var(--ink);
      background:
        radial-gradient(ellipse at 8% 0%,rgba(0,210,255,.19),transparent 28rem),
        radial-gradient(ellipse at 92% 8%,rgba(255,65,170,.18),transparent 30rem),
        radial-gradient(ellipse at 50% 100%,rgba(119,80,255,.17),transparent 34rem),
        linear-gradient(145deg,#090D20 0%,#111631 48%,#0B1025 100%);
    }
    [data-testid="stHeader"] {background:rgba(9,13,32,.55);}
    .block-container {max-width:1440px;padding-top:1.1rem;padding-bottom:3rem;}
    h1,h2,h3,h4 {font-family:'Space Grotesk',sans-serif!important;color:#F7F8FF!important;letter-spacing:-.035em;}
    p,li,label,.stCaption {color:#C6CEE2;}
    [data-testid="stMarkdownContainer"] strong {color:#FFFFFF;}
    section[data-testid="stSidebar"] {
      background:linear-gradient(180deg,#151B3B 0%,#10152F 100%);
      border-right:1px solid rgba(255,255,255,.09);
    }
    section[data-testid="stSidebar"] * {color:#EEF2FF!important;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#AEBBDD!important;}
    section[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.13);}
    .brand {font:700 1.25rem 'Space Grotesk';letter-spacing:-.04em;}
    .brand span {color:var(--cyan);}
    .side-chip {display:inline-block;padding:5px 9px;border-radius:99px;background:rgba(77,235,255,.1);
      color:#80F4FF;border:1px solid rgba(77,235,255,.22);font-size:.68rem;font-weight:800;letter-spacing:.1em;}
    .hero {
      position:relative;overflow:hidden;isolation:isolate;border-radius:30px;
      padding:2.15rem 2.35rem;margin:.2rem 0 1.35rem;
      background:linear-gradient(112deg,rgba(29,39,87,.98),rgba(42,35,105,.95) 48%,rgba(9,115,139,.94));
      border:1px solid rgba(255,255,255,.19);
      box-shadow:0 24px 70px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.18);
      animation:rise .7s ease both;
    }
    .hero:before {content:"";position:absolute;z-index:-1;width:420px;height:420px;right:-80px;top:-190px;
      border-radius:50%;background:radial-gradient(circle,rgba(77,235,255,.32),rgba(155,123,255,.08) 48%,transparent 70%);
      animation:floatGlow 8s ease-in-out infinite;}
    .hero:after {content:"♥";position:absolute;z-index:-1;right:7%;top:-40px;font-size:250px;line-height:1;
      color:rgba(255,255,255,.055);animation:heartbeat 2.4s ease-in-out infinite;}
    .hero-top {display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:1rem;}
    .pill {display:inline-flex;align-items:center;gap:6px;padding:6px 11px;border-radius:99px;
      background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16);
      color:#DFFBFF;font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
    .live-dot {width:7px;height:7px;border-radius:50%;background:#B8FF73;box-shadow:0 0 12px #B8FF73;}
    .hero h1 {font-size:clamp(2.2rem,5vw,3.7rem)!important;line-height:1.02;margin:0 0 .65rem!important;}
    .hero h1 span {background:linear-gradient(90deg,#6EF4FF,#C2A7FF,#FF8BCB);-webkit-background-clip:text;background-clip:text;color:transparent;}
    .hero p {max-width:760px;font-size:1.03rem;color:#D8E3FF!important;margin:0;}
    .hero-foot {display:flex;gap:10px;flex-wrap:wrap;margin-top:1.4rem;}
    .hero-stat {padding:9px 13px;border-radius:13px;background:rgba(4,9,28,.24);border:1px solid rgba(255,255,255,.13);
      color:#E7F4FF;font-size:.78rem;}
    .hero-stat b {color:#76F3FF;}
    .ecg {height:34px;width:min(420px,75%);margin-top:1.1rem;opacity:.9;}
    .ecg svg {height:100%;width:100%;}
    .ecg path {fill:none;stroke:#75F4FF;stroke-width:2.5;stroke-linecap:round;stroke-dasharray:850;stroke-dashoffset:850;
      animation:draw 3s ease .25s forwards;filter:drop-shadow(0 0 7px #4DEBFF);}
    @keyframes draw {to{stroke-dashoffset:0;}}
    @keyframes rise {from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
    @keyframes heartbeat {0%,100%{transform:scale(1)}12%{transform:scale(1.09)}24%{transform:scale(1)}38%{transform:scale(1.05)}50%{transform:scale(1)}}
    @keyframes floatGlow {0%,100%{transform:translate(0,0)}50%{transform:translate(-28px,22px)}}
    .section-kicker {font-size:.7rem;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:#6DEEFF;margin-bottom:.25rem;}
    .section-sub {color:#AEBBD8;font-size:.9rem;margin-top:-.4rem;margin-bottom:1rem;}
    .glass-card {background:linear-gradient(145deg,rgba(37,47,89,.78),rgba(19,26,57,.82));
      border:1px solid rgba(255,255,255,.13);border-radius:20px;padding:17px 19px;
      box-shadow:0 12px 32px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.08);
      transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;}
    .glass-card:hover {transform:translateY(-4px);border-color:rgba(77,235,255,.42);box-shadow:0 18px 38px rgba(0,0,0,.28),0 0 24px rgba(77,235,255,.07);}
    .stat-label {color:#AEBBDD;font-size:.69rem;font-weight:900;letter-spacing:.1em;text-transform:uppercase;}
    .stat-value {font:700 1.65rem 'Space Grotesk';margin-top:7px;color:#FFFFFF;}
    .stat-note {font-size:.72rem;color:#7CEEFF;margin-top:4px;}
    .glow-line {height:2px;border-radius:5px;background:linear-gradient(90deg,#4DEBFF,#9B7BFF,#FF5DB1,transparent);margin:1.2rem 0;}
    div[data-testid="stForm"] {background:rgba(19,26,57,.8);border:1px solid rgba(255,255,255,.13);
      border-radius:20px;padding:1rem 1.1rem;box-shadow:0 16px 38px rgba(0,0,0,.2);}
    div[data-baseweb="input"]>div,div[data-baseweb="select"]>div {
      background:#171F43!important;border:1px solid #414D7A!important;border-radius:11px!important;color:#F7F8FF!important;}
    input,textarea {color:#FFFFFF!important;}
    [data-baseweb="popover"],[data-baseweb="menu"] {background:#171F43!important;}
    [role="option"] {color:#F7F8FF!important;}
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button {
      border:0!important;border-radius:12px!important;min-height:2.85rem;font-weight:900!important;color:#091226!important;
      background:linear-gradient(100deg,#55EDFF 0%,#8FE8D0 45%,#C5A5FF 100%)!important;
      box-shadow:0 8px 24px rgba(77,235,255,.2),inset 0 1px 0 rgba(255,255,255,.75)!important;
      transition:transform .2s ease,filter .2s ease!important;}
    .stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover {transform:translateY(-2px) scale(1.01);filter:brightness(1.08);}
    .stTabs [data-baseweb="tab-list"] {gap:8px;background:rgba(12,17,42,.62);padding:7px;border:1px solid rgba(255,255,255,.1);border-radius:16px;}
    .stTabs [data-baseweb="tab"] {border-radius:11px;padding:10px 15px;color:#B8C5E3!important;}
    .stTabs [aria-selected="true"] {background:linear-gradient(110deg,rgba(77,235,255,.18),rgba(155,123,255,.2))!important;
      border:1px solid rgba(77,235,255,.3)!important;color:#FFFFFF!important;}
    [data-testid="stDataFrame"] {border:1px solid rgba(255,255,255,.14);border-radius:14px;overflow:hidden;}
    [data-testid="stMetric"] {background:rgba(255,255,255,.045);padding:12px;border:1px solid rgba(255,255,255,.1);border-radius:14px;}
    [data-testid="stMetricLabel"] {color:#B8C5E3!important;}
    [data-testid="stMetricValue"] {color:#FFFFFF!important;}
    .result-card {border-radius:20px;padding:1.25rem 1.4rem;border:1px solid rgba(77,235,255,.42);
      background:linear-gradient(120deg,rgba(9,117,133,.26),rgba(41,46,104,.65));
      box-shadow:0 14px 36px rgba(0,0,0,.2);animation:rise .45s ease both;}
    .result-card.alert {border-color:rgba(255,93,177,.55);background:linear-gradient(120deg,rgba(135,35,92,.35),rgba(65,37,98,.68));}
    .result-tag {color:#7EF2FF;font-size:.7rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase;}
    .result-card.alert .result-tag {color:#FF9ACD;}
    .result-title {font:700 1.55rem 'Space Grotesk';color:white;margin:.35rem 0;}
    .notice {border:1px solid rgba(255,180,94,.34);background:rgba(112,72,24,.19);color:#FFE0B2;padding:13px 15px;border-radius:14px;font-size:.85rem;}
    .info-box {border:1px solid rgba(77,235,255,.24);background:rgba(33,104,130,.15);color:#D6F7FF;padding:13px 15px;border-radius:14px;font-size:.85rem;}
    .emergency {background:linear-gradient(110deg,#8B1748,#B52D48,#7939A0);border:1px solid rgba(255,255,255,.2);
      border-radius:22px;padding:1.3rem 1.45rem;box-shadow:0 16px 40px rgba(0,0,0,.25);}
    .emergency h3 {margin:0;color:white!important;}
    .emergency p {color:#FFE4EF!important;}
    .phone {display:inline-block;background:#FFFFFF;color:#921C51!important;text-decoration:none!important;
      padding:10px 16px;border-radius:11px;font-weight:900;margin:.4rem .5rem .2rem 0;}
    .small-muted {color:#A9B6D5;font-size:.78rem;}
    .footer {color:#8391B4;font-size:.75rem;text-align:center;padding:1.4rem 0 .4rem;}
    @media(max-width:760px) {
      .block-container {padding-left:1rem;padding-right:1rem;}
      .hero {padding:1.45rem;border-radius:22px;}
      .hero:after {font-size:150px;right:-20px;}
      .stat-value {font-size:1.25rem;}
      .stTabs [data-baseweb="tab"] {padding:8px 9px;font-size:.78rem;}
    }
    @media(prefers-reduced-motion:reduce) {
      *,*:before,*:after {animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- UTILITIES ---------------------------------
def pct(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def synthetic_profile(rng):
    """Generate one fictional profile inside the model's accepted feature ranges."""
    return {
        "age": int(rng.integers(29, 79)),
        "sex": int(rng.integers(0, 2)),
        "cp": int(rng.choice([1, 2, 3, 4], p=[.22, .25, .28, .25])),
        "trestbps": int(np.clip(rng.normal(132, 19), 90, 200)),
        "chol": int(np.clip(rng.normal(230, 48), 130, 380)),
        "fbs": int(rng.choice([0, 1], p=[.82, .18])),
        "restecg": int(rng.choice([0, 1, 2], p=[.52, .42, .06])),
        "thalach": int(np.clip(rng.normal(145, 25), 80, 205)),
        "exang": int(rng.choice([0, 1], p=[.66, .34])),
        "oldpeak": round(float(np.clip(rng.gamma(1.5, 1.05), 0, 5)), 1),
        "slope": int(rng.choice([1, 2, 3], p=[.25, .48, .27])),
        "ca": int(rng.choice([0, 1, 2, 3], p=[.58, .22, .14, .06])),
        "thal": int(rng.choice([3, 6, 7], p=[.55, .13, .32])),
    }


def make_3d_performance_chart(frame):
    models = list(frame.index)
    metrics = list(frame.columns)
    fig = plt.figure(figsize=(10.5, 6.1), facecolor="#111831")
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#4DEBFF", "#FF5DB1", "#B8FF73", "#A78BFA"]
    dx, dy = .42, .42
    for j, metric in enumerate(metrics):
        vals = frame[metric].to_numpy(dtype=float)
        xs = np.arange(len(models))
        ys = np.full(len(models), j)
        ax.bar3d(xs, ys, np.zeros(len(models)), dx, dy, vals,
                 color=colors[j % len(colors)], alpha=.96, shade=True,
                 edgecolor="#FFFFFF", linewidth=.35)
        for x, y, z in zip(xs, ys, vals):
            ax.text(x + dx/2, y + dy/2, z + .025, f"{z:.0%}",
                    color="white", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(len(models)) + dx/2)
    ax.set_xticklabels(models, color="#EAF3FF", fontsize=9)
    ax.set_yticks(np.arange(len(metrics)) + dy/2)
    ax.set_yticklabels(metrics, color="#D5E2FF", fontsize=8)
    ax.set_zlim(0, 1.12)
    ax.set_zlabel("Score", color="#D5E2FF", labelpad=8)
    ax.set_title("MODEL PERFORMANCE · 3D VIEW", color="white", pad=18, fontsize=13, fontweight="bold")
    ax.view_init(elev=26, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.10, .14, .27, .95))
        axis.line.set_color("#8392B8")
    ax.tick_params(colors="#D5E2FF")
    fig.tight_layout()
    return fig


def make_3d_confusion(cm_array, model_name):
    fig = plt.figure(figsize=(7.6, 5.4), facecolor="#111831")
    ax = fig.add_subplot(111, projection="3d")
    coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
    values = [int(cm_array[0, 0]), int(cm_array[0, 1]),
              int(cm_array[1, 0]), int(cm_array[1, 1])]
    colors = ["#4DEBFF", "#FFB45E", "#FF5D91", "#B8FF73"]
    for (x, y), value, color in zip(coords, values, colors):
        ax.bar3d(x, y, 0, .55, .55, value, color=color, alpha=.96,
                 shade=True, edgecolor="white")
        ax.text(x + .27, y + .27, value + .5, str(value),
                color="white", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks([.27, 1.27])
    ax.set_xticklabels(["Actual 0", "Actual 1"], color="#EAF3FF")
    ax.set_yticks([.27, 1.27])
    ax.set_yticklabels(["Predicted 0", "Predicted 1"], color="#EAF3FF")
    ax.set_zlabel("Samples", color="#D5E2FF")
    ax.set_title(f"{model_name} · CONFUSION MATRIX", color="white", pad=14, fontsize=12, fontweight="bold")
    ax.view_init(elev=25, azim=-52)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.10, .14, .27, .95))
        axis.line.set_color("#8392B8")
    ax.tick_params(colors="#D5E2FF")
    fig.tight_layout()
    return fig


# ----------------------------- LOAD MODEL -------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    st.error("The model or metrics file is missing. Check the models/ folder in the repository.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        metrics_data = json.load(handle)
except Exception as exc:
    st.error(f"Unable to load the saved model: {exc}")
    st.stop()

selected_model = metrics_data.get("selected_model", "Not specified")
results = metrics_data.get("results", {})
selected_metrics = results.get(selected_model, {})

# ----------------------------- SIDEBAR ----------------------------------
with st.sidebar:
    st.markdown('<div class="brand">💓 Cardio<span>Predict</span></div>', unsafe_allow_html=True)
    st.markdown('<span class="side-chip">HEART HEALTH INTELLIGENCE</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Your workspace**")
    st.caption("Explore a sample, run a model prediction, stress-test scenarios and inspect model performance.")
    st.markdown("**Under the hood**")
    st.caption("Python · Streamlit · Pandas · NumPy")
    st.caption("Scikit-learn · Joblib · Matplotlib 3D")
    st.markdown("---")
    st.markdown("**Model pipeline**")
    st.caption("13 input features → saved preprocessing pipeline → classifier → binary output")
    st.markdown("---")
    st.caption("Prototype notice: not clinically validated. Never use this app to diagnose, screen or decide whether to seek emergency care.")

# ----------------------------- HERO -------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-top">
        <span class="pill"><span class="live-dot"></span> Interactive model workspace</span>
        <span class="pill">13 feature signals</span>
        <span class="pill">3D analytics</span>
      </div>
      <h1>Understand the signals.<br><span>Explore the pattern.</span></h1>
      <p>CardioPredict turns a set of heart-related measurements into an easy-to-explore machine-learning output.
      Run a single profile, generate a batch of fictional scenarios, and see how the model behaves.</p>
      <div class="ecg"><svg viewBox="0 0 520 70" preserveAspectRatio="none">
        <path d="M2 38 L82 38 L101 35 L116 41 L132 38 L157 38 L174 13 L192 59 L212 27 L229 38 L290 38 L309 34 L325 41 L340 38 L367 38 L383 18 L401 56 L419 29 L436 38 L518 38"/>
      </svg></div>
      <div class="hero-foot">
        <div class="hero-stat"><b>Model</b> · {model_name}</div>
        <div class="hero-stat"><b>Dataset</b> · Cleveland heart-disease dataset</div>
        <div class="hero-stat"><b>Mode</b> · Explore + test</div>
      </div>
    </div>
    """.replace("{model_name}", str(selected_model)),
    unsafe_allow_html=True,
)

# ----------------------------- TOP METRICS ------------------------------
metrics_cols = st.columns(4)
cards = [
    ("SELECTED MODEL", selected_model, "Chosen during model training", "#4DEBFF"),
    ("TEST ACCURACY", pct(selected_metrics.get("accuracy")), "Held-out test subset", "#B8FF73"),
    ("PRECISION · GROUP 1", pct(selected_metrics.get("precision")), "Positive-class precision", "#FF8BCB"),
    ("RECALL · GROUP 1", pct(selected_metrics.get("recall")), "Positive-class recall", "#B9A2FF"),
]
for col, (label, value, note, accent) in zip(metrics_cols, cards):
    with col:
        st.markdown(
            f'<div class="glass-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="color:{accent}">{value}</div>'
            f'<div class="stat-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# ----------------------------- WORKSPACE TABS ---------------------------
live_tab, lab_tab, analytics_tab, guide_tab, emergency_tab = st.tabs(
    ["✦ Live explorer", "▦ Scenario lab", "◈ 3D analytics", "♡ Heart guide", "✚ Emergency & map"]
)

# ----------------------------- LIVE EXPLORER ----------------------------
with live_tab:
    st.markdown('<div class="section-kicker">01 / Live explorer</div>', unsafe_allow_html=True)
    st.header("Run a profile through the model")
    st.markdown('<div class="section-sub">Enter recorded measurements, or generate a fictional profile to see how the interface works.</div>', unsafe_allow_html=True)

    gen_col, info_col = st.columns([1, 2])
    with gen_col:
        if st.button("🎲 Generate a sample profile", use_container_width=True, key="generate_one"):
            rng = np.random.default_rng()
            sample = synthetic_profile(rng)
            for key, value in sample.items():
                st.session_state["field_" + key] = value
            st.session_state["last_prediction"] = None
            st.rerun()
    with info_col:
        st.markdown('<div class="info-box">Sample profiles are computer-generated fictional test inputs. They are not real patient records, clinical reference cases or medical advice.</div>', unsafe_allow_html=True)

    with st.form("live_prediction_form", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### ◉ Personal and resting measurements")
            age = st.number_input("Age · years", min_value=18, max_value=100, key="field_age",
                                  help="Age in years.")
            sex = st.selectbox("Sex · dataset coding", [0, 1], key="field_sex",
                               format_func=lambda x: "Female · 0" if x == 0 else "Male · 1")
            cp = st.selectbox("Chest pain category", [1, 2, 3, 4], key="field_cp",
                              format_func=lambda x: {1:"1 · Typical angina",2:"2 · Atypical angina",3:"3 · Non-anginal pain",4:"4 · Asymptomatic in dataset"}[x],
                              help="Dataset code, not a symptom self-assessment.")
            trestbps = st.number_input("Resting blood pressure · mm Hg", min_value=80, max_value=250, key="field_trestbps")
            chol = st.number_input("Serum cholesterol · mg/dL", min_value=100, max_value=600, key="field_chol")
            fbs = st.selectbox("Fasting blood sugar above 120 mg/dL?", [0, 1], key="field_fbs",
                               format_func=lambda x: "No · 0" if x == 0 else "Yes · 1")
            restecg = st.selectbox("Resting ECG category", [0, 1, 2], key="field_restecg",
                                   format_func=lambda x: {0:"0 · Normal",1:"1 · ST-T wave abnormality",2:"2 · LV hypertrophy category"}[x])
        with right:
            st.markdown("#### ◈ Exercise and cardiac measurements")
            thalach = st.number_input("Maximum heart rate achieved · bpm", min_value=60, max_value=250, key="field_thalach")
            exang = st.selectbox("Exercise-induced angina?", [0, 1], key="field_exang",
                                 format_func=lambda x: "No · 0" if x == 0 else "Yes · 1")
            oldpeak = st.number_input("ST depression · oldpeak", min_value=0.0, max_value=10.0, step=.1, key="field_oldpeak")
            slope = st.selectbox("Peak exercise ST-segment slope", [1, 2, 3], key="field_slope",
                                  format_func=lambda x: {1:"1 · Upsloping",2:"2 · Flat",3:"3 · Downsloping"}[x])
            ca = st.selectbox("Major vessels seen by fluoroscopy", [0, 1, 2, 3], key="field_ca")
            thal = st.selectbox("Thal test category", [3, 6, 7], key="field_thal",
                                 format_func=lambda x: {3:"3 · Normal",6:"6 · Fixed defect",7:"7 · Reversible defect"}[x])
            st.markdown('<div class="small-muted">Use values from a recorded clinical report. Do not estimate medical measurements.</div>', unsafe_allow_html=True)
        run_prediction = st.form_submit_button("Run model analysis  →", use_container_width=True)

    if run_prediction:
        profile = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }
        input_df = pd.DataFrame([profile], columns=FEATURES)
        try:
            predicted = int(model.predict(input_df)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                idx = classes.index(1) if 1 in classes else min(1, len(probs)-1)
                probability = float(probs[idx])
            st.session_state["last_prediction"] = {
                "predicted": predicted, "probability": probability, "profile": profile
            }
        except Exception as exc:
            st.error(f"Model analysis failed: {exc}")

    output = st.session_state.get("last_prediction")
    if output:
        st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">02 / Model output</div>', unsafe_allow_html=True)
        predicted = output["predicted"]
        if predicted == 1:
            st.markdown(
                '<div class="result-card alert"><div class="result-tag">Output group · 1</div>'
                '<div class="result-title">Pattern aligned with the dataset disease-present group</div>'
                '<div class="small-muted">The model assigned this input to the group labelled disease-present in its historical training data. This does not establish that a person has cardiovascular disease.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card"><div class="result-tag">Output group · 0</div>'
                '<div class="result-title">Pattern aligned with the dataset no-disease group</div>'
                '<div class="small-muted">The model assigned this input to the group labelled no disease in its historical training data. This does not rule out disease or guarantee that a person is healthy.</div></div>',
                unsafe_allow_html=True,
            )
        prob = output["probability"]
        if prob is not None:
            p1, p2 = st.columns([1, 2])
            with p1:
                st.metric("Model probability · Group 1", pct(prob))
            with p2:
                st.progress(max(0.0, min(1.0, prob)))
                st.caption("This is the classifier's output probability, not a clinically calibrated personal-risk percentage or severity score.")
        with st.expander("View the 13 values sent to the model"):
            st.dataframe(pd.DataFrame([output["profile"]]), use_container_width=True, hide_index=True)
        st.markdown('<div class="notice"><b>Important:</b> This prototype is not clinically validated. It must not be used for diagnosis, screening, treatment decisions or emergency triage. Seek professional medical assessment for health concerns.</div>', unsafe_allow_html=True)

# ----------------------------- SCENARIO LAB -----------------------------
with lab_tab:
    st.markdown('<div class="section-kicker">02 / Scenario lab</div>', unsafe_allow_html=True)
    st.header("Stress-test the model with synthetic profiles")
    st.markdown('<div class="section-sub">Generate a larger batch of fictional input combinations and inspect the model outputs. This is a software-testing feature, not a clinical study.</div>', unsafe_allow_html=True)
    controls = st.columns([1, 1, 2])
    with controls[0]:
        sample_count = st.selectbox("Profiles to generate", [25, 50, 100, 250, 500], index=2)
    with controls[1]:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    with controls[2]:
        st.markdown('<div class="info-box">The generator creates varied values within the app\'s accepted input ranges. It does not create real people or add data to model training.</div>', unsafe_allow_html=True)
    if st.button("⚡ Generate and analyse batch", key="run_batch", use_container_width=True):
        rng = np.random.default_rng(int(seed))
        rows = [synthetic_profile(rng) for _ in range(int(sample_count))]
        batch = pd.DataFrame(rows, columns=FEATURES)
        try:
            predictions = model.predict(batch)
            batch["model_output_group"] = predictions.astype(int)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(batch[FEATURES])
                classes = list(getattr(model, "classes_", [0, 1]))
                idx = classes.index(1) if 1 in classes else min(1, probs.shape[1]-1)
                batch["model_probability_group_1"] = probs[:, idx].round(4)
            st.session_state["synthetic_batch"] = batch
            st.session_state["synthetic_batch_seed"] = int(seed)
        except Exception as exc:
            st.error(f"Batch analysis failed: {exc}")

    batch = st.session_state.get("synthetic_batch")
    if batch is not None:
        total = len(batch)
        group1 = int((batch["model_output_group"] == 1).sum())
        group0 = total - group1
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Synthetic profiles", total)
        with m2:
            st.metric("Output group 1", group1)
        with m3:
            st.metric("Output group 0", group0)
        st.caption(f"Reproducible random seed: {st.session_state.get('synthetic_batch_seed', '—')}")
        bar_df = pd.DataFrame({"Model output": ["Group 0", "Group 1"], "Profiles": [group0, group1]}).set_index("Model output")
        st.bar_chart(bar_df, color=["#4DEBFF"])
        st.markdown("#### Generated sample data")
        st.dataframe(batch.head(50), use_container_width=True, hide_index=True)
        csv_bytes = batch.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download generated profiles as CSV", csv_bytes,
                           file_name="cardiopredict_synthetic_test_profiles.csv",
                           mime="text/csv", use_container_width=True)
        st.caption("Only the first 50 rows are previewed above. Download includes the full generated batch.")
        st.markdown('<div class="notice"><b>Interpretation:</b> The group counts describe only how this model classified these artificial inputs. They are not disease prevalence, accuracy, risk rates or evidence about real patients.</div>', unsafe_allow_html=True)

# ----------------------------- ANALYTICS ---------------------------------
with analytics_tab:
    st.markdown('<div class="section-kicker">03 / Model analytics</div>', unsafe_allow_html=True)
    st.header("Explore the model's test performance")
    st.markdown('<div class="section-sub">The metrics below come from the saved evaluation results on the held-out test split.</div>', unsafe_allow_html=True)
    if results:
        rows = []
        for name, values in results.items():
            rows.append({
                "Algorithm": name,
                "Accuracy": safe_float(values.get("accuracy")),
                "Precision · Group 1": safe_float(values.get("precision")),
                "Recall · Group 1": safe_float(values.get("recall")),
                "F1 · Group 1": safe_float(values.get("f1_score")),
            })
        comparison = pd.DataFrame(rows).set_index("Algorithm")
        st.dataframe(comparison.style.format("{:.1%}"), use_container_width=True)
        st.markdown("#### 3D performance landscape")
        fig = make_3d_performance_chart(comparison)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("#### Confusion matrix explorer")
        chosen = st.selectbox("Choose an algorithm", list(results.keys()), key="cm_model")
        matrix = results[chosen].get("confusion_matrix")
        if matrix and len(matrix) == 2:
            matrix_array = np.asarray(matrix)
            a, b = st.columns([.9, 1.1])
            with a:
                st.dataframe(
                    pd.DataFrame(matrix_array,
                                 index=["Actual group 0", "Actual group 1"],
                                 columns=["Predicted group 0", "Predicted group 1"]),
                    use_container_width=True,
                )
                st.caption("Rows are actual dataset labels; columns are model predictions.")
            with b:
                fig_cm = make_3d_confusion(matrix_array, chosen)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            st.markdown("- **True negative:** correctly assigned to Group 0.\n- **False positive:** assigned to Group 1 when the dataset label was Group 0.\n- **False negative:** assigned to Group 0 when the dataset label was Group 1.\n- **True positive:** correctly assigned to Group 1.")
    else:
        st.info("No saved evaluation results were found. Check models/metrics.json.")

# ----------------------------- HEART GUIDE -------------------------------
with guide_tab:
    st.markdown('<div class="section-kicker">04 / Heart guide</div>', unsafe_allow_html=True)
    st.header("Cardiovascular health, in everyday language")
    st.markdown('<div class="section-sub">A quick guide to the measurements represented in the original dataset.</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="glass-card"><div class="stat-label">01 · BLOOD PRESSURE</div><div class="stat-value">Pressure</div><p>Measures the force of blood against artery walls. Persistent high blood pressure should be discussed with a healthcare professional.</p></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="glass-card"><div class="stat-label">02 · CHOLESTEROL</div><div class="stat-value">Blood fats</div><p>Cholesterol is a fatty substance in the blood. A lipid profile helps clinicians understand different cholesterol components.</p></div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="glass-card"><div class="stat-label">03 · HEART RATE</div><div class="stat-value">Beats</div><p>Heart rate is the number of heartbeats per minute. Exercise-test readings need context and professional interpretation.</p></div>', unsafe_allow_html=True)
    st.markdown("#### The 13 model inputs")
    descriptions = [
        ("Age", "Age in years."),
        ("Sex", "Original dataset coding: 0 or 1."),
        ("Chest pain category", "One of four historical dataset categories."),
        ("Resting blood pressure", "Blood pressure recorded at rest, in mm Hg."),
        ("Cholesterol", "Serum cholesterol, in mg/dL."),
        ("Fasting blood sugar", "Whether fasting blood sugar was above 120 mg/dL."),
        ("Resting ECG", "Coded resting electrocardiogram category."),
        ("Maximum heart rate", "Maximum rate achieved during the exercise test."),
        ("Exercise-induced angina", "Recorded yes/no exercise-test field."),
        ("Oldpeak", "ST depression measurement in the exercise test."),
        ("ST slope", "Coded slope category during peak exercise."),
        ("Major vessels", "Number of major vessels recorded by fluoroscopy."),
        ("Thal", "Historical coded thal test result."),
    ]
    left, right = st.columns(2)
    for i, (name, desc) in enumerate(descriptions):
        target = left if i % 2 == 0 else right
        with target:
            st.markdown(f"**{i+1:02d}. {name}**")
            st.caption(desc)
    st.markdown('<div class="info-box">The original Cleveland dataset uses historical clinical test fields and numeric codes. Some inputs are not measurements a person can safely estimate at home. Use the values only when they are available from an appropriate clinical report.</div>', unsafe_allow_html=True)
    st.markdown("#### Everyday habits that support heart health")
    st.markdown("- Avoid tobacco and smoking.\n- Stay physically active in a way that is appropriate for your health.\n- Choose a balanced diet and limit excess salt.\n- Attend recommended health checks.\n- Follow treatment prescribed by your clinician.\n- Seek medical advice for new, persistent or worrying symptoms.")

# ----------------------------- EMERGENCY & MAP ---------------------------
with emergency_tab:
    st.markdown('<div class="section-kicker">05 / Emergency support</div>', unsafe_allow_html=True)
    st.header("Find help. Don't wait for an app result.")
    st.markdown(
        '<div class="emergency"><h3>Possible medical emergency?</h3>'
        '<p>Severe or new chest pressure, pain spreading to the arm or jaw, severe breathing difficulty, fainting or collapse needs urgent medical attention. Do not wait for a CardioPredict result.</p>'
        '<a class="phone" href="tel:112">Call 112 · Emergency</a>'
        '<a class="phone" href="tel:102">Call 102 · Ambulance</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("In India, 112 is the nationwide integrated emergency number. 102 is listed as a national ambulance service; local availability and dispatch arrangements may vary.")
    st.markdown("#### Locate nearby hospitals")
    st.markdown('<div class="section-sub">Search by your city, neighbourhood or PIN code. Google Maps will open with nearby hospital listings.</div>', unsafe_allow_html=True)
    place = st.text_input("Enter a location", placeholder="For example: Jayanagar, Bengaluru")
    if place.strip():
        q = quote("emergency hospital near " + place.strip())
        url = "https://www.google.com/maps/search/?api=1&query=" + q
        st.link_button("Open nearby hospitals in Google Maps ↗", url, use_container_width=True)
    else:
        st.info("Enter a location to create your Google Maps hospital search.")
    st.markdown('<div class="notice"><b>Important:</b> This app does not access GPS, call hospitals, dispatch an ambulance or confirm emergency-department availability. For an emergency, call the emergency number directly.</div>', unsafe_allow_html=True)

# ----------------------------- FOOTER -----------------------------------
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">CARDIOPREDICT · HEART HEALTH INTELLIGENCE · MACHINE-LEARNING PROJECT<br>'
    'Synthetic scenarios are fictional software-test inputs · Prototype not clinically validated · Not for diagnosis or emergency triage</div>',
    unsafe_allow_html=True,
)
