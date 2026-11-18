import json
import os
from urllib.parse import quote

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from src.assistant import render_cardio_assistant
from src.auth import render_auth_gate, render_logout_control

# CardioPredict - product-style interface and synthetic scenario explorer.
# The saved model, original 13 features, preprocessing and training pipeline
# are intentionally unchanged. Synthetic profiles are for software testing
# only and are never added to the training dataset.

st.set_page_config(
    page_title="CardioPredict | Heart & Life",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
      color-scheme: light;
      --ink: #172B3F;
      --body: #536577;
      --muted: #758494;
      --navy: #142F49;
      --rose: #C45C70;
      --rose-dark: #9E4058;
      --teal: #168C91;
      --line: #E5EAF0;
      --paper: #FCFAF8;
      --white: #FFFFFF;
      --shadow: 0 14px 38px rgba(20,47,73,.075);
    }
    html, body, [class*="css"], button, input, textarea, select {
      font-family: 'DM Sans','Segoe UI',sans-serif !important;
    }
    .stApp { background: linear-gradient(180deg,#FCFAF8 0%,#F6F8FA 100%) !important; color:var(--ink)!important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background:transparent!important; color:var(--ink)!important; }
    [data-testid="stHeader"] { background:rgba(252,250,248,.94)!important; border-bottom:1px solid var(--line)!important; }
    .block-container { max-width:1440px!important; padding:1.4rem 1.6rem 3rem!important; }
    h1,h2,h3,h4,h5,h6,[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3 {
      font-family:'Manrope','DM Sans',sans-serif!important; color:var(--navy)!important;
      letter-spacing:-.035em!important; font-weight:800!important; text-shadow:none!important;
    }
    h1 {font-size:clamp(2rem,4vw,2.8rem)!important;line-height:1.12!important}
    h2 {font-size:clamp(1.65rem,3vw,2.1rem)!important;line-height:1.2!important}
    h3 {font-size:1.35rem!important;line-height:1.3!important}
    .stApp p,.stApp li,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li {color:var(--body)!important;line-height:1.68!important}
    .stApp strong,[data-testid="stMarkdownContainer"] strong {color:var(--navy)!important;font-weight:800!important}
    [data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] label,.stApp label {color:var(--ink)!important;font-weight:700!important}
    [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {color:var(--muted)!important}
    a {color:var(--teal)!important;text-underline-offset:3px}
    section[data-testid="stSidebar"] {background:linear-gradient(180deg,#F1F5F8,#EAF0F4)!important;border-right:1px solid #DCE5EC!important}
    section[data-testid="stSidebar"] * {color:var(--ink)!important}
    .brand {font:800 1.4rem 'Manrope',sans-serif;letter-spacing:-.05em;color:var(--navy)!important}
    .brand span {color:var(--rose)!important}
    .side-chip {display:inline-block;padding:6px 11px;border-radius:99px;background:#F9E8EC!important;color:#8D3E55!important;border:1px solid #EBCBD3;font-size:.72rem;font-weight:800;letter-spacing:.08em}
    .hero {position:relative;overflow:hidden;isolation:isolate;border-radius:26px;padding:2.15rem 2.35rem;margin:.2rem 0 1.35rem;background:linear-gradient(112deg,#102B45 0%,#194C67 62%,#A94761 100%)!important;border:1px solid rgba(255,255,255,.2);box-shadow:0 22px 48px rgba(16,43,69,.16);animation:cp-rise .45s ease both}
    .hero:before {content:"";position:absolute;z-index:-1;width:420px;height:420px;right:-110px;top:-230px;border-radius:50%;background:radial-gradient(circle,rgba(77,206,199,.24),rgba(77,206,199,.04) 60%,transparent 72%)}
    .hero-top,.hero-foot {display:flex;align-items:center;gap:9px;flex-wrap:wrap}
    .hero-top {margin-bottom:1rem}
    .pill {display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border-radius:99px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.24);color:#FFF!important;font-size:.75rem;font-weight:800;letter-spacing:.045em}
    .live-dot {width:8px;height:8px;border-radius:50%;background:#8CE0BE;box-shadow:0 0 0 3px rgba(140,224,190,.16)}
    .hero h1 {color:#FFF!important;font-family:'Manrope',sans-serif!important;font-size:clamp(2.1rem,5vw,3.5rem)!important;line-height:1.06!important;margin:0 0 .7rem!important;max-width:850px}
    .hero h1 span {color:#8DE3DF!important;-webkit-text-fill-color:#8DE3DF!important}
    .hero p {max-width:770px;color:#EAF5F8!important;font-size:1.04rem!important;line-height:1.65!important;margin:0}
    .hero-foot {margin-top:1.3rem}
    .hero-stat {padding:9px 13px;border-radius:12px;background:rgba(5,24,39,.24);border:1px solid rgba(255,255,255,.18);color:#FFF!important;font-size:.84rem}
    .hero-stat b {color:#9CE7E5!important}
    .ecg {height:34px;width:min(420px,75%);margin-top:1.05rem}
    .ecg svg {height:100%;width:100%}
    .ecg path {fill:none;stroke:#A7ECE9;stroke-width:2.5;stroke-linecap:round;stroke-dasharray:850;stroke-dashoffset:850;animation:cp-draw 3s ease .2s forwards}
    @keyframes cp-draw {to{stroke-dashoffset:0}}
    @keyframes cp-rise {from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    .section-kicker {color:var(--rose-dark)!important;font-size:.74rem!important;font-weight:900!important;letter-spacing:.13em!important;text-transform:uppercase;margin-bottom:.3rem}
    .section-sub {color:var(--muted)!important;font-size:.97rem!important;line-height:1.6!important;margin-top:-.35rem;margin-bottom:1rem}
    .glass-card {height:100%;background:linear-gradient(145deg,#FFF,#F7F9FB)!important;border:1px solid var(--line)!important;border-radius:20px;padding:18px 20px;box-shadow:var(--shadow);transition:transform .18s ease,box-shadow .18s ease}
    .glass-card:hover {transform:translateY(-2px);box-shadow:0 17px 36px rgba(20,47,73,.11)}
    .stat-label {color:var(--muted)!important;font-size:.75rem!important;font-weight:900!important;letter-spacing:.075em!important;text-transform:uppercase}
    .stat-value {color:var(--navy)!important;font:800 1.8rem 'Manrope',sans-serif!important;margin-top:7px;line-height:1.25}
    .stat-note {color:var(--muted)!important;font-size:.88rem!important;margin-top:5px;line-height:1.5!important}
    .glow-line {height:2px;border-radius:5px;background:linear-gradient(90deg,#C45C70,#45B9B6,#A94761,transparent);margin:1.25rem 0}
    div[data-testid="stForm"] {background:rgba(255,255,255,.96)!important;border:1px solid var(--line)!important;border-radius:20px!important;padding:1.05rem 1.15rem!important;box-shadow:var(--shadow)}
    /* One consistent light-field rule for all widgets; no competing dark overrides. */
    .stApp [data-baseweb="input"]>div,.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="textarea"]>div,
    .stApp [data-testid="stTextArea"] textarea {background:#FFFFFF!important;border:1px solid #CBD7E1!important;border-radius:11px!important;box-shadow:none!important}
    .stApp input,.stApp textarea,.stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] * {
      color:#172B3F!important;-webkit-text-fill-color:#172B3F!important;caret-color:#168C91!important;opacity:1!important;
    }
    .stApp input::placeholder,.stApp textarea::placeholder {color:#718293!important;-webkit-text-fill-color:#718293!important;opacity:1!important}
    .stApp [data-testid="stNumberInput"] button {background:#EEF3F6!important;color:#173A55!important;border:1px solid #CBD7E1!important}
    .stApp [data-testid="stNumberInput"] button svg {fill:#173A55!important;color:#173A55!important}
    .stApp [data-baseweb="popover"],.stApp [data-baseweb="menu"],.stApp ul[role="listbox"] {background:#FFFFFF!important;border:1px solid #DCE5EC!important;box-shadow:0 12px 28px rgba(20,47,73,.14)!important}
    .stApp [role="option"],.stApp [role="listbox"] * {background:#FFFFFF!important;color:#172B3F!important;-webkit-text-fill-color:#172B3F!important}
    .stApp [role="option"]:hover {background:#F9E8EC!important}
    .stApp [data-baseweb="input"]>div:focus-within,.stApp [data-baseweb="select"]>div:focus-within,.stApp textarea:focus {
      border-color:#168C91!important;box-shadow:0 0 0 3px rgba(22,140,145,.15)!important
    }
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button,.stLinkButton>a {
      min-height:2.85rem!important;border:1px solid #A94761!important;border-radius:11px!important;padding:.55rem 1.05rem!important;
      background:linear-gradient(100deg,#A94761,#C45C70)!important;color:#FFF!important;-webkit-text-fill-color:#FFF!important;
      font-size:.96rem!important;font-weight:800!important;box-shadow:0 7px 18px rgba(169,71,97,.16)!important;
      transition:transform .18s ease,filter .18s ease!important;text-decoration:none!important
    }
    .stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover,.stLinkButton>a:hover {transform:translateY(-1px);filter:brightness(1.06)}
    .stButton>button:focus-visible,.stFormSubmitButton>button:focus-visible,input:focus-visible,textarea:focus-visible {outline:3px solid rgba(22,140,145,.25)!important;outline-offset:2px!important}
    .stTabs [data-baseweb="tab-list"] {gap:6px!important;background:#EDF2F5!important;border:1px solid #DCE5EC!important;border-radius:14px!important;padding:6px!important}
    .stTabs [data-baseweb="tab"] {min-height:2.7rem!important;padding:8px 13px!important;border-radius:9px!important;color:#40566A!important;font-size:.9rem!important;font-weight:750!important;white-space:nowrap!important}
    .stTabs [data-baseweb="tab"] p {color:inherit!important;font-size:inherit!important}
    .stTabs [aria-selected="true"] {background:#FFFFFF!important;border:1px solid #D5E0E8!important;color:#9E4058!important;box-shadow:0 2px 8px rgba(20,47,73,.07)}
    .stTabs [data-baseweb="tab-highlight"] {background:var(--rose)!important}
    [data-testid="stDataFrame"],[data-testid="stTable"] {border:1px solid var(--line)!important;border-radius:12px!important;overflow:hidden!important;background:#FFF!important}
    [data-testid="stMetric"] {background:#FFF!important;border:1px solid var(--line)!important;border-radius:14px!important;padding:13px 15px!important;box-shadow:0 5px 16px rgba(20,47,73,.05)}
    [data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"],[data-testid="stMetricValue"] *,[data-testid="stMetricDelta"],[data-testid="stMetricDelta"] * {color:var(--ink)!important}
    [data-testid="stMetricValue"] {font-family:'Manrope',sans-serif!important}
    [data-testid="stProgressBar"]>div {background:#E8EEF2!important}
    [data-testid="stProgressBar"]>div>div {background:linear-gradient(90deg,#A94761,#168C91)!important}
    [data-testid="stExpander"] {border:1px solid var(--line)!important;border-radius:12px!important;background:rgba(255,255,255,.9)!important}
    [data-testid="stExpander"] summary,[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span {color:var(--navy)!important;font-weight:750!important}
    [data-testid="stAlert"] {border-radius:12px!important;border:1px solid var(--line)!important}
    .info-box {background:#EAF3F6!important;border:1px solid #D3E4EA!important;border-left:4px solid var(--teal)!important;color:#24485D!important;padding:13px 16px;border-radius:12px;font-size:.95rem!important;line-height:1.6!important}
    .notice {background:#FFF5E9!important;border:1px solid #EAD8B9!important;border-left:4px solid #B48242!important;color:#674C2B!important;padding:13px 16px;border-radius:12px;font-size:.95rem!important;line-height:1.6!important}
    .result-card {border-radius:17px;padding:1.2rem 1.35rem;border:1px solid #B9D9C9;border-left:5px solid #4D896F;background:linear-gradient(120deg,#EAF5EF,#F7FBF8)!important;box-shadow:0 9px 24px rgba(55,104,72,.08)}
    .result-card.alert {border-color:#E5BBC2;border-left-color:#C45C70;background:linear-gradient(120deg,#FBECEF,#FFF8F7)!important}
    .result-tag {color:#367B63!important;font-size:.74rem;font-weight:900;letter-spacing:.1em;text-transform:uppercase}
    .result-card.alert .result-tag {color:#A33B50!important}
    .result-title {color:var(--navy)!important;font:800 1.32rem 'Manrope',sans-serif;margin:.4rem 0;line-height:1.35}
    .result-card .small-muted {color:#526B5D!important}.result-card.alert .small-muted {color:#704F56!important}
    .emergency {background:linear-gradient(112deg,#102B45,#194C67 58%,#A94761)!important;border:1px solid #194C67;border-radius:19px;padding:1.3rem 1.45rem;box-shadow:0 12px 30px rgba(16,43,69,.16)}
    .emergency h3 {color:#FFF!important;margin-top:0}.emergency p {color:#EAF5F8!important}
    .phone {display:inline-block;background:#FFF!important;color:#173A55!important;text-decoration:none!important;padding:10px 16px;border-radius:10px;font-weight:900;margin:.4rem .5rem .2rem 0}
    .footer {color:var(--muted)!important;font-size:.82rem!important;line-height:1.7!important;text-align:center;padding:1.35rem 0 .4rem}
    .auth-shell {max-width:1080px;margin:2.4rem auto 1rem;display:grid;grid-template-columns:1.04fr .96fr;gap:0;border:1px solid #E1E7EC;border-radius:28px;overflow:hidden;background:#FFF;box-shadow:0 28px 70px rgba(20,47,73,.12)}
    .auth-brand {padding:3rem 2.6rem;background:linear-gradient(145deg,#102B45,#194C67 68%,#A94761);color:#FFF;display:flex;flex-direction:column;justify-content:space-between;min-height:520px}
    .auth-brand h1 {color:#FFF!important;font:800 clamp(2.1rem,4vw,3.15rem) 'Manrope',sans-serif!important;line-height:1.08!important;margin:2rem 0 1rem!important}
    .auth-brand p {color:#EAF5F8!important;font-size:1rem!important;line-height:1.7!important}
    .auth-brand .eyebrow {color:#A9E7E1!important;font-size:.75rem;font-weight:900;letter-spacing:.15em}
    .auth-brand .trust {color:#DCEAF0!important;font-size:.84rem}
    .auth-panel {padding:2.6rem 2.3rem;background:#FFF}
    .auth-panel h2 {margin-top:.25rem!important}
    .auth-mark {font:800 1.25rem 'Manrope',sans-serif;color:#FFF}
    .auth-mark span {color:#A9E7E1}
    @media(max-width:760px) {
      .block-container {padding:.9rem .8rem 2.5rem!important}
      .hero {padding:1.5rem;border-radius:19px}
      .auth-shell {display:block;margin:1rem auto}
      .auth-brand {min-height:unset;padding:2rem 1.5rem}
      .auth-brand h1 {font-size:2.1rem!important;margin:1.1rem 0 .6rem!important}
      .auth-panel {padding:1.7rem 1.35rem}
      .stTabs [data-baseweb="tab-list"] {overflow-x:auto!important}
      .stTabs [data-baseweb="tab"] {padding:7px 9px!important;font-size:.83rem!important}
      .glass-card {padding:15px}
    }
    @media(prefers-reduced-motion:reduce) {*{animation-duration:.01ms!important;transition-duration:.01ms!important}}
    </style>
    """,
    unsafe_allow_html=True,
)

if not render_auth_gate():
    st.stop()

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


def make_performance_chart(frame):
    """Compare held-out model metrics with aligned, readable horizontal bars."""
    metrics=list(frame.columns); models=list(frame.index)
    fig,ax=plt.subplots(figsize=(10.5,5.4),facecolor="#FFFFFF"); ax.set_facecolor("#FFFFFF")
    palette={"Logistic Regression":"#BD5969","Random Forest":"#D77983"}; fallback=["#BD5969","#D77983","#567C91"]
    y=np.arange(len(metrics)); height=.34
    for i,name in enumerate(models):
        values=frame.loc[name,metrics].to_numpy(dtype=float)
        bars=ax.barh(y+(i-(len(models)-1)/2)*height,values,height=height*.88,
                     color=palette.get(str(name),fallback[i%len(fallback)]),edgecolor="#FFFFFF",
                     linewidth=1.4,label=str(name),zorder=3)
        for bar,value in zip(bars,values):
            ax.text(min(value+.014,1.045),bar.get_y()+bar.get_height()/2,f"{value:.1%}",
                    va="center",ha="left",color="#49363B",fontsize=10,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([str(m).replace("_"," ").title() for m in metrics],
                                          color="#49363B",fontsize=10,fontweight="medium")
    ax.invert_yaxis(); ax.set_xlim(0,1.14); ax.set_xticks(np.linspace(0,1,6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0,1,6)])
    ax.set_xlabel("Evaluation score",color="#66545A",labelpad=10,fontsize=10)
    ax.set_title("MODEL PERFORMANCE COMPARISON",color="#743F4A",pad=18,fontsize=14,fontweight="bold")
    ax.xaxis.grid(True,color="#F0D8DA",linewidth=.9,zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x",colors="#8B777D",labelsize=9,length=0,pad=7); ax.tick_params(axis="y",length=0,pad=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.02),ncol=max(1,min(len(models),2)),
              frameon=False,labelcolor="#49363B",fontsize=9)
    fig.subplots_adjust(left=.23,right=.96,top=.82,bottom=.15)
    return fig


def make_confusion_matrix_chart(cm_array,model_name):
    """Display confusion-matrix counts with a soft rose colour scale and labels."""
    from matplotlib.colors import LinearSegmentedColormap
    values=np.asarray(cm_array,dtype=int)
    fig,ax=plt.subplots(figsize=(7.4,5.3),facecolor="#FFFFFF"); ax.set_facecolor("#FFFFFF")
    cmap=LinearSegmentedColormap.from_list("cardiopredict_rose",["#E8F4F7","#B9E1E3","#55B8BC","#BD5969"])
    vmax=max(int(values.max()),1); ax.imshow(values,cmap=cmap,vmin=0,vmax=vmax,aspect="equal")
    ax.set_xticks([0,1],labels=["Group 0","Group 1"]); ax.set_yticks([0,1],labels=["Group 0","Group 1"])
    ax.set_xlabel("Predicted dataset group",color="#66545A",labelpad=10,fontsize=10)
    ax.set_ylabel("Actual dataset group",color="#66545A",labelpad=10,fontsize=10)
    ax.set_title(f"{model_name.upper()} · CONFUSION MATRIX",color="#743F4A",pad=18,fontsize=13,fontweight="bold")
    ax.tick_params(axis="both",colors="#49363B",labelsize=10,length=0,pad=9)
    descriptions=[["True negative","False positive"],["False negative","True positive"]]; threshold=vmax*.52
    for row in range(2):
        for col in range(2):
            ink="#FFFFFF" if values[row,col]>=threshold and values[row,col]>0 else "#49343B"
            ax.text(col,row,f"{values[row,col]}\n{descriptions[row][col]}",ha="center",va="center",
                    color=ink,fontsize=9,fontweight="bold")
    ax.set_xticks(np.arange(-.5,2,1),minor=True); ax.set_yticks(np.arange(-.5,2,1),minor=True)
    ax.grid(which="minor",color="#FFFFFF",linewidth=3); ax.tick_params(which="minor",bottom=False,left=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.20,right=.96,top=.84,bottom=.17)
    return fig


def make_scenario_chart(group0,group1):
    """Plot synthetic output counts using the same accessible heart-health palette."""
    fig,ax=plt.subplots(figsize=(8.5,3.2),facecolor="#FFFFFF"); ax.set_facecolor("#FFFFFF")
    labels=["Output group 0","Output group 1"]; counts=[int(group0),int(group1)]
    bars=ax.barh(labels,counts,color=["#367B63","#D77983"],height=.56,
                 edgecolor="#FFFFFF",linewidth=1.4,zorder=3)
    maximum=max(max(counts),1); ax.set_xlim(0,maximum*1.18)
    ax.xaxis.grid(True,color="#F0D8DA",linewidth=.8,zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x",colors="#8B777D",labelsize=9,length=0)
    ax.tick_params(axis="y",colors="#49363B",labelsize=10,length=0,pad=9)
    ax.set_xlabel("Number of fictional test profiles",color="#F1E5E8",fontsize=9,labelpad=8)
    ax.set_title("SYNTHETIC OUTPUT DISTRIBUTION",color="#743F4A",fontsize=12,fontweight="bold",pad=13)
    for bar,count in zip(bars,counts):
        ax.text(count+maximum*.025,bar.get_y()+bar.get_height()/2,f"{count:,}",
                va="center",ha="left",color="#49363B",fontsize=10,fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.22,right=.96,top=.82,bottom=.22)
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
    st.markdown('<span class="side-chip">HEART &amp; LIFE · HEALTH INTELLIGENCE</span>', unsafe_allow_html=True)
    st.markdown("---")
    render_logout_control()
    st.markdown("**Your workspace**")
    st.caption("Explore a sample, run a model prediction, stress-test scenarios and inspect model performance.")
    st.markdown("**Under the hood**")
    st.caption("Python · Streamlit · Pandas · NumPy")
    st.caption("Scikit-learn · Joblib · Matplotlib charts")
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
        <span class="pill"><span class="live-dot"></span> HEART &amp; LIFE · HEART HEALTH LEARNING</span>
        <span class="pill">13 feature signals</span>
        <span class="pill">Cardiovascular analytics</span>
      </div>
      <h1>HEART &amp; LIFE.<br><span>Understand. Care. Live well.</span></h1>
      <p>CardioPredict brings heart health and the science of living well into one learning space.
      Explore cardiovascular measurements, understand model patterns and discover how everyday heart-health knowledge supports informed conversations with healthcare professionals.</p>
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
    ("SELECTED MODEL", selected_model, "Chosen during model training", "#743F4A"),
    ("TEST ACCURACY", pct(selected_metrics.get("accuracy")), "Held-out test subset", "#BD5969"),
    ("PRECISION · GROUP 1", pct(selected_metrics.get("precision")), "Positive-class precision", "#D77983"),
    ("RECALL · GROUP 1", pct(selected_metrics.get("recall")), "Positive-class recall", "#367B63"),
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
live_tab, lab_tab, analytics_tab, guide_tab, emergency_tab, assistant_tab = st.tabs(
    ["✦ Live explorer", "▦ Scenario lab", "♡ Heart analytics", "♡ Heart guide", "✚ Emergency & map", "✧ AI voice assistant"]
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
        fig_batch = make_scenario_chart(group0, group1)
        st.pyplot(fig_batch, use_container_width=True)
        plt.close(fig_batch)
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
        st.markdown("#### Model performance comparison")
        fig = make_performance_chart(comparison)
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
                fig_cm = make_confusion_matrix_chart(matrix_array, chosen)
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

# ----------------------------- AI ASSISTANT ------------------------------
with assistant_tab:
    st.markdown('<div class="section-kicker">06 / CardioPredict assistant</div>', unsafe_allow_html=True)
    st.header("Ask CardioPredict")
    st.markdown('<div class="section-sub">Ask by typing or speaking. The assistant explains all 13 model inputs, the Cleveland model and the limits of its output.</div>', unsafe_allow_html=True)
    current_profile = (st.session_state.get("last_prediction") or {}).get("profile") or {key: st.session_state.get("field_" + key, DEFAULTS[key]) for key in FEATURES}
    render_cardio_assistant(current_profile=current_profile, prediction=st.session_state.get("last_prediction"))
# ----------------------------- FOOTER -----------------------------------
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">CARDIOPREDICT · HEART & LIFE · HEART HEALTH LEARNING<br>'
    'Synthetic scenarios are fictional software-test inputs · Prototype not clinically validated · Not for diagnosis or emergency triage</div>',
    unsafe_allow_html=True,
)
