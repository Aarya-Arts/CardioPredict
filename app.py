import json
import os
from urllib.parse import quote
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from src.assistant import render_cardio_assistant
from src.auth import render_auth_gate, render_logout_control

BRAND_MARK_PATH = Path(__file__).resolve().parent / "assets" / "cardiopredict-mark.svg"
try:
    BRAND_MARK_SVG = BRAND_MARK_PATH.read_text(encoding="utf-8")
except OSError:
    BRAND_MARK_SVG = ""

# CardioPredict - product-style interface and synthetic scenario explorer.
# The active prediction form uses the 12-feature Mendeley candidate model.
# The original Cleveland model and metrics remain preserved as backup artifacts.
# Synthetic profiles are fictional software tests, never training records.

st.set_page_config(
    page_title="CardioPredict | Heart & Life",
    page_icon=BRAND_MARK_SVG or "💗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "models/cvd_model_new_dataset_candidate.joblib"
METRICS_PATH = "models/cvd_metrics_new_dataset_candidate.json"
MODEL_SCHEMA_ID = "mendeley-cvd-12-v1"
FEATURES = [
    "age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak",
    "noofmajorvessels", "gender", "chestpain", "fastingbloodsugar",
    "restingrelectro", "exerciseangia", "slope",
]
DEFAULTS = {
    "age": 49, "restingBP": 147, "serumcholestrol": 318,
    "maxheartrate": 146, "oldpeak": 2.4, "noofmajorvessels": 1,
    "gender": 1, "chestpain": 1, "fastingbloodsugar": 0,
    "restingrelectro": 1, "exerciseangia": 0, "slope": 2,
}
if st.session_state.get("_cardiopredict_model_schema") != MODEL_SCHEMA_ID:
    for key, value in DEFAULTS.items():
        st.session_state["field_" + key] = value
    st.session_state["last_prediction"] = None
    st.session_state["synthetic_batch"] = None
    st.session_state["_cardiopredict_model_schema"] = MODEL_SCHEMA_ID
else:
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
    
    /* HEART & LIFE — second approved blush/rose visual direction */
    :root {
      --ink:#172B42; --body:#536477; --muted:#748394; --navy:#142F49;
      --rose:#C91446; --rose-dark:#9E173C; --teal:#168C91;
      --line:#F0D8E0; --paper:#FFF8FA; --white:#FFFFFF;
      --shadow:0 16px 38px rgba(117,34,65,.09);
    }
    .stApp {background:linear-gradient(135deg,#FFF8FA 0%,#FCECF1 45%,#FFF9FB 100%)!important}
    [data-testid="stHeader"] {background:rgba(255,248,250,.94)!important;border-bottom:1px solid #F1D9E1!important}
    section[data-testid="stSidebar"] {background:linear-gradient(180deg,#FFF4F7,#F9E8EF)!important;border-right:1px solid #F0D8E0!important}
    .brand-bar {display:flex;align-items:center;justify-content:space-between;gap:1.1rem;
      padding:.75rem 1.1rem;margin:.15rem 0 1.25rem;background:rgba(255,255,255,.88);
      border:1px solid #F1D5DF;border-radius:18px;box-shadow:0 10px 28px rgba(117,34,65,.065)}
    .brand-lockup {display:flex;align-items:center;gap:.7rem;min-width:230px}
    .brand-lockup svg {width:58px;height:58px;flex:0 0 58px;filter:drop-shadow(0 5px 10px rgba(201,20,70,.16))}
    .brand-name {font:800 1.48rem 'Manrope',sans-serif;letter-spacing:-.055em;color:#172B42;line-height:1.05}
    .brand-name span {color:#C91446}
    .brand-subtitle {margin-top:5px;font-size:.69rem;letter-spacing:.32em;font-weight:900;color:#27384D}
    .brand-nav {display:flex;align-items:center;justify-content:center;gap:clamp(.8rem,2vw,2rem);flex-wrap:wrap}
    .brand-nav a {text-decoration:none!important;color:#24374B!important;font-weight:750;font-size:.9rem;padding:.48rem .2rem;border-bottom:2px solid transparent}
    .brand-nav a.active,.brand-nav a:hover {color:#C91446!important;border-color:#C91446}
    .brand-status {font-size:.72rem;letter-spacing:.1em;font-weight:900;color:#9E173C;background:#FBE6ED;border:1px solid #F1C9D6;border-radius:999px;padding:.55rem .8rem;white-space:nowrap}
    .sidebar-brand {display:flex;align-items:center;gap:.65rem;margin-bottom:.25rem}
    .sidebar-brand svg {width:44px;height:44px}
    .sidebar-brand .brand-copy {font:800 1.12rem 'Manrope',sans-serif;color:#172B42;letter-spacing:-.04em}
    .sidebar-brand .brand-copy span {color:#C91446}
    .sidebar-brand small {display:block;font:800 .57rem 'DM Sans',sans-serif;letter-spacing:.25em;color:#6C7887;margin-top:3px}
    .hero {display:grid;grid-template-columns:minmax(0,1.08fr) minmax(280px,.92fr);align-items:center;gap:1.2rem;
      min-height:365px;padding:2.1rem 2.2rem;margin:.25rem 0 1.25rem;
      background:linear-gradient(112deg,#FFF9FB 0%,#FBE5EC 53%,#F5B6C8 100%)!important;
      border:1px solid #F0C8D5;border-radius:25px;box-shadow:0 20px 46px rgba(148,27,65,.12);overflow:hidden}
    .hero:before,.hero:after {display:none!important}
    .hero-copy {position:relative;z-index:1;min-width:0}
    .hero-top {margin-bottom:1.15rem}
    .pill {background:#FFFFFF!important;color:#8F173A!important;border:1px solid #F1C4D2!important;
      box-shadow:0 4px 12px rgba(151,25,64,.055);font-size:.72rem}
    .live-dot {background:#D7194C!important;box-shadow:0 0 0 3px rgba(215,25,76,.12)!important}
    .hero h1 {color:#142F49!important;font-size:clamp(2.35rem,4.8vw,3.65rem)!important;max-width:700px}
    .hero h1 span {color:#C91446!important;-webkit-text-fill-color:#C91446!important}
    .hero .hero-tagline {font:700 clamp(1.08rem,2vw,1.42rem) 'Manrope',sans-serif!important;color:#172B42!important;margin:.75rem 0 .85rem!important;line-height:1.35!important}
    .hero .hero-tagline span {color:#C91446!important}
    .hero p {max-width:620px;color:#43566A!important;font-size:1rem!important;line-height:1.65!important}
    .hero-actions {display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;margin:1.25rem 0 .25rem}
    .hero-cta {display:inline-flex;align-items:center;gap:.55rem;padding:.78rem 1.05rem;border-radius:999px;
      text-decoration:none!important;font-weight:850;font-size:.91rem;border:1px solid #C91446}
    .hero-cta.primary {background:linear-gradient(100deg,#F04768,#C91446);color:#FFFFFF!important;
      box-shadow:0 8px 20px rgba(201,20,70,.2)}
    .hero-cta.secondary {background:rgba(255,255,255,.62);color:#8F173A!important;border-color:#C75A78}
    .hero-foot {margin-top:1.05rem;gap:.55rem}
    .hero-stat {background:rgba(255,255,255,.76);border:1px solid #F0C8D5;color:#35485B!important;
      padding:8px 11px;border-radius:11px;font-size:.78rem}
    .hero-stat b {color:#9E173C!important}
    .hero-art {min-width:0;display:flex;align-items:center;justify-content:center;position:relative}
    .hero-art svg {width:100%;max-width:520px;height:auto;overflow:visible;filter:drop-shadow(0 14px 22px rgba(180,20,70,.15))}
    .feature-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:0 0 1.25rem}
    .feature-card {display:flex;align-items:center;gap:.75rem;background:rgba(255,255,255,.9);
      border:1px solid #F0D8E0;border-radius:17px;padding:1rem 1.05rem;box-shadow:0 8px 23px rgba(117,34,65,.055)}
    .feature-icon {width:45px;height:45px;display:grid;place-items:center;flex:0 0 45px;border-radius:50%;
      background:#FCE5EC;color:#C91446;font-size:1.35rem}
    .feature-card strong {display:block;color:#172B42!important;font:800 .9rem 'Manrope',sans-serif}
    .feature-card small {display:block;color:#69788A;font-size:.76rem;line-height:1.4;margin-top:3px}
    .glass-card {background:linear-gradient(145deg,#FFFFFF,#FFF7FA)!important;border:1px solid #F0D8E0!important;
      box-shadow:0 10px 25px rgba(117,34,65,.065)}
    .stat-value {color:#9E173C!important}
    .glow-line {background:linear-gradient(90deg,#C91446,#F49DB4,#D8A5B9,transparent)!important}
    .stTabs [data-baseweb="tab-list"] {background:#FBEAF0!important;border-color:#F0D3DE!important}
    .stTabs [aria-selected="true"] {background:#FFFFFF!important;color:#A31540!important;border-color:#F0C8D5!important}
    .stTabs [data-baseweb="tab-highlight"] {background:#C91446!important}
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button,.stLinkButton>a {
      background:linear-gradient(100deg,#F04768,#C91446)!important;border-color:#C91446!important;
      box-shadow:0 7px 18px rgba(201,20,70,.16)!important}
    .auth-brand-lockup {display:flex;align-items:center;gap:1rem}
    .auth-brand-lockup svg {width:72px;height:72px;flex:0 0 72px;filter:drop-shadow(0 6px 12px rgba(201,20,70,.18))}
    .auth-brand-subtitle {font-size:.63rem;letter-spacing:.24em;font-weight:900;color:#536477;margin-top:5px}
    .auth-brand {background:linear-gradient(145deg,#FFF1F6 0%,#F8D8E3 63%,#F2B4C7 100%)!important;
      border-right:1px solid #F0D0DA!important;color:#172B42!important}
    .auth-brand h1 {color:#142F49!important}
    .auth-brand p {color:#43566A!important}
    .auth-brand .eyebrow {color:#A31540!important}
    .auth-brand .trust {color:#536477!important}
    .auth-mark {color:#172B42!important}
    .auth-mark span {color:#C91446!important}
    .auth-panel {background:#FFFFFF!important}
    @media(max-width:900px) {
      .brand-bar {flex-wrap:wrap}
      .brand-nav {order:3;width:100%;justify-content:space-around}
      .hero {grid-template-columns:1fr;gap:.4rem}
      .hero-art {max-height:270px}
      .hero-art svg {max-height:260px}
      .feature-grid {grid-template-columns:repeat(2,minmax(0,1fr))}
    }
    @media(max-width:560px) {
      .brand-lockup {min-width:0}
      .brand-lockup svg {width:47px;height:47px;flex-basis:47px}
      .brand-name {font-size:1.2rem}
      .brand-status {font-size:.62rem;padding:.45rem .6rem}
      .brand-nav {gap:.45rem}
      .brand-nav a {font-size:.78rem}
      .hero {padding:1.35rem;border-radius:19px}
      .hero h1 {font-size:2.15rem!important}
      .hero-art {max-height:215px}
      .feature-grid {grid-template-columns:1fr}
    }

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
    """Generate a fictional profile inside the new dataset's documented input ranges."""
    return {
        "age": int(rng.integers(20, 81)),
        "restingBP": int(np.clip(rng.normal(152, 30), 94, 200)),
        "serumcholestrol": int(np.clip(rng.normal(311, 132), 126, 564)),
        "maxheartrate": int(np.clip(rng.normal(145, 34), 71, 202)),
        "oldpeak": round(float(np.clip(rng.normal(2.7, 1.7), 0, 6.2)), 1),
        "noofmajorvessels": int(rng.choice([0, 1, 2, 3], p=[.34, .42, .18, .06])),
        "gender": int(rng.choice([0, 1], p=[.235, .765])),
        "chestpain": int(rng.choice([0, 1, 2, 3], p=[.42, .224, .312, .044])),
        "fastingbloodsugar": int(rng.choice([0, 1], p=[.704, .296])),
        "restingrelectro": int(rng.choice([0, 1, 2], p=[.454, .344, .202])),
        "exerciseangia": int(rng.integers(0, 2)),
        "slope": int(rng.choice([1, 2, 3], p=[.365, .393, .242])),
    }


def make_performance_chart(frame):
    """Compare held-out model metrics with aligned, readable horizontal bars."""
    metrics=list(frame.columns); models=list(frame.index)
    fig,ax=plt.subplots(figsize=(10.5,5.4),facecolor="#FFFFFF"); ax.set_facecolor("#FFFFFF")
    palette={"Logistic Regression":"#C91446","Random Forest":"#F05C7B"}; fallback=["#C91446","#F05C7B","#567C91"]
    y=np.arange(len(metrics)); height=.34
    for i,name in enumerate(models):
        values=frame.loc[name,metrics].to_numpy(dtype=float)
        bars=ax.barh(y+(i-(len(models)-1)/2)*height,values,height=height*.88,
                     color=palette.get(str(name),fallback[i%len(fallback)]),edgecolor="#FFFFFF",
                     linewidth=1.4,label=str(name),zorder=3)
        for bar,value in zip(bars,values):
            ax.text(min(value+.014,1.045),bar.get_y()+bar.get_height()/2,f"{value:.1%}",
                    va="center",ha="left",color="#172B42",fontsize=10,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([str(m).replace("_"," ").title() for m in metrics],
                                          color="#172B42",fontsize=10,fontweight="medium")
    ax.invert_yaxis(); ax.set_xlim(0,1.14); ax.set_xticks(np.linspace(0,1,6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0,1,6)])
    ax.set_xlabel("Evaluation score",color="#536477",labelpad=10,fontsize=10)
    ax.set_title("SELECTED MODEL · HELD-OUT TEST METRICS",color="#8F173A",pad=18,fontsize=14,fontweight="bold")
    ax.xaxis.grid(True,color="#F2CED9",linewidth=.9,zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x",colors="#748394",labelsize=9,length=0,pad=7); ax.tick_params(axis="y",length=0,pad=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.02),ncol=max(1,min(len(models),2)),
              frameon=False,labelcolor="#172B42",fontsize=9)
    fig.subplots_adjust(left=.23,right=.96,top=.82,bottom=.15)
    return fig


def make_confusion_matrix_chart(cm_array,model_name):
    """Display confusion-matrix counts with a soft rose colour scale and labels."""
    from matplotlib.colors import LinearSegmentedColormap
    values=np.asarray(cm_array,dtype=int)
    fig,ax=plt.subplots(figsize=(7.4,5.3),facecolor="#FFFFFF"); ax.set_facecolor("#FFFFFF")
    cmap=LinearSegmentedColormap.from_list("cardiopredict_rose",["#E8F4F7","#B9E1E3","#55B8BC","#C91446"])
    vmax=max(int(values.max()),1); ax.imshow(values,cmap=cmap,vmin=0,vmax=vmax,aspect="equal")
    ax.set_xticks([0,1],labels=["Group 0\n(absence-labelled)","Group 1\n(presence-labelled)"]); ax.set_yticks([0,1],labels=["Group 0\n(absence-labelled)","Group 1\n(presence-labelled)"])
    ax.set_xlabel("Predicted dataset group",color="#536477",labelpad=10,fontsize=10)
    ax.set_ylabel("Actual dataset group",color="#536477",labelpad=10,fontsize=10)
    ax.set_title(f"{model_name.upper()} · CONFUSION MATRIX",color="#8F173A",pad=18,fontsize=13,fontweight="bold")
    ax.tick_params(axis="both",colors="#172B42",labelsize=10,length=0,pad=9)
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
    bars=ax.barh(labels,counts,color=["#367B63","#F05C7B"],height=.56,
                 edgecolor="#FFFFFF",linewidth=1.4,zorder=3)
    maximum=max(max(counts),1); ax.set_xlim(0,maximum*1.18)
    ax.xaxis.grid(True,color="#F2CED9",linewidth=.8,zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x",colors="#748394",labelsize=9,length=0)
    ax.tick_params(axis="y",colors="#172B42",labelsize=10,length=0,pad=9)
    ax.set_xlabel("Number of fictional test profiles",color="#F1E5E8",fontsize=9,labelpad=8)
    ax.set_title("SYNTHETIC OUTPUT DISTRIBUTION",color="#8F173A",fontsize=12,fontweight="bold",pad=13)
    for bar,count in zip(bars,counts):
        ax.text(count+maximum*.025,bar.get_y()+bar.get_height()/2,f"{count:,}",
                va="center",ha="left",color="#172B42",fontsize=10,fontweight="bold")
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
selected_metrics = metrics_data.get("held_out_test_metrics") or results.get(selected_model, {})
saved_features = metrics_data.get("features")
if saved_features and list(saved_features) != FEATURES:
    st.error("The selected model's saved feature schema does not match the prediction form. Check the candidate model and metrics files.")
    st.stop()
if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
    st.error("The loaded model expects a different number of input fields. Prediction has been stopped to prevent a schema mismatch.")
    st.stop()

# ----------------------------- SIDEBAR ----------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">{BRAND_MARK_SVG}<div class="brand-copy">Cardio<span>Predict</span><small>HEART &amp; LIFE</small></div></div>',
        unsafe_allow_html=True,
    )
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
    st.caption("12 model input features → saved preprocessing pipeline → classifier → binary output")
    st.markdown("---")
    st.caption("Prototype notice: not clinically validated. Never use this app to diagnose, screen or decide whether to seek emergency care.")

st.markdown(
    f"""
    <header class="brand-bar" id="top-nav">
      <div class="brand-lockup">
        {BRAND_MARK_SVG}
        <div><div class="brand-name">Cardio<span>Predict</span></div><div class="brand-subtitle">H E A R T &nbsp; &amp; &nbsp; L I F E</div></div>
      </div>
      <div class="brand-status">HEART HEALTH LEARNING</div>
    </header>
    """,
    unsafe_allow_html=True,
)

def set_active_page(page):
    st.session_state["active_page"] = page

NAVIGATION_PAGES = [
    "⌂ Home",
    "♡ Predict",
    "▦ Scenario Lab",
    "▥ Analytics",
    "♥ Heart Guide",
    "✚ Emergency & Map",
    "✧ AI Guide",
    "ⓘ About",
]
st.session_state.setdefault("active_page", NAVIGATION_PAGES[0])

st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > div[role="radiogroup"] {
      display:flex; gap:.35rem; flex-wrap:wrap; align-items:center;
    }
    div[data-testid="stRadio"] label {
      background:#FFFFFF; border:1px solid #E5EAF0; border-radius:999px;
      padding:.38rem .78rem; margin:0; transition:all .15s ease;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
      background:#FBEAF0; border-color:#C91446;
    }
    div[data-testid="stRadio"] label p {
      color:#24374B; font-weight:750; margin:0;
    }
    div[data-testid="stRadio"] label:has(input:checked) p { color:#9E4058; }
    </style>
    """,
    unsafe_allow_html=True,
)
active_page = st.radio(
    "Main navigation",
    options=NAVIGATION_PAGES,
    horizontal=True,
    key="active_page",
    label_visibility="collapsed",
)

# ----------------------------- HERO -------------------------------------
if active_page == NAVIGATION_PAGES[0]:
    st.markdown(
        f"""
        <section class="hero" id="top">
          <div class="hero-copy">
            <div class="hero-top">
              <span class="pill"><span class="live-dot"></span> HEART &amp; LIFE · HEALTH INTELLIGENCE</span>
              <span class="pill">12 model features</span>
            </div>
            <h1>Cardio<span>Predict</span><br>HEART &amp; LIFE</h1>
            <div class="hero-tagline">Know Today. Live <span>Healthier</span> Tomorrow.</div>
            <p>Explore cardiovascular measurements, understand how machine-learning models work,
            and use the AI guide to learn more about the science behind heart-health data.</p>
            <div class="hero-foot">
              <div class="hero-stat"><b>Model</b> · {selected_model}</div>
              <div class="hero-stat"><b>Dataset</b> · Mendeley CVD</div>
              <div class="hero-stat"><b>Workspace</b> · Explore + learn</div>
            </div>
          </div>
          <div class="hero-art" aria-hidden="true">
            <svg viewBox="0 0 560 430" role="img" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <radialGradient id="heartAura"><stop stop-color="#FF7898" stop-opacity=".55"/><stop offset="1" stop-color="#F5B8CA" stop-opacity="0"/></radialGradient>
                <linearGradient id="heartFill" x1="130" y1="40" x2="430" y2="390" gradientUnits="userSpaceOnUse"><stop stop-color="#FF7995"/><stop offset=".45" stop-color="#D9194D"/><stop offset="1" stop-color="#850D35"/></linearGradient>
                <linearGradient id="heartLight" x1="180" y1="100" x2="360" y2="350" gradientUnits="userSpaceOnUse"><stop stop-color="#FFE2EA"/><stop offset=".55" stop-color="#F34B73"/><stop offset="1" stop-color="#B30D3E"/></linearGradient>
                <filter id="heartGlow"><feGaussianBlur stdDeviation="9" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              </defs>
              <circle cx="300" cy="214" r="196" fill="url(#heartAura)"/>
              <ellipse cx="286" cy="224" rx="240" ry="102" transform="rotate(-22 286 224)" fill="none" stroke="#FFFFFF" stroke-width="5" opacity=".88"/>
              <ellipse cx="286" cy="224" rx="240" ry="102" transform="rotate(-22 286 224)" fill="none" stroke="#E34D78" stroke-width="2" opacity=".7"/>
              <path d="M279 358C245 326 130 246 128 162c-2-70 73-108 127-47 44-67 133-47 139 21 8 87-72 164-115 222Z" fill="url(#heartFill)" stroke="#FFF0F4" stroke-width="5" filter="url(#heartGlow)"/>
              <path d="M279 328c-41-39-94-90-95-151-1-43 41-64 73-26l21 28 23-31c29-39 76-14 73 26-4 55-57 114-95 154Z" fill="#8D113B" opacity=".45"/>
              <path d="M252 105 248 65c-2-20 8-31 23-31 14 0 21 13 19 30l-5 41M309 109l15-47c6-18 20-25 33-18 12 8 12 22 3 39l-27 47M356 129l30-37c11-13 27-15 37-4 9 11 5 25-9 37l-48 43" fill="none" stroke="#C91446" stroke-width="22" stroke-linecap="round"/>
              <path d="M238 220c29-35 47-46 69-42 30 5 42 34 31 61-10 26-35 58-57 78-25-27-54-66-56-91-2-11 3-19 13-28Z" fill="url(#heartLight)" opacity=".84"/>
              <path d="M149 232h87l18-1 18-42 25 83 25-123 23 83 17 1h96" fill="none" stroke="#FFFFFF" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" filter="url(#heartGlow)"/>
              <path d="M65 361c84 39 226 34 340-30 45-25 78-61 91-95" fill="none" stroke="#D83162" stroke-width="5" stroke-linecap="round" opacity=".7"/>
              <circle cx="443" cy="83" r="7" fill="#FFFFFF"/><circle cx="96" cy="138" r="5" fill="#E83C69"/><circle cx="457" cy="286" r="5" fill="#FFFFFF"/>
              <path d="M455 43h34M472 26v34" stroke="#D7194C" stroke-width="10" stroke-linecap="round"/>
            </svg>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    hero_actions = st.columns([1.25, 1.25, 5])
    with hero_actions[0]:
        st.button("♡ Start Prediction", key="hero_predict", on_click=set_active_page,
                  args=("♡ Predict",), use_container_width=True)
    with hero_actions[1]:
        st.button("✧ Ask AI Guide", key="hero_ai_guide", on_click=set_active_page,
                  args=("✧ AI Guide",), use_container_width=True)

    quick_columns = st.columns(4)
    quick_cards = [
        ("Machine Learning", "Explore the Mendeley cardiovascular dataset model", "♡ Predict", "Open prediction"),
        ("AI Health Guide", "Ask questions about heart-health data", "✧ AI Guide", "Ask the AI guide"),
        ("Model Analytics", "Review evaluation metrics and charts", "▥ Analytics", "View analytics"),
        ("Heart & Life", "Learn about cardiovascular measurements", "♥ Heart Guide", "Open heart guide"),
    ]
    for index, (column, card) in enumerate(zip(quick_columns, quick_cards)):
        title, description, destination, action_label = card
        with column:
            st.markdown(
                f'<div class="glass-card"><div class="stat-label">{title}</div>'
                f'<div class="stat-note">{description}</div></div>',
                unsafe_allow_html=True,
            )
            st.button(action_label + " →", key=f"home_quick_link_{index}",
                      on_click=set_active_page, args=(destination,), use_container_width=True)

    # ----------------------------- TOP METRICS ------------------------------
    metrics_cols = st.columns(4)
    cards = [
        ("SELECTED MODEL", selected_model, "Chosen during model training", "#8F173A"),
        ("TEST ACCURACY", pct(selected_metrics.get("accuracy")), "Internal held-out test split", "#C91446"),
        ("PRECISION · GROUP 1", pct(selected_metrics.get("precision")), "Positive-class precision", "#F05C7B"),
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



# ----------------------------- LIVE EXPLORER ----------------------------
def render_prediction_page():
    st.markdown('<div id="live-explorer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">01 / Live explorer</div>', unsafe_allow_html=True)
    st.header("Run a profile through the model")
    st.markdown('<div class="section-sub">Enter values from a recorded report, or generate a fictional profile to explore the model interface.</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="info-box">Sample profiles are fictional software-test inputs. They are not real patient records, clinical reference cases or medical advice.</div>', unsafe_allow_html=True)

    with st.form("live_prediction_form", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### ◉ Personal and resting measurements")
            age = st.number_input("Age · years", min_value=20, max_value=80, key="field_age")
            gender = st.selectbox(
                "Gender · dataset coding", [0, 1], key="field_gender",
                format_func=lambda x: "Female · 0" if x == 0 else "Male · 1",
            )
            chestpain = st.selectbox(
                "Chest pain category", [0, 1, 2, 3], key="field_chestpain",
                format_func=lambda x: {
                    0: "0 · Typical angina",
                    1: "1 · Atypical angina",
                    2: "2 · Non-anginal pain",
                    3: "3 · Asymptomatic",
                }[x],
                help="These are historical dataset codes, not a symptom self-assessment.",
            )
            restingBP = st.number_input(
                "Resting blood pressure · mm Hg", min_value=94, max_value=200,
                key="field_restingBP",
            )
            serumcholestrol = st.number_input(
                "Serum cholesterol · mg/dL", min_value=126, max_value=564,
                key="field_serumcholestrol",
                help="Range shown in the dataset description. Use a recorded laboratory result, not an estimate.",
            )
            fastingbloodsugar = st.selectbox(
                "Fasting blood sugar above 120 mg/dL?", [0, 1],
                key="field_fastingbloodsugar",
                format_func=lambda x: "No · 0" if x == 0 else "Yes · 1",
            )
        with right:
            st.markdown("#### ◈ ECG and exercise-test measurements")
            restingrelectro = st.selectbox(
                "Resting ECG category", [0, 1, 2], key="field_restingrelectro",
                format_func=lambda x: {
                    0: "0 · Normal",
                    1: "1 · ST-T wave abnormality",
                    2: "2 · Possible/definite LVH",
                }[x],
            )
            maxheartrate = st.number_input(
                "Maximum heart rate achieved · bpm", min_value=71, max_value=202,
                key="field_maxheartrate",
            )
            exerciseangia = st.selectbox(
                "Exercise-induced angina?", [0, 1], key="field_exerciseangia",
                format_func=lambda x: "No · 0" if x == 0 else "Yes · 1",
            )
            oldpeak = st.number_input(
                "ST depression · oldpeak", min_value=0.0, max_value=6.2,
                step=0.1, key="field_oldpeak",
            )
            slope = st.selectbox(
                "Peak exercise ST-segment slope", [1, 2, 3], key="field_slope",
                format_func=lambda x: {
                    1: "1 · Upsloping", 2: "2 · Flat", 3: "3 · Downsloping"
                }[x],
            )
            noofmajorvessels = st.selectbox(
                "Number of major vessels", [0, 1, 2, 3],
                key="field_noofmajorvessels",
            )
            st.markdown('<div class="small-muted">Several fields come from ECG or exercise testing and cannot safely be guessed. Use recorded values only.</div>', unsafe_allow_html=True)

        run_prediction = st.form_submit_button("Run model analysis  →", use_container_width=True)

    if run_prediction:
        # Clear the previous output before attempting a new prediction so a failed
        # submission can never leave a stale result visible.
        st.session_state["last_prediction"] = None
        profile = {
            "age": age,
            "restingBP": restingBP,
            "serumcholestrol": serumcholestrol,
            "maxheartrate": maxheartrate,
            "oldpeak": oldpeak,
            "noofmajorvessels": noofmajorvessels,
            "gender": gender,
            "chestpain": chestpain,
            "fastingbloodsugar": fastingbloodsugar,
            "restingrelectro": restingrelectro,
            "exerciseangia": exerciseangia,
            "slope": slope,
        }
        input_df = pd.DataFrame([profile], columns=FEATURES)
        try:
            predicted = int(model.predict(input_df)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                idx = classes.index(1) if 1 in classes else min(1, len(probs) - 1)
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
                '<div class="result-card alert"><div class="result-tag">Dataset label · 1</div>'
                '<div class="result-title">Pattern aligned with the dataset disease-present group</div>'
                '<div class="small-muted">The model assigned this input to the group labelled disease-present in its historical training data. This does not establish that a person has cardiovascular disease.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card"><div class="result-tag">Dataset label · 0</div>'
                '<div class="result-title">Pattern aligned with the dataset no-disease group</div>'
                '<div class="small-muted">The model assigned this input to the group labelled no disease in its historical training data. This does not rule out disease or guarantee that a person is healthy.</div></div>',
                unsafe_allow_html=True,
            )
        prob = output["probability"]
        if prob is not None:
            p1, p2 = st.columns([1, 2])
            with p1:
                st.metric("Model output score · Group 1", pct(prob))
            with p2:
                st.progress(max(0.0, min(1.0, prob)))
                st.caption("This is the classifier's output probability, not a clinically calibrated personal-risk percentage or severity score.")
        with st.expander("View the 12 values sent to the model"):
            st.dataframe(pd.DataFrame([output["profile"]], columns=FEATURES), use_container_width=True, hide_index=True)
        st.markdown('<div class="notice"><b>Important:</b> This educational prototype is not clinically validated. Its output must not be used for diagnosis, screening, treatment decisions or emergency triage. Seek professional medical assessment for health concerns.</div>', unsafe_allow_html=True)


# ----------------------------- SCENARIO LAB -----------------------------
def render_scenario_lab():
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
def render_analytics_page():
    st.markdown('<div class="section-kicker">03 / Model analytics</div>', unsafe_allow_html=True)
    st.header("Explore the model's test performance")
    st.markdown('<div class="section-sub">These are internal experimental results from a held-out split of the uploaded dataset. They are not clinical validation results.</div>', unsafe_allow_html=True)

    test_metrics = metrics_data.get("held_out_test_metrics")
    cv_results = metrics_data.get("cv_model_comparison", {})
    if test_metrics:
        st.info(
            f"Selected candidate: {selected_model} · "
            f"{metrics_data.get('records_after_slope_validation', '—')} records after data checks · "
            f"{test_metrics.get('test_records', '—')} held-out test records."
        )
        metric_cols = st.columns(5)
        metric_items = [
            ("Accuracy", test_metrics.get("accuracy")),
            ("Precision", test_metrics.get("precision")),
            ("Recall", test_metrics.get("recall")),
            ("F1-score", test_metrics.get("f1_score")),
            ("ROC-AUC", test_metrics.get("roc_auc")),
        ]
        for col, (label, value) in zip(metric_cols, metric_items):
            with col:
                st.metric(label, pct(value))

        if cv_results:
            st.markdown("#### Candidate comparison · 5-fold CV on training partition")
            cv_rows = [
                {
                    "Algorithm": name,
                    "Mean ROC-AUC": safe_float(values.get("roc_auc_mean")),
                    "Standard deviation": safe_float(values.get("roc_auc_std")),
                }
                for name, values in cv_results.items()
            ]
            cv_frame = pd.DataFrame(cv_rows).set_index("Algorithm")
            st.dataframe(
                cv_frame.style.format({
                    "Mean ROC-AUC": "{:.3f}",
                    "Standard deviation": "{:.3f}",
                }),
                use_container_width=True,
            )

        st.markdown("#### Selected model · held-out test metrics")
        test_frame = pd.DataFrame(
            [{
                "Accuracy": safe_float(test_metrics.get("accuracy")),
                "Precision": safe_float(test_metrics.get("precision")),
                "Recall": safe_float(test_metrics.get("recall")),
                "F1-score": safe_float(test_metrics.get("f1_score")),
                "ROC-AUC": safe_float(test_metrics.get("roc_auc")),
            }],
            index=[selected_model],
        )
        fig = make_performance_chart(test_frame)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        matrix = test_metrics.get("confusion_matrix")
        if matrix and len(matrix) == 2:
            st.markdown("#### Confusion matrix · held-out test split")
            matrix_array = np.asarray(matrix, dtype=int)
            a, b = st.columns([.9, 1.1])
            with a:
                st.dataframe(
                    pd.DataFrame(
                        matrix_array,
                        index=["Actual group 0 · absence-labelled", "Actual group 1 · presence-labelled"],
                        columns=["Predicted group 0 · absence-labelled", "Predicted group 1 · presence-labelled"],
                    ),
                    use_container_width=True,
                )
                st.caption("Group 0 = absence-labelled in this dataset; Group 1 = presence-labelled. Rows are actual labels and columns are model predictions. These are historical dataset labels, not clinical diagnoses.")
            with b:
                fig_cm = make_confusion_matrix_chart(matrix_array, selected_model)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)

        st.markdown(
            '<div class="notice"><b>Interpretation:</b> The dataset is small and comes from one hospital. '
            'The high internal scores require independent external validation and must not be presented as '
            'real-world or clinical accuracy. The model output is educational only.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Candidate evaluation metrics are missing or use an unexpected format. Check the candidate metrics JSON file.")


# ----------------------------- HEART GUIDE -------------------------------
def render_heart_guide():
    st.markdown('<div class="section-kicker">04 / Heart guide</div>', unsafe_allow_html=True)
    st.header("Cardiovascular measurements in everyday language")
    st.markdown('<div class="section-sub">A quick guide to the 12 input fields used by the Cardiovascular Disease Dataset candidate model.</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="glass-card"><div class="stat-label">01 · BLOOD PRESSURE</div><div class="stat-value">Pressure</div><p>Resting blood pressure is recorded in mm Hg. A single reading needs context and should be interpreted by a healthcare professional.</p></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="glass-card"><div class="stat-label">02 · CHOLESTEROL</div><div class="stat-value">Blood fats</div><p>Serum cholesterol is a laboratory measurement. A full lipid profile provides more context than one total-cholesterol value.</p></div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="glass-card"><div class="stat-label">03 · HEART RATE</div><div class="stat-value">Beats</div><p>Maximum heart rate is recorded during an exercise test. It is not a value to estimate without a test report.</p></div>', unsafe_allow_html=True)

    st.markdown("#### The 12 model inputs")
    descriptions = [
        ("Age", "Age in years."),
        ("Gender", "Dataset code: 0 = female, 1 = male."),
        ("Chest pain category", "0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic."),
        ("Resting blood pressure", "Recorded at rest, in mm Hg."),
        ("Serum cholesterol", "Laboratory total cholesterol, in mg/dL."),
        ("Fasting blood sugar", "0 = not above 120 mg/dL; 1 = above 120 mg/dL."),
        ("Resting ECG", "0 = normal; 1 = ST-T wave abnormality; 2 = possible/definite LVH."),
        ("Maximum heart rate", "Maximum rate achieved during the exercise test, in bpm."),
        ("Exercise-induced angina", "0 = no; 1 = yes, as recorded during the exercise test."),
        ("Oldpeak", "ST depression value recorded during the exercise test."),
        ("ST slope", "1 = upsloping; 2 = flat; 3 = downsloping."),
        ("Number of major vessels", "Coded count from 0 to 3."),
    ]
    left, right = st.columns(2)
    for i, (name, desc) in enumerate(descriptions):
        target = left if i % 2 == 0 else right
        with target:
            st.markdown(f"**{i+1:02d}. {name}**")
            st.caption(desc)
    st.markdown(
        '<div class="info-box">Several inputs require laboratory, ECG or exercise-test records. '
        'Do not guess these values. The numeric codes are dataset encodings, not a self-assessment guide.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Everyday habits that support heart health")
    st.markdown("- Avoid tobacco and smoking.\n- Stay physically active in a way that is appropriate for your health.\n- Choose a balanced diet and limit excess salt.\n- Attend recommended health checks.\n- Follow treatment prescribed by your clinician.\n- Seek medical advice for new, persistent or worrying symptoms.")


# ----------------------------- EMERGENCY & MAP ---------------------------
def render_emergency_page():
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
def render_ai_guide():
    st.markdown('<div id="ai-guide"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">06 / CardioPredict assistant</div>', unsafe_allow_html=True)
    st.header("Ask CardioPredict")
    st.markdown('<div class="section-sub">Ask by typing or speaking. The assistant explains the model inputs, dataset codes and limits of the educational prediction output.</div>', unsafe_allow_html=True)
    current_profile = (st.session_state.get("last_prediction") or {}).get("profile") or {key: st.session_state.get("field_" + key, DEFAULTS[key]) for key in FEATURES}
    render_cardio_assistant(current_profile=current_profile, prediction=st.session_state.get("last_prediction"))
def render_about_page():
    st.markdown('<div class="section-kicker">ABOUT / CARDIOPREDICT</div>', unsafe_allow_html=True)
    st.header("About CardioPredict – Heart & Life")
    st.markdown(
        '<div class="section-sub">A machine-learning mini-project that demonstrates how a cardiovascular dataset can be explored through an interactive educational interface.</div>',
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown('<div class="glass-card"><div class="stat-label">DATASET</div><div class="stat-value">Mendeley</div><div class="stat-note">Cardiovascular Disease Dataset · 2021</div></div>', unsafe_allow_html=True)
    with summary_cols[1]:
        st.markdown('<div class="glass-card"><div class="stat-label">MODEL INPUTS</div><div class="stat-value">12</div><div class="stat-note">Dataset predictor attributes</div></div>', unsafe_allow_html=True)
    with summary_cols[2]:
        st.markdown(f'<div class="glass-card"><div class="stat-label">SELECTED ALGORITHM</div><div class="stat-value">{selected_model}</div><div class="stat-note">Chosen using cross-validation ROC-AUC</div></div>', unsafe_allow_html=True)

    st.markdown("### Project objective")
    st.markdown(
        "CardioPredict provides a learning environment for exploring cardiovascular dataset attributes, "
        "observing model outputs, comparing classification metrics and understanding the limitations of "
        "machine-learning predictions. It is intended for academic demonstration and software experimentation."
    )

    st.markdown("### Dataset and machine-learning workflow")
    st.markdown(
        """
        1. Load the Cardiovascular Disease Dataset published on Mendeley Data by Doppala and Bhattacharyya (2021).
        2. Exclude the patient identifier from model features.
        3. Validate the documented ST-slope codes and exclude records with undocumented slope value 0.
        4. Treat zero serum-cholesterol values as missing and impute them within the training pipeline.
        5. Compare Logistic Regression, Random Forest and Gradient Boosting using 5-fold stratified cross-validation on the training partition.
        6. Evaluate the selected candidate on a separate stratified holdout test split.
        7. Display the dataset labels and evaluation metrics with explicit educational and non-clinical limitations.
        """
    )
    st.markdown(
        "Dataset source: Doppala, B. P., & Bhattacharyya (2021), "
        "[Cardiovascular_Disease_Dataset, Mendeley Data, Version 1](https://doi.org/10.17632/dzz48mvjht.1). "
        "Licensed under CC BY 4.0; attribution is provided for academic reuse."
    )

    st.markdown("### Scope and limitations")
    st.markdown(
        "The dataset contains 1,000 records described as originating from one multispecialty hospital. "
        "The reported metrics are internal experimental results from this dataset and do not establish "
        "performance on independent hospitals, populations or real-world clinical settings. The model is "
        "not clinically validated and must not be used for diagnosis, screening, treatment decisions or emergency triage."
    )
    st.info("Several fields require actual laboratory, ECG or exercise-test records. Do not guess measurements. For health concerns, consult a qualified healthcare professional.")



# ----------------------------- PAGE ROUTER -------------------------------
PAGE_RENDERERS = {
    "♡ Predict": render_prediction_page,
    "▦ Scenario Lab": render_scenario_lab,
    "▥ Analytics": render_analytics_page,
    "♥ Heart Guide": render_heart_guide,
    "✚ Emergency & Map": render_emergency_page,
    "✧ AI Guide": render_ai_guide,
    "ⓘ About": render_about_page,
}
if active_page in PAGE_RENDERERS:
    PAGE_RENDERERS[active_page]()


# ----------------------------- FOOTER -----------------------------------
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer" id="about-project">CARDIOPREDICT · HEART & LIFE · HEART HEALTH LEARNING<br>'
    'Synthetic scenarios are fictional software-test inputs · Prototype not clinically validated · Not for diagnosis or emergency triage</div>',
    unsafe_allow_html=True,
)
