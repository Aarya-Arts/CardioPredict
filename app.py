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
    .footer {color:#D8E2F5;font-size:.78rem;text-align:center;padding:1.4rem 0 .4rem;}
    /* High-contrast readability layer: keep the colourful background, make every word legible. */
    body, .stApp, [data-testid="stAppViewContainer"] {color:#FFFFFF!important;}
    .stApp p, .stApp li, .stApp label, .stApp span, .stApp small,
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    .section-sub, .small-muted, .stat-label, .stat-note, .footer {
      color:#E5ECFA!important;
    }
    h1,h2,h3,h4,h5,h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {color:#FFFFFF!important;text-shadow:0 1px 10px rgba(0,0,0,.18);}
    .section-kicker {color:#8AF5FF!important;}
    .hero p {color:#F1F7FF!important;}
    .hero-stat, .pill {color:#FFFFFF!important;}
    .glass-card p, .glass-card .stat-label, .glass-card .stat-note {color:#F0F4FF!important;}
    .info-box {color:#F0FCFF!important;background:rgba(12,77,100,.55)!important;border-color:#55DDF0!important;}
    .notice {color:#FFF2DC!important;background:rgba(104,63,14,.55)!important;border-color:#F5B95D!important;}
    .result-title, .result-tag, .result-card .small-muted {color:#FFFFFF!important;}
    .result-card {background:linear-gradient(120deg,rgba(0,105,125,.78),rgba(39,48,112,.88))!important;}
    .result-card.alert {background:linear-gradient(120deg,rgba(126,25,78,.88),rgba(66,34,111,.92))!important;}
    .emergency p {color:#FFFFFF!important;}
    .stTabs [data-baseweb="tab"] {color:#F0F4FF!important;font-weight:700!important;}
    .stTabs [aria-selected="true"] {color:#FFFFFF!important;}
    div[data-baseweb="input"] input, div[data-baseweb="select"] *,
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] {color:#FFFFFF!important;}
    [data-testid="stNumberInput"] button {color:#FFFFFF!important;background:#29365F!important;}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {color:#FFFFFF!important;}
    [data-testid="stDataFrame"] * {color:#F8FAFF!important;}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {color:#FFFFFF!important;}
    [data-testid="stAlert"] * {color:#FFFFFF!important;}
    ::placeholder {color:#C7D2E8!important;opacity:1!important;}
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

    /* Accessible warm-teal colour system and larger type */
    :root {
      --ink:#183342; --muted:#405968; --panel:#F7FAF9;
      --line:rgba(22,62,73,.16); --cyan:#267D80; --pink:#C96E63;
      --violet:#668A9A; --lime:#6F967A; --orange:#B8793F;
    }
    .stApp {
      color:#203744!important;
      background:linear-gradient(145deg,#EEF4F3 0%,#E5EFF0 52%,#F5F1E9 100%)!important;
    }
    [data-testid="stHeader"] {background:rgba(238,244,243,.94)!important;}
    .block-container {padding-top:1.35rem!important;}
    h1,h2,h3,h4,h5,h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
      color:#173B49!important;text-shadow:none!important;
    }
    .stApp p,.stApp li,.stApp label,.stApp span,.stApp small,
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,
    .section-sub,.small-muted,.stat-label,.stat-note,.footer {
      color:#344F5C!important;
    }
    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMarkdownContainer"] {
      font-size:18px!important;
    }
    .stApp p,.stApp li {font-size:1.08rem!important;line-height:1.65!important;}
    .stApp label,[data-testid="stWidgetLabel"] {font-size:1.04rem!important;font-weight:650!important;color:#203F4D!important;}
    [data-testid="stCaptionContainer"],.stCaption,.small-muted,.section-sub {font-size:1rem!important;line-height:1.55!important;}
    h1 {font-size:2.35rem!important;} h2 {font-size:1.9rem!important;} h3 {font-size:1.48rem!important;} h4 {font-size:1.25rem!important;}
    section[data-testid="stSidebar"] {background:#E3ECEB!important;border-right:1px solid #C6D7D7!important;}
    section[data-testid="stSidebar"] * {color:#203F4D!important;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#405C67!important;}
    .brand span,.section-kicker {color:#287D80!important;}
    .side-chip {background:#D5E9E6!important;color:#205E61!important;border-color:#A9CFCA!important;font-size:.82rem!important;}
    .hero {
      background:linear-gradient(112deg,#173F4C 0%,#28616A 55%,#557D76 100%)!important;
      border:1px solid rgba(255,255,255,.35)!important;
      box-shadow:0 18px 42px rgba(27,66,74,.18)!important;
    }
    .hero h1,.hero h1 span {color:#FFFFFF!important;background:none!important;-webkit-text-fill-color:#FFFFFF!important;}
    .hero p,.hero-stat,.pill {color:#FFFFFF!important;}
    .hero-stat b {color:#C6E7D8!important;}
    .pill {background:rgba(255,255,255,.14)!important;border-color:rgba(255,255,255,.28)!important;font-size:.85rem!important;}
    .live-dot {background:#B7D8B5!important;box-shadow:none!important;}
    .ecg path {stroke:#D4E9DD!important;filter:none!important;}
    .glass-card {
      background:linear-gradient(145deg,#FFFFFF,#F2F7F6)!important;
      border:1px solid #D3E1E0!important;box-shadow:0 8px 24px rgba(35,72,78,.09)!important;
    }
    .glass-card:hover {border-color:#8AB9B4!important;box-shadow:0 12px 28px rgba(35,72,78,.14)!important;}
    .glass-card p,.glass-card .stat-label,.glass-card .stat-note {color:#34525E!important;}
    .stat-value {color:#173F4C!important;font-size:1.9rem!important;}
    .stat-label {font-size:.88rem!important;letter-spacing:.06em!important;}
    .stat-note {font-size:.92rem!important;color:#45616B!important;}
    .glow-line {background:linear-gradient(90deg,#397F82,#82AFA4,#C78A70,transparent)!important;}
    div[data-testid="stForm"] {background:#F8FBFA!important;border-color:#D3E1E0!important;}
    div[data-baseweb="input"]>div,div[data-baseweb="select"]>div {
      background:#FFFFFF!important;border:1px solid #AFC4C5!important;color:#173B49!important;
    }
    input,textarea,div[data-baseweb="input"] input,div[data-baseweb="select"] *,
    [data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] {color:#173B49!important;font-size:1.08rem!important;}
    [data-baseweb="popover"],[data-baseweb="menu"] {background:#FFFFFF!important;}
    [role="option"] {color:#173B49!important;font-size:1.02rem!important;}
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button {
      color:#FFFFFF!important;background:linear-gradient(100deg,#286F73,#397F79 55%,#668C7A)!important;
      box-shadow:0 6px 16px rgba(37,103,103,.18)!important;font-size:1.02rem!important;min-height:3.2rem!important;
    }
    .stTabs [data-baseweb="tab-list"] {background:#DDE9E8!important;border-color:#C8D9D8!important;}
    .stTabs [data-baseweb="tab"] {color:#34535E!important;font-size:1rem!important;font-weight:750!important;}
    .stTabs [aria-selected="true"] {background:#FFFFFF!important;border:1px solid #A8C6C2!important;color:#1F6568!important;}
    [data-testid="stMetric"] {background:#F8FBFA!important;border-color:#D3E1E0!important;}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricDelta"] {color:#173F4C!important;}
    .result-card {background:linear-gradient(120deg,#DCEFEB,#E9F0F5)!important;border-color:#8CBDB2!important;}
    .result-card.alert {background:linear-gradient(120deg,#F7E8E2,#F1E9E7)!important;border-color:#D5A092!important;}
    .result-title,.result-tag,.result-card .small-muted {color:#173F4C!important;}
    .result-tag {color:#286F73!important;}
    .result-card.alert .result-tag {color:#A6534D!important;}
    .info-box {color:#214F58!important;background:#E1F0EF!important;border-color:#9CCBC5!important;font-size:1rem!important;}
    .notice {color:#654629!important;background:#F7EEDF!important;border-color:#DCC49A!important;font-size:1rem!important;}
    .emergency {background:linear-gradient(110deg,#8E4E4A,#A96055,#806477)!important;}
    .emergency h3,.emergency p {color:#FFFFFF!important;}
    .phone {color:#78413F!important;font-size:1rem!important;}
    [data-testid="stExpander"] {background:#F8FBFA!important;border:1px solid #D3E1E0!important;border-radius:12px!important;}
    [data-testid="stExpander"] summary,[data-testid="stExpander"] summary p {color:#173F4C!important;font-size:1rem!important;}
    [data-testid="stDataFrame"] {background:#FFFFFF!important;}
    [data-testid="stDataFrame"] * {color:#203F4D!important;}
    [data-testid="stAlert"] * {color:#203F4D!important;}
    ::placeholder {color:#647C85!important;opacity:1!important;}
    @media(max-width:760px) {
      .block-container {padding-left:1rem!important;padding-right:1rem!important;padding-top:.8rem!important;}
      .stApp,[data-testid="stMarkdownContainer"] {font-size:17px!important;}
      .stApp p,.stApp li {font-size:1.04rem!important;line-height:1.6!important;}
      h1 {font-size:2rem!important;} h2 {font-size:1.65rem!important;} h3 {font-size:1.32rem!important;}
      .hero {padding:1.45rem!important;border-radius:20px!important;}
      .hero h1 {font-size:2.15rem!important;}
      .stat-value {font-size:1.45rem!important;}
      .stTabs [data-baseweb="tab"] {padding:9px 10px!important;font-size:.88rem!important;}
      .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button {min-height:3.25rem!important;}
    }

    
    /* FINAL ACCESSIBILITY OVERRIDE — LIGHT SURFACES + DARK TEXT */
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], .stApp,
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
      background:#F3F7F6 !important;
      color:#203B46 !important;
    }
    [data-testid="stHeader"] {background:#F3F7F6 !important;}
    [data-testid="stMain"] {background:#F3F7F6 !important;}
    .stApp *:not(svg):not(path) {color:#203B46;}
    h1,h2,h3,h4,h5,h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
      color:#173B49 !important;text-shadow:none !important;
    }
    .stApp p,.stApp li,.stApp label,.stApp small,
    [data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
      color:#294653 !important;
    }
    /* Sidebar is a light navigation panel; it starts collapsed on phones. */
    section[data-testid="stSidebar"] {
      background:#E7F0EE !important;border-right:1px solid #C4D8D4 !important;
    }
    section[data-testid="stSidebar"] * {color:#203F4D !important;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#365662 !important;}
    /* Dark hero is intentional: every text element in it stays white. */
    .hero {
      background:linear-gradient(118deg,#174653 0%,#236B70 58%,#547F70 100%) !important;
      border:1px solid #A8C9C2 !important;
      box-shadow:0 16px 38px rgba(28,73,76,.16) !important;
    }
    .hero * {color:#FFFFFF !important;}
    .hero h1,.hero h1 span {
      color:#FFFFFF !important;background:none !important;
      -webkit-text-fill-color:#FFFFFF !important;
    }
    .hero p {color:#F5FAF8 !important;}
    .hero-stat {background:rgba(5,28,34,.36)!important;border-color:rgba(255,255,255,.3)!important;}
    .hero-stat b {color:#D6EEE2 !important;}
    .pill {background:rgba(255,255,255,.16)!important;border:1px solid rgba(255,255,255,.34)!important;color:#FFFFFF!important;}
    .live-dot {background:#C3E6B8!important;}
    .ecg path {stroke:#D9F0E5!important;filter:none!important;}
    /* Cards, forms, expanders and notices use opaque light surfaces. */
    .glass-card {
      background:#FFFFFF !important;border:1px solid #D0DFDC !important;
      box-shadow:0 7px 20px rgba(32,70,73,.08) !important;
    }
    .glass-card * {color:#294653 !important;}
    .glass-card .stat-value {color:#173F4C !important;}
    .stat-label {color:#3C5963 !important;font-size:.9rem!important;}
    .stat-value {color:#173F4C!important;font-size:1.9rem!important;}
    .stat-note {color:#45616B!important;font-size:.95rem!important;}
    div[data-testid="stForm"],[data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:#FFFFFF !important;border-color:#D2E0DE !important;
    }
    div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,
    [data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
      background:#FFFFFF !important;color:#183B48 !important;
      border:1px solid #AABFBE !important;
    }
    input,textarea,[data-baseweb="input"] input,
    [data-baseweb="select"] *,[data-testid="stWidgetLabel"] {
      color:#183B48 !important;
    }
    input::placeholder,textarea::placeholder,::placeholder {color:#61777D!important;opacity:1!important;}
    [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {
      background:#FFFFFF !important;border:1px solid #C6D6D3 !important;
    }
    [role="option"],[role="option"] * {background:#FFFFFF!important;color:#183B48!important;}
    [data-testid="stNumberInput"] button {
      background:#E4EFEC !important;color:#214D53 !important;border:1px solid #B8CECA!important;
    }
    /* Buttons always use white text on a solid dark teal fill. */
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button,
    [data-testid="stLinkButton"] a {
      background:#286D70 !important;color:#FFFFFF !important;
      border:1px solid #205D61 !important;border-radius:12px!important;
      min-height:3.15rem!important;font-size:1.04rem!important;font-weight:800!important;
      box-shadow:0 5px 14px rgba(34,91,92,.16)!important;
    }
    .stButton>button *, .stFormSubmitButton>button *,
    .stDownloadButton>button *, [data-testid="stLinkButton"] a * {
      color:#FFFFFF !important;
    }
    .stButton>button:hover,.stFormSubmitButton>button:hover,
    .stDownloadButton>button:hover {background:#1E5C60!important;}
    /* Tabs have dark labels on a light strip; active tab is distinctly shaded. */
    .stTabs [data-baseweb="tab-list"] {
      background:#E2ECEA!important;border:1px solid #C7D9D5!important;
      border-radius:14px!important;
    }
    .stTabs [data-baseweb="tab"],.stTabs [data-baseweb="tab"] * {
      color:#294B56!important;font-weight:750!important;
    }
    .stTabs [aria-selected="true"],.stTabs [aria-selected="true"] * {
      background:#FFFFFF!important;color:#165E62!important;
    }
    /* Metrics and tables remain readable even when Streamlit's base theme is dark. */
    [data-testid="stMetric"],[data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],[data-testid="stMetricDelta"] {
      background:#FFFFFF!important;color:#173F4C!important;
    }
    [data-testid="stMetric"] {border:1px solid #D1E0DD!important;border-radius:13px!important;}
    [data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"] * {color:#173F4C!important;}
    [data-testid="stDataFrame"] {
      background:#FFFFFF!important;border:1px solid #D1E0DD!important;border-radius:12px!important;
    }
    [data-testid="stDataFrame"] * {color:#203F4D!important;}
    [data-testid="stExpander"] summary,[data-testid="stExpander"] summary * {
      color:#203F4D!important;background:#FFFFFF!important;
    }
    [data-testid="stAlert"] {
      background:#EAF2F0!important;border:1px solid #BDD3CE!important;
    }
    [data-testid="stAlert"] * {color:#244651!important;}
    /* Informational/result panels: all use dark text on pale backgrounds. */
    .info-box,.info-box * {
      color:#214F58!important;background:#E2F0ED!important;
      border-color:#A9CEC5!important;
    }
    .notice,.notice * {
      color:#614A2C!important;background:#F7F0E3!important;
      border-color:#DCC9A7!important;
    }
    .result-card {
      background:#E4F1ED!important;border:1px solid #9FC8BC!important;
      box-shadow:0 8px 22px rgba(32,70,73,.08)!important;
    }
    .result-card * {color:#214550!important;}
    .result-card.alert {
      background:#F5EAE5!important;border-color:#D9B3A5!important;
    }
    .result-card.alert * {color:#67423D!important;}
    .result-title {font-size:1.55rem!important;font-weight:800!important;}
    .result-tag {font-size:.9rem!important;font-weight:900!important;}
    /* Emergency panel is the other intentional dark panel. */
    .emergency {
      background:linear-gradient(115deg,#784A4A,#985D55,#66566E)!important;
      border:1px solid #B9918C!important;
    }
    .emergency * {color:#FFFFFF!important;}
    .phone,.phone * {
      background:#FFFFFF!important;color:#673D3B!important;
      font-weight:900!important;
    }
    .section-kicker {color:#276E70!important;font-size:.85rem!important;}
    .section-sub,.small-muted,.footer {color:#405B64!important;}
    .footer {font-size:.9rem!important;}
    .glow-line {background:linear-gradient(90deg,#347C7A,#8BAE9F,#C58B70,transparent)!important;}
    /* Larger, consistent type for older and low-vision users. */
    .stApp,[data-testid="stMarkdownContainer"] {font-size:18px!important;}
    .stApp p,.stApp li {font-size:1.08rem!important;line-height:1.65!important;}
    .stApp label,[data-testid="stWidgetLabel"] {font-size:1.05rem!important;font-weight:700!important;}
    [data-testid="stCaptionContainer"],.stCaption,.small-muted,.section-sub {
      font-size:1rem!important;line-height:1.55!important;
    }
    h1 {font-size:2.35rem!important;} h2 {font-size:1.95rem!important;}
    h3 {font-size:1.5rem!important;} h4 {font-size:1.28rem!important;}
    @media(max-width:760px) {
      .block-container {padding-left:1rem!important;padding-right:1rem!important;padding-top:.8rem!important;}
      .hero {padding:1.45rem!important;border-radius:20px!important;}
      .hero h1 {font-size:2.1rem!important;}
      h1 {font-size:2.05rem!important;} h2 {font-size:1.72rem!important;}
      h3 {font-size:1.35rem!important;}
      .stApp p,.stApp li {font-size:1.04rem!important;line-height:1.6!important;}
      .stat-value {font-size:1.5rem!important;}
      .stTabs [data-baseweb="tab"] {padding:9px 10px!important;font-size:.9rem!important;}
    }

    
    /* CARDIOPREDICT HEART-HEALTH THEME — warm, vibrant and accessible */
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], .stApp {
      background:#FFF8F6 !important;
      color:#432D34 !important;
    }
    [data-testid="stHeader"] {background:rgba(255,248,246,.96)!important;}
    .block-container {max-width:1440px!important;padding-top:1.15rem!important;padding-bottom:3rem!important;}

    /* Dark text on all ordinary light surfaces */
    .stApp p,.stApp li,.stApp label,.stApp small,
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"] {
      color:#49353B!important;
    }
    h1,h2,h3,h4,h5,h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
      color:#542532!important;text-shadow:none!important;
      font-family:'DM Sans',sans-serif!important;letter-spacing:-.025em!important;
    }
    h1 {font-size:2.35rem!important;} h2 {font-size:1.95rem!important;}
    h3 {font-size:1.5rem!important;} h4 {font-size:1.28rem!important;}
    .stApp,[data-testid="stMarkdownContainer"] {font-size:18px!important;}
    .stApp p,.stApp li {font-size:1.08rem!important;line-height:1.65!important;}
    .stApp label,[data-testid="stWidgetLabel"] {font-size:1.05rem!important;font-weight:700!important;}
    [data-testid="stCaptionContainer"],.stCaption,.small-muted,.section-sub {
      font-size:1rem!important;line-height:1.55!important;
    }

    /* Light navigation panel */
    section[data-testid="stSidebar"] {
      background:#F9EAE9!important;border-right:1px solid #E9C9CB!important;
    }
    section[data-testid="stSidebar"] * {color:#542B35!important;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#684650!important;}
    .brand span {color:#B83250!important;}
    .side-chip {background:#F7DDE0!important;color:#8F2940!important;border:1px solid #E9B9C1!important;font-size:.82rem!important;}

    /* Hero is an intentional dark surface: every word within is white */
    .hero {
      background:linear-gradient(112deg,#6D1F36 0%,#A52D49 50%,#D65C68 100%)!important;
      border:1px solid rgba(255,255,255,.35)!important;
      box-shadow:0 20px 48px rgba(111,31,54,.2),inset 0 1px 0 rgba(255,255,255,.18)!important;
      border-radius:28px!important;
    }
    .hero * {color:#FFFFFF!important;}
    .hero h1,.hero h1 span {
      color:#FFFFFF!important;background:none!important;
      -webkit-text-fill-color:#FFFFFF!important;
    }
    .hero p {color:#FFF6F4!important;font-size:1.1rem!important;}
    .hero-stat {background:rgba(60,8,26,.26)!important;border-color:rgba(255,255,255,.28)!important;}
    .hero-stat b {color:#DDF0D7!important;}
    .pill {background:rgba(255,255,255,.15)!important;border:1px solid rgba(255,255,255,.35)!important;color:#FFFFFF!important;}
    .live-dot {background:#C7E7A2!important;box-shadow:0 0 10px rgba(199,231,162,.45)!important;}
    .ecg path {stroke:#FFE7E4!important;filter:drop-shadow(0 0 5px rgba(255,231,228,.4))!important;}

    /* White cards with burgundy text and gentle shadows */
    .glass-card {
      background:linear-gradient(145deg,#FFFFFF,#FFF2F0)!important;
      border:1px solid #EED6D4!important;
      box-shadow:0 9px 24px rgba(111,48,60,.08)!important;
    }
    .glass-card * {color:#49353B!important;}
    .glass-card .stat-value {color:#7D263E!important;}
    .glass-card:hover {border-color:#DCA8AC!important;box-shadow:0 13px 30px rgba(111,48,60,.13)!important;}
    .stat-label {color:#704550!important;font-size:.9rem!important;}
    .stat-value {color:#7D263E!important;font-size:1.9rem!important;}
    .stat-note {color:#557B5C!important;font-size:.95rem!important;}
    .section-kicker {color:#A8324C!important;font-size:.85rem!important;}
    .section-sub,.small-muted,.footer {color:#654A50!important;}
    .glow-line {background:linear-gradient(90deg,#B83250,#E77B74,#82A878,transparent)!important;}

    /* Forms and controls: white fields + dark readable values */
    div[data-testid="stForm"],[data-testid="stExpander"] {
      background:#FFFFFF!important;border:1px solid #E8D7D5!important;
      border-radius:18px!important;box-shadow:0 7px 18px rgba(111,48,60,.055)!important;
    }
    div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,
    [data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
      background:#FFFFFF!important;color:#432D34!important;
      border:1px solid #CDBABD!important;border-radius:10px!important;
    }
    input,textarea,[data-baseweb="input"] input,
    [data-baseweb="select"] *,[data-testid="stWidgetLabel"] {
      color:#432D34!important;font-size:1.06rem!important;
    }
    input::placeholder,textarea::placeholder,::placeholder {color:#806C70!important;opacity:1!important;}
    [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {
      background:#FFFFFF!important;border:1px solid #D8C7C8!important;
    }
    [role="option"],[role="option"] * {background:#FFFFFF!important;color:#432D34!important;}
    [data-testid="stNumberInput"] button {
      background:#F8E5E5!important;color:#812B43!important;border:1px solid #E2C5C6!important;
    }

    /* Primary controls: solid berry-red with white text */
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button,
    [data-testid="stLinkButton"] a {
      background:#A92F4A!important;color:#FFFFFF!important;
      border:1px solid #92253E!important;border-radius:12px!important;
      min-height:3.15rem!important;font-size:1.04rem!important;font-weight:800!important;
      box-shadow:0 6px 16px rgba(143,39,62,.16)!important;
    }
    .stButton>button *,.stFormSubmitButton>button *,
    .stDownloadButton>button *,[data-testid="stLinkButton"] a * {color:#FFFFFF!important;}
    .stButton>button:hover,.stFormSubmitButton>button:hover,
    .stDownloadButton>button:hover {background:#87233B!important;border-color:#87233B!important;}

    /* Tabs and interactive state */
    .stTabs [data-baseweb="tab-list"] {
      background:#F6E7E5!important;border:1px solid #E8D0D0!important;border-radius:15px!important;
    }
    .stTabs [data-baseweb="tab"],.stTabs [data-baseweb="tab"] * {
      color:#653C46!important;font-size:.98rem!important;font-weight:750!important;
    }
    .stTabs [aria-selected="true"],.stTabs [aria-selected="true"] * {
      background:#FFFFFF!important;color:#9D2B46!important;
    }

    /* Metric tiles, tables, expanders and Streamlit alerts */
    [data-testid="stMetric"] {background:#FFFFFF!important;border:1px solid #E8D7D5!important;border-radius:13px!important;}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricDelta"],
    [data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *,[data-testid="stMetricDelta"] * {
      color:#542532!important;
    }
    [data-testid="stDataFrame"] {background:#FFFFFF!important;border:1px solid #E8D7D5!important;border-radius:12px!important;}
    [data-testid="stDataFrame"] * {color:#432D34!important;}
    [data-testid="stExpander"] summary,[data-testid="stExpander"] summary * {
      color:#542532!important;background:#FFFFFF!important;
    }
    [data-testid="stAlert"] {background:#FBEFED!important;border:1px solid #EBC9C8!important;}
    [data-testid="stAlert"] * {color:#542532!important;}

    /* Result, help and warning panels: pale backgrounds with dark text */
    .info-box,.info-box * {color:#285A4A!important;background:#EAF4E9!important;border-color:#BBD7B5!important;}
    .notice,.notice * {color:#704921!important;background:#FFF3DF!important;border-color:#E9D1A5!important;}
    .result-card {
      background:linear-gradient(120deg,#EAF4E9,#F5F8EE)!important;
      border:1px solid #BDD5B2!important;box-shadow:0 8px 22px rgba(68,111,70,.08)!important;
    }
    .result-card * {color:#28543D!important;}
    .result-card.alert {
      background:linear-gradient(120deg,#FBE8E7,#FFF1EA)!important;
      border-color:#E4B5B2!important;
    }
    .result-card.alert * {color:#74283D!important;}
    .result-title {font-size:1.55rem!important;font-weight:800!important;}
    .result-tag {font-size:.9rem!important;font-weight:900!important;}

    /* Emergency support gets a deep rose surface with white text */
    .emergency {
      background:linear-gradient(115deg,#76243B,#A8324C,#C75D66)!important;
      border:1px solid #D89AA0!important;
    }
    .emergency * {color:#FFFFFF!important;}
    .phone,.phone * {background:#FFFFFF!important;color:#84283F!important;font-weight:900!important;}

    .footer {font-size:.9rem!important;}
    @media(max-width:760px) {
      .block-container {padding-left:1rem!important;padding-right:1rem!important;padding-top:.8rem!important;}
      .hero {padding:1.45rem!important;border-radius:20px!important;}
      .hero h1 {font-size:2.1rem!important;}
      h1 {font-size:2.05rem!important;} h2 {font-size:1.72rem!important;} h3 {font-size:1.35rem!important;}
      .stApp p,.stApp li {font-size:1.04rem!important;line-height:1.6!important;}
      .stat-value {font-size:1.5rem!important;}
      .stTabs [data-baseweb="tab"] {padding:9px 10px!important;font-size:.9rem!important;}
    }

    
    /* HEART-PALETTE SELECT MENUS — Sex, chest-pain, angina, slope, vessels, Thal, etc. */
    div[data-baseweb="select"] > div {
      background:#FFFFFF!important;
      border:1.5px solid #D7AEB5!important;
      border-radius:12px!important;
      box-shadow:0 2px 7px rgba(128,39,62,.05)!important;
      color:#542532!important;
    }
    div[data-baseweb="select"] > div:hover {
      border-color:#B83250!important;
      box-shadow:0 0 0 2px rgba(184,50,80,.10)!important;
    }
    div[data-baseweb="select"] > div:focus-within {
      border-color:#A8324C!important;
      box-shadow:0 0 0 3px rgba(184,50,80,.16)!important;
    }
    div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
    div[data-baseweb="select"] [role="combobox"],
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
      color:#542532!important;
      font-size:1.06rem!important;
      font-weight:600!important;
    }
    div[data-baseweb="select"] svg {fill:#A8324C!important;color:#A8324C!important;}
    /* Dropdown popover should be a clear white menu with a soft rose border. */
    [data-baseweb="popover"],[data-baseweb="menu"],
    [data-baseweb="popover"] > div,[data-baseweb="menu"] > div,
    ul[role="listbox"] {
      background:#FFFFFF!important;
      border:1px solid #E5C9CC!important;
      border-radius:12px!important;
      box-shadow:0 12px 30px rgba(83,32,47,.16)!important;
    }
    [data-baseweb="popover"] *,[data-baseweb="menu"] *,
    ul[role="listbox"] * {
      color:#542532!important;
      font-size:1.02rem!important;
    }
    [role="option"],[data-baseweb="menu"] [role="option"],
    ul[role="listbox"] li {
      background:#FFFFFF!important;
      color:#542532!important;
      padding-top:10px!important;
      padding-bottom:10px!important;
      min-height:42px!important;
      font-weight:550!important;
    }
    [role="option"]:hover,[role="option"][data-highlighted="true"],
    [role="option"][aria-selected="true"]:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    ul[role="listbox"] li:hover {
      background:#A8324C!important;
      color:#FFFFFF!important;
    }
    [role="option"]:hover *,[role="option"][data-highlighted="true"] *,
    [data-baseweb="menu"] [role="option"]:hover *,
    ul[role="listbox"] li:hover * {
      color:#FFFFFF!important;
    }
    [role="option"][aria-selected="true"],
    [role="option"][data-selected="true"] {
      background:#F8E2E5!important;
      color:#84283F!important;
      font-weight:800!important;
      box-shadow:inset 3px 0 0 #B83250!important;
    }
    [role="option"][aria-selected="true"] *,
    [role="option"][data-selected="true"] * {
      color:#84283F!important;
      font-weight:800!important;
    }


    /* COMPLETE LIGHT & FRESH HEART-HEALTH PALETTE — no dark panels */
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], .stApp, [data-testid="stHeader"] {
      background:#FFF9F7!important;color:#493B40!important;
    }
    .block-container {background:transparent!important;}
    .stApp p,.stApp li,.stApp label,.stApp small,
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"] {
      color:#51434A!important;
    }
    h1,h2,h3,h4,h5,h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
      color:#704653!important;text-shadow:none!important;
    }
    /* All-light hero: blush, peach and soft coral */
    .hero {
      background:linear-gradient(115deg,#FBE2E5 0%,#FCE9DF 52%,#E8F3E9 100%)!important;
      border:1px solid #EBCBD0!important;
      box-shadow:0 12px 30px rgba(177,111,120,.10)!important;
    }
    .hero:before {background:radial-gradient(circle,#F7C9D0,#F9E4D8 48%,transparent 72%)!important;}
    .hero:after {color:rgba(183,91,111,.10)!important;}
    .hero *, .hero h1,.hero h1 span,.hero p,.hero-stat,.hero-stat b,.pill {
      color:#704653!important;-webkit-text-fill-color:#704653!important;
    }
    .hero h1 span {background:none!important;}
    .hero-stat {background:#FFF7F4!important;border:1px solid #EED4D1!important;}
    .hero-stat b {color:#A95769!important;}
    .pill {background:#FFF7F5!important;border:1px solid #EBCBD0!important;}
    .live-dot {background:#91BE9A!important;box-shadow:none!important;}
    .ecg path {stroke:#C87583!important;filter:none!important;}
    /* Fresh pastel navigation, cards and statistic tiles */
    section[data-testid="stSidebar"] {background:#F8EFEE!important;border-right:1px solid #EAD8D7!important;}
    section[data-testid="stSidebar"] * {color:#59444B!important;}
    .side-chip {background:#FBE4E6!important;color:#9C5363!important;border-color:#EBCBD0!important;}
    .glass-card {background:linear-gradient(145deg,#FFFFFF,#FFF3F1)!important;border:1px solid #F0D9D6!important;box-shadow:0 7px 20px rgba(142,92,97,.07)!important;}
    .glass-card * {color:#59444B!important;}
    .glass-card .stat-value,.stat-value {color:#A95769!important;}
    .stat-label {color:#815A64!important;}
    .stat-note {color:#64866D!important;}
    .section-kicker {color:#B96578!important;}
    .section-sub,.small-muted,.footer {color:#68545A!important;}
    .glow-line {background:linear-gradient(90deg,#E8AAB5,#F2C5A8,#A9D2B3,#B8D7E9,transparent)!important;}
    /* Form and input surfaces stay white with readable labels */
    div[data-testid="stForm"],[data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:#FFFFFF!important;border:1px solid #EADBD9!important;border-radius:16px!important;
      box-shadow:0 5px 16px rgba(142,92,97,.045)!important;
    }
    div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,
    [data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
      background:#FFFFFF!important;color:#493B40!important;border:1px solid #DCC9C9!important;
    }
    input,textarea,[data-baseweb="input"] input,[data-baseweb="select"] *,
    [data-testid="stWidgetLabel"] {color:#493B40!important;font-size:1.06rem!important;}
    input::placeholder,textarea::placeholder,::placeholder {color:#8B777B!important;opacity:1!important;}
    [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {background:#FFFFFF!important;border:1px solid #EAD4D5!important;}
    [role="option"],[role="option"] * {background:#FFFFFF!important;color:#59444B!important;}
    [role="option"]:hover,[role="option"][data-highlighted="true"],
    [role="option"][aria-selected="true"]:hover {background:#FBE5E8!important;color:#814B59!important;}
    [role="option"]:hover *,[role="option"][data-highlighted="true"] * {color:#814B59!important;}
    [role="option"][aria-selected="true"],[role="option"][data-selected="true"] {
      background:#F8E8E9!important;color:#874F5B!important;font-weight:800!important;
      box-shadow:inset 3px 0 0 #D995A1!important;
    }
    [role="option"][aria-selected="true"] *,[role="option"][data-selected="true"] * {color:#874F5B!important;}
    div[data-baseweb="select"] svg {fill:#B86F7D!important;color:#B86F7D!important;}
    [data-testid="stNumberInput"] button {background:#FBEAEC!important;color:#8D5662!important;border:1px solid #E9D0D2!important;}
    /* Pastel buttons with dark berry text — no dark fills */
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button,
    [data-testid="stLinkButton"] a {
      background:linear-gradient(100deg,#F8DDE2,#FBE8DB 55%,#E2F0E3)!important;
      color:#784653!important;border:1px solid #E7C5CB!important;border-radius:12px!important;
      min-height:3.2rem!important;font-size:1.04rem!important;font-weight:800!important;
      box-shadow:0 4px 12px rgba(160,105,111,.09)!important;
    }
    .stButton>button *,.stFormSubmitButton>button *,
    .stDownloadButton>button *,[data-testid="stLinkButton"] a * {color:#784653!important;}
    .stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover,
    [data-testid="stLinkButton"] a:hover {background:#F4D5DB!important;border-color:#DFAEB8!important;}
    /* Soft tabs, pastel selected state */
    .stTabs [data-baseweb="tab-list"] {background:#F7ECEB!important;border:1px solid #EAD8D8!important;border-radius:15px!important;}
    .stTabs [data-baseweb="tab"],.stTabs [data-baseweb="tab"] * {color:#70535C!important;font-weight:750!important;}
    .stTabs [aria-selected="true"],.stTabs [aria-selected="true"] * {background:#FFFFFF!important;color:#A65D70!important;}
    /* Metrics and data tables */
    [data-testid="stMetric"] {background:#FFFFFF!important;border:1px solid #EADBD9!important;border-radius:13px!important;}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricDelta"],
    [data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *,[data-testid="stMetricDelta"] * {color:#704653!important;}
    [data-testid="stDataFrame"] {background:#FFFFFF!important;border:1px solid #EADBD9!important;border-radius:12px!important;}
    [data-testid="stDataFrame"] * {color:#493B40!important;}
    [data-testid="stExpander"] summary,[data-testid="stExpander"] summary * {color:#59444B!important;background:#FFFFFF!important;}
    [data-testid="stAlert"] {background:#F0F6EE!important;border:1px solid #D1E3CE!important;}
    [data-testid="stAlert"] * {color:#4E684F!important;}
    /* Result, information and warning panels — all pastel */
    .info-box,.info-box * {color:#426C59!important;background:#EAF5EC!important;border-color:#C7E0CA!important;}
    .notice,.notice * {color:#805E3C!important;background:#FFF3E3!important;border-color:#EED8B8!important;}
    .result-card {background:linear-gradient(120deg,#EAF5EC,#EEF6F4)!important;border:1px solid #C6DEC9!important;box-shadow:0 7px 18px rgba(99,139,103,.07)!important;}
    .result-card * {color:#42634C!important;}
    .result-card.alert {background:linear-gradient(120deg,#FBE9E9,#FFF0E8)!important;border-color:#EAC8C5!important;}
    .result-card.alert * {color:#895260!important;}
    .result-title {font-size:1.55rem!important;font-weight:800!important;}
    .result-tag {font-size:.9rem!important;font-weight:900!important;}
    /* Emergency panel is also light: pale peach with legible text */
    .emergency {background:linear-gradient(115deg,#FBE5E4,#FCEBDD,#F4EAF3)!important;border:1px solid #E9C9C9!important;box-shadow:0 8px 20px rgba(155,105,110,.07)!important;}
    .emergency h3,.emergency p,.emergency * {color:#704653!important;}
    .phone,.phone * {background:#FFFFFF!important;color:#A04F63!important;font-weight:900!important;border:1px solid #E8C8CE!important;}
    .footer {font-size:.92rem!important;}
    /* Preserve larger accessible type */
    .stApp,[data-testid="stMarkdownContainer"] {font-size:18px!important;}
    .stApp p,.stApp li {font-size:1.08rem!important;line-height:1.65!important;}
    .stApp label,[data-testid="stWidgetLabel"] {font-size:1.05rem!important;font-weight:700!important;}
    [data-testid="stCaptionContainer"],.stCaption,.small-muted,.section-sub {font-size:1rem!important;line-height:1.55!important;}
    @media(max-width:760px) {
      .block-container {padding-left:1rem!important;padding-right:1rem!important;padding-top:.8rem!important;}
      .hero {padding:1.35rem!important;border-radius:20px!important;}
      .hero h1 {font-size:2.05rem!important;}
      h1 {font-size:2.05rem!important;} h2 {font-size:1.72rem!important;} h3 {font-size:1.35rem!important;}
      .stApp p,.stApp li {font-size:1.04rem!important;line-height:1.6!important;}
      .stat-value {font-size:1.5rem!important;}
      .stTabs [data-baseweb="tab"] {padding:9px 10px!important;font-size:.9rem!important;}
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
    """Aligned 2D comparison chart using a clear heart-health palette."""
    metrics = list(frame.columns)
    models = list(frame.index)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="#FFF9F7")
    ax.set_facecolor("#FFFDFC")
    colors = ["#D96C7B", "#E9A0A8", "#83B89A", "#75AEB0"]
    y = np.arange(len(metrics))
    height = 0.34
    for i, model in enumerate(models):
        vals = frame.loc[model, metrics].to_numpy(dtype=float)
        bars = ax.barh(y + (i - (len(models)-1)/2) * height, vals,
                       height=height * .88, color=colors[i % len(colors)],
                       edgecolor="#FFFFFF", linewidth=1.2, label=str(model))
        for bar, val in zip(bars, vals):
            ax.text(min(val + .012, 1.015), bar.get_y() + bar.get_height()/2,
                    f"{val:.0%}", va="center", ha="left",
                    color="#69444D", fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([str(m).replace("_", " ").title() for m in metrics],
                       color="#684B53", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Evaluation score", color="#765A60", labelpad=9)
    ax.set_title("MODEL PERFORMANCE COMPARISON", color="#A94F63",
                 pad=16, fontsize=14, fontweight="bold")
    ax.xaxis.grid(True, color="#F0DDE0", linewidth=.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="#806B70", labelsize=9)
    ax.tick_params(axis="y", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF",
              edgecolor="#F0DDE0", labelcolor="#684B53", fontsize=9)
    fig.subplots_adjust(left=.22, right=.96, top=.86, bottom=.16)
    return fig


def make_3d_confusion(cm_array, model_name):
    """Aligned confusion matrix heatmap with legible counts and labels."""
    values = np.asarray(cm_array, dtype=int)
    fig, ax = plt.subplots(figsize=(7.4, 5.3), facecolor="#FFF9F7")
    ax.set_facecolor("#FFFDFC")
    palette = ["#F7DDE1", "#EBAAB4", "#D97A8A", "#A94F63"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(palette)
    vmax = max(int(values.max()), 1)
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks([0, 1], labels=["Predicted: Group 0", "Predicted: Group 1"])
    ax.set_yticks([0, 1], labels=["Actual: Group 0", "Actual: Group 1"])
    ax.set_title(f"{model_name} · CONFUSION MATRIX", color="#A94F63",
                 pad=16, fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", colors="#684B53", labelsize=9, length=0, pad=9)
    threshold = vmax * .55
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{values[row, col]}",
                    ha="center", va="center",
                    color="#FFFFFF" if values[row, col] >= threshold else "#704653",
                    fontsize=17, fontweight="bold")
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.27, right=.96, top=.86, bottom=.20)
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
        <span class="pill"><span class="live-dot"></span> Interactive model workspace</span>
        <span class="pill">13 feature signals</span>
        <span class="pill">Heart analytics</span>
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
    ("SELECTED MODEL", selected_model, "Chosen during model training", "#286D70"),
    ("TEST ACCURACY", pct(selected_metrics.get("accuracy")), "Held-out test subset", "#527B53"),
    ("PRECISION · GROUP 1", pct(selected_metrics.get("precision")), "Positive-class precision", "#A85E4D"),
    ("RECALL · GROUP 1", pct(selected_metrics.get("recall")), "Positive-class recall", "#5C7591"),
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
    ["✦ Live explorer", "▦ Scenario lab", "♡ Heart analytics", "♡ Heart guide", "✚ Emergency & map"]
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
        st.bar_chart(bar_df, color=["#347C7A"])
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
