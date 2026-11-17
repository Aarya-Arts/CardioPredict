import json
import os
from urllib.parse import quote

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

# ============================================================
# CardioPredict | Educational cardiovascular ML demonstration
# This file changes only the user interface. The trained model,
# dataset, 13 features, preprocessing and training pipeline stay
# unchanged.
# ============================================================

st.set_page_config(
    page_title="CardioPredict | Heart Health Learning Studio",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/cvd_model.joblib"
METRICS_PATH = "models/metrics.json"

# ----------------------- Styling ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --navy:#10243A; --teal:#0D9488; --ink:#172B3E; --muted:#64748B; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      background:radial-gradient(circle at 88% 2%,rgba(13,148,136,.10),transparent 26rem),
      linear-gradient(180deg,#F4F8FB 0%,#F8FAFC 55%,#EEF4F8 100%);
      color:var(--ink);
    }
    [data-testid="stHeader"] { background:rgba(248,250,252,.82); }
    .block-container { max-width:1280px; padding-top:1.35rem; padding-bottom:3rem; }
    section[data-testid="stSidebar"] {
      background:linear-gradient(180deg,#10243A 0%,#173B55 100%);
      border-right:1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * { color:#E8F1F7 !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#C7D7E4 !important; }
    section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#10243A !important; letter-spacing:-.025em; }
    p,label,.stCaption { color:#42566B; }
    .hero {
      position:relative; overflow:hidden; isolation:isolate;
      padding:2rem 2.15rem; border-radius:24px; margin-bottom:1.2rem;
      background:linear-gradient(115deg,#10243A 0%,#17445D 58%,#087F78 100%);
      box-shadow:0 18px 42px rgba(16,36,58,.16); color:white;
      animation:rise .7s ease both;
    }
    .hero:after {
      content:"♥"; position:absolute; right:4%; top:-55px; z-index:-1;
      font-size:230px; line-height:1; color:rgba(255,255,255,.055);
      animation:pulse 2.1s ease-in-out infinite;
    }
    .hero:before {
      content:""; position:absolute; z-index:-1; width:310px; height:210px;
      right:3%; bottom:-60px; border-radius:50%;
      background:radial-gradient(circle,rgba(45,212,191,.24),transparent 68%);
      filter:blur(8px); animation:drift 7s ease-in-out infinite;
    }
    .hero-kicker {
      display:inline-block; padding:6px 11px; border-radius:999px;
      background:rgba(204,251,241,.14); border:1px solid rgba(204,251,241,.25);
      color:#BDF7EA; font-size:.74rem; font-weight:800; letter-spacing:.09em;
      text-transform:uppercase; margin-bottom:.8rem;
    }
    .hero h1 { color:white !important; font-size:clamp(1.8rem,4vw,2.7rem); margin:0 0 .5rem 0; }
    .hero p { color:#D8E8EF !important; font-size:1rem; max-width:800px; margin:0; }
    .ecg { width:min(440px,75%); height:48px; margin-top:1rem; opacity:.9; }
    .ecg svg { width:100%; height:100%; }
    .ecg path { fill:none; stroke:#99F6E4; stroke-width:2.4; stroke-linecap:round;
      stroke-dasharray:900; stroke-dashoffset:900; animation:draw 2.7s ease .3s forwards;
      filter:drop-shadow(0 0 5px rgba(153,246,228,.6)); }
    @keyframes pulse { 0%,100%{transform:scale(1)} 12%{transform:scale(1.12)} 25%{transform:scale(1)} 38%{transform:scale(1.07)} 50%{transform:scale(1)} }
    @keyframes draw { to { stroke-dashoffset:0; } }
    @keyframes drift { 0%,100%{opacity:.25;transform:translate(0,0)} 50%{opacity:.6;transform:translate(-12px,8px)} }
    @keyframes rise { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
    .eyebrow { color:#0D9488; font-weight:800; letter-spacing:.11em; text-transform:uppercase; font-size:.72rem; margin-bottom:.25rem; }
    .subtle { color:#64748B; margin-top:-.4rem; margin-bottom:1rem; }
    .metric-card {
      background:rgba(255,255,255,.95); border:1px solid #E2E8F0; border-radius:17px;
      padding:16px 18px; min-height:108px; box-shadow:0 5px 18px rgba(16,36,58,.045);
      transition:transform .22s ease,box-shadow .22s ease;
    }
    .metric-card:hover { transform:translateY(-4px); box-shadow:0 14px 28px rgba(16,36,58,.10); }
    .metric-label { color:#64748B; font-size:.75rem; font-weight:700; }
    .metric-value { color:#10243A; font:800 1.48rem 'Manrope',sans-serif; margin-top:8px; }
    .metric-note { color:#0D9488; font-size:.72rem; font-weight:600; margin-top:4px; }
    div[data-testid="stForm"] { background:rgba(255,255,255,.92); border:1px solid #E2E8F0; border-radius:17px; padding:1rem 1.1rem; box-shadow:0 10px 30px rgba(16,36,58,.045); }
    div[data-baseweb="input"]>div,div[data-baseweb="select"]>div { background:white; border-color:#CBD5E1; border-radius:10px; }
    .stButton>button,.stFormSubmitButton>button {
      border-radius:11px; border:0; font-weight:800; min-height:2.8rem;
      background:linear-gradient(100deg,#0D9488,#0F766E); color:white !important;
      box-shadow:0 7px 16px rgba(13,148,136,.18);
    }
    .stButton>button:hover,.stFormSubmitButton>button:hover { background:linear-gradient(100deg,#0F766E,#115E59); transform:translateY(-1px); }
    .result {
      border-radius:18px; padding:1.2rem 1.3rem; border:1px solid #99F6E4;
      background:linear-gradient(120deg,#ECFDF5,#F0FDFA); color:#115E59;
      box-shadow:0 10px 24px rgba(13,148,136,.07); animation:rise .45s ease both;
    }
    .result.flagged { border-color:#FECACA; background:linear-gradient(120deg,#FFF1F2,#FFF7ED); color:#9F1239; }
    .result-kicker { font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; font-weight:800; opacity:.8; }
    .result-heading { font:800 1.5rem 'Manrope',sans-serif; margin:.3rem 0; }
    .notice { background:#FFF8E7; border:1px solid #FDE68A; color:#854D0E; border-radius:12px; padding:.9rem 1rem; font-size:.86rem; }
    .info { background:#EAF3F8; border:1px solid #D6E5EE; color:#36536A; border-radius:12px; padding:.9rem 1rem; font-size:.88rem; }
    .emergency {
      background:linear-gradient(115deg,#7F1D1D,#B91C1C); color:white; border-radius:18px;
      padding:1.2rem 1.35rem; box-shadow:0 12px 28px rgba(127,29,29,.16);
    }
    .emergency h3 { color:white !important; margin:0 0 .35rem 0; }
    .emergency p { color:#FEE2E2 !important; margin:.2rem 0; }
    .phone {
      display:inline-block; background:white; color:#991B1B !important; text-decoration:none !important;
      padding:9px 15px; border-radius:10px; font-weight:900; font-size:1.05rem; margin:.4rem .45rem .2rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #DCE5EC; }
    .stTabs [data-baseweb="tab"] { background:rgba(255,255,255,.65); border:1px solid #E2E8F0; border-radius:10px 10px 0 0; padding:10px 15px; }
    .stTabs [aria-selected="true"] { background:white !important; color:#0F766E !important; border-bottom:3px solid #0D9488 !important; font-weight:800; }
    .footer { color:#8091A2; font-size:.76rem; text-align:center; padding-top:1rem; }
    @media(max-width:700px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .hero { padding:1.4rem; }
      .hero:after { right:-25px; font-size:150px; }
      .metric-value { font-size:1.2rem; }
    }
    @media(prefers-reduced-motion:reduce) {
      *,*:before,*:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------- Helpers ----------------------------
def pct(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_3d_metric_chart(comparison_df):
    """3D grouped bars: models on X, metrics on Y, score on Z."""
    models = list(comparison_df.index)
    metrics = list(comparison_df.columns)
    fig = plt.figure(figsize=(10.5, 6.2), facecolor="#F8FAFC")
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#0F766E", "#38BDF8", "#F59E0B", "#8B5CF6"]
    dx, dy = 0.42, 0.42
    for yi, metric in enumerate(metrics):
        values = comparison_df[metric].to_numpy(dtype=float)
        xs = np.arange(len(models))
        ys = np.full(len(models), yi)
        ax.bar3d(xs, ys, np.zeros(len(models)), dx, dy, values,
                 color=colors[yi % len(colors)], alpha=.92, shade=True,
                 edgecolor="#FFFFFF", linewidth=.35)
        for x, y, z in zip(xs, ys, values):
            ax.text(x + dx/2, y + dy/2, z + .025, f"{z:.0%}",
                    ha="center", va="bottom", fontsize=8, color="#17324A")
    ax.set_xticks(np.arange(len(models)) + dx/2)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(np.arange(len(metrics)) + dy/2)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_zlim(0, 1.12)
    ax.set_zlabel("Score", labelpad=8)
    ax.set_title("Classifier performance · 3D comparison", pad=18,
                 fontsize=14, fontweight="bold", color="#10243A")
    ax.view_init(elev=25, azim=-58)
    ax.xaxis.pane.set_facecolor((.92, .96, .98, .8))
    ax.yaxis.pane.set_facecolor((.92, .96, .98, .8))
    ax.zaxis.pane.set_facecolor((.97, .98, .99, .8))
    ax.grid(True, alpha=.2)
    fig.tight_layout()
    return fig


def make_3d_confusion_matrix(cm_array, model_name):
    """3D columns for the four confusion-matrix cells."""
    fig = plt.figure(figsize=(7.6, 5.5), facecolor="#F8FAFC")
    ax = fig.add_subplot(111, projection="3d")
    labels = [
        "Correctly predicted\nno-disease group",
        "False positive\n(predicted group 1)",
        "Missed positive\n(predicted group 0)",
        "Correctly predicted\ndisease-present group",
    ]
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    values = [int(cm_array[0, 0]), int(cm_array[0, 1]),
              int(cm_array[1, 0]), int(cm_array[1, 1])]
    colors = ["#0F766E", "#F59E0B", "#E11D48", "#2563EB"]
    for (x, y), value, color, label in zip(positions, values, colors, labels):
        ax.bar3d(x, y, 0, .55, .55, value, color=color, shade=True,
                 alpha=.94, edgecolor="white")
        ax.text(x + .27, y + .27, value + .6, str(value),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks([.27, 1.27])
    ax.set_xticklabels(["Actual 0", "Actual 1"])
    ax.set_yticks([.27, 1.27])
    ax.set_yticklabels(["Predicted 0", "Predicted 1"])
    ax.set_zlabel("Number of test samples")
    ax.set_zlim(0, max(values + [1]) * 1.25)
    ax.set_title(f"{model_name} · 3D confusion matrix",
                 fontsize=13, fontweight="bold", color="#10243A", pad=15)
    ax.view_init(elev=24, azim=-52)
    ax.grid(True, alpha=.2)
    fig.tight_layout()
    return fig


# ----------------------- Load artifacts ---------------------
if not os.path.exists(MODEL_PATH):
    st.error("The trained model file is missing. Please check models/cvd_model.joblib.")
    st.stop()
if not os.path.exists(METRICS_PATH):
    st.error("The model metrics file is missing. Please check models/metrics.json.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics_data = json.load(file)
except Exception as exc:
    st.error(f"Could not load the model or metrics: {exc}")
    st.stop()

selected_model = metrics_data.get("selected_model", "Not specified")
results = metrics_data.get("results", {})
selected_metrics = results.get(selected_model, {})

# ----------------------- Sidebar ----------------------------
with st.sidebar:
    st.markdown("## ❤️ CardioPredict")
    st.caption("HEART HEALTH LEARNING STUDIO")
    st.markdown("---")
    st.markdown("**What this app does**")
    st.write("Demonstrates how a machine-learning classifier processes 13 attributes from the historical UCI Cleveland dataset.")
    st.markdown("**Algorithms**")
    st.caption("Logistic Regression · Random Forest")
    st.markdown("**Technology**")
    st.caption("Python · Pandas · Scikit-learn · Matplotlib · Streamlit")
    st.markdown("---")
    st.caption("Academic prototype only. It is not a medical device, diagnosis or emergency triage service.")

# ----------------------- Hero -------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Machine learning · Educational prototype</div>
      <div class="ecg"><svg viewBox="0 0 520 70" preserveAspectRatio="none">
        <path d="M2 38 L82 38 L101 35 L116 41 L132 38 L157 38 L174 13 L192 59 L212 27 L229 38 L290 38 L309 34 L325 41 L340 38 L367 38 L383 18 L401 56 L419 29 L436 38 L518 38"/>
      </svg></div>
      <h1>CardioPredict</h1>
      <p>A clear, visual introduction to cardiovascular disease and a machine-learning model.
      Explore the inputs, understand the model's output in everyday language, and view its test-set performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------- Plain-language explainer ----------
with st.expander("❤️ What is cardiovascular disease?", expanded=False):
    st.write(
        "Cardiovascular disease (CVD) is a broad term for conditions affecting the heart and blood vessels. "
        "Coronary heart disease is one type; it can occur when the blood supply to the heart muscle is reduced, "
        "often because fatty deposits narrow the coronary arteries."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Common risk factors**")
        st.markdown("- High blood pressure\n- High cholesterol\n- Diabetes or high blood sugar\n- Smoking and tobacco use\n- Physical inactivity\n- Family history and increasing age")
    with c2:
        st.markdown("**Possible warning symptoms**")
        st.markdown("- Chest pressure, tightness or pain\n- Pain spreading to the arm, back, neck or jaw\n- Shortness of breath\n- Cold sweat, nausea or light-headedness\n- Unusual sudden weakness")
    st.caption("Some people have no obvious symptoms. Only a qualified healthcare professional can assess symptoms and diagnose disease.")

# ----------------------- KPI row ----------------------------
kpi_cols = st.columns(4)
kpi_data = [
    ("MODEL SELECTED", selected_model, "Chosen during training"),
    ("TEST ACCURACY", pct(selected_metrics.get("accuracy")), "Held-out test split"),
    ("PRECISION · GROUP 1", pct(selected_metrics.get("precision")), "Positive-class precision"),
    ("RECALL · GROUP 1", pct(selected_metrics.get("recall")), "Positive-class recall"),
]
for col, (label, value, note) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
prediction_tab, analytics_tab, learn_tab, help_tab = st.tabs(
    ["♥ Prediction", "▥ Model performance", "ⓘ Learn about CVD", "✚ Emergency & hospitals"]
)

# ----------------------- Prediction -------------------------
with prediction_tab:
    st.markdown('<div class="eyebrow">Step 1 · Enter the sample information</div>', unsafe_allow_html=True)
    st.subheader("Patient attributes")
    st.markdown(
        '<div class="subtle">These are the same 13 features used by the trained model. '
        'The short descriptions explain the dataset fields; use actual recorded values rather than guessing.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Basic and resting measurements")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=50,
                                  help="Age in years.")
            sex = st.selectbox("Sex (dataset code)", [0, 1],
                               format_func=lambda x: "Female · code 0" if x == 0 else "Male · code 1",
                               help="The numeric coding used in the original dataset.")
            cp = st.selectbox(
                "Chest pain category",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 · Typical angina",
                    2: "2 · Atypical angina",
                    3: "3 · Non-anginal pain",
                    4: "4 · Asymptomatic in dataset",
                }[x],
                help="Historical dataset category, not a self-diagnosis. A person should not classify chest pain themselves.",
            )
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=250, value=120,
                                       help="Blood pressure measured while resting.")
            chol = st.number_input("Serum cholesterol (mg/dL)", min_value=100, max_value=600, value=200,
                                   help="Total serum cholesterol measurement recorded in the dataset.")
            fbs = st.selectbox("Fasting blood sugar above 120 mg/dL?", [0, 1],
                               format_func=lambda x: "No · code 0" if x == 0 else "Yes · code 1")
            restecg = st.selectbox(
                "Resting ECG category", [0, 1, 2],
                format_func=lambda x: {
                    0: "0 · Normal",
                    1: "1 · ST-T wave abnormality",
                    2: "2 · Probable/definite left ventricular hypertrophy",
                }[x],
                help="ECG category from the original dataset. Interpretation requires a clinician.",
            )
        with right:
            st.markdown("#### Exercise and cardiac measurements")
            thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=250, value=150,
                                      help="Maximum heart rate recorded during the exercise test.")
            exang = st.selectbox("Exercise-induced angina?", [0, 1],
                                 format_func=lambda x: "No · code 0" if x == 0 else "Yes · code 1")
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0,
                                      value=1.0, step=0.1,
                                      help="ECG measurement during exercise relative to rest. This is not a value to estimate.")
            slope = st.selectbox(
                "Peak exercise ST-segment slope", [1, 2, 3],
                format_func=lambda x: {
                    1: "1 · Upsloping",
                    2: "2 · Flat",
                    3: "3 · Downsloping",
                }[x],
            )
            ca = st.selectbox("Major vessels seen by fluoroscopy", [0, 1, 2, 3],
                              help="Number of major vessels (0 to 3) recorded by the test.")
            thal = st.selectbox(
                "Thal test category", [3, 6, 7],
                format_func=lambda x: {
                    3: "3 · Normal",
                    6: "6 · Fixed defect",
                    7: "7 · Reversible defect",
                }[x],
                help="Historical coded test result. A clinician should interpret the original report.",
            )
        submitted = st.form_submit_button("Run educational model prediction  →",
                                          type="primary", use_container_width=True)

    if submitted:
        input_row = pd.DataFrame([{
            "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach,
            "oldpeak": oldpeak, "sex": sex, "cp": cp, "fbs": fbs,
            "restecg": restecg, "exang": exang, "slope": slope, "ca": ca, "thal": thal,
        }])
        try:
            predicted_class = int(model.predict(input_row)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_row)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                positive_index = classes.index(1) if 1 in classes else min(1, len(probabilities) - 1)
                probability = float(probabilities[positive_index])

            st.markdown("---")
            st.markdown('<div class="eyebrow">Step 2 · Understand the result</div>', unsafe_allow_html=True)
            st.subheader("What the model output means")

            if predicted_class == 1:
                st.markdown(
                    '<div class="result flagged"><div class="result-kicker">Model output · Group 1</div>'
                    '<div class="result-heading">Pattern associated with the disease-present group</div>'
                    '<div>The model places this set of entered measurements in the group that was labelled '
                    'as disease-present in its training dataset. This is not proof that a person has heart disease.</div></div>',
                    unsafe_allow_html=True,
                )
                st.warning(
                    "What should you do? Do not panic or self-diagnose from this result. "
                    "If this is about a real person, discuss health concerns and the original test results with a qualified doctor. "
                    "If there are emergency symptoms, call emergency services immediately."
                )
            else:
                st.markdown(
                    '<div class="result"><div class="result-kicker">Model output · Group 0</div>'
                    '<div class="result-heading">Pattern associated with the no-disease group</div>'
                    '<div>The model places this set of entered measurements in the group labelled no disease '
                    'in its training dataset. This does not rule out heart disease or guarantee that a person is healthy.</div></div>',
                    unsafe_allow_html=True,
                )
                st.info(
                    "A result in this group is not a medical clearance. Persistent symptoms or health concerns "
                    "still need assessment by a qualified healthcare professional."
                )

            st.write("")
            if probability is not None:
                st.markdown("**Model probability for Group 1**")
                st.progress(max(0.0, min(1.0, probability)))
                st.metric("Estimated model probability", pct(probability))
                st.caption(
                    "This number describes the model's output for this sample. It is not a clinically calibrated "
                    "personal risk percentage, disease severity score, or emergency score."
                )

            with st.expander("What do Group 0 and Group 1 mean?"):
                st.markdown(
                    "- **Group 0:** Original dataset target was 0, meaning no disease was recorded in that dataset row.\n"
                    "- **Group 1:** Original dataset targets 1, 2, 3 or 4 were combined into one disease-present group.\n\n"
                    "Your model is a binary classifier. It does not produce Class 2, moderate, severe or critical categories."
                )
            st.markdown(
                '<div class="notice"><b>Important limitation:</b> CardioPredict is a student academic prototype '
                'trained on a small historical dataset. It has not been clinically validated and must not be used '
                'to diagnose, screen, triage, prescribe treatment, or decide whether emergency care is needed.</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Prediction could not be generated. Please check the model and input features. Details: {exc}")

# ----------------------- Analytics -------------------------
with analytics_tab:
    st.markdown('<div class="eyebrow">Model evaluation</div>', unsafe_allow_html=True)
    st.subheader("How well did the algorithms perform on the test data?")
    st.markdown(
        '<div class="subtle">These figures describe performance on the held-out test subset, '
        'not the accuracy of a medical diagnosis in the real world.</div>',
        unsafe_allow_html=True,
    )

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
        comparison_df = pd.DataFrame(rows).set_index("Algorithm")
        st.dataframe(comparison_df.style.format("{:.1%}"), use_container_width=True)

        st.markdown("#### 3D algorithm comparison")
        st.caption("Each column represents a test metric. Higher bars indicate a higher score on this particular test split.")
        fig = make_3d_metric_chart(comparison_df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Confusion matrix")
        chosen_model = st.selectbox("Select an algorithm", list(results.keys()))
        cm_values = results[chosen_model].get("confusion_matrix")
        if cm_values and len(cm_values) == 2:
            cm_array = np.asarray(cm_values)
            c1, c2 = st.columns([1, 1.15])
            with c1:
                st.markdown("**Exact test counts**")
                cm_table = pd.DataFrame(
                    cm_array,
                    index=["Actual Group 0", "Actual Group 1"],
                    columns=["Predicted Group 0", "Predicted Group 1"],
                )
                st.dataframe(cm_table, use_container_width=True)
                st.caption("Rows are actual dataset labels; columns are model predictions.")
            with c2:
                fig_cm = make_3d_confusion_matrix(cm_array, chosen_model)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            st.markdown(
                "- **Correct Group 0:** model correctly predicted a sample from the no-disease dataset group.\n"
                "- **False positive:** model predicted Group 1 when the actual dataset label was Group 0.\n"
                "- **Missed positive:** model predicted Group 0 when the actual dataset label was Group 1.\n"
                "- **Correct Group 1:** model correctly predicted a sample from the disease-present dataset group."
            )
    else:
        st.info("No evaluation metrics were found. Run the training script first.")

# ----------------------- Learn about CVD -------------------
with learn_tab:
    st.markdown('<div class="eyebrow">Health education</div>', unsafe_allow_html=True)
    st.subheader("Understanding cardiovascular health")
    st.write(
        "Cardiovascular diseases affect the heart and blood vessels. Coronary artery disease is one form. "
        "It can develop when deposits build up inside arteries and reduce blood flow to the heart."
    )
    a, b = st.columns(2)
    with a:
        st.markdown("#### Risk factors people can discuss with a clinician")
        st.markdown(
            "- High blood pressure\n"
            "- High LDL cholesterol\n"
            "- Diabetes\n"
            "- Smoking or tobacco exposure\n"
            "- Low physical activity\n"
            "- Diet high in salt or unhealthy fats\n"
            "- Family history and age"
        )
    with b:
        st.markdown("#### Helpful heart-health habits")
        st.markdown(
            "- Avoid tobacco and smoking\n"
            "- Be physically active as appropriate for your health\n"
            "- Choose a balanced diet\n"
            "- Attend recommended health checks\n"
            "- Follow prescribed treatment\n"
            "- Seek professional advice for symptoms"
        )
    st.markdown(
        '<div class="info"><b>Remember:</b> Risk factors are not a diagnosis. Some risk factors can be changed, '
        'while others cannot. A healthcare professional can assess an individual using history, examination and appropriate tests.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### When symptoms may be an emergency")
    st.error(
        "New or severe chest pressure/pain, pain spreading to the arm/jaw/back, severe shortness of breath, "
        "fainting, cold sweating or sudden collapse can be signs of a medical emergency. Call emergency services; "
        "do not wait for CardioPredict to produce a result."
    )

# ----------------------- Emergency & hospitals -------------
with help_tab:
    st.markdown('<div class="eyebrow">Emergency information</div>', unsafe_allow_html=True)
    st.subheader("Get medical help when it is needed")
    st.markdown(
        '<div class="emergency"><h3>Possible medical emergency?</h3>'
        '<p>Do not use this prediction app to decide whether to call for help. '
        'If someone has possible heart-attack symptoms, severe breathing difficulty, faints or collapses, '
        'call emergency services immediately.</p>'
        '<a class="phone" href="tel:112">Call 112 · Emergency</a>'
        '<a class="phone" href="tel:102">Call 102 · Ambulance service</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("India: 112 is the nationwide integrated emergency response number. 102 is listed by the National Portal of India as the National Ambulance Service. Availability and dispatch depend on local services.")
    st.markdown("#### Find hospitals near a location")
    st.write(
        "This hosted app cannot reliably access a visitor's exact GPS position automatically. "
        "Enter a city, area or PIN code below to open a live Google Maps search. Check the hospital's "
        "emergency department availability and call before travelling when it is safe to do so."
    )
    location_text = st.text_input("Your city, area or PIN code", value="Mysuru, Karnataka",
                                  placeholder="For example: Mysuru, Karnataka")
    if location_text.strip():
        hospital_query = quote("emergency hospital near " + location_text.strip())
        map_url = "https://www.google.com/maps/search/?api=1&query=" + hospital_query
        st.link_button("Find nearby hospitals on Google Maps ↗", map_url, use_container_width=True)
        general_query = quote("hospital near " + location_text.strip())
        st.markdown(
            f"[Open general hospital search on Google Maps](https://www.google.com/maps/search/?api=1&query={general_query})"
        )
    st.markdown(
        '<div class="notice"><b>Safety note:</b> Map listings, routes, phone numbers and emergency-department '
        'availability can change. CardioPredict does not dispatch an ambulance, transmit a patient's location, '
        'contact hospitals, or confirm that a facility can accept an emergency case.</div>',
        unsafe_allow_html=True,
    )

# ----------------------- Footer ----------------------------
st.markdown("---")
st.markdown(
    '<div class="footer">CARDIOPREDICT · BE CSE MINI PROJECT · EDUCATIONAL MACHINE-LEARNING DEMONSTRATION<br>'
    'Not a medical device · Not clinically validated · Do not use for diagnosis or emergency triage</div>',
    unsafe_allow_html=True,
)
