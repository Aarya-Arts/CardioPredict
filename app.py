import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# CardioPredict - academic cardiovascular ML demonstration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict | Cardiovascular ML",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/cvd_model.joblib"
METRICS_PATH = "models/metrics.json"

# ---------- Visual theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
      --navy: #10243A;
      --navy2: #173B55;
      --teal: #0D9488;
      --teal-light: #CCFBF1;
      --ink: #15283B;
      --muted: #64748B;
      --surface: #FFFFFF;
      --line: #E2E8F0;
    }
    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp {
      background:
        radial-gradient(circle at 88% 2%, rgba(13,148,136,.09), transparent 25rem),
        linear-gradient(180deg, #F4F8FB 0%, #F7FAFC 55%, #EEF4F8 100%);
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(247,250,252,.82); }
    .block-container { max-width: 1240px; padding-top: 1.7rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #10243A 0%, #173B55 100%);
      border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * { color: #E8F1F7 !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #C7D7E4 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #10243A !important; letter-spacing: -.025em; }
    p, label, .stCaption { color: #42566B; }
    .hero {
      position: relative; overflow: hidden;
      padding: 2rem 2.15rem; border-radius: 24px; margin-bottom: 1.3rem;
      background: linear-gradient(115deg, #10243A 0%, #17445D 58%, #087F78 100%);
      box-shadow: 0 18px 42px rgba(16,36,58,.16);
      color: white;
    }
    .hero:after {
      content: "♥"; position: absolute; right: 4%; top: -48px;
      font-size: 220px; line-height: 1; color: rgba(255,255,255,.055);
      font-family: Arial, sans-serif; transform: rotate(-12deg);
    }
    .hero-kicker {
      display: inline-block; padding: 6px 11px; border-radius: 999px;
      background: rgba(204,251,241,.14); border: 1px solid rgba(204,251,241,.25);
      color: #BDF7EA; font-size: .76rem; font-weight: 700; letter-spacing: .09em;
      text-transform: uppercase; margin-bottom: .8rem;
    }
    .hero h1 { color: white !important; font-size: clamp(1.8rem, 4vw, 2.65rem); margin: 0 0 .5rem 0; }
    .hero p { color: #D8E8EF !important; font-size: 1rem; max-width: 760px; margin: 0; }
    .section-eyebrow {
      color: #0D9488; font-weight: 800; letter-spacing: .11em;
      text-transform: uppercase; font-size: .72rem; margin-bottom: .25rem;
    }
    .section-subtitle { color: #64748B; margin-top: -.45rem; margin-bottom: 1.1rem; }
    .kpi {
      background: rgba(255,255,255,.94); border: 1px solid #E2E8F0;
      border-radius: 17px; padding: 17px 18px; min-height: 112px;
      box-shadow: 0 5px 18px rgba(16,36,58,.045);
    }
    .kpi-label { color: #64748B; font-size: .78rem; font-weight: 600; margin-bottom: 9px; }
    .kpi-value { color: #10243A; font-family: 'Manrope',sans-serif; font-size: 1.55rem; font-weight: 800; line-height: 1.15; }
    .kpi-note { color: #0D9488; font-size: .72rem; margin-top: 6px; font-weight: 600; }
    div[data-testid="stForm"], div[data-testid="stExpander"], div[data-testid="stDataFrame"] {
      border: 1px solid #E2E8F0; border-radius: 16px;
    }
    div[data-testid="stForm"] { background: rgba(255,255,255,.88); padding: 1rem 1.1rem; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
      background: #FFFFFF; border-color: #CBD5E1; border-radius: 10px;
    }
    .stButton > button, .stFormSubmitButton > button {
      border-radius: 11px; border: 0; font-weight: 700; min-height: 2.8rem;
      background: linear-gradient(100deg, #0D9488, #0F766E); color: #FFFFFF !important;
      box-shadow: 0 7px 16px rgba(13,148,136,.18);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
      background: linear-gradient(100deg, #0F766E, #115E59); color: white;
      border: 0; transform: translateY(-1px);
    }
    .result-card {
      border-radius: 17px; padding: 1.1rem 1.25rem; margin-top: .4rem;
      border: 1px solid #99F6E4; background: linear-gradient(120deg,#ECFDF5,#F0FDFA);
      color: #115E59;
    }
    .result-card.positive {
      border-color: #FECACA; background: linear-gradient(120deg,#FFF1F2,#FFF7ED); color: #9F1239;
    }
    .result-title { font-size: .75rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 800; opacity: .78; }
    .result-main { font-family: 'Manrope',sans-serif; font-size: 1.45rem; font-weight: 800; margin-top: .25rem; }
    .notice {
      background: #FFF8E7; border: 1px solid #FDE68A; color: #854D0E;
      border-radius: 12px; padding: .85rem 1rem; font-size: .86rem;
    }
    .info-strip {
      background: #EAF3F8; border: 1px solid #D6E5EE; border-radius: 12px;
      padding: .8rem 1rem; color: #36536A; font-size: .88rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #DCE5EC; }
    .stTabs [data-baseweb="tab"] {
      background: rgba(255,255,255,.65); border: 1px solid #E2E8F0;
      border-radius: 10px 10px 0 0; padding: 10px 16px; color: #42566B;
    }
    .stTabs [aria-selected="true"] {
      background: #FFFFFF !important; color: #0F766E !important;
      border-bottom: 3px solid #0D9488 !important; font-weight: 800;
    }
    .footer { color: #8091A2; font-size: .78rem; text-align: center; padding-top: 1rem; }
    
    /* Cinematic motion layer — decorative only; prediction logic and inputs stay untouched */
    @keyframes cp-heartbeat {
      0%, 100% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(45,212,191,0)); }
      12% { transform: scale(1.13); filter: drop-shadow(0 0 18px rgba(45,212,191,.45)); }
      24% { transform: scale(1); }
      36% { transform: scale(1.08); }
      48% { transform: scale(1); }
    }
    @keyframes cp-ecg-draw { from { stroke-dashoffset: 900; } to { stroke-dashoffset: 0; } }
    @keyframes cp-glow-drift { 0%,100% { opacity:.25; transform:translate3d(0,0,0) scale(1); } 50% { opacity:.6; transform:translate3d(-12px,8px,0) scale(1.08); } }
    @keyframes cp-rise { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
    @keyframes cp-shimmer { 0% { transform:translateX(-130%) skewX(-18deg); } 100% { transform:translateX(240%) skewX(-18deg); } }
    .hero { isolation:isolate; animation:cp-rise .75s cubic-bezier(.2,.8,.2,1) both; }
    .hero:after { animation:cp-heartbeat 2.1s ease-in-out infinite; transform-origin:center; }
    .hero:before {
      content:""; position:absolute; z-index:-1; width:310px; height:210px;
      right:3%; bottom:-60px; border-radius:50%;
      background:radial-gradient(circle,rgba(45,212,191,.22),transparent 68%);
      filter:blur(8px); animation:cp-glow-drift 7s ease-in-out infinite;
    }
    .hero-ecg { position:relative; z-index:1; width:min(430px,72%); height:54px; margin-top:1.25rem; opacity:.88; }
    .hero-ecg svg { width:100%; height:100%; overflow:visible; }
    .hero-ecg path { fill:none; stroke:#99F6E4; stroke-width:2.3; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:900; stroke-dashoffset:900; animation:cp-ecg-draw 2.8s ease-in-out .35s forwards; filter:drop-shadow(0 0 5px rgba(153,246,228,.6)); }
    .kpi { position:relative; overflow:hidden; transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease; animation:cp-rise .65s ease both; }
    .kpi:after { content:""; position:absolute; top:0; left:0; width:42%; height:2px; background:linear-gradient(90deg,transparent,#2DD4BF,transparent); opacity:0; }
    .kpi:hover { transform:translateY(-5px); box-shadow:0 15px 30px rgba(16,36,58,.10); border-color:#99F6E4; }
    .kpi:hover:after { opacity:1; animation:cp-shimmer 1.1s ease; }
    .stTabs [data-baseweb="tab"] { transition:all .2s ease; }
    .stTabs [data-baseweb="tab"]:hover { transform:translateY(-2px); border-color:#99F6E4; }
    div[data-testid="stForm"] { box-shadow:0 12px 34px rgba(16,36,58,.045); transition:box-shadow .25s ease; }
    div[data-testid="stForm"]:hover { box-shadow:0 16px 40px rgba(16,36,58,.075); }
    .result-card { animation:cp-rise .55s cubic-bezier(.2,.8,.2,1) both; box-shadow:0 10px 24px rgba(13,148,136,.07); }
    .result-card.positive { box-shadow:0 10px 24px rgba(225,29,72,.07); }
    .section-eyebrow { display:inline-flex; align-items:center; gap:8px; }
    .section-eyebrow:before { content:""; width:20px; height:2px; background:#0D9488; border-radius:3px; }
    .stProgress > div > div > div > div { background:linear-gradient(90deg,#0D9488,#5EEAD4); }
    @media (prefers-reduced-motion: reduce) {
      *, *:before, *:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; scroll-behavior:auto !important; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model and metrics ----------
if not os.path.exists(MODEL_PATH):
    st.error(f"Trained model not found at `{MODEL_PATH}`. Run your training script first.")
    st.stop()
if not os.path.exists(METRICS_PATH):
    st.error(f"Metrics file not found at `{METRICS_PATH}`. Run your training script first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
except Exception as exc:
    st.error(f"Could not load the saved model or metrics: {exc}")
    st.stop()

selected_model = metrics_data.get("selected_model", "Not specified")
results = metrics_data.get("results", {})
selected_metrics = results.get(selected_model, {})

# The saved training report's precision/recall/F1 values are class-1 metrics.
# Keep the labels explicit rather than calling them weighted averages.
def pct(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ❤️ CardioPredict")
    st.markdown("**CARDIOVASCULAR ML STUDIO**")
    st.caption("BE CSE · Mini Project")
    st.divider()
    st.markdown("### Project snapshot")
    st.write("**Dataset**")
    st.caption("UCI Cleveland Heart Disease")
    st.write("**Algorithms**")
    st.caption("Logistic Regression · Random Forest")
    st.write("**Application**")
    st.caption("Python · Scikit-learn · Streamlit")
    st.divider()
    st.markdown("### Evaluation")
    st.metric("Selected classifier", selected_model)
    if selected_metrics:
        st.metric("Held-out accuracy", pct(selected_metrics.get("accuracy")))
    st.divider()
    st.caption("Academic prototype only. Not a clinical decision-support system.")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Machine Learning · Academic Demonstration</div>
      <div class="hero-ecg" aria-hidden="true">
        <svg viewBox="0 0 520 70" preserveAspectRatio="none">
          <path d="M2 38 L82 38 L101 35 L116 41 L132 38 L157 38 L174 13 L192 59 L212 27 L229 38 L290 38 L309 34 L325 41 L340 38 L367 38 L383 18 L401 56 L419 29 L436 38 L518 38"/>
        </svg>
      </div>
      <h1>Cardiovascular Disease Prediction</h1>
      <p>Explore a supervised learning workflow built around the UCI Cleveland dataset.
      Enter coded sample attributes, inspect model estimates, and compare classifier performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("SELECTED MODEL", selected_model, "Best model in saved evaluation"),
    ("TEST ACCURACY", pct(selected_metrics.get("accuracy")), "Held-out test split"),
    ("CLASS 1 PRECISION", pct(selected_metrics.get("precision")), "Positive-class precision"),
    ("CLASS 1 F1-SCORE", pct(selected_metrics.get("f1_score")), "Positive-class F1-score"),
]
for col, (label, value, note) in zip([k1, k2, k3, k4], kpis):
    with col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
prediction_tab, performance_tab, about_tab = st.tabs(
    ["♥  Prediction workspace", "▥  Model analytics", "ⓘ  Project overview"]
)

# ---------- Prediction ----------
with prediction_tab:
    st.markdown('<div class="section-eyebrow">Interactive model</div>', unsafe_allow_html=True)
    st.subheader("Enter sample attributes")
    st.markdown(
        '<div class="section-subtitle">Use the dataset coding shown in each field. '
        'All values are passed to the saved preprocessing/model pipeline.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Personal & resting measurements")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
            sex = st.selectbox("Sex (dataset coding)", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            cp = st.selectbox("Chest pain type (1–4)", [1, 2, 3, 4])
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=250, value=120)
            chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            restecg = st.selectbox("Resting ECG result (0–2)", [0, 1, 2])
        with right:
            st.markdown("#### Exercise & cardiac measurements")
            thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=250, value=150)
            exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Peak exercise ST slope (1–3)", [1, 2, 3])
            ca = st.selectbox("Major vessels (0–3)", [0, 1, 2, 3])
            thal = st.selectbox("Thal result (dataset code)", [3, 6, 7])

        submitted = st.form_submit_button("Generate model estimate  →", type="primary", use_container_width=True)

    if submitted:
        input_row = pd.DataFrame([{
            "age": age,
            "trestbps": trestbps,
            "chol": chol,
            "thalach": thalach,
            "oldpeak": oldpeak,
            "sex": sex,
            "cp": cp,
            "fbs": fbs,
            "restecg": restecg,
            "exang": exang,
            "slope": slope,
            "ca": ca,
            "thal": thal,
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
            st.markdown('<div class="section-eyebrow">Model response</div>', unsafe_allow_html=True)
            st.subheader("Prediction result")
            r1, r2 = st.columns([1.15, 1])
            with r1:
                if predicted_class == 1:
                    st.markdown(
                        '<div class="result-card positive"><div class="result-title">Classifier output</div>'
                        '<div class="result-main">Positive class · 1</div>'
                        '<div>For this supplied sample, the model predicted class 1.</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-card"><div class="result-title">Classifier output</div>'
                        '<div class="result-main">Negative class · 0</div>'
                        '<div>For this supplied sample, the model predicted class 0.</div></div>',
                        unsafe_allow_html=True,
                    )
            with r2:
                if probability is not None:
                    st.metric("Estimated probability for class 1", pct(probability))
                    st.progress(max(0.0, min(1.0, probability)))
                    st.caption("Model probability for the positive class; not a personal medical risk estimate.")
            st.markdown(
                '<div class="notice"><b>Educational-use notice:</b> This prototype uses a small historical dataset. '
                'Its output is not a diagnosis, screening result, or personal risk score. Do not use it for medical '
                'decisions. For health concerns, consult a qualified healthcare professional.</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Prediction could not be generated. Check that the saved model matches the input features. Details: {exc}")

# ---------- Model analytics ----------
with performance_tab:
    st.markdown('<div class="section-eyebrow">Evaluation results</div>', unsafe_allow_html=True)
    st.subheader("Held-out test-set performance")
    st.markdown(
        '<div class="section-subtitle">Metrics are loaded from the training output. '
        'Precision, recall and F1-score below refer to class 1, matching the saved report values.</div>',
        unsafe_allow_html=True,
    )

    if results:
        comparison_rows = []
        for model_name, values in results.items():
            comparison_rows.append({
                "Model": model_name,
                "Accuracy": values.get("accuracy", 0),
                "Precision (class 1)": values.get("precision", 0),
                "Recall (class 1)": values.get("recall", 0),
                "F1-score (class 1)": values.get("f1_score", 0),
            })
        comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
        st.dataframe(
            comparison_df.style.format("{:.1%}"),
            use_container_width=True,
        )

        st.markdown("#### Classifier comparison")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        metric_names = list(comparison_df.columns)
        model_names = list(comparison_df.index)
        x = np.arange(len(model_names))
        width = 0.19
        palette = ["#0F766E", "#38BDF8", "#F59E0B", "#6366F1"]
        for i, metric in enumerate(metric_names):
            offsets = x + (i - (len(metric_names) - 1) / 2) * width
            ax.bar(offsets, comparison_df[metric].values, width, label=metric, color=palette[i])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", color="#475569")
        ax.set_title("Model performance by metric", loc="left", color="#10243A", fontsize=14, fontweight="bold", pad=14)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="y", length=0, colors="#64748B")
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Confusion matrix")
        chosen_model = st.selectbox("Choose classifier", list(results.keys()))
        cm = results[chosen_model].get("confusion_matrix")
        if cm and len(cm) == 2:
            cm_array = np.asarray(cm)
            fig_cm, ax_cm = plt.subplots(figsize=(6.4, 4.2))
            image = ax_cm.imshow(cm_array, cmap="GnBu")
            ax_cm.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
            ax_cm.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
            ax_cm.set_title(f"{chosen_model} · Confusion Matrix", loc="left", fontsize=13, fontweight="bold", color="#10243A", pad=14)
            threshold = cm_array.max() / 2 if cm_array.size else 0
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm_array[i, j]), ha="center", va="center",
                               color="white" if cm_array[i, j] > threshold else "#10243A",
                               fontsize=16, fontweight="bold")
            ax_cm.spines[:].set_visible(False)
            ax_cm.tick_params(length=0, pad=10)
            fig_cm.colorbar(image, ax=ax_cm, fraction=.045, pad=.04)
            fig_cm.tight_layout()
            st.pyplot(fig_cm, use_container_width=False)
            plt.close(fig_cm)
            st.caption("Rows show actual classes; columns show predicted classes.")
        else:
            st.info("A 2 × 2 confusion matrix is not available for this classifier.")
    else:
        st.info("No evaluation results found. Run the model training script first.")

# ---------- About ----------
with about_tab:
    st.markdown('<div class="section-eyebrow">Project documentation</div>', unsafe_allow_html=True)
    st.subheader("About this mini project")
    st.write(
        "CardioPredict is an academic demonstration of a supervised binary-classification workflow. "
        "It compares Logistic Regression and Random Forest using the UCI Cleveland Heart Disease dataset."
    )
    a, b = st.columns(2)
    with a:
        st.markdown("#### Workflow")
        st.markdown(
            "1. Load the public dataset and inspect its attributes.  \n"
            "2. Prepare numerical and categorical features.  \n"
            "3. Train Logistic Regression and Random Forest.  \n"
            "4. Evaluate the models on a held-out test split.  \n"
            "5. Save the selected model and metrics.  \n"
            "6. Demonstrate predictions through this Streamlit interface."
        )
    with b:
        st.markdown("#### Technology stack")
        st.markdown(
            "- **Language:** Python  \n"
            "- **ML library:** Scikit-learn  \n"
            "- **Data handling:** Pandas and NumPy  \n"
            "- **Visualizations:** Matplotlib  \n"
            "- **Web interface:** Streamlit  \n"
            "- **Dataset:** UCI Cleveland Heart Disease"
        )
    st.markdown(
        '<div class="info-strip"><b>Dataset target coding:</b> Original target 0 is mapped to class 0; '
        'original targets 1–4 are mapped to class 1 in this project.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        '<div class="notice"><b>Limitations:</b> The dataset is small and historical. '
        'Test-set performance does not establish clinical validity, generalizability, or safety. '
        'This application is intended only for academic demonstration.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">CARDIOPREDICT · BE CSE MINI PROJECT · MACHINE-LEARNING ACADEMIC DEMONSTRATION</div>',
    unsafe_allow_html=True,
)
