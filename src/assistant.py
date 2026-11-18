"""CardioPredict educational AI and voice assistant."""
import os
import json
import streamlit as st
import streamlit.components.v1 as components

MODEL_NAME = "gemini-2.5-flash"
FIELD_GUIDE = """
13 historical Cleveland inputs:
age: age in years.
sex: original dataset code 0 or 1; do not infer identity from this code.
cp: chest pain category code 1 typical angina, 2 atypical angina,
3 non-anginal pain, 4 asymptomatic in the dataset.
trestbps: resting blood pressure in mm Hg.
chol: serum cholesterol in mg/dL.
fbs: fasting blood sugar above 120 mg/dL (0=no, 1=yes).
restecg: resting ECG category (0 normal, 1 ST-T abnormality, 2 LV hypertrophy).
thalach: maximum heart rate achieved during exercise test, bpm.
exang: exercise-induced angina recorded during test (0=no, 1=yes).
oldpeak: exercise-induced ST depression relative to rest.
slope: peak exercise ST slope (1 upsloping, 2 flat, 3 downsloping).
ca: number of major vessels recorded by fluoroscopy (0-3).
thal: historical thal category (3 normal, 6 fixed defect, 7 reversible defect).
Original target 0 maps to class 0; original targets 1-4 map to class 1.
These codes and classes are dataset labels, not a medical diagnosis.
"""
INSTRUCTIONS = """
You are CardioPredict's educational assistant. Explain the app, all 13 fields,
the Cleveland dataset, Logistic Regression, Random Forest, preprocessing,
metrics and model outputs in plain language.
Safety:
- This is a small historical UCI Cleveland academic prototype, not clinically
  validated and not validated for Indian patients.
- Never diagnose, rule out disease, prescribe, or claim true personal risk.
  Model probability is not calibrated clinical risk.
- Do not encourage users to guess clinical measurements; use clinical reports
  and consult qualified healthcare professionals.
- Severe/new chest pressure, pain spreading to arm/jaw, severe breathing
  difficulty, fainting or collapse: seek urgent medical help now; do not wait
  for the app. In India call 112.
- Never claim this follows Indian standards or WHO risk charts.
- Never invent studies, metrics or medical facts. Do not request identifiers.
- Synthetic profiles are fictional software tests, not real patient records.
"""
def _api_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""

def _speak(text, number):
    safe = json.dumps(str(text))
    html = f"""
    <button id="s{number}" style="background:#087f8c;color:white;border:0;
    border-radius:9px;padding:9px 14px;font-weight:700;cursor:pointer">
    🔊 Speak answer</button>
    <script>
    document.getElementById("s{number}").onclick=()=>{{
      if (!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const u=new SpeechSynthesisUtterance({safe});
      u.lang="en-IN"; u.rate=0.96; window.speechSynthesis.speak(u);
    }};
    </script>
    """
    components.html(html, height=48)

def _answer(question, history, context, key):
    from google import genai
    from google.genai import types
    previous = "\n".join(
        ("User: " if x["role"] == "user" else "Assistant: ") + x["content"]
        for x in history[-10:]
    )
    prompt = f"""
Field reference:
{FIELD_GUIDE}
Optional user-approved current context:
{context or "No personal input values or prediction have been shared."}
Recent conversation:
{previous or "No previous messages."}
New question: {question}
Answer clearly and briefly, mentioning limitations where relevant.
"""
    client = genai.Client(api_key=key)
    result = client.models.generate_content(
        model=MODEL_NAME, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTIONS, temperature=0.25, max_output_tokens=650
        )
    )
    if not getattr(result, "text", None):
        raise RuntimeError("The AI service returned an empty answer.")
    return result.text.strip()

def render_cardio_assistant(current_profile=None, prediction=None):
    if "cp_chat_history" not in st.session_state:
        st.session_state.cp_chat_history = []
    if "cp_chat_count" not in st.session_state:
        st.session_state.cp_chat_count = 0

    st.markdown(
        '<div class="info-box"><b>Ask about:</b> the 13 input fields, using '
        'CardioPredict, model algorithms, metrics and prediction meaning.</div>',
        unsafe_allow_html=True
    )
    with st.expander("Privacy and safety", expanded=False):
        st.write(
            "Questions are sent to the configured Gemini API. Google's free tier "
            "may use submitted prompts to improve its products. Do not enter names, "
            "contact details, IDs or identifiable patient information. This is "
            "an educational assistant, not a medical service."
        )
        st.markdown("[Gemini API pricing and data-use details](https://ai.google.dev/gemini-api/docs/pricing)")

    share = st.checkbox(
        "Include my current 13 input values and latest prediction in this chat. "
        "I understand these values will be sent to the AI provider.",
        value=False, key="cp_share_health_values"
    )
    context = ""
    if share and current_profile:
        context = "Current 13 values: " + repr(current_profile)
        if prediction:
            context += "\nLatest model class: " + str(prediction.get("predicted"))
            context += "; model probability for class 1: " + str(prediction.get("probability"))
        st.caption("Values are shared only because you checked the consent box.")

    key = _api_key()
    if not key:
        st.warning(
            "The chat interface is in place, but it needs a Gemini API key. "
            "Create one in Google AI Studio and add it to Streamlit secrets as "
            "GEMINI_API_KEY. Never commit the key to GitHub."
        )
        st.link_button("Create a Gemini API key", "https://aistudio.google.com/app/apikey")
        return

    for i, item in enumerate(st.session_state.cp_chat_history):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _speak(item["content"], i)

    left, right = st.columns([1, 1])
    with left:
        language = st.selectbox(
            "Voice language", ["en-IN", "hi-IN", "kn-IN"],
            format_func=lambda x: {"en-IN":"English (India)","hi-IN":"Hindi","kn-IN":"Kannada"}[x],
            key="cp_voice_lang"
        )
    voice_text = None
    with right:
        try:
            from streamlit_mic_recorder import speech_to_text
            voice_text = speech_to_text(
                language=language, start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop",
                just_once=True, use_container_width=True, key="cp_voice_recognition"
            )
        except Exception:
            st.caption("Voice input unavailable in this browser. You can type instead.")

    typed = st.chat_input("Ask CardioPredict…")
    question = (voice_text or typed or "").strip()
    if voice_text:
        st.info("Recognized speech: " + voice_text)

    if question:
        if st.session_state.cp_chat_count >= 30:
            st.warning("The 30-question demo limit for this session has been reached.")
            return
        st.session_state.cp_chat_history.append({"role":"user","content":question})
        st.session_state.cp_chat_count += 1
        with st.spinner("Preparing an explanation…"):
            try:
                answer = _answer(question, st.session_state.cp_chat_history[:-1], context, key)
            except Exception as exc:
                answer = "I couldn't reach the AI service. Check the API key, internet connection and free-tier quota. Detail: " + str(exc)[:300]
            st.session_state.cp_chat_history.append({"role":"assistant","content":answer})
        st.rerun()

    st.caption(f"Questions this session: {st.session_state.cp_chat_count}/30. Spoken answers use browser speech synthesis.")
    if st.button("Clear conversation", key="cp_clear_conversation"):
        st.session_state.cp_chat_history = []
        st.session_state.cp_chat_count = 0
        st.rerun()
