"""CardioPredict educational AI and voice assistant."""
import json
import os
import re
import time

import streamlit as st
import streamlit.components.v1 as components

# Load a local .env file when running in VS Code. Streamlit Cloud uses Secrets.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "gemini-3.8-flash"
FALLBACK_MODEL_NAME = "gemini-3.5-flash-lite"

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
You are CardioPredict's educational assistant. Explain the app, its 13 input
fields, the Cleveland dataset, Logistic Regression, Random Forest, preprocessing,
evaluation metrics and model outputs in clear, student-friendly language.

Medical and privacy safety:
- This is a small historical UCI Cleveland academic prototype. It is not
  clinically validated and has not been validated for Indian patients.
- Never diagnose, rule out disease, prescribe medicines or claim true personal
  risk. The model's output/probability is not calibrated clinical risk.
- Do not encourage users to guess clinical measurements. Recommend using
  actual reports and discussing concerns with qualified healthcare professionals.
- Severe or new chest pressure, pain spreading to the arm/jaw, severe breathing
  difficulty, fainting or collapse: seek urgent medical help now; do not wait
  for this app. In India call 112.
- Never claim this follows Indian standards or WHO risk charts.
- Never invent studies, metrics, citations or medical facts. Do not request
  names, contact details, IDs or other identifying information.
- Synthetic profiles are fictional software tests, not real patient records.
"""

LANGUAGE_NAMES = {
    "en-IN": "English (India)",
    "hi-IN": "Hindi in Devanagari script",
    "kn-IN": "Kannada in Kannada script",
}


def _api_key():
    """Read the key from local environment or Streamlit Cloud Secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def _speak(text, number, language="en-IN"):
    """Speak clean text rather than Markdown symbols."""
    speech_text = str(text)
    speech_text = re.sub(r"```[\\s\\S]*?```", " ", speech_text)
    speech_text = re.sub(r"!?\\[([^\\]]+)\\]\\([^)]+\\)", r"\\1", speech_text)
    speech_text = re.sub(r"^\\s{0,3}#{1,6}\\s*", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"(\\*\\*|__)(.*?)\\1", r"\\2", speech_text)
    speech_text = re.sub(r"(?<!\\w)[*_~`]", "", speech_text)
    speech_text = re.sub(r"^\\s*[-+*]\\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"^\\s*\\d+[.)]\\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"\\n{2,}", ". ", speech_text)
    speech_text = re.sub(r"\\s+", " ", speech_text).strip()
    safe = json.dumps(speech_text)
    html = f"""
    <button id="s{number}" style="background:#BD5969;color:white;border:0;
    border-radius:9px;padding:9px 14px;font-weight:700;cursor:pointer">
    🔊 Speak answer</button>
    <script>
    document.getElementById("s{number}").onclick=()=>{{
      if (!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const u=new SpeechSynthesisUtterance({safe});
      u.lang={json.dumps(language)};
      u.rate=0.96;
      window.speechSynthesis.speak(u);
    }};
    </script>
    """
    components.html(html, height=48)


def _answer(question, history, context, key, language):
    """Ask Gemini, falling back to a second Flash model on temporary outages."""
    from google import genai
    from google.genai import types

    previous = "\n".join(
        ("User: " if item["role"] == "user" else "Assistant: ") + item["content"]
        for item in history[-10:]
    )
    language_name = LANGUAGE_NAMES.get(language, "English (India)")
    prompt = f"""
RESPONSE LANGUAGE — IMPORTANT:
Write the entire answer in {language_name}.
- en-IN: use natural, clear English.
- hi-IN: use natural Hindi in Devanagari script.
- kn-IN: use natural Kannada in Kannada script.
Do not switch to English unless the user explicitly asks for English.
Keep medical and technical explanations simple. Preserve standard technical
terms in English only when that makes the explanation clearer, and explain them
in the selected language.

Field reference:
{FIELD_GUIDE}

Optional user-approved current context:
{context or "No personal input values or prediction have been shared."}

Recent conversation:
{previous or "No previous messages."}

New question:
{question}

Answer clearly and briefly. Follow all medical, privacy and prototype limits.
"""
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        max_output_tokens=650,
    )

    models = (MODEL_NAME, FALLBACK_MODEL_NAME)
    for model_index, model_name in enumerate(models):
        # Retry temporary service overloads on the primary model. If it remains
        # unavailable, try the backup model while preserving the same prompt.
        attempts = 3 if model_index == 0 else 2
        for attempt in range(attempts):
            try:
                result = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                answer = getattr(result, "text", None)
                if not answer or not answer.strip():
                    raise RuntimeError(
                        f"{model_name} returned an empty answer."
                    )
                return answer.strip()
            except Exception as exc:
                message = str(exc).upper()
                temporary = any(code in message for code in (
                    "503", "UNAVAILABLE", "INTERNAL SERVER ERROR", "500"
                ))
                if not temporary:
                    raise
                if attempt < attempts - 1:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                if model_index == len(models) - 1:
                    raise
                # Move to the backup model after the primary's retries fail.

    raise RuntimeError("Both Gemini models are temporarily unavailable.")


def render_cardio_assistant(current_profile=None, prediction=None):
    if "cp_chat_history" not in st.session_state:
        st.session_state.cp_chat_history = []
    if "cp_chat_count" not in st.session_state:
        st.session_state.cp_chat_count = 0

    st.markdown(
        '<div class="info-box"><b>HEART & LIFE · AI GUIDE</b><br>'
        'Ask about the 13 input fields, heart-health concepts, model algorithms, '
        'metrics and what a CardioPredict result means.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Privacy and safety", expanded=False):
        st.write(
            "Typed questions are sent to the configured Gemini API. Voice input "
            "is processed by the browser speech-recognition component. Google's "
            "free tier may use submitted prompts to improve its products. Do not "
            "enter names, contact details, IDs or identifiable patient information. "
            "This is an educational prototype, not a medical service."
        )
        st.markdown(
            "[Gemini API data-use details](https://ai.google.dev/gemini-api/docs/usage-policies)"
        )

    share = st.checkbox(
        "Include my current 13 input values and latest prediction in this chat. "
        "I understand these values will be sent to the AI provider.",
        value=False,
        key="cp_share_health_values",
    )
    context = ""
    if share and current_profile:
        context = "Current 13 values: " + repr(current_profile)
        if prediction:
            context += "\nLatest model class: " + str(prediction.get("predicted"))
            context += "; model probability for class 1: " + str(
                prediction.get("probability")
            )
        st.caption("Values are shared only because you checked the consent box.")

    key = _api_key()
    if not key:
        st.warning(
            "The assistant is ready, but no Gemini API key was found. Add "
            "GEMINI_API_KEY to Streamlit Cloud Secrets or your local .env file. "
            "Never commit the key to GitHub."
        )
        st.link_button(
            "Create a Gemini API key",
            "https://aistudio.google.com/app/apikey",
        )
        return

    left, right = st.columns([1, 1])
    with left:
        language = st.selectbox(
            "Response and voice language",
            ["en-IN", "hi-IN", "kn-IN"],
            format_func=lambda value: {
                "en-IN": "English (India)",
                "hi-IN": "Hindi",
                "kn-IN": "Kannada",
            }[value],
            key="cp_voice_lang",
        )
    voice_text = None
    with right:
        try:
            from streamlit_mic_recorder import speech_to_text
            voice_text = speech_to_text(
                language=language,
                start_prompt="🎙️ Speak",
                stop_prompt="⏹️ Stop",
                just_once=True,
                use_container_width=True,
                key="cp_voice_recognition",
            )
        except Exception:
            st.caption(
                "Voice input is unavailable in this browser. You can still type."
            )

    for index, item in enumerate(st.session_state.cp_chat_history):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _speak(item["content"], index, language)

    typed = st.chat_input("Ask CardioPredict…")
    question = (voice_text or typed or "").strip()
    if voice_text:
        st.info("Recognized speech: " + voice_text)

    if question:
        if st.session_state.cp_chat_count >= 30:
            st.warning("The 30-question demo limit for this session has been reached.")
            return

        st.session_state.cp_chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.cp_chat_count += 1

        with st.spinner("Preparing your Heart & Life explanation…"):
            try:
                answer = _answer(
                    question,
                    st.session_state.cp_chat_history[:-1],
                    context,
                    key,
                    language,
                )
            except Exception as exc:
                detail = str(exc)[:240]
                answer = (
                    "I couldn't get a response from Gemini right now. Please try "
                    "again shortly. If the problem continues, check the API key "
                    "and Gemini API quota. Technical detail: " + detail
                )

        st.session_state.cp_chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    st.caption(
        f"Questions this session: {st.session_state.cp_chat_count}/30. "
        "Spoken answers use your browser's available voice. Voice availability "
        "varies by device and installed language packs."
    )
    if st.button("Clear conversation", key="cp_clear_conversation"):
        st.session_state.cp_chat_history = []
        st.session_state.cp_chat_count = 0
        st.rerun()
