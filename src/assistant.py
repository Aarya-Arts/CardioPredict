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
12 predictors in the current Cardiovascular Disease Dataset model:
age: age in years.
gender: dataset code 0=female, 1=male.
chestpain: 0=typical angina, 1=atypical angina,
2=non-anginal pain, 3=asymptomatic. These are dataset categories, not a symptom quiz.
restingBP: resting blood pressure in mm Hg.
serumcholestrol: serum cholesterol in mg/dL. Zero values in the source data are treated as missing during training.
fastingbloodsugar: fasting blood sugar above 120 mg/dL (0=no, 1=yes).
restingrelectro: resting ECG code (0=normal, 1=ST-T wave abnormality, 2=possible/definite LVH).
maxheartrate: maximum heart rate achieved during an exercise test, bpm.
exerciseangia: exercise-induced angina recorded during testing (0=no, 1=yes).
oldpeak: exercise-test ST depression value.
slope: peak exercise ST slope (1=upsloping, 2=flat, 3=downsloping).
noofmajorvessels: coded number of major vessels, from 0 to 3.
patientid is an identifier and is not used as a predictor.
Rows with undocumented slope code 0 are excluded from the cleaned training data.
Target 0 means the record belongs to the dataset's absence-labelled group;
target 1 means the record belongs to the dataset's presence-labelled group.
These are historical dataset labels, not a medical diagnosis or a measure of personal clinical risk.
"""

INSTRUCTIONS = """
You are CardioPredict's educational assistant. Explain the app, its 12 input
fields, the Mendeley Cardiovascular Disease Dataset, the selected candidate
classifier, preprocessing, evaluation metrics and model outputs in clear,
student-friendly language.

Emotional awareness and natural conversation:
- Notice clear emotional cues such as worry, fear, frustration, sadness, or confusion without claiming certainty about the user's feelings.
- If a cue is present, acknowledge it briefly and kindly, then answer the actual question. Do not overdo sympathy or use forced cheerfulness.
- Use natural everyday spoken Hindi in Devanagari and Kannada in Kannada script. Avoid stiff literal translations and unnecessary English.
- Keep medical explanations calm, simple, respectful, and accurate. Never diagnose emotions or medical conditions.

Medical and privacy safety:
- This is a small educational prototype trained on a dataset described as
  coming from one multispecialty hospital in India. It is not clinically
  validated and must not be represented as validated for any population.
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
    speech_text = re.sub(r"```[\s\S]*?```", " ", speech_text)
    speech_text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", speech_text)
    speech_text = re.sub(r"^\s{0,3}#{1,6}\s*", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", speech_text)
    speech_text = re.sub(r"(?<!\w)[*_~`]", "", speech_text)
    speech_text = re.sub(r"^\s*[-+*]\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"^\s*\d+[.)]\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"\n{2,}", ". ", speech_text)
    speech_text = re.sub(r"\s+", " ", speech_text).strip()
    safe = json.dumps(speech_text)
    html = f"""
    <div style="display:flex;gap:8px;align-items:center">
      <button id="s{number}" style="background:#BD5969;color:white;border:0;
      border-radius:9px;padding:9px 14px;font-weight:700;cursor:pointer">
      🔊 Speak answer</button>
      <button id="stop{number}" style="background:#F2E6E7;color:#743F4A;border:1px solid #D9BFC2;
      border-radius:9px;padding:9px 14px;font-weight:700;cursor:pointer">
      ⏹ Stop speaking</button>
    </div>
    <script>
    const speakButton = document.getElementById("s{number}");
    const stopButton = document.getElementById("stop{number}");
    speakButton.onclick=async()=>{{
      if (!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const u=new SpeechSynthesisUtterance({safe});
      u.lang={json.dumps(language)};
      const langPrefix={json.dumps(language.split("-")[0])}.toLowerCase();
      const voices = await new Promise(resolve => {{
        const current = window.speechSynthesis.getVoices();
        if (current.length) return resolve(current);

        let finished = false;
        const finish = () => {{
          if (finished) return;
          finished = true;
          resolve(window.speechSynthesis.getVoices());
        }};

        window.speechSynthesis.addEventListener(
          "voiceschanged", finish, {{ once: true }}
        );
        setTimeout(finish, 1200);
      }});

      const normalize = value =>
        (value || "").replace("_", "-").toLowerCase();
      const matches = voices.filter(v =>
        normalize(v.lang).startsWith(langPrefix)
      );
      const exact = matches.find(v =>
        normalize(v.lang) === normalize({json.dumps(language)})
      );
      const chosen = exact || matches.find(v => v.default) || matches[0];

      if (!chosen && langPrefix !== "en") {{
        alert("No voice for " + {json.dumps(language)} +
          " is available in this browser or device. The text response still works.");
        return;
      }}
      if (chosen) u.voice = chosen;
      u.rate=langPrefix==="hi" ? 0.92 : (langPrefix==="kn" ? 0.93 : 0.96);
      u.pitch=1.0;
      window.speechSynthesis.speak(u);
    }};
    stopButton.onclick=()=>{{
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }};
    </script>
    """
    components.html(html, height=58)

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
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        max_output_tokens=650,
    )

    models = (MODEL_NAME, FALLBACK_MODEL_NAME)
    for model_index, model_name in enumerate(models):
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
            quota_exhausted = any(marker in message for marker in (
                "429", "RESOURCE_EXHAUSTED", "QUOTA EXCEEDED",
                "RATE LIMIT EXCEEDED",
            ))
            temporary = any(code in message for code in (
                "503", "504", "DEADLINE_EXCEEDED", "DEADLINE EXCEEDED",
                "GATEWAY TIMEOUT", "UNAVAILABLE", "INTERNAL SERVER ERROR", "500"
            ))

            if not quota_exhausted and not temporary:
                raise

            if model_index == len(models) - 1:
                raise

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
        "Spoken answers prefer an installed voice matching your selected language. "
        "Voice naturalness depends on the browser and installed language packs."
    )
    if st.button("Clear conversation", key="cp_clear_conversation"):
        st.session_state.cp_chat_history = []
        st.session_state.cp_chat_count = 0
        st.rerun()
