"""Safely update the local AI Guide's dataset context after the model migration.

This script edits only FIELD_GUIDE and two dataset-specific instruction passages.
It deliberately preserves the user's local Gemini and browser speech changes.
"""
from pathlib import Path
from datetime import datetime
import shutil
import sys

path = Path(__file__).resolve().with_name("assistant.py")
if not path.exists():
    print(f"Cannot find assistant module: {path}")
    raise SystemExit(1)

source = path.read_text(encoding="utf-8")
start_marker = 'FIELD_GUIDE = """'
start = source.find(start_marker)
if start < 0:
    print("FIELD_GUIDE block not found; no changes made.")
    raise SystemExit(1)

content_start = start + len(start_marker)
end = source.find('"""', content_start)
if end < 0:
    print("FIELD_GUIDE closing marker not found; no changes made.")
    raise SystemExit(1)

new_guide = '''FIELD_GUIDE = """
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
"""'''

source = source[:start] + new_guide + source[end + 3:]

old_intro = """You are CardioPredict's educational assistant. Explain the app, its 13 input
fields, the Cleveland dataset, Logistic Regression, Random Forest, preprocessing,
evaluation metrics and model outputs in clear, student-friendly language."""
new_intro = """You are CardioPredict's educational assistant. Explain the app, its 12 input
fields, the Mendeley Cardiovascular Disease Dataset, the selected candidate
classifier, preprocessing, evaluation metrics and model outputs in clear,
student-friendly language."""
if source.count(old_intro) != 1:
    print("Expected AI Guide introduction was not found exactly once; no changes made.")
    raise SystemExit(1)
source = source.replace(old_intro, new_intro, 1)

old_safety = """- This is a small historical UCI Cleveland academic prototype. It is not
  clinically validated and has not been validated for Indian patients."""
new_safety = """- This is a small educational prototype trained on a dataset described as
  coming from one multispecialty hospital in India. It is not clinically
  validated and must not be represented as validated for any population."""
if source.count(old_safety) != 1:
    print("Expected medical-safety passage was not found exactly once; no changes made.")
    raise SystemExit(1)
source = source.replace(old_safety, new_safety, 1)

# Confirm the user's local voice/Gemini code is still present before writing.
if "def _speak(" not in source or "def _answer(" not in source:
    print("Expected speech or Gemini function missing; no changes made.")
    raise SystemExit(1)

backup = path.with_name(
    f"assistant.py.before-dataset-context-{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
)
shutil.copy2(path, backup)
path.write_text(source, encoding="utf-8")

try:
    compile(source, str(path), "exec")
except SyntaxError as exc:
    shutil.copy2(backup, path)
    print(f"Syntax check failed; original restored from {backup.name}: {exc}")
    raise SystemExit(1)

print("AI Guide dataset context updated.")
print(f"Backup created: {backup.name}")
print("Your local Gemini and speech functions were preserved.")
