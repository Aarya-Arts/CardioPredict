# Cardiovascular Disease Prediction – Mini Project

## Project overview
A classroom demonstration that compares Logistic Regression and Random Forest on the UCI Cleveland Heart Disease dataset and provides a Streamlit interface for a model estimate.

**Important:** This is not a medical device, diagnostic system, or clinically validated risk calculator. The historical dataset is small and should not be used for real patient decisions.

## Folder structure
- `src/train_model.py` – downloads, preprocesses, trains, evaluates, and saves the selected model.
- `app.py` – Streamlit prediction interface.
- `requirements.txt` – Python dependencies.
- `data/` – downloaded dataset (created when training runs).
- `models/` – trained model and evaluation metrics (created when training runs).

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP file.
3. Open Command Prompt in the extracted project folder.
4. Create a virtual environment:
   `py -m venv .venv`
5. Activate it:
   `.venv\Scripts\activate`
6. Install dependencies:
   `python -m pip install -r requirements.txt`
7. Train and evaluate:
   `python src/train_model.py`
8. Launch the app:
   `streamlit run app.py`
9. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Methodology
- Download Cleveland processed data from the UCI repository.
- Convert `?` values to missing values and coerce columns to numeric.
- Convert target 0 to class 0 and targets 1–4 to class 1.
- Split data into training and test sets (80:20, stratified, random_state=42).
- Apply median imputation and standard scaling to numeric fields.
- Apply most-frequent imputation and one-hot encoding to categorical fields.
- Compare Logistic Regression and Random Forest using accuracy, precision, recall, F1-score and confusion matrix.
- Select the model with the highest test F1-score and save it.

## Before submission
- Run the project on your own computer.
- Replace any report placeholders with your real team, college, guide, screenshots and measured results.
- Do not claim model performance until the training script has actually run.
- Keep the report's medical disclaimer.
