# CardioPredict – HEART & LIFE

## Project overview

CardioPredict is an academic Streamlit prototype for exploring a cardiovascular dataset and demonstrating a machine-learning classification workflow. This branch uses the **Cardiovascular Disease Dataset** published on Mendeley Data by Doppala and Bhattacharyya (2021).

The source dataset contains 1,000 records. After the documented data checks (including excluding records with an undocumented ST-slope value), the saved candidate workflow contains **820 records**. The patient identifier is excluded, leaving **12 model predictor features**.

**Important:** This project is an educational prototype, not a medical device or clinically validated risk calculator. Its predictions and internal test metrics must not be used for diagnosis, screening, treatment decisions or emergency triage.

## Current model artifacts

The app loads the Mendeley candidate artifacts already stored in `models/`:

- `models/cvd_model_new_dataset_candidate.joblib` – saved preprocessing and Gradient Boosting classifier pipeline.
- `models/cvd_metrics_new_dataset_candidate.json` – dataset metadata, data-check counts, five-fold training-partition comparison and held-out test metrics.
- `models/cvd_model.joblib` and `models/metrics.json` – older Cleveland workflow artifacts retained as backup; they are not used by this branch's active app.

The selected candidate is **Gradient Boosting**. The saved held-out test split contains 164 records and reports accuracy 98.17%, precision 100.00%, recall 97.41%, F1-score 98.69% and ROC-AUC 99.80%. These are internal results on one dataset split, not estimates of clinical or real-world performance.

## Application modules

- **Home** – project overview and selected-model summary.
- **Predict** – accepts the 12 dataset predictor fields and runs the saved model pipeline.
- **Scenario Lab** – generates fictional profiles for software testing and offers a CSV export.
- **Analytics** – displays held-out test metrics, five-fold candidate comparison and the confusion matrix.
- **Heart Guide** – explains the dataset fields and general heart-health concepts.
- **Emergency & Map** – provides emergency guidance and opens a Google Maps hospital search.
- **AI Guide** – optional Gemini-powered educational assistant with text, voice input and spoken responses.
- **About** – dataset source, workflow, technology and limitations.

## Technology

Python, Streamlit, Pandas, NumPy, scikit-learn, Joblib and Matplotlib. Supabase provides email authentication; Google Gemini powers the optional AI Guide.

## Setup (Windows)

Use **Python 3.11 or newer**. The saved model was produced with scikit-learn 1.9.1, which is pinned in `requirements.txt`.

1. Clone or download this repository and open its project folder.
2. Create a virtual environment: `py -3.11 -m venv .venv`.
3. Activate it: `.venv\\Scripts\\activate`.
4. Install dependencies: `python -m pip install -r requirements.txt`.
5. Configure the required service credentials in Streamlit secrets (local `.streamlit/secrets.toml`, or the Streamlit Cloud app settings):

   ```toml
   SUPABASE_URL = "your-supabase-project-url"
   SUPABASE_ANON_KEY = "your-supabase-anon-or-publishable-key"
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

   The Supabase URL and anon/publishable key are required for the sign-in gate. The Gemini key enables the AI Guide. Keep secrets out of source control; never use a Supabase service-role key in this app.

6. Launch the application: `streamlit run app.py`.

**Do not run `src/train_model.py` to rebuild this Mendeley model.** That file belongs to the earlier Cleveland-dataset workflow and creates the older backup artifacts. The current Mendeley candidate model and its metrics are already saved in the repository and are loaded directly by `app.py`.

## Dataset reference

Doppala, B. P., & Bhattacharyya (2021). *Cardiovascular Disease Dataset*, Mendeley Data, Version 1. [https://doi.org/10.17632/dzz48mvjht.1](https://doi.org/10.17632/dzz48mvjht.1). License: CC BY 4.0.

## Before submission

- Confirm that Streamlit Community Cloud deploys the `redesign/mendeley-dataset-primary` branch.
- Test authentication, prediction, batch generation, CSV download, AI Guide and mobile layout in the deployed app.
- Replace presentation placeholders with the actual team, USNs and guide details.
- Describe all reported metrics as internal held-out dataset results; do not claim clinical validation.
