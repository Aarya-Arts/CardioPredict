# CardioPredict — New Dataset Audit

## Dataset identity
- File: `Cardiovascular_Disease_Dataset.csv`
- Source: Doppala, Bhanu Prakash; Bhattacharyya, Debnath (2021), *Cardiovascular_Disease_Dataset*, Mendeley Data, Version 1.
- DOI: `10.17632/dzz48mvjht.1`
- Licence: CC BY 4.0 (confirm attribution in the project report).
- Source description: 1,000 subjects and 12 predictor features.
- This is a dataset described as acquired from one multispecialty hospital in India; it should not be treated as representative of all populations or as clinically validated.

## Structure confirmed from the uploaded file
- Raw rows: 1000
- Raw columns: 14
- Columns: patientid, age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar, restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels, target
- Blank/null cells in the raw CSV: 0
- Duplicate rows: 0
- Duplicate patient IDs: 0
- Target counts (0 = absence, 1 = presence): {"0": 420, "1": 580}

`patientid` is an identifier, not a model feature. It must be excluded from training to avoid identifier memorization.

## Data-quality findings and adopted policy
1. `slope=0`: 180 records. The source description defines slope categories as 1=upsloping, 2=flat, 3=downsloping. These rows are excluded from the cleaned training dataset.
2. `serumcholestrol=0`: 53 records. Zero is not a plausible measured cholesterol concentration. Values are replaced with missing (`NaN`) in the cleaned training CSV and imputed using the median inside the model pipeline, fitted separately within each training fold. This avoids calculating imputation from the test fold.
3. Nonzero cholesterol values outside the source-described interval 126–564 mg/dL: 12 records; values are {"85": 1, "86": 4, "87": 2, "601": 3, "602": 2}. They are retained provisionally but flagged for review rather than silently discarded. Confirm them against the original source before final submission.
4. The cleaned working CSV contains 820 rows. Target counts after slope validation: {"0": 240, "1": 580}.

No clinical diagnosis or risk thresholds are inferred from these records.

## Candidate model run (local experiment)
A stratified 80/20 holdout split (random state 42) and 5-fold stratified cross-validation on the training partition were used. Preprocessing is fitted inside each model pipeline, and `patientid` is excluded. Three candidate algorithms were compared by mean cross-validation ROC-AUC.

| Candidate | Mean CV ROC-AUC | Holdout accuracy | Precision | Recall | F1-score | Holdout ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting (selected) | 0.9929 | 0.9817 | 1.0000 | 0.9741 | 0.9869 | 0.9980 |
| Random Forest | 0.9911 | — | — | — | — | — |
| Logistic Regression | 0.9860 | — | — | — | — | — |

The selected candidate was Gradient Boosting. These values are from an internal random holdout on this same small dataset and are unusually high. They must not be described as real-world or clinical accuracy. In this file, `slope=3` occurs only with target 1, and `slope=0` occurs only with target 0 before the invalid-code rows are excluded. This strong association may reflect source-label construction or encoding artifacts. The dataset is from one hospital and has no independent external validation in this experiment.

## Safe integration approach
- Keep the existing Cleveland dataset, `models/cvd_model.joblib`, `models/metrics.json`, and current app unchanged until this candidate is evaluated.
- Run `python src/train_new_dataset.py` from the project root after copying this kit into the project.
- The script selects among Logistic Regression, Random Forest and Gradient Boosting based on 5-fold ROC-AUC on the training partition; it reports the held-out test metrics.
- It saves only `models/cvd_model_new_dataset_candidate.joblib` and `models/cvd_metrics_new_dataset_candidate.json`.
- Do not rename the candidate over the current model until `app.py` is updated to use the new 12-feature schema. The existing Cleveland form/model likely expects a different 13-feature schema (including a `thal` input absent from this dataset). The UI and prediction input mapping must be adapted together.

## Attribution
Doppala, B. P., & Bhattacharyya, D. (2021). *Cardiovascular_Disease_Dataset* (Version 1). Mendeley Data. https://doi.org/10.17632/dzz48mvjht.1

## Disclaimer
CardioPredict is an educational prototype. Its model output is not a medical diagnosis, a substitute for a clinician, or a validated clinical decision-support tool.
