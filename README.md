# Heart Disease Risk Prediction & Population Health Analytics

**Problem.** Given a short health survey, which people should a screening programme invite for a cardiac
check-up? Missing a real case is far worse than an extra check-up, the outcome is rare (~8.6%), and
the people running the programme need to understand and trust the scores.

**What this project does** — analytics and data science in one pipeline:

| Stage | Analytics side | Data-science side |
|---|---|---|
| 1. Cohort analytics (`sql/analytics.sql`) | Prevalence by age × sex, crude relative risks, general health, BMI class, comorbidity burden, lifestyle, sleep | — |
| 2. Epidemiology (`src/analytics.py`) | Crude vs **adjusted odds ratios with 95% CIs** — shows how age confounds raw associations | Logistic regression (statsmodels) |
| 3. Model selection (`src/model.py`) | — | LR, RF, LightGBM × {none, class weights, SMOTE} under stratified CV; ROC-AUC, **PR-AUC**, Brier |
| 4. Decision layer | Risk tiers, cumulative gains, people flagged | **Isotonic calibration**, threshold chosen on validation for ≥ 80% recall, reported on an untouched test set |
| 5. Trust | **Fairness audit** by sex, age band and race | SHAP global + per-person explanations |
| 6. App (`app.py`) | Population dashboard | Risk calculator with tier, referral flag and driver chart |

See **[RESULTS.md](RESULTS.md)** for the numbers (regenerated on every run).

## Run it

```bash
pip install -r requirements.txt
python run_all.py          # ~10 min on a laptop (5 models x 3-fold CV on 256K rows)
streamlit run app.py
```

## Using the real Kaggle data (do this before quoting numbers)

1. Download `heart_2020_cleaned.csv` from [Kaggle — Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
   (use the **2020** file; the 2022 files have different columns)
2. Delete `data/raw/*` (including the hidden `.synthetic` marker) and put the CSV there
3. `python run_all.py`

The default run uses `src/generate_data.py`, which writes a file with the same 18 columns, size (319,795 rows)
and prevalence (8.6%), with risk factors correlated as in real populations.

## Key design decisions (interview talking points)

- **PR-AUC over accuracy.** With 8.6% positives, predicting "no" for everyone is 91% accurate. PR-AUC shows
  how well the model finds the rare positives; random PR-AUC equals the prevalence.
- **SMOTE and class weights don't fix ranking.** They change probabilities, not the order of people. Rather than
  distort the probabilities, keep them honest (calibration) and handle recall with the threshold.
- **Calibration matters for a risk score.** A class-weighted model tells a 50-year-old they're at 40% risk when the
  real rate is 10%. Isotonic calibration makes "15% risk" mean 15 in 100 similar people.
- **Threshold chosen on validation, reported on test.** Picking it on test would overstate recall.
- **Crude vs adjusted odds ratios.** Stroke and difficulty walking look far more dangerous in raw tables because
  they're common in older people. Adjusting separates the factor from age — the core skill in health analytics.
- **Race is not a feature.** It is used only to audit recall and false-positive rates across groups.
- **Association, not causation.** This is cross-sectional, self-reported survey data. The app says so.

## Project layout

```
├── run_all.py             # one-command pipeline, writes RESULTS.md
├── app.py                 # Streamlit risk calculator + dashboards
├── sql/analytics.sql      # 9 named cohort queries (UNION ALL factor table, FILTER, window shares)
├── src/
│   ├── generate_data.py   # BRFSS-format simulator (used when data/raw is empty)
│   ├── data.py            # loading, validation, encoding
│   ├── analytics.py       # SQL + crude/adjusted odds ratios + charts
│   ├── model.py           # CV comparison, calibration, threshold, gains, tiers, SHAP, fairness
│   └── viz.py             # chart style
├── reports/               # query outputs, metrics JSON, fairness audit, figures/
└── models/                # calibrated model + SHAP booster + threshold
```

*Educational project. Not a medical device and not medical advice.*
