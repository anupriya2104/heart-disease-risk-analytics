# Results (synthetic data — rerun on the real Kaggle file before quoting these numbers)

## Population analytics (319,795 respondents, 8.59% prevalence)
- Each decade of age multiplies the odds of heart disease by **1.63** (adjusted).
- Strongest adjusted factor: **Stroke** (OR 2.97).
- **Confounding:** stroke's crude OR 5.71 → 2.97 adjusted; difficulty walking 2.8 → 1.27. Much of the raw association is age.
- Prevalence rises from 4.8% with no major conditions to 43.7% with 3+ (**9.1×**).

## Model comparison (3-fold stratified CV, 255,836 rows)
| Model | ROC-AUC | PR-AUC | Brier | Recall @ 0.5 |
|---|---|---|---|---|
| Logistic regression (balanced) | 0.807 | 0.316 | 0.1821 | 75% |
| Random forest (balanced) | 0.806 | 0.311 | 0.1719 | 73% |
| LightGBM (no weighting) | 0.808 | 0.320 | 0.0678 | 8% |
| LightGBM (class weights) | 0.808 | 0.321 | 0.1762 | 74% |
| LightGBM + SMOTE | 0.775 | 0.255 | 0.0813 | 17% |

Reweighting and SMOTE do not improve ranking (SMOTE vs plain LightGBM AUC Δ = -0.0337); they only shift
probabilities. So the final model is **plain LightGBM + isotonic calibration**, with recall handled by the threshold.

## Final model on the untouched test set (63,959 rows)
- ROC-AUC **0.8081** (logistic regression 0.8072), PR-AUC **0.3205** (3.7× random)
- Calibration: Brier 0.0678 calibrated vs 0.1773 class-weighted; mean predicted risk 8.6% vs 37.1% (true 8.6%)
- Default 0.5 threshold catches only **6%** of cases → screening threshold **0.073** chosen on validation:
  recall **80%**, precision 18%, specificity 66%, flags 38% of people
- Logistic regression ranks almost as well (AUC gap +0.001): risk here is close to additive. LightGBM is kept for its calibrated probabilities and captured interactions, not for raw AUC
- Top 10% highest-risk capture **39%** of cases; top 20% capture **59%**

| Risk tier | % of people | % of cases | Observed rate | Mean predicted |
|---|---|---|---|---|
| Low | 51% | 13% | 2.1% | 2.1% |
| Moderate | 32% | 33% | 9.0% | 9.0% |
| High | 13% | 30% | 20.7% | 20.9% |
| Very high | 5% | 24% | 42.1% | 41.8% |

Top SHAP drivers: age, gen_health, male, smoking, sleep_hours, bmi.

Fairness (recall at the threshold): sex {'Female': 0.711, 'Male': 0.86}, age {'18-44': 0.081, '45-64': 0.567, '65+': 0.943}, race groups (n ≥ 500)
79%–85%. Race is excluded from the features and used only for this audit.
**Finding:** one global threshold gives only 8% recall for 18–44s, because their base rate is low. Recommended
fix: age-band-specific thresholds (or a separate young-adult programme), evaluated with the same audit.

## Resume bullets (auto-generated from the numbers above)
- Analysed **320K CDC BRFSS survey records** with SQL cohort analysis and adjusted logistic regression; quantified risk drivers (OR 1.63 per decade of age, stroke OR 2.97) and showed how age confounding inflates crude associations.
- Compared logistic regression, random forest and LightGBM with class weighting and SMOTE under stratified CV; selected a **calibrated LightGBM (ROC-AUC 0.81, PR-AUC 0.32, Brier 0.068 vs 0.177 class-weighted)** after finding SMOTE and reweighting distorted probabilities without improving ranking.
- Tuned the decision threshold for screening: **80% recall** vs 6% at the default; top 20% of risk scores captured 59% of cases.
- Built risk tiers (top tier: 42% observed rate, 24% of cases from 5% of people), SHAP explanations and a fairness audit across sex, age and race that surfaced a young-adult recall gap; deployed a Streamlit risk calculator with per-person driver explanations.
