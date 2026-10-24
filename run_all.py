"""One command end-to-end: SQL cohort analytics + odds ratios -> model comparison, calibration, threshold, SHAP, fairness -> RESULTS.md"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import analytics  # noqa: E402
import model  # noqa: E402

t0 = time.time()
print("== Stage 1: population analytics ==", flush=True)
analytics.main()
print("\n== Stage 2: modelling (several minutes: 5 models x 3-fold CV on 256K rows) ==", flush=True)
model.main()

a = json.loads((ROOT / "reports" / "analytics_summary.json").read_text())
m = json.loads((ROOT / "reports" / "model_summary.json").read_text())
op, lr, d05 = m["operating_point"], m["lr_same_recall"], m["default_0_5"]
synthetic = (ROOT / "data" / "raw" / ".synthetic").exists()
cv_rows = "\n".join(f"| {r['model']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['brier']:.4f} | {r['recall_at_0.5']:.0%} |" for r in m["cv"])
tier_rows = "\n".join(f"| {t['tier']} | {t['population_share']:.0%} | {t['case_share']:.0%} | {t['observed']:.1%} | {t['mean_predicted']:.1%} |"
                      for t in m["tiers"])
vh = next(t for t in m["tiers"] if t["tier"] == "Very high")
hi = next(t for t in m["tiers"] if t["tier"] == "High")

auc_gap = m["test_auc"] - m["test_auc_lr"]
if m["flagged_reduction_vs_lr_pct"] >= 3:
    lr_line = (f"Same recall with logistic regression flags {lr['flagged_share']:.0%} → LightGBM flags "
               f"**{m['flagged_reduction_vs_lr_pct']:.0f}% fewer** people for follow-up")
    lr_bullet = f", flagging {m['flagged_reduction_vs_lr_pct']:.0f}% fewer people than logistic regression at equal recall"
else:
    lr_line = (f"Logistic regression ranks almost as well (AUC gap {auc_gap:+.3f}): risk here is close to additive. "
               "LightGBM is kept for its calibrated probabilities and captured interactions, not for raw AUC")
    lr_bullet = ""
young = m["fair_recall_by_age"].get("18-44", float("nan"))

results = f"""# Results{' (synthetic data — rerun on the real Kaggle file before quoting these numbers)' if synthetic else ''}

## Population analytics ({a['rows']:,} respondents, {a['prevalence_pct']:.2f}% prevalence)
- Each decade of age multiplies the odds of heart disease by **{a['or_per_decade_of_age']}** (adjusted).
- Strongest adjusted factor: **{a['top_adjusted_factor']}** (OR {a['top_adjusted_or']}).
- **Confounding:** stroke's crude OR {a['stroke_crude_or']} → {a['stroke_adjusted_or']} adjusted; difficulty walking {a['diffwalk_crude_or']} → {a['diffwalk_adjusted_or']}. Much of the raw association is age.
- Prevalence rises from {a['comorb_0_prev']:.1f}% with no major conditions to {a['comorb_max_prev']:.1f}% with 3+ (**{a['comorb_multiple']}×**).

## Model comparison (3-fold stratified CV, {m['n_train_val']:,} rows)
| Model | ROC-AUC | PR-AUC | Brier | Recall @ 0.5 |
|---|---|---|---|---|
{cv_rows}

Reweighting and SMOTE do not improve ranking (SMOTE vs plain LightGBM AUC Δ = {m['smote_vs_plain_auc']:+.4f}); they only shift
probabilities. So the final model is **plain LightGBM + isotonic calibration**, with recall handled by the threshold.

## Final model on the untouched test set ({m['n_test']:,} rows)
- ROC-AUC **{m['test_auc']}** (logistic regression {m['test_auc_lr']}), PR-AUC **{m['test_pr_auc']}** ({m['test_pr_auc'] / m['prevalence']:.1f}× random)
- Calibration: Brier {m['brier_calibrated']} calibrated vs {m['brier_weighted']} class-weighted; mean predicted risk {m['mean_pred_calibrated']:.1%} vs {m['mean_pred_weighted']:.1%} (true {m['prevalence']:.1%})
- Default 0.5 threshold catches only **{d05['recall']:.0%}** of cases → screening threshold **{m['threshold']:.3f}** chosen on validation:
  recall **{op['recall']:.0%}**, precision {op['precision']:.0%}, specificity {op['specificity']:.0%}, flags {op['flagged_share']:.0%} of people
- {lr_line}
- Top 10% highest-risk capture **{m['top10_capture']:.0%}** of cases; top 20% capture **{m['top20_capture']:.0%}**

| Risk tier | % of people | % of cases | Observed rate | Mean predicted |
|---|---|---|---|---|
{tier_rows}

Top SHAP drivers: {', '.join(m['top_shap'])}.

Fairness (recall at the threshold): sex {m['fair_recall_by_sex']}, age {m['fair_recall_by_age']}, race groups (n ≥ 500)
{m['fair_race_recall_range'][0]:.0%}–{m['fair_race_recall_range'][1]:.0%}. Race is excluded from the features and used only for this audit.
**Finding:** one global threshold gives only {young:.0%} recall for 18–44s, because their base rate is low. Recommended
fix: age-band-specific thresholds (or a separate young-adult programme), evaluated with the same audit.

## Resume bullets (auto-generated from the numbers above)
- Analysed **{a['rows'] / 1000:.0f}K CDC BRFSS survey records** with SQL cohort analysis and adjusted logistic regression; quantified risk drivers (OR {a['or_per_decade_of_age']} per decade of age, stroke OR {a['stroke_adjusted_or']}) and showed how age confounding inflates crude associations.
- Compared logistic regression, random forest and LightGBM with class weighting and SMOTE under stratified CV; selected a **calibrated LightGBM (ROC-AUC {m['test_auc']:.2f}, PR-AUC {m['test_pr_auc']:.2f}, Brier {m['brier_calibrated']:.3f} vs {m['brier_weighted']:.3f} class-weighted)** after finding SMOTE and reweighting distorted probabilities without improving ranking.
- Tuned the decision threshold for screening: **{op['recall']:.0%} recall** vs {d05['recall']:.0%} at the default{lr_bullet}; top 20% of risk scores captured {m['top20_capture']:.0%} of cases.
- Built risk tiers (top tier: {vh['observed']:.0%} observed rate, {vh['case_share']:.0%} of cases from {vh['population_share']:.0%} of people), SHAP explanations and a fairness audit across sex, age and race that surfaced a young-adult recall gap; deployed a Streamlit risk calculator with per-person driver explanations.
"""
(ROOT / "RESULTS.md").write_text(results)
print(f"\nDone in {time.time() - t0:.0f}s -> RESULTS.md. Launch: streamlit run app.py")
