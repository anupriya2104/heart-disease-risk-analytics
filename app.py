"""
Streamlit app: individual risk calculator (calibrated probability, tier, top drivers) + population dashboard.
Run:  streamlit run app.py      (after `python run_all.py`)
Educational project — not a medical device and not medical advice.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from data import AGE_MID, GEN_ORD  # noqa: E402

R = ROOT / "reports"
BLUE, ORANGE = "#2a78d6", "#eb6834"
st.set_page_config(page_title="Heart Disease Risk Analytics", layout="wide")
if not (ROOT / "models" / "heart_risk.joblib").exists():
    st.error("Run `python run_all.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    m = joblib.load(ROOT / "models" / "heart_risk.joblib")
    return m, shap.TreeExplainer(m["booster"])


@st.cache_data
def load_reports():
    return (json.loads((R / "analytics_summary.json").read_text()), json.loads((R / "model_summary.json").read_text()),
            pd.read_csv(R / "sql_prevalence_by_age_sex.csv"), pd.read_csv(R / "odds_ratios.csv"),
            pd.read_csv(R / "risk_tiers.csv"), pd.read_csv(R / "fairness_audit.csv"), pd.read_csv(R / "cv_model_comparison.csv"))


M, explainer = load_model()
a, m, age_sex, orr, tiers, fair, cv = load_reports()
op = m["operating_point"]

st.title("Heart Disease Risk Prediction & Population Health Analytics")
st.caption(f"{a['rows']:,} CDC BRFSS survey respondents · {a['prevalence_pct']:.1f}% report heart disease · "
           "educational project, not medical advice")
k = st.columns(5)
k[0].metric("Test ROC-AUC", f"{m['test_auc']:.3f}")
k[1].metric("PR-AUC", f"{m['test_pr_auc']:.3f}", f"{m['test_pr_auc'] / m['prevalence']:.1f}× random")
k[2].metric("Recall at screening threshold", f"{op['recall']:.0%}")
k[3].metric("Population flagged", f"{op['flagged_share']:.0%}")
k[4].metric("Top 20% risk holds", f"{m['top20_capture']:.0%} of cases")

tab1, tab2, tab3 = st.tabs(["Risk calculator", "Population insights", "Model & fairness report"])

with tab1:
    c1, c2, c3 = st.columns(3)
    age_cat = c1.selectbox("Age", list(AGE_MID), index=8)
    sex = c1.radio("Sex", ["Female", "Male"], horizontal=True)
    bmi = c1.slider("BMI", 15.0, 50.0, 27.0, 0.5)
    gen = c1.select_slider("Self-rated general health", list(GEN_ORD), value="Good")
    diab = c2.selectbox("Diabetes", ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"])
    stroke = c2.checkbox("Ever had a stroke")
    kidney = c2.checkbox("Kidney disease")
    walk = c2.checkbox("Difficulty walking / climbing stairs")
    smoke = c2.checkbox("Smoked 100+ cigarettes in life")
    active = c3.checkbox("Physically active in last 30 days", value=True)
    alcohol = c3.checkbox("Heavy drinker")
    asthma = c3.checkbox("Asthma")
    skin = c3.checkbox("Skin cancer")
    phys = c3.slider("Days of poor physical health (last 30)", 0, 30, 0)
    ment = c3.slider("Days of poor mental health (last 30)", 0, 30, 0)
    sleep = c3.slider("Sleep hours per night", 3, 12, 7)

    row = {"age": AGE_MID[age_cat], "male": int(sex == "Male"), "bmi": bmi, "gen_health": GEN_ORD[gen],
           "physical_health_days": phys, "mental_health_days": ment, "sleep_hours": sleep,
           "diabetes_yes": int(diab == "Yes"), "diabetes_borderline": int(diab == "No, borderline diabetes"),
           "diabetes_pregnancy": int(diab == "Yes (during pregnancy)"), "smoking": int(smoke), "alcoholdrinking": int(alcohol),
           "stroke": int(stroke), "diffwalking": int(walk), "physicalactivity": int(active), "asthma": int(asthma),
           "kidneydisease": int(kidney), "skincancer": int(skin)}
    X = pd.DataFrame([row])[M["features"]]
    risk = float(M["calibrated"].predict_proba(X)[0, 1])
    tier = next(n for lo, hi, n in M["tiers"] if lo <= risk < hi)
    st.divider()
    o1, o2, o3 = st.columns(3)
    o1.metric("Estimated probability", f"{risk:.1%}", f"{risk / m['prevalence']:.1f}× population average", delta_color="off")
    o2.metric("Risk tier", tier)
    o3.metric("Screening flag", "Refer for check-up" if risk >= M["threshold"] else "Routine",
              help=f"Threshold {M['threshold']:.3f}, chosen so the model catches ~{op['recall']:.0%} of cases.")
    sv = explainer.shap_values(X)
    sv = (sv[1] if isinstance(sv, list) else sv)[0]
    contrib = pd.DataFrame({"factor": X.columns, "value": X.iloc[0].values, "shap": sv})
    contrib = contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index).head(8)[::-1]
    contrib["direction"] = np.where(contrib.shap > 0, "Raises risk", "Lowers risk")
    fig = px.bar(contrib, x="shap", y="factor", color="direction", orientation="h",
                 color_discrete_map={"Raises risk": ORANGE, "Lowers risk": BLUE},
                 title="What drives this estimate (SHAP, log-odds)", hover_data=["value"])
    st.plotly_chart(fig.update_layout(height=380), width="stretch")
    st.caption("Not medical advice. Estimates come from self-reported survey data and show association, not causation.")

with tab2:
    l, r = st.columns(2)
    fig = px.line(age_sex, x="AgeCategory", y="prevalence_pct", color="Sex", markers=True,
                  title="Prevalence by age and sex (%)", color_discrete_sequence=[BLUE, ORANGE])
    l.plotly_chart(fig, width="stretch")
    o = orr.sort_values("adjusted_or")
    fig = px.scatter(o, x="adjusted_or", y="factor", error_x=o.ci_high - o.adjusted_or, error_x_minus=o.adjusted_or - o.ci_low,
                     log_x=True, title="Adjusted odds ratios (95% CI)", color_discrete_sequence=[BLUE])
    fig.add_scatter(x=o.crude_or, y=o.factor, mode="markers", name="Crude OR", marker=dict(color="#b9b8b2", symbol="diamond"))
    fig.add_vline(x=1, line_dash="dash", line_color="#52514e")
    r.plotly_chart(fig, width="stretch")
    st.info(f"Confounding check: stroke's crude odds ratio is {a['stroke_crude_or']} but {a['stroke_adjusted_or']} after adjusting "
            f"for age, sex and other conditions; difficulty walking drops from {a['diffwalk_crude_or']} to {a['diffwalk_adjusted_or']}. "
            f"Each decade of age multiplies the odds by {a['or_per_decade_of_age']}.")
    st.image(str(R / "figures" / "03_comorbidity_genhealth.png"))

with tab3:
    st.dataframe(cv.round(4), width="stretch")
    l, r = st.columns(2)
    l.image(str(R / "figures" / "05_calibration.png"))
    r.image(str(R / "figures" / "06_gains.png"))
    st.subheader("Risk tiers (test set)")
    st.dataframe(tiers.round(3), width="stretch")
    st.subheader("Fairness audit at the screening threshold")
    st.dataframe(fair.round(3), width="stretch")
    st.image(str(R / "figures" / "10_fairness.png"))
