"""
Stage 1 - population health analytics:
  SQL cohort analysis + crude vs age/sex-adjusted odds ratios (logistic regression with 95% CIs).
The crude-vs-adjusted comparison shows how much of each factor's apparent risk is really just age.
"""
import json
import re

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import AGE_MID, FIG, REPORTS, ROOT, bmi_class, encode, load
from viz import MUTED, SERIES, plt, save

FACTORS = {"stroke": "Stroke", "kidneydisease": "Kidney disease", "diabetes_yes": "Diabetes", "diffwalking": "Difficulty walking",
           "smoking": "Smoking", "skincancer": "Skin cancer", "asthma": "Asthma", "physicalactivity": "Physically active",
           "alcoholdrinking": "Heavy drinking", "male": "Male sex"}


def run_sql(df):
    s = df.copy()
    s["hd"] = (s.HeartDisease == "Yes").astype(int)
    s["age_mid"] = s.AgeCategory.map(AGE_MID)
    s["bmi_class"] = bmi_class(s.BMI).astype(str)
    con = duckdb.connect()
    con.register("survey", s)
    blocks = re.split(r"--\s*name:\s*(\w+)", (ROOT / "sql" / "analytics.sql").read_text())[1:]
    res = {}
    for name, sql in zip(blocks[::2], blocks[1::2]):
        res[name] = con.execute(sql).df()
        res[name].to_csv(REPORTS / f"sql_{name}.csv", index=False)
    return res


def odds_ratios(X, y):
    """Crude OR (factor alone) vs adjusted OR (factor + age + sex + all other factors)."""
    rows = []
    adj = sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0)
    ci = np.exp(adj.conf_int())
    for f, label in FACTORS.items():
        crude = sm.Logit(y, sm.add_constant(X[[f]].astype(float))).fit(disp=0)
        rows.append({"factor": label, "crude_or": float(np.exp(crude.params[f])),
                     "adjusted_or": float(np.exp(adj.params[f])), "ci_low": float(ci.loc[f, 0]), "ci_high": float(ci.loc[f, 1]),
                     "p_value": float(adj.pvalues[f])})
    per_decade = float(np.exp(adj.params["age"] * 10))
    return pd.DataFrame(rows).sort_values("adjusted_or", ascending=False), per_decade, adj


def charts(r, orr):
    # prevalence by age & sex
    a = r["prevalence_by_age_sex"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (sex, g) in enumerate(a.groupby("Sex")):
        ax.plot(g.AgeCategory, g.prevalence_pct, marker="o", ms=5, color=SERIES[i], label=sex)
        ax.annotate(sex, (len(g) - 1, g.prevalence_pct.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=MUTED)
    ax.set_ylabel("Heart disease prevalence, %")
    ax.set_title("Prevalence rises steeply with age — and is higher for men at every age")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.legend(loc="upper left")
    save(fig, FIG / "01_prevalence_age_sex.png")

    # crude vs adjusted OR (forest plot)
    o = orr.sort_values("adjusted_or")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(o))
    ax.hlines(y, o.ci_low, o.ci_high, color=SERIES[0], lw=2)
    ax.scatter(o.adjusted_or, y, color=SERIES[0], s=40, zorder=3, label="Adjusted OR (95% CI)")
    ax.scatter(o.crude_or, y, color="#b9b8b2", marker="D", s=30, zorder=2, label="Crude OR")
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, o.factor)
    ax.set_xlabel("Odds ratio for heart disease (log scale)")
    ax.set_title("Crude vs adjusted odds ratios: much of the crude risk is age")
    ax.legend(loc="lower right")
    save(fig, FIG / "02_odds_ratios.png")

    # comorbidity burden + general health
    c, g = r["comorbidity_burden"], r["gen_health"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [f"{int(k)}{'+' if k == 3 else ''}" for k in c.conditions]
    axs[0].bar(labels, c.prevalence_pct, color=SERIES[0], width=0.6)
    for x, (v, p) in enumerate(zip(c.prevalence_pct, c.population_pct)):
        axs[0].text(x, v + 0.5, f"{v:.0f}%\n({p:.0f}% of pop.)", ha="center", fontsize=8)
    axs[0].set_ylim(0, c.prevalence_pct.max() * 1.3)
    axs[0].set_xlabel("Number of risk conditions*")
    axs[0].set_title("Prevalence by comorbidity count, %")
    axs[1].bar(g.GenHealth, g.prevalence_pct, color=SERIES[1], width=0.6)
    for x, v in enumerate(g.prevalence_pct):
        axs[1].text(x, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9)
    axs[1].set_title("Prevalence by self-rated general health, %")
    fig.text(0.02, -0.03, "*stroke, diabetes, kidney disease, smoking, difficulty walking", fontsize=8, color=MUTED)
    save(fig, FIG / "03_comorbidity_genhealth.png")


def main():
    df, log = load()
    r = run_sql(df)
    X, y = encode(df)
    orr, per_decade, _ = odds_ratios(X, y)
    orr.to_csv(REPORTS / "odds_ratios.csv", index=False)
    charts(r, orr)

    o = orr.set_index("factor")
    c = r["comorbidity_burden"]
    lift = r["risk_factor_lift"].set_index("factor")
    summary = {
        **log, "prevalence_pct": float(r["overview"].prevalence_pct.iloc[0]),
        "cases": int(r["overview"].heart_disease_cases.iloc[0]),
        "or_per_decade_of_age": round(per_decade, 2),
        "top_adjusted_factor": o.index[0], "top_adjusted_or": round(float(o.adjusted_or.iloc[0]), 2),
        "stroke_crude_or": round(float(o.loc["Stroke", "crude_or"]), 2), "stroke_adjusted_or": round(float(o.loc["Stroke", "adjusted_or"]), 2),
        "diffwalk_crude_or": round(float(o.loc["Difficulty walking", "crude_or"]), 2),
        "diffwalk_adjusted_or": round(float(o.loc["Difficulty walking", "adjusted_or"]), 2),
        "comorb_0_prev": float(c.prevalence_pct.iloc[0]), "comorb_max_prev": float(c.prevalence_pct.iloc[-1]),
        "comorb_multiple": round(float(c.prevalence_pct.iloc[-1] / c.prevalence_pct.iloc[0]), 1),
        "stroke_relative_risk": float(lift.loc["Stroke", "relative_risk"]),
    }
    (REPORTS / "analytics_summary.json").write_text(json.dumps(summary, indent=2))
    print(orr.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
