"""
Stage 2 - heart disease risk model for SCREENING (the cost of missing a case >> cost of a follow-up test).

1. Model + imbalance strategy comparison with stratified CV (ROC-AUC, PR-AUC, Brier)
2. Calibrated LightGBM (isotonic) -> probabilities that mean what they say
3. Threshold chosen on a validation split for >= 80% recall; reported on an untouched test split
4. Lift / gains, risk tiers, SHAP, and a fairness audit across sex, age band and race (race is NOT a feature)
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import FIG, REPORTS, ROOT, encode, load
from viz import MUTED, SERIES, plt, save

RECALL_TARGET = 0.80
SEED = 7
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.06, num_leaves=31, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, random_state=SEED)
TIERS = [(0, 0.05, "Low"), (0.05, 0.15, "Moderate"), (0.15, 0.30, "High"), (0.30, 1.01, "Very high")]


def scores(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def at_threshold(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"threshold": float(t), "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
            "specificity": tn / (tn + fp), "flagged_share": float(pred.mean()), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    df, _ = load()
    X, y = encode(df)
    idx = np.arange(len(X))
    i_tmp, i_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    i_fit, i_val = train_test_split(i_tmp, test_size=0.15, stratify=y.iloc[i_tmp], random_state=SEED)
    Xtmp, ytmp = X.iloc[i_tmp], y.iloc[i_tmp]
    print(f"train+val {len(i_tmp):,} | test {len(i_test):,} | prevalence {y.mean():.2%}")

    # ---------- 1. model & imbalance strategy comparison (3-fold stratified CV on train+val)
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    spw = float((ytmp == 0).sum() / (ytmp == 1).sum())
    candidates = {
        "Logistic regression (balanced)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random forest (balanced)": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=20,
                                                           class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "LightGBM (no weighting)": lgb.LGBMClassifier(**LGB_PARAMS),
        "LightGBM (class weights)": lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=spw),
        "LightGBM + SMOTE": imb_pipeline(SMOTE(random_state=SEED, sampling_strategy=0.5), lgb.LGBMClassifier(**LGB_PARAMS)),
    }
    cv_rows, oof = [], {}
    for name, est in candidates.items():
        p = cross_val_predict(est, Xtmp, ytmp, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
        oof[name] = p
        cv_rows.append({"model": name, **scores(ytmp, p),
                        "recall_at_0.5": float(((p >= 0.5) & (ytmp == 1)).sum() / (ytmp == 1).sum())})
        print(f"  {name:32s} AUC {cv_rows[-1]['roc_auc']:.4f}  PR-AUC {cv_rows[-1]['pr_auc']:.4f}  Brier {cv_rows[-1]['brier']:.4f}")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(REPORTS / "cv_model_comparison.csv", index=False)

    # ---------- 2. final model: LightGBM + isotonic calibration, fit on train split only
    Xfit, yfit, Xval, yval = X.iloc[i_fit], y.iloc[i_fit], X.iloc[i_val], y.iloc[i_val]
    Xte, yte = X.iloc[i_test], y.iloc[i_test]
    cal = CalibratedClassifierCV(lgb.LGBMClassifier(**LGB_PARAMS), method="isotonic", cv=3).fit(Xfit, yfit)
    weighted = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=spw).fit(Xfit, yfit)
    p_val, p_te = cal.predict_proba(Xval)[:, 1], cal.predict_proba(Xte)[:, 1]
    p_te_w = weighted.predict_proba(Xte)[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xfit, yfit)
    p_te_lr = lr.predict_proba(Xte)[:, 1]

    # ---------- 3. threshold for >= 80% recall, chosen on VALIDATION, reported on TEST
    prec, rec, thr = precision_recall_curve(yval, p_val)
    ok = np.where(rec[:-1] >= RECALL_TARGET)[0]
    t_star = float(thr[ok[-1]])  # highest threshold that still meets the recall target
    test_op = at_threshold(yte.values, p_te, t_star)
    default_op = at_threshold(yte.values, p_te, 0.5)
    # same recall with logistic regression -> how many more people flagged?
    pl, rl, tl = precision_recall_curve(yval, lr.predict_proba(Xval)[:, 1])
    t_lr = float(tl[np.where(rl[:-1] >= RECALL_TARGET)[0][-1]])
    lr_op = at_threshold(yte.values, p_te_lr, t_lr)
    print("test @ threshold:", {k: round(v, 3) if isinstance(v, float) else v for k, v in test_op.items()})

    # ---------- 4a. lift / gains + risk tiers
    order = np.argsort(-p_te)
    ys = yte.values[order]
    gains = np.cumsum(ys) / ys.sum()
    top10 = float(gains[int(0.1 * len(ys)) - 1])
    top20 = float(gains[int(0.2 * len(ys)) - 1])
    test_df = df.iloc[i_test].assign(risk=p_te, y=yte.values)
    test_df["tier"] = pd.cut(test_df.risk, [t[0] for t in TIERS] + [1.01], labels=[t[2] for t in TIERS], right=False)
    tiers = test_df.groupby("tier", observed=False).agg(people=("y", "size"), cases=("y", "sum"),
                                                        mean_predicted=("risk", "mean"), observed=("y", "mean"))
    tiers["population_share"] = tiers.people / tiers.people.sum()
    tiers["case_share"] = tiers.cases / tiers.cases.sum()
    tiers.to_csv(REPORTS / "risk_tiers.csv")

    # ---------- 4b. fairness audit at the chosen threshold
    test_df["flag"] = test_df.risk >= t_star
    test_df["age_band"] = pd.cut(X.iloc[i_test].age, [0, 44, 64, 120], labels=["18-44", "45-64", "65+"])
    fair_rows = []
    for attr in ["Sex", "age_band", "Race"]:
        for g, d in test_df.groupby(attr, observed=True):
            pos, neg = d[d.y == 1], d[d.y == 0]
            fair_rows.append({"attribute": attr, "group": g, "n": len(d), "prevalence": d.y.mean(),
                              "recall_tpr": pos.flag.mean() if len(pos) else np.nan, "fpr": neg.flag.mean(),
                              "precision": d[d.flag].y.mean() if d.flag.any() else np.nan,
                              "mean_predicted": d.risk.mean(), "auc": roc_auc_score(d.y, d.risk) if d.y.nunique() > 1 else np.nan})
    fair = pd.DataFrame(fair_rows)
    fair.to_csv(REPORTS / "fairness_audit.csv", index=False)

    # ---------- 4c. SHAP on the (uncalibrated) booster: same ranking, additive log-odds
    booster = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xfit, yfit)
    samp = Xte.sample(4000, random_state=SEED)
    sv = shap.TreeExplainer(booster).shap_values(samp)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(axis=0), index=samp.columns).sort_values(ascending=False)

    charts(cv, yte, p_te, p_te_w, gains, tiers, imp, sv, samp, fair, test_op, t_star)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"calibrated": cal, "booster": booster, "threshold": t_star, "features": list(X.columns),
                 "tiers": TIERS}, ROOT / "models" / "heart_risk.joblib")

    best = cv.set_index("model")
    race = fair[fair.attribute == "Race"].dropna(subset=["recall_tpr"])
    race_big = race[race.n >= 500]
    summary = {
        "n_train_val": len(i_tmp), "n_test": len(i_test), "prevalence": float(y.mean()),
        "cv": cv.round(4).to_dict(orient="records"),
        "test_auc": round(roc_auc_score(yte, p_te), 4), "test_pr_auc": round(average_precision_score(yte, p_te), 4),
        "test_auc_lr": round(roc_auc_score(yte, p_te_lr), 4),
        "brier_calibrated": round(brier_score_loss(yte, p_te), 4), "brier_weighted": round(brier_score_loss(yte, p_te_w), 4),
        "mean_pred_weighted": round(float(p_te_w.mean()), 4), "mean_pred_calibrated": round(float(p_te.mean()), 4),
        "threshold": round(t_star, 4), "operating_point": test_op, "default_0_5": default_op, "lr_same_recall": lr_op,
        "flagged_reduction_vs_lr_pct": round(100 * (1 - test_op["flagged_share"] / lr_op["flagged_share"]), 1),
        "top10_capture": round(top10, 3), "top20_capture": round(top20, 3),
        "tiers": tiers.round(4).reset_index().to_dict(orient="records"),
        "top_shap": imp.head(6).round(3).to_dict(),
        "fair_recall_by_sex": fair[fair.attribute == "Sex"].set_index("group").recall_tpr.round(3).to_dict(),
        "fair_recall_by_age": fair[fair.attribute == "age_band"].set_index("group").recall_tpr.round(3).to_dict(),
        "fair_race_recall_range": [round(float(race_big.recall_tpr.min()), 3), round(float(race_big.recall_tpr.max()), 3)],
        "smote_vs_plain_auc": round(float(best.loc["LightGBM + SMOTE", "roc_auc"] - best.loc["LightGBM (no weighting)", "roc_auc"]), 4),
    }
    (REPORTS / "model_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cv", "tiers", "operating_point", "default_0_5", "lr_same_recall")},
                     indent=2, default=float))


def charts(cv, yte, p, p_w, gains, tiers, imp, sv, samp, fair, op, t_star):
    # CV comparison
    c = cv.sort_values("pr_auc")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, col, title in ((axs[0], "roc_auc", "ROC-AUC"), (axs[1], "pr_auc", "PR-AUC (better metric at 8.6% prevalence)")):
        colors = [SERIES[0] if m == "LightGBM (no weighting)" else "#b9b8b2" for m in c.model]
        ax.barh(c.model, c[col], color=colors, height=0.6)
        for yy, v in enumerate(c[col]):
            ax.text(v, yy, f" {v:.3f}", va="center", fontsize=8.5)
        ax.set_xlim(c[col].min() * 0.9, c[col].max() * 1.06)
        ax.set_title(title)
    fig.suptitle("3-fold stratified CV: models and imbalance strategies", x=0.02, ha="left", fontweight="bold")
    save(fig, FIG / "04_cv_comparison.png")

    # calibration
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for probs, name, col in ((p_w, "Class-weighted LightGBM", SERIES[1]), (p, "Calibrated LightGBM (isotonic)", SERIES[0])):
        fy, fx = calibration_curve(yte, probs, n_bins=12, strategy="quantile")
        ax.plot(fx, fy, marker="o", ms=4, color=col, label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed rate")
    ax.set_title("Calibration: does 30% risk mean 30%?")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, FIG / "05_calibration.png")

    # gains
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    xs = np.arange(1, len(gains) + 1) / len(gains)
    ax.plot(xs * 100, gains * 100, color=SERIES[0], label="Model")
    ax.plot([0, 100], [0, 100], ls="--", color=MUTED, lw=1, label="Random")
    for q in (0.1, 0.2):
        v = gains[int(q * len(gains)) - 1] * 100
        ax.scatter([q * 100], [v], color=SERIES[0], s=30, zorder=3)
        ax.annotate(f"top {q:.0%} → {v:.0f}% of cases", (q * 100, v), xytext=(10, -14), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("% of population screened (highest risk first)"); ax.set_ylabel("% of heart disease cases found")
    ax.set_title("Cumulative gains")
    ax.legend(loc="lower right")
    save(fig, FIG / "06_gains.png")

    # risk tiers
    t = tiers.reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = np.arange(len(t))
    ax.bar(xs - 0.18, t.population_share * 100, 0.34, color="#b9b8b2", label="% of population")
    ax.bar(xs + 0.18, t.case_share * 100, 0.34, color=SERIES[0], label="% of all cases")
    for x, obs in zip(xs, t.observed):
        ax.text(x, max(t.population_share.max(), t.case_share.max()) * 100 * 1.02, f"{obs:.0%} have HD", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(xs, [f"{n}\n({lo:.0%}–{min(hi, 1):.0%})" for lo, hi, n in TIERS])
    ax.set_ylim(0, max(t.population_share.max(), t.case_share.max()) * 118)
    ax.set_title("Risk tiers: where the cases are")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    save(fig, FIG / "07_risk_tiers.png")

    # SHAP
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("Mean |SHAP|, log-odds")
    ax.set_title("What drives predicted risk (SHAP)")
    save(fig, FIG / "08_shap_importance.png")
    shap.summary_plot(sv, samp, max_display=12, show=False, plot_size=(8, 5))
    plt.gcf().axes[0].set_title("SHAP beeswarm", loc="left", fontweight="bold")
    save(plt.gcf(), FIG / "09_shap_beeswarm.png")

    # fairness
    f = fair[(fair.attribute != "Race") | (fair.n >= 500)].copy()
    f["label"] = f.attribute.replace({"age_band": "Age"}) + ": " + f.group.astype(str)
    f = f.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(f.label, f.recall_tpr * 100, color=SERIES[0], height=0.6)
    ax.axvline(op["recall"] * 100, color=MUTED, ls="--", lw=1)
    ax.text(op["recall"] * 100, -0.9, f" overall {op['recall']:.0%}", fontsize=8.5, color=MUTED, va="center")
    ax.set_ylim(-1.3, len(f) - 0.5)
    for yy, (r_, fp) in enumerate(zip(f.recall_tpr, f.fpr)):
        ax.text(r_ * 100 + 0.5, yy, f"{r_:.0%}  (FPR {fp:.0%})", va="center", fontsize=8)
    ax.set_xlim(0, 115)
    ax.set_xlabel(f"Recall at threshold {t_star:.3f}, %")
    ax.set_title("Fairness audit: recall and false-positive rate by group")
    save(fig, FIG / "10_fairness.png")


if __name__ == "__main__":
    main()
