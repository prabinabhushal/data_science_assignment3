from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in [PROC, FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)

matches = pd.read_csv(RAW / "fifa_2026_matches_verified.csv", parse_dates=["date"])
ranking = pd.read_csv(RAW / "fifa_ranking_2026-06-11.csv")

# Recreate pre-match features using only information available before each match.
rmap = ranking.set_index("team_code")["fifa_rank_2026_06_11"].to_dict()
pmap = ranking.set_index("team_code")["fifa_points_2026_06_11"].to_dict()

matches["tournament_day"] = (matches["date"] - pd.Timestamp("2026-06-11")).dt.days + 1
matches["knockout_stage"] = (matches["stage"] != "First Stage").astype(int)
matches["home_rank"] = matches.home.map(rmap)
matches["away_rank"] = matches.away.map(rmap)
matches["home_points"] = matches.home.map(pmap)
matches["away_points"] = matches.away.map(pmap)
matches["rank_difference"] = matches.away_rank - matches.home_rank
matches["points_difference"] = matches.home_points - matches.away_points
hosts = {"CAN", "MEX", "USA"}
matches["home_host"] = matches.home.isin(hosts).astype(int)
matches["away_host"] = matches.away.isin(hosts).astype(int)

last_seen = {}
home_rest, away_rest = [], []
for _, r in matches.sort_values(["date", "match_number"]).iterrows():
    home_rest.append((r.date - last_seen[r.home]).days if r.home in last_seen else 0)
    away_rest.append((r.date - last_seen[r.away]).days if r.away in last_seen else 0)
    last_seen[r.home] = r.date
    last_seen[r.away] = r.date

ordered = matches.sort_values(["date", "match_number"]).copy()
ordered["home_rest_days"] = home_rest
ordered["away_rest_days"] = away_rest
ordered["rest_difference"] = ordered.home_rest_days - ordered.away_rest_days
matches = ordered.sort_index()
matches["goal_difference"] = matches.home_score - matches.away_score

lr21 = matches[["match_id","home_name","away_name","rank_difference","points_difference",
                 "rest_difference","tournament_day","knockout_stage","home_host","away_host",
                 "match_number","goal_difference"]].copy()
lr21.columns = ["match_id","team_a","team_b","fifa_rank_difference","fifa_points_difference",
                "rest_days_difference","tournament_day","knockout_stage",
                "home_is_host_nation","away_is_host_nation","match_number","goal_difference"]

rows = []
for _, r in matches.iterrows():
    rows.append([r.match_id,r.home_name,r.away_name,r.home_rank,r.away_rank,r.rank_difference,
                 r.home_points,r.away_points,r.home_rest_days,r.away_rest_days,r.tournament_day,
                 r.knockout_stage,r.home_host,r.away_host,r.match_number,r.home_score])
    rows.append([r.match_id,r.away_name,r.home_name,r.away_rank,r.home_rank,-r.rank_difference,
                 r.away_points,r.home_points,r.away_rest_days,r.home_rest_days,r.tournament_day,
                 r.knockout_stage,r.away_host,r.home_host,r.match_number,r.away_score])

lr22 = pd.DataFrame(rows, columns=[
    "match_id","team","opponent","team_fifa_rank","opponent_fifa_rank","fifa_rank_difference",
    "team_fifa_points","opponent_fifa_points","team_rest_days","opponent_rest_days",
    "tournament_day","knockout_stage","team_is_host_nation","opponent_is_host_nation",
    "match_number","goals_scored"
])

P21 = ["fifa_rank_difference","fifa_points_difference","rest_days_difference","tournament_day",
       "knockout_stage","home_is_host_nation","away_is_host_nation","match_number"]
P22 = ["fifa_rank_difference","team_fifa_points","team_rest_days","opponent_rest_days",
       "knockout_stage","team_is_host_nation","opponent_is_host_nation","match_number"]

def analyse(data, predictors, target):
    X = data[predictors]
    y = data[target]
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    cv = GroupKFold(n_splits=5)
    scores = cross_validate(LinearRegression(), X, y, cv=cv, groups=data["match_id"],
                            scoring={"r2":"r2","mae":"neg_mean_absolute_error",
                                     "mse":"neg_mean_squared_error"})
    chronological_matches = (data[["match_id", "tournament_day", "match_number"]]
                             .drop_duplicates("match_id")
                             .sort_values(["tournament_day", "match_number"]))
    split_at = int(len(chronological_matches) * 0.8)
    train_ids = set(chronological_matches.iloc[:split_at]["match_id"])
    test_ids = set(chronological_matches.iloc[split_at:]["match_id"])
    train_rows = data["match_id"].isin(train_ids)
    test_rows = data["match_id"].isin(test_ids)
    holdout_model = LinearRegression().fit(X.loc[train_rows], y.loc[train_rows])
    holdout_predictions = holdout_model.predict(X.loc[test_rows])

    summary = {
        "n": len(data),
        "r2": ols.rsquared,
        "adjusted_r2": ols.rsquared_adj,
        "cv_r2": scores["test_r2"].mean(),
        "cv_mae": -scores["test_mae"].mean(),
        "cv_rmse": np.sqrt(-scores["test_mse"].mean()),
        "holdout_r2": r2_score(y.loc[test_rows], holdout_predictions),
        "holdout_mae": mean_absolute_error(y.loc[test_rows], holdout_predictions),
        "holdout_rmse": np.sqrt(mean_squared_error(y.loc[test_rows], holdout_predictions)),
        "holdout_matches": len(test_ids),
        "breusch_pagan_p": het_breuschpagan(ols.resid, sm.add_constant(X))[1],
        "shapiro_p": shapiro(ols.resid).pvalue,
    }
    exog = sm.add_constant(X)
    summary["max_vif"] = max(variance_inflation_factor(exog.values, i)
                              for i in range(1, exog.shape[1]))
    return ols, summary

PREDICTOR_LABELS = {
    "fifa_rank_difference": "FIFA rank difference (positive favors home/team)",
    "fifa_points_difference": "FIFA points difference (home minus away)",
    "rest_days_difference": "Rest days difference (home minus away)",
    "tournament_day": "Tournament day",
    "knockout_stage": "Knockout-stage match",
    "home_is_host_nation": "Home team is a host nation",
    "away_is_host_nation": "Away team is a host nation",
    "team_fifa_points": "Team FIFA points",
    "team_rest_days": "Team rest days",
    "opponent_rest_days": "Opponent rest days",
    "team_is_host_nation": "Team is a host nation",
    "opponent_is_host_nation": "Opponent is a host nation",
    "match_number": "Official match number",
}


def save_coefficient_table(model, predictors, path):
    confidence = model.conf_int().loc[predictors]
    p_values = model.pvalues.loc[predictors]
    table = pd.DataFrame({
        "variable": predictors,
        "predictor": [PREDICTOR_LABELS[name] for name in predictors],
        "coefficient": model.params.loc[predictors].values,
        "standard_error": model.bse.loc[predictors].values,
        "95% CI lower": confidence[0].values,
        "95% CI upper": confidence[1].values,
        "p_value": p_values.values,
        "significance": ["p < .001" if p < 0.001 else "p < .05" if p < 0.05 else "not significant"
                         for p in p_values],
    })
    table.to_csv(path, index=False)

# ------------------------------------
def save_predictions(data, model, target, path):
    predictions = data[["match_id", "tournament_day", "match_number"]].copy()
    for column in ("team_a", "team_b", "team", "opponent"):
        if column in data:
            predictions[column] = data[column]
    predictions["actual"] = data[target].to_numpy()
    predictions["predicted"] = model.fittedvalues.to_numpy()
    predictions["residual"] = predictions["actual"] - predictions["predicted"]
    predictions.to_csv(path, index=False)
    return predictions


def save_diagnostic_figures(predictions, target_label, model_label, prefix):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions["actual"], predictions["predicted"], alpha=0.72,
               color="#167D72", edgecolor="white", linewidth=0.5)
    bounds = [min(predictions["actual"].min(), predictions["predicted"].min()),
              max(predictions["actual"].max(), predictions["predicted"].max())]
    ax.plot(bounds, bounds, color="#D05B46", linestyle="--", linewidth=1.5,
            label="Perfect predictions")
    ax.set(title=f"{model_label}: in-sample actual vs predicted",
           xlabel=f"Actual {target_label}", ylabel=f"Predicted {target_label}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{prefix}_actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions["predicted"], predictions["residual"], alpha=0.72,
               color="#167D72", edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#D05B46", linestyle="--", linewidth=1.5)
    ax.set(title=f"{model_label}: in-sample residuals vs predicted",
           xlabel=f"Predicted {target_label}", ylabel="Residual (actual minus predicted)")
    fig.tight_layout()
    fig.savefig(FIG / f"{prefix}_residuals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
# ------------------------------------

ols21, s21 = analyse(lr21, P21, "goal_difference")
ols22, s22 = analyse(lr22, P22, "goals_scored")

# Required validation assertions.
assert len(lr21) == 104
assert len(P21) == 8
assert len(lr22) == 208
assert len(P22) == 8
assert lr21.match_id.nunique() == 104
assert lr22.match_id.nunique() == 104
assert lr22.groupby("match_id").size().eq(2).all()
assert lr21.isna().sum().sum() == 0
assert lr22.isna().sum().sum() == 0
assert matches.home_name.isna().sum() == 0 and matches.away_name.isna().sum() == 0

lr21.to_csv(PROC / "lr21_dataset_rebuilt.csv", index=False)
lr22.to_csv(PROC / "lr22_dataset_rebuilt.csv", index=False)

save_coefficient_table(ols21, P21, TAB / "lr21_coefficients_rebuilt.csv")
save_coefficient_table(ols22, P22, TAB / "lr22_coefficients_rebuilt.csv")

summary_rows = []
for label, target, predictors, summary in [
    ("LR 2.1: Match goal difference", "Home goals minus away goals", P21, s21),
    ("LR 2.2: Team goals scored", "Goals scored by a team", P22, s22),
]:
    summary_rows.append({
        "Model": label,
        "Target": target,
        "Observations": summary["n"],
        "Predictors": len(predictors),
        "Full-data R-squared": summary["r2"],
        "Adjusted R-squared": summary["adjusted_r2"],
        "Grouped CV R-squared": summary["cv_r2"],
        "Grouped CV MAE (goals)": summary["cv_mae"],
        "Grouped CV RMSE (goals)": summary["cv_rmse"],
        "Chronological holdout matches": summary["holdout_matches"],
        "Chronological holdout R-squared": summary["holdout_r2"],
        "Chronological holdout MAE (goals)": summary["holdout_mae"],
        "Chronological holdout RMSE (goals)": summary["holdout_rmse"],
        "Breusch-Pagan p-value": summary["breusch_pagan_p"],
        "Shapiro-Wilk p-value": summary["shapiro_p"],
        "Maximum VIF": summary["max_vif"],
    })
pd.DataFrame(summary_rows).to_csv(TAB / "model_summary_rebuilt.csv", index=False)

predictions21 = save_predictions(lr21, ols21, "goal_difference",
                                 TAB / "lr21_predictions_rebuilt.csv")
predictions22 = save_predictions(lr22, ols22, "goals_scored",
                                 TAB / "lr22_predictions_rebuilt.csv")
save_diagnostic_figures(predictions21, "goal difference", "LR 2.1", "lr21")
save_diagnostic_figures(predictions22, "goals scored", "LR 2.2", "lr22")

print("LR 2.1 (match goal difference)")
print(f"  Grouped CV R-squared: {s21['cv_r2']:.3f}; MAE: {s21['cv_mae']:.3f}; RMSE: {s21['cv_rmse']:.3f}")
print(f"  Chronological holdout R-squared: {s21['holdout_r2']:.3f}; MAE: {s21['holdout_mae']:.3f}; RMSE: {s21['holdout_rmse']:.3f}")
print("LR 2.2 (team goals scored)")
print(f"  Grouped CV R-squared: {s22['cv_r2']:.3f}; MAE: {s22['cv_mae']:.3f}; RMSE: {s22['cv_rmse']:.3f}")
print(f"  Chronological holdout R-squared: {s22['holdout_r2']:.3f}; MAE: {s22['holdout_mae']:.3f}; RMSE: {s22['holdout_rmse']:.3f}")
print("Validation passed. Rebuilt datasets, model tables, prediction tables and diagnostic figures.")
