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

