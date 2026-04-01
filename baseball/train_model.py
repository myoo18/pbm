# train_model.py
"""
Trains XGBoost HR prediction model using Statcast data.

Improvements over v0.0.1:
  - All 30 MLB park factors (was 13)
  - LHP/RHP platoon split features
  - Pitcher handedness feature
  - Platoon advantage feature
  - Pitcher minimum raised to 50 pitches (was 5 starts)
  - Comprehensive model assessment (calibration, overfit check, year-by-year, thresholds)

Run:
    uv run train_model.py

First run: ~2-3 hrs (downloads Statcast for each season)
Subsequent runs: much faster with pybaseball cache enabled
"""

from pybaseball import statcast, cache
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, average_precision_score,
)
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings("ignore")
cache.enable()

# ── Features (must match feature_builder.py exactly) ─────────────────────────

FEATURES = [
    # Batter rolling HR rates
    "hr_rate_7d", "hr_rate_15d", "hr_rate_30d", "hr_rate_season",
    # Platoon splits
    "hr_rate_vs_lhp_30d", "hr_rate_vs_rhp_30d",
    # Exit velocity
    "avg_ev_7d", "avg_ev_30d", "ev_trend",
    # Quality of contact
    "barrel_rate_30d", "hard_hit_pct", "sweet_spot_pct",
    # Streaks
    "hot_streak", "cold_streak",
    # Volume
    "avg_pa_per_game",
    # Pitcher vulnerability
    "pitcher_hr_per_9", "pitcher_barrel_rate_allowed",
    "pitcher_hard_hit_allowed", "pitcher_fb_pct",
    # Handedness / platoon
    "pitcher_hand", "platoon_advantage",
    # Park
    "park_factor",
]

# ── Park factors (all 30 stadiums, 2024 HR factors) ───────────────────────────

PARK_FACTORS = {
    "coors field":              1.35,
    "great american ball park": 1.25,
    "citizens bank park":       1.20,
    "yankee stadium":           1.18,
    "fenway park":              1.10,
    "wrigley field":            1.08,
    "guaranteed rate field":    1.07,
    "globe life field":         1.05,
    "truist park":              1.03,
    "chase field":              1.02,
    "american family field":    1.01,
    "camden yards":             1.00,
    "rogers centre":            1.00,
    "nationals park":           0.99,
    "loandepot park":           0.98,
    "sutter health park":       0.98,   # Athletics temp home (Sacramento)
    "angel stadium":            0.97,
    "progressive field":        0.97,
    "kauffman stadium":         0.96,
    "target field":             0.95,
    "minute maid park":         0.95,
    "daikin park":              0.95,
    "dodger stadium":           0.95,
    "uniqlo field at dodger stadium": 0.95,
    "citi field":               0.94,
    "busch stadium":            0.93,
    "tropicana field":          0.92,
    "pnc park":                 0.91,
    "comerica park":            0.90,
    "petco park":               0.87,
    "oracle park":              0.88,
    "t-mobile park":            0.86,
}

TEAM_TO_PARK = {
    "COL": "coors field",
    "CIN": "great american ball park",
    "PHI": "citizens bank park",
    "NYY": "yankee stadium",
    "BOS": "fenway park",
    "CHC": "wrigley field",
    "CWS": "guaranteed rate field",
    "TEX": "globe life field",
    "ATL": "truist park",
    "ARI": "chase field",
    "MIL": "american family field",
    "BAL": "camden yards",
    "TOR": "rogers centre",
    "WSH": "nationals park",
    "MIA": "loandepot park",
    "ATH": "sutter health park",
    "LAA": "angel stadium",
    "CLE": "progressive field",
    "KC":  "kauffman stadium",
    "MIN": "target field",
    "HOU": "minute maid park",
    "LAD": "dodger stadium",
    "NYM": "citi field",
    "STL": "busch stadium",
    "TB":  "tropicana field",
    "PIT": "pnc park",
    "DET": "comerica park",
    "SD":  "petco park",
    "SF":  "oracle park",
    "SEA": "t-mobile park",
}


def _streak(arr: np.ndarray, value: int) -> int:
    count = 0
    for v in reversed(arr):
        if v == value:
            count += 1
        else:
            break
    return count


def build_pitcher_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-pitcher per-date rolling vulnerability table.
    Uses only data BEFORE each game date — no leakage.
    Minimum 50 cumulative pitches before a pitcher gets stats.

    Returns DataFrame with columns:
        pitcher, game_date,
        pitcher_hr_per_9, pitcher_barrel_rate_allowed,
        pitcher_hard_hit_allowed, pitcher_fb_pct, pitcher_hand
    """
    print("  Building pitcher rolling stats...")

    df = df.copy()
    df["is_hr"]     = (df["events"] == "home_run").fillna(False).astype(int)
    df["is_barrel"] = (df.get("launch_speed_angle", pd.Series(dtype=float)) == 6).fillna(False).astype(int)
    df["is_hard"]   = (df["launch_speed"] >= 95).fillna(False).astype(int)
    df["is_fb"]     = df["pitch_type"].isin(["FF", "SI", "FC"]).astype(int)
    df["is_bip"]    = (df["type"] == "X").fillna(False).astype(int)

    pgame = (
        df.groupby(["pitcher", "game_date"])
        .agg(
            hrs=("is_hr", "sum"),
            ip=("inning", "max"),
            barrels=("is_barrel", "sum"),
            hard=("is_hard", "sum"),
            bip=("is_bip", "sum"),
            pitches=("pitch_type", "count"),
            fb=("is_fb", "sum"),
            hand=("p_throws", lambda x: "L" if (x == "L").mean() > 0.5 else "R"),
        )
        .reset_index()
        .sort_values(["pitcher", "game_date"])
    )

    rows = []
    for pitcher_id, grp in pgame.groupby("pitcher"):
        grp = grp.reset_index(drop=True)
        pitcher_hand_val = 1 if grp.iloc[0]["hand"] == "L" else 0

        for i in range(1, len(grp)):
            past = grp.iloc[:i]
            if past["pitches"].sum() < 50:   # need at least ~5 starts of history
                continue

            today_date = grp.iloc[i]["game_date"]
            total_ip   = past["ip"].sum()
            total_bip  = past["bip"].sum()
            total_p    = past["pitches"].sum()

            rows.append({
                "pitcher":                     pitcher_id,
                "game_date":                   today_date,
                "pitcher_hr_per_9":            past["hrs"].sum() / max(total_ip, 1) * 9,
                "pitcher_barrel_rate_allowed": past["barrels"].sum() / max(total_bip, 1),
                "pitcher_hard_hit_allowed":    past["hard"].sum() / max(total_bip, 1),
                "pitcher_fb_pct":              past["fb"].sum() / max(total_p, 1),
                "pitcher_hand":                pitcher_hand_val,
            })

    return pd.DataFrame(rows)


def build_training_data(seasons: list[int]) -> pd.DataFrame:
    """
    For each batter-game, build the full feature set as it would have
    looked BEFORE that game — no leakage.
    """
    all_rows = []

    for year in seasons:
        print(f"\nProcessing {year}...")
        df = statcast(f"{year}-03-20", f"{year}-10-05")
        df = df.sort_values("game_date")
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

        # ── Pitcher rolling lookup ────────────────────────────────────────────
        pitcher_lookup = build_pitcher_lookup(df)

        # ── Batter per-game aggregation ───────────────────────────────────────
        df["is_hr"]     = (df["events"] == "home_run").fillna(False).astype(int)
        df["is_pa"]     = df["events"].notna().astype(int)
        df["is_barrel"] = (df.get("launch_speed_angle", pd.Series(dtype=float)) == 6).fillna(False).astype(int)
        df["is_hard"]   = (df["launch_speed"] >= 95).fillna(False).astype(int)
        df["is_sweet"]  = df["launch_angle"].between(8, 32).fillna(False).astype(int)

        # Starting pitcher faced (most common)
        game_pitcher = (
            df.groupby(["batter", "game_date"])["pitcher"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
            .rename(columns={"pitcher": "pitcher_id"})
        )

        # Opposing pitcher handedness per game
        game_hand = (
            df.groupby(["batter", "game_date"])["p_throws"]
            .agg(lambda x: "L" if (x == "L").mean() > 0.5 else "R")
            .reset_index()
            .rename(columns={"p_throws": "opp_hand"})
        )

        # Batter's own handedness (constant per player — take first value)
        game_stand = (
            df.groupby(["batter", "game_date"])["stand"]
            .first()
            .reset_index()
            .rename(columns={"stand": "batter_hand"})
        )

        # Home team for park factor
        game_park = (
            df.groupby(["batter", "game_date"])["home_team"]
            .first()
            .reset_index()
        )

        game_results = (
            df.groupby(["batter", "game_date"])
            .agg(
                hrs=("is_hr", "sum"),
                pas=("is_pa", "sum"),
                avg_ev=("launch_speed", "mean"),
                barrels=("is_barrel", "sum"),
                hard=("is_hard", "sum"),
                sweet=("is_sweet", "sum"),
            )
            .reset_index()
        )
        game_results["hit_hr"] = (game_results["hrs"] > 0).astype(int)
        game_results["avg_ev"] = game_results["avg_ev"].fillna(88.0)  # league avg EV for no-BIP games
        game_results = game_results.merge(game_pitcher, on=["batter", "game_date"])
        game_results = game_results.merge(game_park,    on=["batter", "game_date"])
        game_results = game_results.merge(game_hand,    on=["batter", "game_date"])
        game_results = game_results.merge(game_stand,   on=["batter", "game_date"])

        # ── Build lag features per batter ─────────────────────────────────────
        for batter_id, grp in game_results.groupby("batter"):
            grp = grp.sort_values("game_date").reset_index(drop=True)
            if len(grp) < 20:
                continue

            for i in range(15, len(grp)):
                past  = grp.iloc[:i]
                today = grp.iloc[i]

                def r(num, den, w):
                    n = past.tail(w)[num].sum()
                    d = past.tail(w)[den].sum()
                    return float(n / d) if d > 0 else 0.0

                ev7  = float(past.tail(7)["avg_ev"].mean())
                ev30 = float(past.tail(30)["avg_ev"].mean())

                # Park factor
                home = today.get("home_team", "")
                park = TEAM_TO_PARK.get(str(home).upper(), "")
                park_factor = PARK_FACTORS.get(park, 1.00)

                # Pitcher features
                pitcher_id   = today["pitcher_id"]
                pitcher_date = today["game_date"]
                pm = pitcher_lookup[
                    (pitcher_lookup["pitcher"] == pitcher_id) &
                    (pitcher_lookup["game_date"] == pitcher_date)
                ]

                if pm.empty:
                    p_hr9    = 1.10
                    p_barrel = 0.065
                    p_hard   = 0.360
                    p_fb     = 0.560
                    p_hand   = 0      # default RHP
                else:
                    p_hr9    = float(pm.iloc[0]["pitcher_hr_per_9"])
                    p_barrel = float(pm.iloc[0]["pitcher_barrel_rate_allowed"])
                    p_hard   = float(pm.iloc[0]["pitcher_hard_hit_allowed"])
                    p_fb     = float(pm.iloc[0]["pitcher_fb_pct"])
                    p_hand   = int(pm.iloc[0]["pitcher_hand"])

                # Platoon splits
                past_lhp = past[past["opp_hand"] == "L"]
                past_rhp = past[past["opp_hand"] == "R"]
                hr_vs_lhp = float(
                    past_lhp.tail(30)["hrs"].sum() / max(past_lhp.tail(30)["pas"].sum(), 1)
                )
                hr_vs_rhp = float(
                    past_rhp.tail(30)["hrs"].sum() / max(past_rhp.tail(30)["pas"].sum(), 1)
                )

                # Platoon advantage
                batter_hand = today.get("batter_hand", "R")
                opp_hand    = today.get("opp_hand", "R")
                if batter_hand == "S":
                    platoon_adv = 0.5
                elif batter_hand != opp_hand:
                    platoon_adv = 1.0
                else:
                    platoon_adv = 0.0

                all_rows.append({
                    "batter":    batter_id,
                    "game_date": today["game_date"],
                    "hit_hr":    int(today["hit_hr"]),

                    "hr_rate_7d":          r("hrs", "pas", 7),
                    "hr_rate_15d":         r("hrs", "pas", 15),
                    "hr_rate_30d":         r("hrs", "pas", 30),
                    "hr_rate_season":      float(past["hrs"].sum() / max(past["pas"].sum(), 1)),
                    "hr_rate_vs_lhp_30d":  hr_vs_lhp,
                    "hr_rate_vs_rhp_30d":  hr_vs_rhp,

                    "avg_ev_7d":   ev7,
                    "avg_ev_30d":  ev30,
                    "ev_trend":    ev7 - ev30,

                    "barrel_rate_30d": r("barrels", "pas", 30),
                    "hard_hit_pct":    float(past["hard"].sum() / max(past["pas"].sum(), 1)),
                    "sweet_spot_pct":  float(past["sweet"].sum() / max(past["pas"].sum(), 1)),

                    "avg_pa_per_game": float(past["pas"].mean()),

                    "hot_streak":  _streak((past["hrs"] > 0).astype(int).values, 1),
                    "cold_streak": _streak((past["hrs"] > 0).astype(int).values, 0),

                    "pitcher_hr_per_9":            p_hr9,
                    "pitcher_barrel_rate_allowed": p_barrel,
                    "pitcher_hard_hit_allowed":    p_hard,
                    "pitcher_fb_pct":              p_fb,
                    "pitcher_hand":                p_hand,
                    "platoon_advantage":           platoon_adv,

                    "park_factor": park_factor,
                })

    return pd.DataFrame(all_rows)


from feature_builder import CalibratedXGB


def assess(model, X_tr, y_tr, X_te, y_te, df, train_mask):
    """Comprehensive model assessment — calibration, overfit, thresholds."""
    preds_te = model.predict_proba(X_te)[:, 1]
    preds_tr = model.predict_proba(X_tr)[:, 1]

    auc_te  = roc_auc_score(y_te, preds_te)
    auc_tr  = roc_auc_score(y_tr, preds_tr)
    ll_te   = log_loss(y_te, preds_te)
    ll_tr   = log_loss(y_tr, preds_tr)
    brier   = brier_score_loss(y_te, preds_te)
    pr_auc  = average_precision_score(y_te, preds_te)
    # Unwrap calibration wrapper if present
    base = getattr(model, "estimator", model)
    n_trees = base.best_iteration + 1 if hasattr(base, "best_iteration") else getattr(base, "n_estimators", "?")

    print("\n" + "=" * 72)
    print("MODEL ASSESSMENT")
    print("=" * 72)

    print("\n--- CORE METRICS ---")
    print(f"  Trees used      : {n_trees} / {getattr(base, 'n_estimators', '?')}")
    print(f"  Train AUC-ROC   : {auc_tr:.4f}")
    print(f"  Test  AUC-ROC   : {auc_te:.4f}   (target > 0.65)")
    print(f"  AUC overfit gap : {auc_tr - auc_te:.4f}   (good if < 0.05)")
    print(f"  Train Log Loss  : {ll_tr:.4f}")
    print(f"  Test  Log Loss  : {ll_te:.4f}")
    print(f"  Brier Score     : {brier:.4f}   (baseline: {float(y_te.mean() * (1-y_te.mean())):.4f})")
    print(f"  PR-AUC          : {pr_auc:.4f}   (baseline: {y_te.mean():.4f} = HR rate)")

    print("\n--- CALIBRATION (predicted vs actual HR rate by decile) ---")
    print(f"  {'Pred Range':<14}  {'Pred Avg':>9}  {'Actual%':>9}  {'N':>8}")
    bins = np.percentile(preds_te, np.linspace(0, 100, 11))
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (preds_te >= lo) & (preds_te < hi) if i < 9 else (preds_te >= lo)
        if mask.sum() > 0:
            print(
                f"  {lo:.3f}–{hi:.3f}       "
                f"{preds_te[mask].mean():>9.3f}  "
                f"{y_te.values[mask].mean():>9.3f}  "
                f"{mask.sum():>8,}"
            )

    print("\n--- YEAR-BY-YEAR BREAKDOWN (test set) ---")
    test_df = df[~train_mask].copy()
    test_df["_pred"] = preds_te
    test_df["_year"] = pd.to_datetime(test_df["game_date"].astype(str)).dt.year
    for yr in sorted(test_df["_year"].unique()):
        ym = test_df["_year"] == yr
        if ym.sum() > 100:
            yr_auc = roc_auc_score(y_te.values[ym.values], test_df.loc[ym, "_pred"].values)
            hr_pct = y_te.values[ym.values].mean()
            print(f"  {yr}: AUC={yr_auc:.4f}  HR%={hr_pct:.3f}  n={ym.sum():,}")

    print("\n--- BETTING THRESHOLD ANALYSIS (test set) ---")
    print(f"  {'Min model%':<12}  {'Precision':>10}  {'Recall':>8}  {'N bets':>8}")
    for thresh in [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.20]:
        above = preds_te >= thresh
        if above.sum() > 0:
            precision = y_te.values[above].mean()
            recall    = y_te.values[above].sum() / max(y_te.sum(), 1)
            print(
                f"  ≥ {thresh*100:.0f}%          "
                f"{precision*100:>9.1f}%  "
                f"{recall*100:>7.1f}%  "
                f"{above.sum():>8,}"
            )

    print("\n--- TOP 15 FEATURES (by gain) ---")
    imp = pd.Series(
        base.get_booster().get_score(importance_type="gain"),
        name="gain"
    ).sort_values(ascending=False)
    for feat, val in imp.head(15).items():
        print(f"  {feat:<35}  {val:.1f}")

    print("=" * 72)
    return preds_te


def train(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES)
    X  = df[FEATURES]
    y  = df["hit_hr"]

    split_date = pd.to_datetime("2025-04-01").date()
    train_mask = df["game_date"] < split_date
    X_tr, X_te = X[train_mask], X[~train_mask]
    y_tr, y_te = y[train_mask], y[~train_mask]

    print(f"\nTrain rows : {len(X_tr):,}")
    print(f"Test rows  : {len(X_te):,}")
    hr_rate = float(y_tr.mean())
    print(f"HR rate    : {hr_rate:.3f} train / {y_te.mean():.3f} test")

    # Split train into fit / calibration (80/20 by time — no shuffling)
    cal_cut   = int(len(X_tr) * 0.8)
    X_fit,  X_cal  = X_tr.iloc[:cal_cut],  X_tr.iloc[cal_cut:]
    y_fit,  y_cal  = y_tr.iloc[:cal_cut],  y_tr.iloc[cal_cut:]

    base_model = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.015,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        # No scale_pos_weight — it inflates all predictions toward 0.5.
        # Use base_score = actual HR rate so the model starts from the right prior.
        base_score=hr_rate,
        eval_metric="logloss",
        early_stopping_rounds=75,
        random_state=42,
        device="cuda",
        verbosity=0,
    )
    base_model.fit(
        X_fit, y_fit,
        eval_set=[(X_te, y_te)],
        verbose=50,
    )

    # Isotonic calibration — maps raw XGBoost scores to real probabilities.
    # Fit on the held-out calibration slice (not used to fit trees → no leakage).
    print("\nCalibrating with isotonic regression...")
    raw_cal = base_model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)

    calibrated = CalibratedXGB(base_model, iso)

    assess(calibrated, X_tr, y_tr, X_te, y_te, df, train_mask)

    joblib.dump(calibrated, "hr_model.pkl")
    print("\nModel saved → hr_model.pkl  (XGBoost + isotonic calibration)")
    return calibrated


if __name__ == "__main__":
    df = build_training_data(seasons=[2022, 2023, 2024, 2025, 2026])
    train(df)
