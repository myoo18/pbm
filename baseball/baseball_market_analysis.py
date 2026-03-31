# baseball_market_analysis.py
"""
MLB Props — HR Lines + Model Edge Finder
========================================
Full pipeline:
  1. Pulls today's starting pitchers from MLB Stats API (free, no key)
  2. Pulls live HR props from sharp books via The Odds API
  3. Fetches rolling Statcast features per batter + pitcher
  4. Scores with trained XGBoost model
  5. Surfaces positive-edge plays

Setup:
    uv add requests pandas python-dotenv pybaseball xgboost scikit-learn joblib numpy
    echo "ODDS_API_KEY=your_key" > .env

Run daily (before first pitch):
    uv run baseball_market_analysis.py

Train model first (one time):
    uv run train_model.py
"""

from __future__ import annotations

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import date
from dotenv import load_dotenv
from pathlib import Path

from feature_builder import (
    get_batter_features,
    get_pitcher_features,
    get_park_factor,
    get_game_details_today,
    get_roster_player_teams,
)

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ODDS_API_KEY")
BASE_URL   = "https://api.the-odds-api.com/v4"
SPORT      = "baseball_mlb"
TODAY      = date.today().strftime("%Y-%m-%d")
MODEL_PATH = Path(__file__).parent / "hr_model.pkl"

BOOKS   = ["draftkings", "fanduel", "betmgm", "caesars", "bet365"]
MARKETS = ["batter_home_runs"]

EDGE_THRESHOLD = 5.0

FEATURES = [
    "hr_rate_7d", "hr_rate_15d", "hr_rate_30d", "hr_rate_season",
    "hr_rate_vs_lhp_30d", "hr_rate_vs_rhp_30d",
    "avg_ev_7d", "avg_ev_30d", "ev_trend",
    "barrel_rate_30d", "hard_hit_pct", "sweet_spot_pct",
    "hot_streak", "cold_streak",
    "avg_pa_per_game",
    "pitcher_hr_per_9", "pitcher_barrel_rate_allowed",
    "pitcher_hard_hit_allowed", "pitcher_fb_pct",
    "pitcher_hand", "platoon_advantage",
    "park_factor",
]

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: uv run train_model.py")
    print(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ── Odds API ──────────────────────────────────────────────────────────────────

def implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def get_games() -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        params={"apiKey": API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    games = resp.json()
    print(f"Found {len(games)} games today from Odds API")
    return games


def get_props(game: dict) -> list[dict]:
    game_id   = game["id"]
    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")

    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events/{game_id}/odds",
        params={
            "apiKey":     API_KEY,
            "regions":    "us",
            "markets":    ",".join(MARKETS),
            "oddsFormat": "american",
            "bookmakers": ",".join(BOOKS),
        },
        timeout=10,
    )
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  {away_team} @ {home_team} — quota remaining: {remaining}")
    resp.raise_for_status()

    rows = []
    for bm in resp.json().get("bookmakers", []):
        for market in bm.get("markets", []):
            for outcome in market.get("outcomes", []):
                price = outcome["price"]
                rows.append({
                    "bookmaker":    bm["key"],
                    "market":       market["key"],
                    "player":       outcome.get("description", ""),
                    "side":         outcome["name"],
                    "line":         outcome.get("point"),
                    "odds":         price,
                    "implied_prob": round(implied_prob(price) * 100, 1),
                    "home_team":    home_team,
                    "away_team":    away_team,
                })
    return rows


# ── Pitcher assignment ────────────────────────────────────────────────────────

def build_player_pitcher_map(
    props_df: pd.DataFrame,
    mlb_games: list[dict],
    player_teams: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """
    Maps each prop player to their opposing starting pitcher + venue.

    Uses the MLB Stats API roster to determine which team each batter
    plays for, then assigns the correct opposing pitcher (home batters
    face the away starter; away batters face the home starter).

    Falls back to the old home-batter assumption if the player isn't
    found in the roster.
    """
    # Build team name → game lookup (both home and away)
    team_game: dict[str, dict] = {}
    for g in mlb_games:
        team_game[g["home_team"]] = g
        team_game[g["away_team"]] = g

    result: dict[str, tuple[str, str]] = {}
    fallback_players: list[str] = []

    for _, row in props_df.drop_duplicates("player").iterrows():
        player    = row["player"]
        player_tm = player_teams.get(player)

        if player_tm and player_tm in team_game:
            game  = team_game[player_tm]
            venue = game.get("venue", "")
            # Correct opposing pitcher based on player's actual team
            if game["home_team"] == player_tm:
                pitcher = game.get("away_starter", "") or game.get("home_starter", "")
            else:
                pitcher = game.get("home_starter", "") or game.get("away_starter", "")
            result[player] = (pitcher, venue)
        else:
            # Fallback: use the game from the Odds API team names + assume home batter
            fallback_players.append(player)
            home_team = row["home_team"]
            game      = team_game.get(home_team, {})
            pitcher   = game.get("away_starter", "") if game else ""
            venue     = game.get("venue", "") if game else ""
            result[player] = (pitcher, venue)

    if fallback_players:
        shown = ", ".join(fallback_players[:5])
        extra = f" +{len(fallback_players)-5} more" if len(fallback_players) > 5 else ""
        print(f"  [{len(fallback_players)} players used fallback pitcher assignment]: {shown}{extra}")

    return result


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_props(
    df: pd.DataFrame,
    model,
    mlb_games: list[dict],
    player_teams: dict[str, str],
) -> pd.DataFrame:
    player_pitcher_map = build_player_pitcher_map(df, mlb_games, player_teams)

    unique_players  = df["player"].unique()
    unique_pitchers = {p for p, _ in player_pitcher_map.values() if p}

    print(f"\nFetching batter features for {len(unique_players)} players...")
    batter_cache: dict[str, dict] = {}
    failed_batters = 0
    for player in unique_players:
        print(f"  → {player}")
        feats = get_batter_features(player, TODAY)
        batter_cache[player] = feats
        if not feats:
            failed_batters += 1
        time.sleep(0.3)
    print(f"  Batter lookup: {len(unique_players)-failed_batters}/{len(unique_players)} found")

    print(f"\nFetching pitcher features for {len(unique_pitchers)} starters...")
    pitcher_cache: dict[str, dict] = {}
    failed_pitchers = 0
    for pitcher in unique_pitchers:
        print(f"  → {pitcher}")
        feats = get_pitcher_features(pitcher, TODAY)
        pitcher_cache[pitcher] = feats
        if feats == {} :
            failed_pitchers += 1
        time.sleep(0.3)
    print(f"  Pitcher lookup: {len(unique_pitchers)-failed_pitchers}/{len(unique_pitchers)} found")

    model_probs   = []
    pitcher_names = []

    for _, row in df.iterrows():
        player         = row["player"]
        pitcher, venue = player_pitcher_map.get(player, ("", ""))

        b_feats = batter_cache.get(player, {}).copy()
        p_feats = pitcher_cache.get(pitcher, {}).copy()

        # Extract metadata (not model features)
        batter_stand = b_feats.pop("_batter_stand", "R")

        feats = {**b_feats, **p_feats}

        # Platoon advantage — computed from batter stand + pitcher hand
        p_hand        = feats.get("pitcher_hand", 0)
        pitcher_throws = "L" if p_hand == 1 else "R"
        if batter_stand == "S":
            feats["platoon_advantage"] = 0.5
        elif batter_stand != pitcher_throws:
            feats["platoon_advantage"] = 1.0
        else:
            feats["platoon_advantage"] = 0.0

        feats["park_factor"] = get_park_factor(venue)

        feat_vec = pd.DataFrame([{k: feats.get(k, np.nan) for k in FEATURES}])

        if feat_vec.isnull().all(axis=1).iloc[0]:
            model_probs.append(np.nan)
        else:
            prob = model.predict_proba(feat_vec)[0][1]
            model_probs.append(round(prob * 100, 1))

        pitcher_names.append(pitcher)

    df = df.copy()
    df["starter"]    = pitcher_names
    df["model_prob"] = model_probs
    df["edge"]       = (df["model_prob"] - df["implied_prob"]).round(1)
    return df


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    cols = ["player", "starter", "bookmaker", "odds", "implied_prob", "model_prob", "edge"]

    scored = df[df["model_prob"].notna()]
    skipped = df["model_prob"].isna().sum()
    if skipped:
        print(f"\n  [{skipped} props skipped — batter data unavailable]")

    print("\n" + "=" * 90)
    print("ALL HR PROPS — OVERS (model scored)")
    print("=" * 90)
    print(scored[cols].sort_values("edge", ascending=False).to_string(index=False))

    playable = scored[scored["edge"] >= EDGE_THRESHOLD]
    if playable.empty:
        print(f"\nNo plays above {EDGE_THRESHOLD}% edge today.")
    else:
        print(f"\n{'=' * 90}")
        print(f"✅  PLAYABLE EDGES  (model edge ≥ {EDGE_THRESHOLD}%)")
        print("=" * 90)
        print(playable[cols].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model()

    print("\nFetching today's starters from MLB Stats API...")
    mlb_games = get_game_details_today(TODAY)
    for g in mlb_games:
        print(f"  {g['away_team']} @ {g['home_team']} "
              f"| {g['away_starter'] or 'TBD'} vs {g['home_starter'] or 'TBD'} "
              f"| {g['venue']}")

    print("\nFetching player-team roster...")
    player_teams = get_roster_player_teams(int(TODAY[:4]))

    odds_games = get_games()
    all_rows   = []
    for game in odds_games:
        all_rows.extend(get_props(game))
        time.sleep(1)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No props found — check your API tier supports player props.")
    else:
        overs = (
            df[df["side"] == "Over"]
            .sort_values("implied_prob")
            .drop_duplicates(subset=["player", "bookmaker"])
            .reset_index(drop=True)
        )

        scored = score_props(overs, model, mlb_games, player_teams)
        print_summary(scored)
        scored.to_csv("props_with_edge.csv", index=False)
        print("\nSaved → props_with_edge.csv")
