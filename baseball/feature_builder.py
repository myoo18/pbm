# feature_builder.py
"""
Builds batter + pitcher feature vectors for inference (daily scoring).
"""

from __future__ import annotations

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup, cache

cache.enable()

# ── League average fallbacks ──────────────────────────────────────────────────

PITCHER_DEFAULTS = {
    "pitcher_hr_per_9":            1.10,
    "pitcher_barrel_rate_allowed": 0.065,
    "pitcher_hard_hit_allowed":    0.360,
    "pitcher_fb_pct":              0.560,
    "pitcher_hand":                0,     # default RHP
}

# ── Park factors (all 30 stadiums) ───────────────────────────────────────────

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
    "sutter health park":       0.98,
    "angel stadium":            0.97,
    "progressive field":        0.97,
    "kauffman stadium":         0.96,
    "target field":             0.95,
    "minute maid park":         0.95,
    "dodger stadium":           0.95,
    "citi field":               0.94,
    "busch stadium":            0.93,
    "tropicana field":          0.92,
    "pnc park":                 0.91,
    "comerica park":            0.90,
    "petco park":               0.87,
    "oracle park":              0.88,
    "t-mobile park":            0.86,
}

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"


# ── MLB Stats API ─────────────────────────────────────────────────────────────

def get_todays_starters(game_date: str | None = None) -> dict[str, str]:
    """Returns {team_name: starter_full_name} for every game today."""
    target = game_date or date.today().strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            f"{MLB_STATS_API}/schedule",
            params={
                "sportId": 1,
                "date":    target,
                "hydrate": "probablePitcher,team",
                "fields":  "dates,games,teams,probablePitcher,fullName,team,name",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[MLB API error] {e}")
        return {}

    starters: dict[str, str] = {}
    for game_day in data.get("dates", []):
        for game in game_day.get("games", []):
            for side in ("home", "away"):
                team_info = game.get("teams", {}).get(side, {})
                team_name = team_info.get("team", {}).get("name", "")
                pitcher   = team_info.get("probablePitcher", {}).get("fullName", "")
                if team_name and pitcher:
                    starters[team_name] = pitcher

    print(f"  Found {len(starters)} starters from MLB Stats API")
    return starters


def get_game_details_today(game_date: str | None = None) -> list[dict]:
    """
    Returns list of:
        {home_team, away_team, home_starter, away_starter, venue}
    for every game today.
    """
    target = game_date or date.today().strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            f"{MLB_STATS_API}/schedule",
            params={
                "sportId": 1,
                "date":    target,
                "hydrate": "probablePitcher,team,venue",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[MLB API error] {e}")
        return []

    games = []
    for game_day in data.get("dates", []):
        for game in game_day.get("games", []):
            teams = game.get("teams", {})
            games.append({
                "home_team":    teams.get("home", {}).get("team", {}).get("name", ""),
                "away_team":    teams.get("away", {}).get("team", {}).get("name", ""),
                "home_starter": teams.get("home", {}).get("probablePitcher", {}).get("fullName", ""),
                "away_starter": teams.get("away", {}).get("probablePitcher", {}).get("fullName", ""),
                "venue":        game.get("venue", {}).get("name", "").lower().strip(),
            })

    return games


def get_roster_player_teams(season: int) -> dict[str, str]:
    """
    Returns {player_full_name: team_name} for all active MLB players.
    Used to correctly assign opposing pitchers in scoring.
    """
    try:
        resp = requests.get(
            f"{MLB_STATS_API}/sports/1/players",
            params={"season": season, "fields": "people,fullName,currentTeam,id"},
            timeout=15,
        )
        resp.raise_for_status()
        result = {}
        for person in resp.json().get("people", []):
            name = person.get("fullName", "")
            team = person.get("currentTeam", {}).get("name", "")
            if name and team:
                result[name] = team
        print(f"  Loaded {len(result)} player-team mappings from roster API")
        return result
    except Exception as e:
        print(f"[Roster API error] {e} — pitcher assignment may be less accurate")
        return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _streak(arr: np.ndarray, value: int) -> int:
    count = 0
    for v in reversed(arr):
        if v == value:
            count += 1
        else:
            break
    return count


def _lookup_id(name: str) -> int | None:
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    first, last = parts[0], " ".join(parts[1:])
    try:
        res = playerid_lookup(last, first, fuzzy=True)
        if res.empty:
            return None
        return int(res.iloc[0]["key_mlbam"])
    except Exception:
        return None


# ── Batter features ───────────────────────────────────────────────────────────

def get_batter_features(player_name: str, as_of: str) -> dict:
    """
    Returns rolling Statcast features for a batter.
    Includes LHP/RHP split rates and batter_stand metadata.
    Returns {} if player not found or insufficient data.

    Note: '_batter_stand' is metadata (not a model feature) used by the
    scoring step to compute platoon_advantage.
    """
    mlbam_id = _lookup_id(player_name)
    if mlbam_id is None:
        print(f"    [batter not found] {player_name}")
        return {}

    start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        df = statcast_batter(start, as_of, player_id=mlbam_id)
    except Exception as e:
        print(f"    [statcast error] {player_name}: {e}")
        return {}

    if df.empty or len(df) < 10:
        print(f"    [thin sample] {player_name} — {len(df)} pitches, skipping")
        return {}

    df["is_hr"]     = (df["events"] == "home_run").fillna(False).astype(int)
    df["is_pa"]     = df["events"].notna().astype(int)
    df["is_barrel"] = (df.get("launch_speed_angle", pd.Series(dtype=float)) == 6).fillna(False).astype(int)
    df["is_hard"]   = (df["launch_speed"] >= 95).fillna(False).astype(int)
    df["is_sweet"]  = df["launch_angle"].between(8, 32).fillna(False).astype(int)

    daily = (
        df.groupby("game_date")
        .agg(
            hrs=("is_hr", "sum"),
            pas=("is_pa", "sum"),
            avg_ev=("launch_speed", "mean"),
            barrels=("is_barrel", "sum"),
            hard=("is_hard", "sum"),
            sweet=("is_sweet", "sum"),
            opp_hand=("p_throws", lambda x: "L" if (x == "L").mean() > 0.5 else "R"),
            batter_stand=("stand", "first"),
        )
        .sort_index()
    )

    if len(daily) < 7:
        print(f"    [thin sample] {player_name} — {len(daily)} game days, skipping")
        return {}

    def rate(num, den, w):
        n = daily[num].tail(w).sum()
        d = daily[den].tail(w).sum()
        return float(n / d) if d > 0 else 0.0

    ev7  = float(daily["avg_ev"].tail(7).mean())
    ev30 = float(daily["avg_ev"].tail(30).mean())
    tot  = daily["pas"].sum()

    # LHP/RHP splits
    lhp = daily[daily["opp_hand"] == "L"]
    rhp = daily[daily["opp_hand"] == "R"]
    hr_vs_lhp = float(lhp.tail(30)["hrs"].sum() / max(lhp.tail(30)["pas"].sum(), 1))
    hr_vs_rhp = float(rhp.tail(30)["hrs"].sum() / max(rhp.tail(30)["pas"].sum(), 1))

    batter_stand = str(daily["batter_stand"].iloc[-1]) if not daily.empty else "R"

    return {
        "hr_rate_7d":          rate("hrs", "pas", 7),
        "hr_rate_15d":         rate("hrs", "pas", 15),
        "hr_rate_30d":         rate("hrs", "pas", 30),
        "hr_rate_season":      float(daily["hrs"].sum() / max(tot, 1)),
        "hr_rate_vs_lhp_30d":  hr_vs_lhp,
        "hr_rate_vs_rhp_30d":  hr_vs_rhp,
        "avg_ev_7d":           ev7,
        "avg_ev_30d":          ev30,
        "ev_trend":            ev7 - ev30,
        "barrel_rate_30d":     rate("barrels", "pas", 30),
        "hard_hit_pct":        float(daily["hard"].sum() / max(tot, 1)),
        "sweet_spot_pct":      float(daily["sweet"].sum() / max(tot, 1)),
        "avg_pa_per_game":     float(daily["pas"].mean()),
        "hot_streak":          _streak((daily["hrs"] > 0).astype(int).values, 1),
        "cold_streak":         _streak((daily["hrs"] > 0).astype(int).values, 0),
        "_batter_stand":       batter_stand,   # metadata — not a model feature
    }


# ── Pitcher features ──────────────────────────────────────────────────────────

def get_pitcher_features(pitcher_name: str, as_of: str) -> dict:
    """
    Returns rolling Statcast vulnerability metrics + pitcher handedness.
    Returns league-average defaults if pitcher not found or thin sample.
    Minimum: 50 pitches (~5 starts).
    """
    if not pitcher_name:
        return PITCHER_DEFAULTS.copy()

    mlbam_id = _lookup_id(pitcher_name)
    if mlbam_id is None:
        print(f"    [pitcher not found] {pitcher_name} — using league avg")
        return PITCHER_DEFAULTS.copy()

    start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        df = statcast_pitcher(start, as_of, player_id=mlbam_id)
    except Exception as e:
        print(f"    [statcast pitcher error] {pitcher_name}: {e}")
        return PITCHER_DEFAULTS.copy()

    if df.empty or len(df) < 50:
        print(f"    [thin sample] {pitcher_name} ({len(df)} pitches) — using league avg")
        return PITCHER_DEFAULTS.copy()

    df["is_hr"]     = (df["events"] == "home_run").fillna(False).astype(int)
    df["is_barrel"] = (df.get("launch_speed_angle", pd.Series(dtype=float)) == 6).fillna(False).astype(int)
    df["is_hard"]   = (df["launch_speed"] >= 95).fillna(False).astype(int)
    df["is_bip"]    = (df["type"] == "X").fillna(False).astype(int)
    df["is_fb"]     = df["pitch_type"].isin(["FF", "SI", "FC"]).astype(int)

    # IP: sum of unique max innings per game start
    ip = df.groupby("game_date")["inning"].max().sum()
    total_bip = df["is_bip"].sum()
    total_p   = len(df)
    p_hand    = 1 if (df["p_throws"] == "L").mean() > 0.5 else 0

    return {
        "pitcher_hr_per_9":            float(df["is_hr"].sum() / max(ip, 1) * 9),
        "pitcher_barrel_rate_allowed": float(df["is_barrel"].sum() / max(total_bip, 1)),
        "pitcher_hard_hit_allowed":    float(df["is_hard"].sum() / max(total_bip, 1)),
        "pitcher_fb_pct":              float(df["is_fb"].sum() / max(total_p, 1)),
        "pitcher_hand":                p_hand,
    }


# ── Park factor ───────────────────────────────────────────────────────────────

def get_park_factor(venue: str) -> float:
    return PARK_FACTORS.get(venue.strip().lower(), 1.00)
