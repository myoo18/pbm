"""
MLB Props — Current Playable Lines
Pulls live HR props from the sharpest/most liquid books only.

Setup:
    uv add requests pandas python-dotenv
    echo "ODDS_API_KEY=your_key" > .env
    uv run mlb_props_now.py
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "baseball_mlb"

# Sharpest/most liquid books — best for finding real edges
BOOKS = [
    "pinnacle",     # sharpest line in the market
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "bet365",
]

MARKETS = [
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "pitcher_strikeouts",
]


def implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def get_games() -> list[str]:
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        params={"apiKey": API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    games = resp.json()
    print(f"Found {len(games)} games today")
    return [g["id"] for g in games]


def get_props(game_id: str) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events/{game_id}/odds",
        params={
            "apiKey":      API_KEY,
            "regions":     "us",
            "markets":     ",".join(MARKETS),
            "oddsFormat":  "american",
            "bookmakers":  ",".join(BOOKS),
        },
        timeout=10,
    )
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  game {game_id} — quota remaining: {remaining}")
    resp.raise_for_status()

    rows = []
    data = resp.json()
    for bm in data.get("bookmakers", []):
        for market in bm.get("markets", []):
            for outcome in market.get("outcomes", []):
                price = outcome["price"]
                rows.append({
                    "bookmaker":   bm["key"],
                    "market":      market["key"],
                    "player":      outcome.get("description", ""),
                    "side":        outcome["name"],
                    "line":        outcome.get("point"),
                    "odds":        price,
                    "implied_prob": round(implied_prob(price) * 100, 1),
                })
    return rows


if __name__ == "__main__":
    game_ids = get_games()

    all_rows = []
    for gid in game_ids:
        all_rows.extend(get_props(gid))

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No props found — check your API tier supports player props.")
    else:
        # Show only "Over" side — that's the side we're betting for HR props
        df = df[df["side"] == "Over"].sort_values(
            ["player", "market", "implied_prob"]
        )
        print(df.to_string(index=False))
        df.to_csv("props.csv", index=False)
        print(f"\nSaved to props_now.csv")