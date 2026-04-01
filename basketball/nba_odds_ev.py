"""
nba_odds_ev.py
--------------
Find the highest-EV NBA player props using The Odds API.

Two complementary EV methods:
  1. Consensus EV  — compares each book's offered odds against the
                     vig-removed market consensus across all fetched books.
                     Pure odds-based; no external stats library required.
  2. Model EV      — uses recent nba_api game logs to estimate true
                     probability, then compares to book odds.
                     Requires:  pip install nba-api scipy

Setup:
    pip install requests python-dotenv pandas
    Optional: pip install nba-api scipy
    echo "ODDS_API_KEY=your_key_here" > .env

Quick start:
    python nba_odds_ev.py                        # consensus EV, all markets
    python nba_odds_ev.py --method model         # use nba_api model EV
    python nba_odds_ev.py --min-ev 2.0           # only show > 2 % edge
    python nba_odds_ev.py --markets pts reb ast  # limit markets

Programmatic use:
    from nba_odds_ev import OddsEVScanner
    scanner = OddsEVScanner()
    results = scanner.scan()
    scanner.print_report(results)
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── optional nba_api / scipy ──────────────────────────────────────────────────
try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players as _nba_players_static
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

API_KEY  = os.getenv("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_nba"
SEASON   = "2024-25"

# Books to include when building consensus (more = better consensus)
DEFAULT_BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "bet365"
]

# All supported player-prop markets and their shorthand aliases
MARKET_INFO: dict[str, dict] = {
    "player_points":                   {"stat": "PTS",  "alias": ["pts",  "points"]},
    "player_rebounds":                 {"stat": "REB",  "alias": ["reb",  "rebounds"]},
    "player_assists":                  {"stat": "AST",  "alias": ["ast",  "assists"]},
    "player_threes":                   {"stat": "3PM",  "alias": ["3pm",  "threes", "3s"]},
    "player_steals":                   {"stat": "STL",  "alias": ["stl",  "steals"]},
    "player_blocks":                   {"stat": "BLK",  "alias": ["blk",  "blocks"]},
    "player_turnovers":                {"stat": "TOV",  "alias": ["tov",  "turnovers"]},
    "player_points_rebounds_assists":  {"stat": "PRA",  "alias": ["pra"]},
    "player_points_rebounds":          {"stat": "PR",   "alias": ["pr"]},
    "player_points_assists":           {"stat": "PA",   "alias": ["pa"]},
    "player_rebounds_assists":         {"stat": "RA",   "alias": ["ra"]},
}

# NBA-API column mapping (same as nba_props_ev.py)
_NBA_COL: dict[str, str] = {
    "PTS": "PTS", "REB": "REB", "AST": "AST",
    "STL": "STL", "BLK": "BLK", "TOV": "TOV",
    "3PM": "FG3M", "PRA": "PRA", "PR": "PR", "PA": "PA", "RA": "RA",
}
_POISSON_STATS = {"STL", "BLK", "TOV", "3PM"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BookLine:
    """One book's over/under odds for a specific player + market + line."""
    book:       str
    over_odds:  int
    under_odds: int

    @property
    def over_decimal(self) -> float:
        return _american_to_decimal(self.over_odds)

    @property
    def under_decimal(self) -> float:
        return _american_to_decimal(self.under_odds)

    @property
    def over_implied(self) -> float:
        return _implied_prob(self.over_odds)

    @property
    def under_implied(self) -> float:
        return _implied_prob(self.under_odds)

    @property
    def vig_free_over(self) -> float:
        """Vig-removed fair probability for the Over."""
        total = self.over_implied + self.under_implied
        return self.over_implied / total

    @property
    def vig_free_under(self) -> float:
        return self.under_implied / (self.over_implied + self.under_implied)


@dataclass
class PropKey:
    """Unique identity for a player + market + line combination."""
    player: str
    market: str          # e.g. "player_points"
    stat:   str          # e.g. "PTS"
    line:   float

    def __hash__(self):
        return hash((self.player.lower(), self.market, self.line))

    def __eq__(self, other):
        return (
            self.player.lower() == other.player.lower()
            and self.market == other.market
            and self.line == other.line
        )


@dataclass
class EVOpportunity:
    """A single +EV opportunity at a specific book."""
    player:         str
    stat:           str
    line:           float
    side:           str          # "Over" | "Under"
    book:           str
    american_odds:  int
    decimal_odds:   float
    consensus_prob: float        # vig-free consensus probability
    model_prob:     Optional[float] = None   # nba_api model probability
    ev_pct:         float = 0.0              # consensus-based EV %
    model_ev_pct:   Optional[float] = None   # model-based EV %
    edge:           float = 0.0              # model_prob − consensus_prob
    num_books:      int = 0


@dataclass
class PropSummary:
    """All analysis for one PropKey."""
    key:            PropKey
    books:          list[BookLine]
    consensus_over: float         # vig-free average P(Over) across books
    consensus_under: float
    opportunities:  list[EVOpportunity] = field(default_factory=list)
    model_prob_over: Optional[float] = None
    error:          Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Odds math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _american_to_decimal(odds: int) -> float:
    if odds > 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def _implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _ev_pct(prob: float, decimal_odds: float) -> float:
    return prob * decimal_odds - 1.0


def _consensus_probs(book_lines: list[BookLine]) -> tuple[float, float]:
    """
    Average vig-free over/under probabilities across all books.
    This is the 'market consensus' — the best estimate of true probability.
    """
    if not book_lines:
        return 0.5, 0.5
    over_probs  = [bl.vig_free_over  for bl in book_lines]
    under_probs = [bl.vig_free_under for bl in book_lines]
    return statistics.mean(over_probs), statistics.mean(under_probs)


# ─────────────────────────────────────────────────────────────────────────────
# The Odds API client
# ─────────────────────────────────────────────────────────────────────────────

class OddsAPIClient:
    """Thin wrapper around The Odds API v4."""

    def __init__(self, api_key: str = API_KEY):
        if not api_key:
            raise ValueError(
                "ODDS_API_KEY not set. Add it to your .env file:\n"
                "  ODDS_API_KEY=your_key_here"
            )
        self._key = api_key
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"

    def _get(self, path: str, params: dict) -> dict | list:
        params["apiKey"] = self._key
        resp = self._session.get(f"{BASE_URL}{path}", params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"  [{path.split('/')[-1]}] quota used={used} remaining={remaining}")
        resp.raise_for_status()
        return resp.json()

    def get_events(self) -> list[dict]:
        """Return today's NBA events."""
        return self._get(f"/sports/{SPORT}/events", {"dateFormat": "iso"})

    def get_event_props(
        self,
        event_id: str,
        markets: list[str],
        books: list[str],
    ) -> dict:
        return self._get(
            f"/sports/{SPORT}/events/{event_id}/odds",
            {
                "regions":    "us",
                "markets":    ",".join(markets),
                "oddsFormat": "american",
                "bookmakers": ",".join(books),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# NBA game-log model (optional — requires nba_api)
# ─────────────────────────────────────────────────────────────────────────────

class _GameLogModel:
    """Estimates P(stat > line) from recent NBA game logs."""

    def __init__(self, request_delay: float = 0.7):
        if not HAS_NBA_API:
            raise ImportError("pip install nba-api")
        self._delay = request_delay
        self._cache: dict[str, list[dict]] = {}

    def prob_over(self, player: str, stat: str, line: float, num_games: int = 20) -> float:
        games = self._get_log(player)
        col = _NBA_COL.get(stat.upper())
        if col is None:
            raise ValueError(f"Unknown stat: {stat}")

        values = [float(g[col]) for g in games[:num_games] if col in g]
        if len(values) < 5:
            raise ValueError(f"Only {len(values)} games found for {player}")

        mean = statistics.mean(values)
        std  = statistics.pstdev(values)

        if stat.upper() in _POISSON_STATS:
            return self._poisson_over(mean, line)
        return self._normal_over(mean, std, line)

    def _get_log(self, player: str) -> list[dict]:
        key = player.lower()
        if key not in self._cache:
            pid = self._find_id(player)
            time.sleep(self._delay)
            log = playergamelog.PlayerGameLog(
                player_id=pid, season=SEASON, season_type_all_star="Regular Season"
            )
            df = log.get_data_frames()[0]
            games = df.to_dict(orient="records")
            for g in games:
                g["PRA"] = g["PTS"] + g["REB"] + g["AST"]
                g["PR"]  = g["PTS"] + g["REB"]
                g["PA"]  = g["PTS"] + g["AST"]
                g["RA"]  = g["REB"] + g["AST"]
            self._cache[key] = games
        return self._cache[key]

    @staticmethod
    def _find_id(name: str) -> int:
        name_lower = name.lower()
        matches = [
            p for p in _nba_players_static.get_players()
            if name_lower in p["full_name"].lower()
        ]
        if not matches:
            raise ValueError(f"Player not found: '{name}'")
        exact = [p for p in matches if p["full_name"].lower() == name_lower]
        return (exact or matches)[0]["id"]

    @staticmethod
    def _normal_over(mean: float, std: float, line: float) -> float:
        if std < 0.01:
            return 1.0 if mean > line else 0.0
        if HAS_SCIPY:
            return float(1.0 - _scipy_stats.norm.cdf(line, loc=mean, scale=std))
        z = (line - mean) / (std * math.sqrt(2))
        return 0.5 * math.erfc(z)

    @staticmethod
    def _poisson_over(lam: float, line: float) -> float:
        k_max = int(line)
        if HAS_SCIPY:
            return float(1.0 - _scipy_stats.poisson.cdf(k_max, lam))
        cdf = 0.0
        log_lam = math.log(lam) if lam > 0 else -math.inf
        log_fact = 0.0
        for k in range(k_max + 1):
            if k > 0:
                log_fact += math.log(k)
            cdf += math.exp(k * log_lam - lam - log_fact)
        return max(0.0, 1.0 - cdf)


# ─────────────────────────────────────────────────────────────────────────────
# Main scanner
# ─────────────────────────────────────────────────────────────────────────────

class OddsEVScanner:
    """
    Fetches live NBA player props from The Odds API and ranks them by EV.

    Args:
        api_key:      The Odds API key (defaults to ODDS_API_KEY env var).
        books:        Which bookmakers to include.
        markets:      List of market keys or aliases (e.g. ["pts", "reb"]).
                      Defaults to all supported markets.
        min_books:    Minimum number of books required to build a valid consensus.
        use_model:    If True, also run the nba_api stat model.
        model_games:  Number of recent games used in the stat model.
        min_ev_pct:   Minimum EV % to include in results (e.g. 2.0 for 2 %).
    """

    def __init__(
        self,
        api_key:     str = API_KEY,
        books:       list[str] = None,
        markets:     list[str] = None,
        min_books:   int = 2,
        use_model:   bool = False,
        model_games: int = 20,
        min_ev_pct:  float = 0.0,
    ):
        self._client    = OddsAPIClient(api_key)
        self._books     = books or DEFAULT_BOOKS
        self._markets   = self._resolve_markets(markets)
        self._min_books = min_books
        self._min_ev    = min_ev_pct / 100.0
        self._model: Optional[_GameLogModel] = None
        self._model_games = model_games
        if use_model:
            if not HAS_NBA_API:
                print("⚠  nba-api not installed — skipping model EV. pip install nba-api")
            else:
                self._model = _GameLogModel()

    # ── public API ────────────────────────────────────────────────────────────

    def scan(self) -> list[EVOpportunity]:
        """
        Fetch today's props and return all +EV opportunities sorted by EV%.
        """
        print(f"\nFetching NBA events...")
        events = self._client.get_events()
        if not events:
            print("No NBA games today.")
            return []
        print(f"Found {len(events)} game(s). Fetching props...")

        prop_map: dict[PropKey, list[BookLine]] = defaultdict(list)

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            print(f"\n  {away} @ {home}")
            try:
                data = self._client.get_event_props(
                    event["id"], self._markets, self._books
                )
            except requests.HTTPError as exc:
                print(f"  ⚠ HTTP {exc.response.status_code} — skipping")
                continue

            self._parse_event(data, prop_map)

        print(f"\nParsed {len(prop_map)} unique player/market/line combinations.")
        summaries = self._build_summaries(prop_map)
        opportunities = self._rank_opportunities(summaries)
        return opportunities

    def print_report(
        self,
        opportunities: list[EVOpportunity],
        top_n: int = 30,
    ) -> None:
        SEP  = "─" * 76
        DSEP = "═" * 76

        print(f"\n{DSEP}")
        print(f"  NBA PLAYER PROPS — EXPECTED VALUE REPORT (Consensus Method)")
        if self._model:
            print(f"  Model EV (nba_api) shown where available")
        print(f"{DSEP}")

        shown = [o for o in opportunities if o.ev_pct >= self._min_ev][:top_n]

        if not shown:
            print("\n  No +EV opportunities found with current filters.\n")
            print(f"{DSEP}\n")
            return

        for o in shown:
            odds_str = f"+{o.american_odds}" if o.american_odds > 0 else str(o.american_odds)
            ev_str   = f"{o.ev_pct * 100:+.2f}%"
            con_str  = f"{o.consensus_prob * 100:.1f}%"
            tag      = "✅" if o.ev_pct > 0 else "·"

            model_str = ""
            if o.model_ev_pct is not None:
                model_tag = "✅" if o.model_ev_pct > 0 else "❌"
                model_str = (
                    f"  │  model: {o.model_prob * 100:.1f}% "
                    f"EV={o.model_ev_pct * 100:+.2f}% {model_tag}"
                )

            print(f"\n{SEP}")
            print(
                f"  {tag} {o.player:<28}  {o.stat} {o.line}  {o.side.upper()}"
            )
            print(
                f"     {o.book:<16}  {odds_str:>6}  "
                f"consensus={con_str}  EV={ev_str}  "
                f"({o.num_books} books){model_str}"
            )

        print(f"\n{DSEP}")
        print(f"  Showing {len(shown)} opportunities  |  min EV filter: {self._min_ev * 100:.1f}%")
        print(f"{DSEP}\n")

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_event(
        self,
        data: dict,
        prop_map: dict[PropKey, list[BookLine]],
    ) -> None:
        for bm in data.get("bookmakers", []):
            book = bm["key"]
            for market in bm.get("markets", []):
                mkey = market["key"]
                stat = MARKET_INFO.get(mkey, {}).get("stat")
                if not stat:
                    continue

                # Group outcomes by player + line
                player_lines: dict[tuple[str, float], dict[str, int]] = defaultdict(dict)
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "").strip()
                    side   = outcome["name"]   # "Over" | "Under"
                    line   = outcome.get("point")
                    price  = outcome["price"]
                    if player and line is not None and side in ("Over", "Under"):
                        player_lines[(player, float(line))][side] = int(price)

                for (player, line), sides in player_lines.items():
                    if "Over" not in sides or "Under" not in sides:
                        continue
                    pkey = PropKey(player=player, market=mkey, stat=stat, line=line)
                    prop_map[pkey].append(
                        BookLine(
                            book=book,
                            over_odds=sides["Over"],
                            under_odds=sides["Under"],
                        )
                    )

    # ── analysis ──────────────────────────────────────────────────────────────

    def _build_summaries(
        self,
        prop_map: dict[PropKey, list[BookLine]],
    ) -> list[PropSummary]:
        summaries = []
        for key, book_lines in prop_map.items():
            if len(book_lines) < self._min_books:
                continue

            cons_over, cons_under = _consensus_probs(book_lines)
            summary = PropSummary(
                key=key,
                books=book_lines,
                consensus_over=cons_over,
                consensus_under=cons_under,
            )

            # Optional: model probability
            if self._model:
                try:
                    summary.model_prob_over = self._model.prob_over(
                        key.player, key.stat, key.line, self._model_games
                    )
                except Exception as exc:
                    summary.error = str(exc)

            summaries.append(summary)

        return summaries

    def _rank_opportunities(self, summaries: list[PropSummary]) -> list[EVOpportunity]:
        opps: list[EVOpportunity] = []

        for s in summaries:
            for bl in s.books:
                for side, cons_prob, dec_odds, am_odds in [
                    ("Over",  s.consensus_over,  bl.over_decimal,  bl.over_odds),
                    ("Under", s.consensus_under, bl.under_decimal, bl.under_odds),
                ]:
                    ev = _ev_pct(cons_prob, dec_odds)

                    model_ev  = None
                    model_prob = None
                    if s.model_prob_over is not None:
                        mp = s.model_prob_over if side == "Over" else 1.0 - s.model_prob_over
                        model_prob = mp
                        model_ev   = _ev_pct(mp, dec_odds)

                    opps.append(EVOpportunity(
                        player=s.key.player,
                        stat=s.key.stat,
                        line=s.key.line,
                        side=side,
                        book=bl.book,
                        american_odds=am_odds,
                        decimal_odds=dec_odds,
                        consensus_prob=cons_prob,
                        model_prob=model_prob,
                        ev_pct=ev,
                        model_ev_pct=model_ev,
                        edge=(model_prob - cons_prob) if model_prob is not None else 0.0,
                        num_books=len(s.books),
                    ))

        # Sort: primary = consensus EV desc, secondary = model EV desc
        opps.sort(
            key=lambda o: (
                o.ev_pct,
                o.model_ev_pct if o.model_ev_pct is not None else -1.0,
            ),
            reverse=True,
        )
        return opps

    # ── market resolution ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_markets(markets: list[str] | None) -> list[str]:
        if not markets:
            return list(MARKET_INFO.keys())

        resolved = set()
        alias_map: dict[str, str] = {}
        for mkey, info in MARKET_INFO.items():
            alias_map[mkey] = mkey
            for a in info["alias"]:
                alias_map[a.lower()] = mkey

        for m in markets:
            key = alias_map.get(m.lower())
            if key:
                resolved.add(key)
            else:
                print(f"⚠  Unknown market alias '{m}' — skipping. "
                      f"Valid: {list(alias_map.keys())}")
        return list(resolved)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find the highest-EV NBA player props via The Odds API."
    )
    p.add_argument(
        "--min-ev", type=float, default=2.0, metavar="PCT",
        help="Minimum EV %% to display (default: 2.0)",
    )
    p.add_argument(
        "--top", type=int, default=25,
        help="Maximum rows to show in report (default: 25)",
    )
    p.add_argument(
        "--markets", nargs="+", default=None,
        metavar="MARKET",
        help=(
            "Markets to scan. Aliases: pts, reb, ast, 3pm, stl, blk, tov, "
            "pra, pr, pa, ra. Default: all."
        ),
    )
    p.add_argument(
        "--books", nargs="+", default=None,
        metavar="BOOK",
        help="Bookmakers to include (default: dk, fd, betmgm, caesars, bet365, pinnacle…)",
    )
    p.add_argument(
        "--min-books", type=int, default=2,
        help="Minimum number of books required per prop (default: 2)",
    )
    p.add_argument(
        "--model", action="store_true",
        help="Also compute model EV using nba_api game logs (slower)",
    )
    p.add_argument(
        "--model-games", type=int, default=20,
        help="Number of recent games for the stat model (default: 20)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Show all props including -EV (overrides --min-ev)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    min_ev = 0.0 if args.all else args.min_ev

    scanner = OddsEVScanner(
        books=args.books,
        markets=args.markets,
        min_books=args.min_books,
        use_model=args.model,
        model_games=args.model_games,
        min_ev_pct=min_ev,
    )

    results = scanner.scan()
    scanner.print_report(results, top_n=args.top)


if __name__ == "__main__":
    main()
