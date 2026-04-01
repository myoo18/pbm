# pbm
Predictive baseball model — generates expected values for HR props and surfaces edges vs sportsbook lines.

## Setup

Install uv: https://docs.astral.sh/uv/getting-started/installation/

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env   # add your API key
```

Get an Odds API key: https://the-odds-api.com/

## Usage

```bash
# Train the model (first time only, ~15 min with GPU)
cd baseball
python train_model.py

# Score today's HR props and find edges
python baseball_market_analysis.py

# Odds calculator
python odds.py 410
python odds.py -115 -105
```

## Model — hr_model.pkl

XGBoost classifier + isotonic calibration. Predicts probability a batter hits a HR in a given game.

**Data:** Statcast pitch-level data, 2022–2026 (pybaseball)  
**Split:** Train < 2025-04-01 / Test ≥ 2025-04-01 (time-based, no leakage)

### Features (23)

| Group | Features |
|-------|----------|
| Batter HR rates | hr_rate_7d, 15d, 30d, season |
| Platoon splits | hr_rate_vs_lhp_30d, hr_rate_vs_rhp_30d |
| Exit velocity | avg_ev_7d, avg_ev_30d, ev_trend |
| Quality of contact | barrel_rate_30d, hard_hit_pct, sweet_spot_pct |
| Streaks | hot_streak, cold_streak |
| Volume | avg_pa_per_game |
| Pitcher vulnerability | pitcher_hr_per_9, barrel_rate_allowed, hard_hit_allowed, fb_pct |
| Handedness | pitcher_hand, platoon_advantage |
| Park | park_factor (all 30 stadiums) |

### Results — v0.0.4

```
Train rows : 125,869   Test rows : 41,041
HR rate    : 11.1% train / 11.3% test
Trees used : 277 / 1200
```

| Metric | Value | Notes |
|--------|-------|-------|
| Test AUC-ROC | 0.6280 | target > 0.65 |
| Train AUC-ROC | 0.6597 | |
| Overfit gap | 0.0317 | good if < 0.05 |
| Brier Score | 0.0987 | baseline 0.1006 |
| PR-AUC | 0.1638 | baseline (HR rate) 0.1134 |
| Test Log Loss | 0.3438 | |

**Calibration** — predicted vs actual HR rate by decile:

| Pred Range | Pred Avg | Actual% | N |
|------------|----------|---------|---|
| 0.0%–5.9% | 4.0% | 3.1% | 2,764 |
| 5.9%–8.5% | 6.7% | 6.1% | 5,073 |
| 8.5%–8.9% | 8.5% | 8.5% | 2,364 |
| 8.9%–10.8% | 9.2% | 8.6% | 4,743 |
| 10.8%–11.2% | 10.9% | 9.7% | 3,427 |
| 11.2%–11.4% | 11.2% | 10.6% | 3,607 |
| 11.4%–13.0% | 11.6% | 12.3% | 5,264 |
| 13.0%–15.5% | 13.3% | 14.3% | 3,661 |
| 15.5%–17.9% | 15.6% | 15.6% | 4,604 |
| 17.9%–66.7% | 19.7% | 19.0% | 5,534 |

**Betting threshold analysis:**

| Min model% | Precision | Recall | N bets |
|------------|-----------|--------|--------|
| ≥ 9% | 13.6% | 81.4% | 27,781 |
| ≥ 11% | 14.6% | 71.2% | 22,670 |
| ≥ 13% | 16.6% | 49.2% | 13,799 |
| ≥ 15% | 17.4% | 37.9% | 10,138 |
| ≥ 18% | 21.8% | 8.4% | 1,785 |
| ≥ 20% | 22.0% | 7.4% | 1,562 |

**Top features by gain:**

1. barrel_rate_30d
2. hr_rate_season
3. avg_pa_per_game
4. hard_hit_pct
5. avg_ev_30d
6. hr_rate_vs_rhp_30d
7. park_factor
8. pitcher_hr_per_9

### Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| n_estimators | 1200 | early stopped at 277 |
| max_depth | 5 | |
| learning_rate | 0.015 | |
| min_child_weight | 8 | was 30 — blocked all splits |
| subsample | 0.75 | |
| colsample_bytree | 0.75 | |
| gamma | 0.1 | min split gain |
| reg_alpha | 0.05 | L1 |
| reg_lambda | 1.0 | L2 |
| base_score | HR rate | prevents probability inflation |
| calibration | isotonic | maps scores to real probabilities |
| device | cuda | GPU training |
