# pbm
predictive baseball model to generate expected values for current player and team matchups

install uv 
https://docs.astral.sh/uv/getting-started/installation/

uv venv
uv pip install -r requirements.txt

get the odss api key from
https://the-odds-api.com/


is this shit free? : 
https://developer.sportradar.com/baseball/reference/mlb-overview

get the weather api 
https://api.weather.gov/

training paramaters for 0-0-3
  ┌───────────────────────┬────────┬───────┬──────────────────────────────────────────────────────────────────────┐
  │         Param         │ Before │ After │                                Reason                                │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ min_child_weight      │ 30     │ 8     │ Was blocking almost every split — the main cause of 14-tree collapse │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ early_stopping_rounds │ 30     │ 75    │ More patience before giving up                                       │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ learning_rate         │ 0.03   │ 0.015 │ Slower steps → smoother loss curve → less likely to plateau early    │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ n_estimators          │ 800    │ 1200  │ Lower LR needs more trees to reach the same point                    │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ max_depth             │ 4      │ 5     │ Slightly more capacity                                               │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ gamma                 │ —      │ 0.1   │ Minimum gain required to make a split — prevents noisy splits        │
  ├───────────────────────┼────────┼───────┼──────────────────────────────────────────────────────────────────────┤
  │ reg_alpha             │ —      │ 0.05  │ Light L1 regularisation                                              │
  └───────────────────────┴────────┴───────┴──────────────────────────────────────────────────────────────────────┘