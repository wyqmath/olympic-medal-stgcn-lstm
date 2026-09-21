# Olympic Medal Prediction with an STGCN-LSTM Model

Code and processed data for the manuscript "STGCN-LSTM for Olympic Medal Prediction: Relational Forecasting and Hosting-Effect Analysis". The model forecasts country-level medal counts with a graph attention network over a country-similarity graph, is evaluated in a leakage-free nested walk-forward design across 13 editions (1976-2024), and the hosting effect is quantified with panel econometrics and causal estimators.

## Layout

- `data/panel_clean.csv` — processed country-edition panel (1896-2024, 1,435 rows), the input to every analysis script
- `data/summerOly_hosts_with_ioc.csv` — host city/country and IOC code per edition
- `code/` — all pipeline scripts; outputs are written under `code/gen/`

Scripts: panel construction (`build_panel.py`), the model (`stgcn_predict.py`, `stgcn_variants.py`), the nested walk-forward evaluation (`run_nested_variants.py`, `nested_backtest.py`, `backtest_period.py`), baselines (`baselines_compare.py`), the 2028 projection (`forward_2028.py`), sensitivity and scenarios (`sensitivity_grid.py`, `scenario_gdp.py`, `boundary_by_gdp.py`), econometrics (`feols.py`, `econometrics_*.py`, `causal_host_effect.py`), and the manuscript figures (`fig_*.py`).

## Data

`data/panel_clean.csv` holds one row per country and Summer Games edition with the medal counts (gold/silver/bronze/total), a host flag, and GDP, GDP per capita and population from the Maddison Project Database 2023 (units in the column names).

The raw sources are not redistributed: the Olympic results files come from MCM 2025 Problem C and are available from COMAP, and the macroeconomic series come from the Maddison Project Database 2023. The panel itself can be reconstructed with `python code/build_panel.py`.

## Reproducing the results

Run the scripts from `code/`; each writes its outputs under `gen/`. The order is `run_nested_variants.py` → `nested_backtest.py` → `backtest_period.py` → `baselines_compare.py` → `forward_2028.py` → `sensitivity_grid.py` → `scenario_gdp.py` → the `econometrics_*.py` scripts → `causal_host_effect.py` → `boundary_by_gdp.py`, with the `fig_*.py` scripts redrawing the manuscript figures. The neural steps use a GPU when available; the econometric steps are CPU-only. Trained weights are not distributed — every reported number retrains from `data/panel_clean.csv` with the fixed seed set inside the scripts.
