# Olympic Medal Prediction with an STGCN-LSTM Model

Code and processed data for the manuscript "STGCN-LSTM for Olympic Medal Prediction: Relational Forecasting and Hosting-Effect Analysis".

## Layout

- `data/panel_clean.csv` contains 1,435 delegation-edition observations from 1896 to 2024.
- `data/summerOly_hosts_with_ioc.csv` supplies the host records for the scenario analysis.
- `code/` contains the analysis scripts. Outputs are written to `code/gen/`.

## Data

The panel combines Summer Olympic results from COMAP's 2025 MCM Problem C with GDP and population from the Maddison Project Database 2023. The raw data are available from those sources.

## Reproducing the results

Run the commands from `code/` using the processed panel.

~~~bash
python run_nested_variants.py --outdir gen/nested/variants
python nested_backtest.py --no-equivalence-check
python baselines_compare.py
python evaluate_tokyo_roc.py
~~~

Run `fit_tokyo_linked.py` once for each of `plain`, `huber`, `host_huber`, `pure_lstm`, `tt_ensemble`, `tabular`, `xgboost`, and `hgbr`. The first command is `python fit_tokyo_linked.py plain`.

~~~bash
python rebuild_linked_backtest.py
python run_ablation.py --threads 8
python score_ablation.py
python forward_2028.py
python sensitivity_grid.py
python draw_linked_backtest.py
~~~

`draw_linked_backtest.py` produces Tables 1-2 and Figures 4-5 and S2. `score_ablation.py` produces Tables 3-4. `forward_2028.py` produces the forecasts in Tables 5-6. Run `scenario_gdp.py` for uniform GDP perturbations and `scenario_gdp.py --scope host --outdir gen/scenarios_host` for perturbations limited to the host.

For the hosting analysis, run `econometrics_host_effect.py`, `econometrics_host_dynamic.py`, `econometrics_gdp_break.py`, `econometrics_var_decomp.py`, and `causal_host_effect.py` with Python. Table S4 is written to `gen/host_sample_sensitivity.csv` and Table S5 to `gen/causal/tau_table.csv`. Run `fig_medal2024.py` and `fig_hosting_dynamic.py` for Figures 1 and 7. `causal_host_effect.py` also draws Figures 6 and S1.

Neural models use CUDA when available and can run on CPU. The scripts fit the models from the processed data.
