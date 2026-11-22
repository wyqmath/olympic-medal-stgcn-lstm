# Code and Data

Code and processed data for forecasting Summer Olympic medal totals and analyzing hosting effects. The repository includes historical backtests, model comparisons, and component ablations.

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

For GDP sensitivity and hosting analysis, run:

~~~bash
python scenario_gdp.py
python scenario_gdp.py --scope host --outdir gen/scenarios_host
python econometrics_host_effect.py
python econometrics_host_dynamic.py
python econometrics_gdp_break.py
python econometrics_var_decomp.py
python causal_host_effect.py
python fig_medal2024.py
python fig_hosting_dynamic.py
~~~

Neural models use CUDA when available and can run on CPU. The scripts fit the models from the processed data.
