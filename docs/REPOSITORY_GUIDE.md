# Repository Guide

This file is the plain map of what is in the folder, what each code file does, and how it connects back to the project questions.

## What this repo is trying to answer

The project asks three practical questions:

1. When does fusion start to look better than scaling fission for a Mars colony.
2. How much do dust storms hurt power reliability.
3. Which fusion concept looks the most promising on paper.

The code answers those questions with classification, cost modeling, Monte Carlo simulation, and forecast evaluation.

## Top level folders

`mars_power/`

Shared Python code.

`Data/`

Input data plus generated outputs. If you rerun the analysis, most of the CSV and PNG files in here will update.

`docs/`

Repository notes and source records.

`report/`

LaTeX report files.

`presentation/`

LaTeX presentation files and theme files.

`run_analysis.py`

Main script for running the analyses.

## Code map

`mars_power/common.py`

Defines the shared constants, file paths, Jezero storm filtering, and the demand equations used across the repo.

`mars_power/reliability.py`

Contains the Mars year simulator used by the baseline reliability, scenario comparison, solar sizing, fission scaling, and sensitivity analyses.

`mars_power/costs.py`

Builds the resource classification table and the all in LCOE table.

`mars_power/forecasting.py`

Builds the synthetic demand series, the Linear, SARIMA, and Transformer proxy forecasts, the DM test, and the pinball loss metric.

`mars_power/analyses.py`

Holds the top level functions that create the project tables and figures.

## Analysis map

`inventory`

Checks the tracked input tables used by the final project and reports their size and shape.

Project role: setup and audit.

`classification`

Builds the McKelvey Box style comparison for solar, fission, and the three fusion concepts.

Project role: question 3, early feasibility framing.

`lcoe`

Computes the all in cost model with launch cost included.

Project role: question 1 and question 3, cost comparison.

`baseline-reliability`

Runs the 6 person baseline reliability simulation for solar plus fission.

Project role: question 2, baseline reliability.

`scenarios`

Compares six colony scenarios across size and supply mix.

Project role: question 2, reliability by scenario.

`forecast`

Compares forecast models using MAPE, DM tests, and pinball loss.

Project role: final modeling summary.

`classification-sensitivity`

Tests how the fusion classification changes if the TRL mapping is more pessimistic or more optimistic.

Project role: question 3 sensitivity check.

`lcoe-sensitivity`

Sweeps launch cost assumptions and checks whether rankings move around.

Project role: question 1 sensitivity check.

`solar-sizing`

Sweeps solar capacity for 100 person and 2,000 person colonies with and without fission.

Project role: question 2 and question 1, supply sizing.

`fission-scaling`

Sweeps the number of Kilopower units and tracks reliability plus CAPEX.

Project role: question 1, how hard fission scaling gets.

`reliability-sensitivity`

Varies storm rate and battery size, then builds a heat map and sensitivity summary.

Project role: question 2 robustness check.

## Data map

Tracked input files:

- `Data/MDAD.csv`
- `Data/dust_penalty_curve.csv`
- `Data/fusion_reactor_specs.csv`
- `Data/habitat_engineering_constants.csv`
- `Data/nuclear_capacity_data_annual.csv`
- `Data/storage_specs.csv`

The source record for these files is in `docs/DATA_SOURCES.md`.

Main generated tables:

- `Data/resource_classification.csv`
- `Data/all_in_cost_model.csv`
- `Data/monte_carlo_reliability.csv`
- `Data/scenario_reliability.csv`
- `Data/forecast_metrics.csv`
- `Data/mckelvey_sensitivity.csv`
- `Data/lcoe_sensitivity.csv`
- `Data/solar_sizing_sensitivity.csv`
- `Data/fission_scaling.csv`
- `Data/mc_storm_sensitivity.csv`
- `Data/mc_battery_sensitivity.csv`
- `Data/mc_sensitivity_grid.csv`

Main generated figures:

- `Data/mckelvey_box.png`
- `Data/lcoe_comparison.png`
- `Data/monte_carlo_reliability.png`
- `Data/scenario_reliability.png`
- `Data/final_synthesis.png`
- `Data/mckelvey_sensitivity.png`
- `Data/lcoe_sensitivity.png`
- `Data/solar_sizing_sensitivity.png`
- `Data/fission_scaling.png`
- `Data/monte_carlo_sensitivity.png`
