# Mars Colony Nuclear Fusion

This repo contains our INDENG 290 final project on Mars colony power systems. The analysis compares solar, fission, and several fusion concepts through feasibility, cost, and reliability under Mars dust storms.

## Repo map

`mars_power/`

Shared project code. This is where the modeling logic lives.

`Data/`

Tracked input CSV files plus generated CSV and PNG outputs.

`docs/`

Repository guide and data source notes.

`report/`

LaTeX report and compiled PDF.

`presentation/`

LaTeX presentation source, theme files, and compiled PDF.

`run_analysis.py`

Main script for running the project.

## Main files

`mars_power/common.py`

Paths, constants, data loading, storm sampling, and shared demand math.

`mars_power/reliability.py`

The Mars year Monte Carlo simulator used by the reliability analyses.

`mars_power/costs.py`

McKelvey classification helpers and the all in LCOE model.

`mars_power/forecasting.py`

Synthetic demand generation, forecast proxies, metrics, and the DM test.

`mars_power/analyses.py`

Top level functions that create each CSV and figure.

## Quick start

Create a local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Run the full project:

```bash
.venv/bin/python run_analysis.py
```

If you want a smaller test run or only want some analyses, open `run_analysis.py` and change the settings at the top of the file.

The source record for the input CSV files is in `docs/DATA_SOURCES.md`.

## Output files

The main outputs used in the report and presentation are:

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

## Notes

The detailed file map is in `docs/REPOSITORY_GUIDE.md`.
