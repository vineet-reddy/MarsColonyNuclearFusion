from __future__ import annotations
# Shared constants and helper functions used across the project.
# We keep the base assumptions here so the cost, forecast,
# and reliability parts are all using the same setup.

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
DOCS_DIR = PROJECT_ROOT / "docs"

N_SOLS = 668
HOURS_PER_SOL = 24.6
BASE_STORM_RATE = 0.0057
BASE_FISSION_CAPACITY_FACTOR = 0.92
FISSION_UPTIME_PROBABILITY = 0.995
DEFAULT_BATTERY_ROUND_TRIP_EFFICIENCY = 0.90

KILOPOWER_UNIT_KW = 40
KILOPOWER_UNIT_MASS_KG = 1500
KILOPOWER_UNIT_CAPEX_M = 30
BASE_LAUNCH_COST_PER_KG = 1500

JEZERO_LAT_RANGE = (-12, 48)
JEZERO_LON_RANGE = (47, 107)


def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, **kwargs)


def load_habitat_row() -> pd.Series:
    return load_csv("habitat_engineering_constants.csv").iloc[0]


def load_dust_curve() -> pd.DataFrame:
    return load_csv("dust_penalty_curve.csv")


def load_storage_specs() -> pd.DataFrame:
    return load_csv("storage_specs.csv")


def load_fusion_specs() -> pd.DataFrame:
    return load_csv("fusion_reactor_specs.csv")


def load_technology_screening_inputs() -> pd.DataFrame:
    return load_csv("technology_screening_inputs.csv")


def load_nuclear_capacity() -> pd.DataFrame:
    return load_csv("nuclear_capacity_data_annual.csv")


def get_liion_round_trip_efficiency(storage_specs: pd.DataFrame | None = None) -> float:
    specs = storage_specs if storage_specs is not None else load_storage_specs()
    matches = specs.loc[specs["technology"] == "Li-ion Battery", "round_trip_efficiency"]
    if matches.empty:
        return DEFAULT_BATTERY_ROUND_TRIP_EFFICIENCY
    return float(matches.iloc[0])


# These demand helpers show up in both the forecast section and the reliability runs.
def base_demand_kw(n_people: int, habitat: pd.Series) -> float:
    water_recyclers = max(1, n_people // 6)
    return float(
        n_people * habitat["plug_load_kw_per_person"]
        + water_recyclers * habitat["water_recycler_kw"]
        + n_people * habitat["moxie_o2_kw"]
        + habitat["comms_kw"]
        + habitat["medical_lab_kw"]
        + n_people * habitat["lighting_kw_per_person"]
    )


def thermal_load_kw(n_people: int, habitat: pd.Series, sol_index: int) -> float:
    season = 1 + 0.3 * np.sin(2 * np.pi * sol_index / N_SOLS)
    shell_scale = n_people / 6
    return float(
        habitat["wall_u_value_w_m2k"]
        * habitat["habitat_surface_area_m2"]
        * shell_scale
        * 83
        * season
        / 1000
    )


def total_demand_kw(n_people: int, habitat: pd.Series, sol_index: int) -> float:
    base = base_demand_kw(n_people, habitat)
    metabolic = n_people * habitat["metabolic_heat_w_per_person"] / 1000
    return base + max(0.0, thermal_load_kw(n_people, habitat, sol_index) - metabolic)


def get_storm_durations() -> np.ndarray:
    mdad = load_csv("MDAD.csv")
    mdad.columns = mdad.columns.str.strip()

    for column in ("Centroid latitude", "Centroid longitude", "Sol"):
        mdad[column] = pd.to_numeric(mdad[column], errors="coerce")

    mdad["Sequence ID"] = mdad["Sequence ID"].fillna("").astype(str).str.strip()
    local = mdad[
        mdad["Centroid latitude"].between(*JEZERO_LAT_RANGE)
        & mdad["Centroid longitude"].between(*JEZERO_LON_RANGE)
        & mdad["Sequence ID"].ne("")
    ]

    durations = (
        local.groupby("Sequence ID")["Sol"]
        .agg(lambda values: values.max() - values.min() + 1)
        .to_numpy()
    )
    return durations.astype(int)


def sample_storm_mask(
    rng: np.random.RandomState,
    storm_durations: np.ndarray,
    storm_rate: float = BASE_STORM_RATE,
) -> np.ndarray:
    in_storm = np.zeros(N_SOLS, dtype=bool)
    sol = 0
    while sol < N_SOLS:
        if rng.rand() < storm_rate:
            duration = max(1, int(rng.choice(storm_durations)))
            in_storm[sol : min(sol + duration, N_SOLS)] = True
            sol += duration
            continue
        sol += 1
    return in_storm


def solar_efficiency(dust_curve: pd.DataFrame, sol_index: int, in_storm: bool) -> float:
    dust = float(dust_curve["solar_efficiency_factor"].iloc[min(sol_index, len(dust_curve) - 1)])
    if in_storm:
        dust *= 0.15
    return dust


def save_figure(fig, filename: str) -> Path:
    output_path = DATA_DIR / filename
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path
