from __future__ import annotations
# Core Mars year reliability simulator.
# This file is here for the report question about how dust storms change power
# reliability once solar, fission, and storage are all in the mix.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mars_power.common import (
    BASE_FISSION_CAPACITY_FACTOR,
    BASE_STORM_RATE,
    DEFAULT_BATTERY_ROUND_TRIP_EFFICIENCY,
    FISSION_UPTIME_PROBABILITY,
    HOURS_PER_SOL,
    N_SOLS,
    sample_storm_mask,
    solar_efficiency,
    total_demand_kw,
)


@dataclass(frozen=True)
class ReliabilityScenario:
    name: str
    people: int
    solar_kw: float
    fission_kw: float
    battery_kwh: float


def simulate_once(
    scenario: ReliabilityScenario,
    storm_durations: np.ndarray,
    habitat: pd.Series,
    dust_curve: pd.DataFrame,
    rng: np.random.RandomState,
    *,
    storm_rate: float = BASE_STORM_RATE,
    battery_rte: float = DEFAULT_BATTERY_ROUND_TRIP_EFFICIENCY,
) -> dict[str, float]:
    battery_soc = scenario.battery_kwh * 0.8
    shortfall_hours = 0.0
    max_shortfall_kw = 0.0
    in_storm = sample_storm_mask(rng, storm_durations, storm_rate=storm_rate)

    # We step through one Mars year sol by sol so storm exposure and battery drawdown show up clearly.
    for sol_index in range(N_SOLS):
        demand_kw = total_demand_kw(scenario.people, habitat, sol_index)
        dust = solar_efficiency(dust_curve, sol_index, in_storm[sol_index])
        solar_kw = scenario.solar_kw * 0.5 * dust
        fission_kw = scenario.fission_kw * (
            BASE_FISSION_CAPACITY_FACTOR if rng.rand() < FISSION_UPTIME_PROBABILITY else 0.0
        )
        net_kw = solar_kw + fission_kw - demand_kw

        if net_kw >= 0:
            battery_soc = min(
                scenario.battery_kwh,
                battery_soc + net_kw * HOURS_PER_SOL * battery_rte,
            )
            continue

        deficit_kwh = abs(net_kw) * HOURS_PER_SOL
        if battery_soc >= deficit_kwh:
            battery_soc -= deficit_kwh
            continue

        shortfall_hours += HOURS_PER_SOL
        max_shortfall_kw = max(max_shortfall_kw, abs(net_kw))
        battery_soc = 0.0

    return {
        "shortfall_hours": shortfall_hours,
        "max_shortfall_kw": max_shortfall_kw,
        "storm_sols": int(in_storm.sum()),
        "reliability": 1 - shortfall_hours / (N_SOLS * HOURS_PER_SOL),
    }


def simulate_many(
    scenario: ReliabilityScenario,
    n_sims: int,
    storm_durations: np.ndarray,
    habitat: pd.Series,
    dust_curve: pd.DataFrame,
    rng: np.random.RandomState,
    *,
    storm_rate: float = BASE_STORM_RATE,
    battery_rte: float = DEFAULT_BATTERY_ROUND_TRIP_EFFICIENCY,
) -> pd.DataFrame:
    rows = []
    for sim_index in range(n_sims):
        row = simulate_once(
            scenario,
            storm_durations,
            habitat,
            dust_curve,
            rng,
            storm_rate=storm_rate,
            battery_rte=battery_rte,
        )
        row["sim"] = sim_index
        row["scenario"] = scenario.name
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_reliability(results: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_reliability": float(results["reliability"].mean()),
        "p5_reliability": float(results["reliability"].quantile(0.05)),
        "pct_zero_shortfall": float((results["shortfall_hours"] == 0).mean()),
        "mean_storm_sols": float(results["storm_sols"].mean()),
    }
