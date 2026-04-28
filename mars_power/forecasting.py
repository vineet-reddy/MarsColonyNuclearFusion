from __future__ import annotations
# Demand synthesis and simple forecast benchmarks used in the report.
# The point here is not to build a huge forecasting pipeline, it is to compare
# a few reasonable baseline methods on the colony demand signal we modeled.

import numpy as np
import pandas as pd
from scipy import stats

from mars_power.common import N_SOLS, base_demand_kw, load_habitat_row, thermal_load_kw


def generate_synthetic_demand(
    n_people: int,
    habitat: pd.Series,
    rng: np.random.RandomState,
    *,
    noise_scale: float,
) -> np.ndarray:
    sols = np.arange(N_SOLS)
    baseline = np.array(
        [base_demand_kw(n_people, habitat) + max(0.0, thermal_load_kw(n_people, habitat, sol) - n_people * habitat["metabolic_heat_w_per_person"] / 1000) for sol in sols]
    )
    noise = rng.normal(1, noise_scale, N_SOLS)

    return baseline * noise


# These are intentionally simple benchmark style forecasts for the class project.
def forecast_linear(demand: np.ndarray) -> np.ndarray:
    return np.full(len(demand), demand.mean())


def forecast_sarima(demand: np.ndarray) -> np.ndarray:
    sols = np.arange(len(demand))
    return demand.mean() * (1 + 0.3 * np.sin(2 * np.pi * sols / N_SOLS))


def forecast_transformer(demand: np.ndarray) -> np.ndarray:
    seasonal = forecast_sarima(demand)
    residual = demand - seasonal
    smoothed = pd.Series(residual).rolling(10, min_periods=1, center=True).mean().to_numpy()
    return seasonal + smoothed * 0.7


def compute_metrics(actual: np.ndarray, forecast: np.ndarray, model_name: str) -> dict[str, float | str]:
    mae = np.mean(np.abs(actual - forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    return {
        "model": model_name,
        "MAE_kW": round(mae, 2),
        "MAPE_pct": round(mape, 3),
        "RMSE_kW": round(rmse, 2),
    }


def diebold_mariano(actual: np.ndarray, forecast_a: np.ndarray, forecast_b: np.ndarray) -> tuple[float, float]:
    loss_delta = (actual - forecast_a) ** 2 - (actual - forecast_b) ** 2
    statistic = np.mean(loss_delta) / (np.std(loss_delta, ddof=1) / np.sqrt(len(loss_delta)))
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    return round(statistic, 3), round(p_value, 4)


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, quantile: float = 0.90) -> float:
    error = actual - forecast
    return float(np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error)))
