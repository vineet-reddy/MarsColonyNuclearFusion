from __future__ import annotations
# Top level analysis functions for the report tables and figures.
# Each function lines up pretty closely with one of the project questions,
# so run_analysis.py can stay short and readable.

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mars_power.common import (
    BASE_LAUNCH_COST_PER_KG,
    BASE_STORM_RATE,
    DATA_DIR,
    KILOPOWER_UNIT_CAPEX_M,
    KILOPOWER_UNIT_KW,
    KILOPOWER_UNIT_MASS_KG,
    get_liion_round_trip_efficiency,
    get_storm_durations,
    load_dust_curve,
    load_habitat_row,
    load_storage_specs,
    load_technology_screening_inputs,
    save_figure,
)
from mars_power.costs import apply_launch_cost, base_cost_table, build_resource_classification
from mars_power.forecasting import (
    compute_metrics,
    diebold_mariano,
    forecast_linear,
    forecast_sarima,
    forecast_transformer,
    generate_synthetic_demand,
    pinball_loss,
)
from mars_power.reliability import ReliabilityScenario, simulate_many, summarize_reliability


# Simple inventory step so it is obvious which tracked inputs the project is using.
DATA_FILES = [
    ("Dust Storms", "MDAD.csv", True),
    ("EIA Nuclear Annual", "nuclear_capacity_data_annual.csv", True),
    ("Fusion Reactor Specs", "fusion_reactor_specs.csv", True),
    ("Technology Screening Inputs", "technology_screening_inputs.csv", True),
    ("Habitat Constants", "habitat_engineering_constants.csv", True),
    ("Storage Specs", "storage_specs.csv", True),
    ("Dust Penalty Curve", "dust_penalty_curve.csv", True),
]


def _print_header(title: str, subtitle: str | None = None) -> None:
    print("=" * 60)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 60)


def run_data_inventory() -> None:
    _print_header("Data Inventory")

    present = []
    missing = []
    for label, filename, required_for_core in DATA_FILES:
        path = DATA_DIR / filename
        if path.exists():
            frame = pd.read_csv(path, nrows=5)
            size_mb = path.stat().st_size / 1e6
            print(f"OK       {label:20s} {filename:32s} {size_mb:8.1f} MB")
            print(f"         preview shape: {frame.shape[0]} rows x {frame.shape[1]} cols")
            present.append(label)
        else:
            note = "core input" if required_for_core else "optional input"
            print(f"MISSING  {label:20s} {filename:32s} {note}")
            missing.append(label)

    print("\nSummary")
    print(f"  Present: {len(present)}")
    print(f"  Missing: {len(missing)}")


def run_mckelvey_classification() -> pd.DataFrame:
    _print_header("McKelvey Box Classification")

    resources = build_resource_classification()
    print(
        resources[
            ["source", "category", "certainty_of_existence", "chance_of_commerciality", "net_power_kw"]
        ].to_string(index=False)
    )

    resources.to_csv(DATA_DIR / "resource_classification.csv", index=False)

    colors = {
        "Proved Reserve": "#2ecc71",
        "Prospective Resource": "#f39c12",
        "Contingent Resource": "#e74c3c",
    }
    legend_labels = {
        "Proved Reserve": "Proved analogue",
        "Prospective Resource": "Prospective analogue",
        "Contingent Resource": "Contingent analogue",
    }
    offsets = {
        "Solar PV": (10, -15),
        "Fission (Kilopower-class)": (-110, 10),
        "CFS SPARC": (14, -28),
        "Princeton FRC": (10, 14),
        "Avalanche Orbitron": (10, -15),
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in resources.iterrows():
        size = np.log10(row["net_power_kw"] + 1) * 150
        ax.scatter(
            row["certainty_of_existence"],
            row["chance_of_commerciality"],
            s=size,
            c=colors[row["category"]],
            alpha=0.82,
            edgecolors="black",
            linewidth=1.4,
        )
        ax.annotate(
            f"{row['source']}\n({row['net_power_kw']:.0f} kW)",
            (row["certainty_of_existence"], row["chance_of_commerciality"]),
            textcoords="offset points",
            xytext=offsets.get(row["source"], (8, 8)),
            fontsize=8,
        )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Certainty of Deployable System")
    ax.set_ylabel("Chance of Mars Commercial Operation")
    ax.set_title("Adapted McKelvey Chart: Mars Power Deployability")
    ax.text(0.75, 0.75, "Proved\nAnalogue", fontsize=11, alpha=0.25, ha="center", weight="bold")
    ax.text(0.25, 0.75, "Commercial\nPotential", fontsize=9, alpha=0.25, ha="center")
    ax.text(0.84, 0.12, "Known System,\nMars Barriers", fontsize=9, alpha=0.25, ha="center")
    ax.text(0.25, 0.25, "Low Confidence", fontsize=9, alpha=0.25, ha="center")
    patches = [mpatches.Patch(color=color, label=legend_labels[label]) for label, color in colors.items()]
    ax.legend(handles=patches, loc="upper left")

    out_png = save_figure(fig, "mckelvey_box.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'resource_classification.csv'}")
    print(f"Saved: {out_png}")
    return resources


def run_lcoe_model() -> pd.DataFrame:
    _print_header("All In LCOE Cost Model")

    costs = apply_launch_cost()
    print(
        costs[
            ["source", "total_capex_M", "launch_cost_M", "lifetime_energy_MWh", "lcoe_dollar_per_kWh"]
        ].to_string(index=False)
    )
    costs.to_csv(DATA_DIR / "all_in_cost_model.csv", index=False)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#f39c12", "#2ecc71", "#e74c3c", "#e74c3c", "#e74c3c"]
    bars = ax_left.barh(costs["source"], costs["lcoe_dollar_per_kWh"], color=colors, edgecolor="black")
    ax_left.set_xlabel("LCOE ($/kWh)")
    ax_left.set_title("LCOE with Launch Cost Included")
    for bar, value in zip(bars, costs["lcoe_dollar_per_kWh"]):
        ax_left.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"${value:.2f}", va="center")

    x = np.arange(len(costs))
    ax_right.bar(x, costs["capex_earth_M"], label="Earth CAPEX", color="#3498db")
    ax_right.bar(x, costs["launch_cost_M"], bottom=costs["capex_earth_M"], label="Launch", color="#e67e22")
    ax_right.bar(
        x,
        costs["annual_opex_M"] * costs["lifetime_years"],
        bottom=costs["capex_earth_M"] + costs["launch_cost_M"],
        label="Lifetime OPEX",
        color="#95a5a6",
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(costs["source"], rotation=30, ha="right")
    ax_right.set_ylabel("Cost ($M)")
    ax_right.set_title("Cost Breakdown")
    ax_right.legend()

    out_png = save_figure(fig, "lcoe_comparison.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'all_in_cost_model.csv'}")
    print(f"Saved: {out_png}")
    return costs


def run_monte_carlo_reliability(*, test_mode: bool) -> pd.DataFrame:
    n_sims = 50 if test_mode else 1000
    mode_label = "TEST (50 sims)" if test_mode else "FULL (1000 sims)"
    _print_header("Monte Carlo Reliability", f"Mode: {mode_label}")

    habitat = load_habitat_row()
    dust_curve = load_dust_curve()
    storm_durations = get_storm_durations()
    storage_specs = load_storage_specs()
    rng = np.random.RandomState(42)

    scenario = ReliabilityScenario(
        name="6p Solar+Fission",
        people=6,
        solar_kw=50,
        fission_kw=40,
        battery_kwh=500,
    )
    # This is the baseline case from the report before we branch into the bigger scenarios.
    results = simulate_many(
        scenario,
        n_sims,
        storm_durations,
        habitat,
        dust_curve,
        rng,
        battery_rte=get_liion_round_trip_efficiency(storage_specs),
    )
    results.to_csv(DATA_DIR / "monte_carlo_reliability.csv", index=False)

    summary = summarize_reliability(results)
    print(f"  Mean reliability: {summary['mean_reliability']:.4f}")
    print(f"  5th percentile:   {summary['p5_reliability']:.4f}")
    print(f"  Zero shortfall:   {(results['shortfall_hours'] == 0).sum()}/{n_sims}")
    print(f"  Mean storm sols:  {summary['mean_storm_sols']:.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(results["reliability"], bins=50, color="#2ecc71", edgecolor="black", alpha=0.8)
    axes[0, 0].axvline(0.90, color="red", linestyle="--", label="90% threshold")
    axes[0, 0].axvline(results["reliability"].mean(), color="blue", linestyle="--", label="Mean")
    axes[0, 0].set_xlabel("Reliability")
    axes[0, 0].set_title("Reliability Distribution")
    axes[0, 0].legend()

    axes[0, 1].scatter(results["storm_sols"], results["shortfall_hours"], alpha=0.3, s=10, c="#e74c3c")
    axes[0, 1].set_xlabel("Storm Sols per Mars Year")
    axes[0, 1].set_ylabel("Shortfall Hours")
    axes[0, 1].set_title("Shortfall vs Storm Exposure")

    sorted_rel = np.sort(results["reliability"])
    cdf = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
    axes[1, 0].plot(sorted_rel, cdf, color="#3498db", linewidth=2)
    axes[1, 0].axvline(0.90, color="red", linestyle="--", label="90% threshold")
    axes[1, 0].set_xlabel("Reliability")
    axes[1, 0].set_ylabel("Cumulative Probability")
    axes[1, 0].set_title("Reliability CDF")
    axes[1, 0].legend()

    nonzero = results.loc[results["max_shortfall_kw"] > 0, "max_shortfall_kw"]
    if not nonzero.empty:
        axes[1, 1].hist(nonzero, bins=30, color="#f39c12", edgecolor="black", alpha=0.8)
    axes[1, 1].set_xlabel("Max Shortfall (kW)")
    axes[1, 1].set_title("Peak Shortfall Distribution")

    fig.suptitle(f"Monte Carlo Reliability: {n_sims} Mars Year Simulations")
    out_png = save_figure(fig, "monte_carlo_reliability.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'monte_carlo_reliability.csv'}")
    print(f"Saved: {out_png}")
    return results


def run_scenario_comparison(*, test_mode: bool) -> pd.DataFrame:
    n_sims = 50 if test_mode else 500
    mode_label = "TEST (50 sims)" if test_mode else "FULL (500 sims)"
    _print_header("Scenario Comparison", f"Mode: {mode_label}")

    habitat = load_habitat_row()
    dust_curve = load_dust_curve()
    storm_durations = get_storm_durations()
    storage_specs = load_storage_specs()
    rng = np.random.RandomState(42)
    battery_rte = get_liion_round_trip_efficiency(storage_specs)

    scenarios = [
        ReliabilityScenario("6p Solar-only", 6, 80, 0, 500),
        ReliabilityScenario("6p Solar+Fission", 6, 50, 40, 500),
        ReliabilityScenario("100p Solar-only", 100, 1800, 0, 10000),
        ReliabilityScenario("100p Solar+Fission", 100, 600, 500, 5000),
        ReliabilityScenario("2000p Solar-only", 2000, 35000, 0, 200000),
        ReliabilityScenario("2000p Solar+Fission", 2000, 10000, 8000, 100000),
    ]

    frames = []
    for scenario in scenarios:
        print(f"  Running: {scenario.name}")
        frames.append(
            simulate_many(
                scenario,
                n_sims,
                storm_durations,
                habitat,
                dust_curve,
                rng,
                battery_rte=battery_rte,
            )[["scenario", "shortfall_hours", "reliability"]]
        )

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(DATA_DIR / "scenario_reliability.csv", index=False)

    summary = (
        results.groupby("scenario")
        .agg(
            mean_reliability=("reliability", "mean"),
            p5_reliability=("reliability", lambda values: values.quantile(0.05)),
            pct_zero_sf=("shortfall_hours", lambda values: (values == 0).mean()),
        )
        .round(4)
        .sort_values("mean_reliability")
    )
    print(summary.to_string())

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#e74c3c" if "Solar-only" in name else "#2ecc71" for name in summary.index]

    ax_left.barh(summary.index, summary["mean_reliability"], color=colors, edgecolor="black")
    ax_left.axvline(0.90, color="red", linestyle="--", alpha=0.7, label="90% threshold")
    ax_left.set_xlabel("Mean Reliability")
    ax_left.set_title("Reliability by Scenario")
    ax_left.legend()

    box_data = [results.loc[results["scenario"] == name, "reliability"].to_numpy() for name in summary.index]
    boxplot = ax_right.boxplot(box_data, patch_artist=True, labels=summary.index)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_right.axhline(0.90, color="red", linestyle="--")
    ax_right.set_ylabel("Reliability")
    ax_right.set_title("Reliability Spread")
    ax_right.tick_params(axis="x", rotation=35)

    out_png = save_figure(fig, "scenario_reliability.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'scenario_reliability.csv'}")
    print(f"Saved: {out_png}")
    return results


def run_final_synthesis() -> pd.DataFrame:
    _print_header("Final Synthesis")

    habitat = load_habitat_row()
    rng = np.random.RandomState(42)

    metric_rows = []
    for people in (6, 100, 2000):
        actual = generate_synthetic_demand(people, habitat, rng, noise_scale=0.08)
        linear = forecast_linear(actual)
        sarima = forecast_sarima(actual)
        transformer = forecast_transformer(actual)

        for name, forecast in (("Linear", linear), ("SARIMA", sarima), ("Transformer", transformer)):
            row = compute_metrics(actual, forecast, name)
            row["colony_size"] = people
            row["pinball_loss_90"] = round(pinball_loss(actual, forecast), 4)
            metric_rows.append(row)

        dm_ls, p_ls = diebold_mariano(actual, linear, sarima)
        dm_st, p_st = diebold_mariano(actual, sarima, transformer)
        dm_lt, p_lt = diebold_mariano(actual, linear, transformer)
        print(f"\n{people} people")
        print(f"  Linear vs SARIMA:      DM={dm_ls:.3f}, p={p_ls:.4f}")
        print(f"  SARIMA vs Transformer: DM={dm_st:.3f}, p={p_st:.4f}")
        print(f"  Linear vs Transformer: DM={dm_lt:.3f}, p={p_lt:.4f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(DATA_DIR / "forecast_metrics.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'forecast_metrics.csv'}")
    print(metrics.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colony_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    for index, people in enumerate((6, 100, 2000)):
        subset = metrics.loc[metrics["colony_size"] == people]
        x = np.arange(len(subset))
        axes[0, 0].bar(x + index * 0.25, subset["MAPE_pct"], 0.25, color=colony_colors[index], edgecolor="black", label=f"{people}p")
    axes[0, 0].set_xticks([0.25, 1.25, 2.25])
    axes[0, 0].set_xticklabels(["Linear", "SARIMA", "Transformer"])
    axes[0, 0].set_ylabel("MAPE (%)")
    axes[0, 0].set_title("Forecast Accuracy")
    axes[0, 0].legend()

    actual_100 = generate_synthetic_demand(100, habitat, np.random.RandomState(142), noise_scale=0.08)
    axes[0, 1].plot(actual_100, alpha=0.5, label="Synthetic Actual", linewidth=0.8)
    axes[0, 1].plot(forecast_sarima(actual_100), label="SARIMA", linewidth=1.5)
    axes[0, 1].plot(forecast_transformer(actual_100), label="Transformer", linewidth=1.5)
    axes[0, 1].set_xlabel("Sol")
    axes[0, 1].set_ylabel("Demand (kW)")
    axes[0, 1].set_title("100 Person Colony Forecasts")
    axes[0, 1].legend()

    q90 = np.percentile(actual_100, 90)
    axes[1, 0].hist(actual_100, bins=40, alpha=0.7, color="#3498db", edgecolor="black")
    axes[1, 0].axvline(q90, color="red", linestyle="--", linewidth=2, label=f"P90 = {q90:.0f} kW")
    axes[1, 0].axvline(np.mean(actual_100), color="blue", linestyle="--", label=f"Mean = {np.mean(actual_100):.0f} kW")
    axes[1, 0].set_xlabel("Demand (kW)")
    axes[1, 0].set_title("Demand Distribution")
    axes[1, 0].legend()

    for index, people in enumerate((6, 100, 2000)):
        subset = metrics.loc[metrics["colony_size"] == people]
        x = np.arange(len(subset))
        axes[1, 1].bar(
            x + index * 0.25,
            subset["pinball_loss_90"],
            0.25,
            color=colony_colors[index],
            edgecolor="black",
            label=f"{people}p",
        )
    axes[1, 1].set_xticks([0.25, 1.25, 2.25])
    axes[1, 1].set_xticklabels(["Linear", "SARIMA", "Transformer"])
    axes[1, 1].set_ylabel("Pinball Loss (90th percentile)")
    axes[1, 1].set_title("Safety Margin Loss")
    axes[1, 1].legend()

    fig.suptitle("Final Synthesis: Forecast Accuracy and Reliability")
    out_png = save_figure(fig, "final_synthesis.png")
    plt.close(fig)
    print(f"Saved: {out_png}")
    return metrics


def run_mckelvey_sensitivity() -> pd.DataFrame:
    _print_header("McKelvey Box Sensitivity")

    scenarios = {
        "Pessimistic": {"trl_delta": -1, "mars_delta": -0.05},
        "Base": {"trl_delta": 0, "mars_delta": 0.0},
        "Optimistic": {"trl_delta": 1, "mars_delta": 0.05},
    }
    colors = {
        "Proved Reserve": "#2ecc71",
        "Prospective Resource": "#f39c12",
        "Contingent Resource": "#e74c3c",
    }
    legend_labels = {
        "Proved Reserve": "Proved analogue",
        "Prospective Resource": "Prospective analogue",
        "Contingent Resource": "Contingent analogue",
    }
    offsets = {
        "Solar PV": (8, -12),
        "Fission (Kilopower-class)": (-92, 6),
        "CFS SPARC": (12, -20),
        "Princeton FRC": (10, 10),
        "Avalanche Orbitron": (8, -14),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    rows = []
    for ax, (scenario_name, delta) in zip(axes, scenarios.items()):
        screening = load_technology_screening_inputs()
        fusion_mask = ~screening["source"].isin(["Solar PV", "Fission (Kilopower-class)"])
        screening.loc[fusion_mask, "technology_readiness_level"] = (
            screening.loc[fusion_mask, "technology_readiness_level"] + delta["trl_delta"]
        ).clip(lower=1, upper=9)
        screening.loc[fusion_mask, "mars_operating_score"] = (
            screening.loc[fusion_mask, "mars_operating_score"] + delta["mars_delta"]
        ).clip(lower=0.0, upper=1.0)
        resources = build_resource_classification(screening_inputs=screening)

        for _, row in resources.iterrows():
            size = np.log10(row["net_power_kw"] + 1) * 150
            ax.scatter(
                row["certainty_of_existence"],
                row["chance_of_commerciality"],
                s=size,
                c=colors[row["category"]],
                alpha=0.85,
                edgecolors="black",
                linewidth=1.4,
            )
            offset = offsets.get(row["source"], (6, 6))
            ax.annotate(row["source"], (row["certainty_of_existence"], row["chance_of_commerciality"]), textcoords="offset points", xytext=offset, fontsize=8)
            rows.append(
                {
                    "scenario": scenario_name,
                    "source": row["source"],
                    "certainty": row["certainty_of_existence"],
                    "Pc": row["chance_of_commerciality"],
                    "trl": row["technology_readiness_level"],
                    "mars_operating_score": row["mars_operating_score"],
                    "category": row["category"],
                }
            )

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Certainty of Deployable System")
        ax.set_title(f"{scenario_name} TRL Mapping")
        ax.text(0.75, 0.75, "Proved\nAnalogue", fontsize=8, alpha=0.2, ha="center", weight="bold")
        ax.text(0.25, 0.75, "Commercial\nPotential", fontsize=8, alpha=0.2, ha="center")
        ax.text(0.84, 0.12, "Known System,\nMars Barriers", fontsize=8, alpha=0.2, ha="center")

    axes[0].set_ylabel("Chance of Mars Commercial Operation")
    axes[2].legend(handles=[mpatches.Patch(color=color, label=legend_labels[label]) for label, color in colors.items()], loc="upper left", fontsize=8)
    fig.suptitle("Adapted McKelvey Sensitivity")
    out_png = save_figure(fig, "mckelvey_sensitivity.png")
    plt.close(fig)

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(DATA_DIR / "mckelvey_sensitivity.csv", index=False)
    print(f"Saved: {DATA_DIR / 'mckelvey_sensitivity.csv'}")
    print(f"Saved: {out_png}")

    print("\nClassification changes")
    for source in sensitivity["source"].unique():
        categories = sensitivity.loc[sensitivity["source"] == source, "category"].unique()
        print(f"  {source:28s}: {' -> '.join(categories)}")
    return sensitivity


def run_lcoe_sensitivity() -> pd.DataFrame:
    _print_header("LCOE Sensitivity To Launch Cost")

    launch_cases = {
        "$1,000/kg": 1000,
        "$1,500/kg": 1500,
        "$2,000/kg": 2000,
    }
    frames = []
    for name, value in launch_cases.items():
        case = apply_launch_cost(launch_cost_per_kg=value)
        case["scenario"] = name
        frames.append(case)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(DATA_DIR / "lcoe_sensitivity.csv", index=False)

    pivot = results.pivot(index="source", columns="scenario", values="lcoe_dollar_per_kWh")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(pivot))
    width = 0.27
    palette = ["#2ecc71", "#f39c12", "#e74c3c"]
    for index, (scenario_name, color) in enumerate(zip(launch_cases, palette)):
        ax_left.barh(x + (index - 1) * width, pivot[scenario_name], width, label=scenario_name, color=color, edgecolor="black", alpha=0.85)
    ax_left.set_yticks(x)
    ax_left.set_yticklabels(pivot.index)
    ax_left.set_xlabel("LCOE ($/kWh)")
    ax_left.set_title("LCOE by Launch Cost")
    ax_left.legend()

    change = pivot.sub(pivot["$1,500/kg"], axis=0)
    change.plot(kind="barh", ax=ax_right, color=["#2ecc71", "gray", "#e74c3c"], edgecolor="black", alpha=0.85)
    ax_right.set_xlabel("Change vs $1,500/kg ($/kWh)")
    ax_right.set_title("LCOE Change vs Base Case")
    ax_right.axvline(0, color="black", linewidth=0.8)

    out_png = save_figure(fig, "lcoe_sensitivity.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'lcoe_sensitivity.csv'}")
    print(f"Saved: {out_png}")

    ranks = pivot.rank(ascending=True).astype(int)
    print("\nRank stability")
    for source in ranks.index:
        row = ranks.loc[source].tolist()
        status = "stable" if len(set(row)) == 1 else f"changes {row}"
        print(f"  {source:28s}: {status}")
    return results


def _sweep_solar(
    people: int,
    solar_values: np.ndarray,
    fission_kw: float,
    battery_kwh: float,
    n_sims: int,
    storm_durations: np.ndarray,
    habitat: pd.Series,
    dust_curve: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows = []
    for solar_kw in solar_values:
        scenario = ReliabilityScenario("sweep", people, float(solar_kw), fission_kw, battery_kwh)
        results = simulate_many(scenario, n_sims, storm_durations, habitat, dust_curve, rng)
        rows.append(
            {
                "solar_kw": int(solar_kw),
                "mean_reliability": round(results["reliability"].mean(), 4),
                "p5_reliability": round(results["reliability"].quantile(0.05), 4),
            }
        )
    return pd.DataFrame(rows)


def run_solar_sizing_sensitivity(*, test_mode: bool) -> pd.DataFrame:
    n_sims = 20 if test_mode else 100
    n_points = 6 if test_mode else 12
    _print_header("Solar Array Sizing Sensitivity", f"Mode: {'TEST' if test_mode else 'FULL'} ({n_sims} sims, {n_points} sweep points)")

    habitat = load_habitat_row()
    dust_curve = load_dust_curve()
    storm_durations = get_storm_durations()
    rng = np.random.RandomState(42)

    solar_100 = np.linspace(200, 3000, n_points).astype(int)
    solar_2000 = np.linspace(5000, 50000, n_points).astype(int)

    df_100_sf = _sweep_solar(100, solar_100, 500, 5000, n_sims, storm_durations, habitat, dust_curve, rng)
    df_100_sol = _sweep_solar(100, solar_100, 0, 5000, n_sims, storm_durations, habitat, dust_curve, rng)
    df_2000_sf = _sweep_solar(2000, solar_2000, 8000, 100000, n_sims, storm_durations, habitat, dust_curve, rng)
    df_2000_sol = _sweep_solar(2000, solar_2000, 0, 100000, n_sims, storm_durations, habitat, dust_curve, rng)

    for frame, label in (
        (df_100_sf, "100p_solar_fission"),
        (df_100_sol, "100p_solar_only"),
        (df_2000_sf, "2000p_solar_fission"),
        (df_2000_sol, "2000p_solar_only"),
    ):
        frame["scenario"] = label

    results = pd.concat([df_100_sf, df_100_sol, df_2000_sf, df_2000_sol], ignore_index=True)
    results.to_csv(DATA_DIR / "solar_sizing_sensitivity.csv", index=False)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    chart_specs = [
        (ax_left, df_100_sf, df_100_sol, "100 Person Colony", 1, "kW"),
        (ax_right, df_2000_sf, df_2000_sol, "2,000 Person Colony", 1000, "MW"),
    ]
    for axis, solar_fission, solar_only, title, scale, unit in chart_specs:
        axis.plot(solar_fission["solar_kw"] / scale, solar_fission["mean_reliability"], "g-o", linewidth=2, label="Solar+Fission")
        axis.fill_between(solar_fission["solar_kw"] / scale, solar_fission["p5_reliability"], solar_fission["mean_reliability"], alpha=0.15, color="green")
        axis.plot(solar_only["solar_kw"] / scale, solar_only["mean_reliability"], "r-s", linewidth=2, label="Solar-only")
        axis.fill_between(solar_only["solar_kw"] / scale, solar_only["p5_reliability"], solar_only["mean_reliability"], alpha=0.15, color="red")
        axis.axhline(0.90, color="black", linestyle="--", linewidth=1.5, label="90% target")
        axis.set_xlabel(f"Solar Array Capacity ({unit})")
        axis.set_ylabel("Mean Reliability")
        axis.set_title(title)
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize=9)

    fig.suptitle("Solar Array Sizing Sensitivity")
    out_png = save_figure(fig, "solar_sizing_sensitivity.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'solar_sizing_sensitivity.csv'}")
    print(f"Saved: {out_png}")

    print("\nMinimum solar for 90% mean reliability")
    for label, frame in (
        ("100p Solar+Fission", df_100_sf),
        ("100p Solar-only", df_100_sol),
        ("2000p Solar+Fission", df_2000_sf),
        ("2000p Solar-only", df_2000_sol),
    ):
        passing = frame.loc[frame["mean_reliability"] >= 0.90]
        if passing.empty:
            print(f"  {label}: not reached in sweep")
        else:
            print(f"  {label}: {passing['solar_kw'].min():,.0f} kW")
    return results


def _sweep_fission(
    people: int,
    unit_counts: list[int],
    solar_kw: float,
    battery_kwh: float,
    n_sims: int,
    storm_durations: np.ndarray,
    habitat: pd.Series,
    dust_curve: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows = []
    for count in unit_counts:
        fission_kw = count * KILOPOWER_UNIT_KW
        scenario = ReliabilityScenario("sweep", people, solar_kw, fission_kw, battery_kwh)
        results = simulate_many(scenario, n_sims, storm_durations, habitat, dust_curve, rng)
        capex = count * KILOPOWER_UNIT_CAPEX_M + count * KILOPOWER_UNIT_MASS_KG * BASE_LAUNCH_COST_PER_KG / 1e6
        rows.append(
            {
                "n_units": count,
                "fission_kw": fission_kw,
                "capex_M": round(capex, 1),
                "mean_reliability": round(results["reliability"].mean(), 4),
                "p5_reliability": round(results["reliability"].quantile(0.05), 4),
            }
        )
    return pd.DataFrame(rows)


def run_fission_scaling(*, test_mode: bool) -> pd.DataFrame:
    n_sims = 20 if test_mode else 100
    _print_header("Fission Scaling Exploration", f"Mode: {'TEST (20 sims)' if test_mode else 'FULL (100 sims)'}")

    habitat = load_habitat_row()
    dust_curve = load_dust_curve()
    storm_durations = get_storm_durations()
    rng = np.random.RandomState(42)

    units_100 = [1, 2, 4] if test_mode else [1, 2, 4, 6, 8, 12, 16]
    units_2000 = [20, 40, 80] if test_mode else [20, 40, 60, 80, 100, 120, 160, 200]

    df_100 = _sweep_fission(100, units_100, 600, 5000, n_sims, storm_durations, habitat, dust_curve, rng)
    df_2000 = _sweep_fission(2000, units_2000, 10000, 100000, n_sims, storm_durations, habitat, dust_curve, rng)
    df_100["colony"] = "100p"
    df_2000["colony"] = "2000p"

    results = pd.concat([df_100, df_2000], ignore_index=True)
    results.to_csv(DATA_DIR / "fission_scaling.csv", index=False)
    print(results.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for column, (frame, title) in enumerate(((df_100, "100 Person Colony"), (df_2000, "2,000 Person Colony"))):
        top_axis = axes[0, column]
        top_axis.plot(frame["n_units"], frame["mean_reliability"], "b-o", linewidth=2, label="Mean reliability")
        top_axis.fill_between(frame["n_units"], frame["p5_reliability"], frame["mean_reliability"], alpha=0.2, color="blue")
        top_axis.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="90% target")
        top_axis.set_xlabel(f"Kilopower Units (x{KILOPOWER_UNIT_KW} kW each)")
        top_axis.set_ylabel("Reliability")
        top_axis.set_title(f"{title}: Reliability vs Fission Units")
        top_axis.set_ylim(0, 1.02)
        top_axis.legend()

        bottom_axis = axes[1, column]
        colors = ["#3498db"] * len(frame)
        passing = frame.loc[frame["mean_reliability"] >= 0.90]
        if not passing.empty:
            first_index = frame.index.get_loc(passing.index[0])
            colors[first_index] = "#e74c3c"
        bottom_axis.bar(frame["n_units"].astype(str), frame["capex_M"], color=colors, edgecolor="black", alpha=0.85)
        minimum_label = ""
        if not passing.empty:
            minimum_label = f" (min for 90% is {int(passing['n_units'].iloc[0])} units)"
        bottom_axis.set_title(f"{title}: Fission Cost{minimum_label}")
        bottom_axis.set_xlabel("Kilopower Units")
        bottom_axis.set_ylabel("Fission CAPEX ($M with launch)")

    fig.suptitle("Fission Scaling Exploration")
    out_png = save_figure(fig, "fission_scaling.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'fission_scaling.csv'}")
    print(f"Saved: {out_png}")

    print("\nMinimum units for 90% mean reliability")
    for label, frame in (("100p", df_100), ("2000p", df_2000)):
        passing = frame.loc[frame["mean_reliability"] >= 0.90]
        if passing.empty:
            print(f"  {label}: not reached in sweep")
        else:
            row = passing.iloc[0]
            print(f"  {label}: {int(row['n_units'])} units ({row['fission_kw']:.0f} kW, ${row['capex_M']:.0f}M)")
    return results


def run_monte_carlo_sensitivity(*, test_mode: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_sims = 20 if test_mode else 150
    n_sims_grid = 20 if test_mode else 80
    _print_header("Monte Carlo Sensitivity", f"Mode: {'TEST' if test_mode else 'FULL'} ({n_sims}/{n_sims_grid} sims)")

    habitat = load_habitat_row()
    dust_curve = load_dust_curve()
    storm_durations = get_storm_durations()
    rng = np.random.RandomState(42)
    storage_specs = load_storage_specs()
    battery_rte = get_liion_round_trip_efficiency(storage_specs)
    base_scenario = ReliabilityScenario("6p Solar+Fission", 6, 50, 40, 500)

    storm_multipliers = [0.50, 0.75, 1.00, 1.25, 1.50]
    battery_multipliers = [0.25, 0.50, 1.00, 2.00, 4.00]

    storm_rows = []
    for multiplier in storm_multipliers:
        scenario = base_scenario
        results = simulate_many(
            scenario,
            n_sims,
            storm_durations,
            habitat,
            dust_curve,
            rng,
            storm_rate=BASE_STORM_RATE * multiplier,
            battery_rte=battery_rte,
        )
        storm_rows.append(
            {
                "multiplier": multiplier,
                "storm_rate": round(BASE_STORM_RATE * multiplier, 5),
                "mean_reliability": round(results["reliability"].mean(), 4),
                "p5_reliability": round(results["reliability"].quantile(0.05), 4),
            }
        )

    battery_rows = []
    for multiplier in battery_multipliers:
        scenario = ReliabilityScenario(base_scenario.name, base_scenario.people, base_scenario.solar_kw, base_scenario.fission_kw, base_scenario.battery_kwh * multiplier)
        results = simulate_many(
            scenario,
            n_sims,
            storm_durations,
            habitat,
            dust_curve,
            rng,
            battery_rte=battery_rte,
        )
        battery_rows.append(
            {
                "multiplier": multiplier,
                "battery_kwh": int(base_scenario.battery_kwh * multiplier),
                "mean_reliability": round(results["reliability"].mean(), 4),
                "p5_reliability": round(results["reliability"].quantile(0.05), 4),
            }
        )

    grid_rows = []
    for storm_multiplier in storm_multipliers:
        for battery_multiplier in battery_multipliers:
            scenario = ReliabilityScenario(
                base_scenario.name,
                base_scenario.people,
                base_scenario.solar_kw,
                base_scenario.fission_kw,
                base_scenario.battery_kwh * battery_multiplier,
            )
            results = simulate_many(
                scenario,
                n_sims_grid,
                storm_durations,
                habitat,
                dust_curve,
                rng,
                storm_rate=BASE_STORM_RATE * storm_multiplier,
                battery_rte=battery_rte,
            )
            grid_rows.append(
                {
                    "storm_mult": storm_multiplier,
                    "battery_mult": battery_multiplier,
                    "mean_reliability": round(results["reliability"].mean(), 4),
                }
            )

    storm_df = pd.DataFrame(storm_rows)
    battery_df = pd.DataFrame(battery_rows)
    grid_df = pd.DataFrame(grid_rows)
    grid_pivot = grid_df.pivot(index="storm_mult", columns="battery_mult", values="mean_reliability")

    storm_df.to_csv(DATA_DIR / "mc_storm_sensitivity.csv", index=False)
    battery_df.to_csv(DATA_DIR / "mc_battery_sensitivity.csv", index=False)
    grid_df.to_csv(DATA_DIR / "mc_sensitivity_grid.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(storm_df["multiplier"], storm_df["mean_reliability"], "b-o", linewidth=2)
    axes[0, 0].fill_between(storm_df["multiplier"], storm_df["p5_reliability"], storm_df["mean_reliability"], alpha=0.2)
    axes[0, 0].axhline(0.90, color="red", linestyle="--", label="90% target")
    axes[0, 0].axvline(1.0, color="gray", linestyle=":", label="Base storm rate")
    axes[0, 0].set_xlabel("Storm Rate Multiplier")
    axes[0, 0].set_ylabel("Reliability")
    axes[0, 0].set_title("Reliability vs Storm Rate")
    axes[0, 0].legend()

    axes[0, 1].semilogx(battery_df["battery_kwh"], battery_df["mean_reliability"], "g-s", linewidth=2)
    axes[0, 1].fill_between(battery_df["battery_kwh"], battery_df["p5_reliability"], battery_df["mean_reliability"], alpha=0.2, color="green")
    axes[0, 1].axhline(0.90, color="red", linestyle="--", label="90% target")
    axes[0, 1].axvline(base_scenario.battery_kwh, color="gray", linestyle=":", label="Base battery")
    axes[0, 1].set_xlabel("Battery Capacity (kWh)")
    axes[0, 1].set_ylabel("Reliability")
    axes[0, 1].set_title("Reliability vs Battery Capacity")
    axes[0, 1].legend()

    heatmap = axes[1, 0].imshow(grid_pivot.values, cmap="RdYlGn", vmin=0.85, vmax=1.0, aspect="auto")
    axes[1, 0].set_xticks(range(len(battery_multipliers)))
    axes[1, 0].set_xticklabels([f"{mult}x\n({int(base_scenario.battery_kwh * mult)} kWh)" for mult in battery_multipliers], fontsize=7)
    axes[1, 0].set_yticks(range(len(storm_multipliers)))
    axes[1, 0].set_yticklabels([f"{mult}x" for mult in storm_multipliers], fontsize=8)
    axes[1, 0].set_xlabel("Battery Multiplier")
    axes[1, 0].set_ylabel("Storm Rate Multiplier")
    axes[1, 0].set_title("Reliability Heat Map")
    plt.colorbar(heatmap, ax=axes[1, 0], label="Mean Reliability")
    for row_index in range(len(storm_multipliers)):
        for column_index in range(len(battery_multipliers)):
            axes[1, 0].text(column_index, row_index, f"{grid_pivot.values[row_index, column_index]:.3f}", ha="center", va="center", fontsize=7)

    storm_range = storm_df["mean_reliability"].max() - storm_df["mean_reliability"].min()
    battery_range = battery_df["mean_reliability"].max() - battery_df["mean_reliability"].min()
    params = ["Storm Rate", "Battery Capacity"]
    ranges = [storm_range, battery_range]
    colors = ["#e74c3c" if value == max(ranges) else "#3498db" for value in ranges]
    bars = axes[1, 1].barh(params, ranges, color=colors, edgecolor="black", alpha=0.85)
    for bar, value in zip(bars, ranges):
        axes[1, 1].text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2, f"delta={value:.4f}", va="center")
    axes[1, 1].set_xlabel("Reliability Range")
    axes[1, 1].set_title("Sensitivity Summary")

    fig.suptitle("Monte Carlo Sensitivity: 6 Person Colony")
    out_png = save_figure(fig, "monte_carlo_sensitivity.png")
    plt.close(fig)
    print(f"Saved: {DATA_DIR / 'mc_storm_sensitivity.csv'}")
    print(f"Saved: {DATA_DIR / 'mc_battery_sensitivity.csv'}")
    print(f"Saved: {DATA_DIR / 'mc_sensitivity_grid.csv'}")
    print(f"Saved: {out_png}")
    return storm_df, battery_df, grid_df
