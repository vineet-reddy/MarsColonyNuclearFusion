from __future__ import annotations
# Cost and resource classification helpers for the fusion comparison.
# This file backs the parts of the report that ask when fusion starts to look
# believable and how its cost stacks up against solar and fission.

import pandas as pd

from mars_power.common import BASE_LAUNCH_COST_PER_KG, load_technology_screening_inputs


def classify_resource(certainty: float, commerciality: float) -> str:
    if certainty >= 0.5 and commerciality >= 0.5:
        return "Proved Reserve"
    if certainty >= 0.5 or commerciality >= 0.5:
        return "Prospective Resource"
    return "Contingent Resource"


def base_cost_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "Solar PV",
                "unit_power_kw": 50,
                "hardware_mass_kg": 2500,
                "deployment_mass_kg": 500,
                "capex_earth_M": 5,
                "annual_opex_M": 0.2,
                "lifetime_years": 15,
                "capacity_factor": 0.25,
                "notes": "Dust losses and long nights keep the capacity factor low.",
            },
            {
                "source": "Fission (Kilopower-class)",
                "unit_power_kw": 40,
                "hardware_mass_kg": 1500,
                "deployment_mass_kg": 300,
                "capex_earth_M": 30,
                "annual_opex_M": 0.5,
                "lifetime_years": 15,
                "capacity_factor": 0.92,
                "notes": "Modeled as a high availability baseload source.",
            },
            {
                "source": "CFS SPARC",
                "unit_power_kw": 50000,
                "hardware_mass_kg": 100000,
                "deployment_mass_kg": 20000,
                "capex_earth_M": 5000,
                "annual_opex_M": 50,
                "lifetime_years": 30,
                "capacity_factor": 0.80,
                "notes": "Large tokamak concept with high mass and strong scale.",
            },
            {
                "source": "Princeton FRC",
                "unit_power_kw": 2500,
                "hardware_mass_kg": 5000,
                "deployment_mass_kg": 1000,
                "capex_earth_M": 500,
                "annual_opex_M": 10,
                "lifetime_years": 20,
                "capacity_factor": 0.75,
                "notes": "Compact concept with midrange size and cost.",
            },
            {
                "source": "Avalanche Orbitron",
                "unit_power_kw": 500,
                "hardware_mass_kg": 500,
                "deployment_mass_kg": 100,
                "capex_earth_M": 100,
                "annual_opex_M": 2,
                "lifetime_years": 20,
                "capacity_factor": 0.70,
                "notes": "Smallest concept, but still low TRL.",
            },
        ]
    )


def build_resource_classification(
    costs: pd.DataFrame | None = None,
    screening_inputs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cost_table = apply_launch_cost() if costs is None else costs.copy()
    screening = (
        load_technology_screening_inputs() if screening_inputs is None else screening_inputs.copy()
    )
    table = cost_table.merge(screening, on="source", how="left", validate="one_to_one")
    table = table.rename(columns={"notes_x": "cost_notes", "notes_y": "screening_notes"})
    if table["technology_readiness_level"].isna().any():
        missing = table.loc[table["technology_readiness_level"].isna(), "source"].tolist()
        raise ValueError(f"Missing screening inputs for: {missing}")

    min_lcoe = float(table["lcoe_dollar_per_kWh"].min())
    table["trl_score"] = table["technology_readiness_level"] / 9.0
    table["lcoe_score"] = (min_lcoe / table["lcoe_dollar_per_kWh"]).clip(upper=1.0)
    table["certainty_of_existence"] = (
        0.60 * table["trl_score"]
        + 0.25 * table["capacity_factor"]
        + 0.15 * table["mars_operating_score"]
    )
    raw_commerciality = (
        0.55 * table["capacity_factor"]
        + 0.25 * table["mars_operating_score"]
        + 0.20 * table["lcoe_score"]
    )
    table["chance_of_commerciality"] = table["certainty_of_existence"] * raw_commerciality
    table["category"] = [
        classify_resource(certainty, commerciality)
        for certainty, commerciality in zip(
            table["certainty_of_existence"], table["chance_of_commerciality"], strict=True
        )
    ]
    table["net_power_kw"] = table["unit_power_kw"]
    table["score_method"] = (
        "D=0.60*(TRL/9)+0.25*capacity_factor+0.15*mars_operating_score; "
        "P=D*(0.55*capacity_factor+0.25*mars_operating_score+0.20*lcoe_score)"
    )

    output_columns = [
        "source",
        "category",
        "certainty_of_existence",
        "chance_of_commerciality",
        "net_power_kw",
        "technology_readiness_level",
        "trl_score",
        "capacity_factor",
        "mars_operating_score",
        "lcoe_dollar_per_kWh",
        "lcoe_score",
        "cost_notes",
        "screening_notes",
        "score_method",
    ]
    return table[output_columns]


def apply_launch_cost(
    costs: pd.DataFrame | None = None,
    launch_cost_per_kg: float = BASE_LAUNCH_COST_PER_KG,
) -> pd.DataFrame:
    table = base_cost_table() if costs is None else costs.copy()
    # We count launch mass here because it is part of the actual Mars deployment cost.
    table["launch_cost_M"] = (
        (table["hardware_mass_kg"] + table["deployment_mass_kg"]) * launch_cost_per_kg / 1e6
    )
    table["total_capex_M"] = table["capex_earth_M"] + table["launch_cost_M"]
    table["lifetime_energy_MWh"] = (
        table["unit_power_kw"] * table["capacity_factor"] * 8760 * table["lifetime_years"] / 1000
    )
    total_cost_M = table["total_capex_M"] + table["annual_opex_M"] * table["lifetime_years"]
    table["lcoe_dollar_per_MWh"] = total_cost_M * 1e6 / table["lifetime_energy_MWh"]
    table["lcoe_dollar_per_kWh"] = table["lcoe_dollar_per_MWh"] / 1000
    return table
