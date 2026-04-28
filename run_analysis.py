from mars_power.analyses import (
    run_data_inventory,
    run_final_synthesis,
    run_fission_scaling,
    run_lcoe_model,
    run_lcoe_sensitivity,
    run_mckelvey_classification,
    run_mckelvey_sensitivity,
    run_monte_carlo_reliability,
    run_monte_carlo_sensitivity,
    run_scenario_comparison,
    run_solar_sizing_sensitivity,
)

TEST_MODE = False

# Leave this empty to run everything.
ONLY_THESE: list[str] = []

# Add names here if you want to skip a step.
SKIP_THESE: set[str] = set()


def run_baseline_reliability() -> None:
    run_monte_carlo_reliability(test_mode=TEST_MODE)


def run_scenarios() -> None:
    run_scenario_comparison(test_mode=TEST_MODE)


def run_solar_sizing() -> None:
    run_solar_sizing_sensitivity(test_mode=TEST_MODE)


def run_fission_sizing() -> None:
    run_fission_scaling(test_mode=TEST_MODE)


def run_reliability_sensitivity() -> None:
    run_monte_carlo_sensitivity(test_mode=TEST_MODE)


ANALYSES = [
    ("inventory", run_data_inventory),
    ("classification", run_mckelvey_classification),
    ("lcoe", run_lcoe_model),
    ("baseline-reliability", run_baseline_reliability),
    ("scenarios", run_scenarios),
    ("forecast", run_final_synthesis),
    ("classification-sensitivity", run_mckelvey_sensitivity),
    ("lcoe-sensitivity", run_lcoe_sensitivity),
    ("solar-sizing", run_solar_sizing),
    ("fission-scaling", run_fission_sizing),
    ("reliability-sensitivity", run_reliability_sensitivity),
]


def main() -> None:
    print("=" * 60)
    print("Mars Colony Nuclear Fusion")
    print("=" * 60)

    selected = ANALYSES
    if ONLY_THESE:
        selected = [(name, func) for name, func in ANALYSES if name in ONLY_THESE]
    if SKIP_THESE:
        selected = [(name, func) for name, func in selected if name not in SKIP_THESE]

    for name, func in selected:
        print(f"\n[{name}]")
        func()

if __name__ == "__main__":
    main()
