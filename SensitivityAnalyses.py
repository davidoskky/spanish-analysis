import pandas as pd
import numpy as np
from copy import deepcopy
from dta_handling import load_data
from eff_typology import assign_typology1
from New_Simulation import (
    apply_individual_split,
    simulate_wealth_tax,
    generate_summary_table,
    typology_impact_summary,
    compute_effective_tax_rates,
    summarize_cap_and_tax_shares,
)

# --- Sensitivity Simulation Function ---
def simulate_wealth_tax_sensitivity(
    df,
    brackets=None,
    exemption=700_000,
    income_cap_rate=0.6,
    apply_cap=True,
    elasticity=0.0,
    valuation_adjustment_fn=None,
    exemption_fn=None,
):
    df = df.copy()

    if valuation_adjustment_fn is not None:
        df = valuation_adjustment_fn(df)

    if exemption_fn is not None:
        df["exempt_total"] = exemption_fn(df)
    else:
        df["exempt_total"] = 0

    non_taxable_assets = (
        df["p2_71"].fillna(0) + df["timpvehic"].fillna(0) + df["p2_84"].fillna(0)
    ) / df["np1"]
    base = df["netwealth_individual"] - non_taxable_assets - df["exempt_total"]
    df["taxable_wealth"] = np.maximum(base - exemption, 0)

    if brackets is None:
        brackets = [
            (0, 167129.45, 0.002),
            (167129.46, 334246.88, 0.003),
            (334246.89, 668499.75, 0.005),
            (668499.76, 1336999.51, 0.009),
            (1336999.52, 2673999.01, 0.013),
            (2673999.02, 5347998.03, 0.017),
            (5347998.04, 10695996.06, 0.021),
            (10695996.07, float("inf"), 0.035),
        ]

    def compute_tax(taxable):
        tax = 0
        for lower, upper, rate in brackets:
            if taxable > lower:
                taxed_amount = min(taxable, upper) - lower
                tax += taxed_amount * rate
            else:
                break
        return tax

    df["sim_tax"] = df["taxable_wealth"].apply(compute_tax)

    if apply_cap:
        income = df["income_individual"]
        cap = income * income_cap_rate
        df["adjusted_sim_tax"] = df["sim_tax"]
        df["adjusted_final_tax"] = np.minimum(df["sim_tax"], cap)
    else:
        df["adjusted_final_tax"] = df["sim_tax"]

    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - elasticity)

    return df

# --- Define optional adjustment functions ---
def apply_valuation_manipulation(df, real_estate_discount=0.15, business_discount=0.20):
    df = df.copy()
    df["p2_70"] = df["p2_70"].fillna(0) * (1 - real_estate_discount)
    df["valhog"] = df["valhog"].fillna(0) * (1 - business_discount)
    return df

def get_split_method(method):
    def apply(df):
        df = df.copy()
        if method == "equal":
            df["netwealth_individual"] = df["riquezanet"] / df["np1"]
        elif method == "adults_only":
            # Convert to numeric BEFORE clipping
            earners = pd.to_numeric(df["nnumadtrab"], errors="coerce")
            earners = earners.clip(lower=1).fillna(df["np1"])
            df["netwealth_individual"] = df["riquezanet"] / earners
        elif method == "head_only":
            df["netwealth_individual"] = df["riquezanet"]
        return df
    return apply


def apply_income_split(df):
    df = df.copy()
    if "nnumadtrab" in df.columns:
        earners = pd.to_numeric(df["nnumadtrab"], errors="coerce")
        earners = earners.clip(lower=1).fillna(df["np1"])
        df["income_split_factor"] = earners
    else:
        df["income_split_factor"] = df["np1"]
    df["income_individual"] = df["renthog21_eur22"] / df["income_split_factor"]
    return df

# --- Main Function ---
def main():
    df_base = load_data()
    df_base = assign_typology1(df_base)

    elasticities = [0.0, 0.1, 0.2]
    valuation_scenarios = [(0.10, 0.15), (0.15, 0.20)]
    income_cap_rates = [0.5, 0.6, 0.7]
    wealth_split_methods = ["equal", "adults_only", "head_only"]
    exemption_thresholds = [700_000, 500_000, 300_000]

    results = {}

    for e in elasticities:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, elasticity=e)
        results[f"elasticity_{e}"] = sim_df

    for cap in income_cap_rates:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, income_cap_rate=cap)
        results[f"income_cap_{cap}"] = sim_df

    for method in wealth_split_methods:
        df = deepcopy(df_base)
        df = get_split_method(method)(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df)
        results[f"split_{method}"] = sim_df

    for threshold in exemption_thresholds:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, exemption=threshold)
        results[f"exemption_{threshold}"] = sim_df

    summary_tables = {}
    for key, df in results.items():
        summary_tables[key] = generate_summary_table(df)

if __name__ == "__main__":
    main()
