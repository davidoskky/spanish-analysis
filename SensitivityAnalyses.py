import pandas as pd
import numpy as np
from copy import deepcopy
from dta_handling import load_data
from eff_typology import assign_typology1
from New_Simulation import (
    generate_summary_table,
    typology_impact_summary,
)
from constants import (Residence_Ownership, Business_Value, Business_Ownership ,Primary_Residence, Num_Workers, PEOPLE_IN_HOUSEHOLD, Net_Wealth, Income, wealth_percentile, working_status, income_percentile)
import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Bracket = Tuple[float, float, float]

def simulate_wealth_tax_sensitivity(
    df: pd.DataFrame,
    brackets: Optional[List[Bracket]] = None,
    exemption: float = 700_000,
    income_cap_rate: float = 0.6,
    apply_cap: bool = True,
    elasticity: float = 0.0,
    valuation_adjustment_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    exemption_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """
    Simulates wealth tax burden under various parameters.
    """

    df = df.copy()

    if valuation_adjustment_fn:
        df = valuation_adjustment_fn(df)

    df["exempt_total"] = exemption_fn(df) if exemption_fn else 0

    non_taxable_assets = (
        df.get("p2_71", 0).fillna(0) +
        df.get("timpvehic", 0).fillna(0) +
        df.get("p2_84", 0).fillna(0)
    ) / df["np1"]

    base_wealth = df["netwealth_individual"] - non_taxable_assets - df["exempt_total"]
    df["taxable_wealth"] = np.maximum(base_wealth - exemption, 0)

    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - elasticity)

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

    def compute_tax(taxable_wealth: float) -> float:
        tax = 0.0
        for lower, upper, rate in brackets:
            if taxable_wealth > lower:
                taxed_amount = min(taxable_wealth, upper) - lower
                tax += taxed_amount * rate
            else:
                break
        return tax

    df["sim_tax"] = df["taxable_wealth"].apply(compute_tax)
    df["sim_tax_eroded"] = df["taxable_wealth_eroded"].apply(compute_tax)

    if apply_cap:
        df["final_tax"] = np.minimum(df["sim_tax_eroded"], df["income_individual"] * income_cap_rate)
    else:
        df["final_tax"] = df["sim_tax_eroded"]

    df["cap_relief"] = df["sim_tax_eroded"] - df["final_tax"]

    logger.info("Wealth tax simulation completed.")

    return df


# Sensitivity Parameters
def apply_valuation_manipulation(df, real_estate_discount=0.15, business_discount=0.20):
    df = df.copy()
    df[Primary_Residence] = df[Primary_Residence].fillna(0) * (1 - real_estate_discount)
    df[Business_Value] = df[Business_Value].fillna(0) * (1 - business_discount)
    return df

def get_split_method(method):
    def apply(df):
        df = df.copy()
        if method == "equal":
            df["netwealth_individual"] = df[Net_Wealth] / df[PEOPLE_IN_HOUSEHOLD]
        elif method == "adults_only":
           
            earners = pd.to_numeric(df[Num_Workers], errors="coerce")
            earners = earners.clip(lower=1).fillna(df["np1"])
            df["netwealth_individual"] = df[Net_Wealth] / earners
        elif method == "head_only":
            df["netwealth_individual"] = df[Net_Wealth]
        return df
    return apply

def apply_income_split(df):
    df = df.copy()
    if Num_Workers in df.columns:
        earners = pd.to_numeric(df[Num_Workers], errors="coerce")
        earners = earners.clip(lower=1).fillna(df[PEOPLE_IN_HOUSEHOLD])
        df["income_split_factor"] = earners
    else:
        df["income_split_factor"] = df[PEOPLE_IN_HOUSEHOLD]
    df["income_individual"] = df[Income] / df["income_split_factor"]

    return df

def apply_adjustments(df):
    df = df.copy()
    # Assumptions based on academic literature (e.g., Durán-Cabré et al. (2021)):
    REGIONAL_REDUCTION = (
    0.33  # 33% lost due to regional exemptions (e.g. Madrid)
)
    adjustment_factor = (1 - REGIONAL_REDUCTION)

    df["adjusted_taxable_wealth"] = df["taxable_wealth_eroded"] * adjustment_factor
    df["adjusted_sim_tax"] = df["sim_tax"] * adjustment_factor
    df["adjusted_final_tax"] = df["final_tax"] * adjustment_factor

    return df

def typology_impact_summary(df, weight_col="facine3"):
    "Table summarizing the impact of wealth tax by typology."
    df = df[df["final_tax"].notnull() & df["sim_tax"].notnull()]

    typology_df = (
        df.groupby("mismatch_type")
        .apply(
            lambda g: pd.Series(
                {
                    "Population Share": g[weight_col].sum() / df[weight_col].sum(),
                    "Avg Final Tax": np.average(g["adjusted_final_tax"], weights=g[weight_col]),
                    "Avg Sim Tax": np.average(g["adjusted_sim_tax"], weights=g[weight_col]),
                    "Cap Relief Share": (g["cap_relief"] > 1e-6).mean(),
                    "Total Revenue": (g["adjusted_final_tax"] * g[weight_col]).sum(),
                }
            )
        )
        .reset_index()
    )

    print("\n--- Typology Impact Table ---")
    print(typology_df.to_string(index=False))
    return typology_df

def main():
    df_base = load_data()
    df_base = assign_typology1(df_base)

    elasticities = [0.0, 0.1, 0.2]
    valuation_scenarios = [(0.10, 0.15), (0.15, 0.20)]
    income_cap_rates = [0.5, 0.7, 0.8]
    wealth_split_methods = ["equal", "adults_only", "head_only"]
    exemption_thresholds = [1000_000, 800_000, 500_000]

    results = {}

    for re_disc, biz_disc in valuation_scenarios:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
    df = apply_income_split(df)
    valuation_fn = lambda d: apply_valuation_manipulation(d, real_estate_discount=re_disc, business_discount=biz_disc)
    sim_df = simulate_wealth_tax_sensitivity(df, valuation_adjustment_fn=valuation_fn)
    sim_df = apply_adjustments(sim_df)
    results[f"valutation_scenario{valuation_scenarios}"] = sim_df

    for e in elasticities:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, elasticity=e)
        sim_df = apply_adjustments(sim_df)
        results[f"elasticity_{e}"] = sim_df

    for cap in income_cap_rates:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, income_cap_rate=cap)
        sim_df = apply_adjustments(sim_df)
        results[f"income_cap_{cap}"] = sim_df

    for method in wealth_split_methods:
        df = deepcopy(df_base)
        df = get_split_method(method)(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df)
        sim_df = apply_adjustments(sim_df)
        results[f"split_{method}"] = sim_df

    for threshold in exemption_thresholds:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, exemption=threshold)
        sim_df = apply_adjustments(sim_df)
        results[f"exemption_{threshold}"] = sim_df

# summary statistics for each scenario
    summary_tables = {}
    for key, df in results.items():
        summary_name = f"summary_{key}"
        summary_tables[summary_name] = generate_summary_table(df)
        print(f"\nScenario: {key}")
        typology_impact_summary(df)

if __name__ == "__main__":
    main()
