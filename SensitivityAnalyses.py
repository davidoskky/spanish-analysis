import pandas as pd
import numpy as np
from copy import deepcopy
from dta_handling import load_data
from eff_typology import assign_typology
from New_Simulation import (
    apply_valuation_manipulation,
    typology_impact_summary,
    simulate_pit_liability,
    apply_income_cap,
)
from constants import (
    Residence_Ownership,
    Business_Value,
    Business_Ownership,
    Primary_Residence,
    Num_Workers,
    PEOPLE_IN_HOUSEHOLD,
    Net_Wealth,
    Income,
)
import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Bracket = Tuple[float, float, float]


def compute_legal_exemptions(df):
    """
    Estimates total legal exemptions that can be subtracted from taxable wealth.

    Two main categories are considered:
    - Primary residence exemption (if owned)
    - Business asset exemption (applied probabilistically)

    The idea is to replicate legal treatments where exemptions reduce the tax base
    before applying any tax rates.
    """
    # Primary residence exemption
    owns_home = df[Residence_Ownership] == "Ownership"
    primary_home_val = df[Primary_Residence].fillna(0)
    exempt_home_value = np.where(owns_home, np.minimum(primary_home_val, 300_000), 0)

    # Business exemption if household has declared business value
    business_exemption_rate = (
        0.30  # Probability of applying exemption (Duran-Cabré et al.)
    )
    has_business_value = df[Business_Ownership] == 1
    apply_business_exempt = (
        np.random.rand(len(df)) < business_exemption_rate
    ) & has_business_value
    business_exempt = np.where(apply_business_exempt, df[Business_Value].fillna(0), 0)

    return exempt_home_value + business_exempt


def simulate_wealth_tax_sensitivity(
    df: pd.DataFrame,
    brackets: Optional[List[Bracket]] = None,
    exemption: float = 700_000,
    income_cap_rate: float = 0.6,
    apply_cap: bool = True,
    elasticity: float = 0.0,
    exemption_fn: Optional[
        Callable[[pd.DataFrame], pd.Series]
    ] = compute_legal_exemptions,
) -> pd.DataFrame:
    df = df.copy()
    df["exempt_total"] = exemption_fn(df) if exemption_fn else 0

    non_taxable_assets = (
        df.get("p2_71", 0).fillna(0)
        + df.get("timpvehic", 0).fillna(0)
        + df.get("p2_84", 0).fillna(0)
    ) / df["np1"]

    base_wealth = df["netwealth_individual"] - non_taxable_assets - df["exempt_total"]
    df["base_wealth"] = base_wealth
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
        df["final_tax"] = np.minimum(
            df["sim_tax_eroded"], df["income_individual"] * income_cap_rate
        )
    else:
        df["final_tax"] = df["sim_tax_eroded"]

    df["cap_relief"] = df["sim_tax_eroded"] - df["final_tax"]

    logger.info("Wealth tax simulation completed.")

    return df


def apply_valuation_manipulation(df, real_estate_discount=0.15, business_discount=0.20):
    df = df.copy()
    df[Primary_Residence] = df[Primary_Residence].fillna(0) * (1 - real_estate_discount)
    df[Business_Value] = df[Business_Value].fillna(0) * (1 - business_discount)
    return df


def get_split_method(method):
    def apply(df):
        df = df.copy()
        earners = pd.to_numeric(df[Num_Workers], errors="coerce").fillna(0)
        hh_size = df[PEOPLE_IN_HOUSEHOLD]
        split_factor = earners.where(earners != 0, hh_size).clip(lower=1)

        if method == "equal":
            df["netwealth_individual"] = df[Net_Wealth] / split_factor
        elif method == "head_only":
            df["netwealth_individual"] = df[Net_Wealth]
        return df

    return apply


def apply_income_split(df):
    df = df.copy()
    earners = pd.to_numeric(df[Num_Workers], errors="coerce").fillna(0)
    hh_size = df[PEOPLE_IN_HOUSEHOLD]
    split_factor = earners.where(earners != 0, hh_size).clip(lower=1)

    df["income_split_factor"] = split_factor
    df["income_individual"] = df[Income] / split_factor
    return df


def revenue_summary(df, weight_col="facine3"):
    revenue_collected = (df["final_tax"] * df[weight_col]).sum()
    revenue_without_cap = (df["sim_tax"] * df[weight_col]).sum()
    cap_relief = revenue_without_cap - revenue_collected

    erosion_base = (df["base_wealth"] - df["taxable_wealth_eroded"]).clip(lower=0)
    erosion_total_loss = (erosion_base * df[weight_col]).sum()

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Revenue Collected (with cap)",
                "Revenue Without Cap",
                "Cap Relief (Revenue Lost)",
                "Behavioral Erosion (Implicit Loss)",
            ],
            "EUR": [
                revenue_collected,
                revenue_without_cap,
                cap_relief,
                erosion_total_loss,
            ],
        }
    )

    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False))
    return summary_df


def typology_impact_summary(df, weight_col="facine3"):
    df = df[df["final_tax"].notnull() & df["sim_tax"].notnull()]

    typology_df = (
        df.groupby("mismatch_type")
        .apply(
            lambda g: pd.Series(
                {
                    "Population Share": g[weight_col].sum() / df[weight_col].sum(),
                    "Avg Final Tax": np.average(g["final_tax"], weights=g[weight_col]),
                    "Avg Sim Tax": np.average(g["sim_tax"], weights=g[weight_col]),
                    "Cap Relief Share": (g["cap_relief"] > 1e-6).mean(),
                    "Total Revenue": (g["final_tax"] * g[weight_col]).sum(),
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
    df_base = assign_typology(df_base)

    elasticities = [0.0, 0.1, 0.2]
    valuation_scenarios = [(0.10, 0.15), (0.15, 0.20)]
    income_cap_rates = [0.5, 0.7, 0.8]
    wealth_split_methods = ["equal", "head_only"]
    exemption_thresholds = [1_000_000, 800_000, 500_000]

    summary_tables = {}

    for re_disc, biz_disc in valuation_scenarios:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        df = apply_valuation_manipulation(df, re_disc, biz_disc)
        sim_df = simulate_wealth_tax_sensitivity(df)
        sim_df = simulate_pit_liability(sim_df)
        sim_df = apply_income_cap(sim_df)
        summary_tables[f"summary_valuation_{re_disc}_{biz_disc}"] = revenue_summary(
            sim_df
        )
        typology_impact_summary(sim_df)
        print(f"\nScenario: valuation_{re_disc}_{biz_disc}")

    for e in elasticities:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, elasticity=e)
        sim_df = simulate_pit_liability(sim_df)
        sim_df = apply_income_cap(sim_df)
        summary_tables[f"summary_elasticity_{e}"] = revenue_summary(sim_df)
        typology_impact_summary(sim_df)
        print(f"\nScenario: elasticity_{e}")

    for cap in income_cap_rates:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df)
        sim_df = simulate_pit_liability(sim_df)
        sim_df = apply_income_cap(sim_df, income_cap_rate=cap)
        summary_tables[f"summary_cap_{cap}"] = revenue_summary(sim_df)
        typology_impact_summary(sim_df)
        print(f"\nScenario: cap_{cap}")

    for method in wealth_split_methods:
        df = deepcopy(df_base)
        df = get_split_method(method)(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df)
        sim_df = simulate_pit_liability(sim_df)
        sim_df = apply_income_cap(sim_df)
        summary_tables[f"summary_split_{method}"] = revenue_summary(sim_df)
        typology_impact_summary(sim_df)
        print(f"\nScenario: split_{method}")

    for threshold in exemption_thresholds:
        df = deepcopy(df_base)
        df = get_split_method("equal")(df)
        df = apply_income_split(df)
        sim_df = simulate_wealth_tax_sensitivity(df, exemption=threshold)
        sim_df = simulate_pit_liability(sim_df)
        sim_df = apply_income_cap(sim_df)
        summary_tables[f"summary_exemption_{threshold}"] = revenue_summary(sim_df)
        typology_impact_summary(sim_df)
        print(f"\nScenario: exemption_{threshold}")


if __name__ == "__main__":
    main()
