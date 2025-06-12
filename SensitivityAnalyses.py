import pandas as pd
import numpy as np
from dta_handling import load_data
from eff_typology import assign_typology1
from New_Simulation import (
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
    """
    Customizable wealth tax simulation function for sensitivity analysis.

    Parameters:
    - df: Input DataFrame (already preprocessed).
    - brackets: List of (lower, upper, rate) tuples for tax schedule.
    - exemption: Individual exemption threshold (EUR).
    - income_cap_rate: Maximum tax/income ratio allowed (cap).
    - apply_cap: Whether to apply income cap adjustment.
    - elasticity: Optional behavioral elasticity for erosion (currently placeholder).
    - valuation_adjustment_fn: Optional function to adjust asset values (e.g., undervaluation).
    - exemption_fn: Optional function to compute legal exemptions.

    Returns:
    - df_result: DataFrame with computed tax outputs.
    """
    df = df.copy()

    # --- Step 1: Adjust valuation if needed ---
    if valuation_adjustment_fn is not None:
        df = valuation_adjustment_fn(df)

    # --- Step 2: Compute exemptions ---
    if exemption_fn is not None:
        df["exempt_total"] = exemption_fn(df)
    else:
        df["exempt_total"] = 0

    # --- Step 3: Compute base ---
    non_taxable_assets = (
        df["p2_71"].fillna(0) + df["timpvehic"].fillna(0) + df["p2_84"].fillna(0)
    ) / df["np1"]
    base = df["netwealth_individual"] - non_taxable_assets - df["exempt_total"]
    df["taxable_wealth"] = np.maximum(base - exemption, 0)

    # --- Step 4: Apply tax schedule ---
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

    # --- Step 5: Apply income cap if needed ---
    if apply_cap:
        income = df["income_individual"]
        cap = income * income_cap_rate
        df["adjusted_sim_tax"] = df["sim_tax"]
        df["adjusted_final_tax"] = np.minimum(df["sim_tax"], cap)
    else:
        df["adjusted_final_tax"] = df["sim_tax"]

    # --- Step 6: Placeholder for behavioral erosion (if any) ---
    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - elasticity)

    return df
