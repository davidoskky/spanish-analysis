import numpy as np
import pandas as pd
from ineqpy.inequality import gini

from constants import Net_Wealth
from statistic import top_share


def summarize_cap_and_tax_shares(df):
    top_10 = df["wealth_rank"] > 0.90
    top_1 = df["wealth_rank"] > 0.99
    top_01 = df["wealth_rank"] > 0.999

    total_relief = (df["cap_relief"] * df["facine3"]).sum()
    total_final_tax = (df["adjusted_final_tax"] * df["facine3"]).sum()

    top10_relief = (df.loc[top_10, "cap_relief"] * df.loc[top_10, "facine3"]).sum()
    top1_relief = (df.loc[top_1, "cap_relief"] * df.loc[top_1, "facine3"]).sum()
    top01_relief = (df.loc[top_01, "cap_relief"] * df.loc[top_01, "facine3"]).sum()

    top10_tax = (df.loc[top_10, "adjusted_final_tax"] * df.loc[top_10, "facine3"]).sum()
    top1_tax = (df.loc[top_1, "adjusted_final_tax"] * df.loc[top_1, "facine3"]).sum()
    top01_tax = (df.loc[top_01, "adjusted_final_tax"] * df.loc[top_01, "facine3"]).sum()

    print("Cap Relief Share:")
    print(f"  Top 10%: {top10_relief / total_relief:.2%}")
    print(f"  Top 1%:  {top1_relief / total_relief:.2%}")
    print(f"  Top 0.1%: {top01_relief / total_relief:.2%}\n")

    print("Final Tax Share:")
    print(f"  Top 10%: {top10_tax / total_final_tax:.2%}")
    print(f"  Top 1%:  {top1_tax / total_final_tax:.2%}")
    print(f"  Top 0.1%: {top01_tax / total_final_tax:.2%}")


def report_effective_tax_rates(df):
    def weighted_avg(series, weights):
        mask = series.notna()
        return np.average(series[mask], weights=weights[mask])

    top_10 = df["wealth_rank"] > 0.90
    top_1 = df["wealth_rank"] > 0.99

    eff_tax_top10 = weighted_avg(
        df.loc[top_10, "eff_tax_rate"], df.loc[top_10, "facine3"]
    )
    eff_tax_top1 = weighted_avg(df.loc[top_1, "eff_tax_rate"], df.loc[top_1, "facine3"])
    eff_tax_top10_nocap = weighted_avg(
        df.loc[top_10, "eff_tax_nocap"], df.loc[top_10, "facine3"]
    )
    eff_tax_top1_nocap = weighted_avg(
        df.loc[top_1, "eff_tax_nocap"], df.loc[top_1, "facine3"]
    )

    print("\n--- Effective Tax Rates ---")
    print(f"With Cap - Top 10%: {eff_tax_top10:.3%}")
    print(f"With Cap - Top 1%:  {eff_tax_top1:.3%}")
    print(f"Without Cap - Top 10%: {eff_tax_top10_nocap:.3%}")
    print(f"Without Cap - Top 1%:  {eff_tax_top1_nocap:.3%}")

    return df


def typology_impact_summary(df, weight_col="facine3"):
    typology_df = (
        df.groupby("mismatch_type")
        .apply(
            lambda g: pd.Series(
                {
                    "Population Share": g[weight_col].sum() / df[weight_col].sum(),
                    "Avg Final Tax": np.average(
                        g["adjusted_final_tax"], weights=g[weight_col]
                    ),
                    "Cap Relief Share": (g["cap_relief"] > 1e-6).mean(),
                    "Migration Rate": g["Migration_Exit"].mean(),
                    "Total Revenue": (g["adjusted_final_tax"] * g[weight_col]).sum(),
                }
            )
        )
        .reset_index()
    )
    print("\n--- Typology Impact Table ---")
    print(typology_df.to_string(index=False))
    return typology_df


def generate_summary_table(df, weight_col="facine3"):
    revenue_collected = (df["adjusted_final_tax"] * df[weight_col]).sum()
    revenue_without_cap = (df["adjusted_tax_afterBR"] * df[weight_col]).sum()
    cap_relief = revenue_without_cap - revenue_collected

    if "Migration_Exit" in df.columns:
        revenue_after_migration = (
            df.loc[~df["Migration_Exit"], "adjusted_final_tax"]
            * df.loc[~df["Migration_Exit"], weight_col]
        ).sum()
        migration_loss = revenue_collected - revenue_after_migration
    else:
        revenue_after_migration = np.nan
        migration_loss = np.nan

    erosion_base = (df["taxable_wealth"] - df["taxable_wealth_eroded"]).clip(lower=0)
    erosion_total_loss = (erosion_base * df[weight_col]).sum()

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Revenue Collected (with cap)",
                "Revenue Without Cap",
                "Cap Relief (Revenue Lost)",
                "Revenue After Migration",
                "Migration Loss",
                "Behavioral Erosion (Implicit Loss)",
            ],
            "EUR": [
                revenue_collected,
                revenue_without_cap,
                cap_relief,
                revenue_after_migration,
                migration_loss,
                erosion_total_loss,
            ],
        }
    )

    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False))
    return summary_df

import numpy as np

def gini(values, weights=None):
    """
    Compute Gini coefficient of a numpy array or pandas Series.

    Parameters:
    - values: array-like, income or wealth values
    - weights: array-like, same length as values

    Returns:
    - Gini coefficient as float between 0 and 1
    """
    values = np.asarray(values)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights)

    # Sort by values
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Compute cumulative values and weights
    cumw = np.cumsum(sorted_weights)
    cumxw = np.cumsum(sorted_values * sorted_weights)

    # Relative mean difference (Gini formula)
    gini_numerator = np.sum(sorted_weights * (cumxw - sorted_values * sorted_weights / 2))
    gini_denominator = cumxw[-1] * cumw[-1]
    
    return 1 - 2 * gini_numerator / gini_denominator


def compute_inequality_metrics(df):
    metrics = {
        "Gini Before Tax": gini(df[Net_Wealth], weights=df["facine3"]),
        "Gini After Tax (cap)": gini(df["wealth_after_cap"], weights=df["facine3"]),
        "Gini After Tax (no cap)": gini(df["wealth_after_no_cap"], weights=df["facine3"]),

        "Top 10% Share Before": top_share(df, Net_Wealth, "facine3", 0.10),
        "Top 10% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.10),
        "Top 10% Share After (no cap)": top_share(df, "wealth_after_no_cap", "facine3", 0.10),

        "Top 1% Share Before": top_share(df, Net_Wealth, "facine3", 0.01),
        "Top 1% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.01),
        "Top 1% Share After (no cap)": top_share(df, "wealth_after_no_cap", "facine3", 0.01),
    }

    print("\n--- Inequality Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4%}")
    return metrics


def payer_coverage(df):
    payers = (df["final_tax"] > 0).mean()
    print(f"Coverage: {payers:.2%} of population pays any WT.")


def loss_breakdown(df):
    gross = (df["sim_tax_original"] * df["facine3"]).sum()
    cap_loss = ((df["tax_afterBR"] - df["final_tax"]) * df["facine3"]).sum()
    regional_loss = ((df["final_tax"] - df["adjusted_final_tax"]) * df["facine3"]).sum()
    behav_loss = ((df["sim_tax_original"] - df["tax_afterBR"]) * df["facine3"]).sum()

    print(f"Cap loss:      {cap_loss / gross:.1%} of gross")
    print(f"Regional loss: {regional_loss / gross:.1%}")
    print(f"Behavioural:   {behav_loss / gross:.1%}")


def generate_summary_table2(df: pd.DataFrame, weight_col="facine3") -> None:
    """
    Generate and print summary of tax revenue at different simulation stages.
    Assumes sim_tax_original is set before erosion.
    """
    weight = df[weight_col]

    # Revenue at each stage
    revenue_pre_erosion = (df["sim_tax_original"] * weight).sum()
    revenue_post_erosion = (df["tax_afterBR"] * weight).sum()
    revenue_after_cap = (df["final_tax"] * weight).sum()
    revenue_after_regional = (df["adjusted_final_tax"] * weight).sum()

    # Losses
    erosion_loss = revenue_pre_erosion - revenue_post_erosion
    cap_relief_loss = revenue_post_erosion - revenue_after_cap
    regional_loss = revenue_after_cap - revenue_after_regional

    print("\n--- Revenue Summary ---")
    print(f"Revenue Before Erosion:            €{revenue_pre_erosion:,.0f}")
    print(f"Revenue After Behavioral Erosion:  €{revenue_post_erosion:,.0f}")
    print(f"Revenue After Income Cap:          €{revenue_after_cap:,.0f}")
    print(f"Revenue After Regional Adjustments:€{revenue_after_regional:,.0f}")
    print(f"\nLoss Due to Behavioral Erosion:    €{erosion_loss:,.0f}")
    print(f"Loss Due to Income Cap Relief:     €{cap_relief_loss:,.0f}")
    print(f"Loss Due to Regional Adjustments:  €{regional_loss:,.0f}")
