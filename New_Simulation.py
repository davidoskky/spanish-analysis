import pandas as pd
import numpy as np

from constants import (
    PROGRESSIVE_TAX_BRACKETS,
    NON_TAXABLE_ASSET_COLS,
)
from dta_handling import load_data
from eff_typology import assign_typology

from ineqpy.inequality import gini

from preprocessing import individual_split, apply_valuation_manipulation
from wealth_tax import simulate_household_wealth_tax, simulate_pit_liability


def apply_wealth_tax_income_cap(
    df: pd.DataFrame, income_cap_rate: float = 0.60, min_wealth_tax_share: float = 0.20
):
    """
    Apply an income-based cap to the wealth tax (WT) as per Spanish tax rules.

    Ensures that the total tax burden (PIT + WT) does not exceed a set percentage
    (e.g. 60%) of an individual's income. If it does, the WT is reduced—but not
    below a minimum share (e.g. 20%) of the original wealth tax.

    Parameters:
    - income_cap_rate: ceiling threshold (default = 60%)
    - min_wt_share: minimum WT share to preserve (default = 20%)

    Returns:
    - df: DataFrame with capped WT and relief columns
    """
    df = df.copy()

    income_limit = df["income_individual"] * income_cap_rate
    wealth_tax = df["sim_tax"]
    income_tax = df["pit_liability"]

    total_tax = wealth_tax + income_tax
    over_cap = total_tax > income_limit

    max_allowed_relief = wealth_tax * (1 - min_wealth_tax_share)

    excess = total_tax - income_limit
    wt_relief = np.minimum(excess, max_allowed_relief)
    wt_relief = np.where(over_cap, wt_relief, 0.0)

    df["cap_relief"] = wt_relief
    df["final_tax"] = wealth_tax - wt_relief

    return df


def apply_behavioral_response(
    df,
    ref_tax_rate=0.004,
    max_erosion: float = 0.35,
    wealth_col: str = "netwealth_individual",
):
    """
    Apply behavioral erosion based on wealth-ranked elasticity to simulate real-world avoidance.
    Must be called after initial simulate_wealth_tax(), before income cap.

        Calculate the behavioural‐erosion factor θ for a vector of effective tax rates.

    θ_i = 1 − ((1 − τ_eff_i) / (1 − τ_ref))^ε
      • τ_eff_i  : individual effective wealth-tax rate
      • τ_ref    : reference rate (≈ population average)
      • ε        : elasticity of taxable wealth wrt. net-of-tax rate
      • θ is capped at `max_erosion` and floored at 0

     Sources:
    - Jakobsen et al. (2020), QJE
    - Seim (2017), AER
    - Duran-Cabré et al. (2023), WP
    """

    df = df.copy()

    schedule = [
        (0.9999, 1.10),
        (0.999, 0.80),
        (0.990, 0.40),
        (0.900, 0.20),
    ]
    eff = df["sim_tax"] / (df[wealth_col] + 1e-6)

    thresholds, values = zip(*schedule)
    conditions = [df["wealth_rank"] > t for t in thresholds]
    elasticity = np.select(conditions, values, default=0.10)

    # 3. Behavioural erosion factor θ
    theta = 1 - ((1 - eff) / (1 - ref_tax_rate)) ** elasticity
    theta = np.clip(theta, 0, max_erosion)
    theta[(eff <= 0) | (eff >= 1) | np.isnan(eff)] = 0.0

    df["behavioral_erosion"] = theta
    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - theta)

    return df


def simulate_migration_attrition(
    df: pd.DataFrame,
    wealth_threshold: float = 0.995,
    base_migration_prob: float = 0.04,
    elasticity: float = 1.76,
) -> pd.DataFrame:
    """
    Simulates tax-motivated migration or wealth erosion among top wealth holders,
    based on behavioral responses modeled in Jakobsen et al. (2020).

    This function probabilistically identifies individuals likely to "exit"
    the tax base (e.g., through migration, legal restructuring, or non-compliance)
    as a function of their effective wealth tax burden.

    Parameters:
    - top_pct (float): threshold above which individuals are considered part of the top wealth group (default: 99.8th percentile)
    - base_prob (float): baseline probability of migration at zero tax (default: 4%)
    - elasticity (float): behavioral response elasticity of migration to net-of-tax rate

    Returns:
    - df (DataFrame): updated DataFrame with migration exit flags and adjusted tax contributions
    """
    df = df.copy()
    df["Migration_Exit"] = False

    net_of_tax = 1 - df["final_tax"] / (df["netwealth_individual"] + 1e-6)

    # migration probability using exponential behavioral model ---
    # Based on stock elasticity to net-of-tax rate
    exit_prob = base_migration_prob * np.exp(elasticity * (1 - net_of_tax))

    # TODO: Check whether it is correct that the probability applies to the whole population and not just to the top wealth group
    top_wealth_group = df["wealth_rank"] > wealth_threshold
    will_migrate = (np.random.rand(len(df)) < exit_prob) & top_wealth_group

    df.loc[will_migrate, "Migration_Exit"] = True
    df.loc[will_migrate, ["sim_tax", "final_tax", "taxable_wealth_eroded"]] = 0

    return df


def apply_regional_tax_adjustments(
    df: pd.DataFrame, tax_reduction: float = 0.3
) -> pd.DataFrame:
    """Adjust taxable wealth and tax values to account for regional exemptions
    in the Spanish wealth tax system, based on estimates from Durán-Cabré et al. (2021).

    Assumes a fixed 30% reduction due to regional policies in areas like Madrid, Galicia, and Andalucía.
    """
    df = df.copy()
    adjustment_factor = 1 - tax_reduction

    df["adjusted_taxable_wealth"] = df["taxable_wealth_eroded"] * adjustment_factor
    df["adjusted_sim_tax"] = df["sim_tax"] * adjustment_factor
    df["adjusted_final_tax"] = df["final_tax"] * adjustment_factor

    return df


def generate_summary_table(df, weight_col="facine3"):
    revenue_collected = (df["adjusted_final_tax"] * df[weight_col]).sum()
    revenue_without_cap = (df["adjusted_sim_tax"] * df[weight_col]).sum()
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
                    "Avg Sim Tax": np.average(
                        g["adjusted_sim_tax"], weights=g[weight_col]
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


def compute_effective_tax_rates(df):
    df = df.copy()
    df["eff_tax_rate"] = df["adjusted_final_tax"] / (df["netwealth_individual"] + 1e-6)
    df["eff_tax_rate"] = df["eff_tax_rate"].replace([np.inf, -np.inf], np.nan)

    df["eff_tax_nocap"] = df["adjusted_sim_tax"] / (df["netwealth_individual"] + 1e-6)
    df["eff_tax_nocap"] = df["eff_tax_nocap"].replace([np.inf, -np.inf], np.nan)

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


def compute_net_wealth_post_tax(df):
    df = df.copy()
    df["wealth_after_cap"] = df["netwealth_individual"] - df[
        "adjusted_final_tax"
    ].fillna(0)
    df["wealth_after_no_cap"] = df["netwealth_individual"] - df[
        "adjusted_sim_tax"
    ].fillna(0)
    return df


def top_share(df, col, weight, pct):
    df = df[[col, weight]].dropna().sort_values(col, ascending=False).copy()
    df["cum_weight"] = df[weight].cumsum()
    cutoff = df[weight].sum() * pct
    df["in_top"] = df["cum_weight"] <= cutoff
    top_sum = (df.loc[df["in_top"], col] * df.loc[df["in_top"], weight]).sum()
    total = (df[col] * df[weight]).sum()
    return top_sum / total


def compute_inequality_metrics(df):
    df = compute_net_wealth_post_tax(df)

    metrics = {
        "Gini Before Tax": gini(
            income="netwealth_individual", weights="facine3", data=df
        ),
        "Gini After Tax (cap)": gini(
            income="wealth_after_cap", weights="facine3", data=df
        ),
        "Gini After Tax (no cap)": gini(
            income="wealth_after_no_cap", weights="facine3", data=df
        ),
        "Top 10% Share Before": top_share(df, "netwealth_individual", "facine3", 0.10),
        "Top 10% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.10),
        "Top 10% Share After (no cap)": top_share(
            df, "wealth_after_no_cap", "facine3", 0.10
        ),
        "Top 1% Share Before": top_share(df, "netwealth_individual", "facine3", 0.01),
        "Top 1% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.01),
        "Top 1% Share After (no cap)": top_share(
            df, "wealth_after_no_cap", "facine3", 0.01
        ),
    }

    print("\n--- Inequality Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4%}")
    return metrics


def main():
    np.random.seed(42)

    df = load_data()
    df = assign_typology(df)

    df = individual_split(df)
    df["wealth_rank"] = df["riquezanet"].rank(pct=True)

    df = simulate_household_wealth_tax(
        df,
        exemption_amount=700_000,
        brackets=PROGRESSIVE_TAX_BRACKETS,
        asset_cols=NON_TAXABLE_ASSET_COLS,
    )
    df = apply_valuation_manipulation(df)
    df = apply_behavioral_response(df)
    df = simulate_pit_liability(df)
    df = apply_wealth_tax_income_cap(df)
    df = simulate_migration_attrition(df)
    print(df["Migration_Exit"].value_counts())
    df = apply_regional_tax_adjustments(df)

    generate_summary_table(df)
    typology_impact_summary(df)

    # Plots
    # plot_tax_rate_by_wealth(df)
    # plot_cap_relief_by_income(df)

    compute_effective_tax_rates(df)
    summarize_cap_and_tax_shares(df)

    df["wealth_after_cap"] = df["netwealth_individual"] - df["final_tax"].fillna(0)
    df["wealth_after_no_cap"] = df["netwealth_individual"] - df["sim_tax"].fillna(0)

    compute_inequality_metrics(df)


if __name__ == "__main__":
    main()
