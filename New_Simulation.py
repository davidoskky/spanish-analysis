import pandas as pd
import numpy as np

from constants import (
    PROGRESSIVE_TAX_BRACKETS,
    NON_TAXABLE_ASSET_COLS,
    Net_Wealth,
)
from dta_handling import load_data
from eff_typology import assign_typology

from preprocessing import individual_split, apply_valuation_manipulation
from reporting import (
    summarize_cap_and_tax_shares,
    report_effective_tax_rates,
    typology_impact_summary,
    generate_summary_table,
    compute_inequality_metrics,
    payer_coverage,
    loss_breakdown,
)
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


def compute_effective_tax_rates(df):
    df = df.copy()
    df["eff_tax_rate"] = df["adjusted_final_tax"] / (df["netwealth_individual"] + 1e-6)
    df["eff_tax_rate"] = df["eff_tax_rate"].replace([np.inf, -np.inf], np.nan)

    df["eff_tax_nocap"] = df["adjusted_sim_tax"] / (df["netwealth_individual"] + 1e-6)
    df["eff_tax_nocap"] = df["eff_tax_nocap"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_net_wealth_post_tax(df):
    df = df.copy()
    df["wealth_after_cap"] = df["netwealth_individual"] - df[
        "adjusted_final_tax"
    ].fillna(0)
    df["wealth_after_no_cap"] = df["netwealth_individual"] - df[
        "adjusted_sim_tax"
    ].fillna(0)
    return df


def check_valid_input_data(df):
    assert not (df[Net_Wealth].isna()).any()


def main():
    np.random.seed(42)

    df = load_data()

    check_valid_input_data(df)
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

    df = compute_effective_tax_rates(df)
    report_effective_tax_rates(df)
    summarize_cap_and_tax_shares(df)

    df["wealth_after_cap"] = df["netwealth_individual"] - df["final_tax"].fillna(0)
    df["wealth_after_no_cap"] = df["netwealth_individual"] - df["sim_tax"].fillna(0)
    df = compute_net_wealth_post_tax(df)

    compute_inequality_metrics(df)
    payer_coverage(df)
    loss_breakdown(df)


if __name__ == "__main__":
    main()
