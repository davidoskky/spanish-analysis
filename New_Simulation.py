import pandas as pd
import numpy as np

from constants import (
    PROGRESSIVE_TAX_BRACKETS,
<<<<<<< HEAD
    wealth_percentile,
=======
    NON_TAXABLE_ASSET_COLS,
    Net_Wealth,
>>>>>>> d04b5686ed761e61e7e664ee9ac31b9493d4b6c8
)
from dta_handling import load_data
from eff_typology import assign_typology

<<<<<<< HEAD
from ineqpy.inequality import gini

def individual_split(df):
    """
    Defines 'individual' income and wealth as household-level values,
    without performing any splitting.

    This allows the rest of the simulation code to remain unchanged,
    while treating the household as a single taxpayer unit.
    """
    df = df.copy()

    # No splitting: just assign values directly
    df["income_individual"] = df[Income]
    df["netwealth_individual"] = df[Net_Wealth]

    print(df["income_individual"].describe())
    return df


def apply_valuation_manipulation(df, real_estate_discount=0.15, business_discount=0.20):
    """
    Adjusts reported asset values for typical underreporting in household surveys.

    Applies empirical discounts to real estate and business holdings, in line with literature
    showing systematic undervaluation in self-reported data.

    References:
      - Alstadsæter et al. (2019), AER
      - Advani & Tarrant (2022), IFS
      - Duran-Cabré et al. (2023)

    Parameters:
    - real_estate_discount: fraction to reduce real estate values by (default: 15%)
    - business_discount: fraction to reduce business asset values by (default: 20%)
    """
    df = df.copy()
    df[Primary_Residence] = df[Primary_Residence] * (1 - real_estate_discount)
    df[Business_Value] = df[Business_Value] * (1 - business_discount)
    return df


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
    business_exemption_rate = 0.30  # Based on literature(Duran-Cabré et al. 2021)
    has_business_value = df[Business_Ownership] == 1
    apply_business_exempt = (
        np.random.rand(len(df)) < business_exemption_rate
    ) & has_business_value
    business_exempt = np.where(apply_business_exempt, df[Business_Value].fillna(0), 0)

    return exempt_home_value + business_exempt

def simulate_pit_liability(df: pd.DataFrame, correction_top1=0.15, weight_col="facine3"):
    """
    Simulates Spanish PIT liability with a basic personal allowance.
    Also applies an upward correction to the top 1% to approximate unreported capital income.

    Parameters:
    - correction_top1: fractional increase in PIT for top 1% wealth (default: 15%)
    - weight_col: name of weight column (default: 'facine3')
    """
    df = df.copy()

    # Personal allowance (2022): €5,550 per taxpayer
    personal_allowance = 5550
    taxable_income = np.maximum(df["income_individual"] - personal_allowance, 0)

    df["pit_liability"] = taxable_income.apply(
        lambda amount: calculate_tax_liability(amount, SPANISH_PIT_2022_BRACKETS)
    )

    # Weighted wealth rank to identify top 1%
    df = df.sort_values("netwealth_individual")
    df["cum_weight"] = df[weight_col].cumsum()
    total_weight = df[weight_col].sum()
    df["wealth_rank"] = df["cum_weight"] / total_weight

    # Apply capital income correction to top 1%
    is_top_1 = df["wealth_rank"] > 0.99
    df["pit_liability_adjusted"] = df["pit_liability"]
    df.loc[is_top_1, "pit_liability_adjusted"] *= (1 + correction_top1)

    # Show PIT before correction for context
    total_pit = (df["pit_liability"] * df[weight_col]).sum()
    total_pit_adjusted = (df["pit_liability_adjusted"] * df[weight_col]).sum()

    print(f"Total PIT (before correction):  €{total_pit:,.2f}")
    print(f"Total PIT (after correction):   €{total_pit_adjusted:,.2f}")

    return df
=======
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
>>>>>>> d04b5686ed761e61e7e664ee9ac31b9493d4b6c8


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
    eligible = df["netwealth_individual"] < 1_000_000_000
    income_limit = df["income_individual"] * income_cap_rate
    wealth_tax = df["sim_tax"]
    income_tax = df["pit_liability_adjusted"].fillna(0)


    total_tax = wealth_tax + income_tax
    over_cap = (total_tax > income_limit)

    max_allowed_relief = wealth_tax * (1 - min_wealth_tax_share)

    excess = total_tax - income_limit
    wt_relief = np.minimum(excess, max_allowed_relief)
    wt_relief = np.where(over_cap, wt_relief, 0.0)

    df["cap_relief"] = wt_relief
    df["final_tax"] = wealth_tax - wt_relief

    return df


<<<<<<< HEAD
def calculate_tax_liability(
    amount: float, brackets: list[tuple[float, float, float]]
) -> float:
    """
    Compute total tax liability using progressive brackets.
    """
    return sum(
        max(0, min(amount, upper_limit) - lower_limit) * rate
        for lower_limit, upper_limit, rate in brackets
    )


def simulate_household_wealth_tax(
    df: pd.DataFrame, exemption_amount: int = 700_000
) -> pd.DataFrame:
    """
    Simulate a progressive wealth tax based on individual net wealth,
    taking into account legal exemptions and non-taxable assets.

    Returns:
    -------
    pd.DataFrame
        Original DataFrame with added columns:
            - exempt_total: legal exemption calculated for each individual.
            - taxable_wealth: wealth subject to tax after exemptions.
            - sim_tax: simulated tax owed under a progressive tax system.
    """

    df = df.copy()

    df["exempt_total"] = compute_legal_exemptions(df)

    earners = df[Num_Workers]
    adult_equivalent = earners.clip(lower=1)

    # Non-taxable assets: art, vehicles, pension funds
    non_taxable_assets = (
        df["p2_71"].fillna(0) + df["timpvehic"].fillna(0) + df["p2_84"].fillna(0)
    ) / adult_equivalent

    # Taxable wealth = net wealth - non-taxable assets - legal exemptions - base exemption
    adjusted_wealth = (
        df["netwealth_individual"] - non_taxable_assets - df["exempt_total"]
    )
    df["taxable_wealth"] = np.maximum(adjusted_wealth - exemption_amount, 0)

    df["sim_tax"] = df["taxable_wealth"].apply(
        lambda amount: calculate_tax_liability(amount, PROGRESSIVE_TAX_BRACKETS)
    )

    return df


def assign_behavioral_erosion_from_elasticity(
    row, ref_tax_rate=0.004, elasticity=0.25, max_erosion=0.10
=======
def apply_behavioral_response(
    df,
    ref_tax_rate=0.004,
    max_erosion: float = 0.35,
    wealth_col: str = "netwealth_individual",
>>>>>>> d04b5686ed761e61e7e664ee9ac31b9493d4b6c8
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
<<<<<<< HEAD
    net_wealth = row.get(Net_Wealth, 0)
    sim_tax = row.get("sim_tax", 0)
    tax_base = row.get("taxable_wealth", 0)

    if net_wealth <= 1e-6 or sim_tax <= 0:
        return 0.0
    eff_rate = sim_tax / tax_base


    if eff_rate <= 0 or eff_rate >= 1:
        return 0.0

    erosion = 1 - ((1 - eff_rate) / (1 - ref_tax_rate)) ** elasticity

    return min(max(erosion, 0), max_erosion)


def get_grouped_elasticity(row):
    """
    Assign elasticity based on wealth rank group.
    """
    p = row.get("wealth_rank", 0)
    if p > 0.9999:
        return 0.07
    if p > 0.999:
        return 0.05
    elif p > 0.99:
        return 0.04
    elif p > 0.90:
        return 0.020
    else:
        return 0.01


def apply_behavioral_response(df, ref_tax_rate=0.004):
    """
    Apply behavioral erosion based on wealth-ranked elasticity to simulate real-world avoidance.
    Must be called after initial simulate_wealth_tax(), before income cap.
    """
=======

>>>>>>> d04b5686ed761e61e7e664ee9ac31b9493d4b6c8
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

def recalculate_wealth_tax_on_eroded_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute sim_tax using taxable_wealth_eroded instead of the original base.
    This ensures behavioral erosion actually reduces the tax owed.
    """
    df = df.copy()
    df["sim_tax"] = df["taxable_wealth_eroded"].apply(
        lambda amount: calculate_tax_liability(amount, PROGRESSIVE_TAX_BRACKETS)
    )
    return df

def simulate_migration_attrition(
    df: pd.DataFrame,
    wealth_threshold: float = 0.998,
    base_migration_prob: float = 0.02,
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
    df: pd.DataFrame, tax_reduction: float = 0.093
) -> pd.DataFrame:
    """Adjust taxable wealth and tax values to account for regional exemptions such as Andalusia
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

def compute_weighted_wealth_rank(df, wealth_col="netwealth_individual", weight_col="weight"):
    df = df.sort_values(by=wealth_col).copy()
    df["cum_weight"] = df[weight_col].cumsum()
    total_weight = df[weight_col].sum()
    df["wealth_rank"] = df["cum_weight"] / total_weight
    return df


def main():
    np.random.seed(42)

    df = load_data()

    check_valid_input_data(df)
    df = assign_typology(df)

    df = individual_split(df)
    df = compute_weighted_wealth_rank(df, "netwealth_individual", "facine3")



<<<<<<< HEAD
    df = simulate_household_wealth_tax(df, exemption_amount=700_000)
    #df = apply_valuation_manipulation(df)
=======
    df = simulate_household_wealth_tax(
        df,
        exemption_amount=700_000,
        brackets=PROGRESSIVE_TAX_BRACKETS,
        asset_cols=NON_TAXABLE_ASSET_COLS,
    )
    df = apply_valuation_manipulation(df)
>>>>>>> d04b5686ed761e61e7e664ee9ac31b9493d4b6c8
    df = apply_behavioral_response(df)
    df = recalculate_wealth_tax_on_eroded_base(df)
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

    relieved = df["cap_relief"] > 0
    share_relieved = (df[relieved]["facine3"].sum() / df["facine3"].sum()) * 100
    avg_relief = (df[relieved]["cap_relief"] * df[relieved]["facine3"]).sum() / df["facine3"].sum()

    print(f"% of population receiving relief: {share_relieved:.2f}%")
    print(f"Average relief per capita (weighted): €{avg_relief:,.2f}")


if __name__ == "__main__":
    main()
