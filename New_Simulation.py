import pandas as pd
import numpy as np

from constants import (
    PROGRESSIVE_TAX_BRACKETS,
    wealth_percentile,
    Net_Wealth,
    Income,
    Primary_Residence,
    Business_Value,
    Residence_Ownership,
    Business_Ownership,
    Num_Workers,
    SPANISH_PIT_2022_BRACKETS,
    
)
from dta_handling import load_data
from eff_typology import assign_typology


from ineqpy.inequality import gini

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
    taxable_income = np.maximum(df[Income] - personal_allowance, 0)

    df["pit_liability"] = taxable_income.apply(
        lambda amount: calculate_tax_liability(amount, SPANISH_PIT_2022_BRACKETS)
    )

    # Show PIT before correction for context
    total_pit = (df["pit_liability"] * df[weight_col]).sum()

    print(f"Total PIT (before correction):  €{total_pit:,.2f}")
    return df

from reporting import (
    summarize_cap_and_tax_shares,
    report_effective_tax_rates,
    typology_impact_summary,
    generate_summary_table2,
    compute_inequality_metrics,
    payer_coverage,
    loss_breakdown,
)
from wealth_tax import simulate_household_wealth_tax, simulate_pit_liability2


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
    eligible = df[Net_Wealth] < 20_000_000
    income_limit = df[Income] * income_cap_rate
    wealth_tax = df["tax_afterBR"].fillna(0)
    income_tax = df["pit_liability"].fillna(0)


    total_tax = wealth_tax + income_tax
    over_cap = (total_tax > income_limit) & eligible

    max_allowed_relief = wealth_tax * (1 - min_wealth_tax_share)

    excess = total_tax - income_limit
    wt_relief = np.minimum(excess, max_allowed_relief)
    wt_relief = np.where(over_cap, wt_relief, 0.0)

    df["cap_relief"] = wt_relief
    df["final_tax"] = wealth_tax - wt_relief
    wealth_tax = df["tax_afterBR"].fillna(0)


    return df


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
        df[Net_Wealth] - non_taxable_assets - df["exempt_total"]
    )
    df["taxable_wealth"] = np.maximum(adjusted_wealth - exemption_amount, 0)

    df["sim_tax"] = df["taxable_wealth"].apply(
        lambda amount: calculate_tax_liability(amount, PROGRESSIVE_TAX_BRACKETS)
    )
    df["sim_tax_original"] = df["sim_tax"]  # preserve pre-erosion value

    return df

def assign_behavioral_erosion_from_elasticity(
    row, ref_tax_rate=0.004, elasticity=0.25, max_erosion=0.10
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
        return 0.06
    if p > 0.999:
        return 0.05
    elif p > 0.99:
        return 0.04
    elif p > 0.90:
        return 0.020
    else:
        return 0.01
    

def apply_behavioral_response(df, max_erosion=0.3):
    """
    Applies behavioral erosion based on wealth rank, assigning elasticities
    in line with evidence from Jakobsen et al. (2020), Duran-Cabré et al. (2019), etc.

    Parameters:
    - max_erosion: Maximum erosion allowed (cap), default 30%
    """
    df = df.copy()

    # Assign elasticity based on rank (percentile)
    def elasticity(rank):
        if rank > 0.9999:
            return 1.50
        elif rank > 0.999:
            return 0.80
        elif rank > 0.99:
            return 0.50
        elif rank > 0.90:
            return 0.20
        else:
            return 0.10

    df["elasticity"] = df["wealth_rank"].apply(elasticity)

    # Compute behavioral erosion θ_i = min(elasticity * base_rate, max_erosion)
    # Assume an average perceived statutory burden (0.004 = 0.4%) baseline
    ref_rate = 0.004
    df["behavioral_erosion"] = np.clip(df["elasticity"] * ref_rate, 0, max_erosion)

    # Apply erosion to taxable wealth
    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - df["behavioral_erosion"])

    # Diagnostics
    taxpayers = df["sim_tax_original"] > 0
    erosion_weighted = (df.loc[taxpayers, "behavioral_erosion"] * df.loc[taxpayers, "facine3"]).sum() / df.loc[taxpayers, "facine3"].sum()

    print("\n[Behavioral Response by Rank]")
    print(df["behavioral_erosion"].describe())
    print(f"Avg erosion among payers: {erosion_weighted:.4%}")
    print(f"Taxpayers affected: {(taxpayers.mean()*100):.2f}%")

    return df





   #def apply_behavioral_response(df, ref_tax_rate=0.003, max_erosion=0.8):
    """
    Apply behavioral erosion based on wealth-ranked elasticity to simulate real-world avoidance.
    Must be called after initial simulate_wealth_tax(), before income cap.
    """
    df = df.copy()
    

    schedule = [
        (0.9999, 1.9),
        (0.999, 0.8),
        (0.990, 0.5),
        (0.900, 0.2),
    ]
    eff = df["sim_tax"] / (df["taxable_wealth"] + 1e-6)

    thresholds, values = zip(*schedule)
    conditions = [df["wealth_rank"] > t for t in thresholds]
    elasticity = np.select(conditions, values, default=0.35)

    # 3. Behavioural erosion factor θ
    theta = 1 - ((1 - eff) / (1 - ref_tax_rate)) ** elasticity
    theta = np.clip(theta, 0, max_erosion)
    theta[(eff <= 0) | (eff >= 1) | np.isnan(eff)] = 0.0

    df["behavioral_erosion"] = theta
    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - theta)
    print(df["behavioral_erosion"].describe())


    return df

def recalculate_wealth_tax_on_eroded_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute sim_tax using taxable_wealth_eroded instead of the original base.
    This ensures behavioral erosion actually reduces the tax owed.
    """
    df = df.copy()
    df["tax_afterBR"] = df["taxable_wealth_eroded"].apply(
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

    net_of_tax = 1 - df["final_tax"] / (df[Net_Wealth] + 1e-6)

    # migration probability using exponential behavioral model ---
    # Based on stock elasticity to net-of-tax rate
    exit_prob = base_migration_prob * np.exp(elasticity * (1 - net_of_tax))

    # TODO: Check whether it is correct that the probability applies to the whole population and not just to the top wealth group
    top_wealth_group = df["wealth_rank"] > wealth_threshold
    will_migrate = (np.random.rand(len(df)) < exit_prob) & top_wealth_group

    df.loc[will_migrate, "Migration_Exit"] = True
    df.loc[will_migrate, ["sim_tax_original", "tax_afterBR", "final_tax", "taxable_wealth_eroded"]] = 0


    return df


def apply_regional_tax_adjustments(
    df: pd.DataFrame, tax_reduction: float = 0.083
) -> pd.DataFrame:
    """Adjust taxable wealth and tax values to account for regional exemptions such as Andalusia
    """
    df = df.copy()
    adjustment_factor = 1 - tax_reduction

    df["adjusted_sim_tax_original"] = df["sim_tax_original"] * adjustment_factor
    df["adjusted_tax_afterBR"] = df["tax_afterBR"] * adjustment_factor
    df["adjusted_final_tax"] = df["final_tax"] * adjustment_factor

    return df


def compute_effective_tax_rates(df):
    df = df.copy()
    df["eff_tax_rate"] = df["adjusted_final_tax"] / (df[Net_Wealth] + 1e-6)
    df["eff_tax_rate"] = df["eff_tax_rate"].replace([np.inf, -np.inf], np.nan)

    df["eff_tax_nocap"] = df["adjusted_tax_afterBR"] / (df[Net_Wealth] + 1e-6)
    df["eff_tax_nocap"] = df["eff_tax_nocap"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_net_wealth_post_tax(df):
    df = df.copy()
    df["wealth_after_cap"] = df[Net_Wealth] - df[
        "adjusted_final_tax"
    ].fillna(0)
    df["wealth_after_no_cap"] = df[Net_Wealth] - df[
        "adjusted_tax_afterBR"
    ].fillna(0)
    return df


def check_valid_input_data(df):
    assert not (df[Net_Wealth].isna()).any()


def compute_weighted_wealth_rank(df, wealth_col=Net_Wealth, weight_col="facine3"):
    df = df.copy()
    
    # Sort and calculate cumulative weight
    df_sorted = df[[wealth_col, weight_col]].copy()
    df_sorted["orig_index"] = df_sorted.index  # preserve original position
    df_sorted = df_sorted.sort_values(by=wealth_col, kind="mergesort").reset_index(drop=True)
    
    df_sorted["cum_weight"] = df_sorted[weight_col].cumsum()
    total_weight = df_sorted[weight_col].sum()
    df_sorted["wealth_rank"] = df_sorted["cum_weight"] / total_weight

    # Merge back by original index
    df = df.merge(df_sorted[["orig_index", "wealth_rank"]], left_index=True, right_on="orig_index")
    df.drop(columns=["orig_index"], inplace=True)

    return df


def main():
    np.random.seed(42)

    df = load_data()

    check_valid_input_data(df)
    df = assign_typology(df)
    df = compute_weighted_wealth_rank(df, Net_Wealth, "facine3")



    df = simulate_household_wealth_tax(df, exemption_amount=700_000)
    df["sim_tax_original"] = df["sim_tax"]
    #df = apply_valuation_manipulation(df)
    #df = assign_behavioral_erosion_from_elasticity(df)
    df = apply_behavioral_response(df)
    df = recalculate_wealth_tax_on_eroded_base(df)
    df = simulate_pit_liability(df)
    df = apply_wealth_tax_income_cap(df)
    df = simulate_migration_attrition(df)
    print(df["Migration_Exit"].value_counts())
    df = apply_regional_tax_adjustments(df)

    generate_summary_table2(df)
    typology_impact_summary(df)

    # Plots
    # plot_tax_rate_by_wealth(df)
    # plot_cap_relief_by_income(df)

    df = compute_effective_tax_rates(df)
    report_effective_tax_rates(df)
    summarize_cap_and_tax_shares(df)

    df["wealth_after_cap"] = df[Net_Wealth] - df["final_tax"].fillna(0)
    df["wealth_after_no_cap"] = df[Net_Wealth] - df["tax_afterBR"].fillna(0)

    df = compute_net_wealth_post_tax(df)

    compute_inequality_metrics(df)
    payer_coverage(df)
    loss_breakdown(df)

    relieved = df["cap_relief"] > 0
    share_relieved = (df[relieved]["facine3"].sum() / df["facine3"].sum()) * 100
    avg_relief = (df[relieved]["cap_relief"] * df[relieved]["facine3"]).sum() / df["facine3"].sum()

    print(f"% of population receiving relief: {share_relieved:.2f}%")
    print(f"Average relief per capita (weighted): €{avg_relief:,.2f}")

    print("Avg behavioral erosion:", df["behavioral_erosion"].mean())
    print("Total revenue lost to erosion: €", ((df["sim_tax_original"] - df["tax_afterBR"]) * df["facine3"]).sum())



if __name__ == "__main__":
    main()
