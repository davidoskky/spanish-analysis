import pandas as pd
import numpy as np
from constants import (Residence_Ownership, Business_Value, Business_Ownership ,Primary_Residence, Num_Workers, PEOPLE_IN_HOUSEHOLD, Net_Wealth, Income, wealth_percentile, working_status, income_percentile)

def individual_split(df):
    """
    Decomposes household-level net wealth and income into individual-level equivalents.
    
    Since income and wealth are reported per household, this function tries to approximate 
    per-capita figures by dividing by the number of economic contributors (working adults).
    Where no earners are reported, household size is used as a fallback proxy.
    """
   
    df = df.copy()
    
    # Approximate the number of working individuals in the household
    earners = pd.to_numeric(df[Num_Workers], errors="coerce").fillna(0)
    hh_size = df[PEOPLE_IN_HOUSEHOLD]

    # Use earners where possible; otherwise, fall back on household size to avoid div/0
    # Note: using clip to prevent very small divisor values (at least one economic agent)
    adult_split_factor = earners.where(earners != 0, hh_size).clip(lower=1)

    df["netwealth_individual"] = df[Net_Wealth] / adult_split_factor
    df["income_individual"] = df[Income] / adult_split_factor

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
    df[Primary_Residence] = df[Primary_Residence].fillna(0) * (1 - real_estate_discount)
    df[Business_Value] = df[Business_Value].fillna(0) * (1 - business_discount)
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
    apply_business_exempt = (np.random.rand(len(df)) < business_exemption_rate) & has_business_value
    business_exempt = np.where(apply_business_exempt, df[Business_Value].fillna(0), 0) 

    return exempt_home_value + business_exempt

def apply_income_cap(df, income_cap_rate=0.60, min_wt_share=0.20):
    """
    Implements an income-based ceiling on individual tax liability.

    This function ensures that no individual pays more than a fixed proportion 
    of their income in simulated wealth taxes, with a safeguard to preserve 
    a minimum portion of the original tax estimate.

    Parameters:
    - income_cap_rate (float): Maximum fraction of income an individual can be taxed (default = 60%)
    - min_wt_share (float): Minimum share of original tax that must still be paid (default = 20%)

    Returns:
    - df (DataFrame): Modified DataFrame including capped tax amounts and intermediate calculations
    """
    df = df.copy()

    cap_threshold = df["income_individual"] * income_cap_rate
    wt = df["sim_tax"]

    df["cap_applicable"] = wt > cap_threshold 

    excess = wt - cap_threshold
    max_relief = wt * (1 - min_wt_share)
    relief = np.where(df["cap_applicable"], np.minimum(excess, max_relief), 0.0)

    df["cap_relief"] = relief
    df["final_tax"] = wt - relief

    return df

def simulate_wealth_tax(df, exemption=700_000, income_cap_rate=0.6):
    df = df.copy()

    # tax rates and brackets based on the Spanish wealth tax structure
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

    df["exempt_total"] = compute_legal_exemptions(df)

    earners = pd.to_numeric(df["nnumadtrab"], errors="coerce").fillna(0)
    household_size = df[PEOPLE_IN_HOUSEHOLD]
    adult_split_factor = earners.where(earners != 0, household_size).clip(lower=1)

#  non-taxable assets = art, antiques, vehicles, and pension funds
    non_taxable_assets = (
    df["p2_71"].fillna(0) +
    df["timpvehic"].fillna(0) +
    df["p2_84"].fillna(0)
) / adult_split_factor
    # Calculate taxable wealth
    base = df["netwealth_individual"] - non_taxable_assets - df["exempt_total"]
    df["taxable_wealth"] = np.maximum(base - exemption, 0)
    df["sim_tax"] = df["taxable_wealth"].apply(compute_tax)

    return df

def assign_behavioral_erosion_from_elasticity(row, ref_tax_rate=0.004, elasticity=0.64, max_erosion=0.30):
    """
    Compute behavioral erosion factor θ_i = 1 - ((1 - τ_eff) / (1 - τ_ref))^ε
    - τ_eff: effective tax rate for individual i
    - τ_ref: reference average effective rate (e.g., 0.004)
    - ε: elasticity of taxable wealth (e.g., 0.64)
    
     Sources:
    - Jakobsen et al. (2020), QJE
    - Seim (2017), AER
    - Duran-Cabré et al. (2023), WP
    """
    net_wealth = row.get("income_individual", 0)
    sim_tax = row.get("sim_tax", 0)

    if net_wealth <= 1e-6 or sim_tax <= 0:
        return 0.0
    eff_rate = sim_tax / net_wealth

    if eff_rate <= 0 or eff_rate >= 1:
        return 0.0

    erosion = 1 - ((1 - eff_rate) / (1 - ref_tax_rate)) ** elasticity

    return min(max(erosion, 0), max_erosion)


def get_grouped_elasticity(row):
    """
    Assign elasticity based on wealth rank group.
    """
    p = row.get("wealth_rank", 0)
    if p > 0.999:
        return 0.80
    elif p > 0.99:
        return 0.50
    elif p > 0.90:
        return 0.35
    else:
        return 0.10


def apply_behavioral_response(df, ref_tax_rate=0.004):
    """
    Apply behavioral erosion based on wealth-ranked elasticity to simulate real-world avoidance.
    Must be called after initial simulate_wealth_tax(), before income cap.
    """
    df = df.copy()

    df["behavioral_erosion"] = df.apply(
        lambda row: assign_behavioral_erosion_from_elasticity(
            row,
            ref_tax_rate=ref_tax_rate,
            elasticity=get_grouped_elasticity(row)
        ),
        axis=1,
    )

    df["taxable_wealth_eroded"] = df["taxable_wealth"] * (1 - df["behavioral_erosion"])
    return df

def simulate_migration_attrition(df, top_pct=0.995, base_prob=0.04, elasticity=1.76):
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

    eff_tax_rate = df["final_tax"] / (df["netwealth_individual"] + 1e-6)
    net_of_tax = 1 - eff_tax_rate

    # migration probability using exponential behavioral model ---
    # Based on stock elasticity to net-of-tax rate
    exit_prob = base_prob * np.exp(elasticity * (1 - net_of_tax))

    top_wealth_group = df["wealth_rank"] > top_pct
    will_migrate = (np.random.rand(len(df)) < exit_prob) & top_wealth_group

    df.loc[will_migrate, "Migration_Exit"] = True
    df.loc[will_migrate, ["sim_tax", "final_tax", "taxable_wealth_eroded"]] = 0  # All tax revenue lost

    return df
def apply_adjustments(df):
    df = df.copy()
    # Assumptions based on academic literature (Durán-Cabré et al. (2021):
    REGIONAL_REDUCTION = (
    0.33  # 33% lost due to regional exemptions (e.g. Madrid, Galicia, Andalucía, etc.)
)
    adjustment_factor = (1 - REGIONAL_REDUCTION)

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
            df.loc[~df["Migration_Exit"], "adjusted_final_tax"] * df.loc[~df["Migration_Exit"], weight_col]
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
                    "Avg Final Tax": np.average(g["adjusted_final_tax"], weights=g[weight_col]),
                    "Avg Sim Tax": np.average(g["adjusted_sim_tax"], weights=g[weight_col]),
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

# DISTRIBUTIONAL PLOTS
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tax_rate_by_wealth(df):
    df_sorted = df.sort_values("wealth_rank").reset_index(
        drop=True
    )
    df_sorted["eff_tax_rate"] = df_sorted["adjusted_final_tax"] / (
        df_sorted["netwealth_individual"] + 1e-6
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="wealth_rank", y="eff_tax_rate", data=df_sorted)
    plt.title("Effective Tax Rate by Wealth Percentile")
    plt.xlabel("Wealth Percentile")
    plt.ylabel("Effective Tax Rate")
    plt.grid(True)
    plt.show()


def plot_cap_relief_by_income(df):
    df = df.copy()
    df["income_decile"] = pd.qcut(
        df["income_individual"], 10, labels=[f"D{i}" for i in range(1, 11)]
    )
    summary = df.groupby("income_decile")["cap_relief"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="income_decile", y="cap_relief", data=summary)
    plt.title("Average Cap Relief by Income Decile")
    plt.xlabel("Income Decile")
    plt.ylabel("Average Cap Relief (EUR)")
    plt.grid(True)
    plt.show()


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

    eff_tax_top10 = weighted_avg(df.loc[top_10, "eff_tax_rate"], df.loc[top_10, "facine3"])
    eff_tax_top1 = weighted_avg(df.loc[top_1, "eff_tax_rate"], df.loc[top_1, "facine3"])
    eff_tax_top10_nocap = weighted_avg(df.loc[top_10, "eff_tax_nocap"], df.loc[top_10, "facine3"])
    eff_tax_top1_nocap = weighted_avg(df.loc[top_1, "eff_tax_nocap"], df.loc[top_1, "facine3"])

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
    df["wealth_after_cap"] = df["netwealth_individual"] - df["adjusted_final_tax"].fillna(0)
    df["wealth_after_no_cap"] = df["netwealth_individual"] - df["adjusted_sim_tax"].fillna(0)
    return df

# Inequality metrics like Gini coefficient and top shares

def gini(array, weights):
    df = pd.DataFrame({"x": array, "w": weights}).dropna()
    df = df.sort_values("x")
    xw = df["x"] * df["w"]
    cumw = df["w"].cumsum()
    cumxw = xw.cumsum()
    rel_cumw = cumw / cumw.iloc[-1]
    rel_cumxw = cumxw / cumxw.iloc[-1]
    return 1 - 2 * np.trapz(rel_cumxw, rel_cumw)

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
    w = df["facine3"]
    metrics = {
        "Gini Before Tax": gini(df["netwealth_individual"], w),
        "Gini After Tax (cap)": gini(df["wealth_after_cap"], w),
        "Gini After Tax (no cap)": gini(df["wealth_after_no_cap"], w),
        "Top 10% Share Before": top_share(df, "netwealth_individual", "facine3", 0.10),
        "Top 10% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.10),
        "Top 10% Share After (no cap)": top_share(df, "wealth_after_no_cap", "facine3", 0.10),
        "Top 1% Share Before": top_share(df, "netwealth_individual", "facine3", 0.01),
        "Top 1% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.01),
        "Top 1% Share After (no cap)": top_share(df, "wealth_after_no_cap", "facine3", 0.01),
    }

    print("\n--- Inequality Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4%}")
    return metrics

def main():
    from dta_handling import load_data
    from eff_typology import assign_typology1

    np.random.seed(42)

    df = load_data()
    df = assign_typology1(df)

    df = individual_split(df)
    df["wealth_rank"] = df["riquezanet"].rank(pct=True)

    df = apply_valuation_manipulation(df)
    df = simulate_wealth_tax(df)                     
    df = apply_behavioral_response(df)            
    df = apply_income_cap(df) 
    df = simulate_migration_attrition(df) 
    print(df["Migration_Exit"].value_counts())
    df = apply_adjustments(df)                       

    generate_summary_table(df)
    typology_impact_summary(df)

    # Plots
     # plot_tax_rate_by_wealth(df)
     # plot_cap_relief_by_income(df)

    compute_effective_tax_rates(df)
    summarize_cap_and_tax_shares(df)


    df["wealth_after_cap"] = df["netwealth_individual"] - df[
    "final_tax"
    ].fillna(0)
    df["wealth_after_no_cap"] = df["netwealth_individual"] - df[
    "sim_tax"
    ].fillna(0)

    compute_inequality_metrics(df)


if __name__ == "__main__":
    main()
