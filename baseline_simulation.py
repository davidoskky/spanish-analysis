import pandas as pd
import numpy as np
import random
import logging
import unicodedata

from data_loaders import (
    load_eff_data,
    load_population_and_revenue_data,
    generate_eff_group_stats,
)
from synthetic_data import generate_and_adjust_households, generate_households_by_size

np.random.seed(42)
random.seed(42)
logging.basicConfig(level=logging.INFO)

# --- Set paths & constants -----------------------------------------------
POP_FILE = "Regional_Age_Bin_Population_Shares.csv"
EFF_FILE = "eff_data.xlsx"
OUT_FILE = "mini_sim.csv"
N_HH = 20_000  # start small


def reweight_to_match_percentile_shares(
    dataframe, value_col="Net_Wealth", weight_col="Weight", percentiles=10
):
    df_copy = dataframe.copy()
    df_copy = df_copy[df_copy[value_col] >= 0].reset_index(drop=True)
    df_copy["Wealth_Rank"] = df_copy[value_col].rank(method="first", pct=True)
    df_copy["Wealth_Percentile"] = pd.qcut(
        df_copy["Wealth_Rank"], q=percentiles, labels=False
    )
    df_copy["Weighted_Wealth"] = df_copy[value_col] * df_copy[weight_col]
    actual_shares = df_copy.groupby("Wealth_Percentile")["Weighted_Wealth"].sum()
    actual_shares /= actual_shares.sum()

    target_shares = np.array(
        [0.00, 0.01, 0.02, 0.04, 0.07, 0.10, 0.13, 0.18, 0.25, 0.20]
    )
    target_shares /= target_shares.sum()

    scaling_factors = target_shares / actual_shares.values
    df_copy["Scaling_Factor"] = df_copy["Wealth_Percentile"].map(
        dict(enumerate(scaling_factors))
    )
    df_copy["Adjusted_Weight"] = df_copy[weight_col] * df_copy["Scaling_Factor"]
    return df_copy


def calculate_population_over_30(pop_file):
    df = pd.read_csv(pop_file)
    df["Region"] = df["Region"].str.replace(r"^\d+\s+", "", regex=True)
    df["Region"] = df["Region"].apply(
        lambda x: unicodedata.normalize("NFKD", x.strip())
        .encode("ascii", errors="ignore")
        .decode("utf-8")
        .lower()
    )

    province_to_region = {
        "madrid": "madrid",
        "madrid, comunidad de": "madrid",
        "barcelona": "catalonia",
        "girona": "catalonia",
        "lleida": "catalonia",
        "tarragona": "catalonia",
        "cataluna": "catalonia",
        "valencia/valencia": "valencia",
        "alicante/alacant": "valencia",
        "castellon/castello": "valencia",
        "comunitat valenciana": "valencia",
        "coruna, a": "galicia",
        "lugo": "galicia",
        "ourense": "galicia",
        "pontevedra": "galicia",
        "asturias, principado de": "asturias",
        "asturias": "asturias",
        "caceres": "extremadura",
        "badajoz": "extremadura",
    }

    df["Autonomous_Region"] = df["Region"].map(province_to_region)
    df = df[df["Autonomous_Region"].notna()].copy()

    over_30_bins = [
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65+",
    ]
    df = df[df["Age Bin"].isin(over_30_bins)]

    region_pop = df.groupby("Autonomous_Region")["Population"].sum()
    selected_regions = [
        "madrid",
        "catalonia",
        "valencia",
        "galicia",
        "asturias",
        "extremadura",
    ]
    total_population = region_pop.loc[
        region_pop.index.intersection(selected_regions)
    ].sum()

    print(f" Estimated population over 30 in selected regions: {total_population:,.0f}")
    return total_population


def compute_region_targets(region_weights, total_households):
    region_targets = {
        row["Region"]: int(round(row["Population"] * total_households))
        for _, row in region_weights.iterrows()
    }
    print("📌 Computed household targets by region:")
    for region, count in region_targets.items():
        print(f"  {region}: {count} households")
    return region_targets


def scale_region_shares_to_population(
    region_weights_df, simulated_population, avg_household_size=2.5
):
    """
    Scales region shares to household counts based on a simulated target population size.

    Parameters:
        region_weights_df (pd.DataFrame): Must contain 'Region' and 'Population' columns (as share of total).
        simulated_population (int): Target simulated population size (not total population).
        avg_household_size (float): Average household size (default = 2.5).

    Returns:
        pd.DataFrame with added 'Num_Households' column.
    """
    df = region_weights_df.copy()
    total_households = int(round(simulated_population / avg_household_size))
    df["Num_Households"] = (df["Population"] * total_households).round().astype(int)

    # Adjust rounding mismatch
    diff = total_households - df["Num_Households"].sum()
    if diff != 0:
        idx = df["Num_Households"].idxmax()
        df.loc[idx, "Num_Households"] += diff

    print("📌 Computed Num_Households from scaled population:")
    print(df[["Region", "Num_Households"]])

    return df


def recalculate_wealth_ranks(df, value_col="Net_Wealth", weight_col="Final_Weight"):
    df = df[df[value_col] >= 0].copy()
    df = df.sort_values(by=value_col, ascending=False)

    total_weight = df[weight_col].sum()
    if total_weight == 0:
        raise ValueError("Total weight is zero; cannot compute wealth ranks.")

    df["CumWeight"] = df[weight_col].cumsum()
    df["Wealth_Rank_Weighted"] = df["CumWeight"] / total_weight
    return df


def share_in_top_percentile(df, value_col, weight_col, top_pct=0.01):
    df = df[df[value_col] > 0].copy()
    df = df.sort_values(by=value_col, ascending=False)

    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return 0.0

    df["CumWeight"] = df[weight_col].cumsum()
    cutoff = total_weight * top_pct
    df_top = df[df["CumWeight"] <= cutoff]

    total_weighted_value = (df[value_col] * df[weight_col]).sum()
    top_weighted_value = (df_top[value_col] * df_top[weight_col]).sum()

    return (
        top_weighted_value / total_weighted_value if total_weighted_value > 0 else 0.0
    )


from scipy.stats import pareto


def calibrate_top_wealth_share_dual(
    df, top1_pct=0.01, top10_pct=0.10, alpha=2, base_scale=2e5, seed=42
):
    rng = np.random.default_rng(seed)
    dfs = []

    for region, subdf in df.groupby("Region"):
        subdf = subdf.sort_values("Total_Assets", ascending=False).reset_index(
            drop=True
        )
        n = len(subdf)
        top1_n = int(n * top1_pct)
        top10_n = int(n * top10_pct)

        top1_mask = subdf.index < top1_n
        top10_mask = (subdf.index >= top1_n) & (subdf.index < top10_n)

        scale_top1 = base_scale * 14
        scale_top10 = base_scale * 4

        subdf.loc[top1_mask, "Total_Assets"] = pareto.rvs(
            alpha, scale=scale_top1, size=top1_n, random_state=rng
        )
        subdf.loc[top10_mask, "Total_Assets"] = pareto.rvs(
            alpha, scale=scale_top10, size=(top10_n - top1_n), random_state=rng
        )

        subdf["Debts"] = subdf["Total_Assets"] * subdf["Debt_Ratio"]
        subdf["Net_Wealth"] = subdf["Total_Assets"] - subdf["Debts"]

        total_weight = subdf["Weight"].sum()
        if total_weight == 0:
            print(f"⚠️ Region {region} has zero total weight, skipping weight scaling.")
            dfs.append(subdf)
            continue

        # Reset weights uniformly before scaling
        subdf["Weight"] = total_weight / len(subdf)

        # Proportional re-scaling
        if subdf.loc[top1_mask, "Weight"].sum() > 0:
            subdf.loc[top1_mask, "Weight"] *= (top1_pct * total_weight) / subdf.loc[
                top1_mask, "Weight"
            ].sum()
        if subdf.loc[top10_mask, "Weight"].sum() > 0:
            subdf.loc[top10_mask, "Weight"] *= (
                (top10_pct - top1_pct) * total_weight
            ) / subdf.loc[top10_mask, "Weight"].sum()
        remaining_mask = ~(top1_mask | top10_mask)
        if subdf.loc[remaining_mask, "Weight"].sum() > 0:
            subdf.loc[remaining_mask, "Weight"] *= (
                (1.0 - top10_pct) * total_weight
            ) / subdf.loc[remaining_mask, "Weight"].sum()

        dfs.append(subdf)

    df = pd.concat(dfs, ignore_index=True)

    df_sorted = (
        df[df["Net_Wealth"] > 0].sort_values("Net_Wealth", ascending=False).copy()
    )
    df_sorted["CumWeight"] = df_sorted["Weight"].cumsum()
    total_weight = df_sorted["Weight"].sum()

    top1_cut = total_weight * top1_pct
    top10_cut = total_weight * top10_pct

    top1_df = df_sorted[df_sorted["CumWeight"] <= top1_cut]
    top10_df = df_sorted[df_sorted["CumWeight"] <= top10_cut]

    top1_share = (top1_df["Net_Wealth"] * top1_df["Weight"]).sum() / (
        df_sorted["Net_Wealth"] * df_sorted["Weight"]
    ).sum()
    top10_share = (top10_df["Net_Wealth"] * top10_df["Weight"]).sum() / (
        df_sorted["Net_Wealth"] * df_sorted["Weight"]
    ).sum()

    print("\n📊 Calibrated Top Wealth Shares (Post Pareto Tail Injection)")
    print(f"Top 1% Net Wealth Share:  {top1_share:.2%}")
    print(f"Top 10% Net Wealth Share: {top10_share:.2%}")

    return df


def assign_declarant_weights(df):
    df = df.copy()
    df["Declarant_Weight"] = df.get("Weight", 1.0)
    return df


def get_personal_exemption(region):
    return 500_000 if region in ["catalonia", "extremadura", "valencia"] else 700_000


def compute_total_exemption(row):
    personal_exemption = get_personal_exemption(row["Region"])
    primary_exempt = min(row.get("Adj_Real_Assets", 0), 300_000)
    return personal_exemption + primary_exempt + row.get("Business_Exemption", 0.0)


def assign_erosion(row):
    # Step 1: Base erosion by wealth rank
    if row["Wealth_Rank"] > 0.999:
        base_erosion = 0.30
    elif row["Wealth_Rank"] > 0.99:
        base_erosion = 0.25
    elif row["Wealth_Rank"] > 0.90:
        base_erosion = 0.15
    elif row["Wealth_Rank"] > 0.75:
        base_erosion = 0.07
    else:
        base_erosion = 0.02

    # Step 2: Apply modifiers based on asset composition
    modifier = 1.0
    if row.get("Business_Asset_Ratio", 0) > 0.2:
        modifier += 0.10
    if row.get("Real_Asset_Ratio", 0) > 0.4:
        modifier += 0.02
    if row.get("Financial_Asset_Ratio", 0) > 0.4:
        modifier += 0.08
    if row.get("Income", 0) < 0.6 * row.get("Adj_Net_Wealth", 1):
        modifier += 0.05

    # Final erosion factor
    erosion_factor = min(base_erosion * modifier, 0.40)

    # Step 3: Probabilistic dropout based on adjusted net wealth
    nw = row.get("Adj_Net_Wealth", 0)
    if nw < 2_000_000:
        dropout_prob = 0.0
    elif nw < 10_000_000:
        dropout_prob = (nw - 2_000_000) / 8_000_000 * 0.10
    else:
        dropout_prob = 0.10 + min((nw - 10_000_000) / 90_000_000 * 0.20, 0.20)

    dropout = np.random.binomial(1, dropout_prob)

    return pd.Series(
        {
            "Erosion_Factor": erosion_factor,
            "Dropout": dropout,
            "Dropout_Prob": dropout_prob,
        }
    )


def generate_tax_diagnostics(df):
    filtered = df[df["Is_Taxpayer"] == True]
    diagnostics = {
        "Total_Tax_Revenue": (filtered["Wealth_Tax"] * filtered["Final_Weight"]).sum(),
        "Top_1_Wealth_Share": share_in_top_percentile(
            filtered, "Net_Wealth", "Final_Weight", 0.01
        ),
        "Declarant_Count": filtered.shape[0],
    }
    print("\n📊 Diagnostics:")
    for k, v in diagnostics.items():
        print(f"{k}: {v:,.2f}")
    return diagnostics


def calculate_ip_tax(base, region):
    if region == "madrid":
        return 0

    brackets_by_region = {
        "valencia": [
            (167129.45, 0.0025),
            (334252.88, 0.0035),
            (668499.75, 0.0055),
            (1336999.51, 0.0095),
            (2673999.01, 0.0135),
            (5347998.03, 0.0175),
            (10695996.06, 0.0215),
            (float("inf"), 0.035),
        ],
        "catalonia": [
            (167129.45, 0.002),
            (334252.88, 0.003),
            (668499.75, 0.005),
            (1336999.51, 0.009),
            (2673999.01, 0.013),
            (5347998.03, 0.017),
            (10695996.06, 0.021),
            (20000000.0, 0.0348),
            (float("inf"), 0.0348),
        ],
        "galicia": [
            (167129.45, 0.002),
            (334252.88, 0.003),
            (668499.75, 0.005),
            (1336999.51, 0.009),
            (2673999.01, 0.013),
            (5347998.03, 0.017),
            (10695996.06, 0.021),
            (float("inf"), 0.035),
        ],
        "default": [
            (167129.45, 0.002),
            (334252.88, 0.003),
            (668499.75, 0.005),
            (1336999.51, 0.009),
            (2673999.01, 0.013),
            (5347998.03, 0.017),
            (10695996.06, 0.021),
            (float("inf"), 0.025),
        ],
    }
    brackets = brackets_by_region.get(region, brackets_by_region["default"])
    tax = 0
    last_limit = 0
    for limit, rate in brackets:
        if base > limit:
            tax += (limit - last_limit) * rate
            last_limit = limit
        else:
            tax += (base - last_limit) * rate
            break
    return tax


def simulate_pit(income):
    brackets = [
        (12450, 0.19),
        (20200, 0.24),
        (35200, 0.30),
        (60000, 0.37),
        (300000, 0.45),
        (float("inf"), 0.47),
    ]
    tax, last = 0, 0
    for limit, rate in brackets:
        if income > limit:
            tax += (limit - last) * rate
            last = limit
        else:
            tax += (income - last) * rate
            break
    return tax


def apply_tax_cap_and_adjustments(df):
    df = df.copy()
    df["Cap"] = 0.60 * df["Income"]
    over_limit = df["Wealth_Tax"] + df["PIT_Liability"] > df["Cap"]
    df.loc[over_limit, "Wealth_Tax"] = np.maximum(
        0.2 * df.loc[over_limit, "Wealth_Tax"],
        df.loc[over_limit, "Cap"] - df.loc[over_limit, "PIT_Liability"],
    )
    df["Weighted_Wealth_Tax"] = df["Wealth_Tax"] * df["Declarant_Weight"]
    return df


# Constants for migration logic
MIGRATION_THRESHOLDS = {
    "top_01": 0.999,
    "top_1": 0.99,
    "top_5": 0.95,
}
BASE_PROBABILITIES = {
    "top_01": 0.02,
    "top_1": 0.01,
    "top_5": 0.003,
}


def compute_migration_probability(row):
    """Returns the adjusted migration probability for an individual."""
    wealth_rank = row.get("Wealth_Rank", 0)
    prob = 0.0

    # Base probability by wealth tier
    if wealth_rank > MIGRATION_THRESHOLDS["top_01"]:
        prob = BASE_PROBABILITIES["top_01"]
    elif wealth_rank > MIGRATION_THRESHOLDS["top_1"]:
        prob = BASE_PROBABILITIES["top_1"]
    elif wealth_rank > MIGRATION_THRESHOLDS["top_5"]:
        prob = BASE_PROBABILITIES["top_5"]

    # Adjust for wealth tax intensity if available
    if "Wealth_Tax_Baseline" in row and "Adj_Net_Wealth" in row:
        ratio = row["Wealth_Tax_Baseline"] / (row["Adj_Net_Wealth"] + 1e-6)
        prob *= 1 + 2 * min(ratio, 0.02)  # clip ratio to avoid extreme spikes

    return min(prob, 1.0)  # ensure probability is valid


def apply_migration_consequences(df):
    """Zeroes out tax and wealth if individual is a migrant."""
    df.loc[df["Migration_Exit"], "Taxable_Wealth_Eroded"] = 0.0
    df.loc[df["Migration_Exit"], "Wealth_Tax"] = 0.0
    return df


def apply_migration_module(df):
    df = df.copy()

    # Compute migration probabilities
    df["Migration_Prob"] = df.apply(compute_migration_probability, axis=1)
    df["Migration_Exit"] = np.random.rand(len(df)) < df["Migration_Prob"]

    # Apply tax consequences
    df = apply_migration_consequences(df)

    # Optional: Print diagnostics
    migrated_count = df["Migration_Exit"].sum()
    print(
        f"🏃 Migration module applied: {migrated_count:,} individuals exited due to tax burden."
    )

    return df


def run_tax_simulation(df):
    df = df.copy()
    df = apply_region_multipliers(df, region_scaling)

    # 1. Adjust asset values
    df["Adj_Real_Assets"] = df["Real_Assets"] * 0.75
    df["Adj_Financial_Assets"] = df["Financial_Assets"]
    df["Adj_Business_Assets"] = df["Business_Assets"] * 0.70
    df["Adj_Total_Assets"] = (
        df["Adj_Real_Assets"] + df["Adj_Financial_Assets"] + df["Adj_Business_Assets"]
    )
    df["Adj_Net_Wealth"] = df["Adj_Total_Assets"] - df["Debts"]

    # 2. Apply exemptions
    df["Business_Exemption"] = 0.0
    eligible = (df["Business_Asset_Ratio"] > 0.5) & (df["Income"] > 30_000)
    df.loc[eligible, "Business_Exemption"] = df.loc[eligible, "Business_Assets"]

    df["Primary_Residence_Exempt"] = df["Adj_Real_Assets"].clip(upper=300_000)
    df["Personal_Exemption"] = df["Region"].apply(get_personal_exemption)

    reclass_mask = df["Business_Asset_Ratio"] > 0.2
    df["Business_Reclass"] = 0.0
    df.loc[reclass_mask, "Business_Reclass"] = (
        df.loc[reclass_mask, "Adj_Business_Assets"] * 0.2
    )
    df["Adj_Business_Assets"] -= df["Business_Reclass"]
    df["Adj_Total_Assets"] = (
        df["Adj_Real_Assets"] + df["Adj_Financial_Assets"] + df["Adj_Business_Assets"]
    )
    df["Adj_Net_Wealth"] = df["Adj_Total_Assets"] - df["Debts"]

    # 3. Compute tax base
    df["Gross_Tax_Base"] = (
        df["Adj_Net_Wealth"] - df["Primary_Residence_Exempt"] - df["Business_Exemption"]
    )
    df["Net_Tax_Base"] = (df["Gross_Tax_Base"] - df["Personal_Exemption"]).clip(lower=0)
    df["Gross_Assets"] = df["Adj_Total_Assets"]
    df["Is_Declarant"] = (df["Net_Tax_Base"] > 0) | (df["Gross_Assets"] > 2_000_000)
    df["Is_Taxpayer"] = df["Is_Declarant"]
    df["Exemption"] = df.apply(compute_total_exemption, axis=1)

    # 4. Apply erosion and liquidity adjustments
    erosion_df = df.apply(assign_erosion, axis=1)
    df = df.join(erosion_df)
    df["Liquid_Heavy"] = df["Financial_Asset_Ratio"] > 0.4
    df.loc[df["Liquid_Heavy"], "Income"] *= 0.95

    df["Business_High"] = df["Business_Asset_Ratio"] > 0.2
    df["Business_Low"] = ~df["Business_High"]
    df.loc[df["Business_High"], "Erosion_Factor"] *= 1.05
    df.loc[df["Business_Low"], "Erosion_Factor"] *= 1.10
    df["Erosion_Factor"] = df["Erosion_Factor"].clip(upper=0.3)

    # 5. Final tax computations
    df["Taxable_Wealth"] = (df["Adj_Net_Wealth"] - df["Exemption"]).clip(lower=0)
    df["Taxable_Wealth_Eroded"] = df["Taxable_Wealth"] * (1 - df["Erosion_Factor"])

    df = assign_declarant_weights(df)
    df = apply_migration_module(df)

    df["Final_Weight"] = 0.0
    df.loc[df["Is_Taxpayer"], "Final_Weight"] = df.loc[
        df["Is_Taxpayer"], "Declarant_Weight"
    ]

    df["Taxable_Wealth_Baseline"] = df["Taxable_Wealth"]
    df["Wealth_Tax_Baseline"] = df.apply(
        lambda row: calculate_ip_tax(row["Taxable_Wealth_Baseline"], row["Region"]),
        axis=1,
    )
    df["Weighted_Wealth_Tax_Baseline"] = df["Wealth_Tax_Baseline"] * df["Final_Weight"]
    df["Wealth_Tax"] = df.apply(
        lambda row: calculate_ip_tax(row["Taxable_Wealth_Eroded"], row["Region"]),
        axis=1,
    )
    df.loc[df["Dropout"] > 0, "Wealth_Tax"] = 0

    df["PIT_Liability"] = df["Income"].apply(simulate_pit)
    df = apply_tax_cap_and_adjustments(df)

    return df


# Apply erosion to the baseline to simulate behavioral impact
def apply_baseline_behavioral_erosion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assert "Erosion_Factor" in df.columns, (
        "Erosion_Factor must be computed before applying behavioral erosion"
    )
    df["Taxable_Wealth_Baseline_Eroded"] = df["Taxable_Wealth_Baseline"] * (
        1 - df["Erosion_Factor"]
    )
    df["Wealth_Tax_Baseline_Eroded"] = df.apply(
        lambda row: calculate_ip_tax(
            row["Taxable_Wealth_Baseline_Eroded"], row["Region"]
        ),
        axis=1,
    )
    df["Weighted_Wealth_Tax_Baseline_Eroded"] = (
        df["Wealth_Tax_Baseline_Eroded"] * df["Final_Weight"]
    )
    return df


region_scaling = {
    "asturias": 1,
    "catalonia": 1,
    "extremadura": 1,
    "galicia": 1,
    "valencia": 1,
    "madrid": 1,
}


def apply_region_multipliers(df, multipliers, recompute=True):
    """
    Apply region-specific scaling factors to Total_Assets and optionally recompute
    all dependent monetary columns. This function modifies monetary values and
    assumes required ratios are present and non-null.

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'Region' and monetary columns.
        multipliers (dict): Region to scaling factor mapping.
        recompute (bool): Whether to recompute dependent monetary columns.

    Returns:
        pd.DataFrame: Updated DataFrame with scaled and optionally recomputed columns.
    """

    df = df.copy()
    all_regions = df["Region"].unique()

    # Validation for required ratio columns
    required_ratios = ["Debt_Ratio", "Real_Asset_Ratio", "Financial_Asset_Ratio"]
    for col in required_ratios:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values")

    for region in all_regions:
        if region not in multipliers:
            print(
                f"⚠️ Warning: No scaling multiplier found for region '{region}'. Using default factor 1.0."
            )
        factor = multipliers.get(region, 1.0)
        mask = df["Region"] == region

        # Scale total assets
        df.loc[mask, "Total_Assets"] *= factor

        if recompute:
            # Recompute dependent fields
            df.loc[mask, "Debts"] = (
                df.loc[mask, "Total_Assets"] * df.loc[mask, "Debt_Ratio"]
            )
            df.loc[mask, "Net_Wealth"] = (
                df.loc[mask, "Total_Assets"] - df.loc[mask, "Debts"]
            )
            df.loc[mask, "Real_Assets"] = (
                df.loc[mask, "Total_Assets"] * df.loc[mask, "Real_Asset_Ratio"]
            )
            df.loc[mask, "Financial_Assets"] = (
                df.loc[mask, "Total_Assets"] * df.loc[mask, "Financial_Asset_Ratio"]
            )

            if "Business_Asset_Ratio" in df.columns:
                if df.loc[mask, "Business_Asset_Ratio"].isna().any():
                    print(
                        f"⚠️ Warning: NaN values in 'Business_Asset_Ratio' for region '{region}'"
                    )
                df.loc[mask, "Business_Assets"] = (
                    df.loc[mask, "Total_Assets"] * df.loc[mask, "Business_Asset_Ratio"]
                )
            else:
                print(
                    f"ℹ️ Info: 'Business_Asset_Ratio' not found. Setting Business_Assets to 0 for region '{region}'"
                )
                df.loc[mask, "Business_Assets"] = 0.0

    df.reset_index(drop=True, inplace=True)
    print("\n✅ Region multipliers applied. Preview of updated DataFrame:")
    print(df.head())
    return df


def scale_final_weights_by_taxpayer_counts(df, region_targets_quota):
    df = df.copy()
    df["Final_Weight"] = 0.0
    df["Region"] = df["Region"].str.strip().str.lower()

    for region, target_count in region_targets_quota.items():
        mask = (df["Region"] == region) & (df["Is_Taxpayer"] == 1)
        regional_declarants = df[mask]

        if regional_declarants.empty:
            print(f"⚠️ No taxpayers found in region '{region}', skipping.")
            continue

        total_weight = regional_declarants["Weight"].sum()
        if total_weight == 0:
            print(f"⚠️ Zero simulated taxpayer weight in region '{region}', skipping.")
            continue

        scaling_factor = target_count / total_weight
        df.loc[mask, "Final_Weight"] = df.loc[mask, "Weight"] * scaling_factor

    return df


def compare_to_observed(households_df: pd.DataFrame) -> pd.DataFrame:
    observed_df = pd.read_csv("Cleaned_Regional_Wealth_Tax_Data.csv")

    observed_clean = observed_df[
        observed_df["Variable"].str.strip().str.lower() == "resultado de la declaración"
    ].copy()
    observed_clean["Region"] = observed_clean["Region"].str.strip().str.lower()

    observed_clean["Total_Revenue"] = (
        observed_clean["Importe"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    observed_clean["Total_Revenue"] = pd.to_numeric(
        observed_clean["Total_Revenue"], errors="coerce"
    )
    observed_clean = observed_clean[["Region", "Total_Revenue"]]

    df = households_df.copy()
    df["Region"] = df["Region"].str.strip().str.lower()
    df = df[df["Region"] != "madrid"]

    df = df[df["Is_Taxpayer"] == True]
    df["Weighted_Wealth_Tax"] = df["Wealth_Tax"] * df["Final_Weight"]

    actual_region_revenue = df.groupby("Region", as_index=False)[
        "Weighted_Wealth_Tax"
    ].sum()
    actual_region_revenue.rename(
        columns={"Weighted_Wealth_Tax": "Simulated_Actual_Revenue"}, inplace=True
    )

    merged = pd.merge(actual_region_revenue, observed_clean, on="Region", how="left")
    merged["Gap_%"] = (
        100
        * (merged["Simulated_Actual_Revenue"] - merged["Total_Revenue"])
        / merged["Total_Revenue"]
    )
    merged["Gap_%"] = merged["Gap_%"].map("{:.2f}%".format)

    total_sim = merged["Simulated_Actual_Revenue"].sum()
    total_obs = merged["Total_Revenue"].sum()
    total_gap_pct = 100 * (total_sim - total_obs) / total_obs
    totals = pd.DataFrame(
        [
            {
                "Region": "TOTAL",
                "Simulated_Actual_Revenue": total_sim,
                "Total_Revenue": total_obs,
                "Gap_%": f"{total_gap_pct:.2f}%",
            }
        ]
    )

    result = pd.concat([merged, totals], ignore_index=True)
    print("\n📊 Revenue Comparison with Observed:")
    print(result.to_string(index=False))
    return result


def finalize_weights(df, normalization_target=8_984_492):
    df = df.copy()
    total_weight = df["Final_Weight"].sum()
    normalization_factor = normalization_target / total_weight
    df["Final_Weight"] *= normalization_factor
    return df


def main():
    try:
        # === CONFIGURATION ===
        POP_FILE = "Regional_Age_Bin_Population_Shares.csv"
        INCOME_FILE = "eff_incomedata.csv"
        STATS_FILE = "eff_data.xlsx"
        OUTPUT_FILE = "simulated_thesis.csv"
        TOTAL_HOUSEHOLDS = 100_000

        REGION_TARGETS_QUOTA = {
            "asturias": 3871,
            "catalonia": 84867,
            "extremadura": 1196,
            "galicia": 8027,
            "valencia": 26905,
        }

        # === LOAD DATA ===
        base_population = calculate_population_over_30(POP_FILE)
        group_stats = generate_eff_group_stats(load_eff_data(STATS_FILE))
        revenue_df, region_weights = load_population_and_revenue_data(POP_FILE)

        # === SYNTHETIC HOUSEHOLD GENERATION ===
        household_meta = generate_households_by_size(region_weights, TOTAL_HOUSEHOLDS)
        regions = household_meta["Region"].values
        household_sizes = household_meta["Household_Size"].values

        individuals, household_sizes_lookup = generate_and_adjust_households(
            group_stats,
            region_weights,
            INCOME_FILE,
            household_sizes=household_sizes,
            regions=regions,
        )

        assert "Household_Size" in individuals.columns, "❌ Household_Size missing."

        # === TOP WEALTH CALIBRATION ===
        individuals = calibrate_top_wealth_share_dual(individuals)

        # === INITIAL WEIGHTING ===
        individuals = reweight_to_match_percentile_shares(individuals)
        individuals["Weight"] = individuals["Adjusted_Weight"]
        individuals["Final_Weight"] = individuals["Weight"]

        # === NORMALIZE TO TOTAL POPULATION ===
        individuals = finalize_weights(
            individuals, normalization_target=base_population
        )

        # === TAX SIMULATION ===
        taxed_individuals = run_tax_simulation(individuals)
        taxed_individuals = apply_baseline_behavioral_erosion(taxed_individuals)

        # === POST-SIMULATION MERGES & ADJUSTMENTS ===
        taxed_individuals = taxed_individuals.merge(
            household_sizes_lookup, on="Original_ID", how="left"
        )
        if taxed_individuals["Household_Size"].isna().any():
            raise ValueError("Household_Size merge failed after tax simulation.")

        # === FINAL SCALING ===
        taxed_individuals = scale_final_weights_by_taxpayer_counts(
            taxed_individuals, REGION_TARGETS_QUOTA
        )
        taxed_individuals = recalculate_wealth_ranks(
            taxed_individuals, weight_col="Final_Weight"
        )
        taxed_individuals = finalize_weights(
            taxed_individuals, normalization_target=base_population
        )

        # === REPORT & OUTPUT ===
        generate_tax_diagnostics(taxed_individuals)
        taxed_individuals.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Simulation results saved to {OUTPUT_FILE}")

        compare_to_observed(taxed_individuals)

    except Exception as e:
        logging.exception(f"Pipeline execution failed: {e}")
        print("⚠️ Simulation failed due to error above.")


if __name__ == "__main__":
    main()
