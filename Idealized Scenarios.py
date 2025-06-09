import pandas as pd
import numpy as np
from synthetic_data import build_population
from data_loaders import load_eff_data, generate_eff_group_stats, load_population_data

np.random.seed(42)

# === Load and Generate Synthetic Population ===
eff = load_eff_data("eff_data.xlsx")
group_stats = generate_eff_group_stats(eff)
region_weights = load_population_data("Regional_Age_Bin_Population_Shares.csv")
individuals, _ = build_population(
    stats_by_group=group_stats,
    region_weights=region_weights,
    income_data_path="eff_data.xlsx",
    total_households=100_000,
    seed=42
)

households = individuals.copy()

# --- Liquidity-Adjusted Wealth Tax Regime with Behavior ---
households["Liquidity_Tax_Base"] = (
    0.25 * households["Real_Assets"] +
    1.00 * households["Financial_Assets"] +
    0.60 * households["Business_Assets"]
)

def improved_liquidity_erosion(row):
    if row["Wealth_Rank"] > 0.999:
        base_erosion = 0.12
    elif row["Wealth_Rank"] > 0.99:
        base_erosion = 0.08
    elif row["Wealth_Rank"] > 0.90:
        base_erosion = 0.05
    elif row["Wealth_Rank"] > 0.75:
        base_erosion = 0.02
    else:
        base_erosion = 0.005

    modifier = 1.0
    if row["Business_Asset_Ratio"] > 0.2:
        modifier += 0.05
    if row["Real_Asset_Ratio"] > 0.4:
        modifier += 0.03
    if row["Financial_Asset_Ratio"] > 0.4:
        modifier += 0.02
    if row["Income"] < 0.6 * row["Liquidity_Tax_Base"]:
        modifier += 0.02

    dropout = 0.001 if row["Wealth_Rank"] > 0.999 else 0.0
    erosion = min(base_erosion * modifier, 0.3)

    return pd.Series({
        "Liquidity_Erosion": erosion,
        "Liquidity_Dropout": dropout
    })

households = households.join(households.apply(improved_liquidity_erosion, axis=1))
households["Liquidity_Tax_Base_Eroded"] = households["Liquidity_Tax_Base"] * (1 - households["Liquidity_Erosion"])
households.loc[households["Liquidity_Dropout"] > 0, "Liquidity_Tax_Base_Eroded"] = 0

def calculate_liquidity_tax(base):
    brackets = [
        (400_000, 0.01),
        (700_000, 0.015),
        (1_000_000, 0.025),
        (5_000_000, 0.035),
        (10_000_000, 0.045),
        (float("inf"), 0.06)
    ]
    taxable_base = max(base - 400_000, 0)
    tax = 0
    last_limit = 0
    for limit, rate in brackets:
        if taxable_base > limit:
            tax += (limit - last_limit) * rate
            last_limit = limit
        else:
            tax += (taxable_base - last_limit) * rate
            break
    return tax

households["Liquidity_Wealth_Tax"] = households["Liquidity_Tax_Base_Eroded"].apply(calculate_liquidity_tax)
households["Cap"] = 0.60 * households["Income"]
over_cap = households["Liquidity_Wealth_Tax"] + households["PIT_Liability"] > households["Cap"]
households.loc[over_cap, "Liquidity_Wealth_Tax"] = np.maximum(
    0.2 * households.loc[over_cap, "Liquidity_Wealth_Tax"],
    households.loc[over_cap, "Cap"] - households.loc[over_cap, "PIT_Liability"]
)

households["Weighted_Liquidity_Tax"] = households["Liquidity_Wealth_Tax"] * households["Weight"]

# === Uniform National Wealth Tax Regime ===

households["Gross_Tax_Base_Uniform"] = households["Adj_Net_Wealth"] - households["Primary_Residence_Exempt"]

def get_uniform_exemption(region):
    return 700_000

households["Uniform_Exemption"] = households["Region"].apply(get_uniform_exemption)
households["Net_Tax_Base_Uniform"] = (households["Gross_Tax_Base_Uniform"] - households["Uniform_Exemption"]).clip(lower=0)

def calculate_uniform_tax(base):
    brackets = [
        (167129.45, 0.002), (334252.88, 0.003), (668499.75, 0.005),
        (1336999.51, 0.009), (2673999.01, 0.013), (5347998.03, 0.017),
        (10695996.06, 0.021), (float("inf"), 0.025)
    ]
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

households["Wealth_Tax_Uniform"] = households["Net_Tax_Base_Uniform"].apply(calculate_uniform_tax)
households["Cap_Uniform"] = 0.60 * households["Income"]
over_limit = households["Wealth_Tax_Uniform"] + households["PIT_Liability"] > households["Cap_Uniform"]
households.loc[over_limit, "Wealth_Tax_Uniform"] = np.maximum(
    0.2 * households.loc[over_limit, "Wealth_Tax_Uniform"],
    households.loc[over_limit, "Cap_Uniform"] - households.loc[over_limit, "PIT_Liability"]
)

households["Taxable_Wealth_Uniform"] = households["Net_Tax_Base_Uniform"]

def uniform_tax_erosion(row):
    if row["Wealth_Rank"] > 0.999:
        base_erosion = 0.08
    elif row["Wealth_Rank"] > 0.99:
        base_erosion = 0.05
    elif row["Wealth_Rank"] > 0.90:
        base_erosion = 0.03
    elif row["Wealth_Rank"] > 0.75:
        base_erosion = 0.01
    else:
        base_erosion = 0.0

    modifier = 1.0
    if row["Business_Asset_Ratio"] > 0.2:
        modifier += 0.03
    if row["Real_Asset_Ratio"] > 0.4:
        modifier += 0.01
    if row["Income"] < 0.6 * row["Adj_Net_Wealth"]:
        modifier += 0.01

    dropout = 0.0005 if row["Wealth_Rank"] > 0.999 else 0.0
    erosion = min(base_erosion * modifier, 0.2)

    return pd.Series({
        "Uniform_Erosion": erosion,
        "Uniform_Dropout": dropout
    })

households = households.join(households.apply(uniform_tax_erosion, axis=1))
households["Taxable_Wealth_Uniform_Eroded"] = households["Taxable_Wealth_Uniform"] * (1 - households["Uniform_Erosion"])
households.loc[households["Uniform_Dropout"] > 0, "Wealth_Tax_Uniform"] = 0
households["Wealth_Tax_Uniform"] = households["Taxable_Wealth_Uniform_Eroded"].apply(calculate_uniform_tax)

# 6. Weight and summarize
households["Weighted_Wealth_Tax_Uniform"] = households["Wealth_Tax_Uniform"] * households["Declarant_Weight"]
uniform_revenue = households.groupby("Region")["Weighted_Wealth_Tax_Uniform"].sum().reset_index()
uniform_revenue.columns = ["Region", "Uniform_Wealth_Tax_Revenue"]
# === might delete later ==

# Recreate top 1% group after all tax columns exist
households_sorted = households.sort_values("Adj_Net_Wealth", ascending=False)
households_sorted["Cumulative_Weight"] = households_sorted["Weight"].cumsum()
total_weight = households_sorted["Weight"].sum()
top_1 = households_sorted[households_sorted["Cumulative_Weight"] <= 0.01 * total_weight]

# === 📊 Final Evaluation: Revenue, Redistribution, and Tax Burden ===
import numpy as np
# === Gini Coefficient Function (Correct, Weighted) ===
def gini(array, weights=None):
    """
    Compute Gini coefficient with optional weights.
    Parameters:
        array (array-like): Income or wealth values
        weights (array-like): Sample weights (same length as array)
    Returns:
        Gini coefficient (float) between 0 and 1
    """
    array = np.asarray(array)
    if weights is None:
        weights = np.ones_like(array)
    else:
        weights = np.asarray(weights)

    sorted_indices = np.argsort(array)
    x_sorted = array[sorted_indices]
    w_sorted = weights[sorted_indices]

    cum_weights = np.cumsum(w_sorted)
    cum_income = np.cumsum(x_sorted * w_sorted)

    cum_weights_rel = cum_weights / cum_weights[-1]
    cum_income_rel = cum_income / cum_income[-1]

    b = np.trapezoid(cum_income_rel, cum_weights_rel)
    g = 1 - 2 * b
    return g

# === Gini Calculations Per Tax Regime ===
gini_pre = gini(households["Adj_Net_Wealth"], households["Weight"])
gini_post_baseline = gini(households["Adj_Net_Wealth"] - households["Wealth_Tax"], households["Weight"])
gini_post_liquidity = gini(households["Adj_Net_Wealth"] - households["Liquidity_Wealth_Tax"], households["Weight"])
gini_post_uniform = gini(households["Adj_Net_Wealth"] - households["Wealth_Tax_Uniform"], households["Weight"])

print("\nGini Coefficients by Tax Regime")
print(f"Before Tax:       {gini_pre:.4f}")
print(f"After Baseline:   {gini_post_baseline:.4f}")
print(f"After Liquidity:  {gini_post_liquidity:.4f}")
print(f"After Uniform:    {gini_post_uniform:.4f}")

# --- 3. Revenue Metrics ---
potential_revenue = households["No_Exemption_Tax"].multiply(households["Declarant_Weight"]).sum()
legal_revenue = households["Wealth_Tax_Baseline"].multiply(households["Declarant_Weight"]).sum()
actual_revenue = households["Wealth_Tax"].multiply(households["Declarant_Weight"]).sum()
# === Liquidity-Adjusted Regime Revenue ===
# Define potential revenue (no erosion, no dropout)
households["Liquidity_Tax_Base_No_Erosion"] = households["Liquidity_Tax_Base"]
households["Liquidity_Tax_No_Erosion"] = households["Liquidity_Tax_Base_No_Erosion"].apply(calculate_liquidity_tax)
households["Weighted_Liquidity_Tax_No_Erosion"] = households["Liquidity_Tax_No_Erosion"] * households["Declarant_Weight"]
liquidity_potential_revenue = households["Weighted_Liquidity_Tax_No_Erosion"].sum()
liquidity_actual_revenue = households["Weighted_Liquidity_Tax"].sum()
# Legal revenue for liquidity = no erosion, but with exemptions and cap
# In your case, liquidity base already excludes exemptions, so legal == potential
liquidity_legal_revenue = liquidity_potential_revenue

# === Uniform National Regime Revenue ===
households["Uniform_Tax_No_Erosion"] = households["Taxable_Wealth_Uniform"].apply(calculate_uniform_tax)
households["Weighted_Uniform_Tax_No_Erosion"] = households["Uniform_Tax_No_Erosion"] * households["Declarant_Weight"]
uniform_potential_revenue = households["Weighted_Uniform_Tax_No_Erosion"].sum()
uniform_actual_revenue = households["Weighted_Wealth_Tax_Uniform"].sum()
uniform_legal_revenue = uniform_potential_revenue  # Same logic as liquidity

# Revenue summary as DataFrame
revenue_summary = pd.DataFrame({
    "Metric": [
        "Potential Revenue",
        "Legal Revenue",
        "Actual Revenue"
    ],
    "Baseline": [
        potential_revenue,
        legal_revenue,
        actual_revenue
    ],
    "Liquidity-Adjusted": [
        liquidity_potential_revenue,
        liquidity_legal_revenue,
        liquidity_actual_revenue
    ],
    "Uniform": [
        uniform_potential_revenue,
        uniform_legal_revenue,
        uniform_actual_revenue
    ]
})
print("\n💰 Revenue Summary:")
print(revenue_summary)


# Calculate share of tax paid under each regime
# Sort by wealth and calculate cumulative weight
households_sorted = households.sort_values("Adj_Net_Wealth", ascending=False)
households_sorted["Cumulative_Weight"] = households_sorted["Final_Weight"].cumsum()
total_weight = households_sorted["Final_Weight"].sum()

# Top 10% and Top 1% cutoffs
top10 = households_sorted[households_sorted["Cumulative_Weight"] <= 0.10 * total_weight]
top1 = households_sorted[households_sorted["Cumulative_Weight"] <= 0.01 * total_weight]

def tax_share(tax_column, top_df, total_col):
    return 100 * (top_df[tax_column] * top_df["Final_Weight"]).sum() / \
                 (households[tax_column] * households["Final_Weight"]).sum()

print("\n💰 Tax Concentration by Regime:")
print(f"Top 10% share - Baseline:   {tax_share('Wealth_Tax', top10, 'Final_Weight'):.2f}%")
print(f"Top 10% share - Liquidity:  {tax_share('Liquidity_Wealth_Tax', top10, 'Final_Weight'):.2f}%")
print(f"Top 10% share - Uniform:    {tax_share('Wealth_Tax_Uniform', top10, 'Final_Weight'):.2f}%")

print(f"Top 1% share  - Baseline:   {tax_share('Wealth_Tax', top1, 'Final_Weight'):.2f}%")
print(f"Top 1% share  - Liquidity:  {tax_share('Liquidity_Wealth_Tax', top1, 'Final_Weight'):.2f}%")
print(f"Top 1% share  - Uniform:    {tax_share('Wealth_Tax_Uniform', top1, 'Final_Weight'):.2f}%")

# === Diagnostic Block ===
print("\n\U0001F527 Model Diagnostics Summary")
print("Total population (Final Weight):", households["Final_Weight"].sum())
print("Total declarants (Declarant Weight):", households["Declarant_Weight"].sum())
print("Top 1% Wealth Share:", ((households[households["Wealth_Rank"] > 0.99]["Adj_Net_Wealth"] * households[households["Wealth_Rank"] > 0.99]["Final_Weight"]).sum()) / (households["Adj_Net_Wealth"] * households["Final_Weight"]).sum())
print("Top 10% Wealth Share:", ((households[households["Wealth_Rank"] > 0.90]["Adj_Net_Wealth"] * households[households["Wealth_Rank"] > 0.90]["Final_Weight"]).sum()) / (households["Adj_Net_Wealth"] * households["Final_Weight"]).sum())
print("Mean Erosion (Baseline):", households["Erosion_Factor"].mean())
print("Dropout Rate (Baseline):", households["Dropout"].mean())

# === Internal Validation Block ===
print("\n\U0001F9EA Internal Validation Checks")

# 1. Check weighted tax totals
baseline_total_tax = (households["Wealth_Tax"] * households["Final_Weight"]).sum()
liquidity_total_tax = (households["Liquidity_Wealth_Tax"] * households["Final_Weight"]).sum()
uniform_total_tax = (households["Wealth_Tax_Uniform"] * households["Final_Weight"]).sum()

print(f"Baseline total simulated revenue: €{baseline_total_tax:,.2f}")
print(f"Liquidity-adjusted simulated revenue: €{liquidity_total_tax:,.2f}")
print(f"Uniform simulated revenue: €{uniform_total_tax:,.2f}")

# 2. Check for negative tax liabilities or wealth
assert (households[["Wealth_Tax", "Liquidity_Wealth_Tax", "Wealth_Tax_Uniform"]] >= 0).all().all(), "Negative tax liability detected."
assert (households["Adj_Net_Wealth"] >= 0).all(), "Negative wealth detected."
print("No negative tax liabilities or wealth values. ✓")

# 3. Sensitivity test: Adjust exemption threshold and measure revenue change
exemption_adjustment = 100_000
households["Adj_Net_Wealth_Adjusted"] = (households["Adj_Net_Wealth"] - exemption_adjustment).clip(lower=0)

test_tax = households["Adj_Net_Wealth_Adjusted"].apply(lambda x: x * 0.01)  # 1% flat tax
adjusted_total_revenue = (test_tax * households["Final_Weight"]).sum()
print(f"\nSensitivity Check — Revenue under 1% flat tax post adjustment: €{adjusted_total_revenue:,.2f}")

print("Top 1% Raw Net Wealth Share:", (
    (households[households["Wealth_Rank"] > 0.99]["Net_Wealth"] *
     households[households["Wealth_Rank"] > 0.99]["Final_Weight"]).sum()
) / (households["Net_Wealth"] * households["Final_Weight"]).sum())

print("Top 10% Raw Net Wealth Share:", (
    (households[households["Wealth_Rank"] > 0.90]["Net_Wealth"] *
     households[households["Wealth_Rank"] > 0.90]["Final_Weight"]).sum()
) / (households["Net_Wealth"] * households["Final_Weight"]).sum())

