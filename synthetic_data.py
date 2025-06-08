"""
generators.py

Synthetic‐population generators for the wealth–tax simulator.

This module provides:

  • generate_households_by_size:
      Allocate total_households across regions and randomly assign
      household sizes (1–5 persons).

  • generate_and_adjust_households:
      Take those households, assign them wealth‐percentile categories,
      merge in group‐level asset and debt stats, add realistic noise,
      compute derived asset/debt/income fields, and split any very‐rich
      households into multiple tax units.

  • expand_households_to_individuals:
      Turn each household (or tax unit) into person‐level rows, scaling
      all monetary and weight fields by the household size.

  • build_population:
      A one‐stop wrapper that runs the above steps in sequence to yield
      a person‐level DataFrame plus a small lookup of original household sizes.
"""

import numpy as np
import pandas as pd

from constants import REGION_COLUMN_NAME
from data_loaders import load_eff_income_map


def generate_households_by_size(
    region_weights: pd.DataFrame,
    total_households: int,
    size_probs=None,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    For each region, draw Num_Households = round(pop_share*total_households),
    then sample a household size for each.

    Returns
    -------
        DataFrame
        One row per household, columns ['Region', 'Household_Size'].
    """
    if size_probs is None:
        size_probs = [0.25, 0.35, 0.20, 0.15, 0.05]
    people_per_household = len(size_probs)
    rng = np.random.default_rng(rng_seed)
    df = region_weights.copy()
    df["Num_Households"] = (df["Population"] * total_households).round().astype(int)
    diff = total_households - df["Num_Households"].sum()
    if diff:
        df.loc[df["Num_Households"].idxmax(), "Num_Households"] += diff

    rows = []
    for region, group in df.iterrows():
        sizes = rng.choice(
            range(1, people_per_household), size=group.Num_Households, p=size_probs
        )
        rows += [(region, sz) for sz in sizes]

    return pd.DataFrame(rows, columns=[REGION_COLUMN_NAME, "Household_Size"])


def simulate_tax_unit_population(
    group_statistics: pd.DataFrame,
    population_shares: pd.DataFrame,
    income_lookup: dict,
    total_households: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a person‐level synthetic population with assets, income and tax‐unit structure.

    1. Draw households by region & size.
    2. Assign each household a wealth‐percentile rank & category.
    3. Merge in per-category asset/debt stats.
    4. Inject realistic noise into Total_Assets.
    5. Compute derived fields: Debts, Net_Wealth, Real/Financial/Business assets.
    6. Load mean incomes by wealth category and draw each household’s Income.
    7. Split high‐wealth households into multiple tax units & expand to persons.

    Returns
    -------
    persons_df : DataFrame
        One row per individual, with columns:
        ['Original_ID','Tax_Unit_ID','Region','Wealth_Rank','Category',
         'Total_Assets','Debts','Net_Wealth','Real_Assets',
         'Financial_Assets','Business_Assets','Income','Weight',…]
    household_lookup : DataFrame
        Two columns ['Original_ID','Household_Size'] to recover household sizes.
    """
    household_distribution = generate_households_by_size(
        population_shares,
        total_households=total_households,
        rng_seed=seed,
    )
    regions = household_distribution[REGION_COLUMN_NAME].values
    sizes = household_distribution["Household_Size"].values

    # 2) Assign wealth ranks & categories
    n = len(regions)
    ranks = np.linspace(0.0001, 1.0, n)
    categories = pd.cut(
        ranks,
        bins=[0, 0.3, 0.55, 0.75, 0.9, 1.0],
        labels=["<25", "25–50", "50–75", "75–90", "90–100"],
        include_lowest=True,
    ).astype(str)

    households = pd.DataFrame(
        {
            "Region": regions,
            "Household_Size": sizes,
            "Wealth_Rank": ranks,
            "Category": categories,
        }
    )

    households = households.merge(group_statistics, on="Category", how="left").dropna(
        subset=["Total_Assets"]
    )

    np.random.seed(seed)
    households["Total_Assets"] *= np.random.normal(1.0, 0.05, len(households))
    mid_rank = households["Wealth_Rank"].between(0.3, 0.9)
    households.loc[mid_rank, "Total_Assets"] *= np.random.normal(
        1.0, 0.15, mid_rank.sum()
    )
    low_rank = households["Wealth_Rank"] <= 0.5
    households.loc[low_rank, "Total_Assets"] *= np.random.normal(
        1.2, 0.25, low_rank.sum()
    )

    households["Debts"] = households["Total_Assets"] * households["Debt_Ratio"]
    households["Net_Wealth"] = (households["Total_Assets"] - households["Debts"]).clip(
        lower=7000
    )
    households["Real_Assets"] = (
        households["Total_Assets"] * households["Real_Asset_Ratio"]
    )
    households["Financial_Assets"] = (
        households["Total_Assets"] * households["Financial_Asset_Ratio"]
    )
    households["Business_Assets"] = households.get("Business_Assets", 0.0).fillna(0.0)
    mean_incomes = (
        households["Category"].map(lambda c: income_lookup.get(c, 0.0)).clip(lower=1.0)
    )
    rng = np.random.default_rng(seed)
    households["Income"] = rng.normal(mean_incomes, 0.05 * mean_incomes)
    persons_df = expand_households_to_individuals(
        households, base_threshold=1_000_000, rng_seed=seed
    )
    household_lookup = households[["Original_ID", "Household_Size"]].drop_duplicates()

    return persons_df, household_lookup


def expand_households_to_individuals(
    df: pd.DataFrame,
    base_threshold: float = 1_000_000,
    max_split: int = 5,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Split each household-row into 1–max_split tax units (via a Dirichlet draw),
    then expand each unit into individual rows based on Household_Size.

    Returns a person-level DataFrame with weights and all monetary fields.
    """
    rng = np.random.default_rng(rng_seed)
    df = df.copy()
    df["Original_ID"] = df.index
    df["Weight"] = df.get("Weight", 1.0)
    df["Household_Size"] = df.get("Household_Size", 1)

    monetary_cols = [
        "Total_Assets",
        "Debts",
        "Net_Wealth",
        "Real_Assets",
        "Financial_Assets",
        "Business_Assets",
        "Income",
    ]
    unit_rows = []

    for _, row in df.iterrows():
        wealth = row["Net_Wealth"]
        if wealth > base_threshold:
            lam = (wealth - base_threshold) / 5e5
            k = int(np.clip(rng.poisson(lam), 1, max_split))
        else:
            k = 1
        shares = rng.dirichlet(np.linspace(2, 1, k))
        for i, frac in enumerate(shares, start=1):
            unit = row.copy()
            for col in monetary_cols:
                unit[col] *= frac
            unit["Weight"] *= frac
            unit["Tax_Unit_ID"] = i
            unit_rows.append(unit)

    units_df = pd.DataFrame(unit_rows)
    indiv_rows = []
    for _, u in units_df.iterrows():
        for _ in range(int(u["Household_Size"])):
            indiv = u.copy()
            indiv["Weight"] /= u["Household_Size"]
            indiv_rows.append(indiv)

    return pd.DataFrame(indiv_rows)


def build_population(
    stats_by_group: pd.DataFrame,
    region_weights: pd.DataFrame,
    income_data_path: str,
    total_households: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top‐level: from group_stats and region_weights, produce:
      - individuals_df: person-level rows with full asset/income/tax-unit info
      - household_lookup: mapping Original_ID → Household_Size
    """
    income_lookup = load_eff_income_map(income_data_path)

    individuals, lookup = simulate_tax_unit_population(
        stats_by_group, region_weights, income_lookup, total_households, seed=seed
    )
    return individuals, lookup
