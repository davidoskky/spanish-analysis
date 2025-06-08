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


def generate_and_adjust_households(
    stats_by_group: pd.DataFrame,
    region_weights: pd.DataFrame,
    income_data_path: str,
    household_sizes: np.ndarray = None,
    regions: np.ndarray = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create household‐level tax units with simulated assets, debts, and income.

    Steps:
      1) Assign a continuous wealth_rank (0.0001–1.0) and percentile category.
      2) Merge in group_stats (totals, ratios).
      3) Add noise to Total_Assets by rank buckets.
      4) Compute Debts, Net_Wealth, Real/Financial assets.
      5) Read EFF income-per-centile and draw random incomes.
      6) Split high‐wealth households into multiple tax units.
      7) Return both the unit‐level DataFrame and a household_size lookup.

    Parameters
    ----------
    stats_by_group : DataFrame
        Output of generate_eff_group_stats(), indexed by Category.
    region_weights : DataFrame
        Not used internally here, but kept for API symmetry.
    income_data_path : str
        CSV with EFF income‐percentile rows.
    household_sizes : array-like, optional
        Pre-drawn household sizes.
    regions : array-like, optional
        Pre-drawn household regions.

    Returns
    -------
    tuple
      - households_df : DataFrame
          One row per (possibly split) tax unit, columns include
          Original_ID, assets, debts, income, Weight, Region, etc.
      - lookup_df : DataFrame
          Two columns ['Original_ID','Household_Size'] for merging later.
    """
    # If sizes/regions not provided, re‐draw them
    if regions is None or household_sizes is None:
        tmp = generate_households_by_size(
            region_weights, int(region_weights["Population"].sum())
        )
        regions = tmp[REGION_COLUMN_NAME].values
        household_sizes = tmp["Household_Size"].values

    n = len(regions)
    ranks = np.linspace(0.0001, 1.0, n)
    categories = pd.cut(
        ranks,
        bins=[0, 0.3, 0.55, 0.75, 0.9, 1.0],
        labels=[
            "under 25",
            "between 25 and 50",
            "between 50 and 75",
            "between 75 and 90",
            "between 90 and 100",
        ],
        include_lowest=True,
    ).astype(str)

    hh = pd.DataFrame(
        {
            REGION_COLUMN_NAME: regions,
            "Wealth_Rank": ranks,
            "Category": categories,
            "Household_Size": household_sizes,
        }
    )

    # Merge asset statistics
    hh = hh.merge(stats_by_group, on="Category", how="left").dropna(
        subset=["Total_Assets"]
    )

    # Add noise to Total_Assets
    np.random.seed(42)
    hh["Total_Assets"] *= np.random.normal(1.0, 0.05, len(hh))
    mid = hh["Wealth_Rank"].between(0.3, 0.9)
    hh.loc[mid, "Total_Assets"] *= np.random.normal(1.0, 0.15, mid.sum())
    low = hh["Wealth_Rank"] <= 0.5
    hh.loc[low, "Total_Assets"] *= np.random.normal(1.2, 0.25, low.sum())

    # Compute derived fields
    hh["Debts"] = hh["Total_Assets"] * hh["Debt_Ratio"]
    hh["Net_Wealth"] = (hh["Total_Assets"] - hh["Debts"]).clip(lower=7000)
    hh["Real_Assets"] = hh["Total_Assets"] * hh["Real_Asset_Ratio"]
    hh["Financial_Assets"] = hh["Total_Assets"] * hh["Financial_Asset_Ratio"]
    hh["Business_Assets"] = hh.get("Business_Assets", 0.0).fillna(0.0)

    # Assign random incomes from EFF income-by-percentile
    inc = pd.read_csv(income_data_path)
    mask = (
        inc["breakdown"].eq("NET WEALTH PERCENTILE")
        & inc["estadistico"].str.upper().eq("MEAN")
        & (inc["wave"] == 2022)
    )
    mp = dict(
        zip(
            inc.loc[mask, "category"].str.strip().str.lower(),
            inc.loc[mask, "value"] * 1000,
        )
    )
    hh["Income"] = hh["Category"].map(lambda c: max(1, mp.get(c, 0)))
    hh["Income"] = hh["Income"].map(lambda m: np.random.normal(m, 0.05 * m))

    # Split & expand to individuals
    individuals = expand_households_to_individuals(hh, base_threshold=1_000_000)
    lookup = hh[["Original_ID", "Household_Size"]].drop_duplicates()

    return individuals, lookup


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
    total_households: int = 100_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top‐level: from group_stats and region_weights, produce:
      - individuals_df: person-level rows with full asset/income/tax-unit info
      - household_lookup: mapping Original_ID → Household_Size
    """
    hh_meta = generate_households_by_size(
        region_weights, total_households, rng_seed=seed
    )
    individuals, lookup = generate_and_adjust_households(
        stats_by_group,
        region_weights,
        income_data_path,
        household_sizes=hh_meta["Household_Size"].values,
        regions=hh_meta[REGION_COLUMN_NAME].values,
    )
    return individuals, lookup
