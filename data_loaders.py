"""
load.py  – I/O helpers for the wealth–tax simulator
──────────────────────────────────────────────────
Functions
---------
load_eff_data(path, year=2022)                -> DataFrame (row-level EFF)
process_eff_assets_income(eff_df)             -> (pivot_df, group_stats_df)
load_population_and_revenue_data(pop_path,
                                 revenue_path) -> (revenue_df, region_weights_df)
"""

from pathlib import Path
import unicodedata
import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────
EFF_COLUMNS = [
    "Concept",
    "Element",
    "Statistic",
    "Breakdown",
    "Category",
    "Measure",
    "Wave",
    "Value",
]

REAL_ASSETS = [
    "MAIN RESIDENCE",
    "OTHER REAL ESTATE PROPERTIES",
    "CARS AND OTHER VEHICLES",
    "OTHER DURABLE GOODS",
]
FIN_ASSETS = [
    "LISTED SHARES",
    "INVESTMENT FUNDS",
    "FIXED-INCOME SECURITIES",
    "PENSION SCHEMES AND UNIT-LINKED OR MIXED LIFE INSURANCE",
    "ACCOUNTS AND DEPOSITS USABLE FOR PAYMENTS",
    "ACCOUNTS NON USABLE FOR PAYMENTS AND HOUSE-PURCHASE SAVING ACCOUNTS",
    "OTHER FINANCIAL ASSETS",
    "UNLISTED SHARES AND OTHER EQUITY",
]
DEBTS = ["TOTAL DEBT"]

# ────────────────────────────────────────────────────────────
# Full province ➜ autonomous-region mapping   (ASCII-folded keys)
# ────────────────────────────────────────────────────────────
PROVINCE_TO_REGION: dict[str, str] = {
    # ANDALUSIA
    "almeria": "andalusia",
    "cadiz": "andalusia",
    "cordoba": "andalusia",
    "granada": "andalusia",
    "huelva": "andalusia",
    "jaen": "andalusia",
    "malaga": "andalusia",
    "sevilla": "andalusia",
    # ARAGON
    "huesca": "aragon",
    "zaragoza": "aragon",
    "teruel": "aragon",
    # ASTURIAS (single-province region)
    "asturias": "asturias",
    "asturias, principado de": "asturias",
    # BALEARS
    "illes balears": "balearic islands",
    "baleares": "balearic islands",
    "balears": "balearic islands",
    # CANARY ISLANDS
    "santa cruz de tenerife": "canary islands",
    "las palmas": "canary islands",
    # CANTABRIA
    "cantabria": "cantabria",
    # CASTILLA–LA MANCHA
    "albacete": "castilla-la mancha",
    "ciudad real": "castilla-la mancha",
    "cuenca": "castilla-la mancha",
    "guadalajara": "castilla-la mancha",
    "toledo": "castilla-la mancha",
    # CASTILLA Y LEON
    "avila": "castilla y leon",
    "burgos": "castilla y leon",
    "leon": "castilla y leon",
    "palencia": "castilla y leon",
    "salamanca": "castilla y leon",
    "segovia": "castilla y leon",
    "soria": "castilla y leon",
    "valladolid": "castilla y leon",
    "zamora": "castilla y leon",
    # CATALONIA
    "barcelona": "catalonia",
    "girona": "catalonia",
    "lleida": "catalonia",
    "tarragona": "catalonia",
    "cataluna": "catalonia",
    # VALENCIAN COMMUNITY
    "valencia/valencia": "valencia",
    "valencia": "valencia",
    "alicante/alacant": "valencia",
    "alicante": "valencia",
    "castellon/castello": "valencia",
    "castellon": "valencia",
    "comunitat valenciana": "valencia",
    # EXTREMADURA
    "caceres": "extremadura",
    "badajoz": "extremadura",
    # GALICIA
    "coruna, a": "galicia",
    "a coruna": "galicia",
    "lugo": "galicia",
    "ourense": "galicia",
    "pontevedra": "galicia",
    # LA RIOJA
    "rioja, la": "la rioja",
    "la rioja": "la rioja",
    # MADRID
    "madrid": "madrid",
    "madrid, comunidad de": "madrid",
    # MURCIA
    "murcia": "murcia",
    "murcia, region de": "murcia",
    # NAVARRE
    "navarra": "navarre",
    "navarra, comunidad foral de": "navarre",
    # BASQUE COUNTRY
    "alava": "basque country",
    "araba/alava": "basque country",
    "bizkaia": "basque country",
    "vizcaya": "basque country",
    "gipuzkoa": "basque country",
    "guipuzcoa": "basque country",
    # CEUTA & MELILLA (autonomous cities)
    "ceuta": "ceuta",
    "melilla": "melilla",
    "andalucia": "andalusia",
    "aragon": "aragon",
    "balears, illes": "balearic islands",
    "canarias": "canary islands",
    "castilla - la mancha": "castilla-la mancha",
    "castilla y leon": "castilla y leon",
    "extremadura": "extremadura",
    "galicia": "galicia",
    "pais vasco": "basque country",
    "palmas, las": "canary islands",  # same as “las palmas”
    # CSV sometimes carries a country summary line – we’ll drop it later,
    # but mapping it to None keeps the ‘unmapped’ list empty.
    "national total": None,
}


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """ASCII-fold, strip, lowercase – used for every free-text label."""
    return (
        unicodedata.normalize("NFKD", str(s).strip())
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )


# ────────────────────────────────────────────────────────────
# 1.  EFF LOADER  ░░ load_eff_data ░░
# ────────────────────────────────────────────────────────────
def load_eff_data(
    path: str | Path = "eff_data.xlsx",
    sheet: str = "Datos",
    year: int = 2022,
) -> pd.DataFrame:
    """
    Read the ECB EFF file, keep the chosen wave & ‘mean’ rows,
    scale euro values ×1000 (file is in thousands).

    Returns the row-level filtered DataFrame – you’ll aggregate later.
    """
    df = (
        pd.read_excel(path, sheet_name=sheet, skiprows=10, engine="openpyxl")
        .iloc[:, : len(EFF_COLUMNS)]
        .set_axis(EFF_COLUMNS, axis=1)
    )

    df["Wave"] = pd.to_numeric(df["Wave"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce") * 1000
    df["Category"] = df["Category"].astype(str).str.strip().str.replace("\u2013", "-")

    mask = (
        (df["Wave"] == year)
        & (df["Breakdown"] == "NET WEALTH PERCENTILE")
        & (df["Statistic"].str.upper() == "MEAN")
        & df["Value"].notna()
    )
    return df.loc[mask].copy()


def _validate_eff_elements(eff: pd.DataFrame) -> None:
    required = set(REAL_ASSETS + FIN_ASSETS + DEBTS)
    missing = required - set(eff["Element"].unique())
    if missing:
        raise ValueError(f"EFF data missing elements: {missing}")


def _pivot_eff_assets(eff: pd.DataFrame) -> pd.DataFrame:
    return eff.pivot_table(
        index="Category",
        columns="Element",
        values="Value",
        aggfunc="mean",
    ).fillna(0)


def _compute_totals(pv: pd.DataFrame) -> pd.DataFrame:
    pv["Real_Assets"] = pv[REAL_ASSETS].sum(axis=1)
    pv["Financial_Assets"] = pv[FIN_ASSETS].sum(axis=1)
    pv["Total_Assets"] = pv["Real_Assets"] + pv["Financial_Assets"]
    pv["Debts"] = pv[DEBTS].sum(axis=1)
    pv["Net_Wealth"] = pv["Total_Assets"] - pv["Debts"]
    return pv


def _compute_ratios(pv: pd.DataFrame) -> pd.DataFrame:
    total = pv["Total_Assets"].replace(0, np.nan)
    pv["Real_Asset_Ratio"] = pv["Real_Assets"] / total
    pv["Financial_Asset_Ratio"] = pv["Financial_Assets"] / total
    pv["Debt_Ratio"] = pv["Debts"] / total
    return pv


def load_population_and_revenue_data(
    pop_path: str | Path,
    revenue_path: str | Path = "Cleaned_Regional_Wealth_Tax_Data.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (AEAT revenue, region-level population *share*)."""
    # —— Revenue ——————————————————————————
    rev = (
        pd.read_csv(revenue_path)
        .query(
            "Variable.str.strip().str.lower() == 'resultado de la declaración'",
            engine="python",
        )
        .rename(columns={"Importe": "Total_Revenue"})
        .assign(Region=lambda d: d["Region"].str.strip().str.lower())
        .loc[:, ["Region", "Total_Revenue"]]
    )

    # —— Population ——————————————————————
    pop = pd.read_csv(pop_path)
    pop["Region"] = pop["Region"].str.replace(r"^\d+\s+", "", regex=True).map(_norm)
    pop["Autonomous_Region"] = pop["Region"].map(PROVINCE_TO_REGION)

    dropped = pop["Autonomous_Region"].isna().sum()
    unmapped_mask = pop["Autonomous_Region"].isna()
    if unmapped_mask.any():
        missing_labels = pop.loc[unmapped_mask, "Region"].value_counts().sort_index()
        print(f"⚠️  {dropped} province rows not mapped – ignored.")
        print("   Unmapped province names (with row counts):")
        for prov, count in missing_labels.items():
            print(f"     • {prov:25s} {count:>5}")

    pop = pop.dropna(subset=["Autonomous_Region"])
    weights = (
        pop.groupby("Autonomous_Region", as_index=False)["Population"]
        .sum()
        .rename(columns={"Autonomous_Region": "Region"})
        .assign(Population=lambda d: d["Population"] / d["Population"].sum())
    )

    return rev, weights


def _extract_income(eff: pd.DataFrame) -> pd.DataFrame:
    """
      • income_df   – columns [Category, Mean_Income]
      • business_df – columns [Category, Business_Assets]

    Assumes eff["Value"] is already in full‐euro units.
    """
    # Mean income by category
    income_df = (
        eff[eff["Concept"].str.contains("income", case=False)]
        .loc[:, ["Category", "Value"]]
        .rename(columns={"Value": "Mean_Income"})
    )
    return income_df


def _extract_business(eff: pd.DataFrame) -> pd.DataFrame:
    # Business assets by category
    business_df = (
        eff[eff["Element"] == "BUSINESSES RELATED TO SELF-EMPLOYMENT"]
        .loc[:, ["Category", "Value"]]
        .rename(columns={"Value": "Business_Assets"})
    )

    return business_df


def _merge_income_business(
    pivot: pd.DataFrame,
    income_df: pd.DataFrame,
    business_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left‐join the small income_df and business_df back onto the pivot table.

    Input:
      pivot       – DataFrame with index Category, plus all asset columns
      income_df   – [Category, Mean_Income]
      business_df – [Category, Business_Assets]

    Output:
      DataFrame reset to a 0..n index, with new columns:
        • Mean_Income
        • Business_Assets (filled with 0 where missing)
    """
    merged = (
        pivot.reset_index()  # bring Category back as a column
        .merge(income_df, on="Category", how="left")
        .merge(business_df, on="Category", how="left")
        .fillna({"Business_Assets": 0.0})
    )
    return merged


def _compute_business_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Business_Asset_Ratio = Business_Assets / Total_Assets
    Safely handles Total_Assets == 0 by turning those ratios into NaN.
    """
    denom = df["Total_Assets"].replace(0, np.nan)
    df["Business_Asset_Ratio"] = df["Business_Assets"] / denom
    return df


def generate_eff_pivot_df(eff: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full “pivot_df” of asset breakdown by Category.

    Output schema (columns):
      • Category                 (str)  – wealth percentile label
      • <every EFF Element>            – one column per Element, e.g.
        – MAIN RESIDENCE
        – OTHER REAL ESTATE PROPERTIES
        – … etc …
      • Real_Assets              (float)
      • Financial_Assets         (float)
      • Total_Assets             (float)
      • Debts                    (float)
      • Net_Wealth               (float)
      • Real_Asset_Ratio         (float)
      • Financial_Asset_Ratio    (float)
      • Debt_Ratio               (float)
      • Mean_Income              (float)
      • Business_Assets          (float)
      • Business_Asset_Ratio     (float)
    """
    _validate_eff_elements(eff)
    pivot = _pivot_eff_assets(eff)
    pivot = _compute_totals(pivot)
    pivot = _compute_ratios(pivot)
    income_df = _extract_income(eff)
    business_df = _extract_business(eff)
    pivot = _merge_income_business(pivot, income_df, business_df)
    pivot = _compute_business_ratio(pivot)
    pivot["Category"] = pivot["Category"].map(_norm)

    return pivot


def generate_eff_group_stats(eff: pd.DataFrame) -> pd.DataFrame:
    """
    Build the “group_stats” table for the synthetic-population generator.

    This is just a subset of generate_eff_pivot_df’s columns:

      • Category
      • Total_Assets
      • Debts
      • Net_Wealth
      • Real_Asset_Ratio
      • Financial_Asset_Ratio
      • Debt_Ratio
      • Mean_Income
      • Business_Assets
      • Business_Asset_Ratio
    """
    pivot = generate_eff_pivot_df(eff)

    group_cols = [
        "Category",
        "Total_Assets",
        "Debts",
        "Net_Wealth",
        "Real_Asset_Ratio",
        "Financial_Asset_Ratio",
        "Debt_Ratio",
        "Mean_Income",
        "Business_Assets",
        "Business_Asset_Ratio",
    ]
    # sanity check
    missing = set(group_cols) - set(pivot.columns)
    if missing:
        raise RuntimeError(f"Pivot missing expected group_stats cols: {missing}")

    return pivot[group_cols].copy()


# ────────────────────────────────────────────────────────────
# Smoke-test – run `python load.py` to see basic shapes
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    eff_rows = load_eff_data().shape[0]
    pivot_df = generate_eff_group_stats(load_eff_data())
    grp = generate_eff_group_stats(load_eff_data())

    print(f"EFF rows kept: {eff_rows}")
    print("group_stats preview:")
    print(grp.head())

    rev, w = load_population_and_revenue_data("Regional_Age_Bin_Population_Shares.csv")
    print("Revenue rows:", len(rev), " |  Region weights:", len(w))
