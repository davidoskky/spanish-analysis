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
    "Concept", "Element", "Statistic", "Breakdown",
    "Category", "Measure", "Wave", "Value"
]

REAL_ASSETS = [
    "MAIN RESIDENCE", "OTHER REAL ESTATE PROPERTIES",
    "CARS AND OTHER VEHICLES", "OTHER DURABLE GOODS"
]
FIN_ASSETS = [
    "LISTED SHARES", "INVESTMENT FUNDS", "FIXED-INCOME SECURITIES",
    "PENSION SCHEMES AND UNIT-LINKED OR MIXED LIFE INSURANCE",
    "ACCOUNTS AND DEPOSITS USABLE FOR PAYMENTS",
    "ACCOUNTS NON USABLE FOR PAYMENTS AND HOUSE-PURCHASE SAVING ACCOUNTS",
    "OTHER FINANCIAL ASSETS", "UNLISTED SHARES AND OTHER EQUITY"
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


# ────────────────────────────────────────────────────────────
# 2.  AGGREGATION ░░ process_eff_assets_income ░░
# ────────────────────────────────────────────────────────────
def process_eff_assets_income(
    eff_filtered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform row-level EFF into:
    • pivot_df      – every wealth-percentile ‘Category’ with full asset break-down
    • group_stats   – tidy subset used by the synthetic-population generator
    """
    expected = set(REAL_ASSETS + FIN_ASSETS + DEBTS)
    missing = expected.difference(eff_filtered["Element"].unique())
    if missing:
        raise ValueError(f"EFF file missing elements: {sorted(missing)}")

    pv = (
        eff_filtered.pivot_table(
            index="Category",
            columns="Element",
            values="Value",
            aggfunc="mean",
        )
        .fillna(0)
    )

    pv["Real_Assets"] = pv[REAL_ASSETS].sum(1)
    pv["Financial_Assets"] = pv[FIN_ASSETS].sum(1)
    pv["Total_Assets"] = pv["Real_Assets"] + pv["Financial_Assets"]
    pv["Debts"] = pv[DEBTS].sum(1)
    pv["Net_Wealth"] = pv["Total_Assets"] - pv["Debts"]

    # Ratios (guard against division by zero)
    denom = pv["Total_Assets"].replace(0, np.nan)
    pv["Real_Asset_Ratio"] = pv["Real_Assets"] / denom
    pv["Financial_Asset_Ratio"] = pv["Financial_Assets"] / denom
    pv["Debt_Ratio"] = pv["Debts"] / denom

    # Income & business assets
    income = (
        eff_filtered[eff_filtered["Concept"].str.contains("income", case=False)]
        .loc[:, ["Category", "Value"]]
        .rename(columns={"Value": "Mean_Income"})
        .assign(Mean_Income=lambda d: d["Mean_Income"] * 1000)
    )
    business = (
        eff_filtered[eff_filtered["Element"] == "BUSINESSES RELATED TO SELF-EMPLOYMENT"]
        .loc[:, ["Category", "Value"]]
        .rename(columns={"Value": "Business_Assets"})
    )

    pv = (
        pv.reset_index()
        .merge(income, on="Category", how="left")
        .merge(business, on="Category", how="left")
        .fillna({"Business_Assets": 0.0})
    )
    pv["Business_Asset_Ratio"] = pv["Business_Assets"] / denom
    pv["Category"] = pv["Category"].map(_norm)

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
    return pv, pv[group_cols]


# ────────────────────────────────────────────────────────────
# 3.  POPULATION & REVENUE ░░ load_population_and_revenue_data ░░
# ────────────────────────────────────────────────────────────
def load_population_and_revenue_data(
    pop_path: str | Path,
    revenue_path: str | Path = "Cleaned_Regional_Wealth_Tax_Data.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (AEAT revenue, region-level population *share*)."""
    # —— Revenue ——————————————————————————
    rev = (
        pd.read_csv(revenue_path)
        .query("Variable.str.strip().str.lower() == 'resultado de la declaración'", engine="python")
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


# ────────────────────────────────────────────────────────────
# Smoke-test – run `python load.py` to see basic shapes
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    eff_rows = load_eff_data().shape[0]
    _, grp = process_eff_assets_income(load_eff_data())
    print(f"EFF rows kept: {eff_rows}")
    print("group_stats preview:")
    print(grp.head())

    rev, w = load_population_and_revenue_data("Regional_Age_Bin_Population_Shares.csv")
    print("Revenue rows:", len(rev), " |  Region weights:", len(w))
