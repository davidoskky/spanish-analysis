import unicodedata

import pandas as pd

EFF_DF_COLUMNS= ["Concept", "Element", "Statistic", "Breakdown", "Category", "Measure", "Wave", "Value"]
def load_eff_data(file_path="eff_data.xlsx"):
    eff_df = pd.read_excel(file_path, sheet_name="Datos", skiprows=10)
    eff_df.columns = ["Concept", "Element", "Statistic", "Breakdown", "Category", "Measure", "Wave", "Value"]
    eff_df["Wave"] = pd.to_numeric(eff_df["Wave"], errors="coerce")
    eff_df["Value"] = pd.to_numeric(eff_df["Value"], errors="coerce")
    eff_df["Value"] *= 1000
    eff_df["Category"] = eff_df["Category"].astype(str).str.strip().str.replace("\u2013", "-", regex=False)

    filtered = eff_df[
        (eff_df["Wave"] == 2022) &
        (eff_df["Breakdown"] == "NET WEALTH PERCENTILE") &
        (eff_df["Statistic"].str.upper() == "MEAN") &
        (eff_df["Value"].notna())
        ].copy()
    return filtered


def process_eff_assets_income(filtered_df):
    real_assets = [
        "MAIN RESIDENCE", "OTHER REAL ESTATE PROPERTIES",
        "CARS AND OTHER VEHICLES", "OTHER DURABLE GOODS"
    ]
    financial_assets = [
        "LISTED SHARES", "INVESTMENT FUNDS", "FIXED-INCOME SECURITIES",
        "PENSION SCHEMES AND UNIT-LINKED OR MIXED LIFE INSURANCE",
        "ACCOUNTS AND DEPOSITS USABLE FOR PAYMENTS",
        "ACCOUNTS NON USABLE FOR PAYMENTS AND HOUSE-PURCHASE SAVING ACCOUNTS",
        "OTHER FINANCIAL ASSETS", "UNLISTED SHARES AND OTHER EQUITY"
    ]
    debts = ["TOTAL DEBT"]

    expected_columns = set(real_assets + financial_assets + debts)
    if not expected_columns.issubset(set(filtered_df["Element"].unique())):
        missing = expected_columns - set(filtered_df["Element"].unique())
        raise ValueError(f"Missing expected elements in data: {missing}")

    pivot_df = filtered_df.pivot_table(index="Category", columns="Element", values="Value", aggfunc="mean").fillna(0)

    pivot_df["Real_Assets"] = pivot_df[real_assets].sum(axis=1)
    pivot_df["Financial_Assets"] = pivot_df[financial_assets].sum(axis=1)
    pivot_df["Total_Assets"] = pivot_df["Real_Assets"] + pivot_df["Financial_Assets"]
    pivot_df["Debts"] = pivot_df[debts].sum(axis=1)
    pivot_df["Net_Wealth"] = pivot_df["Total_Assets"] - pivot_df["Debts"]

    # Avoid division by zero
    pivot_df["Real_Asset_Ratio"] = pivot_df["Real_Assets"] / pivot_df["Total_Assets"].replace(0, np.nan)
    pivot_df["Financial_Asset_Ratio"] = pivot_df["Financial_Assets"] / pivot_df["Total_Assets"].replace(0, np.nan)
    pivot_df["Debt_Ratio"] = pivot_df["Debts"] / pivot_df["Total_Assets"].replace(0, np.nan)

    income_df = filtered_df[filtered_df["Concept"].str.upper().str.contains("INCOME")][["Category", "Value"]].copy()
    income_df.columns = ["Category", "Mean_Income"]
    income_df["Mean_Income"] *= 1000

    business_df = filtered_df[filtered_df["Element"] == "BUSINESSES RELATED TO SELF-EMPLOYMENT"][
        ["Category", "Value"]].copy()
    business_df.columns = ["Category", "Business_Assets"]

    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.merge(income_df, on="Category", how="left")
    pivot_df = pivot_df.merge(business_df, on="Category", how="left")
    pivot_df["Business_Assets"] = pivot_df["Business_Assets"].fillna(0.0)
    pivot_df["Business_Asset_Ratio"] = pivot_df["Business_Assets"] / pivot_df["Total_Assets"].replace(0, np.nan)
    pivot_df["Category"] = pivot_df["Category"].astype(str).str.strip().str.lower()

    group_stats_df = pivot_df.drop_duplicates(subset="Category")[[
        "Category", "Total_Assets", "Debts", "Net_Wealth",
        "Real_Asset_Ratio", "Financial_Asset_Ratio", "Debt_Ratio",
        "Mean_Income", "Business_Assets", "Business_Asset_Ratio"
    ]]

    return pivot_df, group_stats_df

def load_population_and_revenue_data(pop_file):
    observed_df = pd.read_csv("Cleaned_Regional_Wealth_Tax_Data.csv")
    observed_clean = observed_df[
        observed_df["Variable"].str.strip().str.lower() == "resultado de la declaración"
    ].copy()
    observed_clean["Region"] = observed_clean["Region"].str.strip().str.lower()
    observed_clean = observed_clean.rename(columns={"Importe": "Total_Revenue"})
    revenue_df = observed_clean[["Region", "Total_Revenue"]].copy()

    pop_shares = pd.read_csv(pop_file)
    pop_shares["Region"] = pop_shares["Region"].str.replace(r"^\d+\s+", "", regex=True)
    pop_shares["Region"] = pop_shares["Region"].apply(
        lambda x: unicodedata.normalize("NFKD", x.strip()).encode("ascii", errors="ignore").decode("utf-8").lower()
    )

    province_to_region = {
        "madrid": "madrid", "madrid, comunidad de": "madrid",
        "barcelona": "catalonia", "girona": "catalonia", "lleida": "catalonia", "tarragona": "catalonia", "cataluna": "catalonia",
        "valencia/valencia": "valencia", "alicante/alacant": "valencia", "castellon/castello": "valencia", "comunitat valenciana": "valencia",
        "coruna, a": "galicia", "lugo": "galicia", "ourense": "galicia", "pontevedra": "galicia",
        "asturias, principado de": "asturias", "asturias": "asturias",
        "caceres": "extremadura", "badajoz": "extremadura"
    }

    pop_shares["Autonomous_Region"] = pop_shares["Region"].map(province_to_region)
    dropped = pop_shares["Autonomous_Region"].isna().sum()
    if dropped > 0:
        print(f"⚠️ {dropped} rows dropped due to unmatched province mapping.")
    pop_shares = pop_shares[pop_shares["Autonomous_Region"].notna()].copy()

    region_population = pop_shares.groupby("Autonomous_Region", as_index=False)["Population"].sum()
    region_population.columns = ["Region", "Population"]
    region_population["Population"] = region_population["Population"] / region_population["Population"].sum()

    return revenue_df, region_population
