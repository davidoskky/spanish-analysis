# Dataframe column names
Net_Wealth = "riquezanet"
PEOPLE_IN_HOUSEHOLD = "np1"
Income = "renthog21_eur22"
wealth_percentile = "percriq"
working_status = "nsitlabdom"
income_percentile = "percrent"
Num_Workers = "nnumadtrab"
Primary_Residence = "p2_70"
Business_Value = "valhog"
Residence_Ownership = "np2_1"
Business_Ownership = "havenegval"

# Data
SPANISH_PIT_2022_BRACKETS = [
    (0, 12_450, 0.19),
    (12_450.01, 20_200, 0.24),
    (20_200.01, 35_200, 0.30),
    (35_200.01, 60_000, 0.37),
    (60_000.01, 300_000, 0.45),
    (300_000.01, float("inf"), 0.47),
]

PROGRESSIVE_TAX_BRACKETS = [
    (0, 167129.45, 0.002),
    (167129.46, 334246.88, 0.003),
    (334246.89, 668499.75, 0.005),
    (668499.76, 1336999.51, 0.009),
    (1336999.52, 2673999.01, 0.013),
    (2673999.02, 5347998.03, 0.017),
    (5347998.04, 10695996.06, 0.021),
    (10695996.07, float("inf"), 0.035),
]

def individual_split(df):
    """
    Assign all household wealth and income to a single individual for accurate
    wealth tax simulation. Avoids dilution of wealth in multi-earner households.
    """
    df = df.copy()
    df["netwealth_individual"] = df[Net_Wealth]
    df["income_individual"] = df[Income]
    return df


def individual_split(df):
    """
    Decomposes household-level net wealth and income into individual-level equivalents.

    Since income and wealth are reported per household, this function tries to approximate
    per-capita figures by dividing by the number of economic contributors (working adults).
    Where no earners are reported, one worker is assumed as a fallback proxy.
    """
    df = df.copy()
    adult_split_factor = df[Num_Workers].clip(lower=1)

    df["netwealth_individual"] = df[Net_Wealth] / adult_split_factor
    df["income_individual"] = df[Income] / adult_split_factor

    return df

def individual_split(df):
    """
    Decomposes household-level net wealth and income into individual equivalents.
    
    - Income is split among actual earners (Num_Workers)
    - Wealth is split across 2 adults if multiple earners exist, otherwise 1
    
    This reflects realistic income generation and shared asset ownership patterns.
    """
    df = df.copy()

    # Income: split among actual earners (clip to avoid division by zero)
    income_split_factor = df[Num_Workers].clip(lower=1)

    # Wealth: split across 2 adults if more than one earner, otherwise 1
    wealth_split_factor = np.where(df[Num_Workers] > 1, 2, 1)

    df["income_individual"] = df[Income] / income_split_factor
    df["netwealth_individual"] = df[Net_Wealth] / wealth_split_factor

    return df


def simulate_pit_liability(df: pd.DataFrame):
    """
    Simulates Spanish PIT liability with a basic personal allowance.
    This prevents underestimating PIT and avoids excessive WT income caps.
    """
    df = df.copy()

    # Personal allowance (2022 value): €5,550 per taxpayer
    personal_allowance = 5550
    taxable_income = np.maximum(df["income_individual"] - personal_allowance, 0)

    df["pit_liability"] = taxable_income.apply(
        lambda amount: calculate_tax_liability(amount, SPANISH_PIT_2022_BRACKETS)
    )
    total_pit = df["pit_liability"].sum()
    print(f"Total PIT liability: €{total_pit:,.2f}")

    return df


