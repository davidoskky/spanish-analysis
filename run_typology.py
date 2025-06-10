import pandas as pd
import numpy as np
from dta_handling import df_eff  # or from dta_handling import load_eff_data
from eff_typology import assign_typology

# --- STEP 1: Assign Typology ---
df_eff = assign_typology(df_eff)

# Inspect the result visually
print(df_eff[[
    'riquezanet', 
    'renthog21_eur22', 
    'wealth_decile', 
    'income_quintile', 
    'mismatch_type']].head(10))

# Weighted population share by mismatch type
shares = df_eff.groupby("mismatch_type")["facine3"].sum() / df_eff["facine3"].sum()
print("\nWeighted population share by mismatch type:")
print(shares)

# Average wealth and income by mismatch type
summary_stats = df_eff.groupby("mismatch_type").apply(
    lambda x: pd.Series({
        "mean_wealth": np.average(x["riquezanet"], weights=x["facine3"]),
        "mean_income": np.average(x["renthog21_eur22"], weights=x["facine3"]),
    })
)
print("\nAverage wealth and income by mismatch type:")
print(summary_stats)
