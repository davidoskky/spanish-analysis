import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dta_handling import load_data
from dta_handling import df_eff
from constants import wealth_percentile, income_percentile 
# from data_models import WealthIncomeMismatchType  # Only needed if used below

def assign_typology(df):
    """
    Assigns households to mismatch types based on:
    - 'percriq' (wealth group as string)
    - 'percrent' (income group as string)

    This function re-maps them to numeric percentiles for logic.
    """
    df = df.copy()

    wealth_map = {
        "< P25": 1,
        "P25-P50": 2,
        "P50-P75": 3,
        "P75-P90": 4,
        "> P90": 5
    }

    income_map = {
        "< P20": 1,
        "P20-P40": 2,
        "P40-P60": 3,
        "P60-80": 4,
        "P80-P90": 5,
        "> P90": 6
    }

    df["wealth_percentile"] = df["percriq"].map(wealth_map)
    df["income_percentile"] = df["percrent"].map(income_map)

    def classify_typology(row):
        if row["wealth_percentile"] >= 4 and row["income_percentile"] <= 2:
            return "Wealth-rich, income-poor"
        elif row["wealth_percentile"] <= 2 and row["income_percentile"] >= 5:
            return "Income-rich, wealth-poor"
        else:
            return "Aligned"

    df["mismatch_type"] = df.apply(classify_typology, axis=1)
    return df



def get_typology_counts(df):
    typology_counts = df["mismatch_type"].value_counts().reset_index()
    typology_counts.columns = ["mismatch_type", "count"]
    print("\nCounts of each mismatch type:")
    print(typology_counts)
    return typology_counts


def main():
    df = df_eff
    df = assign_typology(df)
    get_typology_counts(df)
    
    return df


if __name__ == "__main__":
    main()
