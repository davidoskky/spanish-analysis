import pandas as pd
import numpy as np


def assign_typology(df):
    """
    Assigns wealth and income percentiles as numeric ranks and classifies mismatch typologies.
    Conditions:
    - Wealth-rich = top 25% (>= 75th percentile)
    - Wealth-poor = bottom 50% (< 50th percentile)
    - Income-rich = top 20% (>= 80th percentile)
    - Income-poor = bottom 60% (< 60th percentile)
    """
    # Compute percentile ranks
    df = df.copy()
    df["wealth_percentile"] = df["riquezanet"].rank(pct=True)
    df["income_percentile"] = df["renthog21_eur22"].rank(pct=True)

    # Classify mismatch typology
    def classify(row):
        w = row["wealth_percentile"]
        i = row["income_percentile"]

        if w >= 0.75 and i < 0.60:
            return "Wealth-rich, income-poor"
        elif w < 0.50 and i >= 0.80:
            return "Income-rich, wealth-poor"
        else:
            return "Aligned"

    df["mismatch_type"] = df.apply(classify, axis=1)
    return df


def get_typology_statistics(df):
    typology_counts = df["mismatch_type"].value_counts().reset_index()
    typology_counts.columns = ["mismatch_type", "count"]
    print("\nCounts of each mismatch type:")
    print(typology_counts)

    shares = df.groupby("mismatch_type")["facine3"].sum() / df["facine3"].sum()
    print("\nWeighted population share by mismatch type:")
    print(shares)

    # Average wealth and income by mismatch type
    summary_stats = df.groupby("mismatch_type").apply(
        lambda x: pd.Series(
            {
                "mean_wealth": np.average(x["riquezanet"], weights=x["facine3"]),
                "mean_income": np.average(x["renthog21_eur22"], weights=x["facine3"]),
            }
        )
    )
    print("\nAverage wealth and income by mismatch type:")
    print(summary_stats)

    return typology_counts, shares, summary_stats


def main():
    df = df_eff
    df = assign_typology(df)

    get_typology_statistics(df)

    return df


if __name__ == "__main__":
    main()
