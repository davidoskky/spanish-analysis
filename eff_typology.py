import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dta_handling import load_data
from dta_handling import df_eff
from constants import wealth_percentile, income_percentile 

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

     if row["wealth_percentile"] >= 4 and row["income_percentile"] <= 3:
        return "Wealth-rich, income-poor"
     elif row["wealth_percentile"] <= 2 and row["income_percentile"] >= 5:
        return "Income-rich, wealth-poor"
     else:
        return "Aligned"

    df["mismatch_type"] = df.apply(classify_typology, axis=1)

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
        lambda x: pd.Series({
            "mean_wealth": np.average(x["riquezanet"], weights=x["facine3"]),
            "mean_income": np.average(x["renthog21_eur22"], weights=x["facine3"]),
        })
    )
    print("\nAverage wealth and income by mismatch type:")
    print(summary_stats)

    return typology_counts, shares, summary_stats
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_mean_by_percentile_group(df, group_col, value_col, weight_col, title="", ylabel=""):
    """
    Plots weighted average of value_col by percentile group (e.g., percriq or percrent),
    sorted in meaningful economic order.
    """
    if group_col == "percriq":
        ordered_groups = ["< P25", "P25-P50", "P50-P75", "P75-P90", "> P90"]
    elif group_col == "percrent":
        ordered_groups = ["< P20", "P20-P40", "P40-P60", "P60-80", "P80-P90", "> P90"]
    else:
        ordered_groups = sorted(df[group_col].dropna().unique().tolist())

    # Compute weighted means
    grouped = df.groupby(group_col).apply(
        lambda g: np.average(g[value_col], weights=g[weight_col])
    ).reset_index(name="weighted_mean")

    grouped[group_col] = pd.Categorical(grouped[group_col], categories=ordered_groups, ordered=True)
    grouped = grouped.sort_values(group_col)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=grouped, x=group_col, y="weighted_mean", color="steelblue", edgecolor="black")
    
    # Format y-axis with thousand separators and €
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    plt.title(title, fontsize=16)
    plt.xlabel("")  # group_col, if needed
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle="--", axis="y", alpha=0.6)
    plt.tight_layout()
    plt.show()




def main():
    df = df_eff
    df = assign_typology(df)
    get_typology_statistics(df)

    plot_mean_by_percentile_group(
    df=df_eff,
    group_col="percriq",
    value_col="riquezanet",
    weight_col="facine3",
    title="Average Net Wealth by Wealth Group",
    ylabel="Net Wealth (€)"
)

    plot_mean_by_percentile_group(
    df=df_eff,
    group_col="percrent",
    value_col="renthog21_eur22",
    weight_col="facine3",
    title="Average Income by Income Group",
    ylabel="Household Income (€)"
)

    return df


if __name__ == "__main__":
    main()
