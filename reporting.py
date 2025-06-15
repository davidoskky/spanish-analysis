import numpy as np
import pandas as pd
from ineqpy.inequality import gini

from statistic import top_share


def summarize_cap_and_tax_shares(df):
    top_10 = df["wealth_rank"] > 0.90
    top_1 = df["wealth_rank"] > 0.99
    top_01 = df["wealth_rank"] > 0.999

    total_relief = (df["cap_relief"] * df["facine3"]).sum()
    total_final_tax = (df["adjusted_final_tax"] * df["facine3"]).sum()

    top10_relief = (df.loc[top_10, "cap_relief"] * df.loc[top_10, "facine3"]).sum()
    top1_relief = (df.loc[top_1, "cap_relief"] * df.loc[top_1, "facine3"]).sum()
    top01_relief = (df.loc[top_01, "cap_relief"] * df.loc[top_01, "facine3"]).sum()

    top10_tax = (df.loc[top_10, "adjusted_final_tax"] * df.loc[top_10, "facine3"]).sum()
    top1_tax = (df.loc[top_1, "adjusted_final_tax"] * df.loc[top_1, "facine3"]).sum()
    top01_tax = (df.loc[top_01, "adjusted_final_tax"] * df.loc[top_01, "facine3"]).sum()

    print("Cap Relief Share:")
    print(f"  Top 10%: {top10_relief / total_relief:.2%}")
    print(f"  Top 1%:  {top1_relief / total_relief:.2%}")
    print(f"  Top 0.1%: {top01_relief / total_relief:.2%}\n")

    print("Final Tax Share:")
    print(f"  Top 10%: {top10_tax / total_final_tax:.2%}")
    print(f"  Top 1%:  {top1_tax / total_final_tax:.2%}")
    print(f"  Top 0.1%: {top01_tax / total_final_tax:.2%}")


def report_effective_tax_rates(df):
    def weighted_avg(series, weights):
        mask = series.notna()
        return np.average(series[mask], weights=weights[mask])

    top_10 = df["wealth_rank"] > 0.90
    top_1 = df["wealth_rank"] > 0.99

    eff_tax_top10 = weighted_avg(
        df.loc[top_10, "eff_tax_rate"], df.loc[top_10, "facine3"]
    )
    eff_tax_top1 = weighted_avg(df.loc[top_1, "eff_tax_rate"], df.loc[top_1, "facine3"])
    eff_tax_top10_nocap = weighted_avg(
        df.loc[top_10, "eff_tax_nocap"], df.loc[top_10, "facine3"]
    )
    eff_tax_top1_nocap = weighted_avg(
        df.loc[top_1, "eff_tax_nocap"], df.loc[top_1, "facine3"]
    )

    print("\n--- Effective Tax Rates ---")
    print(f"With Cap - Top 10%: {eff_tax_top10:.3%}")
    print(f"With Cap - Top 1%:  {eff_tax_top1:.3%}")
    print(f"Without Cap - Top 10%: {eff_tax_top10_nocap:.3%}")
    print(f"Without Cap - Top 1%:  {eff_tax_top1_nocap:.3%}")

    return df


def typology_impact_summary(df, weight_col="facine3"):
    typology_df = (
        df.groupby("mismatch_type")
        .apply(
            lambda g: pd.Series(
                {
                    "Population Share": g[weight_col].sum() / df[weight_col].sum(),
                    "Avg Final Tax": np.average(
                        g["adjusted_final_tax"], weights=g[weight_col]
                    ),
                    "Avg Sim Tax": np.average(
                        g["adjusted_sim_tax"], weights=g[weight_col]
                    ),
                    "Cap Relief Share": (g["cap_relief"] > 1e-6).mean(),
                    "Migration Rate": g["Migration_Exit"].mean(),
                    "Total Revenue": (g["adjusted_final_tax"] * g[weight_col]).sum(),
                }
            )
        )
        .reset_index()
    )
    print("\n--- Typology Impact Table ---")
    print(typology_df.to_string(index=False))
    return typology_df


def generate_summary_table(df, weight_col="facine3"):
    revenue_collected = (df["adjusted_final_tax"] * df[weight_col]).sum()
    revenue_without_cap = (df["adjusted_sim_tax"] * df[weight_col]).sum()
    cap_relief = revenue_without_cap - revenue_collected

    if "Migration_Exit" in df.columns:
        revenue_after_migration = (
            df.loc[~df["Migration_Exit"], "adjusted_final_tax"]
            * df.loc[~df["Migration_Exit"], weight_col]
        ).sum()
        migration_loss = revenue_collected - revenue_after_migration
    else:
        revenue_after_migration = np.nan
        migration_loss = np.nan

    erosion_base = (df["taxable_wealth"] - df["taxable_wealth_eroded"]).clip(lower=0)
    erosion_total_loss = (erosion_base * df[weight_col]).sum()

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Revenue Collected (with cap)",
                "Revenue Without Cap",
                "Cap Relief (Revenue Lost)",
                "Revenue After Migration",
                "Migration Loss",
                "Behavioral Erosion (Implicit Loss)",
            ],
            "EUR": [
                revenue_collected,
                revenue_without_cap,
                cap_relief,
                revenue_after_migration,
                migration_loss,
                erosion_total_loss,
            ],
        }
    )

    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False))
    return summary_df


def compute_inequality_metrics(df):
    metrics = {
        "Gini Before Tax": gini(
            income="netwealth_individual", weights="facine3", data=df
        ),
        "Gini After Tax (cap)": gini(
            income="wealth_after_cap", weights="facine3", data=df
        ),
        "Gini After Tax (no cap)": gini(
            income="wealth_after_no_cap", weights="facine3", data=df
        ),
        "Top 10% Share Before": top_share(df, "netwealth_individual", "facine3", 0.10),
        "Top 10% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.10),
        "Top 10% Share After (no cap)": top_share(
            df, "wealth_after_no_cap", "facine3", 0.10
        ),
        "Top 1% Share Before": top_share(df, "netwealth_individual", "facine3", 0.01),
        "Top 1% Share After (cap)": top_share(df, "wealth_after_cap", "facine3", 0.01),
        "Top 1% Share After (no cap)": top_share(
            df, "wealth_after_no_cap", "facine3", 0.01
        ),
    }

    print("\n--- Inequality Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4%}")
    return metrics
