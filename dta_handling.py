from pprint import pprint

import numpy as np
import pandas as pd

from constants import PEOPLE_IN_HOUSEHOLD


def load_data(folder: str = "Data"):
    df_eff = pd.concat(
        [
            pd.read_csv(f"{folder}/databol{i}.csv", sep=";").assign(imputation=i)
            for i in range(1, 6)
        ]
    )
    df_eff["facine3"] /= 5  # Adjust weights to account for 5 implicates
    replace_dict = {
        "bage": {
            1: "Under 35",
            2: "35-44",
            3: "45-54",
            4: "55-64",
            5: "65-74",
            6: "Over 75",
        },
         "percrent": {
            1: "< P20",
            2: "P20-P40",
            3: "P40-P60",
            4: "P60-80",
            5: "P80-P90",
            6: "> P90",
        },
        "nsitlabdom": {
            1: "Employee",
            2: "Self-Employed",
            3: "Retired",
            4: "Other Inactive or Unemployed",
        },
        "neducdom": {
            1: "Below Secondary Education",
            2: "Secondary Education",
            3: "University Education",
        },
        "np2_1": {1: "Ownership", 2: "Other"},
        "nnumadtrab": {0: "None", 1: "One", 2: "Two", 3: "Three or More"},
        # PEOPLE_IN_HOUSEHOLD: {5: "5 or more"}, 5 is 5 or more but we are treating it as 5
        # "percriq": {1: "< P25", 2: "P25-P50", 3: "P50-P75", 4: "P75-P90", 5: "> P90"},
    }
    # df_eff[PEOPLE_IN_HOUSEHOLD] = df_eff[PEOPLE_IN_HOUSEHOLD].astype(int)
    df_eff = df_eff.replace(to_replace=replace_dict)
    df_eff[PEOPLE_IN_HOUSEHOLD] = pd.to_numeric(df_eff[PEOPLE_IN_HOUSEHOLD])
     # df_eff["percriq"] = pd.to_numeric(df_eff["percriq"]).fillna(0).astype(int)
    return df_eff


df_eff = pd.concat(
    [
        pd.read_csv(r"Data/databol{}.csv".format(i), sep=";").assign(imputation=i)
        for i in range(1, 6)
    ]
)
df_eff["facine3"] /= 5  # Adjust weights to account for 5 implicates
replace_dict = {
    "bage": {
        1: "Under 35",
        2: "35-44",
        3: "45-54",
        4: "55-64",
        5: "65-74",
        6: "Over 75",
    },
    "percrent": {
        1: "< P20",
        2: "P20-P40",
        3: "P40-P60",
        4: "P60-80",
        5: "P80-P90",
        6: "> P90",
    },
    "nsitlabdom": {
        1: "Employee",
        2: "Self-Employed",
        3: "Retired",
        4: "Other Inactive or Unemployed",
    },
    "neducdom": {
        1: "Below Secondary Education",
        2: "Secondary Education",
        3: "University Education",
    },
    "np2_1": {1: "Ownership", 2: "Other"},
    "nnumadtrab": {0: "None", 1: "One", 2: "Two", 3: "Three or More"},
    PEOPLE_IN_HOUSEHOLD: {5: "5 or more"},
    # "percriq": {1: "< P25", 2: "P25-P50", 3: "P50-P75", 4: "P75-P90", 5: "> P90"},
}
# df_eff[PEOPLE_IN_HOUSEHOLD] = df_eff[PEOPLE_IN_HOUSEHOLD].astype(int)
df_eff = df_eff.replace(to_replace=replace_dict)

print(df_eff)
pprint(list(filter(lambda x: "renthog" in x, df_eff.columns)))


def weighted_median(variable, weights):
    variable = variable.values
    weights = weights.values
    sorted_idx = np.argsort(variable)
    cum_weights = np.cumsum(weights[sorted_idx])
    lower_percentile_idx = np.searchsorted(cum_weights, 0.5 * cum_weights[-1])
    return variable[sorted_idx[lower_percentile_idx]]


# use a group by operation to calculate the weighted statistic for each implicate,
# and then average over the 5 implicates

mean_renthog = (
    df_eff.groupby("imputation")
    .apply(lambda x: np.average(x["renthog21_eur22"], weights=x["facine3"]))
    .mean()
)
print("Mean: {:.2f}".format(mean_renthog))
median_renthog = (
    df_eff.groupby("imputation")
    .apply(
        lambda x: weighted_median(variable=x["renthog21_eur22"], weights=x["facine3"])
    )
    .mean()
)
print("Median: {:.2f}".format(median_renthog))
print("riquezanet — basic stats:")
print(df_eff["riquezanet"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))

print("Total weighted households (facine3 sum):", df_eff["facine3"].sum())
print("Number of observations:", len(df_eff))
print("Average weight per household:", df_eff["facine3"].mean())
print("Max weight:", df_eff["facine3"].max())
print("Min weight:", df_eff["facine3"].min())
