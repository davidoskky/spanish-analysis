import pandas as pd

from dta_handling import df_eff

def assign_typology(df):
    """
    Assigns households to mismatch types based on deciles of net wealth and quintiles of income.
    Returns a modified copy of the dataframe with columns:
    - wealth_decile
    - income_quintile
    - mismatch_type
    """

    df = df.copy()

    # Step 1: Compute wealth deciles and income quintiles within each implicate
    df['wealth_decile'] = df.groupby('imputation')['riquezanet'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 10, labels=False) + 1
    )
    df['income_quintile'] = df.groupby('imputation')['renthog21_eur22'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) + 1
    )

    # Step 2: Classify mismatch types
    def classify_typology(row):
        if row['wealth_decile'] >= 9 and row['income_quintile'] <= 2:
            return 'Wealth-rich, income-poor'
        elif row['wealth_decile'] <= 2 and row['income_quintile'] >= 4:
            return 'Income-rich, wealth-poor'
        else:
            return 'Aligned'

    df['mismatch_type'] = df.apply(classify_typology, axis=1)
    return df

