import pandas as pd

data = {
    'Product': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Sales': [200, 300, 400, None, 500, 600, None, 800, 900, 1000],
    'Cost': [150, 200, None, 250, 300, 350, 400, None, 500, 600],
    'Profit': [50, 100, 150, 200, None, 250, None, 400, 450, None]
}

df = pd.DataFrame(data)
df_copy = df.copy()

df_copy["Sales"] = df_copy["Sales"].fillna(df_copy["Sales"].mean())
df_copy["Cost"] = df_copy["Cost"].fillna(df_copy["Cost"].median())
df_copy["Profit"] = df_copy["Profit"].fillna(0)

df_copy = df_copy.drop_duplicates()

Q1 = df_copy["Sales"].quantile(0.25)
Q3 = df_copy["Sales"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Checks for outliers in Sales
outliers = df_copy[(df_copy["Sales"] <= lower_bound) | (df_copy["Sales"] >= upper_bound)]

df_copy.loc[df_copy["Sales"] < lower_bound, "Sales"] = lower_bound
df_copy.loc[df_copy["Sales"] > upper_bound, "Sales"] = upper_bound


print(outliers)