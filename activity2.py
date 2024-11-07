import pandas as pd

data = {
    'Name': ['John', 'Mary', 'Alex', 'Sarah', 'Michael', 'Laura', 'David', 'Anna'],
    'Age': [28, None, 30, 32, None, 25, 27, 29],
    'Salary': [50000, None, None, 70000, 80000, 45000, None, 55000],
    'Department': ['IT', None, 'IT', None, 'Marketing', 'IT', 'Marketing', None]
}

df = pd.DataFrame(data)
df_copy = df.copy()

# df_copy["Age"] = df_copy["Age"].fillna(df_copy["Age"].median())
# df_copy["Salary"] = df_copy["Salary"].fillna(df_copy["Salary"].mean())
# df_copy["Department"] = df_copy["Department"].fillna(df_copy["Department"].mode()[0])


df_copy = df_copy.dropna(thresh=len(df_copy.columns) - 2)



print(df_copy)
