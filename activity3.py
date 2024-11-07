import pandas as pd

data = {
    'Product': ['Laptop', 'Laptop', 'Smartphone', 'smartphone', 'TV', 'tv', 'Headphones', 'headphones'],
    'Category': ['Electronics', 'electronics', 'Electronics', 'electronics', 'Electronics', 'electronics', 'Electronics', 'electronics'],
    'Price': [1000, 1200, 700, 750, 1500, 1400, 200, 200],
    'Quantity': [5, 3, 8, 10, 2, 4, 15, 15]
}

df = pd.DataFrame(data)
df_copy = df.copy()

df_copy["Category"] = df_copy["Category"].str.title()
df_copy["Product"] = df_copy["Product"].str.title()


df_copy = df_copy.drop_duplicates(keep="first")

print(df_copy)
