import pandas as pd

data = {
    'Product': ['Laptop', 'Smartphone', 'TV', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'Speakers'],
    'Price': [1000, 700, 1500, 200, 800, 50, 25, 500],
    'Quantity': [5, 10, 3, 30, 4, 60, 100, 20]
}

df = pd.DataFrame(data)
df_copy = df.copy()

# Calculate z-scores for price and quantity
df_copy["z-price"] = (df_copy["Price"] - df_copy["Price"].mean()) / df_copy["Price"].std()
df_copy["z-quantity"] = (df_copy["Quantity"] - df_copy["Quantity"].mean()) / df_copy["Quantity"].std()

df_copy.loc[abs(df_copy["z-price"]) > 2, "Price"] = df_copy['Price'].median()
df_copy.loc[abs(df_copy["z-quantity"]) > 2, "Quantity"] = df_copy['Quantity'].median()

print(df_copy)
