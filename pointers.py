import pandas as pd
import numpy as np
from scipy import stats

# Sample data with various issues
data = {
    'Product': ['Laptop', 'laptop', None, 'PHONE', 'TV ', ' Monitor', 'Laptop', 'Mouse', 'TABLET', '  keyboard', 'Mouse pad', 'HeadPhones'],
    'Price': [1000, 1200, 700, -50, 99999, 300, 1000, None, 800, 50, 20, 150],
    'Category': ['Electronics', None, 'electronics', 'ELECTRONICS', None, 'Electronics', 'Electronics', 'electronics', 'ELECTRONICS', None, 'accessories', 'Audio'],
    'Quantity': [5, 3, -1, 10, 2, 4, 5, 15, -3, 100, 50, 25],
    'Rating': [4.5, 4.2, None, 3.8, 5.0, 2.1, 4.5, None, 3.9, 4.0, None, 4.8]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\n")

# 1. Handling Missing Values
print("1. Handling Missing Values:")

# Fill NA with specific value
df_fillna = df.fillna("Unknown")
print("\nFilling NA with specific value:")
print(df_fillna)

# Fill NA with method (ffill, bfill)
df_ffill = df.fillna(method='ffill')  # forward fill
print("\nForward fill:")
print(df_ffill)

# Fill NA with different values for different columns
df_fill_dict = df.fillna({'Product': 'Unknown', 'Price': df['Price'].mean(), 'Category': 'Unspecified', 'Rating': df['Rating'].median()})
print("\nFill NA with dictionary:")
print(df_fill_dict)

# Drop NA rows/columns
df_dropna = df.dropna(how='any')  # 'any' or 'all'
print("\nDrop rows with any NA:")
print(df_dropna)

# 2. Removing Duplicates
print("\n2. Removing Duplicates:")
df_unique = df.drop_duplicates(subset=['Product', 'Price'], keep='first')  # keep='first','last','False'
print(df_unique)

# 3. Handling Outliers using Multiple Methods
print("\n3. Handling Outliers:")

# 3.1 IQR Method
print("\nIQR Method:")
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df_outliers_iqr = df.copy()
df_outliers_iqr.loc[df_outliers_iqr['Price'] > (Q3 + 1.5 * IQR), 'Price'] = Q3 + 1.5 * IQR
df_outliers_iqr.loc[df_outliers_iqr['Price'] < (Q1 - 1.5 * IQR), 'Price'] = Q1 - 1.5 * IQR
print(df_outliers_iqr)

# 3.2 Z-Score Method
print("\nZ-Score Method:")
df_outliers_z = df.copy()
z_scores = stats.zscore(df_outliers_z['Price'].fillna(df_outliers_z['Price'].mean()))
df_outliers_z['z_scores'] = z_scores
df_outliers_z.loc[abs(df_outliers_z['z_scores']) > 2, 'Price'] = df_outliers_z['Price'].median()
print(df_outliers_z)

# 4. Advanced String Cleaning
print("\n4. Advanced String Cleaning:")
df_clean = df.copy()
# Strip whitespace and convert to consistent case
df_clean['Product'] = df_clean['Product'].str.strip().str.title()
df_clean['Category'] = df_clean['Category'].str.strip().str.title()
# Remove special characters
df_clean['Product'] = df_clean['Product'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
# Group similar categories
df_clean['Category'] = df_clean['Category'].replace({'Electronics': 'Electronics', 
                                                    'Accessories': 'Electronics Accessories',
                                                    'Audio': 'Electronics'})
print(df_clean)

# 5. Advanced Value Replacement
print("\n5. Advanced Value Replacement:")
df_replace = df.copy()
# Replace negative values
df_replace['Quantity'] = df_replace['Quantity'].clip(lower=0)  # Replace negative quantities
df_replace['Price'] = df_replace['Price'].clip(lower=0)  # Replace negative prices
# Replace outliers using winsorization
df_replace['Price'] = stats.mstats.winsorize(df_replace['Price'].fillna(df_replace['Price'].median()), limits=[0.05, 0.05])
print(df_replace)

# 6. Enhanced Type Conversion
print("\n6. Enhanced Type Conversion:")
df_convert = df.copy()
# Convert to appropriate types
df_convert['Price'] = pd.to_numeric(df_convert['Price'], errors='coerce')
df_convert['Rating'] = pd.to_numeric(df_convert['Rating'], errors='coerce')
df_convert['Quantity'] = pd.to_numeric(df_convert['Quantity'], errors='coerce')
# Convert categories to categorical type
df_convert['Category'] = df_convert['Category'].astype('category')
print(df_convert.dtypes)

# 7. Advanced Value Range Validation
print("\n7. Advanced Value Range Validation:")
df_valid = df.copy()
# Set valid ranges for each column
df_valid.loc[df_valid['Quantity'] < 0, 'Quantity'] = 0
df_valid.loc[df_valid['Price'] < 0, 'Price'] = 0
df_valid.loc[df_valid['Rating'] > 5, 'Rating'] = 5
df_valid.loc[df_valid['Rating'] < 0, 'Rating'] = 0
print(df_valid)

# 8. Feature Engineering
print("\n8. Feature Engineering:")
df_features = df.copy()
# Create price categories
df_features['Price_Category'] = pd.qcut(df_features['Price'].fillna(df_features['Price'].median()), 
                                      q=3, labels=['Budget', 'Mid-range', 'Premium'])
# Create quantity categories
df_features['Stock_Status'] = pd.cut(df_features['Quantity'], 
                                   bins=[-np.inf, 0, 5, 20, np.inf],
                                   labels=['Out of Stock', 'Low Stock', 'Medium Stock', 'High Stock'])
print(df_features)

# Additional Data Cleaning Steps
print("\n9. Additional Data Cleaning:")

# Check missing values
print("Missing Values Count:")
print(df.isnull().sum())

# New DataFrame with dropped missing values
df_dropped = df.dropna()
print("\nDataFrame after dropping all missing values:")
print(df_dropped)

# Impute values with statistics
df_imputed = df.copy()
df_imputed['Price'].fillna(df_imputed['Price'].mean(), inplace=True)
df_imputed['Rating'].fillna(df_imputed['Rating'].median(), inplace=True)
df_imputed['Category'].fillna(df_imputed['Category'].mode()[0], inplace=True)
df_imputed['Product'].fillna('Unknown', inplace=True)

# Additional duplicate handling
df_no_dupes = df_imputed.drop_duplicates()
df_no_dupes_last = df_imputed.drop_duplicates(keep='last')
df_no_dupes_subset = df_imputed.drop_duplicates(subset=['Product'])

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return text
    return str(text).strip().lower()

# Apply text cleaning to string columns
df_imputed['Product'] = df_imputed['Product'].apply(clean_text)
df_imputed['Category'] = df_imputed['Category'].apply(clean_text)

print("\nFinal cleaned and processed DataFrame:")
print(df_imputed)

# Using apply with lambda to access all columns
def process_row(row):
    # Example processing using multiple columns
    if pd.isna(row['Price']):
        if not pd.isna(row['Rating']) and row['Rating'] > 4.0:
            return row['Quantity'] * 100
        else:
            return 0
    return row['Price'] * row['Quantity']

# Add new column based on multiple column processing
df_imputed['Revenue'] = df_imputed.apply(process_row, axis=1)

# Can also modify multiple columns at once
def clean_row(row):
    row['Product'] = str(row['Product']).strip().title()
    row['Category'] = str(row['Category']).strip().title() 
    row['Price'] = abs(row['Price']) if not pd.isna(row['Price']) else row['Price']
    row['Quantity'] = abs(row['Quantity']) if not pd.isna(row['Quantity']) else row['Quantity']
    return row

# Apply transformations across all columns
df_imputed = df_imputed.apply(clean_row, axis=1)

print("\nDataFrame after applying row-wise transformations:")
print(df_imputed)

