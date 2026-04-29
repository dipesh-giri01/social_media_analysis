
import pandas as pd
import matplotlib.pyplot as plt
#LOAD DATA
#
df = pd.read_csv("Social_Media_Engagement Dataset.csv")
print("--- DATASET SHAPE ---")
print(df.shape)
# BASIC INFO
print("\n=== DATA INFO ===")
print(df.info())
print("\n=-- SUMMARY STATISTICS ---")
print(df.describe())
#CHECK MISSING VALUES
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
# SELECT NUMERICAL COLUMNS
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# OUTLIER DETECTION USING IQR
print("\n--- OUTLIER DETECTION (IQR METHOD) ---")
outlier_summary = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)
print(f"\nColumn: {col}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print(f"Number of Outliers: {len(outliers)}")
# FINAL SUMMARY
print("\n--- FINAL OUTLIER SUMMARY ---")
print(outlier_summary)

    