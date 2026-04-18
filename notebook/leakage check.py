import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = r"D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

numeric_df = df.select_dtypes(include=["number"])

target_corr = numeric_df.corr()["is_attack"].sort_values(ascending=False)

print("\nTop positive correlations with is_attack:")
print(target_corr.head(15))

print("\nTop negative correlations with is_attack:")
print(target_corr.tail(15))

target_corr.to_csv(os.path.join(OUTPUT_DIR, "target_correlation_check.csv"))