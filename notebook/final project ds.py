import os
import glob
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "merged")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

CHUNK_SIZE = 200_000
SAMPLE_PER_PART = 5000
RANDOM_SEED = 42

DROP_COLS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}

FINAL_SAMPLE = os.path.join(PROCESSED_DIR, "cicids2018_sample.parquet")


def clean_chunk(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan)

    to_drop = [c for c in df.columns if c in DROP_COLS]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    if "Label" not in df.columns:
        raise ValueError("Label column not found.")

    df["Label"] = df["Label"].astype(str).str.strip()
    df["is_attack"] = np.where(df["Label"].str.lower() == "benign", 0, 1)

    for col in df.columns:
        if col == "Label":
            continue
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    return df


def create_merged():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    part_idx = 0

    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = clean_chunk(chunk)
            out_path = os.path.join(INTERIM_DIR, f"part_{part_idx:05d}.parquet")
            chunk.to_parquet(out_path, index=False)
            part_idx += 1


def create_sample():
    parts = sorted(glob.glob(os.path.join(INTERIM_DIR, "part_*.parquet")))
    sampled = []

    for p in parts:
        df = pd.read_parquet(p)
        if len(df) > SAMPLE_PER_PART:
            df = df.sample(SAMPLE_PER_PART, random_state=RANDOM_SEED)
        sampled.append(df)

    final_df = pd.concat(sampled, ignore_index=True)
    final_df.to_parquet(FINAL_SAMPLE, index=False)


def main():
    create_merged()
    create_sample()


if __name__ == "__main__":
    main()
    
    
# converting the parquet file to csv to read and check data
df = pd.read_parquet("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_sample.parquet")

print(df.shape)

df.to_csv("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv", index=False)


df = pd.read_csv("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv")

print(df.shape)
print(df["is_attack"].value_counts())
print(df.isnull().sum().sum())