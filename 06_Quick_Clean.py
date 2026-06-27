import pandas as pd

df = pd.read_csv("paf_hrv_dataset.csv")

# 1. Check how many rows we start with
print(f"Original rows: {len(df)}")

# 2. Apply Physiological Limits
# We drop any window where the mean heartbeat gap is > 2000ms (Under 30 BPM)
# or < 300ms (Over 200 BPM), or where SDNN is impossibly high (> 500)
clean_df = df[
    (df['mean_rr'] > 300) & 
    (df['mean_rr'] < 2000) & 
    (df['sdnn'] < 500)
]

# 3. Check how many rows are left
print(f"Cleaned rows: {len(clean_df)}")
print(f"Removed {len(df) - len(clean_df)} corrupted windows.")

# Save the final pristine dataset
clean_df.to_csv("paf_hrv_dataset_clean.csv", index=False)