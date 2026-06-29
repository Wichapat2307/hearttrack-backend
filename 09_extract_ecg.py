"""
extract_demo_npy.py — HeartTrack Raw Signal Extractor
======================================================
This script extracts precise 3-minute clinical windows from the PhysioNet PAFDB
and saves them as raw .npy (NumPy binary) arrays for the web app to process.
"""

import os
import numpy as np
import wfdb

# --- CONFIGURATION ---
# Path to your PhysioNet database files
DB_PATH = "./PAFPDB"
# Path to your React app's public folder
OUTPUT_DIR = "./hearttrack-ui/public/demo_files"
# 3 minutes is the clinical standard for short-term HRV
SNIPPET_MINUTES = 3  

# The 9 core demo scenarios (Test Set: t01 - t12)
EXTRACTIONS = [
    # 🟢 HEALTHY CONTROLS (No history of AFib)
    dict(record="t03", start_min=0,  out_name="t03_healthy_control"),
    dict(record="t05", start_min=0,  out_name="t05_healthy_control"),
    dict(record="t07", start_min=0,  out_name="t07_healthy_control"),

    # 🟡 STABLE PAF BASELINES (Has AFib, but currently resting normally)
    dict(record="t01", start_min=0,  out_name="t01_paf_baseline"),
    dict(record="t09", start_min=0,  out_name="t09_paf_baseline"),
    dict(record="t11", start_min=0,  out_name="t11_paf_baseline"),

    # 🔴 IMMINENT AFIB ATTACKS (The final 3 minutes right before AFib triggers)
    dict(record="t02", start_min=27, out_name="t02_paf_imminent"),
    dict(record="t10", start_min=27, out_name="t10_paf_imminent"),
    dict(record="t12", start_min=27, out_name="t12_paf_imminent"),
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving {SNIPPET_MINUTES}-minute raw ECG arrays to: {OUTPUT_DIR}\n")

    success_count = 0

    for cfg in EXTRACTIONS:
        record_path = os.path.join(DB_PATH, cfg["record"])
        
        try:
            # 1. Read header to get sampling frequency (usually 128 Hz for PAFDB)
            header = wfdb.rdheader(record_path)
            fs = header.fs
            
            # 2. Calculate row indices based on minutes
            start_sample = int(cfg["start_min"] * 60 * fs)
            end_sample = int((cfg["start_min"] + SNIPPET_MINUTES) * 60 * fs)
            
            # 3. Read the actual binary .dat data
            record = wfdb.rdrecord(record_path, sampfrom=start_sample, sampto=end_sample)
            
            # Extract Channel 0 (usually Lead I or ECG1)
            raw_ecg_signal = record.p_signal[:, 0]
            
            # 4. Save directly as a lightweight numpy array (.npy)
            out_file = os.path.join(OUTPUT_DIR, f"{cfg['out_name']}.npy")
            np.save(out_file, raw_ecg_signal)
            
            print(f"✅ Extracted: {cfg['out_name']}.npy | Length: {len(raw_ecg_signal)} samples")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to extract {cfg['record']}: {e}")
            print(f"   Ensure {cfg['record']}.hea and .dat exist in {DB_PATH}")

    print(f"\nDone! Successfully generated {success_count}/{len(EXTRACTIONS)} demo files.")

if __name__ == "__main__":
    main()