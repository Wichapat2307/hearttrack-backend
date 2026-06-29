"""
09_extract_ecg.py  —  HeartTrack PAFPDB NPY Extractor
========================================================
Extracts raw ECG signals as .npy files from PAFPDB.
Each file contains 30 seconds of raw ECG at 128 Hz = 3840 samples.

Output → ./hearttrack-ui/public/demo_files/
"""

import os
import numpy as np
import wfdb

DB_PATH    = "./PAFPDB"
OUTPUT_DIR = "./hearttrack-ui/public/demo_files"
SNIPPET_SEC = 30   # 30 seconds per demo file

EXTRACTIONS = [
    # 🟢 HEALTHY CONTROLS (No history of AFib)
    dict(record="t03", start_min=0,  out_name="demo_healthy_1",
         label_key="normal", short="Healthy Control #1 (t03)",
         description="Healthy subject — no AF history."),
    dict(record="t05", start_min=0,  out_name="demo_healthy_2",
         label_key="normal", short="Healthy Control #2 (t05)",
         description="Healthy subject — no AF history."),
    dict(record="t07", start_min=0,  out_name="demo_healthy_3",
         label_key="normal", short="Healthy Control #3 (t07)",
         description="Healthy subject — no AF history."),

    # 🟡 STABLE PAF BASELINES (Has AFib, but currently resting normally)
    dict(record="t01", start_min=0,  out_name="demo_stable_1",
         label_key="paf_distant", short="Stable PAF Baseline #1 (t01)",
         description="PAF patient, but recording taken hours away from any episode."),
    dict(record="t09", start_min=0,  out_name="demo_stable_2",
         label_key="paf_distant", short="Stable PAF Baseline #2 (t09)",
         description="PAF patient, but recording taken hours away from any episode."),
    dict(record="t11", start_min=0,  out_name="demo_stable_3",
         label_key="paf_distant", short="Stable PAF Baseline #3 (t11)",
         description="PAF patient, but recording taken hours away from any episode."),

    # 🔴 IMMINENT AFIB ATTACKS (The final 30 seconds right before AFib triggers)
    dict(record="t02", start_min=29.5, out_name="demo_imminent_1",
         label_key="paf_imminent", short="Imminent AFib Episode #1 (t02)",
         description="PAF patient, recording ends immediately before AFib onset."),
    dict(record="t10", start_min=29.5, out_name="demo_imminent_2",
         label_key="paf_imminent", short="Imminent AFib Episode #2 (t10)",
         description="PAF patient, recording ends immediately before AFib onset."),
    dict(record="t12", start_min=29.5, out_name="demo_imminent_3",
         label_key="paf_imminent", short="Imminent AFib Episode #3 (t12)",
         description="PAF patient, recording ends immediately before AFib onset."),
]

def extract_ecg_npy(db_dir, record_name, start_min, duration_sec):
    try:
        record_path = os.path.join(db_dir, record_name)
        header = wfdb.rdheader(record_path)
        fs = header.fs

        start_samp = int(start_min * 60 * fs)
        samp_count = int(duration_sec * fs)

        record = wfdb.rdrecord(record_path, sampfrom=start_samp, sampto=start_samp + samp_count)
        signal = record.p_signal[:, 0]
        return signal, fs
    except Exception as e:
        print(f"  [error] Extraction failed for {record_name}: {e}")
        return None, None

def main():
    print(f"Creating output directory: {OUTPUT_DIR}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ok = 0
    for cfg in EXTRACTIONS:
        print(f"[{cfg['label_key']}] Generating {cfg['out_name']} from record {cfg['record']}...")
        
        hea_file = os.path.join(DB_PATH, f"{cfg['record']}.hea")
        if not os.path.exists(hea_file):
            print(f"  [skip] {cfg['record']}.hea not found in {DB_PATH}. skipping...")
            continue

        signal, fs = extract_ecg_npy(DB_PATH, cfg["record"], cfg["start_min"], SNIPPET_SEC)
        if signal is None:
            continue

        # Save binary file
        npy_path = os.path.join(OUTPUT_DIR, f"{cfg['out_name']}.npy")
        np.save(npy_path, signal)
        print(f"  → Successfully exported: {npy_path} ({len(signal)} samples)")
        ok += 1

    print(f"\nDone! Extracted {ok}/{len(EXTRACTIONS)} files successfully.")

if __name__ == "__main__":
    main()