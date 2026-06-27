import os
import wfdb
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

# Configuration
RAW_DATA_DIR = Path("PAFPDB")       # Where your current n01, p01 files are
CLEAN_DATA_DIR = Path("PAFPDB/Cleaned")  # Where clean files will go
CLEAN_DATA_DIR.mkdir(exist_ok=True)

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=128, order=3):
    """Applies a zero-phase Butterworth bandpass filter to a 1D signal array."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def clean_and_export_all():
    # Find all unique record names by checking .hea files
    hea_files = list(RAW_DATA_DIR.glob("*.hea"))
    record_ids = [f.stem for f in hea_files if not f.stem.endswith('c')]
    record_ids.sort()

    print(f"Found {len(record_ids)} baseline records to process...")

    for rid in record_ids:
        print(f"Filtering record: {rid}...", end="\r")
        try:
            # 1. Load raw record
            record = wfdb.rdrecord(str(RAW_DATA_DIR / rid))
            fs = record.fs
            num_leads = record.p_signal.shape[1]
            
            # Create an empty array to hold the clean leads
            clean_signal = np.zeros_like(record.p_signal)
            
            # 2. Filter each lead individually
            for lead_idx in range(num_leads):
                raw_lead = record.p_signal[:, lead_idx]
                clean_signal[:, lead_idx] = butter_bandpass_filter(raw_lead, fs=fs)
                
            # 3. Save as a compressed numpy file along with its metadata
            output_file = CLEAN_DATA_DIR / f"{rid}_clean.npz"
            np.savez_compressed(
                output_file, 
                signal=clean_signal, 
                fs=fs, 
                sig_name=record.sig_name
            )
        except Exception as e:
            print(f"\nError processing {rid}: {e}")

    print("\nProcessing complete! All records cleaned and exported to './cleaned_afpdb/'")

if __name__ == "__main__":
    clean_and_export_all()