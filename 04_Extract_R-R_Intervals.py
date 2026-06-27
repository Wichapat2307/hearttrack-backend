import os
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

# Configuration Paths
CLEAN_DATA_DIR = Path("PAFPDB/Cleaned")
RR_OUTPUT_DIR = Path("PAFPDB/RR_Intervals")
RR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pan_tompkins_peaks(signal, fs):
    """
    Advanced Noise-Tolerant Pan-Tompkins R-peak detector.
    Identifies heartbeat contraction intervals stably across noisy channels.
    """
    median_val = np.median(signal)
    is_inverted = np.abs(np.min(signal) - median_val) > np.abs(np.max(signal) - median_val)

    diff = np.diff(signal)
    squared = diff ** 2
    
    window_len = int(0.15 * fs) 
    integrated = np.convolve(squared, np.ones(window_len)/window_len, mode='same')
    
    # --- THE FIX: OUTLIER-PROOF THRESHOLDING ---
    min_distance = int(0.35 * fs) 
    
    # Use 99th percentile instead of np.max() to ignore extreme noise artifacts
    safe_max = np.percentile(integrated, 99)
    prom_thresh = safe_max * 0.15
    height_thresh = np.mean(integrated) + 0.2 * np.std(integrated)
    # -------------------------------------------
    
    peaks, _ = find_peaks(integrated, distance=min_distance, prominence=prom_thresh, height=height_thresh)
    
    refined_peaks = []
    search_radius = int(0.06 * fs)
    
    for p in peaks:
        start_idx = max(0, p - search_radius)
        end_idx = min(len(signal), p + search_radius)
        if start_idx < end_idx:
            window = signal[start_idx:end_idx]
            if is_inverted:
                true_r = start_idx + np.argmin(window)
            else:
                true_r = start_idx + np.argmax(window)
            refined_peaks.append(true_r)
            
    return np.array(refined_peaks)


def process_all_records():
    npz_files = list(CLEAN_DATA_DIR.glob("*_clean.npz"))
    npz_files.sort()
    
    if not npz_files:
        print(f"No cleaned files found in {CLEAN_DATA_DIR}! Run your cleaner script first.")
        return
        
    print(f"Found {len(npz_files)} records. Extracting heartbeat time intervals...")

    for file_path in npz_files:
        record_id = file_path.stem.replace("_clean", "")
        print(f"Processing heartbeat timings for: {record_id}...", end="\r")
        
        try:
            data = np.load(file_path)
            signal = data['signal']
            fs = int(data['fs'])
            
            # Use Channel 0 as the primary engine
            primary_lead_signal = signal[:, 0]
            
            # Detect peaks
            peak_indices = pan_tompkins_peaks(primary_lead_signal, fs)
            
            # Calculate RR intervals
            rr_intervals_ms = (np.diff(peak_indices) / fs) * 1000.0
            
            # Save
            output_file = RR_OUTPUT_DIR / f"{record_id}_rr.npz"
            np.savez_compressed(
                output_file,
                rr_intervals=rr_intervals_ms,
                peak_indices=peak_indices,
                fs=fs
            )
            
        except Exception as e:
            print(f"\n❌ Error extracting intervals from {record_id}: {e}")

    print(f"\nSuccess! All time intervals extracted and saved into '{RR_OUTPUT_DIR}/'")


if __name__ == "__main__":
    process_all_records()