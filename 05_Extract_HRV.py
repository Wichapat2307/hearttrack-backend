import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.integrate import trapezoid
import antropy as ant

# Paths
RR_DIR = Path("PAFPDB/RR_Intervals")
OUTPUT_CSV = "paf_hrv_dataset.csv"

# 5 minutes in milliseconds
WINDOW_MS = 5 * 60 * 1000 

def determine_label(record_id):
    """
    Class 0 (Safe): All 'n' records and odd 'p' records (p01, p03).
    Class 1 (Pre-AFib): Even 'p' records (p02, p04).
    """
    if record_id.startswith('n'):
        return 0
    elif record_id.startswith('p'):
        record_num = int(record_id.replace('p', ''))
        if record_num % 2 == 0:
            return 1 
        else:
            return 0 
    return 0

def extract_features(rr_intervals):
    """Calculates Time, Frequency, Non-Linear, and Ectopic HRV features."""
    if len(rr_intervals) < 10:  
        return None
        
    # --- 1. ECTOPIC BEAT DOMAIN (NEW!) ---
    # Premature Atrial Contractions (PACs) are a primary trigger for AFib.
    # Clinically defined as an R-R interval that is < 80% of the preceding interval.
    rr_prev = rr_intervals[:-1]
    rr_curr = rr_intervals[1:]
    
    # Find beats that occur at least 20% earlier than expected
    premature_beats = (rr_curr < 0.80 * rr_prev)
    pac_count = np.sum(premature_beats)
    pac_ratio = (pac_count / len(rr_intervals)) * 100
        
    # --- 2. TIME DOMAIN ---
    rr_diff = np.diff(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    
    nn50 = np.sum(np.abs(rr_diff) > 50)
    pnn50 = (nn50 / len(rr_intervals)) * 100
    
    # --- 3. FREQUENCY DOMAIN ---
    rr_seconds = rr_intervals / 1000.0
    cumulative_time = np.cumsum(rr_seconds)
    cumulative_time -= cumulative_time[0] 
    
    fs = 4.0  
    t_interpolated = np.arange(0, cumulative_time[-1], 1 / fs)
    
    if len(t_interpolated) >= 2:
        f_interp = interp1d(cumulative_time, rr_seconds, kind='linear', fill_value="extrapolate")
        rr_interpolated = f_interp(t_interpolated)
        rr_interpolated -= np.mean(rr_interpolated)
        
        nperseg = min(256, len(rr_interpolated))
        frequencies, psd = welch(rr_interpolated, fs=fs, nperseg=nperseg)
        
        lf_mask = (frequencies >= 0.04) & (frequencies <= 0.15)
        hf_mask = (frequencies > 0.15) & (frequencies <= 0.40)
        
        lf = trapezoid(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0.0
        hf = trapezoid(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0.0
        lf_hf_ratio = lf / hf if hf > 0 else 0.0
    else:
        lf, hf, lf_hf_ratio = 0.0, 0.0, 0.0

    # --- 4. NON-LINEAR / CHAOS DOMAIN ---
    # SD1 measures short-term beat-to-beat variability (parasympathetic)
    sd1 = np.sqrt(np.std(rr_prev - rr_curr) ** 2 / 2)
    # SD2 measures long-term continuous variability (sympathetic + parasympathetic)
    sd2 = np.sqrt(np.std(rr_prev + rr_curr) ** 2 / 2)
    
    # Sample Entropy (Measures the unpredictability/chaos of the heartbeats)
    try:
        sampen = ant.sample_entropy(rr_intervals)
    except:
        sampen = 0.0 
        
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'lf': lf,
        'hf': hf,
        'lf_hf_ratio': lf_hf_ratio,
        'sd1': sd1,
        'sd2': sd2,
        'sampen': sampen,
        'pac_count': pac_count,   # New!
        'pac_ratio': pac_ratio    # New!
    }

def build_dataset():
    npz_files = list(RR_DIR.glob("*_rr.npz"))
    if not npz_files:
        print("No RR interval files found!")
        return

    dataset_rows = []
    STEP_MS = 1 * 60 * 1000  
    
    for file_path in npz_files:
        record_id = file_path.stem.replace("_rr", "")
        label = determine_label(record_id)
        
        data = np.load(file_path)
        rr_intervals = data['rr_intervals']
        
        if len(rr_intervals) < 10:
            continue
            
        cumulative_time = np.cumsum(rr_intervals)
        total_time = cumulative_time[-1]
        start_time = 0
        window_idx = 0
        
        while (start_time + WINDOW_MS) <= total_time:
            end_time = start_time + WINDOW_MS
            window_mask = (cumulative_time >= start_time) & (cumulative_time < end_time)
            window_rr = rr_intervals[window_mask]
            
            features = extract_features(window_rr)
            if features is not None:
                row = {'record_id': record_id, 'window_idx': window_idx, **features, 'label': label}
                dataset_rows.append(row)
                
            start_time += STEP_MS 
            window_idx += 1
            
        # Ensure we capture the final 5 minutes
        final_start_time = total_time - WINDOW_MS
        if final_start_time >= 0:
            window_mask = (cumulative_time >= final_start_time) & (cumulative_time <= total_time)
            window_rr = rr_intervals[window_mask]
            features = extract_features(window_rr)
            if features is not None:
                row = {'record_id': record_id, 'window_idx': 999, **features, 'label': label}
                dataset_rows.append(row)

        print(f"Processed {record_id} -> {window_idx + 1} windows extracted.", end="\r")

    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n\nDataset successfully compiled with Ectopic/PAC features! Saved to: {OUTPUT_CSV}")
    print(f"Total Overlapping Windows: {len(df)}")
    if not df.empty:
        print(f"Class 0 (Normal): {len(df[df['label'] == 0])}")
        print(f"Class 1 (Pre-AFib): {len(df[df['label'] == 1])}")

if __name__ == "__main__":
    build_dataset()