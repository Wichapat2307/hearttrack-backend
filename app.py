from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.integrate import trapezoid
import antropy as ant
import io

# 1. Initialize the Web API
app = FastAPI(title="HeartTrack E2E Diagnostic API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the trained CatBoost Model
MODEL_PATH = "models/catboost_afib_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ CatBoost Clinical Model loaded successfully.")
except Exception as e:
    model = None
    print(f"⚠️ Warning: Could not load model at {MODEL_PATH}.")

# 3. QRS Detection (Raw ECG -> Heartbeats)
def extract_rr_from_ecg(ecg_signal, fs=128.0):
    """
    Applies a Pan-Tompkins style algorithm to detect R-peaks in raw ECG.
    """
    # Remove NaN values if any exist
    ecg_signal = np.nan_to_num(ecg_signal)
    
    # 1. Bandpass filter to isolate QRS complexes (0.5 - 40 Hz)
    b, a = butter(1, [0.5, 40.0], btype='bandpass', fs=fs)
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 2. Derivative & Squaring to amplify the peaks
    diff_ecg = np.diff(filtered_ecg)
    squared = diff_ecg ** 2
    
    # 3. Moving average integration to create solid peak envelopes
    window = int(0.12 * fs) # 120ms window
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')
    
    # 4. Find peaks (minimum distance of ~400ms to avoid double-counting T-waves)
    # Height threshold adapts to the patient's specific signal strength
    height_threshold = np.max(integrated) * 0.15 
    min_dist = int(fs * 0.4) 
    
    peaks, _ = find_peaks(integrated, distance=min_dist, height=height_threshold)
    
    # Calculate R-R intervals in milliseconds
    rr_intervals = np.diff(peaks) * (1000.0 / fs)
    return rr_intervals

# 4. The Clinical Math Engine (HRV & PACs)
def extract_features(rr_intervals):
    if len(rr_intervals) < 10:  
        return None
        
    rr_prev = rr_intervals[:-1]
    rr_curr = rr_intervals[1:]
    
    premature_beats = (rr_curr < 0.80 * rr_prev)
    pac_count = np.sum(premature_beats)
    pac_ratio = (pac_count / len(rr_intervals)) * 100
        
    rr_diff = np.diff(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_intervals)) * 100
    
    rr_seconds = rr_intervals / 1000.0
    cumulative_time = np.cumsum(rr_seconds)
    cumulative_time -= cumulative_time[0] 
    
    fs_hrv = 4.0  
    t_interpolated = np.arange(0, cumulative_time[-1], 1 / fs_hrv)
    
    if len(t_interpolated) >= 2:
        f_interp = interp1d(cumulative_time, rr_seconds, kind='linear', fill_value="extrapolate")
        rr_interpolated = f_interp(t_interpolated)
        rr_interpolated -= np.mean(rr_interpolated)
        
        nperseg = min(256, len(rr_interpolated))
        frequencies, psd = welch(rr_interpolated, fs=fs_hrv, nperseg=nperseg)
        
        lf_mask = (frequencies >= 0.04) & (frequencies <= 0.15)
        hf_mask = (frequencies > 0.15) & (frequencies <= 0.40)
        
        lf = trapezoid(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0.0
        hf = trapezoid(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0.0
        lf_hf_ratio = lf / hf if hf > 0 else 0.0
    else:
        lf, hf, lf_hf_ratio = 0.0, 0.0, 0.0

    sd1 = np.sqrt(np.std(rr_prev - rr_curr) ** 2 / 2)
    sd2 = np.sqrt(np.std(rr_prev + rr_curr) ** 2 / 2)
    try:
        sampen = ant.sample_entropy(rr_intervals)
    except:
        sampen = 0.0 
        
    return {
        'mean_rr': float(mean_rr), 'sdnn': float(sdnn), 'rmssd': float(rmssd), 'pnn50': float(pnn50),
        'lf': float(lf), 'hf': float(hf), 'lf_hf_ratio': float(lf_hf_ratio),
        'sd1': float(sd1), 'sd2': float(sd2), 'sampen': float(sampen),
        'pac_count': int(pac_count), 'pac_ratio': float(pac_ratio)
    }

# 5. NEW ENDPOINT: Raw Signal Processing
@app.post("/analyze_ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    """Receives a raw .npy ECG file, finds heartbeats, extracts HRV, and predicts AFib."""
    if not model:
        raise HTTPException(status_code=500, detail="CatBoost model is offline.")
    
    try:
        # Read the binary .npy file directly from memory
        content = await file.read()
        ecg_signal = np.load(io.BytesIO(content))
        
        # 1. Process Signal & Find Beats
        # PAFPDB is sampled at 128 Hz
        rr_intervals = extract_rr_from_ecg(ecg_signal, fs=128.0)
        
        # 2. Extract Clinical Features
        features = extract_features(rr_intervals)
        if features is None:
            raise HTTPException(status_code=400, detail="Could not detect enough heartbeats.")
            
        # 3. Predict Probability
        df_features = pd.DataFrame([features])
        prob = model.predict_proba(df_features)[0][1]
        
        CLINICAL_THRESHOLD = 0.3
        is_afib_imminent = bool(prob >= CLINICAL_THRESHOLD)
        
        # 4. Prepare data for the Frontend Live Plot
        # We downsample the signal heavily so the web browser doesn't freeze.
        # Taking roughly 10 seconds of data (1280 points), downsampled to ~300 points for smooth web rendering.
        plot_snippet = ecg_signal[1000:2280] # Grab 10 seconds from the middle
        plot_snippet_downsampled = plot_snippet[::4].tolist() # Take every 4th point
        
        return {
            "status": "success",
            "risk_probability": float(prob),
            "is_afib_imminent": is_afib_imminent,
            "clinical_threshold": CLINICAL_THRESHOLD,
            "biometrics": features,
            "live_plot_data": plot_snippet_downsampled
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NPY file: {str(e)}")

@app.get("/ping")
def ping():
    return {"status": "awake"}