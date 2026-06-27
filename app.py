from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.integrate import trapezoid
import antropy as ant
import os

# 1. Initialize the Web API
app = FastAPI(title="AFib Diagnostic API", version="1.0")

# Allow the future React frontend to talk to this backend
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
    print(f"⚠️ Warning: Could not load model at {MODEL_PATH}. Did you run 07_Train.py?")

# 3. Define what data the front-end will send us
class PatientData(BaseModel):
    rr_intervals: list[float]

# 4. The Clinical Math Engine (From your Extract_HRV script)
def extract_features(rr_intervals):
    if len(rr_intervals) < 10:  
        return None
        
    rr_prev = rr_intervals[:-1]
    rr_curr = rr_intervals[1:]
    
    # Ectopic Beat Domain (PACs)
    premature_beats = (rr_curr < 0.80 * rr_prev)
    pac_count = np.sum(premature_beats)
    pac_ratio = (pac_count / len(rr_intervals)) * 100
        
    # Time Domain
    rr_diff = np.diff(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_intervals)) * 100
    
    # Frequency Domain
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

    # Non-Linear Domain
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

# 5. The API Endpoints
@app.post("/predict")
def predict_afib(data: PatientData):
    """Receives raw R-R intervals, calculates features, and returns an AFib risk prediction."""
    if not model:
        raise HTTPException(status_code=500, detail="CatBoost model is not loaded on the server.")
        
    rr_array = np.array(data.rr_intervals)
    
    # 1. Extract Clinical Features
    features = extract_features(rr_array)
    if features is None:
        raise HTTPException(status_code=400, detail="Not enough R-R intervals provided for analysis.")
        
    # 2. Format for CatBoost
    df_features = pd.DataFrame([features])
    
    # 3. Predict Probability
    prob = model.predict_proba(df_features)[0][1] # Get probability of Class 1 (AFib)
    
    # 4. Apply our clinical sweet-spot threshold (0.3)
    CLINICAL_THRESHOLD = 0.3
    is_afib_imminent = bool(prob >= CLINICAL_THRESHOLD)
    
    return {
        "status": "success",
        "risk_probability": float(prob),
        "is_afib_imminent": is_afib_imminent,
        "clinical_threshold": CLINICAL_THRESHOLD,
        "biometrics": features
    }

@app.get("/ping")
def ping():
    """Used by UptimeRobot to keep the free server awake 24/7."""
    return {"status": "awake"}