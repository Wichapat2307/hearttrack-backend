"""
app.py — HeartTrack AI Backend v3
===================================
Endpoints:
  POST /predict          ← batch RR intervals (existing)
  POST /analyze_ecg      ← NEW: raw ECG .npy → sliding window analysis
  WS   /ws/ecg           ← real-time ESP32 streaming
  GET  /ping             ← keepalive
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import asyncio
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.integrate import trapezoid
import antropy as ant
import os
from collections import deque
from io import BytesIO

# ─── 1. APP INIT ───────────────────────────────────────────────────────────────
app = FastAPI(title="HeartTrack AI API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 2. MODEL LOAD ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/catboost_afib_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ CatBoost model loaded.")
except Exception as e:
    model = None
    print(f"⚠️  Model not found: {e}")

# ─── 3. FEATURE EXTRACTION ─────────────────────────────────────────────────────
def extract_features(rr_intervals: np.ndarray) -> dict | None:
    if len(rr_intervals) < 10:
        return None
    rr_prev = rr_intervals[:-1]
    rr_curr = rr_intervals[1:]
    premature   = rr_curr < 0.80 * rr_prev
    pac_count   = int(np.sum(premature))
    pac_ratio   = (pac_count / len(rr_intervals)) * 100
    rr_diff     = np.diff(rr_intervals)
    mean_rr     = float(np.mean(rr_intervals))
    sdnn        = float(np.std(rr_intervals, ddof=1))
    rmssd       = float(np.sqrt(np.mean(rr_diff ** 2)))
    pnn50       = float((np.sum(np.abs(rr_diff) > 50) / len(rr_intervals)) * 100)
    rr_sec      = rr_intervals / 1000.0
    cum_time    = np.cumsum(rr_sec) - np.cumsum(rr_sec)[0]
    fs          = 4.0
    t_interp    = np.arange(0, cum_time[-1], 1 / fs)
    if len(t_interp) >= 2:
        f_i         = interp1d(cum_time, rr_sec, kind='linear', fill_value="extrapolate")
        rr_i        = f_i(t_interp) - np.mean(f_i(t_interp))
        nperseg     = min(256, len(rr_i))
        freqs, psd  = welch(rr_i, fs=fs, nperseg=nperseg)
        lf_mask     = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask     = (freqs >  0.15) & (freqs <= 0.40)
        lf          = float(trapezoid(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
        hf          = float(trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
        lf_hf_ratio = lf / hf if hf > 0 else 0.0
    else:
        lf = hf = lf_hf_ratio = 0.0
    sd1 = float(np.sqrt(np.std(rr_prev - rr_curr) ** 2 / 2))
    sd2 = float(np.sqrt(np.std(rr_prev + rr_curr) ** 2 / 2))
    try:
        sampen = float(ant.sample_entropy(rr_intervals))
    except Exception:
        sampen = 0.0
    return {
        "mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50,
        "lf": lf, "hf": hf, "lf_hf_ratio": lf_hf_ratio,
        "sd1": sd1, "sd2": sd2, "sampen": sampen,
        "pac_count": pac_count, "pac_ratio": pac_ratio,
    }


def run_model(features: dict) -> dict:
    df   = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "risk_probability": prob,
        "is_afib_imminent": prob >= 0.3,
        "clinical_threshold": 0.3,
        "biometrics": features,
    }


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Simple R-peak detector using scipy find_peaks."""
    # Bandpass 5–15 Hz to isolate QRS
    b, a       = butter(2, [5 / (fs/2), 15 / (fs/2)], btype='band')
    filtered   = filtfilt(b, a, signal)
    squared    = filtered ** 2
    # Min distance = 200ms refractory period
    min_dist   = int(0.2 * fs)
    height     = np.mean(squared) + 0.5 * np.std(squared)
    peaks, _   = find_peaks(squared, distance=min_dist, height=height)
    return peaks


def ecg_to_rr_ms(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rr_ms array, r_peak_sample_indices)."""
    peaks  = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return np.array([]), peaks
    rr_ms  = np.diff(peaks) / fs * 1000.0
    # Physiological filter
    valid  = (rr_ms >= 200) & (rr_ms <= 2000)
    return rr_ms[valid], peaks


# ─── 4. BATCH /predict (existing) ─────────────────────────────────────────────
class PatientData(BaseModel):
    rr_intervals: list[float]


@app.post("/predict")
def predict_afib(data: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    rr       = np.array(data.rr_intervals)
    features = extract_features(rr)
    if features is None:
        raise HTTPException(status_code=400, detail="Not enough RR intervals (need ≥ 10).")
    return {"status": "success", **run_model(features)}


# ─── 5. NEW /analyze_ecg — NPY upload → sliding window results ────────────────
#
# Request:  multipart/form-data  with field "file" = .npy binary
#           + form field "fs" = sampling rate (default 128)
#           + form field "window_sec" = window size in seconds (default 10)
#           + form field "stride_sec" = stride in seconds (default 5)
#
# Response JSON:
# {
#   "fs": 128,
#   "n_samples": 3840,
#   "signal": [0.12, 0.15, ...],        ← downsampled to max 1000 pts for plotting
#   "signal_fs": 64,                     ← actual fs of the downsampled signal
#   "r_peaks": [45, 173, ...],           ← R-peak indices in DOWNSAMPLED signal
#   "windows": [
#     {
#       "window_idx": 0,
#       "start_sample": 0,               ← in DOWNSAMPLED signal
#       "end_sample": 640,
#       "start_sec": 0.0,
#       "end_sec": 10.0,
#       "risk_probability": 0.12,
#       "is_afib_imminent": false,
#       "biometrics": { ... }
#     }, ...
#   ]
# }

@app.post("/analyze_ecg")
async def analyze_ecg(
    file:       UploadFile = File(...),
    fs:         float = 128.0,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Load NPY
    raw = await file.read()
    try:
        signal = np.load(BytesIO(raw)).astype(np.float64)
        if signal.ndim > 1:
            signal = signal[:, 0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read NPY file: {e}")

    n_samples   = len(signal)
    win_samples = int(window_sec * fs)
    str_samples = int(stride_sec * fs)

    if n_samples < win_samples:
        raise HTTPException(status_code=400, detail=f"Signal too short ({n_samples} samples < {win_samples} needed).")

    # Downsample signal for frontend plotting (max 1500 points)
    ds_factor  = max(1, int(n_samples / 1500))
    sig_ds     = signal[::ds_factor].tolist()
    fs_ds      = fs / ds_factor

    # Detect R-peaks on full signal, then map to downsampled coords
    _, r_peaks_full = ecg_to_rr_ms(signal, fs)
    r_peaks_ds      = (r_peaks_full / ds_factor).astype(int).tolist()

    # Sliding window analysis
    windows = []
    idx     = 0
    start   = 0
    while start + win_samples <= n_samples:
        end       = start + win_samples
        segment   = signal[start:end]
        rr_ms, _  = ecg_to_rr_ms(segment, fs)

        window_result = {
            "window_idx":   idx,
            "start_sample": int(start / ds_factor),
            "end_sample":   int(end   / ds_factor),
            "start_sec":    round(start / fs, 2),
            "end_sec":      round(end   / fs, 2),
        }

        if len(rr_ms) >= 10:
            features = extract_features(rr_ms)
            if features:
                pred = run_model(features)
                window_result.update(pred)
            else:
                window_result.update({"risk_probability": None, "is_afib_imminent": None, "biometrics": None})
        else:
            window_result.update({"risk_probability": None, "is_afib_imminent": None, "biometrics": None})

        windows.append(window_result)
        start += str_samples
        idx   += 1

    return {
        "status":     "success",
        "fs":         fs,
        "fs_ds":      fs_ds,
        "n_samples":  n_samples,
        "signal":     sig_ds,
        "r_peaks":    r_peaks_ds,
        "windows":    windows,
    }


# ─── 6. WEBSOCKET /ws/ecg ──────────────────────────────────────────────────────
WINDOW_SIZE = 50
STRIDE      = 10

class StreamSession:
    def __init__(self):
        self.rr_buffer = deque(maxlen=WINDOW_SIZE)
        self.since_analysis = 0

sessions: dict = {}


@app.websocket("/ws/ecg")
async def websocket_ecg(websocket: WebSocket):
    await websocket.accept()
    sid     = str(id(websocket))
    session = StreamSession()
    sessions[sid] = session

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg.get("type") == "rr":
                value = msg.get("value")
                if not isinstance(value, (int, float)) or not (200 <= value <= 2000):
                    await websocket.send_text(json.dumps({"type": "error", "message": f"RR {value} out of range"}))
                    continue

                session.rr_buffer.append(float(value))
                session.since_analysis += 1

                await websocket.send_text(json.dumps({
                    "type": "rr_echo",
                    "value": value,
                    "hr": round(60000.0 / value, 1),
                    "buffer_size": len(session.rr_buffer),
                }))

                if len(session.rr_buffer) >= WINDOW_SIZE and session.since_analysis >= STRIDE:
                    session.since_analysis = 0
                    if model:
                        rr_arr   = np.array(list(session.rr_buffer))
                        features = extract_features(rr_arr)
                        if features:
                            pred = run_model(features)
                            await websocket.send_text(json.dumps({
                                "type": "analysis",
                                "buffer_size": len(session.rr_buffer),
                                **pred,
                            }))

    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(sid, None)


# ─── 7. KEEPALIVE ──────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"status": "awake", "model_loaded": model is not None}