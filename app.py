"""
app.py — HeartTrack AI Backend
================================
Endpoints:
  POST /predict          ← existing: batch CSV analysis
  WS   /ws/ecg           ← new: real-time ESP32 streaming
  GET  /ping             ← UptimeRobot keepalive
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import asyncio
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.integrate import trapezoid
import antropy as ant
import os
from collections import deque

# ─── 1. APP INIT ───────────────────────────────────────────────────────────────
app = FastAPI(title="HeartTrack AI API", version="2.0")

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
    print(f"⚠️  Model not found at {MODEL_PATH}: {e}")

# ─── 3. FEATURE EXTRACTION (shared by both endpoints) ──────────────────────────
def extract_features(rr_intervals: np.ndarray) -> dict | None:
    if len(rr_intervals) < 10:
        return None

    rr_prev = rr_intervals[:-1]
    rr_curr = rr_intervals[1:]

    # Ectopic beats
    premature_beats = rr_curr < 0.80 * rr_prev
    pac_count  = int(np.sum(premature_beats))
    pac_ratio  = (pac_count / len(rr_intervals)) * 100

    # Time domain
    rr_diff  = np.diff(rr_intervals)
    mean_rr  = float(np.mean(rr_intervals))
    sdnn     = float(np.std(rr_intervals, ddof=1))
    rmssd    = float(np.sqrt(np.mean(rr_diff ** 2)))
    pnn50    = float((np.sum(np.abs(rr_diff) > 50) / len(rr_intervals)) * 100)

    # Frequency domain
    rr_sec       = rr_intervals / 1000.0
    cum_time     = np.cumsum(rr_sec)
    cum_time    -= cum_time[0]
    fs           = 4.0
    t_interp     = np.arange(0, cum_time[-1], 1 / fs)

    if len(t_interp) >= 2:
        f_interp      = interp1d(cum_time, rr_sec, kind='linear', fill_value="extrapolate")
        rr_interp     = f_interp(t_interp) - np.mean(f_interp(t_interp))
        nperseg       = min(256, len(rr_interp))
        freqs, psd    = welch(rr_interp, fs=fs, nperseg=nperseg)
        lf_mask       = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask       = (freqs >  0.15) & (freqs <= 0.40)
        lf            = float(trapezoid(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
        hf            = float(trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
        lf_hf_ratio   = lf / hf if hf > 0 else 0.0
    else:
        lf = hf = lf_hf_ratio = 0.0

    # Nonlinear
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
    """Run model on extracted features and return prediction payload."""
    df = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[0][1])
    THRESHOLD = 0.3
    return {
        "risk_probability": prob,
        "is_afib_imminent": prob >= THRESHOLD,
        "clinical_threshold": THRESHOLD,
        "biometrics": features,
    }


# ─── 4. BATCH PREDICT ENDPOINT (existing CSV upload flow) ──────────────────────
class PatientData(BaseModel):
    rr_intervals: list[float]


@app.post("/predict")
def predict_afib(data: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    rr = np.array(data.rr_intervals)
    features = extract_features(rr)
    if features is None:
        raise HTTPException(status_code=400, detail="Not enough R-R intervals (need ≥ 10).")

    return {"status": "success", **run_model(features)}


# ─── 5. REAL-TIME WEBSOCKET ENDPOINT (ESP32 streaming) ─────────────────────────
#
# Protocol (JSON messages):
#
#   ESP32  → Server:  { "type": "rr",   "value": 843.5 }         ← one RR interval in ms
#   ESP32  → Server:  { "type": "ping" }                          ← keepalive
#   Server → Client:  { "type": "rr_echo",  "value": 843.5,
#                        "hr": 71.2, "buffer_size": 45 }          ← echo + instant HR
#   Server → Client:  { "type": "analysis", "risk_probability": 0.12,
#                        "is_afib_imminent": false,
#                        "biometrics": {...}, "buffer_size": 50 }  ← full analysis (every 10 new RRs)
#   Server → Client:  { "type": "error",   "message": "..." }
#
# The React frontend connects to ws://YOUR_SERVER/ws/ecg
# The ESP32 also connects to the same URL and streams RR intervals
#
# WINDOW SETTINGS
WINDOW_SIZE   = 50   # how many RR intervals to feed the model
STRIDE        = 10   # re-analyse every N new intervals (sliding window)

class StreamSession:
    """Holds per-connection state for a streaming session."""
    def __init__(self):
        self.rr_buffer: deque = deque(maxlen=WINDOW_SIZE)
        self.intervals_since_last_analysis: int = 0

sessions: dict[str, StreamSession] = {}


@app.websocket("/ws/ecg")
async def websocket_ecg(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    sessions[session_id] = StreamSession()
    session = sessions[session_id]
    print(f"🔌 WebSocket connected: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": "Invalid JSON"
                }))
                continue

            msg_type = msg.get("type")

            # ── Keepalive ping ───────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # ── New RR interval ──────────────────────────────────────────────
            if msg_type == "rr":
                value = msg.get("value")
                if not isinstance(value, (int, float)) or not (200 <= value <= 2000):
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"RR value {value} out of physiological range (200–2000 ms)"
                    }))
                    continue

                session.rr_buffer.append(float(value))
                session.intervals_since_last_analysis += 1

                # Echo immediately with instant HR
                instant_hr = round(60000.0 / value, 1)
                await websocket.send_text(json.dumps({
                    "type": "rr_echo",
                    "value": value,
                    "hr": instant_hr,
                    "buffer_size": len(session.rr_buffer),
                }))

                # Full analysis every STRIDE intervals (once buffer is big enough)
                if (len(session.rr_buffer) >= WINDOW_SIZE and
                        session.intervals_since_last_analysis >= STRIDE):
                    session.intervals_since_last_analysis = 0

                    if not model:
                        await websocket.send_text(json.dumps({
                            "type": "error", "message": "Model not loaded on server."
                        }))
                        continue

                    rr_array = np.array(list(session.rr_buffer))
                    features = extract_features(rr_array)
                    if features:
                        prediction = run_model(features)
                        await websocket.send_text(json.dumps({
                            "type": "analysis",
                            "buffer_size": len(session.rr_buffer),
                            **prediction,
                        }))

            # ── Unknown message ──────────────────────────────────────────────
            else:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"Unknown message type: {msg_type}"
                }))

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    finally:
        sessions.pop(session_id, None)


# ─── 6. KEEPALIVE ──────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"status": "awake", "model_loaded": model is not None}