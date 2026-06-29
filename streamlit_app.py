"""
HeartTrack AI — Streamlit App
==============================
Deploy to Streamlit Cloud from your HeartTrack GitHub repo.
Set main file path to: streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap
import wfdb
import os
import time
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, filtfilt, find_peaks, resample
from scipy.integrate import trapezoid
import antropy as ant

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartTrack AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DARK THEME CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #09090b; color: #f4f4f5; }
  section[data-testid="stSidebar"] { background-color: #0f0f11; border-right: 1px solid #27272a; }
  .stApp header { background-color: #09090b; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 14px 16px;
  }
  div[data-testid="metric-container"] label { color: #71717a !important; font-size: 11px !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #f4f4f5 !important; }

  /* Buttons */
  .stButton > button {
    background: #e11d48; color: white; border: none;
    border-radius: 8px; font-weight: 700;
  }
  .stButton > button:hover { background: #be123c; border: none; }

  /* Slider */
  .stSlider > div > div > div > div { background: #e11d48; }

  /* Selectbox */
  .stSelectbox > div > div { background: #18181b; border-color: #27272a; color: #f4f4f5; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #18181b; border-radius: 10px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #71717a; border-radius: 7px; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #e11d48 !important; color: white !important; }

  /* Info/success/error boxes */
  .stAlert { border-radius: 10px; }

  /* Divider */
  hr { border-color: #27272a; }

  h1, h2, h3 { color: #f4f4f5; }
  p, li { color: #a1a1aa; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/catboost_afib_model.pkl"
DB_PATH     = "PAFPDB"
TARGET_FS   = 128
WINDOW_SEC  = 10
STRIDE_SEC  = 5
SNIPPET_SEC = 30
THRESHOLD   = 0.3

DEMO_RECORDS = {
    "🟢 Healthy Control #1  (t03)":      ("t03", 0,  "normal"),
    "🟢 Healthy Control #2  (t05)":      ("t05", 0,  "normal"),
    "🟢 Healthy Control #3  (t07)":      ("t07", 0,  "normal"),
    "🟡 Stable PAF #1  (t01, min 0)":    ("t01", 0,  "paf_distant"),
    "🟡 Stable PAF #2  (t09, min 0)":    ("t09", 0,  "paf_distant"),
    "🟡 Stable PAF #3  (t11, min 0)":    ("t11", 0,  "paf_distant"),
    "🔴 Imminent AFib #1  (t02, min 27)":("t02", 27, "paf_imminent"),
    "🔴 Imminent AFib #2  (t10, min 27)":("t10", 27, "paf_imminent"),
    "🔴 Imminent AFib #3  (t12, min 27)":("t12", 27, "paf_imminent"),
}

RISK_COLORS = {
    "normal":       "#22c55e",
    "paf_distant":  "#f59e0b",
    "paf_imminent": "#ef4444",
}

FEATURE_NAMES = [
    "mean_rr","sdnn","rmssd","pnn50",
    "lf","hf","lf_hf_ratio",
    "sd1","sd2","sampen",
    "pac_count","pac_ratio",
]

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_explainer(_mdl):
    return shap.TreeExplainer(_mdl)

# ─── SIGNAL PROCESSING ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_ecg_snippet(record_name: str, start_min: int) -> tuple:
    """Load and preprocess a 30s ECG snippet from PAFPDB."""
    record_path = os.path.join(DB_PATH, record_name)
    hdr    = wfdb.rdheader(record_path)
    fs     = hdr.fs
    start  = int(start_min * 60 * fs)
    end    = min(start + int(SNIPPET_SEC * fs), hdr.sig_len)
    rec    = wfdb.rdrecord(record_path, sampfrom=start, sampto=end)
    sig    = rec.p_signal[:, 0].astype(np.float64)

    # Resample to TARGET_FS
    if fs != TARGET_FS:
        sig = resample(sig, int(len(sig) * TARGET_FS / fs))

    # High-pass filter to remove baseline wander
    b, a = butter(2, 0.5 / (TARGET_FS / 2), btype='high')
    sig  = filtfilt(b, a, sig)

    # Normalize
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig = sig / mx

    return sig, TARGET_FS


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    b, a     = butter(2, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)
    squared  = filtered ** 2
    height   = np.mean(squared) + 0.5 * np.std(squared)
    peaks, _ = find_peaks(squared, distance=int(0.2*fs), height=height)
    return peaks


def ecg_to_rr(signal: np.ndarray, fs: float):
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return np.array([]), peaks
    rr    = np.diff(peaks) / fs * 1000.0
    valid = (rr >= 200) & (rr <= 2000)
    return rr[valid], peaks


def extract_features(rr: np.ndarray):
    if len(rr) < 10:
        return None
    rr_prev = rr[:-1]; rr_curr = rr[1:]
    pac     = rr_curr < 0.80 * rr_prev
    rr_diff = np.diff(rr)
    cum     = np.cumsum(rr/1000) - np.cumsum(rr/1000)[0]
    fs      = 4.0
    t_i     = np.arange(0, cum[-1], 1/fs)
    if len(t_i) >= 2:
        fi      = interp1d(cum, rr/1000, kind='linear', fill_value='extrapolate')
        rr_i    = fi(t_i) - np.mean(fi(t_i))
        nperseg = min(256, len(rr_i))
        fr, psd = welch(rr_i, fs=fs, nperseg=nperseg)
        lf = float(trapezoid(psd[(fr>=0.04)&(fr<=0.15)], fr[(fr>=0.04)&(fr<=0.15)])) if np.any((fr>=0.04)&(fr<=0.15)) else 0.0
        hf = float(trapezoid(psd[(fr>0.15)&(fr<=0.40)], fr[(fr>0.15)&(fr<=0.40)])) if np.any((fr>0.15)&(fr<=0.40)) else 0.0
        lf_hf = lf/hf if hf > 0 else 0.0
    else:
        lf = hf = lf_hf = 0.0
    try:    sampen = float(ant.sample_entropy(rr))
    except: sampen = 0.0
    return {
        "mean_rr":     float(np.mean(rr)),
        "sdnn":        float(np.std(rr, ddof=1)),
        "rmssd":       float(np.sqrt(np.mean(rr_diff**2))),
        "pnn50":       float((np.sum(np.abs(rr_diff)>50)/len(rr))*100),
        "lf":          lf, "hf": hf, "lf_hf_ratio": lf_hf,
        "sd1":         float(np.sqrt(np.std(rr_prev-rr_curr)**2/2)),
        "sd2":         float(np.sqrt(np.std(rr_prev+rr_curr)**2/2)),
        "sampen":      sampen,
        "pac_count":   int(np.sum(pac)),
        "pac_ratio":   float((np.sum(pac)/len(rr))*100),
    }


@st.cache_data(show_spinner=False)
def run_sliding_windows(signal: np.ndarray, fs: float):
    win = int(WINDOW_SEC * fs)
    stride = int(STRIDE_SEC * fs)
    results = []
    start = 0
    while start + win <= len(signal):
        seg      = signal[start:start+win]
        rr, _    = ecg_to_rr(seg, fs)
        features = extract_features(rr) if len(rr) >= 10 else None
        results.append({
            "start_sec": round(start/fs, 1),
            "end_sec":   round((start+win)/fs, 1),
            "start_idx": start,
            "end_idx":   start+win,
            "rr":        rr,
            "features":  features,
        })
        start += stride
    return results


# ─── PLOTTING ─────────────────────────────────────────────────────────────────
def plot_ecg(signal, fs, r_peaks, windows, current_win_idx, label_key):
    t   = np.arange(len(signal)) / fs
    win = windows[current_win_idx]
    risk = None
    if win["features"] is not None:
        mdl = load_model()
        if mdl:
            df   = pd.DataFrame([win["features"]])
            risk = float(mdl.predict_proba(df)[0][1])

    color = RISK_COLORS.get(label_key, "#e11d48")
    hl_color = (
        "rgba(239,68,68,0.15)"  if risk is not None and risk >= 0.3  else
        "rgba(245,158,11,0.12)" if risk is not None and risk >= 0.15 else
        "rgba(34,197,94,0.10)"  if risk is not None else
        "rgba(100,100,100,0.08)"
    )

    fig = go.Figure()

    # Sliding window highlight
    fig.add_vrect(
        x0=win["start_sec"], x1=win["end_sec"],
        fillcolor=hl_color, layer="below", line_width=0,
    )
    fig.add_vline(x=win["start_sec"], line=dict(color=color, width=1.5, dash="dot"))
    fig.add_vline(x=win["end_sec"],   line=dict(color=color, width=1.5, dash="dot"))

    # ECG signal
    fig.add_trace(go.Scatter(
        x=t, y=signal,
        mode='lines',
        line=dict(color=color, width=1.2),
        name='ECG',
        hovertemplate='%{x:.2f}s<extra></extra>',
    ))

    # R-peak markers
    if len(r_peaks):
        rp_t = r_peaks / fs
        rp_v = signal[r_peaks]
        fig.add_trace(go.Scatter(
            x=rp_t, y=rp_v,
            mode='markers',
            marker=dict(color='#facc15', size=5, symbol='circle'),
            name='R-peaks',
            hovertemplate='R-peak %{x:.2f}s<extra></extra>',
        ))

    # Risk label on window
    if risk is not None:
        fig.add_annotation(
            x=(win["start_sec"]+win["end_sec"])/2,
            y=0.92, yref='paper',
            text=f"<b>{risk*100:.0f}% risk</b>",
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=color, borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        paper_bgcolor='#060d18', plot_bgcolor='#060d18',
        font=dict(color='#a1a1aa', size=11),
        margin=dict(l=40, r=20, t=10, b=40),
        height=220,
        showlegend=False,
        xaxis=dict(
            title='Time (s)', gridcolor='#1a2332', zerolinecolor='#1a2332',
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            title='Amplitude', gridcolor='#1a2332', zerolinecolor='#1a2332',
            showgrid=True, gridwidth=1,
        ),
    )
    return fig


def plot_risk_gauge(probability):
    if probability is None:
        probability = 0.0
    pct   = probability * 100
    color = "#ef4444" if pct >= 30 else "#f59e0b" if pct >= 15 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(color=color, size=36)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#52525b", tickfont=dict(color="#52525b", size=10)),
            bar=dict(color=color, thickness=0.7),
            bgcolor="#18181b",
            borderwidth=0,
            steps=[
                dict(range=[0, 15],  color="#052e16"),
                dict(range=[15, 30], color="#451a03"),
                dict(range=[30, 100],color="#450a0a"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=pct),
        ),
        title=dict(text="AFib Risk", font=dict(color="#71717a", size=13)),
    ))
    fig.update_layout(
        paper_bgcolor='#18181b', plot_bgcolor='#18181b',
        font=dict(color='#f4f4f5'),
        margin=dict(l=20, r=20, t=30, b=10),
        height=200,
    )
    return fig


def plot_poincare(rr: np.ndarray, label_key: str):
    if len(rr) < 2:
        return go.Figure()
    color = RISK_COLORS.get(label_key, "#e11d48")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rr[:-1], y=rr[1:],
        mode='markers',
        marker=dict(color=color, size=5, opacity=0.6,
                    line=dict(color='rgba(0,0,0,0)', width=0)),
        hovertemplate='RR(n)=%{x:.0f}ms<br>RR(n+1)=%{y:.0f}ms<extra></extra>',
    ))
    # Identity line
    mn, mx = rr.min(), rr.max()
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
        mode='lines', line=dict(color='#27272a', dash='dash', width=1), showlegend=False))
    fig.update_layout(
        paper_bgcolor='#18181b', plot_bgcolor='#18181b',
        font=dict(color='#a1a1aa', size=11),
        margin=dict(l=40, r=10, t=30, b=40),
        height=230,
        title=dict(text='Poincaré Plot', font=dict(color='#71717a', size=12), x=0.5),
        xaxis=dict(title='RR(n) ms', gridcolor='#27272a', zerolinecolor='#27272a'),
        yaxis=dict(title='RR(n+1) ms', gridcolor='#27272a', zerolinecolor='#27272a'),
        showlegend=False,
    )
    return fig


def plot_rr_series(rr: np.ndarray, label_key: str):
    color = RISK_COLORS.get(label_key, "#e11d48")
    fig = go.Figure(go.Scatter(
        y=rr, mode='lines+markers',
        line=dict(color=color, width=1.2),
        marker=dict(color=color, size=3),
        hovertemplate='Beat %{x}<br>RR=%{y:.0f}ms<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='#18181b', plot_bgcolor='#18181b',
        font=dict(color='#a1a1aa', size=11),
        margin=dict(l=40, r=10, t=30, b=40),
        height=230,
        title=dict(text='RR Interval Series', font=dict(color='#71717a', size=12), x=0.5),
        xaxis=dict(title='Beat #', gridcolor='#27272a'),
        yaxis=dict(title='RR (ms)', gridcolor='#27272a'),
    )
    return fig


def plot_shap(features: dict, model, explainer):
    df    = pd.DataFrame([features])
    sv    = explainer.shap_values(df)
    # For binary classifiers sv may be list [neg, pos]
    if isinstance(sv, list):
        sv = sv[1]
    vals  = sv[0]
    names = list(features.keys())

    # Sort by absolute impact
    order = np.argsort(np.abs(vals))[::-1][:10]
    sorted_vals  = vals[order]
    sorted_names = [names[i] for i in order]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in sorted_vals]

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:+.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line=dict(color='#52525b', width=1))
    fig.update_layout(
        paper_bgcolor='#18181b', plot_bgcolor='#18181b',
        font=dict(color='#a1a1aa', size=11),
        margin=dict(l=10, r=10, t=30, b=30),
        height=280,
        title=dict(text='SHAP — Feature Impact on Risk Score', font=dict(color='#71717a', size=12), x=0.5),
        xaxis=dict(title='SHAP value (→ increases risk)', gridcolor='#27272a', zerolinecolor='#52525b'),
        yaxis=dict(autorange='reversed'),
    )
    return fig


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🫀 HeartTrack AI")
        st.markdown("<p style='color:#71717a;font-size:13px'>PAF Detection · Clinical Dashboard</p>", unsafe_allow_html=True)
        st.divider()

        mode = st.radio("Mode", ["📂 Demo Recording", "📁 Upload CSV"], label_visibility="collapsed")
        st.divider()

        if "Demo" in mode:
            demo_label = st.selectbox("Select Recording", list(DEMO_RECORDS.keys()))
            record, start_min, label_key = DEMO_RECORDS[demo_label]
            st.markdown(f"""
            <div style='background:#18181b;border:1px solid #27272a;border-radius:8px;padding:12px;margin-top:8px;font-size:12px;color:#71717a'>
            <b style='color:#a1a1aa'>Record:</b> {record}<br>
            <b style='color:#a1a1aa'>Window start:</b> min {start_min}<br>
            <b style='color:#a1a1aa'>Duration:</b> {SNIPPET_SEC}s<br>
            <b style='color:#a1a1aa'>Analysis:</b> {WINDOW_SEC}s windows, {STRIDE_SEC}s stride
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            auto_play = st.toggle("▶ Auto-play windows", value=False)
            play_speed = st.slider("Speed (s/window)", 0.3, 3.0, 1.0, 0.1) if auto_play else 1.0

        else:
            uploaded = st.file_uploader("Upload RR Interval CSV", type=["csv"])
            label_key = "unknown"
            auto_play = False
            play_speed = 1.0

        st.divider()
        st.markdown("<p style='color:#52525b;font-size:11px'>PAFPDB · CatBoost · SHAP</p>", unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
      <span style='font-size:28px;font-weight:800;letter-spacing:-0.02em;color:#f4f4f5'>HeartTrack AI</span>
      <span style='font-size:11px;font-weight:600;color:#71717a;padding:2px 10px;border:1px solid #27272a;border-radius:4px'>Clinical Dashboard</span>
    </div>
    <p style='color:#71717a;margin-bottom:20px;font-size:13px'>
    Paroxysmal Atrial Fibrillation · Real ECG from PAFPDB · CatBoost + SHAP Explainability
    </p>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("⚠️ Model not found at `models/catboost_afib_model.pkl`. Run `07_Train.py` first.")
        return
    explainer = load_explainer(model)

    # ── DEMO MODE ─────────────────────────────────────────────────────────────
    if "Demo" in mode:
        db_ok = os.path.exists(os.path.join(DB_PATH, f"{record}.hea"))
        if not db_ok:
            st.error(f"PAFPDB not found at `{DB_PATH}/`. Make sure the folder is in your repo root.")
            return

        with st.spinner(f"Loading {record} from PAFPDB…"):
            signal, fs = load_ecg_snippet(record, start_min)

        r_peaks = detect_r_peaks(signal, fs)

        with st.spinner("Running sliding window analysis…"):
            windows = run_sliding_windows(signal, fs)

        n_windows = len(windows)
        if n_windows == 0:
            st.error("No windows extracted — signal may be too short.")
            return

        # ── Window selector ───────────────────────────────────────────────────
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            win_idx = st.slider(
                f"Analysis Window  (1–{n_windows})",
                min_value=0, max_value=n_windows-1,
                value=st.session_state.get("win_idx", 0),
                format="Window %d",
                key="win_slider",
            )
        with col_ctrl2:
            w = windows[win_idx]
            st.markdown(f"""
            <div style='background:#18181b;border:1px solid #27272a;border-radius:8px;padding:10px;font-size:12px;color:#71717a;margin-top:8px'>
            ⏱ {w['start_sec']}s → {w['end_sec']}s<br>
            💓 {len(w['rr'])} beats detected
            </div>
            """, unsafe_allow_html=True)

        # ── ECG Plot ──────────────────────────────────────────────────────────
        st.plotly_chart(
            plot_ecg(signal, fs, r_peaks, windows, win_idx, label_key),
            use_container_width=True, config={"displayModeBar": False},
        )

        # ── Run model on current window ───────────────────────────────────────
        win      = windows[win_idx]
        features = win["features"]
        risk     = None
        if features:
            df   = pd.DataFrame([features])
            risk = float(model.predict_proba(df)[0][1])

        # ── Risk + Metrics row ────────────────────────────────────────────────
        col_gauge, col_metrics = st.columns([1, 2])

        with col_gauge:
            st.plotly_chart(plot_risk_gauge(risk), use_container_width=True,
                            config={"displayModeBar": False})
            if risk is not None:
                is_afib = risk >= THRESHOLD
                if is_afib:
                    st.error(f"⚠️ **HIGH RISK** — PAF Detected ({risk*100:.1f}%)")
                elif risk >= 0.15:
                    st.warning(f"🟡 **ELEVATED** — Monitor closely ({risk*100:.1f}%)")
                else:
                    st.success(f"✅ **NORMAL** — Healthy sinus rhythm ({risk*100:.1f}%)")
            else:
                st.info("Not enough beats in this window for analysis.")

        with col_metrics:
            if features:
                st.markdown("**HRV Metrics**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("SDNN",     f"{features['sdnn']:.1f} ms")
                m2.metric("RMSSD",    f"{features['rmssd']:.1f} ms")
                m3.metric("pNN50",    f"{features['pnn50']:.1f}%")
                m4.metric("Mean RR",  f"{features['mean_rr']:.0f} ms")
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("PAC Count",  str(features['pac_count']),
                          delta="⚠️ High" if features['pac_count'] > 3 else None,
                          delta_color="inverse")
                m6.metric("LF/HF",    f"{features['lf_hf_ratio']:.2f}")
                m7.metric("SampEn",   f"{features['sampen']:.3f}")
                m8.metric("SD1",      f"{features['sd1']:.1f} ms")

        st.divider()

        # ── Bottom row: SHAP + Poincaré + RR series ───────────────────────────
        col_shap, col_poincare, col_rr = st.columns(3)

        with col_shap:
            if features and risk is not None:
                st.plotly_chart(plot_shap(features, model, explainer),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown("""
                <p style='font-size:11px;color:#52525b;text-align:center'>
                🔴 Red = increases AFib risk &nbsp;|&nbsp; 🟢 Green = decreases risk
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info("SHAP unavailable — not enough beats in this window.")

        with col_poincare:
            if len(win["rr"]) > 1:
                st.plotly_chart(plot_poincare(win["rr"], label_key),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown("""
                <p style='font-size:11px;color:#52525b;text-align:center'>
                Tight cluster = regular rhythm · Scattered = irregular (AFib)
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info("Not enough RR intervals for Poincaré plot.")

        with col_rr:
            if len(win["rr"]) > 0:
                st.plotly_chart(plot_rr_series(win["rr"], label_key),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No RR intervals detected in this window.")

        # ── Auto-play ─────────────────────────────────────────────────────────
        if auto_play:
            next_idx = (win_idx + 1) % n_windows
            time.sleep(play_speed)
            st.session_state["win_idx"] = next_idx
            st.rerun()

    # ── UPLOAD CSV MODE ───────────────────────────────────────────────────────
    else:
        if not uploaded:
            st.info("Upload a CSV file containing RR intervals in milliseconds (one per line).")
            return

        raw  = uploaded.read().decode()
        vals = [float(x) for line in raw.splitlines()
                for x in line.split(',')
                if x.strip() and not x.strip().startswith('#')
                and x.strip() != 'rr_ms'
                and _is_number(x)]
        rr   = np.array([v for v in vals if 200 <= v <= 2000])

        if len(rr) < 10:
            st.error("Not enough valid RR intervals (need ≥ 10, range 200–2000 ms).")
            return

        st.success(f"✅ Loaded {len(rr)} RR intervals.")
        features = extract_features(rr)
        if features is None:
            st.error("Feature extraction failed.")
            return

        df   = pd.DataFrame([features])
        risk = float(model.predict_proba(df)[0][1])

        col_gauge, col_metrics = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(plot_risk_gauge(risk), use_container_width=True,
                            config={"displayModeBar": False})
            if risk >= THRESHOLD:
                st.error(f"⚠️ **HIGH RISK** — PAF Detected ({risk*100:.1f}%)")
            elif risk >= 0.15:
                st.warning(f"🟡 **ELEVATED** ({risk*100:.1f}%)")
            else:
                st.success(f"✅ **NORMAL** ({risk*100:.1f}%)")

        with col_metrics:
            st.markdown("**HRV Metrics**")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("SDNN",    f"{features['sdnn']:.1f} ms")
            m2.metric("RMSSD",   f"{features['rmssd']:.1f} ms")
            m3.metric("pNN50",   f"{features['pnn50']:.1f}%")
            m4.metric("Mean RR", f"{features['mean_rr']:.0f} ms")

        st.divider()
        col_shap2, col_pc2, col_rr2 = st.columns(3)
        with col_shap2:
            st.plotly_chart(plot_shap(features, model, explainer),
                            use_container_width=True, config={"displayModeBar": False})
        with col_pc2:
            st.plotly_chart(plot_poincare(rr, "unknown"),
                            use_container_width=True, config={"displayModeBar": False})
        with col_rr2:
            st.plotly_chart(plot_rr_series(rr, "unknown"),
                            use_container_width=True, config={"displayModeBar": False})


def _is_number(s):
    try: float(s); return True
    except: return False


if __name__ == "__main__":
    main()