"""
HeartTrack AI — Streamlit App v2
Fresh Clinical UI with ESP32 Wi‑Fi Streaming
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import wfdb
import os
import time
import math
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, filtfilt, find_peaks, resample
from scipy.integrate import trapezoid
import antropy as ant

# ─── Optional ESP32 networking ──────────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartTrack AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FRESH DESIGN TOKENS ──────────────────────────────────────────────────────
# Clean, vibrant palette with a modern clinical feel
BG       = "#f0f7fa"   # soft icy blue background
BG2      = "#ffffff"   # pure white for cards/sidebar
BG3      = "#ffffff"
TINT     = "#e3f0f7"   # light aqua tint for headers
BORDER   = "#cde0ea"
TEXT     = "#1a2d3c"
TEXT2    = "#4b6a80"
TEXT3    = "#8ba6bb"
# Risk colours – more vivid but still tasteful
CRIMSON  = "#d96c7a"   # warm red
CRIMSON2 = "#c05b68"
AMBER    = "#d9a86c"   # golden amber
EMERALD  = "#45b08c"   # fresh green
# Primary accent – a lively teal/cyan
SAPPHIRE = "#2a9d8f"   # teal green
SAPPHIRE_DK = "#21867a"
GOLD     = "#f4a261"   # warm gold for R-peaks
ECG_COLOR = "#00b4d8"  # bright cyan for ECG trace
ECG_GLOW  = "rgba(0,180,216,0.15)"  # glow effect

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Source+Serif+4:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, .stApp {{
    background: {BG};
    color: {TEXT};
    font-family: 'Inter', sans-serif;
  }}

  /* Prevent dimming during reruns */
  .stApp, .main, .block-container,
  div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"],
  div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="element-container"],
  .element-container, [data-stale="true"] {{
    opacity: 1 !important;
    transition: none !important;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {BG2};
    border-right: 1px solid {BORDER};
    box-shadow: 2px 0 16px rgba(0,0,0,0.02);
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* Header */
  .stApp header {{ background: {BG} !important; }}
  .block-container {{
    padding: 1.8rem 2.5rem 3rem !important;
    max-width: 1440px;
  }}

  /* Hide Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Focus states */
  button:focus-visible, input:focus-visible, [role="radio"]:focus-visible {{
    outline: 2px solid {SAPPHIRE} !important;
    outline-offset: 2px !important;
  }}

  /* ── Metric cards ────────────────────────────────────────────────────────── */
  div[data-testid="metric-container"] {{
    background: {BG3};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    transition: all 0.2s ease;
  }}
  div[data-testid="metric-container"]:hover {{
    border-color: #b0d0e0;
    box-shadow: 0 4px 16px rgba(42,157,143,0.08);
    transform: translateY(-2px);
  }}
  div[data-testid="metric-container"] label,
  div[data-testid="stMetric"] label,
  div[data-testid="stMetricLabel"] {{
    color: {TEXT2} !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    opacity: 1 !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricValue"],
  div[data-testid="stMetric"] [data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    opacity: 1 !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricDelta"],
  div[data-testid="stMetric"] [data-testid="stMetricDelta"],
  div[data-testid="stMetricDelta"] {{
    font-size: 11px !important;
    opacity: 1 !important;
  }}
  div[data-testid="metric-container"] *,
  div[data-testid="stMetric"] * {{
    opacity: 1 !important;
  }}

  /* ── Slider ──────────────────────────────────────────────────────────────── */
  .stSlider [data-baseweb="slider"] {{ padding: 0 !important; }}
  .stSlider [data-baseweb="thumb"] {{
    background: {SAPPHIRE} !important;
    border: 3px solid white !important;
    box-shadow: 0 2px 10px rgba(42,157,143,0.4) !important;
  }}
  .stSlider [data-baseweb="track-fill"] {{ background: {SAPPHIRE} !important; }}

  /* ── Buttons ─────────────────────────────────────────────────────────────── */
  .stButton > button {{
    background: {SAPPHIRE};
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 24px;
    transition: all 0.2s;
    box-shadow: 0 2px 10px rgba(42,157,143,0.25);
  }}
  .stButton > button:hover {{
    background: {SAPPHIRE_DK};
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(42,157,143,0.30);
  }}

  /* ── Selectbox ───────────────────────────────────────────────────────────── */
  .stSelectbox [data-baseweb="select"] > div {{
    background: {BG3} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
  }}

  /* ── Radio ───────────────────────────────────────────────────────────────── */
  .stRadio [data-testid="stWidgetLabel"] {{
    color: {TEXT2} !important;
    font-size: 12px !important;
  }}
  .stRadio label {{ color: {TEXT} !important; font-size: 13px !important; }}

  /* ── Toggle ──────────────────────────────────────────────────────────────── */
  .stToggle label {{ color: {TEXT} !important; font-size: 13px !important; }}
  .stToggle [data-baseweb="checkbox"] div[aria-checked="true"] {{
    background: {EMERALD} !important;
  }}

  /* ── Alerts ──────────────────────────────────────────────────────────────── */
  div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border-left: 4px solid {SAPPHIRE} !important;
  }}

  /* ── Divider ────────────────────────────────────────────────────────────── */
  hr {{
    border-color: {BORDER} !important;
    margin: 1.5rem 0 !important;
  }}

  /* ── Tabs ────────────────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {TINT};
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT2};
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {SAPPHIRE} !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(42,157,143,0.25);
  }}

  /* ── Scrollbar ───────────────────────────────────────────────────────────── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BG2}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

  /* ── Custom Cards ───────────────────────────────────────────────────────── */
  .ht-card {{
    background: {BG3};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    transition: box-shadow 0.2s;
  }}
  .ht-card:hover {{
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
  }}

  .ht-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 30px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .ht-badge::before {{
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
  }}
  .ht-badge-normal   {{ background: #e2f3ec; color: #45b08c; }}
  .ht-badge-distant  {{ background: #f8efe4; color: #d9a86c; }}
  .ht-badge-imminent {{ background: #fae8eb; color: #d96c7a; }}

  /* ── Risk Banner ────────────────────────────────────────────────────────── */
  .risk-banner {{
    border-radius: 12px;
    padding: 16px 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 600;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
  }}
  .risk-high    {{ background: #fae8eb; border: 1px solid #f0d3d8; color: #d96c7a; }}
  .risk-medium  {{ background: #f8efe4; border: 1px solid #eedbc8; color: #d9a86c; }}
  .risk-low     {{ background: #e2f3ec; border: 1px solid #c8e4d8; color: #45b08c; }}
  .risk-unknown {{ background: {TINT}; border: 1px solid {BORDER}; color: {TEXT2}; }}

  /* ── Eyebrow header ─────────────────────────────────────────────────────── */
  .ht-eyebrow {{
    font-size: 10.5px;
    font-weight: 700;
    color: {SAPPHIRE_DK};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
  }}
  .ht-eyebrow::before {{
    content: '';
    width: 18px;
    height: 2px;
    background: {SAPPHIRE};
    display: inline-block;
    border-radius: 2px;
  }}

  /* ── Page header ────────────────────────────────────────────────────────── */
  .ht-page-header {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 4px;
  }}
  .ht-page-header h1 {{
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {TEXT};
    font-family: 'Source Serif 4', serif;
    margin: 0;
  }}
  .ht-page-header .badge {{
    font-size: 10px;
    font-weight: 700;
    color: {SAPPHIRE_DK};
    background: {TINT};
    padding: 5px 14px;
    border-radius: 30px;
    border: 1px solid {BORDER};
    letter-spacing: 0.08em;
  }}

  /* ── Footer ─────────────────────────────────────────────────────────────── */
  .ht-footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid {BORDER};
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 11px;
    color: {TEXT3};
  }}
  .ht-footer span {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}

  /* ── Plotly container tweaks ────────────────────────────────────────────── */
  .js-plotly-plot .plotly .main-svg {{
    border-radius: 12px;
  }}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/catboost_afib_model.pkl"
DB_PATH     = "PAFPDB"
TARGET_FS   = 128
WINDOW_SEC  = 15
STRIDE_SEC  = 1
SNIPPET_SEC = 45
THRESHOLD   = 0.3

DEMO_RECORDS = {
    "🟢  Healthy Control #1  ·  t03":     ("t03", 0,  "normal"),
    "🟢  Healthy Control #2  ·  t05":     ("t05", 0,  "normal"),
    "🟢  Healthy Control #3  ·  t07":     ("t07", 0,  "normal"),
    "🟡  Stable PAF #1  ·  t01 (min 0)":  ("t01", 0,  "paf_distant"),
    "🟡  Stable PAF #2  ·  t09 (min 0)":  ("t09", 0,  "paf_distant"),
    "🟡  Stable PAF #3  ·  t11 (min 0)":  ("t11", 0,  "paf_distant"),
    "🔴  Imminent AFib #1  ·  t02 (min 27)": ("t02", 27, "paf_imminent"),
    "🔴  Imminent AFib #2  ·  t10 (min 27)": ("t10", 27, "paf_imminent"),
    "🔴  Imminent AFib #3  ·  t12 (min 27)": ("t12", 27, "paf_imminent"),
}

RISK_COLOR = {
    "normal":       EMERALD,
    "paf_distant":  AMBER,
    "paf_imminent": CRIMSON,
    "unknown":      TEXT2,
}

BADGE_CLASS = {
    "normal": "ht-badge-normal",
    "paf_distant": "ht-badge-distant",
    "paf_imminent": "ht-badge-imminent",
}

BADGE_LABEL = {
    "normal": "NORMAL",
    "paf_distant": "PAF STABLE",
    "paf_imminent": "PAF IMMINENT",
}

def safe_float(v, decimals=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{decimals}f}"

# ─── MODEL ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_explainer(_mdl):
    return shap.TreeExplainer(_mdl)

# ─── DSP ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_ecg(record_name, start_min):
    rp = os.path.join(DB_PATH, record_name)
    hdr = wfdb.rdheader(rp)
    fs  = hdr.fs
    s   = int(start_min * 60 * fs)
    e   = min(s + int(SNIPPET_SEC * fs), hdr.sig_len)
    rec = wfdb.rdrecord(rp, sampfrom=s, sampto=e)
    sig = rec.p_signal[:, 0].astype(np.float64)
    if fs != TARGET_FS:
        sig = resample(sig, int(len(sig) * TARGET_FS / fs))
    b, a = butter(2, 0.5 / (TARGET_FS / 2), btype='high')
    sig  = filtfilt(b, a, sig)
    mx   = np.max(np.abs(sig))
    if mx > 0: sig = sig / mx
    return sig, TARGET_FS

def r_peaks(signal, fs):
    b, a   = butter(2, [5/(fs/2), 15/(fs/2)], btype='band')
    sq     = filtfilt(b, a, signal) ** 2
    h = np.percentile(sq, 75)

    pk, _ = find_peaks(
        sq,
        distance=int(0.25 * fs),
        prominence=np.std(sq) * 0.3,
        height=h
    )
    return pk

def ecg_to_rr(signal, fs):
    pk = r_peaks(signal, fs)
    if len(pk) < 2: return np.array([]), pk
    rr = np.diff(pk) / fs * 1000.0
    return rr[(rr>=200)&(rr<=2000)], pk

def features(rr):
    if len(rr) < 8: return None
    p, c    = rr[:-1], rr[1:]
    diff    = np.diff(rr)
    cum     = np.cumsum(rr/1000) - np.cumsum(rr/1000)[0]
    fs      = 4.0
    ti      = np.arange(0, cum[-1], 1/fs)
    lf=hf=lf_hf=0.0
    if len(ti) >= 2:
        fi   = interp1d(cum, rr/1000, kind='linear', fill_value='extrapolate')
        ri   = fi(ti) - np.mean(fi(ti))
        fr, psd = welch(ri, fs=fs, nperseg=min(256,len(ri)))
        lf   = float(trapezoid(psd[(fr>=0.04)&(fr<=0.15)], fr[(fr>=0.04)&(fr<=0.15)])) if np.any((fr>=0.04)&(fr<=0.15)) else 0.0
        hf   = float(trapezoid(psd[(fr>0.15)&(fr<=0.40)], fr[(fr>0.15)&(fr<=0.40)])) if np.any((fr>0.15)&(fr<=0.40)) else 0.0
        lf_hf = lf/hf if hf > 0 else 0.0
    try:    se = float(ant.sample_entropy(rr))
    except: se = 0.0
    if math.isnan(se) or math.isinf(se): se = 0.0
    return {
        "mean_rr": float(np.mean(rr)),    "sdnn":      float(np.std(rr,ddof=1)),
        "rmssd":   float(np.sqrt(np.mean(diff**2))),
        "pnn50":   float((np.sum(np.abs(diff)>50)/len(rr))*100),
        "lf": lf,  "hf": hf,              "lf_hf_ratio": lf_hf,
        "sd1":     float(np.sqrt(np.std(p-c)**2/2)),
        "sd2":     float(np.sqrt(np.std(p+c)**2/2)),
        "sampen":  se,
        "pac_count": int(np.sum(c < 0.80*p)),
        "pac_ratio": float((np.sum(c < 0.80*p)/len(rr))*100),
    }

@st.cache_data(show_spinner=False)
def sliding_windows(signal, fs):
    win, stride = int(WINDOW_SEC*fs), int(STRIDE_SEC*fs)
    out = []
    s   = 0
    while s + win <= len(signal):
        seg     = signal[s:s+win]
        rr, _   = ecg_to_rr(seg, fs)
        feat    = features(rr) if len(rr) >= 8 else None
        out.append({"start_sec": round(s/fs,1), "end_sec": round((s+win)/fs,1),
                    "start_idx": s, "end_idx": s+win, "rr": rr, "features": feat})
        s += stride
    return out

# ─── ESP32 HELPER ─────────────────────────────────────────────────────────────
def fetch_esp32_data(ip, port, timeout=2):
    """Fetch RR intervals from ESP32 HTTP endpoint."""
    if not REQUESTS_AVAILABLE:
        st.error("The 'requests' library is not installed. Please run: pip install requests")
        return None
    url = f"http://{ip}:{port}/rr"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        text = resp.text.strip()
        # Try to parse as CSV or newline-separated values
        vals = []
        for line in text.splitlines():
            for x in line.split(','):
                x = x.strip()
                if x:
                    try:
                        v = float(x)
                        if 200 <= v <= 2000:
                            vals.append(v)
                    except:
                        pass
        if len(vals) == 0:
            return None
        return np.array(vals)
    except Exception as e:
        return None

# ─── PLOTS ────────────────────────────────────────────────────────────────────
PLOT_BG = "#f4fafe"
GRID_C  = "#e0ecf5"

def plot_ecg(signal, fs, peaks, windows, win_idx, label_key):
    t    = np.arange(len(signal)) / fs
    w    = windows[win_idx]
    rc   = RISK_COLOR.get(label_key, TEXT2)

    mdl  = load_model()
    risk = None
    if w["features"] and mdl:
        risk = float(mdl.predict_proba(pd.DataFrame([w["features"]]))[0][1])

    # Fresh highlight colours
    hl = ("rgba(217,108,122,0.10)" if risk is not None and risk >= 0.3  else
          "rgba(217,168,108,0.10)" if risk is not None and risk >= 0.15 else
          "rgba(69,176,140,0.10)" if risk is not None else
          "rgba(150,150,150,0.05)")
    bc = ("#d96c7a" if risk is not None and risk >= 0.3  else
          "#d9a86c" if risk is not None and risk >= 0.15 else
          "#45b08c" if risk is not None else TEXT3)

    fig = go.Figure()

    # Window highlight
    fig.add_vrect(x0=w["start_sec"], x1=w["end_sec"],
                  fillcolor=hl, layer="below", line_width=0)
    for x in [w["start_sec"], w["end_sec"]]:
        fig.add_vline(x=x, line=dict(color=bc, width=1.2, dash="dot"))

    # ECG trace – vibrant cyan, thicker, with subtle glow via opacity
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode='lines',
        line=dict(color=ECG_COLOR, width=1.8),
        name='ECG',
        hovertemplate='%{x:.2f}s<extra></extra>',
    ))

    # R-peaks – golden markers, slightly larger
    if len(peaks):
        fig.add_trace(go.Scatter(
            x=peaks/fs, y=signal[peaks], mode='markers',
            marker=dict(color=GOLD, size=6, symbol='circle',
                        line=dict(color='white', width=1)),
            name='R-peaks',
            hovertemplate='%{x:.2f}s<extra></extra>',
        ))

    # Risk annotation if available
    if risk is not None:
        fig.add_annotation(
            x=(w["start_sec"]+w["end_sec"])/2, y=0.96, yref='paper',
            text=f"<b>{risk*100:.0f}% risk</b>",
            showarrow=False,
            font=dict(color=bc, size=11, family='JetBrains Mono'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=bc,
            borderwidth=1,
            borderpad=5,
        )

    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=11, family='Inter'),
        margin=dict(l=44, r=16, t=12, b=44),
        height=260,
        showlegend=False,
        xaxis=dict(
            title='Time (s)',
            gridcolor=GRID_C,
            zerolinecolor=GRID_C,
            tickfont=dict(family='JetBrains Mono', size=10),
            showgrid=True,
        ),
        yaxis=dict(
            title='Amplitude (norm.)',
            gridcolor=GRID_C,
            zerolinecolor=GRID_C,
            tickfont=dict(family='JetBrains Mono', size=10),
            showgrid=True,
        ),
    )
    return fig


def plot_gauge(probability):
    pct = (probability or 0) * 100
    if pct >= 30:
        color, band, soft = "#d96c7a", "High Risk", "#fae8eb"
    elif pct >= 15:
        color, band, soft = "#d9a86c", "Elevated", "#f8efe4"
    else:
        color, band, soft = "#45b08c", "Normal", "#e2f3ec"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(color=color, size=42, family='JetBrains Mono')),
        gauge=dict(
            shape="angular",
            axis=dict(
                range=[0, 100],
                tickvals=[0, 15, 30, 50, 75, 100],
                tickwidth=0,
                tickcolor=TEXT3,
                tickfont=dict(color=TEXT3, size=10.5, family='Inter'),
                ticksuffix="%",
            ),
            bar=dict(color=color, thickness=0.5),
            bgcolor="#fafcfe",
            borderwidth=0,
            steps=[
                dict(range=[0, 15],  color="#e2f3ec"),
                dict(range=[15, 30], color="#f8efe4"),
                dict(range=[30, 100], color="#fae8eb"),
            ],
            threshold=dict(
                line=dict(color=TEXT3, width=2),
                thickness=0.82,
                value=30,
            ),
        ),
        title=dict(text="AFib Risk Score", font=dict(color=TEXT2, size=14, family='Inter')),
    ))

    fig.add_annotation(
        x=0.5, y=0.06, xref="paper", yref="paper",
        text=f"<b>{band.upper()}</b>",
        showarrow=False,
        font=dict(color=color, size=12.5, family='Inter'),
        bgcolor=soft,
        bordercolor=color,
        borderwidth=1,
        borderpad=7,
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=34, r=34, t=56, b=28),
        height=340,
    )
    return fig


def plot_poincare(rr, label_key):
    if len(rr) < 2: return go.Figure()
    color = RISK_COLOR.get(label_key, TEXT2)
    mn, mx = rr.min(), rr.max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rr[:-1], y=rr[1:], mode='markers',
        marker=dict(color=color, size=5, opacity=0.7,
                    line=dict(width=0)),
        hovertemplate='RR(n)=%{x:.0f}<br>RR(n+1)=%{y:.0f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color=BORDER, dash='dash', width=1), showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=11, family='Inter'),
        margin=dict(l=44, r=12, t=36, b=44), height=240,
        title=dict(text='Poincaré Plot', font=dict(color=TEXT2, size=12), x=0.5),
        xaxis=dict(title='RR(n) ms', gridcolor=GRID_C,
                   tickfont=dict(family='JetBrains Mono', size=10)),
        yaxis=dict(title='RR(n+1) ms', gridcolor=GRID_C,
                   tickfont=dict(family='JetBrains Mono', size=10)),
        showlegend=False,
    )
    return fig


def plot_rr_series(rr, label_key):
    color = RISK_COLOR.get(label_key, TEXT2)
    fig = go.Figure(go.Scatter(
        y=rr, mode='lines+markers',
        line=dict(color=color, width=1.6),
        marker=dict(color=color, size=4),
        hovertemplate='Beat %{x}<br>%{y:.0f} ms<extra></extra>',
    ))
    fig.add_hline(y=np.mean(rr), line=dict(color=TEXT3, dash='dot', width=1))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=11, family='Inter'),
        margin=dict(l=44, r=12, t=36, b=44), height=240,
        title=dict(text='RR Interval Series', font=dict(color=TEXT2, size=12), x=0.5),
        xaxis=dict(title='Beat #', gridcolor=GRID_C,
                   tickfont=dict(family='JetBrains Mono', size=10)),
        yaxis=dict(title='ms', gridcolor=GRID_C,
                   tickfont=dict(family='JetBrains Mono', size=10)),
    )
    return fig


def plot_shap(feat, mdl, expl):
    df  = pd.DataFrame([feat])
    sv  = expl.shap_values(df)
    if isinstance(sv, list): sv = sv[1]
    vals  = sv[0]
    names = list(feat.keys())
    order = np.argsort(np.abs(vals))[::-1][:10]
    sv    = vals[order]
    sn    = [names[i] for i in order]
    colors= [CRIMSON if v > 0 else EMERALD for v in sv]

    fig = go.Figure(go.Bar(
        x=sv, y=sn, orientation='h',
        marker_color=colors, marker_line_width=0,
        hovertemplate='%{y}: %{x:+.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line=dict(color=BORDER, width=1))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=11, family='Inter'),
        margin=dict(l=10, r=16, t=36, b=44), height=300,
        title=dict(text='SHAP — Feature Contributions', font=dict(color=TEXT2, size=12), x=0.5),
        xaxis=dict(title='Impact on risk score', gridcolor=GRID_C, zerolinecolor=BORDER,
                   tickfont=dict(family='JetBrains Mono', size=10)),
        yaxis=dict(autorange='reversed'),
        bargap=0.3,
    )
    return fig


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    mdl = load_model()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='padding: 0 0 20px 0;'>
          <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='background: {TINT}; border-radius: 12px; padding: 8px 10px; font-size: 22px;'>🫀</div>
            <div>
              <div style='font-size: 22px; font-weight: 700; letter-spacing: -0.02em; color: {TEXT}; font-family: "Source Serif 4", serif;'>HeartTrack AI</div>
              <div style='font-size: 10.5px; color: {TEXT3}; letter-spacing: 0.07em;'>PAF DETECTION · v2</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        mode = st.radio(
            "Select input",
            ["📂 Demo Recording", "📁 Upload CSV", "📡 ESP32 Stream"],
            label_visibility="collapsed"
        )
        st.divider()

        if "Demo" in mode:
            demo_label = st.selectbox(
                "Choose a recording",
                list(DEMO_RECORDS.keys()),
                label_visibility="collapsed"
            )
            record, start_min, label_key = DEMO_RECORDS[demo_label]

            bc = BADGE_CLASS.get(label_key, "")
            bl = BADGE_LABEL.get(label_key, "")
            st.markdown(f"""
            <div class='ht-card' style='margin-top: 6px; padding: 16px 18px;'>
              <span class='ht-badge {bc}'>{bl}</span>
              <div style='margin-top: 12px; color: {TEXT2}; line-height: 1.8; font-size: 12.5px;'>
                <span style='color: {TEXT3};'>Record</span> &nbsp; <b>{record}</b><br>
                <span style='color: {TEXT3};'>Start</span> &nbsp;&nbsp;&nbsp; <b>min {start_min}</b><br>
                <span style='color: {TEXT3};'>Duration</span> &nbsp; <b>{SNIPPET_SEC}s</b><br>
                <span style='color: {TEXT3};'>Window</span> &nbsp;&nbsp; <b>{WINDOW_SEC}s / {STRIDE_SEC}s stride</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            auto_play  = st.toggle("▶ Auto‑play", value=False)
            play_speed = st.slider(
                "Speed (s/window)",
                0.3, 3.0, 1.0, 0.1,
                disabled=not auto_play,
                help="Time between window advances"
            )
        elif "CSV" in mode:
            uploaded  = st.file_uploader(
                "Upload RR interval CSV",
                type=["csv"],
                label_visibility="collapsed",
                help="One RR interval (ms) per line, or comma-separated values. Range 200–2000 ms."
            )
            label_key = "unknown"
            auto_play = False
            play_speed = 1.0
        else:  # ESP32 Stream
            st.markdown("### 📡 ESP32 Wi‑Fi Stream")
            esp_ip = st.text_input("IP Address", value="192.168.1.100")
            esp_port = st.text_input("Port", value="80")
            connect_btn = st.button("Connect", type="primary")
            disconnect_btn = st.button("Disconnect", type="secondary")

            # Manage connection state
            if connect_btn:
                if not REQUESTS_AVAILABLE:
                    st.error("The 'requests' library is required. Please install: pip install requests")
                else:
                    st.session_state.esp_connected = True
                    st.session_state.esp_ip = esp_ip
                    st.session_state.esp_port = esp_port
                    st.session_state.esp_rr_buffer = []  # store last 300 intervals
                    st.session_state.esp_last_update = time.time()
                    st.success(f"Connecting to {esp_ip}:{esp_port}...")
            if disconnect_btn:
                st.session_state.esp_connected = False
                st.session_state.esp_rr_buffer = []
                st.info("Disconnected")

            # Show status
            if st.session_state.get("esp_connected", False):
                st.markdown(f"✅ Connected to `{st.session_state.esp_ip}:{st.session_state.esp_port}`")
                st.caption(f"Buffer size: {len(st.session_state.esp_rr_buffer)} RR intervals")
            else:
                st.markdown("🔴 Not connected")
            label_key = "unknown"
            auto_play = False
            play_speed = 1.0

        st.divider()
        if mdl:
            st.markdown(
                f"<div style='font-size: 11px; color: {EMERALD}; display: flex; align-items: center; gap: 6px;'>"
                f"<span style='font-size: 14px;'>✓</span> CatBoost model loaded</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size: 11px; color: {CRIMSON}; display: flex; align-items: center; gap: 6px;'>"
                f"<span style='font-size: 14px;'>✗</span> Model not found — run training first</div>",
                unsafe_allow_html=True
            )

    # ── Main Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class='ht-page-header'>
      <h1>HeartTrack AI</h1>
      <span class='badge'>● LIVE ANALYSIS</span>
    </div>
    <p style='color: #4e6a82; margin: 0 0 24px 0; font-size: 13px; letter-spacing: 0.01em;'>
      Paroxysmal Atrial Fibrillation detection · Real ECG from PAFPDB · CatBoost + SHAP explainability
    </p>
    """, unsafe_allow_html=True)

    if mdl is None:
        st.error("Model not found at `models/catboost_afib_model.pkl`. Run `07_Train.py` first.")
        return

    expl = load_explainer(mdl)

    # ═══════════════════════════════════════════════════════════════════════════
    # DEMO MODE
    # ═══════════════════════════════════════════════════════════════════════════
    if "Demo" in mode:
        if not os.path.exists(os.path.join(DB_PATH, f"{record}.hea")):
            st.error(f"PAFPDB not found at `{DB_PATH}/`.")
            return

        with st.spinner(f"Loading {record} from PAFPDB…"):
            signal, fs = load_ecg(record, start_min)

        with st.spinner("Running sliding window analysis…"):
            windows = sliding_windows(signal, fs)
            all_pk  = r_peaks(signal, fs)

        n_win = len(windows)
        if n_win == 0:
            st.error("No windows extracted.")
            return

        # ── Window slider ─────────────────────────────────────────────────────
        st.markdown("<div class='ht-eyebrow'>ECG TRACE</div>", unsafe_allow_html=True)
        col_sl, col_info = st.columns([5, 1])
        with col_sl:
            win_idx = st.slider(
                "Analysis window",
                0, n_win-1,
                st.session_state.get("win_idx", 0),
                key="win_slider",
                label_visibility="collapsed",
            )
        with col_info:
            w = windows[win_idx]
            st.markdown(f"""
            <div style='text-align: right; padding-top: 6px;'>
              <span style='font-size: 20px; font-weight: 700; font-family: "JetBrains Mono", monospace; color: {TEXT};'>
                {win_idx+1} / {n_win}
              </span><br>
              <span style='font-size: 11px; color: {TEXT3};'>
                {w['start_sec']}s → {w['end_sec']}s
              </span>
            </div>
            """, unsafe_allow_html=True)

        # ── ECG plot ──────────────────────────────────────────────────────────
        st.plotly_chart(
            plot_ecg(signal, fs, all_pk, windows, win_idx, label_key),
            use_container_width=True,
            config={"displayModeBar": False}
        )

        # ── Model inference ──────────────────────────────────────────────────
        w    = windows[win_idx]
        feat = w["features"]
        risk = None
        if feat:
            risk = float(mdl.predict_proba(pd.DataFrame([feat]))[0][1])

        # ── Risk + Metrics ────────────────────────────────────────────────────
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown("<div class='ht-eyebrow'>RISK ASSESSMENT</div>", unsafe_allow_html=True)
            st.plotly_chart(
                plot_gauge(risk),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            if risk is None:
                st.markdown("""
                <div class='risk-banner risk-unknown'>
                  ⚪ &nbsp; Not enough beats in this window
                </div>""", unsafe_allow_html=True)
            elif risk >= THRESHOLD:
                st.markdown(f"""
                <div class='risk-banner risk-high'>
                  ⚠️ &nbsp; HIGH RISK &nbsp;—&nbsp; PAF Detected &nbsp;
                  <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            elif risk >= 0.15:
                st.markdown(f"""
                <div class='risk-banner risk-medium'>
                  🟡 &nbsp; ELEVATED &nbsp;—&nbsp; Monitor closely &nbsp;
                  <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-banner risk-low'>
                  ✅ &nbsp; NORMAL &nbsp;—&nbsp; Healthy sinus rhythm &nbsp;
                  <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: 10px;'>
              Model‑estimated probability of AFib in this window &nbsp;·&nbsp;
              &lt;15% normal &nbsp;·&nbsp; 15–30% elevated &nbsp;·&nbsp; &gt;30% high risk
            </p>""", unsafe_allow_html=True)

        with col_right:
            if feat:
                st.markdown("<div class='ht-eyebrow' style='margin-bottom: 12px;'>HRV METRICS</div>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("SDNN",    f"{safe_float(feat['sdnn'], 1)} ms")
                m2.metric("RMSSD",   f"{safe_float(feat['rmssd'], 1)} ms")
                m3.metric("pNN50",   f"{safe_float(feat['pnn50'], 1)}%")
                m4.metric("Mean RR", f"{safe_float(feat['mean_rr'], 0)} ms")

                m5, m6, m7, m8 = st.columns(4)
                pac = feat['pac_count']
                m5.metric("PAC Count", str(pac),
                          delta="elevated" if pac > 3 else None,
                          delta_color="inverse")
                m6.metric("LF/HF",  safe_float(feat['lf_hf_ratio'], 2))
                m7.metric("SampEn", safe_float(feat['sampen'], 3))
                m8.metric("SD1",    f"{safe_float(feat['sd1'], 1)} ms")

        st.divider()

        # ── Bottom row: SHAP, Poincaré, RR series ────────────────────────────
        st.markdown("<div class='ht-eyebrow'>MODEL EXPLAINABILITY & SIGNAL DETAIL</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            if feat and risk is not None:
                st.plotly_chart(
                    plot_shap(feat, mdl, expl),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
                st.markdown(f"""
                <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: -8px;'>
                  🔴 Increases AFib risk &nbsp;·&nbsp; 🟢 Decreases risk
                </p>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ht-card' style='height: 300px; display: flex; align-items: center; justify-content: center; color: {TEXT3}; font-size: 13px;'>
                  Not enough beats for SHAP
                </div>""", unsafe_allow_html=True)

        with c2:
            rr = w["rr"]
            if len(rr) > 1:
                st.plotly_chart(
                    plot_poincare(rr, label_key),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
                st.markdown(f"""
                <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: -8px;'>
                  Tight cluster = regular &nbsp;·&nbsp; Scattered = irregular (AFib)
                </p>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ht-card' style='height: 240px; display: flex; align-items: center; justify-content: center; color: {TEXT3}; font-size: 13px;'>
                  Insufficient RR intervals
                </div>""", unsafe_allow_html=True)

        with c3:
            if len(rr) > 0:
                st.plotly_chart(
                    plot_rr_series(rr, label_key),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            else:
                st.markdown(f"""
                <div class='ht-card' style='height: 240px; display: flex; align-items: center; justify-content: center; color: {TEXT3}; font-size: 13px;'>
                  No RR intervals detected
                </div>""", unsafe_allow_html=True)

        # ── Expandable guide ──────────────────────────────────────────────────
        with st.expander("ℹ️  What do these metrics mean?"):
            st.markdown(f"""
            <div style='font-size: 12.5px; line-height: 1.85; color: {TEXT2};'>
              <b style='color: {TEXT};'>SDNN / RMSSD / pNN50</b> — standard heart-rate variability
              measures. Low values can indicate reduced autonomic regulation, a known precursor
              to paroxysmal AFib episodes.<br>
              <b style='color: {TEXT};'>PAC Count</b> — premature atrial contractions detected in
              this window; frequent PACs are a common trigger for AFib onset.<br>
              <b style='color: {TEXT};'>LF/HF Ratio</b> — sympathetic vs. parasympathetic balance
              from frequency-domain analysis.<br>
              <b style='color: {TEXT};'>SampEn</b> — sample entropy; higher values reflect more
              irregular, less predictable rhythm.<br>
              <b style='color: {TEXT};'>SHAP values</b> — show how much each feature pushed the
              model's prediction up (red) or down (green) for this specific window.
            </div>
            """, unsafe_allow_html=True)

        # ── Auto‑play loop ────────────────────────────────────────────────────
        if auto_play:
            time.sleep(play_speed)
            st.session_state["win_idx"] = (win_idx + 1) % n_win
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # UPLOAD CSV MODE
    # ═══════════════════════════════════════════════════════════════════════════
    elif "CSV" in mode:
        if not uploaded:
            st.markdown(f"""
            <div class='ht-card' style='text-align: center; padding: 56px 32px;'>
              <div style='font-size: 40px; margin-bottom: 16px;'>📁</div>
              <div style='font-size: 16px; font-weight: 600; color: {TEXT}; margin-bottom: 8px;'>
                Upload an RR interval CSV
              </div>
              <div style='font-size: 12px; color: {TEXT3};'>
                One RR interval in milliseconds per line · Range 200–2000 ms
              </div>
            </div>
            """, unsafe_allow_html=True)
            return

        raw  = uploaded.read().decode()
        vals = []
        for line in raw.splitlines():
            for x in line.split(','):
                x = x.strip()
                if x and not x.startswith('#') and x != 'rr_ms':
                    try:
                        v = float(x)
                        if 200 <= v <= 2000:
                            vals.append(v)
                    except: pass
        rr = np.array(vals)

        if len(rr) < 10:
            st.error(f"Only {len(rr)} valid intervals found (need ≥ 10, range 200–2000 ms).")
            return

        st.success(f"✓ Loaded {len(rr)} RR intervals from {uploaded.name}")
        feat = features(rr)
        if feat is None:
            st.error("Feature extraction failed.")
            return

        risk = float(mdl.predict_proba(pd.DataFrame([feat]))[0][1])

        col_l, col_r = st.columns([1, 2], gap="large")
        with col_l:
            st.plotly_chart(
                plot_gauge(risk),
                use_container_width=True,
                config={"displayModeBar": False}
            )
            cls = "risk-high" if risk >= THRESHOLD else "risk-medium" if risk >= 0.15 else "risk-low"
            icon = "⚠️" if risk >= THRESHOLD else "🟡" if risk >= 0.15 else "✅"
            label = "HIGH RISK — PAF Detected" if risk >= THRESHOLD else "ELEVATED" if risk >= 0.15 else "NORMAL"
            st.markdown(f"""
            <div class='risk-banner {cls}'>
              {icon} &nbsp; {label} &nbsp;
              <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: 10px;'>
              Model‑estimated probability of AFib in this recording &nbsp;·&nbsp;
              &lt;15% normal &nbsp;·&nbsp; 15–30% elevated &nbsp;·&nbsp; &gt;30% high risk
            </p>""", unsafe_allow_html=True)

        with col_r:
            st.markdown("<div class='ht-eyebrow' style='margin-bottom: 12px;'>HRV METRICS</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SDNN",    f"{safe_float(feat['sdnn'], 1)} ms")
            m2.metric("RMSSD",   f"{safe_float(feat['rmssd'], 1)} ms")
            m3.metric("pNN50",   f"{safe_float(feat['pnn50'], 1)}%")
            m4.metric("Mean RR", f"{safe_float(feat['mean_rr'], 0)} ms")

        st.divider()
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.plotly_chart(
                plot_shap(feat, mdl, expl),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        with c2:
            st.plotly_chart(
                plot_poincare(rr, "unknown"),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        with c3:
            st.plotly_chart(
                plot_rr_series(rr, "unknown"),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # ESP32 STREAM MODE
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        if not st.session_state.get("esp_connected", False):
            st.markdown(f"""
            <div class='ht-card' style='text-align: center; padding: 56px 32px;'>
              <div style='font-size: 40px; margin-bottom: 16px;'>📡</div>
              <div style='font-size: 16px; font-weight: 600; color: {TEXT}; margin-bottom: 8px;'>
                Connect to ESP32 via Wi‑Fi
              </div>
              <div style='font-size: 12px; color: {TEXT3};'>
                Enter IP and Port in the sidebar, then click Connect.
                The ESP32 should serve RR intervals on <code>/rr</code>.
              </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # We are connected – fetch data in a loop
        st.markdown("<div class='ht-eyebrow'>LIVE ESP32 STREAM</div>", unsafe_allow_html=True)

        # Placeholders for dynamic update
        placeholder_ecg = st.empty()
        placeholder_risk = st.empty()
        placeholder_metrics = st.empty()
        placeholder_bottom = st.empty()

        # We'll maintain a rolling buffer of RR intervals (max 300)
        if "esp_rr_buffer" not in st.session_state:
            st.session_state.esp_rr_buffer = []

        # Fetch new data every second
        for _ in range(100):  # limit to prevent infinite loop in one run
            if not st.session_state.get("esp_connected", False):
                st.info("Disconnected by user.")
                break

            new_rr = fetch_esp32_data(st.session_state.esp_ip, st.session_state.esp_port)
            if new_rr is not None and len(new_rr) > 0:
                st.session_state.esp_rr_buffer.extend(new_rr.tolist())
                # Keep only last 300 intervals
                if len(st.session_state.esp_rr_buffer) > 300:
                    st.session_state.esp_rr_buffer = st.session_state.esp_rr_buffer[-300:]

            # If we have enough intervals, compute features and display
            rr_array = np.array(st.session_state.esp_rr_buffer)
            if len(rr_array) >= 8:
                feat = features(rr_array)
                if feat is not None:
                    risk = float(mdl.predict_proba(pd.DataFrame([feat]))[0][1])
                    # Update risk gauge
                    with placeholder_risk.container():
                        col_left, col_right = st.columns([1, 2], gap="large")
                        with col_left:
                            st.markdown("<div class='ht-eyebrow'>RISK ASSESSMENT</div>", unsafe_allow_html=True)
                            st.plotly_chart(
                                plot_gauge(risk),
                                use_container_width=True,
                                config={"displayModeBar": False}
                            )
                            if risk >= THRESHOLD:
                                st.markdown(f"""
                                <div class='risk-banner risk-high'>
                                  ⚠️ &nbsp; HIGH RISK &nbsp;—&nbsp; PAF Detected &nbsp;
                                  <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
                                </div>""", unsafe_allow_html=True)
                            elif risk >= 0.15:
                                st.markdown(f"""
                                <div class='risk-banner risk-medium'>
                                  🟡 &nbsp; ELEVATED &nbsp;—&nbsp; Monitor closely &nbsp;
                                  <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='risk-banner risk-low'>
                                  ✅ &nbsp; NORMAL &nbsp;—&nbsp; Healthy sinus rhythm &nbsp;
                                  <span style='font-family: "JetBrains Mono", monospace;'>{risk*100:.1f}%</span>
                                </div>""", unsafe_allow_html=True)
                            st.markdown(f"""
                            <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: 10px;'>
                              Buffer size: {len(rr_array)} intervals &nbsp;·&nbsp; Last update: {time.strftime('%H:%M:%S')}
                            </p>""", unsafe_allow_html=True)
                        with col_right:
                            st.markdown("<div class='ht-eyebrow' style='margin-bottom: 12px;'>HRV METRICS</div>", unsafe_allow_html=True)
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("SDNN",    f"{safe_float(feat['sdnn'], 1)} ms")
                            m2.metric("RMSSD",   f"{safe_float(feat['rmssd'], 1)} ms")
                            m3.metric("pNN50",   f"{safe_float(feat['pnn50'], 1)}%")
                            m4.metric("Mean RR", f"{safe_float(feat['mean_rr'], 0)} ms")

                            m5, m6, m7, m8 = st.columns(4)
                            pac = feat['pac_count']
                            m5.metric("PAC Count", str(pac),
                                      delta="elevated" if pac > 3 else None,
                                      delta_color="inverse")
                            m6.metric("LF/HF",  safe_float(feat['lf_hf_ratio'], 2))
                            m7.metric("SampEn", safe_float(feat['sampen'], 3))
                            m8.metric("SD1",    f"{safe_float(feat['sd1'], 1)} ms")

                        # Bottom row: Poincaré, RR series, SHAP
                        with placeholder_bottom.container():
                            st.divider()
                            st.markdown("<div class='ht-eyebrow'>MODEL EXPLAINABILITY & SIGNAL DETAIL</div>", unsafe_allow_html=True)
                            c1, c2, c3 = st.columns(3, gap="medium")
                            with c1:
                                if feat:
                                    st.plotly_chart(
                                        plot_shap(feat, mdl, expl),
                                        use_container_width=True,
                                        config={"displayModeBar": False}
                                    )
                                    st.markdown(f"""
                                    <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: -8px;'>
                                      🔴 Increases AFib risk &nbsp;·&nbsp; 🟢 Decreases risk
                                    </p>""", unsafe_allow_html=True)
                            with c2:
                                if len(rr_array) > 1:
                                    st.plotly_chart(
                                        plot_poincare(rr_array, "unknown"),
                                        use_container_width=True,
                                        config={"displayModeBar": False}
                                    )
                                    st.markdown(f"""
                                    <p style='font-size: 10px; color: {TEXT3}; text-align: center; margin-top: -8px;'>
                                      Tight cluster = regular &nbsp;·&nbsp; Scattered = irregular (AFib)
                                    </p>""", unsafe_allow_html=True)
                            with c3:
                                if len(rr_array) > 0:
                                    st.plotly_chart(
                                        plot_rr_series(rr_array, "unknown"),
                                        use_container_width=True,
                                        config={"displayModeBar": False}
                                    )
            else:
                with placeholder_risk.container():
                    st.info(f"⏳ Collecting RR intervals... ({len(rr_array)}/8 needed)")

            # Wait 1 second before next fetch
            time.sleep(1)

        # If we exit the loop, show a message
        if not st.session_state.get("esp_connected", False):
            st.info("Stream ended.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='ht-footer'>
      <span>🫀 HeartTrack AI · CatBoost + SHAP · PAFPDB reference dataset</span>
      <span>For research &amp; educational use only — not a diagnostic device. Always consult a clinician.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()