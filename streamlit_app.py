"""
HeartTrack AI — Streamlit App v2
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

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartTrack AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
# Palette: deep navy base, crimson accent, amber warning, emerald safe
BG       = "#07090f"
BG2      = "#0d1117"
BG3      = "#161b22"
BORDER   = "#21262d"
TEXT     = "#e6edf3"
TEXT2    = "#8b949e"
TEXT3    = "#484f58"
CRIMSON  = "#e11d48"
CRIMSON2 = "#9f1239"
AMBER    = "#d97706"
EMERALD  = "#059669"
SAPPHIRE = "#1d4ed8"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, .stApp {{ background: {BG}; color: {TEXT}; font-family: 'Inter', sans-serif; }}
  section[data-testid="stSidebar"] {{ background: {BG2}; border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
  .stApp header {{ background: {BG} !important; }}
  .block-container {{ padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }}

  /* Remove streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Metric cards — fully custom */
  div[data-testid="metric-container"] {{
    background: {BG3};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px !important;
    transition: border-color 0.2s;
  }}
  div[data-testid="metric-container"]:hover {{ border-color: #30363d; }}
  div[data-testid="metric-container"] label {{
    color: {TEXT3} !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 11px !important;
  }}

  /* Slider */
  .stSlider [data-baseweb="slider"] {{ padding: 0 !important; }}
  .stSlider [data-baseweb="thumb"] {{ background: {CRIMSON} !important; border: 2px solid white !important; }}
  .stSlider [data-baseweb="track-fill"] {{ background: {CRIMSON} !important; }}

  /* Buttons */
  .stButton > button {{
    background: {CRIMSON}; color: white; border: none;
    border-radius: 8px; font-weight: 600; font-size: 13px;
    padding: 8px 20px; transition: all 0.2s;
  }}
  .stButton > button:hover {{ background: {CRIMSON2}; transform: translateY(-1px); }}

  /* Select box */
  .stSelectbox [data-baseweb="select"] > div {{
    background: {BG3} !important; border-color: {BORDER} !important;
    color: {TEXT} !important; border-radius: 8px !important;
  }}

  /* Radio */
  .stRadio [data-testid="stWidgetLabel"] {{ color: {TEXT2} !important; font-size: 12px !important; }}
  .stRadio label {{ color: {TEXT} !important; font-size: 13px !important; }}

  /* Toggle */
  .stToggle label {{ color: {TEXT} !important; font-size: 13px !important; }}

  /* Alerts */
  div[data-testid="stAlert"] {{ border-radius: 10px !important; }}

  /* Divider */
  hr {{ border-color: {BORDER} !important; margin: 1.2rem 0 !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {BG3}; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent; color: {TEXT2}; border-radius: 7px;
    font-weight: 600; font-size: 13px; padding: 8px 18px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {CRIMSON} !important; color: white !important;
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BG2}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

  /* Custom card */
  .ht-card {{
    background: {BG3}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 20px;
  }}
  .ht-badge {{
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 3px 10px; border-radius: 4px; letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .ht-badge-normal   {{ background: #052e16; color: #4ade80; }}
  .ht-badge-distant  {{ background: #451a03; color: #fbbf24; }}
  .ht-badge-imminent {{ background: #450a0a; color: #f87171; }}

  /* Risk banner */
  .risk-banner {{
    border-radius: 10px; padding: 14px 18px;
    display: flex; align-items: center; gap: 12px;
    font-weight: 600; font-size: 14px;
  }}
  .risk-high    {{ background: #450a0a; border: 1px solid #7f1d1d; color: #fca5a5; }}
  .risk-medium  {{ background: #451a03; border: 1px solid #92400e; color: #fcd34d; }}
  .risk-low     {{ background: #052e16; border: 1px solid #14532d; color: #86efac; }}
  .risk-unknown {{ background: {BG3};   border: 1px solid {BORDER}; color: {TEXT2}; }}
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
    h      = np.mean(sq) + 0.5 * np.std(sq)
    pk, _  = find_peaks(sq, distance=int(0.2*fs), height=h)
    return pk

def ecg_to_rr(signal, fs):
    pk = r_peaks(signal, fs)
    if len(pk) < 2: return np.array([]), pk
    rr = np.diff(pk) / fs * 1000.0
    return rr[(rr>=200)&(rr<=2000)], pk

def features(rr):
    if len(rr) < 10: return None
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
        feat    = features(rr) if len(rr) >= 10 else None
        out.append({"start_sec": round(s/fs,1), "end_sec": round((s+win)/fs,1),
                    "start_idx": s, "end_idx": s+win, "rr": rr, "features": feat})
        s += stride
    return out

# ─── PLOTS ────────────────────────────────────────────────────────────────────
PLOT_BG = "#0d1117"
GRID_C  = "#161b22"

def plot_ecg(signal, fs, peaks, windows, win_idx, label_key):
    t    = np.arange(len(signal)) / fs
    w    = windows[win_idx]
    rc   = RISK_COLOR.get(label_key, TEXT2)

    # Risk color for highlight
    mdl  = load_model()
    risk = None
    if w["features"] and mdl:
        risk = float(mdl.predict_proba(pd.DataFrame([w["features"]]))[0][1])

    hl = ("rgba(239,68,68,0.13)"  if risk is not None and risk >= 0.3  else
          "rgba(217,119,6,0.10)"  if risk is not None and risk >= 0.15 else
          "rgba(5,150,105,0.08)"  if risk is not None else
          "rgba(80,80,80,0.06)")
    bc = (CRIMSON if risk is not None and risk >= 0.3  else
          AMBER   if risk is not None and risk >= 0.15 else
          EMERALD if risk is not None else TEXT3)

    fig = go.Figure()

    # Window shading
    fig.add_vrect(x0=w["start_sec"], x1=w["end_sec"],
                  fillcolor=hl, layer="below", line_width=0)
    for x in [w["start_sec"], w["end_sec"]]:
        fig.add_vline(x=x, line=dict(color=bc, width=1.2, dash="dot"))

    # Signal
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode='lines',
        line=dict(color=rc, width=1.0),
        name='ECG', hovertemplate='%{x:.2f}s<extra></extra>',
    ))

    # R-peaks
    if len(peaks):
        fig.add_trace(go.Scatter(
            x=peaks/fs, y=signal[peaks], mode='markers',
            marker=dict(color='#fbbf24', size=4, symbol='circle'),
            name='R-peaks', hovertemplate='%{x:.2f}s<extra></extra>',
        ))

    # Risk annotation
    if risk is not None:
        fig.add_annotation(
            x=(w["start_sec"]+w["end_sec"])/2, y=0.96, yref='paper',
            text=f"<b>{risk*100:.0f}% risk</b>",
            showarrow=False, font=dict(color=bc, size=11, family='JetBrains Mono'),
            bgcolor='rgba(13,17,23,0.85)', bordercolor=bc, borderwidth=1, borderpad=5,
        )

    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=11, family='Inter'),
        margin=dict(l=44, r=16, t=12, b=44),
        height=260,
        showlegend=False,
        xaxis=dict(title='Time (s)', gridcolor=GRID_C, zerolinecolor=GRID_C,
                   tickfont=dict(family='JetBrains Mono', size=10), showgrid=True),
        yaxis=dict(title='Amplitude (norm.)', gridcolor=GRID_C, zerolinecolor=GRID_C,
                   tickfont=dict(family='JetBrains Mono', size=10), showgrid=True),
    )
    return fig


def plot_gauge(probability):
    pct   = (probability or 0) * 100
    color = (CRIMSON if pct >= 30 else AMBER if pct >= 15 else EMERALD)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(color=color, size=40, family='JetBrains Mono')),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0, tickcolor=TEXT3,
                      tickfont=dict(color=TEXT3, size=9)),
            bar=dict(color=color, thickness=0.65),
            bgcolor=BG3,
            borderwidth=0,
            steps=[
                dict(range=[0,15],  color="#052e16"),
                dict(range=[15,30], color="#431407"),
                dict(range=[30,100],color="#450a0a"),
            ],
        ),
        title=dict(text="AFib Risk Score", font=dict(color=TEXT2, size=12, family='Inter')),
    ))
    fig.update_layout(
        paper_bgcolor=BG3, plot_bgcolor=BG3,
        margin=dict(l=16, r=16, t=28, b=4),
        height=190,
    )
    return fig


def plot_poincare(rr, label_key):
    if len(rr) < 2: return go.Figure()
    color = RISK_COLOR.get(label_key, TEXT2)
    mn, mx = rr.min(), rr.max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rr[:-1], y=rr[1:], mode='markers',
        marker=dict(color=color, size=5, opacity=0.65,
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
        line=dict(color=color, width=1.4),
        marker=dict(color=color, size=3),
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
        <div style='padding:4px 0 16px'>
          <div style='font-size:20px;font-weight:800;letter-spacing:-0.02em;color:{TEXT}'>🫀 HeartTrack AI</div>
          <div style='font-size:11px;color:{TEXT3};margin-top:3px;letter-spacing:0.04em'>
            PAF DETECTION  ·  CLINICAL DASHBOARD
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        mode = st.radio("", ["📂 Demo Recording", "📁 Upload CSV"], label_visibility="collapsed")
        st.divider()

        if "Demo" in mode:
            demo_label = st.selectbox("", list(DEMO_RECORDS.keys()), label_visibility="collapsed")
            record, start_min, label_key = DEMO_RECORDS[demo_label]

            bc = BADGE_CLASS.get(label_key, "")
            bl = BADGE_LABEL.get(label_key, "")
            st.markdown(f"""
            <div class='ht-card' style='margin-top:8px;font-size:12px'>
              <span class='ht-badge {bc}'>{bl}</span>
              <div style='margin-top:10px;color:{TEXT2};line-height:1.8'>
                <span style='color:{TEXT3}'>Record</span> &nbsp; {record}<br>
                <span style='color:{TEXT3}'>Start</span> &nbsp;&nbsp;&nbsp; min {start_min}<br>
                <span style='color:{TEXT3}'>Duration</span> &nbsp; {SNIPPET_SEC}s<br>
                <span style='color:{TEXT3}'>Window</span> &nbsp;&nbsp; {WINDOW_SEC}s / {STRIDE_SEC}s stride
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            auto_play  = st.toggle("▶ Auto-play", value=False)
            play_speed = st.slider("Speed (s/window)", 0.3, 3.0, 1.0, 0.1,
                                   disabled=not auto_play)
        else:
            uploaded  = st.file_uploader("RR Interval CSV", type=["csv"],
                                         label_visibility="collapsed")
            label_key = "unknown"
            auto_play = False
            play_speed = 1.0

        st.divider()
        if mdl:
            st.markdown(f"<div style='font-size:11px;color:{TEXT3}'>✓ CatBoost model loaded</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:11px;color:{CRIMSON}'>✗ Model not found</div>",
                        unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px'>
      <span style='font-size:26px;font-weight:800;letter-spacing:-0.03em;color:{TEXT}'>HeartTrack AI</span>
      <span style='font-size:10px;font-weight:700;color:{TEXT3};padding:3px 10px;
             border:1px solid {BORDER};border-radius:4px;letter-spacing:0.08em'>
        CLINICAL DASHBOARD
      </span>
    </div>
    <p style='color:{TEXT3};margin:0 0 20px;font-size:12px;letter-spacing:0.02em'>
      Paroxysmal Atrial Fibrillation &nbsp;·&nbsp; Real ECG from PAFPDB
      &nbsp;·&nbsp; CatBoost + SHAP Explainability
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

        # ── Window control ────────────────────────────────────────────────────
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
            <div style='text-align:right;padding-top:6px'>
              <span style='font-size:18px;font-weight:700;
                font-family:"JetBrains Mono",monospace;color:{TEXT}'>
                {win_idx+1} / {n_win}
              </span><br>
              <span style='font-size:11px;color:{TEXT3}'>
                {w['start_sec']}s → {w['end_sec']}s
              </span>
            </div>
            """, unsafe_allow_html=True)

        # ── ECG plot ──────────────────────────────────────────────────────────
        st.plotly_chart(plot_ecg(signal, fs, all_pk, windows, win_idx, label_key),
                        use_container_width=True, config={"displayModeBar": False})

        # ── Model inference on current window ─────────────────────────────────
        w    = windows[win_idx]
        feat = w["features"]
        risk = None
        if feat:
            risk = float(mdl.predict_proba(pd.DataFrame([feat]))[0][1])

        # ── Risk banner + gauge + metrics ─────────────────────────────────────
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.plotly_chart(plot_gauge(risk),
                            use_container_width=True, config={"displayModeBar": False})

            if risk is None:
                st.markdown(f"""
                <div class='risk-banner risk-unknown'>
                  ⚪ &nbsp; Not enough beats in this window
                </div>""", unsafe_allow_html=True)
            elif risk >= THRESHOLD:
                st.markdown(f"""
                <div class='risk-banner risk-high'>
                  ⚠️ &nbsp; HIGH RISK &nbsp;—&nbsp; PAF Detected &nbsp;
                  <span style='font-family:"JetBrains Mono",monospace'>{risk*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            elif risk >= 0.15:
                st.markdown(f"""
                <div class='risk-banner risk-medium'>
                  🟡 &nbsp; ELEVATED &nbsp;—&nbsp; Monitor closely &nbsp;
                  <span style='font-family:"JetBrains Mono",monospace'>{risk*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-banner risk-low'>
                  ✅ &nbsp; NORMAL &nbsp;—&nbsp; Healthy sinus rhythm &nbsp;
                  <span style='font-family:"JetBrains Mono",monospace'>{risk*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

        with col_right:
            if feat:
                st.markdown(f"<div style='font-size:11px;font-weight:700;color:{TEXT3};"
                            f"letter-spacing:0.08em;text-transform:uppercase;"
                            f"margin-bottom:10px'>HRV Metrics</div>",
                            unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("SDNN",    f"{safe_float(feat['sdnn'],1)} ms")
                m2.metric("RMSSD",   f"{safe_float(feat['rmssd'],1)} ms")
                m3.metric("pNN50",   f"{safe_float(feat['pnn50'],1)}%")
                m4.metric("Mean RR", f"{safe_float(feat['mean_rr'],0)} ms")
                m5,m6,m7,m8 = st.columns(4)
                pac = feat['pac_count']
                m5.metric("PAC Count", str(pac),
                          delta="elevated" if pac > 3 else None,
                          delta_color="inverse")
                m6.metric("LF/HF",  safe_float(feat['lf_hf_ratio'],2))
                m7.metric("SampEn", safe_float(feat['sampen'],3))
                m8.metric("SD1",    f"{safe_float(feat['sd1'],1)} ms")

        st.divider()

        # ── Bottom row ────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            if feat and risk is not None:
                st.plotly_chart(plot_shap(feat, mdl, expl),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown(f"""
                <p style='font-size:10px;color:{TEXT3};text-align:center;margin-top:-8px'>
                  🔴 Increases AFib risk &nbsp;·&nbsp; 🟢 Decreases risk
                </p>""", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ht-card' style='height:300px;display:flex;"
                            f"align-items:center;justify-content:center;color:{TEXT3};font-size:13px'>"
                            f"Not enough beats for SHAP</div>", unsafe_allow_html=True)

        with c2:
            rr = w["rr"]
            if len(rr) > 1:
                st.plotly_chart(plot_poincare(rr, label_key),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown(f"""
                <p style='font-size:10px;color:{TEXT3};text-align:center;margin-top:-8px'>
                  Tight cluster = regular &nbsp;·&nbsp; Scattered = irregular (AFib)
                </p>""", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ht-card' style='height:240px;display:flex;"
                            f"align-items:center;justify-content:center;color:{TEXT3};font-size:13px'>"
                            f"Insufficient RR intervals</div>", unsafe_allow_html=True)

        with c3:
            if len(rr) > 0:
                st.plotly_chart(plot_rr_series(rr, label_key),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(f"<div class='ht-card' style='height:240px;display:flex;"
                            f"align-items:center;justify-content:center;color:{TEXT3};font-size:13px'>"
                            f"No RR intervals detected</div>", unsafe_allow_html=True)

        # ── Auto-play ─────────────────────────────────────────────────────────
        if auto_play:
            time.sleep(play_speed)
            st.session_state["win_idx"] = (win_idx + 1) % n_win
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # UPLOAD CSV MODE
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        if not uploaded:
            st.markdown(f"""
            <div class='ht-card' style='text-align:center;padding:48px;'>
              <div style='font-size:32px;margin-bottom:12px'>📁</div>
              <div style='font-size:15px;font-weight:600;color:{TEXT};margin-bottom:6px'>
                Upload an RR interval CSV
              </div>
              <div style='font-size:12px;color:{TEXT3}'>
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
            st.plotly_chart(plot_gauge(risk), use_container_width=True,
                            config={"displayModeBar": False})
            cls = "risk-high" if risk >= THRESHOLD else "risk-medium" if risk >= 0.15 else "risk-low"
            icon = "⚠️" if risk >= THRESHOLD else "🟡" if risk >= 0.15 else "✅"
            label = "HIGH RISK — PAF Detected" if risk >= THRESHOLD else "ELEVATED" if risk >= 0.15 else "NORMAL"
            st.markdown(f"""
            <div class='risk-banner {cls}'>
              {icon} &nbsp; {label} &nbsp;
              <span style='font-family:"JetBrains Mono",monospace'>{risk*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown(f"<div style='font-size:11px;font-weight:700;color:{TEXT3};"
                        f"letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px'>"
                        f"HRV Metrics</div>", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("SDNN",    f"{safe_float(feat['sdnn'],1)} ms")
            m2.metric("RMSSD",   f"{safe_float(feat['rmssd'],1)} ms")
            m3.metric("pNN50",   f"{safe_float(feat['pnn50'],1)}%")
            m4.metric("Mean RR", f"{safe_float(feat['mean_rr'],0)} ms")

        st.divider()
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.plotly_chart(plot_shap(feat, mdl, expl), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.plotly_chart(plot_poincare(rr, "unknown"), use_container_width=True,
                            config={"displayModeBar": False})
        with c3:
            st.plotly_chart(plot_rr_series(rr, "unknown"), use_container_width=True,
                            config={"displayModeBar": False})


if __name__ == "__main__":
    main()