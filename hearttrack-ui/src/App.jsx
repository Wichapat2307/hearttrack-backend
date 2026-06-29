import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity, HeartPulse, AlertCircle, CheckCircle2,
  UploadCloud, Wifi, WifiOff, Radio, FileText, ChevronDown,
  Play, Pause, SkipBack
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const API_URL = "https://hearttrack-backend.onrender.com";
const WS_URL  = API_URL.replace(/^http/, "ws") + "/ws/ecg";

const RISK_CONFIG = {
  normal:       { color: '#22c55e', bg: '#052e16', border: '#14532d', text: '#86efac', badgeBg: '#14532d' },
  paf_distant:  { color: '#f59e0b', bg: '#451a03', border: '#78350f', text: '#fcd34d', badgeBg: '#78350f' },
  paf_imminent: { color: '#ef4444', bg: '#450a0a', border: '#7f1d1d', text: '#fca5a5', badgeBg: '#7f1d1d' },
  unknown:      { color: '#71717a', bg: '#18181b', border: '#27272a', text: '#a1a1aa', badgeBg: '#27272a' },
};

function Badge({ label, labelKey }) {
  const c = RISK_CONFIG[labelKey] || RISK_CONFIG.unknown;
  return <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: c.badgeBg, color: c.text, whiteSpace: 'nowrap' }}>{label}</span>;
}

function MetricCard({ label, value, unit, highlight }) {
  return (
    <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 10, padding: '14px 16px' }}>
      <div style={{ fontSize: 11, color: '#71717a', marginBottom: 4, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: highlight || '#f4f4f5' }}>
        {value} <span style={{ fontSize: 12, fontWeight: 400, color: '#52525b' }}>{unit}</span>
      </div>
    </div>
  );
}

function RiskGauge({ probability }) {
  const pct   = Math.round((probability ?? 0) * 100);
  const color = pct >= 30 ? '#ef4444' : pct >= 15 ? '#f59e0b' : '#22c55e';
  const angle = -90 + (pct / 100) * 180;
  return (
    <svg viewBox="0 0 180 100" width="180" height="100">
      <path d="M 16 90 A 72 72 0 0 1 164 90" fill="none" stroke="#27272a" strokeWidth="16" strokeLinecap="round" />
      <path d="M 16 90 A 72 72 0 0 1 164 90" fill="none" stroke={color} strokeWidth="16" strokeLinecap="round"
        strokeDasharray={`${pct * 2.26} 226`} style={{ transition: 'all 0.5s ease' }} />
      <line x1="90" y1="90"
        x2={90 + 58 * Math.cos((angle - 180) * Math.PI / 180)}
        y2={90 + 58 * Math.sin((angle - 180) * Math.PI / 180)}
        stroke={color} strokeWidth="2.5" strokeLinecap="round" style={{ transition: 'all 0.5s ease' }} />
      <circle cx="90" cy="90" r="5" fill={color} />
      <text x="90" y="74" textAnchor="middle" fill={color} fontSize="22" fontWeight="700">{pct}%</text>
      <text x="90" y="88" textAnchor="middle" fill="#71717a" fontSize="9">AFib Risk</text>
    </svg>
  );
}

// ─── ANIMATED ECG CANVAS WITH SLIDING WINDOW ──────────────────────────────────
function ECGViewer({ signal, fsDs, rPeaks, windows, currentWindowIdx, labelKey }) {
  const canvasRef = useRef(null);
  const cfg = RISK_CONFIG[labelKey] || RISK_CONFIG.unknown;

  useEffect(() => {
    if (!signal?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');
    const W      = canvas.width;
    const H      = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // ECG paper grid
    const gx = W / 50;
    const gy = H / 10;
    ctx.lineWidth = 0.4;
    ctx.strokeStyle = '#1a2332';
    for (let x = 0; x <= W; x += gx) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += gy) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = '#1e3048';
    for (let x = 0; x <= W; x += gx * 5) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += gy * 5) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    const n   = signal.length;
    const pad = H * 0.12;
    const toX = (i) => (i / (n - 1)) * W;
    const toY = (v) => pad + ((1 - (v + 1) / 2)) * (H - pad * 2);

    // Sliding window highlight
    if (windows?.length && currentWindowIdx != null) {
      const win = windows[currentWindowIdx];
      if (win) {
        const x1  = toX(win.start_sample);
        const x2  = toX(win.end_sample);
        const risk = win.risk_probability;
        const hlColor = risk == null ? '#3b3b3b'
          : risk >= 0.3 ? 'rgba(239,68,68,0.12)'
          : risk >= 0.15 ? 'rgba(245,158,11,0.12)'
          : 'rgba(34,197,94,0.10)';
        const borderColor = risk == null ? '#444'
          : risk >= 0.3 ? '#ef4444'
          : risk >= 0.15 ? '#f59e0b'
          : '#22c55e';

        ctx.fillStyle = hlColor;
        ctx.fillRect(x1, 0, x2 - x1, H);
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x2, 0); ctx.lineTo(x2, H); ctx.stroke();
        ctx.setLineDash([]);

        // Window label
        if (risk != null) {
          ctx.fillStyle = borderColor;
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`${(risk * 100).toFixed(0)}%`, x1 + 4, 14);
        }
      }
    }

    // ECG signal line
    ctx.beginPath();
    ctx.strokeStyle = cfg.color;
    ctx.lineWidth   = 1.4;
    ctx.shadowColor = cfg.color;
    ctx.shadowBlur  = 3;
    signal.forEach((v, i) => {
      const x = toX(i);
      const y = toY(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // R-peak markers
    if (rPeaks?.length) {
      ctx.fillStyle = '#facc15';
      rPeaks.forEach(pi => {
        if (pi < signal.length) {
          ctx.beginPath();
          ctx.arc(toX(pi), toY(signal[pi]), 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    // Time axis labels
    const durSec = n / fsDs;
    ctx.fillStyle = '#52525b';
    ctx.font      = '9px monospace';
    ctx.shadowBlur = 0;
    for (let t = 0; t <= durSec; t += 5) {
      const x = (t / durSec) * W;
      ctx.fillText(`${t}s`, x + 2, H - 3);
    }

  }, [signal, rPeaks, windows, currentWindowIdx, labelKey]);

  return (
    <div style={{ background: '#060d18', border: `1px solid ${cfg.border}`, borderRadius: 10, overflow: 'hidden' }}>
      <canvas ref={canvasRef} width={780} height={190}
        style={{ width: '100%', height: 190, display: 'block' }} />
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [mode, setMode]       = useState('demo');  // 'demo' | 'upload' | 'live'
  const [error, setError]     = useState(null);

  // Demo state
  const [demoFiles, setDemoFiles]     = useState([]);
  const [selectedDemo, setSelectedDemo] = useState(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const [analyzing, setAnalyzing]     = useState(false);
  const [ecgResult, setEcgResult]     = useState(null);   // full /analyze_ecg response
  const [currentWin, setCurrentWin]   = useState(0);
  const [playing, setPlaying]         = useState(false);
  const playRef = useRef(null);

  // Upload state
  const [rrData, setRrData]     = useState([]);
  const [fileName, setFileName] = useState('');
  const [loading, setLoading]   = useState(false);
  const [batchResult, setBatchResult] = useState(null);

  // Live state
  const [wsConnected, setWsConnected] = useState(false);
  const [liveRR, setLiveRR]     = useState([]);
  const [liveHR, setLiveHR]     = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [bufferSize, setBufferSize] = useState(0);
  const wsRef    = useRef(null);
  const liveIdx  = useRef(0);

  useEffect(() => {
    fetch('/demo_files/demo_manifest.json')
      .then(r => r.json()).then(setDemoFiles).catch(() => setDemoFiles([]));
  }, []);

  // Sliding window animation
  useEffect(() => {
    if (playing && ecgResult?.windows?.length) {
      playRef.current = setInterval(() => {
        setCurrentWin(w => {
          if (w >= ecgResult.windows.length - 1) { setPlaying(false); return w; }
          return w + 1;
        });
      }, 800);
    } else {
      clearInterval(playRef.current);
    }
    return () => clearInterval(playRef.current);
  }, [playing, ecgResult]);

  // Load and analyze demo NPY
  const handleDemoSelect = async (demo) => {
    setShowDropdown(false);
    setSelectedDemo(demo);
    setEcgResult(null);
    setCurrentWin(0);
    setPlaying(false);
    setError(null);
    setAnalyzing(true);

    try {
      const npyRes = await fetch(demo.file);
      const npyBlob = await npyRes.blob();
      const formData = new FormData();
      formData.append('file', new File([npyBlob], 'ecg.npy', { type: 'application/octet-stream' }));
      formData.append('fs', demo.fs.toString());
      formData.append('window_sec', '10');
      formData.append('stride_sec', '5');

      const res  = await fetch(`${API_URL}/analyze_ecg`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Server error');
      const data = await res.json();
      setEcgResult(data);
      setCurrentWin(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setAnalyzing(false);
    }
  };

  // Upload CSV batch
  const handleFile = (file) => {
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      const vals = e.target.result.split(/[\n,]+/).map(Number).filter(n => !isNaN(n) && n > 0);
      setRrData(vals); setBatchResult(null); setError(null);
    };
    reader.readAsText(file);
  };

  const analyzeCSV = async () => {
    if (!rrData.length) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rr_intervals: rrData }),
      });
      if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || 'Error');
      setBatchResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  // WebSocket
  const connectWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    ws.onopen  = () => setWsConnected(true);
    ws.onclose = () => { setWsConnected(false); wsRef.current = null; };
    ws.onerror = () => setError('WebSocket failed.');
    ws.onmessage = ({ data }) => {
      try {
        const msg = JSON.parse(data);
        if (msg.type === 'rr_echo') { setLiveHR(msg.hr); setBufferSize(msg.buffer_size); setLiveRR(p => [...p, { i: liveIdx.current++, rr: msg.value }].slice(-80)); }
        if (msg.type === 'analysis') { setLiveResult(msg); setError(null); }
        if (msg.type === 'error')    setError(msg.message);
      } catch (_) {}
    };
    wsRef.current = ws;
  }, []);

  const disconnectWS = useCallback(() => {
    wsRef.current?.close();
    setWsConnected(false); setLiveRR([]); setLiveHR(null); setLiveResult(null); setBufferSize(0);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  // Current window data for display
  const currentWinData  = ecgResult?.windows?.[currentWin];
  const activeResult    = mode === 'demo' ? currentWinData : mode === 'live' ? liveResult : batchResult;
  const isAfib          = activeResult?.is_afib_imminent;
  const riskProb        = activeResult?.risk_probability;
  const resultCfg       = isAfib ? RISK_CONFIG.paf_imminent : RISK_CONFIG.normal;

  const demoGroups = demoFiles.reduce((acc, d) => { (acc[d.label_key] = acc[d.label_key] || []).push(d); return acc; }, {});
  const GROUP_ORDER  = ['normal', 'paf_distant', 'paf_imminent'];
  const GROUP_LABELS = { normal: '🟢 Healthy Control', paf_distant: '🟡 PAF Stable', paf_imminent: '🔴 PAF Imminent' };

  return (
    <div style={{ minHeight: '100vh', background: '#09090b', color: '#f4f4f5', fontFamily: "'Inter', system-ui, sans-serif", paddingBottom: 60 }}>

      {/* HEADER */}
      <header style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '18px 28px', borderBottom: '1px solid #18181b', position: 'sticky', top: 0, background: '#09090b', zIndex: 10 }}>
        <Activity size={24} color="#e11d48" />
        <span style={{ fontWeight: 700, fontSize: 17, letterSpacing: '-0.02em' }}>HeartTrack AI</span>
        <span style={{ fontSize: 10, fontWeight: 600, color: '#71717a', padding: '2px 8px', border: '1px solid #27272a', borderRadius: 4 }}>Clinical Dashboard</span>
        {mode === 'live' && wsConnected && (
          <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, color: '#4ade80' }}>
            <Radio size={13} style={{ animation: 'pulse 1.5s infinite' }} /> LIVE · {bufferSize} RR
          </span>
        )}
      </header>

      <div style={{ maxWidth: 860, margin: '0 auto', padding: '24px 20px' }}>

        {/* TABS */}
        <div style={{ display: 'flex', background: '#18181b', borderRadius: 10, padding: 4, marginBottom: 24 }}>
          {[{ key: 'demo', label: 'Demo Files' }, { key: 'upload', label: 'Upload CSV' }, { key: 'live', label: 'Live ESP32' }].map(({ key, label }) => (
            <button key={key} onClick={() => { setMode(key); setError(null); }} style={{
              flex: 1, padding: '9px 0', borderRadius: 7, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: 600,
              background: mode === key ? '#e11d48' : 'transparent',
              color: mode === key ? '#fff' : '#71717a', transition: 'all 0.2s',
            }}>{label}</button>
          ))}
        </div>

        {/* ── DEMO MODE ── */}
        {mode === 'demo' && (<>
          {/* Demo picker dropdown */}
          <div style={{ position: 'relative', marginBottom: 16 }}>
            <button onClick={() => setShowDropdown(v => !v)} style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              background: '#18181b', border: '1px solid #27272a', borderRadius: 8,
              color: '#a1a1aa', padding: '11px 16px', cursor: 'pointer', fontSize: 14,
            }}>
              <FileText size={15} />
              <span>{selectedDemo ? `${selectedDemo.short}` : 'Select a demo recording from PAFPDB'}</span>
              <ChevronDown size={14} style={{ marginLeft: 'auto', transform: showDropdown ? 'rotate(180deg)' : '', transition: '0.2s' }} />
            </button>

            {showDropdown && (
              <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 30, background: '#18181b', border: '1px solid #27272a', borderRadius: 8, marginTop: 4, maxHeight: 300, overflowY: 'auto' }}>
                {demoFiles.length === 0
                  ? <div style={{ padding: 16, color: '#71717a', fontSize: 13 }}>Run <code style={{ background: '#27272a', padding: '1px 5px', borderRadius: 3 }}>python extract_demo_npy.py</code> first.</div>
                  : GROUP_ORDER.filter(g => demoGroups[g]).map(g => (
                    <div key={g}>
                      <div style={{ padding: '7px 16px', fontSize: 10, fontWeight: 700, color: '#52525b', borderBottom: '1px solid #27272a', letterSpacing: '0.06em', textTransform: 'uppercase' }}>{GROUP_LABELS[g]}</div>
                      {demoGroups[g].map(d => (
                        <button key={d.file} onClick={() => handleDemoSelect(d)} style={{ display: 'flex', alignItems: 'center', gap: 12, width: '100%', textAlign: 'left', padding: '10px 16px', background: 'transparent', border: 'none', cursor: 'pointer', color: '#f4f4f5', borderBottom: '1px solid #27272a', fontSize: 13 }}>
                          <Badge label={d.badge} labelKey={d.label_key} />
                          <div>
                            <div style={{ fontWeight: 600 }}>{d.short}</div>
                            <div style={{ fontSize: 11, color: '#71717a', marginTop: 1 }}>{d.description}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                  ))}
              </div>
            )}
          </div>

          {/* Analyzing spinner */}
          {analyzing && (
            <div style={{ textAlign: 'center', padding: '32px 0', color: '#71717a', fontSize: 14 }}>
              <HeartPulse size={28} color="#e11d48" style={{ display: 'block', margin: '0 auto 12px', animation: 'pulse 1s infinite' }} />
              Uploading ECG to backend & running sliding window analysis…
            </div>
          )}

          {/* ECG viewer + controls */}
          {ecgResult && selectedDemo && (<>
            <ECGViewer
              signal={ecgResult.signal}
              fsDs={ecgResult.fs_ds}
              rPeaks={ecgResult.r_peaks}
              windows={ecgResult.windows}
              currentWindowIdx={currentWin}
              labelKey={selectedDemo.label_key}
            />

            {/* Playback controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10, marginBottom: 20 }}>
              <button onClick={() => { setCurrentWin(0); setPlaying(false); }} style={{ background: '#27272a', border: 'none', borderRadius: 6, padding: '7px 12px', cursor: 'pointer', color: '#a1a1aa' }}>
                <SkipBack size={15} />
              </button>
              <button onClick={() => setPlaying(p => !p)} style={{ background: playing ? '#7f1d1d' : '#e11d48', border: 'none', borderRadius: 6, padding: '7px 16px', cursor: 'pointer', color: '#fff', fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                {playing ? <><Pause size={14}/> Pause</> : <><Play size={14}/> Play Windows</>}
              </button>

              {/* Window scrubber */}
              <input type="range" min={0} max={ecgResult.windows.length - 1} value={currentWin}
                onChange={e => { setCurrentWin(Number(e.target.value)); setPlaying(false); }}
                style={{ flex: 1, accentColor: '#e11d48' }} />

              <span style={{ fontSize: 12, color: '#71717a', whiteSpace: 'nowrap' }}>
                Window {currentWin + 1} / {ecgResult.windows.length}
                {currentWinData && <> · {currentWinData.start_sec}s–{currentWinData.end_sec}s</>}
              </span>
            </div>
          </>)}
        </>)}

        {/* ── UPLOAD MODE ── */}
        {mode === 'upload' && (<>
          <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2rem', background: '#18181b', border: '2px dashed #27272a', borderRadius: 12, cursor: 'pointer', marginBottom: 14 }}>
            <UploadCloud size={38} color="#52525b" style={{ marginBottom: 10 }} />
            <span style={{ fontWeight: 600, color: '#a1a1aa' }}>{fileName ? `✓ ${fileName}` : 'Click to upload .csv (RR intervals)'}</span>
            <input type="file" accept=".csv" onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
          </label>
          {rrData.length > 0 && (
            <div style={{ background: '#18181b', borderRadius: 10, padding: '10px 14px', marginBottom: 14, border: '1px solid #27272a', fontSize: 12, color: '#52525b', fontFamily: 'monospace' }}>
              {rrData.length} intervals · {rrData.slice(0, 25).map(v => v.toFixed(0)).join(', ')}{rrData.length > 25 ? ' …' : ''}
            </div>
          )}
          <button onClick={analyzeCSV} disabled={loading || !rrData.length} style={{
            width: '100%', padding: '13px', borderRadius: 10, border: 'none',
            background: !rrData.length ? '#27272a' : '#e11d48',
            color: !rrData.length ? '#52525b' : '#fff',
            fontSize: 14, fontWeight: 700, cursor: !rrData.length ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          }}>
            <HeartPulse size={17} />
            {loading ? 'Analyzing…' : 'Run AI Diagnosis'}
          </button>
        </>)}

        {/* ── LIVE MODE ── */}
        {mode === 'live' && (<>
          <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 12, padding: 18, marginBottom: 18 }}>
            <p style={{ margin: '0 0 14px', fontSize: 13, color: '#71717a', lineHeight: 1.6 }}>
              ESP32 connects to <code style={{ background: '#27272a', padding: '2px 5px', borderRadius: 4, color: '#a1a1aa', fontSize: 12 }}>{WS_URL}</code>
            </p>
            <button onClick={wsConnected ? disconnectWS : connectWS} style={{
              padding: '10px 20px', borderRadius: 8, border: 'none', fontWeight: 700, fontSize: 13,
              background: wsConnected ? '#27272a' : '#e11d48', color: '#fff',
              display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer',
            }}>
              {wsConnected ? <><WifiOff size={15}/> Disconnect</> : <><Wifi size={15}/> Connect to ESP32</>}
            </button>
          </div>
          {liveRR.length > 0 && (
            <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 12, padding: 14, marginBottom: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: '#71717a' }}>LIVE RR STREAM</span>
                {liveHR && <span style={{ fontSize: 18, fontWeight: 700, color: '#e11d48' }}>{liveHR} <span style={{ fontSize: 11, color: '#71717a', fontWeight: 400 }}>BPM</span></span>}
              </div>
              <ResponsiveContainer width="100%" height={110}>
                <LineChart data={liveRR} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                  <XAxis dataKey="i" hide />
                  <YAxis domain={['auto','auto']} tick={{ fill: '#52525b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', color: '#f4f4f5', fontSize: 11 }} formatter={v => [`${v.toFixed(0)} ms`, 'RR']} labelFormatter={() => ''} />
                  <ReferenceLine y={800} stroke="#3f3f46" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="rr" stroke="#e11d48" dot={false} strokeWidth={1.6} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
          {wsConnected && !liveRR.length && (
            <div style={{ textAlign: 'center', color: '#52525b', padding: '40px 0', fontSize: 13 }}>
              <Radio size={28} style={{ display: 'block', margin: '0 auto 10px', opacity: 0.4 }} />
              Waiting for RR intervals from ESP32…
            </div>
          )}
        </>)}

        {/* ERROR */}
        {error && (
          <div style={{ marginTop: 16, padding: '12px 16px', background: '#450a0a', border: '1px solid #7f1d1d', borderRadius: 10, display: 'flex', gap: 10, color: '#fca5a5', fontSize: 13 }}>
            <AlertCircle size={15} style={{ marginTop: 1, flexShrink: 0 }} /> {error}
          </div>
        )}

        {/* RESULTS PANEL */}
        {activeResult && activeResult.risk_probability != null && (
          <div style={{ marginTop: 20, background: '#18181b', border: `1px solid ${resultCfg.border}`, borderRadius: 14, overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', background: resultCfg.bg, display: 'flex', alignItems: 'center', gap: 12 }}>
              {isAfib ? <AlertCircle size={22} color={resultCfg.color}/> : <CheckCircle2 size={22} color={resultCfg.color}/>}
              <div>
                <div style={{ fontWeight: 700, fontSize: 16, color: resultCfg.text }}>
                  {isAfib ? 'High Risk — PAF Detected' : 'Normal — Healthy Sinus Rhythm'}
                </div>
                <div style={{ fontSize: 12, color: '#71717a', marginTop: 2 }}>
                  Threshold: 30% · CatBoost
                  {selectedDemo && mode === 'demo' && <> · <Badge label={selectedDemo.badge} labelKey={selectedDemo.label_key}/></>}
                  {mode === 'demo' && currentWinData && <> · Window {currentWin + 1} ({currentWinData.start_sec}s–{currentWinData.end_sec}s)</>}
                  {mode === 'live' && <span style={{ marginLeft: 8, color: '#e11d48' }}>● LIVE</span>}
                </div>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <RiskGauge probability={riskProb} />
              </div>
            </div>

            {activeResult.biometrics && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10, padding: 16 }}>
                {[
                  { label: 'SDNN',      value: activeResult.biometrics.sdnn.toFixed(1),        unit: 'ms' },
                  { label: 'RMSSD',     value: activeResult.biometrics.rmssd.toFixed(1),       unit: 'ms' },
                  { label: 'pNN50',     value: activeResult.biometrics.pnn50.toFixed(1),       unit: '%'  },
                  { label: 'Mean RR',   value: activeResult.biometrics.mean_rr.toFixed(0),     unit: 'ms' },
                  { label: 'PAC Count', value: activeResult.biometrics.pac_count,               unit: 'beats', highlight: activeResult.biometrics.pac_count > 3 ? '#ef4444' : null },
                  { label: 'LF/HF',    value: activeResult.biometrics.lf_hf_ratio.toFixed(2), unit: ''   },
                  { label: 'SampEn',   value: activeResult.biometrics.sampen.toFixed(3),       unit: ''   },
                  { label: 'SD1',      value: activeResult.biometrics.sd1.toFixed(1),          unit: 'ms' },
                ].map(m => <MetricCard key={m.label} {...m} />)}
              </div>
            )}
          </div>
        )}
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}*{box-sizing:border-box}`}</style>
    </div>
  );
}