import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity, HeartPulse, AlertCircle, CheckCircle2,
  UploadCloud, Wifi, WifiOff, Radio, FileText, ChevronDown, Clock
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

const API_URL = "https://hearttrack-backend.onrender.com";
const WS_URL  = API_URL.replace(/^http/, "ws") + "/ws/ecg";

// Risk config per label_key
const RISK_CONFIG = {
  normal:       { color: '#22c55e', bg: '#052e16', border: '#14532d', text: '#86efac', badgeBg: '#14532d' },
  paf_distant:  { color: '#f59e0b', bg: '#451a03', border: '#78350f', text: '#fcd34d', badgeBg: '#78350f' },
  paf_imminent: { color: '#ef4444', bg: '#450a0a', border: '#7f1d1d', text: '#fca5a5', badgeBg: '#7f1d1d' },
  training:     { color: '#a78bfa', bg: '#1e1b4b', border: '#3730a3', text: '#c4b5fd', badgeBg: '#3730a3' },
  unknown:      { color: '#71717a', bg: '#18181b', border: '#27272a', text: '#a1a1aa', badgeBg: '#27272a' },
};

function parseCSV(text) {
  return text
    .split('\n')
    .filter(l => l && !l.startsWith('#') && l.trim() !== 'rr_ms')
    .flatMap(l => l.split(','))
    .map(Number)
    .filter(n => !isNaN(n) && n > 0);
}

function Badge({ label, labelKey }) {
  const cfg = RISK_CONFIG[labelKey] || RISK_CONFIG.unknown;
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
      background: cfg.badgeBg, color: cfg.text, whiteSpace: 'nowrap',
    }}>{label}</span>
  );
}

function StatusBadge({ connected, bufferSize }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 14px', borderRadius: 999,
      background: connected ? '#052e16' : '#1c1917',
      border: `1px solid ${connected ? '#16a34a' : '#44403c'}`,
      fontSize: 13, fontWeight: 600,
      color: connected ? '#4ade80' : '#78716c',
    }}>
      {connected
        ? <><Radio size={14} style={{ animation: 'pulse 1.5s infinite' }} /> LIVE · {bufferSize} RR</>
        : <><WifiOff size={14} /> Disconnected</>}
    </div>
  );
}

function RiskGauge({ probability }) {
  const pct = Math.round(probability * 100);
  const color = pct >= 30 ? '#ef4444' : pct >= 15 ? '#f59e0b' : '#22c55e';
  const angle = -90 + (pct / 100) * 180;
  return (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox="0 0 200 110" width="200" height="110">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#27272a" strokeWidth="18" strokeLinecap="round" />
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none"
          stroke={color} strokeWidth="18" strokeLinecap="round"
          strokeDasharray={`${pct * 2.51} 251`}
          style={{ transition: 'stroke-dasharray 0.6s ease, stroke 0.4s' }} />
        <line x1="100" y1="100"
          x2={100 + 65 * Math.cos((angle - 180) * Math.PI / 180)}
          y2={100 + 65 * Math.sin((angle - 180) * Math.PI / 180)}
          stroke={color} strokeWidth="3" strokeLinecap="round"
          style={{ transition: 'all 0.6s ease' }} />
        <circle cx="100" cy="100" r="6" fill={color} />
        <text x="18"  y="115" fill="#71717a" fontSize="11">0%</text>
        <text x="170" y="115" fill="#71717a" fontSize="11">100%</text>
        <text x="100" y="82" textAnchor="middle" fill={color} fontSize="26" fontWeight="700">{pct}%</text>
        <text x="100" y="100" textAnchor="middle" fill="#a1a1aa" fontSize="10">AFib Risk</text>
      </svg>
    </div>
  );
}

function MetricCard({ label, value, unit, highlight }) {
  return (
    <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 10, padding: '14px 16px' }}>
      <div style={{ fontSize: 11, color: '#71717a', marginBottom: 4, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: highlight || '#f4f4f5' }}>
        {value} <span style={{ fontSize: 13, fontWeight: 400, color: '#52525b' }}>{unit}</span>
      </div>
    </div>
  );
}

function LiveChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={120}>
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <XAxis dataKey="i" hide />
        <YAxis domain={['auto', 'auto']} tick={{ fill: '#52525b', fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#18181b', border: '1px solid #27272a', color: '#f4f4f5', fontSize: 12 }}
          formatter={(v) => [`${v.toFixed(0)} ms`, 'RR']}
          labelFormatter={() => ''} />
        <ReferenceLine y={800} stroke="#3f3f46" strokeDasharray="3 3" />
        <Line type="monotone" dataKey="rr" stroke="#e11d48" dot={false} strokeWidth={1.8} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function App() {
  const [mode, setMode]         = useState('upload');
  const [rrData, setRrData]     = useState([]);
  const [fileName, setFileName] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [result, setResult]     = useState(null);
  const [showDemo, setShowDemo] = useState(false);

  // Demo manifest loaded from JSON
  const [demoFiles, setDemoFiles]   = useState([]);
  const [selectedDemo, setSelectedDemo] = useState(null);

  // Live state
  const [wsConnected, setWsConnected] = useState(false);
  const [liveRR, setLiveRR]           = useState([]);
  const [liveHR, setLiveHR]           = useState(null);
  const [liveResult, setLiveResult]   = useState(null);
  const [bufferSize, setBufferSize]   = useState(0);
  const wsRef       = useRef(null);
  const liveIdxRef  = useRef(0);

  // Load demo manifest on mount
  useEffect(() => {
    fetch('/demo_files/demo_manifest.json')
      .then(r => r.json())
      .then(setDemoFiles)
      .catch(() => setDemoFiles([]));
  }, []);

  // Group demos by label_key for the dropdown
  const demoGroups = demoFiles.reduce((acc, d) => {
    if (!acc[d.label_key]) acc[d.label_key] = [];
    acc[d.label_key].push(d);
    return acc;
  }, {});

  const GROUP_ORDER = ['normal', 'paf_distant', 'paf_imminent', 'training'];
  const GROUP_LABELS = {
    normal:       '🟢 Normal Sinus Rhythm',
    paf_distant:  '🟡 PAF — Distant (~45 min away)',
    paf_imminent: '🔴 PAF — Imminent',
    training:     '🟣 Training Set',
  };

  // WebSocket
  const connectWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    ws.onopen  = () => setWsConnected(true);
    ws.onclose = () => { setWsConnected(false); wsRef.current = null; };
    ws.onerror = () => setError('WebSocket connection failed.');
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'rr_echo') {
          setLiveHR(msg.hr);
          setBufferSize(msg.buffer_size);
          setLiveRR(prev => [...prev, { i: liveIdxRef.current++, rr: msg.value }].slice(-80));
        }
        if (msg.type === 'analysis') { setLiveResult(msg); setError(null); }
        if (msg.type === 'error')    { setError(msg.message); }
      } catch (_) {}
    };
    wsRef.current = ws;
  }, []);

  const disconnectWS = useCallback(() => {
    wsRef.current?.close();
    setWsConnected(false);
    setLiveRR([]); setLiveHR(null); setLiveResult(null); setBufferSize(0);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  const handleFile = (file) => {
    if (!file) return;
    setFileName(file.name);
    setSelectedDemo(null);
    const reader = new FileReader();
    reader.onload = (e) => { setRrData(parseCSV(e.target.result)); setResult(null); setError(null); };
    reader.readAsText(file);
  };

  const handleDemoSelect = async (demo) => {
    setShowDemo(false);
    setSelectedDemo(demo);
    setFileName(demo.record);
    setResult(null); setError(null);
    try {
      const res  = await fetch(demo.file);
      const text = await res.text();
      setRrData(parseCSV(text));
    } catch {
      setError('Could not load demo file. Run extract_demo_csvs.py first.');
    }
  };

  const analyzeData = async () => {
    if (!rrData.length) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rr_intervals: rrData }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Server error.');
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const displayResult = mode === 'live' ? liveResult : result;
  const isAfib = displayResult?.is_afib_imminent;
  const resultCfg = isAfib ? RISK_CONFIG.paf_imminent : RISK_CONFIG.normal;

  return (
    <div style={{ minHeight: '100vh', background: '#09090b', color: '#f4f4f5', fontFamily: "'Inter', system-ui, sans-serif", padding: '0 0 60px' }}>
      {/* HEADER */}
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 28px', borderBottom: '1px solid #18181b', background: '#09090b', position: 'sticky', top: 0, zIndex: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Activity size={26} color="#e11d48" />
          <span style={{ fontWeight: 700, fontSize: 18, letterSpacing: '-0.02em' }}>HeartTrack AI</span>
          <span style={{ fontSize: 11, fontWeight: 600, color: '#71717a', padding: '2px 8px', border: '1px solid #27272a', borderRadius: 4 }}>Clinical Dashboard</span>
        </div>
        {mode === 'live' && <StatusBadge connected={wsConnected} bufferSize={bufferSize} />}
      </header>

      <div style={{ maxWidth: 820, margin: '0 auto', padding: '28px 20px' }}>

        {/* MODE TABS */}
        <div style={{ display: 'flex', background: '#18181b', borderRadius: 10, padding: 4, marginBottom: 28 }}>
          {[{ key: 'upload', icon: <UploadCloud size={15}/>, label: 'Upload CSV' },
            { key: 'live',   icon: <Wifi size={15}/>,        label: 'Live ESP32' }]
            .map(({ key, icon, label }) => (
              <button key={key} onClick={() => { setMode(key); setError(null); }} style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                padding: '10px 0', borderRadius: 7, border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
                background: mode === key ? '#e11d48' : 'transparent',
                color: mode === key ? '#fff' : '#71717a', transition: 'all 0.2s',
              }}>{icon} {label}</button>
            ))}
        </div>

        {/* UPLOAD MODE */}
        {mode === 'upload' && (<>
          {/* Upload dropzone */}
          <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2rem', background: '#18181b', border: '2px dashed #27272a', borderRadius: 12, cursor: 'pointer', marginBottom: 12 }}>
            <UploadCloud size={40} color="#52525b" style={{ marginBottom: 12 }} />
            <span style={{ fontWeight: 600, color: '#a1a1aa' }}>
              {fileName && !selectedDemo ? `✓ ${fileName}` : 'Click to upload your .csv file'}
            </span>
            <span style={{ fontSize: 12, color: '#52525b', marginTop: 4 }}>One RR interval (ms) per line, or comma-separated</span>
            <input type="file" accept=".csv" onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
          </label>

          {/* Demo picker */}
          <div style={{ marginBottom: 20, position: 'relative' }}>
            <button onClick={() => setShowDemo(v => !v)} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              background: '#18181b', border: '1px solid #27272a', borderRadius: 8,
              color: '#a1a1aa', padding: '10px 16px', cursor: 'pointer', fontSize: 14, width: '100%',
            }}>
              <FileText size={15} />
              <span>{selectedDemo ? `Demo: ${selectedDemo.record} — ${selectedDemo.short}` : 'Or try a real PAFPDB demo file'}</span>
              <ChevronDown size={14} style={{ marginLeft: 'auto', transform: showDemo ? 'rotate(180deg)' : '', transition: '0.2s' }} />
            </button>

            {showDemo && (
              <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 20, background: '#18181b', border: '1px solid #27272a', borderRadius: 8, marginTop: 4, maxHeight: 320, overflowY: 'auto' }}>
                {demoFiles.length === 0 ? (
                  <div style={{ padding: '16px', color: '#71717a', fontSize: 13 }}>
                    No demo files found. Run <code style={{ background: '#27272a', padding: '1px 6px', borderRadius: 3 }}>python extract_demo_csvs.py</code> first.
                  </div>
                ) : (
                  GROUP_ORDER.filter(g => demoGroups[g]).map(g => (
                    <div key={g}>
                      <div style={{ padding: '8px 16px', fontSize: 11, fontWeight: 700, color: '#52525b', borderBottom: '1px solid #27272a', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                        {GROUP_LABELS[g]}
                      </div>
                      {demoGroups[g].map(demo => (
                        <button key={demo.file} onClick={() => handleDemoSelect(demo)} style={{
                          display: 'flex', alignItems: 'center', gap: 12, width: '100%',
                          textAlign: 'left', padding: '10px 16px', background: 'transparent',
                          border: 'none', cursor: 'pointer', color: '#f4f4f5',
                          borderBottom: '1px solid #27272a', fontSize: 13,
                        }}>
                          <Badge label={demo.badge} labelKey={demo.label_key} />
                          <div>
                            <div style={{ fontWeight: 600 }}>{demo.record}</div>
                            <div style={{ fontSize: 11, color: '#71717a', marginTop: 1 }}>{demo.description}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Selected demo info card */}
          {selectedDemo && (
            <div style={{
              marginBottom: 16, padding: '12px 16px', borderRadius: 10,
              background: (RISK_CONFIG[selectedDemo.label_key] || RISK_CONFIG.unknown).bg,
              border: `1px solid ${(RISK_CONFIG[selectedDemo.label_key] || RISK_CONFIG.unknown).border}`,
              display: 'flex', alignItems: 'flex-start', gap: 12,
            }}>
              <Clock size={16} color={(RISK_CONFIG[selectedDemo.label_key] || RISK_CONFIG.unknown).color} style={{ marginTop: 2, flexShrink: 0 }} />
              <div>
                <div style={{ fontWeight: 600, fontSize: 14, color: (RISK_CONFIG[selectedDemo.label_key] || RISK_CONFIG.unknown).text }}>
                  {selectedDemo.short}
                </div>
                <div style={{ fontSize: 12, color: '#71717a', marginTop: 2 }}>{selectedDemo.description}</div>
              </div>
            </div>
          )}

          {/* Data preview */}
          {rrData.length > 0 && (
            <div style={{ background: '#18181b', borderRadius: 10, padding: '12px 16px', marginBottom: 16, border: '1px solid #27272a' }}>
              <span style={{ fontSize: 12, color: '#71717a' }}>{rrData.length} intervals · </span>
              <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#52525b' }}>
                {rrData.slice(0, 30).map(v => v.toFixed(0)).join(', ')}{rrData.length > 30 ? ' …' : ''}
              </span>
            </div>
          )}

          <button onClick={analyzeData} disabled={loading || !rrData.length} style={{
            width: '100%', padding: '14px', borderRadius: 10, border: 'none',
            background: !rrData.length ? '#27272a' : '#e11d48',
            color: !rrData.length ? '#52525b' : '#fff',
            fontSize: 15, fontWeight: 700, cursor: !rrData.length ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
          }}>
            <HeartPulse size={18} />
            {loading ? 'Analyzing…' : 'Run AI Diagnosis'}
          </button>
        </>)}

        {/* LIVE MODE */}
        {mode === 'live' && (<>
          <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 12, padding: 20, marginBottom: 20 }}>
            <p style={{ margin: '0 0 16px', fontSize: 14, color: '#71717a', lineHeight: 1.6 }}>
              ESP32 connects to <code style={{ background: '#27272a', padding: '2px 6px', borderRadius: 4, color: '#a1a1aa' }}>{WS_URL}</code> and sends <code style={{ background: '#27272a', padding: '2px 6px', borderRadius: 4, color: '#a1a1aa' }}>{`{"type":"rr","value":843.5}`}</code>
            </p>
            <button onClick={wsConnected ? disconnectWS : connectWS} style={{
              padding: '12px 24px', borderRadius: 8, border: 'none', fontWeight: 700, fontSize: 14, cursor: 'pointer',
              background: wsConnected ? '#27272a' : '#e11d48', color: '#fff',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              {wsConnected ? <><WifiOff size={16}/> Disconnect</> : <><Wifi size={16}/> Connect to ESP32</>}
            </button>
          </div>

          {liveRR.length > 0 && (
            <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 12, padding: 16, marginBottom: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#71717a' }}>LIVE RR STREAM</span>
                {liveHR && <span style={{ fontSize: 20, fontWeight: 700, color: '#e11d48' }}>{liveHR} <span style={{ fontSize: 12, color: '#71717a', fontWeight: 400 }}>BPM</span></span>}
              </div>
              <LiveChart data={liveRR} />
            </div>
          )}

          {wsConnected && liveRR.length === 0 && (
            <div style={{ textAlign: 'center', color: '#52525b', padding: '40px 0', fontSize: 14 }}>
              <Radio size={32} style={{ display: 'block', margin: '0 auto 12px', opacity: 0.4 }} />
              Connected — waiting for RR intervals from ESP32…
            </div>
          )}
        </>)}

        {/* ERROR */}
        {error && (
          <div style={{ marginTop: 20, padding: '14px 16px', background: '#450a0a', border: '1px solid #7f1d1d', borderRadius: 10, display: 'flex', alignItems: 'flex-start', gap: 10, color: '#fca5a5', fontSize: 14 }}>
            <AlertCircle size={16} style={{ marginTop: 1, flexShrink: 0 }} />
            {error}
          </div>
        )}

        {/* RESULTS */}
        {displayResult && (
          <div style={{ marginTop: 24, background: '#18181b', border: `1px solid ${resultCfg.border}`, borderRadius: 14, overflow: 'hidden' }}>
            <div style={{ padding: '18px 24px', background: resultCfg.bg, display: 'flex', alignItems: 'center', gap: 14 }}>
              {isAfib ? <AlertCircle size={24} color={resultCfg.color} /> : <CheckCircle2 size={24} color={resultCfg.color} />}
              <div>
                <div style={{ fontWeight: 700, fontSize: 18, color: resultCfg.text }}>
                  {isAfib ? 'High Risk — PAF Detected' : 'Normal — Healthy Sinus Rhythm'}
                </div>
                <div style={{ fontSize: 13, color: '#71717a', marginTop: 2 }}>
                  Threshold: {(displayResult.clinical_threshold * 100).toFixed(0)}% · Model: CatBoost
                  {selectedDemo && <> · <Badge label={selectedDemo.badge} labelKey={selectedDemo.label_key} /></>}
                  {mode === 'live' && <span style={{ marginLeft: 8, color: '#e11d48' }}>● LIVE</span>}
                </div>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <RiskGauge probability={displayResult.risk_probability} />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, padding: 20 }}>
              {[
                { label: 'SDNN',      value: displayResult.biometrics.sdnn.toFixed(1),        unit: 'ms'    },
                { label: 'RMSSD',     value: displayResult.biometrics.rmssd.toFixed(1),       unit: 'ms'    },
                { label: 'pNN50',     value: displayResult.biometrics.pnn50.toFixed(1),       unit: '%'     },
                { label: 'Mean RR',   value: displayResult.biometrics.mean_rr.toFixed(0),     unit: 'ms'    },
                { label: 'PAC Count', value: displayResult.biometrics.pac_count,               unit: 'beats', highlight: displayResult.biometrics.pac_count > 3 ? '#ef4444' : null },
                { label: 'LF/HF',    value: displayResult.biometrics.lf_hf_ratio.toFixed(2), unit: ''      },
                { label: 'SampEn',   value: displayResult.biometrics.sampen.toFixed(3),       unit: ''      },
                { label: 'SD1',      value: displayResult.biometrics.sd1.toFixed(1),          unit: 'ms'    },
              ].map(m => <MetricCard key={m.label} {...m} />)}
            </div>
          </div>
        )}
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}*{box-sizing:border-box}`}</style>
    </div>
  );
}