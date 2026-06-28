import { useState } from 'react';
import { Activity, HeartPulse, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [rrData, setRrData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // --- 1. DATA SIMULATION ---
  const simulateHealthyData = () => {
    // Generates normal resting heart rate intervals (~800ms)
    const normalData = Array.from({ length: 30 }, () => 
      Math.floor(Math.random() * (850 - 750 + 1)) + 750
    );
    setRrData(normalData);
    setResult(null);
    setError(null);
  };

  const simulatePafData = () => {
    // Generates irregular intervals typical of PAF (high variance)
    const irregularData = Array.from({ length: 30 }, () => 
      Math.floor(Math.random() * (1100 - 400 + 1)) + 400
    );
    setRrData(irregularData);
    setResult(null);
    setError(null);
  };

  // --- 2. API COMMUNICATION ENGINE ---
  const analyzeData = async () => {
    if (rrData.length === 0) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Sending data to your live Render Cloud Server
      const response = await fetch("https://hearttrack-backend.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rr_intervals: rrData })
      });
      
      if (!response.ok) {
        throw new Error("Diagnostic server is offline or returned an error.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- 3. USER INTERFACE ---
  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: '12px', borderBottom: '2px solid #eaeaea', paddingBottom: '1rem', marginBottom: '2rem' }}>
        <Activity size={32} color="#e11d48" />
        <h1 style={{ margin: 0, color: '#1f2937' }}>HeartTrack AI Clinical Dashboard</h1>
      </header>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
        <button 
          onClick={simulateHealthyData}
          style={{ padding: '0.75rem 1.5rem', backgroundColor: '#10b981', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Simulate Healthy Heart
        </button>
        <button 
          onClick={simulatePafData}
          style={{ padding: '0.75rem 1.5rem', backgroundColor: '#f59e0b', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Simulate PAF Heart
        </button>
      </div>

      <div style={{ backgroundColor: '#f9fafb', padding: '1.5rem', borderRadius: '8px', marginBottom: '2rem' }}>
        <h3 style={{ margin: '0 0 1rem 0', color: '#4b5563' }}>Current Patient Data (R-R Intervals)</h3>
        <p style={{ fontFamily: 'monospace', wordBreak: 'break-all', color: '#6b7280' }}>
          {rrData.length > 0 ? rrData.join(', ') + ' ...' : 'No data loaded. Click a simulation button above.'}
        </p>
      </div>

      <button 
        onClick={analyzeData}
        disabled={loading || rrData.length === 0}
        style={{ width: '100%', padding: '1rem', backgroundColor: rrData.length === 0 ? '#9ca3af' : '#2563eb', color: 'white', border: 'none', borderRadius: '8px', fontSize: '1.1rem', fontWeight: 'bold', cursor: rrData.length === 0 ? 'not-allowed' : 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}
      >
        <HeartPulse />
        {loading ? 'Analyzing Data in Cloud...' : 'Run AI Diagnosis'}
      </button>

      {error && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#fee2e2', color: '#b91c1c', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertCircle />
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      {result && (
        <div style={{ marginTop: '2rem', padding: '2rem', backgroundColor: result.risk_level === 'High' ? '#fee2e2' : '#d1fae5', border: `2px solid ${result.risk_level === 'High' ? '#f87171' : '#34d399'}`, borderRadius: '12px', textAlign: 'center' }}>
          {result.risk_level === 'High' ? <AlertCircle size={48} color="#ef4444" style={{ margin: '0 auto' }}/> : <CheckCircle2 size={48} color="#10b981" style={{ margin: '0 auto' }}/>}
          <h2 style={{ fontSize: '2rem', margin: '1rem 0 0.5rem 0', color: '#1f2937' }}>
            Risk Level: {result.risk_level}
          </h2>
          <p style={{ fontSize: '1.25rem', color: '#4b5563', margin: 0 }}>
            AI Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
          </p>
        </div>
      )}
    </div>
  );
}