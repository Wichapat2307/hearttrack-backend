import { useState } from 'react';
import { Activity, HeartPulse, AlertCircle, CheckCircle2, UploadCloud } from 'lucide-react';

export default function App() {
  const [rrData, setRrData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [fileName, setFileName] = useState("");

  // --- 1. DATA UPLOAD ENGINE ---
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setFileName(file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      // Parses the CSV: splits by commas or newlines, converts to numbers, ignores empty data
      const dataArray = text.split(/[\n,]+/).map(Number).filter(n => !isNaN(n) && n !== 0);
      
      setRrData(dataArray);
      setResult(null);
      setError(null);
    };
    reader.readAsText(file);
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
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Diagnostic server is offline or returned an error.");
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

      {/* Upload Section */}
      <div style={{ marginBottom: '2rem' }}>
        <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2.5rem', backgroundColor: '#f8fafc', border: '2px dashed #cbd5e1', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s' }}>
          <UploadCloud size={48} color="#64748b" style={{ marginBottom: '1rem' }} />
          <span style={{ fontSize: '1.1rem', fontWeight: '500', color: '#334155' }}>
            {fileName ? `Loaded: ${fileName}` : 'Click to Upload .csv Dataset'}
          </span>
          <input type="file" accept=".csv" onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
      </div>

      <div style={{ backgroundColor: '#f9fafb', padding: '1.5rem', borderRadius: '8px', marginBottom: '2rem' }}>
        <h3 style={{ margin: '0 0 1rem 0', color: '#4b5563' }}>Current Patient Data (R-R Intervals)</h3>
        <p style={{ fontFamily: 'monospace', wordBreak: 'break-all', color: '#6b7280', maxHeight: '100px', overflowY: 'auto' }}>
          {rrData.length > 0 
            ? `${rrData.slice(0, 50).join(', ')} ${rrData.length > 50 ? '...' : ''} (${rrData.length} total intervals loaded)` 
            : 'No data loaded. Upload a file above.'}
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

      {/* Dynamic Results Display */}
      {result && (
        <div style={{ marginTop: '2rem', padding: '2rem', backgroundColor: result.is_afib_imminent ? '#fee2e2' : '#d1fae5', border: `2px solid ${result.is_afib_imminent ? '#f87171' : '#34d399'}`, borderRadius: '12px', textAlign: 'center' }}>
          {result.is_afib_imminent ? <AlertCircle size={48} color="#ef4444" style={{ margin: '0 auto' }}/> : <CheckCircle2 size={48} color="#10b981" style={{ margin: '0 auto' }}/>}
          
          <h2 style={{ fontSize: '2rem', margin: '1rem 0 0.5rem 0', color: '#1f2937' }}>
            {result.is_afib_imminent ? 'High Risk (PAF Detected)' : 'Normal (Healthy)'}
          </h2>
          
          <p style={{ fontSize: '1.25rem', color: '#4b5563', margin: 0 }}>
            AI Probability: <strong>{(result.risk_probability * 100).toFixed(1)}%</strong>
          </p>

          {/* Show the clinical math features your backend calculated! */}
          <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: `1px solid ${result.is_afib_imminent ? '#fca5a5' : '#6ee7b7'}`}}>
              <p style={{ margin: 0, fontSize: '0.9rem', color: '#4b5563' }}>
                <strong>Key Biometrics Analyzed:</strong><br/>
                Heart Rate Variability (SDNN): {result.biometrics.sdnn.toFixed(2)} ms<br/>
                Premature Atrial Contractions (PACs): {result.biometrics.pac_count}
              </p>
          </div>
        </div>
      )}
    </div>
  );
}