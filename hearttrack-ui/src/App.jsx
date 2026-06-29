import React, { useState, useEffect, useRef } from 'react';
import { Heart, Activity, AlertCircle, CheckCircle, FileText, Zap, ShieldCheck, ShieldAlert, ChevronDown, HeartPulse } from 'lucide-react';
import { LineChart, Line, YAxis, ResponsiveContainer } from 'recharts';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeDemo, setActiveDemo] = useState(null);
  
  // State for the "Live" animated plot
  const [livePlotData, setLivePlotData] = useState([]);
  const fullPlotBuffer = useRef([]);
  const animationRef = useRef(null);

  // Stop animation if we load a new file
  useEffect(() => {
    return () => clearInterval(animationRef.current);
  }, []);

  const loadDemoFile = async (fileName, title) => {
    setActiveDemo(title);
    setLoading(true);
    setError(null);
    setResult(null);
    setLivePlotData([]);
    clearInterval(animationRef.current);

    try {
      // 1. Fetch the binary .npy file from the public folder
      const fileResponse = await fetch(`/demo_files/${fileName}`);
      if (!fileResponse.ok) throw new Error("Demo file not found. Did you run the extraction script?");
      const blob = await fileResponse.blob();

      // 2. Wrap it as a FormData upload for FastAPI
      const formData = new FormData();
      formData.append("file", blob, fileName);

      // 3. Send to Backend
      const apiResponse = await fetch("http://127.0.0.1:8000/analyze_ecg", {
        method: "POST",
        body: formData,
      });

      if (!apiResponse.ok) throw new Error("Backend processing failed.");
      
      const data = await apiResponse.json();
      setResult(data);
      
      // 4. Start the "Live Monitor" Animation
      if (data.live_plot_data) {
        fullPlotBuffer.current = data.live_plot_data;
        let index = 0;
        const speed = 2; // Add points per tick
        
        animationRef.current = setInterval(() => {
          setLivePlotData(prev => {
            const nextPoints = fullPlotBuffer.current.slice(index, index + speed).map(val => ({ v: val }));
            index += speed;
            
            // Loop the animation if we reach the end of the snippet
            if (index >= fullPlotBuffer.current.length) {
              index = 0; 
            }
            
            // Keep a rolling window of 150 points on screen
            const newArray = [...prev, ...nextPoints];
            if (newArray.length > 150) return newArray.slice(newArray.length - 150);
            return newArray;
          });
        }, 30); // 30ms refresh rate for smooth sweeping
      }

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-4 sm:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* HEADER */}
        <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
          <div className="flex items-center space-x-4 mb-4 sm:mb-0">
            <div className="bg-rose-50 p-3 rounded-xl">
              <Activity className="w-8 h-8 text-rose-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">HeartTrack E2E</h1>
              <p className="text-sm text-slate-500 font-medium">Raw ECG Signal Processing & AI Diagnosis</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 px-4 py-2 bg-emerald-50 text-emerald-700 rounded-full text-sm font-semibold border border-emerald-100 shadow-sm">
            <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse" />
            <span>Signal API Online</span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* LEFT PANEL: DEMO SELECTOR */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <h2 className="text-sm font-bold text-slate-900 mb-4 flex items-center uppercase tracking-wider">
                <FileText className="w-4 h-4 mr-2 text-slate-400" />
                Select Patient (.npy)
              </h2>
              
              <div className="space-y-6">
                
                {/* Healthy Group */}
                <div>
                  <div className="text-xs font-bold text-emerald-600 mb-2">🟢 HEALTHY CONTROLS</div>
                  <div className="space-y-2">
                    <button onClick={() => loadDemoFile('t03_healthy_control.npy', 'Control Patient t03')} className="w-full text-left px-3 py-2 text-sm bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition">Patient t03 (0-3m)</button>
                  </div>
                </div>

                {/* Stable Group */}
                <div>
                  <div className="text-xs font-bold text-amber-600 mb-2">🟡 STABLE PAF BASELINE</div>
                  <div className="space-y-2">
                    <button onClick={() => loadDemoFile('t01_paf_baseline.npy', 'Stable PAF t01')} className="w-full text-left px-3 py-2 text-sm bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition">Patient t01 (0-3m)</button>
                  </div>
                </div>

                {/* Critical Group */}
                <div>
                  <div className="text-xs font-bold text-rose-600 mb-2">🔴 IMMINENT AFIB EPISODE</div>
                  <div className="space-y-2">
                    <button onClick={() => loadDemoFile('t02_paf_imminent.npy', 'Imminent Attack t02')} className="w-full text-left px-3 py-2 text-sm bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-900 rounded-lg transition font-medium shadow-sm">Patient t02 (27-30m)</button>
                  </div>
                </div>

              </div>

              {loading && (
                <div className="mt-6 p-4 bg-slate-800 rounded-xl text-center">
                  <Activity className="w-6 h-6 text-emerald-400 animate-spin mx-auto mb-2"/>
                  <span className="text-xs font-semibold text-slate-200">Extracting HRV & Computing Risk...</span>
                </div>
              )}

              {error && (
                <div className="mt-4 p-3 bg-red-50 text-red-600 text-sm rounded-xl border border-red-200 flex items-start">
                  <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                  <p>{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT PANEL: DASHBOARD */}
          <div className="lg:col-span-3">
            {result ? (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                
                {/* 1. LIVE ECG MONITOR */}
                <div className="bg-[#0a0a0a] rounded-2xl shadow-xl border border-slate-800 overflow-hidden relative">
                   <div className="absolute top-4 left-4 z-10 flex items-center space-x-3">
                      <HeartPulse className="text-emerald-400 w-5 h-5" />
                      <span className="text-emerald-400 font-mono text-sm tracking-widest">{activeDemo} - LEAD I</span>
                   </div>
                   
                   <div className="h-48 w-full pt-10">
                     <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={livePlotData}>
                           <YAxis domain={['auto', 'auto']} hide={true} />
                           {/* The glowing ECG trace */}
                           <Line 
                             type="monotone" 
                             dataKey="v" 
                             stroke="#10b981" 
                             strokeWidth={2} 
                             dot={false} 
                             isAnimationActive={false} // Custom animation handles motion
                             style={{ filter: 'drop-shadow(0px 0px 4px rgba(16, 185, 129, 0.5))' }}
                           />
                        </LineChart>
                     </ResponsiveContainer>
                   </div>
                   
                   {/* Monitor Grid Overlay */}
                   <div className="absolute inset-0 pointer-events-none opacity-20" style={{ backgroundImage: 'linear-gradient(#22c55e 1px, transparent 1px), linear-gradient(90deg, #22c55e 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
                </div>

                {/* 2. PRIMARY DIAGNOSIS CARD */}
                <div className={`p-6 sm:p-8 rounded-2xl shadow-sm border ${result.is_afib_imminent ? 'bg-rose-50 border-rose-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  <div className="flex items-start sm:items-center space-x-4">
                    {result.is_afib_imminent ? (
                      <div className="bg-rose-100 p-4 rounded-full">
                        <ShieldAlert className="w-10 h-10 text-rose-600" />
                      </div>
                    ) : (
                      <div className="bg-emerald-100 p-4 rounded-full">
                        <ShieldCheck className="w-10 h-10 text-emerald-600" />
                      </div>
                    )}
                    
                    <div className="flex-1">
                      <h2 className={`text-2xl font-bold ${result.is_afib_imminent ? 'text-rose-700' : 'text-emerald-700'}`}>
                        {result.is_afib_imminent ? 'High Risk: AFib Imminent' : 'Normal Sinus Rhythm'}
                      </h2>
                      <p className={`text-sm mt-1 font-medium ${result.is_afib_imminent ? 'text-rose-600/80' : 'text-emerald-600/80'}`}>
                        {result.is_afib_imminent 
                          ? 'CatBoost detected critical ectopic triggers within chaotic HRV environment.' 
                          : 'No critical triggers detected. Heart rate variability is within safe parameters.'}
                      </p>
                    </div>

                    <div className="hidden sm:block text-right">
                       <div className={`text-4xl font-black ${result.is_afib_imminent ? 'text-rose-600' : 'text-emerald-600'}`}>
                         {(result.risk_probability * 100).toFixed(1)}%
                       </div>
                       <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mt-1">Confidence Score</div>
                    </div>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="mt-8 relative">
                     <div className="h-3 w-full bg-slate-200/50 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ${result.is_afib_imminent ? 'bg-gradient-to-r from-rose-400 to-rose-600' : 'bg-gradient-to-r from-emerald-400 to-emerald-600'}`}
                          style={{ width: `${result.risk_probability * 100}%` }}
                        />
                     </div>
                     <div 
                        className="absolute top-4 w-0.5 h-4 bg-slate-800 z-10 flex flex-col items-center"
                        style={{ left: `${result.clinical_threshold * 100}%` }}
                     >
                        <span className="text-[10px] font-bold text-slate-600 mt-2 bg-white px-2 py-0.5 shadow-sm rounded-full border border-slate-200">Threshold ({result.clinical_threshold})</span>
                     </div>
                  </div>
                </div>

                {/* 3. BIOMETRICS GRID */}
                <h3 className="text-sm font-bold text-slate-400 px-1 pt-2 uppercase tracking-widest">Extracted Clinical Biometrics</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className={`p-4 rounded-xl border shadow-sm ${result.biometrics.pac_count > 3 ? 'bg-rose-50 border-rose-100' : 'bg-white border-slate-100'}`}>
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1">PAC Count (Sparks)</p>
                    <p className={`text-2xl font-black ${result.biometrics.pac_count > 3 ? 'text-rose-700' : 'text-slate-900'}`}>{result.biometrics.pac_count}</p>
                    <p className="text-[10px] font-semibold text-slate-500 mt-1 uppercase">Ectopic triggers</p>
                  </div>
                  <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1">RMSSD (Stress)</p>
                    <p className="text-2xl font-black text-slate-900">{result.biometrics.rmssd.toFixed(1)} <span className="text-sm font-normal text-slate-400">ms</span></p>
                    <p className="text-[10px] font-semibold text-slate-500 mt-1 uppercase">Beat variance</p>
                  </div>
                  <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1">Entropy (Chaos)</p>
                    <p className="text-2xl font-black text-slate-900">{result.biometrics.sampen.toFixed(2)}</p>
                    <p className="text-[10px] font-semibold text-slate-500 mt-1 uppercase">Unpredictability</p>
                  </div>
                  <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1">LF/HF Ratio</p>
                    <p className="text-2xl font-black text-slate-900">{result.biometrics.lf_hf_ratio.toFixed(2)}</p>
                    <p className="text-[10px] font-semibold text-slate-500 mt-1 uppercase">Sympathetic tone</p>
                  </div>
                </div>

              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/50 text-center">
                <Activity className="w-16 h-16 text-slate-300 mb-4" />
                <h3 className="text-xl font-bold text-slate-700">Awaiting Signal Analysis</h3>
                <p className="text-slate-500 mt-2 max-w-sm">
                  Select a clinical window from the left panel. The backend will perform QRS-detection, HRV extraction, and CatBoost prediction.
                </p>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}