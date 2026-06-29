"""
extract_demo_csvs.py  —  HeartTrack PAFPDB Demo Extractor
==========================================================
Extracts RR interval CSVs + raw ECG snippets from PAFPDB.

For each demo record it saves:
  test_*.csv          — RR intervals (ms) for AI analysis
  test_*_ecg.json     — 10-second raw ECG snippet for waveform plot

Usage:
  cd HeartTrack
  python extract_demo_csvs.py
"""

import os
import json
import numpy as np
import wfdb

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH    = "./PAFPDB"
OUTPUT_DIR = "./hearttrack-ui/public/demo_files"
ECG_SNIPPET_SEC = 10   # seconds of raw ECG to export per file

EXTRACTIONS = [
    # Healthy controls
    dict(record="t03", start_min=0,  end_min=3,  out_name="test_control_1",
         label_key="normal",
         short="Healthy Control #1  (t03)",
         description="Healthy subject — no AF history. First 3 minutes of recording."),
    dict(record="t05", start_min=0,  end_min=3,  out_name="test_control_2",
         label_key="normal",
         short="Healthy Control #2  (t05)",
         description="Healthy subject — no AF history. First 3 minutes of recording."),
    dict(record="t07", start_min=0,  end_min=3,  out_name="test_control_3",
         label_key="normal",
         short="Healthy Control #3  (t07)",
         description="Healthy subject — no AF history. First 3 minutes of recording."),

    # Stable PAF
    dict(record="t01", start_min=0,  end_min=3,  out_name="test_paf_stable_1",
         label_key="paf_distant", pair="t02",
         short="Stable PAF #1  (t01, min 0–3)",
         description="PAF patient — AF episode ~30 min away. Rhythm still relatively stable."),
    dict(record="t09", start_min=0,  end_min=3,  out_name="test_paf_stable_2",
         label_key="paf_distant", pair="t10",
         short="Stable PAF #2  (t09, min 0–3)",
         description="PAF patient — AF episode ~30 min away. Rhythm still relatively stable."),
    dict(record="t11", start_min=0,  end_min=3,  out_name="test_paf_stable_3",
         label_key="paf_distant", pair="t12",
         short="Stable PAF #3  (t11, min 0–3)",
         description="PAF patient — AF episode ~30 min away. Rhythm still relatively stable."),

    # Imminent AFib
    dict(record="t02", start_min=27, end_min=30, out_name="test_paf_imminent_1",
         label_key="paf_imminent", pair="t01",
         short="Imminent AFib #1  (t02, min 27–30)",
         description="PAF patient — AF episode begins immediately after. Pair of Stable PAF #1."),
    dict(record="t10", start_min=27, end_min=30, out_name="test_paf_imminent_2",
         label_key="paf_imminent", pair="t09",
         short="Imminent AFib #2  (t10, min 27–30)",
         description="PAF patient — AF episode begins immediately after. Pair of Stable PAF #2."),
    dict(record="t12", start_min=27, end_min=30, out_name="test_paf_imminent_3",
         label_key="paf_imminent", pair="t11",
         short="Imminent AFib #3  (t12, min 27–30)",
         description="PAF patient — AF episode begins immediately after. Pair of Stable PAF #3."),
]

BADGE = {
    "normal":       "NORMAL",
    "paf_distant":  "PAF STABLE",
    "paf_imminent": "PAF IMMINENT",
}


def extract_rr_window(db_path, record_name, start_min, end_min):
    record_path = os.path.join(db_path, record_name)
    for ext in ["qrs", "atr"]:
        if not os.path.exists(f"{record_path}.{ext}"):
            continue
        try:
            ann = wfdb.rdann(record_path, ext)
            hdr = wfdb.rdheader(record_path)
            fs  = hdr.fs
            times_sec = np.array(ann.sample) / fs
            mask      = (times_sec >= start_min * 60) & (times_sec <= end_min * 60)
            window_t  = times_sec[mask]
            if len(window_t) < 2:
                return None
            rr = np.diff(window_t) * 1000.0
            rr = rr[(rr >= 200) & (rr <= 2000)]
            print(f"  ✓ RR: {record_name}.{ext}  fs={fs}Hz  → {len(rr)} intervals")
            return rr
        except Exception as e:
            print(f"  ✗ RR {record_name}.{ext}: {e}")
    return None


def extract_ecg_snippet(db_path, record_name, start_min, snippet_sec=10):
    """Extract a short raw ECG snippet starting at start_min."""
    record_path = os.path.join(db_path, record_name)
    if not os.path.exists(f"{record_path}.dat"):
        print(f"  ✗ ECG: {record_name}.dat not found")
        return None
    try:
        hdr      = wfdb.rdheader(record_path)
        fs       = hdr.fs
        n_samps  = hdr.sig_len
        start_s  = int(start_min * 60 * fs)
        end_s    = min(start_s + int(snippet_sec * fs), n_samps)

        record   = wfdb.rdrecord(record_path, sampfrom=start_s, sampto=end_s)
        # Use channel 0; convert to mV using ADC gain
        signal   = record.p_signal[:, 0].tolist()

        # Downsample to 128 Hz for smaller JSON if fs is higher
        if fs > 128:
            factor = int(fs / 128)
            signal = signal[::factor]
            fs_out = fs / factor
        else:
            fs_out = fs

        print(f"  ✓ ECG: {record_name}.dat  {snippet_sec}s snippet  fs={fs_out}Hz  {len(signal)} samples")
        return {"signal": signal, "fs": fs_out, "duration_sec": snippet_sec}
    except Exception as e:
        print(f"  ✗ ECG {record_name}.dat: {e}")
        return None


def save_rr_csv(rr, cfg):
    fname = cfg["out_name"] + ".csv"
    fpath = os.path.join(OUTPUT_DIR, fname)
    pair_note = f"# Paired with: {cfg.get('pair','')}\n" if cfg.get("pair") else ""
    with open(fpath, "w") as f:
        f.write(f"# HeartTrack Demo | Source: PAFPDB {cfg['record']} (min {cfg['start_min']}–{cfg['end_min']})\n")
        f.write(f"# Label: {cfg['short']}\n")
        f.write(f"# Description: {cfg['description']}\n")
        f.write(f"# Category: {cfg['label_key']} | Badge: {BADGE[cfg['label_key']]}\n")
        f.write(pair_note)
        f.write(f"# Intervals: {len(rr)}\n")
        f.write("rr_ms\n")
        for v in rr:
            f.write(f"{v:.2f}\n")
    print(f"  → CSV: {fpath}")
    return fname


def save_ecg_json(ecg_data, cfg):
    fname = cfg["out_name"] + "_ecg.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w") as f:
        json.dump(ecg_data, f)
    print(f"  → ECG: {fpath}  ({len(ecg_data['signal'])} samples)")
    return fname


if __name__ == "__main__":
    print(f"\n📁 DB path    : {DB_PATH}")
    print(f"📤 Output dir : {OUTPUT_DIR}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = []
    ok = 0

    for cfg in EXTRACTIONS:
        print(f"\n[{cfg['label_key']}] {cfg['out_name']}  ←  {cfg['record']} min {cfg['start_min']}–{cfg['end_min']}")

        hea = os.path.join(DB_PATH, f"{cfg['record']}.hea")
        if not os.path.exists(hea):
            print(f"  [skip] {cfg['record']}.hea not found")
            continue

        rr = extract_rr_window(DB_PATH, cfg["record"], cfg["start_min"], cfg["end_min"])
        if rr is None or len(rr) < 10:
            print(f"  ✗ Not enough RR intervals — skipped")
            continue

        # ECG snippet starts at same window start
        ecg = extract_ecg_snippet(DB_PATH, cfg["record"], cfg["start_min"], ECG_SNIPPET_SEC)

        csv_fname = save_rr_csv(rr, cfg)
        ecg_fname = save_ecg_json(ecg, cfg) if ecg else None

        ok += 1
        manifest.append({
            "file":        f"/demo_files/{csv_fname}",
            "ecg_file":    f"/demo_files/{ecg_fname}" if ecg_fname else None,
            "record":      cfg["record"],
            "window":      f"min {cfg['start_min']}–{cfg['end_min']}",
            "label_key":   cfg["label_key"],
            "badge":       BADGE[cfg["label_key"]],
            "short":       cfg["short"],
            "description": cfg["description"],
            "pair":        cfg.get("pair", None),
        })

    manifest_path = os.path.join(OUTPUT_DIR, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📋 Manifest → {manifest_path}")
    print(f"✅ Done — {ok}/9 exported\n")