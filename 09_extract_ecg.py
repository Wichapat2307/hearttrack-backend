"""
extract_demo_npy.py  —  HeartTrack PAFPDB NPY Extractor
========================================================
Extracts raw ECG signals as .npy files from PAFPDB.
Each file contains 30 seconds of raw ECG at 128 Hz = 3840 samples.

Output → ./hearttrack-ui/public/demo_files/

Usage:
  cd HeartTrack
  python extract_demo_npy.py
"""

import os
import json
import numpy as np
import wfdb

DB_PATH    = "./PAFPDB"
OUTPUT_DIR = "./hearttrack-ui/public/demo_files"
SNIPPET_SEC = 30   # 30 seconds per demo file

EXTRACTIONS = [
    dict(record="t03", start_min=0,  out_name="demo_control_1",
         label_key="normal", short="Healthy Control #1 (t03)",
         description="Healthy subject — no AF history."),
    dict(record="t05", start_min=0,  out_name="demo_control_2",
         label_key="normal", short="Healthy Control #2 (t05)",
         description="Healthy subject — no AF history."),
    dict(record="t07", start_min=0,  out_name="demo_control_3",
         label_key="normal", short="Healthy Control #3 (t07)",
         description="Healthy subject — no AF history."),

    dict(record="t01", start_min=0,  out_name="demo_paf_stable_1",
         label_key="paf_distant", short="Stable PAF #1 (t01, min 0)",
         description="PAF patient — AF episode ~30 min away. Rhythm still stable.", pair="t02"),
    dict(record="t09", start_min=0,  out_name="demo_paf_stable_2",
         label_key="paf_distant", short="Stable PAF #2 (t09, min 0)",
         description="PAF patient — AF episode ~30 min away. Rhythm still stable.", pair="t10"),
    dict(record="t11", start_min=0,  out_name="demo_paf_stable_3",
         label_key="paf_distant", short="Stable PAF #3 (t11, min 0)",
         description="PAF patient — AF episode ~30 min away. Rhythm still stable.", pair="t12"),

    dict(record="t02", start_min=27, out_name="demo_paf_imminent_1",
         label_key="paf_imminent", short="Imminent AFib #1 (t02, min 27)",
         description="AF episode begins immediately after this window.", pair="t01"),
    dict(record="t10", start_min=27, out_name="demo_paf_imminent_2",
         label_key="paf_imminent", short="Imminent AFib #2 (t10, min 27)",
         description="AF episode begins immediately after this window.", pair="t09"),
    dict(record="t12", start_min=27, out_name="demo_paf_imminent_3",
         label_key="paf_imminent", short="Imminent AFib #3 (t12, min 27)",
         description="AF episode begins immediately after this window.", pair="t11"),
]

BADGE = {
    "normal":       "NORMAL",
    "paf_distant":  "PAF STABLE",
    "paf_imminent": "PAF IMMINENT",
}


def extract_ecg_npy(db_path, record_name, start_min, snippet_sec):
    record_path = os.path.join(db_path, record_name)
    if not os.path.exists(f"{record_path}.dat"):
        print(f"  ✗ {record_name}.dat not found")
        return None, None
    try:
        hdr     = wfdb.rdheader(record_path)
        fs      = hdr.fs
        start_s = int(start_min * 60 * fs)
        end_s   = min(start_s + int(snippet_sec * fs), hdr.sig_len)

        record  = wfdb.rdrecord(record_path, sampfrom=start_s, sampto=end_s)
        signal  = record.p_signal[:, 0]  # channel 0, physical units (mV)

        # Resample to exactly 128 Hz for consistent frontend handling
        target_fs  = 128
        if fs != target_fs:
            from scipy.signal import resample
            n_target = int(len(signal) * target_fs / fs)
            signal   = resample(signal, n_target)

        # Remove baseline wander with high-pass filter
        from scipy.signal import butter, filtfilt
        b, a   = butter(2, 0.5 / (target_fs / 2), btype='high')
        signal = filtfilt(b, a, signal)

        # Normalize to [-1, 1]
        mx = np.max(np.abs(signal))
        if mx > 0:
            signal = signal / mx

        print(f"  ✓ {record_name}  fs={target_fs}Hz  {len(signal)} samples  ({snippet_sec}s)")
        return signal.astype(np.float32), target_fs
    except Exception as e:
        print(f"  ✗ {record_name}: {e}")
        return None, None


if __name__ == "__main__":
    print(f"\n📁 DB path    : {DB_PATH}")
    print(f"📤 Output dir : {OUTPUT_DIR}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = []
    ok = 0

    for cfg in EXTRACTIONS:
        print(f"\n[{cfg['label_key']}] {cfg['out_name']}  ←  {cfg['record']} min {cfg['start_min']}")

        hea = os.path.join(DB_PATH, f"{cfg['record']}.hea")
        if not os.path.exists(hea):
            print(f"  [skip] {cfg['record']}.hea not found")
            continue

        signal, fs = extract_ecg_npy(DB_PATH, cfg["record"], cfg["start_min"], SNIPPET_SEC)
        if signal is None:
            continue

        # Save as .npy
        npy_path = os.path.join(OUTPUT_DIR, cfg["out_name"] + ".npy")
        np.save(npy_path, signal)
        print(f"  → {npy_path}  ({os.path.getsize(npy_path)//1024} KB)")

        ok += 1
        manifest.append({
            "file":        f"/demo_files/{cfg['out_name']}.npy",
            "record":      cfg["record"],
            "fs":          fs,
            "duration_sec": SNIPPET_SEC,
            "label_key":   cfg["label_key"],
            "badge":       BADGE[cfg["label_key"]],
            "short":       cfg["short"],
            "description": cfg["description"],
            "pair":        cfg.get("pair"),
        })

    manifest_path = os.path.join(OUTPUT_DIR, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📋 Manifest → {manifest_path}")
    print(f"✅ Done — {ok}/9 NPY files exported\n")