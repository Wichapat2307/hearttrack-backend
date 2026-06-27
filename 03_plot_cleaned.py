import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

CLEAN_DATA_DIR = Path("PAFPDB/Cleaned")

# Choose which file to inspect (e.g., "n01", "p02")
RECORD_ID = "n02" 
WINDOW_SECONDS = 5


def pan_tompkins_peaks(signal, fs):
    """
    Advanced Noise-Tolerant Pan-Tompkins R-peak detector.
    Uses global segment polarity detection and adaptive thresholding 
    to prevent noise spikes from causing false double-detections.
    """
    # --- 1. GLOBAL POLARITY DETECTION ---
    # Check if the signal trends further downward or upward relative to its baseline
    median_val = np.median(signal)
    is_inverted = np.abs(np.min(signal) - median_val) > np.abs(np.max(signal) - median_val)

    # --- 2. PRE-PROCESSING (SLOPE & ENERGY) ---
    # Derivative filter to highlight steep QRS complexes
    diff = np.diff(signal)
    
    # Square the signal to make all slopes positive and accentuate high peaks
    squared = diff ** 2
    
    # Moving window integration (widened slightly to 150ms to absorb high-frequency noise)
    window_len = int(0.15 * fs) 
    integrated = np.convolve(squared, np.ones(window_len)/window_len, mode='same')
    
    # --- 3. ADAPTIVE PEAK FILTERING ---
    # Prevent double-counting within a 350ms window (~170 BPM limit)
    min_distance = int(0.35 * fs) 
    
    # Combine a higher prominence threshold with a dynamic baseline floor
    prom_thresh = np.max(integrated) * 0.20
    height_thresh = np.mean(integrated) + 0.2 * np.std(integrated)
    
    peaks, _ = find_peaks(integrated, distance=min_distance, prominence=prom_thresh, height=height_thresh)
    
    # --- 4. INVERSION-PROOF ALIGNMENT ---
    refined_peaks = []
    search_radius = int(0.06 * fs) # 60ms search window around the energy peak
    
    for p in peaks:
        start_idx = max(0, p - search_radius)
        end_idx = min(len(signal), p + search_radius)
        if start_idx < end_idx:
            window = signal[start_idx:end_idx]
            
            # Lock configuration based on the channel's global polarity layout
            if is_inverted:
                true_r = start_idx + np.argmin(window) # Always find the true bottom of the valley
            else:
                true_r = start_idx + np.argmax(window) # Always find the true top of the peak
                
            refined_peaks.append(true_r)
            
    return np.array(refined_peaks)


class CleanViewer:
    def __init__(self, record_id):
        self.record_id = record_id
        file_path = CLEAN_DATA_DIR / f"{record_id}_clean.npz"
        
        if not file_path.exists():
            print(f"Error: Clean file for {record_id} not found. Run clean_dataset.py first!")
            return

        # Load filtered numpy arrays
        data = np.load(file_path)
        self.signal = data['signal']
        self.fs = int(data['fs'])
        self.sig_names = data['sig_name']
        
        self.total_samples = len(self.signal)
        self.num_leads = self.signal.shape[1]
        
        self.window_seconds = WINDOW_SECONDS
        self.window_samples = int(self.window_seconds * self.fs)
        self.start = 0

        # Subplot setup
        self.fig, self.axs = plt.subplots(self.num_leads, 1, sharex=True, figsize=(15, 3 * self.num_leads))
        if self.num_leads == 1: self.axs = [self.axs]

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update()
        plt.show()

    def update(self):
        end = min(self.start + self.window_samples, self.total_samples)
        t = np.arange(self.start, end) / self.fs

        for i in range(self.num_leads):
            ax = self.axs[i]
            ax.clear()
            segment = self.signal[self.start:end, i]
            
            # 1. Plot the continuous clean ECG wave
            ax.plot(t, segment, linewidth=1, color="teal", label="Cleaned ECG")
            
            # --- 2. FIND AND PLOT THE RED DOT R-PEAKS ---
            local_peaks = pan_tompkins_peaks(segment, self.fs)
            if len(local_peaks) > 0:
                # Align peak locations with absolute time axis
                peak_times = (self.start + local_peaks) / self.fs
                peak_amplitudes = segment[local_peaks]
                
                # Draw red circles right on top of the target points
                ax.scatter(peak_times, peak_amplitudes, color="crimson", edgecolors="black", s=40, zorder=3, label="Detected Peak")

            ax.set_ylabel(f"{self.sig_names[i]} (Clean)")
            ax.grid(True, linestyle="--", alpha=0.5)
            if i == 0: ax.legend(loc="upper right")

        self.axs[0].set_title(f"CLEANED Dataset Viewer with Peaks | Record: {self.record_id} | Window: {self.window_seconds}s")
        self.axs[-1].set_xlabel("Time (seconds)")
        self.fig.suptitle("Controls: [Left/Right] Scroll | [Up/Down] Zoom In/Out | [Q] Close", y=0.98, fontsize=9, color='gray')
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = int(self.window_samples * 0.5)
        if event.key == "right":
            self.start = min(self.start + step, self.total_samples - self.window_samples)
        elif event.key == "left":
            self.start = max(self.start - step, 0)
        elif event.key == "up":
            self.window_seconds = max(1, self.window_seconds - 1)
            self.window_samples = int(self.window_seconds * self.fs)
        elif event.key == "down":
            max_seconds = int(self.total_samples / self.fs)
            self.window_seconds = min(max_seconds, self.window_seconds + 1)
            self.window_samples = int(self.window_seconds * self.fs)
            if self.start + self.window_samples > self.total_samples:
                self.start = max(0, self.total_samples - self.window_samples)
        elif event.key == "q":
            plt.close(self.fig)
            return
        self.update()


if __name__ == "__main__":
    viewer = CleanViewer(RECORD_ID)