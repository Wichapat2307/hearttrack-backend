import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. SET YOUR DATA PATH
# If the files are sitting directly in your open VS Code folder, leave this as "."
# If they are inside a subfolder (e.g., "afpdb"), change it to Path("afpdb")
DATA_DIR = Path("PAFPDB") 

# 2. CHOOSE YOUR RECORD TO VIEW
# Simply swap this out to check any file (e.g., "n01", "n02", "p01", "p25")
RECORD_ID = "n02" 

WINDOW_SECONDS = 10


def load_record(record_id):
    # Construct the full path without appending extensions like .dat or .hea
    rec_path = str(DATA_DIR / record_id)

    try:
        record = wfdb.rdrecord(rec_path)
        print(f"--- Successfully Loaded: {record_id} ---")
        print("Signal Shape (Samples, Leads):", record.p_signal.shape)
        print("Sampling Frequency (Fs):", record.fs, "Hz")
        return record
    except FileNotFoundError:
        print(f"Error: Could not find record '{record_id}' in directory '{DATA_DIR.absolute()}'.")
        print("Please check your file names or update the DATA_DIR path variable.")
        exit()


class AFPDBViewer:
    def __init__(self, record):
        self.record = record
        self.signal = record.p_signal
        self.fs = int(record.fs)
        self.total_samples = len(self.signal)
        
        # Automatically detect if the record has 1 or 2 ECG leads
        self.num_leads = self.signal.shape[1]

        self.window_seconds = WINDOW_SECONDS
        self.window_samples = int(self.window_seconds * self.fs)
        self.start = 0

        # Setup stacked subplots for clean lead comparison
        self.fig, self.axs = plt.subplots(
            nrows=self.num_leads, 
            ncols=1, 
            sharex=True, 
            figsize=(15, 4 * self.num_leads)
        )
        
        # Wrap in a list if it's a single-lead signal to keep array indexing stable
        if self.num_leads == 1:
            self.axs = [self.axs]

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update()
        plt.show()

    def update(self):
        end = min(self.start + self.window_samples, self.total_samples)
        t = np.arange(self.start, end) / self.fs

        # Render each lead onto its respective subplot
        for i in range(self.num_leads):
            ax = self.axs[i]
            ax.clear()
            
            segment = self.signal[self.start:end, i]
            ax.plot(t, segment, linewidth=1, color=f"C{i}")

            # Grab lead labels from metadata if available (e.g., 'ECG1', 'ECG2')
            lead_name = self.record.sig_name[i] if hasattr(self.record, 'sig_name') else f"Lead {i}"
            ax.set_ylabel(f"{lead_name} (mV)")
            ax.grid(True, linestyle="--", alpha=0.5)

        # Dynamic titles based on whether it's an 'n' or 'p' record
        record_type = "Pre-AFib Patient" if self.record.record_name.startswith('p') else "Normal Control"
        
        self.axs[0].set_title(
            f"AFPDB Record: {self.record.record_name} ({record_type}) | "
            f"Time: {self.start/self.fs:.1f}s → {end/self.fs:.1f}s | "
            f"Window: {self.window_seconds}s"
        )
        
        self.axs[-1].set_xlabel("Time (s)")
        self.fig.suptitle("Controls: [Left / Right Arrow] Pan | [Up / Down Arrow] Zoom In/Out | [Q] Quit", y=0.98, fontsize=10, color='gray')
        
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        # Shift view by 50% of the current window length
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
    record = load_record(RECORD_ID)
    viewer = AFPDBViewer(record)