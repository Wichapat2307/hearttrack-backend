import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, fbeta_score
from imblearn.over_sampling import SMOTE

# Set random seeds for reproducibility across runs
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. PYTORCH LSTM NEURAL NETWORK
# ==========================================
class AFibLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(AFibLSTM, self).__init__()
        # LSTM Layer remembers the sequence of heartbeats
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        # Fully connected layers to make the final prediction
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1) # Output a single probability (0 to 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        
        # We only care about the prediction at the very last time step of the sequence
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# ==========================================
# 2. SEQUENCE GENERATOR FUNCTION
# ==========================================
def create_sequences(df, feature_cols, seq_length=15):
    """Chunks patient timelines into 3D sequences: (samples, time_steps, features)"""
    X_seq, y_seq = [], []
    
    for record_id, group in df.groupby('record_id'):
        group = group.sort_values('window_idx')
        features = group[feature_cols].values
        labels = group['label'].values
        
        # Slide a window of 'seq_length' across the patient's timeline
        for i in range(len(features) - seq_length):
            X_seq.append(features[i : i + seq_length])
            y_seq.append(labels[i + seq_length - 1]) # Label is the end of the sequence
            
    return np.array(X_seq), np.array(y_seq)

# ==========================================
# 3. TRAINING AND EVALUATION ENGINE
# ==========================================
def run_lstm_experiment(X_train_raw, y_train_raw, X_test, y_test, use_smote, base_features, device, epochs=15):
    print(f"\n--- Running Experiment: LSTM {'WITH' if use_smote else 'WITHOUT'} SMOTE ---")
    
    X_train_seq = X_train_raw.copy()
    y_train_seq = y_train_raw.copy()
    
    if use_smote:
        # --- SMOTE ON TRAINING SET ONLY ---
        print(f"Applying SMOTE to training sequences...")
        print(f"  Pre-SMOTE Training AFib cases: {np.sum(y_train_seq == 1)} / {len(y_train_seq)}")
        
        # Flatten 3D to 2D for SMOTE to process
        N, seq_len, num_features = X_train_seq.shape
        X_train_flat = X_train_seq.reshape(N, seq_len * num_features)
        
        smote = SMOTE(random_state=42)
        X_train_resampled_flat, y_train_resampled = smote.fit_resample(X_train_flat, y_train_seq)
        
        # Reshape back to 3D PyTorch format
        X_train_seq = X_train_resampled_flat.reshape(-1, seq_len, num_features)
        y_train_seq = y_train_resampled
        print(f"  Post-SMOTE Training AFib cases: {np.sum(y_train_seq == 1)} / {len(y_train_seq)}")
    else:
        print(f"Training on raw imbalanced set. AFib cases: {np.sum(y_train_seq == 1)} / {len(y_train_seq)}")
        
    # Convert to PyTorch Tensors
    train_data = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    model = AFibLSTM(input_size=len(base_features), hidden_size=64, num_layers=2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            
            # Handle dimension edge case for final batch
            if predictions.ndim == 0:
                predictions = predictions.unsqueeze(0)
                
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze().cpu().numpy()
            
            if preds.ndim == 0: 
                preds = np.expand_dims(preds, 0)
                
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            
    y_prob = np.array(all_preds)
    y_test_array = np.array(all_targets)
    
    # Run threshold sweep
    thresholds = [0.2, 0.3, 0.4, 0.5]
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test_array, y_pred)
        tn, fp, fn, tp = cm.ravel()

        recall = recall_score(y_test_array, y_pred, zero_division=0)
        precision = precision_score(y_test_array, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f2 = fbeta_score(y_test_array, y_pred, beta=2, zero_division=0)
        accuracy = accuracy_score(y_test_array, y_pred)

        results.append({
            "Threshold": thresh,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Recall": recall,
            "Specificity": specificity,
            "Precision": precision,
            "Accuracy": accuracy,
            "F2-Score": f2
        })
        
    return results, y_prob, y_test_array

# ==========================================
# 4. MAIN EXPERIMENTAL PIPELINE
# ==========================================
def main():
    print("Loading dataset...")
    CSV_PATH = "paf_hrv_dataset_clean.csv"
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "paf_hrv_dataset.csv" 
        if not os.path.exists(CSV_PATH):
            print("Dataset not found! Please run your pipeline first.")
            return
            
    df = pd.read_csv(CSV_PATH)
    
    # Drop delta and lag features; let LSTM handle spatial-temporal patterns naturally
    base_features = [col for col in df.columns if col not in ['record_id', 'window_idx', 'label'] and 'delta' not in col and 'lag' not in col]
    print(f"Features used for LSTM: {len(base_features)} pure biological signals.")
    
    # Patient-Safe Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df['label'], df['record_id']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Scale Features
    scaler = StandardScaler()
    train_df[base_features] = scaler.fit_transform(train_df[base_features])
    test_df[base_features] = scaler.transform(test_df[base_features])
    
    # Structure into sequence steps
    SEQ_LENGTH = 10  
    print(f"Structuring timelines into {SEQ_LENGTH}-step historical sequences...")
    X_train_seq, y_train_seq = create_sequences(train_df, base_features, seq_length=SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(test_df, base_features, seq_length=SEQ_LENGTH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- RUN 1: WITHOUT SMOTE ---
    results_no_smote, y_prob_no_smote, y_test_no_smote = run_lstm_experiment(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
        use_smote=False, base_features=base_features, device=device
    )
    
    # --- RUN 2: WITH SMOTE ---
    results_smote, y_prob_smote, y_test_smote = run_lstm_experiment(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
        use_smote=True, base_features=base_features, device=device
    )
    
    # ==========================================
    # 5. PRINT COMPARISON REPORTS
    # ==========================================
    print("\n" + "="*80)
    print("               LSTM PERFORMANCE COMPARISON (SMOTE VS RAW)")
    print("="*80)
    
    print("\n>>> CONFIGURATION 1: LSTM WITHOUT SMOTE (RAW IMBALANCE)")
    df_no_smote = pd.DataFrame(results_no_smote)
    print_cols = ["Threshold", "TP", "FN", "FP", "Recall", "Specificity", "Precision", "F2-Score"]
    df_no_smote_print = df_no_smote.copy()
    for col in ["Recall", "Specificity", "Precision"]:
        df_no_smote_print[col] = df_no_smote_print[col].map(lambda x: f"{x:.2%}")
    df_no_smote_print["F2-Score"] = df_no_smote_print["F2-Score"].map(lambda x: f"{x:.4f}")
    print(df_no_smote_print[print_cols].to_string(index=False))
    
    print("\n>>> CONFIGURATION 2: LSTM WITH SMOTE (BALANCED TRAINING)")
    df_smote = pd.DataFrame(results_smote)
    df_smote_print = df_smote.copy()
    for col in ["Recall", "Specificity", "Precision"]:
        df_smote_print[col] = df_smote_print[col].map(lambda x: f"{x:.2%}")
    df_smote_print["F2-Score"] = df_smote_print["F2-Score"].map(lambda x: f"{x:.4f}")
    print(df_smote_print[print_cols].to_string(index=False))
    print("="*80)

    # ==========================================
    # 6. EXPORT CLINICAL VISUALIZATION DASHBOARD
    # ==========================================
    BG         = "#F7F9FC"
    PANEL_EDGE = "#DDE3EC"
    NAVY       = "#0D1B2A"
    SLATE      = "#2C3E50"
    CRIMSON    = "#C0392B"
    MINT       = "#27AE60"
    COBALT     = "#2980B9"
    WHITE      = "#FFFFFF"

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.facecolor":    WHITE,
        "figure.facecolor":  BG,
    })

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    fig.text(0.5, 0.96, "Deep Learning AFib Prediction Report", ha="center", va="top",
             fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.93, "Chronological LSTM Architecture: Raw Training vs. SMOTE Sequence Resampling",
             ha="center", va="top", fontsize=11, color=SLATE)

    fig.add_artist(plt.Line2D([0.05, 0.95], [0.915, 0.915], transform=fig.transFigure, color=PANEL_EDGE, linewidth=1.2))

    # --- TOP LEFT: METRIC SWEEP WITHOUT SMOTE ---
    ax_no_smote = fig.add_subplot(gs[0, 0])
    ax_no_smote.plot(df_no_smote["Threshold"], df_no_smote["Recall"], marker='o', linewidth=2, label="Recall (Sensitivity)", color=CRIMSON)
    ax_no_smote.plot(df_no_smote["Threshold"], df_no_smote["Specificity"], marker='s', linewidth=2, label="Specificity", color=MINT)
    ax_no_smote.plot(df_no_smote["Threshold"], df_no_smote["F2-Score"], marker='^', linewidth=2, linestyle='--', label="F2-Score", color=COBALT)
    ax_no_smote.set_title("LSTM without SMOTE: Threshold Trade-off", fontsize=11, fontweight="bold", color=NAVY)
    ax_no_smote.set_xlabel("Decision Threshold", fontsize=9)
    ax_no_smote.set_ylabel("Score", fontsize=9)
    ax_no_smote.set_xticks([0.2, 0.3, 0.4, 0.5])
    ax_no_smote.set_ylim(0, 1.05)
    ax_no_smote.grid(True, linestyle=":", color=PANEL_EDGE)
    ax_no_smote.legend(loc="lower left", fontsize=8)

    # --- TOP RIGHT: METRIC SWEEP WITH SMOTE ---
    ax_smote = fig.add_subplot(gs[0, 1])
    ax_smote.plot(df_smote["Threshold"], df_smote["Recall"], marker='o', linewidth=2, label="Recall (Sensitivity)", color=CRIMSON)
    ax_smote.plot(df_smote["Threshold"], df_smote["Specificity"], marker='s', linewidth=2, label="Specificity", color=MINT)
    ax_smote.plot(df_smote["Threshold"], df_smote["F2-Score"], marker='^', linewidth=2, linestyle='--', label="F2-Score", color=COBALT)
    ax_smote.set_title("LSTM with SMOTE: Threshold Trade-off", fontsize=11, fontweight="bold", color=NAVY)
    ax_smote.set_xlabel("Decision Threshold", fontsize=9)
    ax_smote.set_ylabel("Score", fontsize=9)
    ax_smote.set_xticks([0.2, 0.3, 0.4, 0.5])
    ax_smote.set_ylim(0, 1.05)
    ax_smote.grid(True, linestyle=":", color=PANEL_EDGE)
    ax_smote.legend(loc="lower left", fontsize=8)

    # --- BOTTOM LEFT: CONFUSION MATRIX NO SMOTE (Thresh = 0.3) ---
    ax_cm_no_smote = fig.add_subplot(gs[1, 0])
    row_no_smote_03 = df_no_smote[df_no_smote["Threshold"] == 0.3].iloc[0]
    cm_no_smote_matrix = np.array([
        [int(row_no_smote_03["TN"]), int(row_no_smote_03["FP"])],
        [int(row_no_smote_03["FN"]), int(row_no_smote_03["TP"])]
    ])
    sns.heatmap(cm_no_smote_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Safe (0)", "Pred AFib (1)"],
                yticklabels=["Actual Safe (0)", "Actual AFib (1)"],
                ax=ax_cm_no_smote, annot_kws={"size": 11, "weight": "bold"},
                linewidths=1.5, linecolor=PANEL_EDGE)
    ax_cm_no_smote.set_title("No SMOTE Confusion @ Thresh=0.3", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    plt.setp(ax_cm_no_smote.get_xticklabels(), rotation=10, ha="right")

    # --- BOTTOM RIGHT: CONFUSION MATRIX WITH SMOTE (Thresh = 0.3) ---
    ax_cm_smote = fig.add_subplot(gs[1, 1])
    row_smote_03 = df_smote[df_smote["Threshold"] == 0.3].iloc[0]
    cm_smote_matrix = np.array([
        [int(row_smote_03["TN"]), int(row_smote_03["FP"])],
        [int(row_smote_03["FN"]), int(row_smote_03["TP"])]
    ])
    sns.heatmap(cm_smote_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Safe (0)", "Pred AFib (1)"],
                yticklabels=["Actual Safe (0)", "Actual AFib (1)"],
                ax=ax_cm_smote, annot_kws={"size": 11, "weight": "bold"},
                linewidths=1.5, linecolor=PANEL_EDGE)
    ax_cm_smote.set_title("With SMOTE Confusion @ Thresh=0.3", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    plt.setp(ax_cm_smote.get_xticklabels(), rotation=10, ha="right")

    os.makedirs("models", exist_ok=True)
    output_png = "models/lstm_comparison_dashboard.png"
    plt.savefig(output_png, bbox_inches="tight", dpi=150, facecolor=BG)
    print(f"\n[SUCCESS] Comparative visual report exported! Saved to: {output_png}\n")

if __name__ == "__main__":
    main()