import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Evaluation Metrics
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, fbeta_score

# Data Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def main():
    print("Loading dataset...")
    CSV_PATH = "paf_hrv_dataset_clean.csv"
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please run your extraction and cleaning pipeline first!")
        return

    df = pd.read_csv(CSV_PATH)

    print("Safely splitting patient data (Preventing Data Leakage)...")
    feature_cols = [col for col in df.columns if col not in ['record_id', 'window_idx', 'label']]
    X = df[feature_cols]
    y = df['label']
    groups = df['record_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_0, train_1 = np.bincount(y_train)
    test_0, test_1 = np.bincount(y_test)

    print(f"Training Set: Class 0 (Safe): {train_0}, Class 1 (AFib): {train_1}")
    print(f"Testing Set:  Class 0 (Safe): {test_0},  Class 1 (AFib): {test_1}\n")

    imbalance_ratio = train_0 / train_1

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, scale_pos_weight=imbalance_ratio, random_state=42, eval_metric='logloss', n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=150, auto_class_weights='Balanced', random_state=42, verbose=0
        )
    }

    os.makedirs("models", exist_ok=True)
    thresholds = [0.2, 0.3, 0.4, 0.5]

    print("Training AI Models...")
    trained_models = {}
    for name, model in tqdm(models.items(), desc="Fitting Models"):
        model.fit(X_train, y_train)
        filename = f"models/{name.lower().replace(' ', '_')}_afib_model.pkl"
        joblib.dump(model, filename)
        trained_models[name] = model

    print("\n=======================================================================")
    print("                 CLINICAL PERFORMANCE REPORT (THRESHOLD SWEEP)          ")
    print("=======================================================================")

    # ── DARK DESIGN TOKENS ─────────────────────────────────────────────────────
    BG         = "#0A0F1E"
    PANEL_BG   = "#111827"
    PANEL_EDGE = "#1E2D45"
    GRID_LINE  = "#1A2740"
    HEADER_BG  = "#0D1F35"
    TEXT_PRI   = "#E8EDF5"
    TEXT_SEC   = "#7B91B0"
    GOLD       = "#FFB347"   # Sensitivity
    GOLD_BG    = "#2A1F0A"
    VIOLET     = "#C77DFF"   # Specificity
    VIOLET_BG  = "#160A2A"
    COBALT     = "#4DA6FF"   # Accuracy
    COBALT_BG  = "#0A1A2E"
    ZEBRA      = "#141E2E"
    MINT       = "#00C896"   # confusion matrix green
    CRIMSON    = "#FF4D6D"   # confusion matrix red

    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "text.color":         TEXT_PRI,
        "axes.labelcolor":    TEXT_SEC,
        "axes.edgecolor":     PANEL_EDGE,
        "axes.facecolor":     PANEL_BG,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        TEXT_SEC,
        "ytick.color":        TEXT_SEC,
        "figure.facecolor":   BG,
        "legend.facecolor":   PANEL_BG,
        "legend.edgecolor":   PANEL_EDGE,
        "legend.labelcolor":  TEXT_PRI,
    })

    # ── FIGURE & GRID ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24))
    fig.set_facecolor(BG)

    gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 1.3, 0.85],
                          hspace=0.52, wspace=0.30,
                          top=0.86, bottom=0.04, left=0.05, right=0.97)

    # ── HEADER BLOCK ───────────────────────────────────────────────────────────
    fig.text(0.05, 0.980, "CLINICAL PERFORMANCE REPORT",
             ha="left", va="top",
             fontsize=20, fontweight="bold", color=TEXT_PRI,
             transform=fig.transFigure)
    fig.text(0.05, 0.963, "AFib Detection  ·  Threshold Sweep Dashboard  ·  Confusion Analysis @ thresh=0.3",
             ha="left", va="top",
             fontsize=10, color=TEXT_SEC,
             transform=fig.transFigure)

    # Right-side metric legend chips (3 metrics)
    chip_x = 0.97
    for label, color in [("Sensitivity", GOLD), ("Specificity", VIOLET), ("Accuracy", COBALT)]:
        fig.add_artist(mpatches.FancyBboxPatch(
            (chip_x - 0.082, 0.964), 0.079, 0.014,
            boxstyle="round,pad=0.005",
            transform=fig.transFigure,
            facecolor=color + "22", edgecolor=color, linewidth=1, zorder=4
        ))
        fig.text(chip_x - 0.082 + 0.0395, 0.971, label,
                 ha="center", va="center",
                 fontsize=8, color=color, fontweight="bold",
                 transform=fig.transFigure)
        chip_x -= 0.086

    # Accent bar
    fig.add_artist(mpatches.FancyArrow(
        0.05, 0.956, 0.90, 0,
        width=0.003, head_width=0, head_length=0,
        transform=fig.transFigure, color=GOLD, zorder=5
    ))

    # Separator
    fig.add_artist(plt.Line2D([0.05, 0.97], [0.954, 0.954],
                              transform=fig.transFigure,
                              color=PANEL_EDGE, linewidth=0.8))

    # ── PER-MODEL ROWS ─────────────────────────────────────────────────────────
    for idx, (name, model) in enumerate(trained_models.items()):
        print(f"\n########### MODEL: {name.upper()} ###########")

        y_prob = model.predict_proba(X_test)[:, 1]

        rows = []
        plot_data = {"Threshold": [], "Sensitivity": [], "Specificity": [], "Accuracy": []}
        sweet_spot_cm = None

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            sensitivity = recall_score(y_test, y_pred, zero_division=0)   # same as recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy    = accuracy_score(y_test, y_pred)

            rows.append({
                "Threshold":         f"{thresh:.1f}",
                "Sensitivity":       f"{sensitivity:.2%}",
                "Specificity":       f"{specificity:.2%}",
                "Accuracy":          f"{accuracy:.2%}",
                "Caught (TP)":       tp,
                "Missed (FN)":       fn,
                "False Alarms (FP)": fp,
            })
            plot_data["Threshold"].append(thresh)
            plot_data["Sensitivity"].append(sensitivity)
            plot_data["Specificity"].append(specificity)
            plot_data["Accuracy"].append(accuracy)

            if round(thresh, 1) == 0.3:
                sweet_spot_cm = (tn, fp, fn, tp)

        report_df = pd.DataFrame(rows)
        print(report_df.to_string(index=False))
        print("-" * 70)

        # ── LEFT: Line chart ──────────────────────────────────────────────────
        ax_line = fig.add_subplot(gs[idx, 0])
        ax_line.set_facecolor(PANEL_BG)
        for spine in ax_line.spines.values():
            spine.set_edgecolor(PANEL_EDGE)

        ax_line.axhspan(0.85, 1.08, color=VIOLET, alpha=0.04, zorder=0)

        ax_line.plot(plot_data["Threshold"], plot_data["Sensitivity"],
                     marker="o", linewidth=2.5, markersize=8,
                     label="Sensitivity", color=GOLD,
                     markerfacecolor=BG, markeredgewidth=2.5, zorder=4)
        ax_line.plot(plot_data["Threshold"], plot_data["Specificity"],
                     marker="s", linewidth=2.5, markersize=8,
                     label="Specificity", color=VIOLET,
                     markerfacecolor=BG, markeredgewidth=2.5, zorder=4)
        ax_line.plot(plot_data["Threshold"], plot_data["Accuracy"],
                     marker="^", linewidth=2, markersize=8,
                     linestyle="--", label="Accuracy", color=COBALT,
                     markerfacecolor=BG, markeredgewidth=2.5, zorder=4)

        ax_line.fill_between(plot_data["Threshold"], plot_data["Sensitivity"],
                             alpha=0.10, color=GOLD, zorder=1)
        ax_line.fill_between(plot_data["Threshold"], plot_data["Specificity"],
                             alpha=0.08, color=VIOLET, zorder=1)

        ax_line.text(0.0, 1.10, f"▌ {name.upper()}",
                     transform=ax_line.transAxes,
                     fontsize=9, fontweight="bold", color=GOLD)
        ax_line.set_title("Metrics vs. Classification Threshold",
                          fontsize=10, fontweight="bold", color=TEXT_PRI, pad=6)
        ax_line.set_xlabel("Decision Threshold", fontsize=9, color=TEXT_SEC)
        ax_line.set_ylabel("Score", fontsize=9, color=TEXT_SEC)
        ax_line.set_xticks(thresholds)
        ax_line.set_ylim(0, 1.12)
        ax_line.grid(True, linestyle=":", color=GRID_LINE, alpha=1.0, zorder=0)

        leg = ax_line.legend(loc="lower left", frameon=True,
                             facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                             fontsize=8, handlelength=2.0,
                             labelcolor=TEXT_PRI)

        # ── MIDDLE: Performance table ─────────────────────────────────────────
        ax_table = fig.add_subplot(gs[idx, 1])
        ax_table.set_facecolor(BG)
        ax_table.axis("off")

        ax_table.text(0.0, 1.10, f"▌ {name.upper()}",
                      transform=ax_table.transAxes,
                      fontsize=9, fontweight="bold", color=COBALT)
        ax_table.set_title("Detailed Performance Sweep",
                            fontsize=10, fontweight="bold", color=TEXT_PRI, pad=6)

        table_headers = ["Thresh", "Sensitivity", "Specificity", "Accuracy",
                         "Caught\n(TP)", "Missed\n(FN)", "False\nAlarms"]
        table_rows = [
            [r["Threshold"], r["Sensitivity"], r["Specificity"], r["Accuracy"],
             str(r["Caught (TP)"]), str(r["Missed (FN)"]), str(r["False Alarms (FP)"])]
            for r in rows
        ]

        tbl = ax_table.table(cellText=table_rows, colLabels=table_headers,
                             loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 2.5)

        n_cols = len(table_headers)

        for col_idx in range(n_cols):
            cell = tbl[0, col_idx]
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(weight="bold", color=TEXT_PRI)
            cell.set_edgecolor(PANEL_EDGE)

        for row_idx in range(1, len(table_rows) + 1):
            is_zebra = (row_idx % 2 == 0)
            for col_idx in range(n_cols):
                cell = tbl[row_idx, col_idx]
                cell.set_edgecolor(PANEL_EDGE)
                if col_idx == 1:        # Sensitivity — gold
                    cell.set_facecolor(GOLD_BG)
                    cell.set_text_props(weight="bold", color=GOLD)
                elif col_idx == 2:      # Specificity — violet
                    cell.set_facecolor(VIOLET_BG)
                    cell.set_text_props(weight="bold", color=VIOLET)
                elif col_idx == 3:      # Accuracy — cobalt
                    cell.set_facecolor(COBALT_BG)
                    cell.set_text_props(weight="bold", color=COBALT)
                else:
                    cell.set_facecolor(ZEBRA if is_zebra else PANEL_BG)
                    cell.set_text_props(color=TEXT_PRI)

        # ── RIGHT: Confusion Matrix ───────────────────────────────────────────
        ax_cm = fig.add_subplot(gs[idx, 2])
        ax_cm.set_facecolor(PANEL_BG)
        ax_cm.set_aspect("equal")

        tn, fp, fn, tp_val = sweet_spot_cm

        GREEN_ACCENT = MINT
        GREEN_BG_CM  = "#0A1F15"
        RED_ACCENT   = CRIMSON
        RED_BG_CM    = "#1F0A0F"

        cells = [
            (0, 0, tn,     "TN",  GREEN_ACCENT, GREEN_BG_CM, "True Negative"),
            (0, 1, fp,     "FP",  RED_ACCENT,   RED_BG_CM,   "False Alarm"),
            (1, 0, fn,     "FN",  RED_ACCENT,   RED_BG_CM,   "Missed AFib"),
            (1, 1, tp_val, "TP",  GREEN_ACCENT, GREEN_BG_CM, "True Positive"),
        ]

        for (r, c, val, abbr, accent, bg, sublabel) in cells:
            ax_cm.add_patch(plt.Rectangle((c, 1 - r), 1, 1,
                                          facecolor=bg, edgecolor=accent,
                                          linewidth=2.5, zorder=1))
            ax_cm.add_patch(plt.Rectangle((c + 0.04, 1 - r + 0.04), 0.92, 0.92,
                                          facecolor="none", edgecolor=accent,
                                          linewidth=0.6, alpha=0.4, zorder=2))
            ax_cm.text(c + 0.10, 1 - r + 0.82, abbr,
                       fontsize=11, fontweight="bold", color=accent,
                       va="center", ha="left", zorder=3)
            ax_cm.text(c + 0.50, 1 - r + 0.50, str(val),
                       fontsize=28, fontweight="bold", color=TEXT_PRI,
                       va="center", ha="center", zorder=3)
            ax_cm.text(c + 0.50, 1 - r + 0.18, sublabel,
                       fontsize=7.5, color=TEXT_SEC,
                       va="center", ha="center", zorder=3)

        ax_cm.set_xlim(0, 2)
        ax_cm.set_ylim(0, 2)
        ax_cm.set_xticks([0.5, 1.5])
        ax_cm.set_xticklabels(["Predicted Safe", "Predicted AFib"],
                               fontsize=8.5, color=TEXT_SEC)
        ax_cm.set_yticks([0.5, 1.5])
        ax_cm.set_yticklabels(["Actual AFib", "Actual Safe"],
                               fontsize=8.5, color=TEXT_SEC, rotation=90, va="center")
        ax_cm.tick_params(length=0)
        for spine in ax_cm.spines.values():
            spine.set_visible(False)

        ax_cm.text(0.0, 1.10, f"▌ {name.upper()}",
                   transform=ax_cm.transAxes,
                   fontsize=9, fontweight="bold", color=MINT)
        ax_cm.set_title("Confusion Matrix  @  thresh = 0.3",
                        fontsize=10, fontweight="bold", color=TEXT_PRI, pad=6)

    output_png = "models/clinical_performance_dashboard.png"
    plt.savefig(output_png, bbox_inches="tight", dpi=180, facecolor=BG)
    print(f"\n[SUCCESS] Visual report exported seamlessly! Saved to: {output_png}\n")

if __name__ == "__main__":
    main()