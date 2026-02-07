#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# =======================================================
# 0️⃣ Configurazione Ambiente & Cache
# =======================================================
os.environ["MPLCONFIGDIR"] = "/workspace/cache/matplotlib"
Path("/workspace/cache/matplotlib").mkdir(parents=True, exist_ok=True)

# Costanti per uniformare tutti i grafici
BASELINE_TIME = 11.2
BASELINE_NDCG = 0.3895
X_LIMITS = (3, 16)
Y_LIMITS = (0.20, 0.42)

def main():
    parser = argparse.ArgumentParser(description="Plot trade-off NDCG vs Time con stile annotazioni, Baseline e assi fissi")
    parser.add_argument("--run-dir", type=str, default=None, help="Sottocartella in runs/")
    args = parser.parse_args()

    RUNS_DIR = Path("runs")

    # =======================================================
    # 1️⃣ Selezione CSV
    # =======================================================
    if args.run_dir is not None:
        out_dir = RUNS_DIR / args.run_dir
        csv_path = out_dir / "grid_results.csv"
    else:
        candidates = list(RUNS_DIR.glob("eval_two_tier_*/*/grid_results.csv"))
        if not candidates:
            raise FileNotFoundError("❌ Nessun 'grid_results.csv' trovato.")
        csv_path = max(candidates, key=os.path.getmtime)
        out_dir = csv_path.parent

    print(f"📂 Caricamento: {csv_path}")
    df = pd.read_csv(csv_path)

    # Parametri colonne
    time_col = "avg_time_ms_per_query"
    metric_col = "ndcg_cut_10"
    activation_col = "low_activation_rate"

    # =======================================================
    # 2️⃣ Setup Grafico
    # =======================================================
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # --- A. Disegna il punto della Baseline (Full Index) ---
    plt.scatter(
        BASELINE_TIME, BASELINE_NDCG, 
        color='red', marker='X', s=150, 
        label=f'Baseline (Full Index: {BASELINE_NDCG:.4f})', 
        zorder=5
    )
    plt.text(
        BASELINE_TIME, BASELINE_NDCG + 0.005, 
        "Full Index", color='red', 
        ha='center', fontsize=10, fontweight='bold'
    )

    # --- B. Plot delle modalità di gating ---
    modes = df["mode"].unique()
    
    for mode in modes:
        df_mode = df[df["mode"] == mode].copy()
        df_mode = df_mode.sort_values(by=time_col)
        
        line = plt.plot(
            df_mode[time_col], 
            df_mode[metric_col], 
            marker="o", 
            label=f"Gating: {mode}",
            linewidth=2,
            markersize=8,
            alpha=0.9
        )
        
        current_color = line[0].get_color()

        # Aggiunta annotazioni (% attivazione SOTTO il punto)
        for _, row in df_mode.iterrows():
            plt.annotate(
                f"{int(row[activation_col])}%",
                (row[time_col], row[metric_col]),
                textcoords="offset points", 
                xytext=(0, -18), 
                ha="center", 
                va="top",
                fontsize=9,
                color=current_color,
                fontweight='bold'
            )

    # --- C. Configurazione Assi e Finestra ---
    plt.xlim(X_LIMITS)
    plt.ylim(Y_LIMITS)
    
    plt.xlabel("Tempo medio per query (ms/query)", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    plt.title(f"Trade-off NDCG vs Tempo\n(Baseline: Full Index | Annotazioni: % attivazione)", fontsize=14, pad=20)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Strategie", frameon=True, loc="lower right")
    plt.tight_layout()

    # Salvataggio
    filename = out_dir / "tradeoff_plot_final_fixed.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Grafico salvato correttamente: {filename}")
    plt.show()

if __name__ == "__main__":
    main()