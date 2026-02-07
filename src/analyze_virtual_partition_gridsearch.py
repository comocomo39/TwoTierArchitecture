#!/usr/bin/env python3
import os
import re
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================
# 0️⃣ Configurazione Ambiente
# =======================================================
os.environ["MPLCONFIGDIR"] = "/workspace/cache/matplotlib"
Path("/workspace/cache/matplotlib").mkdir(parents=True, exist_ok=True)

# Valori fissi della Baseline
BASELINE_TIME = 11.2
BASELINE_NDCG = 0.3895

def parse_log_file(log_path):
    """Parsa il file di log ed estrae i dati delle run."""
    data = []
    share_regex = re.compile(r"Computing Masks for High Share ([\d\.]+)")
    run_regex = re.compile(r"Run: Share=[\d\.]+, Mode=(\w+), Val=[\d\.]+")
    metrics_regex = re.compile(r"nDCG@10:\s*([\d\.]+)\s*\|\s*Time:\s*([\d\.]+)ms\s*\|\s*Rate:\s*([\d\.]+)%")

    current_share = None
    current_mode = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        share_match = share_regex.search(line)
        if share_match:
            current_share = float(share_match.group(1))
            continue
            
        run_match = run_regex.search(line)
        if run_match:
            current_mode = run_match.group(1)
            for j in range(1, 3):
                if i + j < len(lines):
                    m_match = metrics_regex.search(lines[i+j])
                    if m_match:
                        data.append({
                            "high_share": current_share,
                            "mode": current_mode,
                            "ndcg": float(m_match.group(1)),
                            "time": float(m_match.group(2)),
                            "rate": float(m_match.group(3))
                        })
                        break
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Plot trade-off da LOG con Baseline e assi fissi")
    parser.add_argument("--run-dir", type=str, default=None, help="Sottocartella in runs/")
    args = parser.parse_args()

    RUNS_DIR = Path("runs")

    # 1. Trova la cartella del run
    if args.run_dir:
        target_dir = RUNS_DIR / args.run_dir
    else:
        candidates = sorted(list(RUNS_DIR.glob("grid_virtual_*/*")), key=os.path.getmtime)
        if not candidates:
            raise FileNotFoundError("❌ Nessuna cartella di run trovata.")
        target_dir = candidates[-1]

    # 2. Cerca il file .log o .txt
    log_files = list(target_dir.glob("*.log")) or list(target_dir.glob("*.txt"))
    if not log_files:
        raise FileNotFoundError(f"❌ Nessun log trovato in {target_dir}")
    
    log_path = log_files[0]
    print(f"📂 Analisi del log: {log_path}")

    # 3. Parsing
    df = parse_log_file(log_path)
    if df.empty:
        print("⚠️ Nessun dato estratto.")
        return

    # Impostazioni grafiche fisse
    sns.set_style("whitegrid")
    colors = {"margin": "#ff7f0e", "entropy": "#1f77b4"}
    markers = {"margin": "o", "entropy": "s"}
    
    # --- LIMITI ASSI FISSI ---
    X_LIMITS = (4, 16)      # Tempo da 0 a 16ms
    Y_LIMITS = (0.20, 0.42) # NDCG da 0.20 a 0.42

    splits = sorted(df["high_share"].unique())

    for hs in splits:
        hs_pct = int(hs * 100)
        df_hs = df[df["high_share"] == hs].copy()
        
        plt.figure(figsize=(10, 6.5))
        
        # A. Disegna il punto della Baseline (Full Index)
        plt.scatter(
            BASELINE_TIME, BASELINE_NDCG, 
            color='red', marker='X', s=120, 
            label=f'Baseline (Full Index: {BASELINE_NDCG:.4f})', 
            zorder=5
        )
        # Etichetta per la baseline
        plt.text(BASELINE_TIME, BASELINE_NDCG + 0.005, "Full Index", color='red', 
                 ha='center', fontsize=9, fontweight='bold')

        # B. Plot delle modalità Margin ed Entropy
        for mode in ["margin", "entropy"]:
            subset = df_hs[df_hs["mode"] == mode].sort_values(by="time")
            if subset.empty: continue

            plt.plot(
                subset["time"], subset["ndcg"], 
                marker=markers[mode], label=f"Gating: {mode.capitalize()}",
                color=colors[mode], linewidth=2, markersize=7, alpha=0.9
            )
            
            # Annotazioni Rate sotto i punti
            for _, row in subset.iterrows():
                plt.annotate(
                    f"{row['rate']:.0f}%", 
                    (row["time"], row["ndcg"]),
                    textcoords="offset points", xytext=(0, -18), 
                    ha="center", va="top", fontsize=8,
                    color=colors[mode], fontweight='bold'
                )

        # C. Configurazione Assi Fissi
        plt.xlim(X_LIMITS)
        plt.ylim(Y_LIMITS)
        
        plt.title(f"Trade-off Virtual Partitioning (Split: {hs_pct}%)", fontsize=14, pad=15)
        plt.xlabel("Tempo medio per query (ms)", fontsize=11)
        plt.ylabel("NDCG@10", fontsize=11)
        plt.legend(frameon=True, loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Salvataggio
        filename = target_dir / f"tradeoff_split_{hs_pct}_fixed.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✅ Generato: {filename.name}")
        plt.close()

if __name__ == "__main__":
    main()