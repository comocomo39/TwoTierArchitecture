# src/analyze_results.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# ========== CONFIG ==========
# imposta qui la cartella del run che vuoi analizzare
# esempio: Path("runs/2025-10-24_18-33-59")
RUN_DIR = Path("runs")  # verrà risolto automaticamente se contiene un solo run
# ============================

def _resolve_eval_dir(run_dir: Path) -> Path:
    """Cerca automaticamente l'ultima sottocartella 'eval'."""
    if (run_dir / "eval").exists():
        return run_dir / "eval"
    evals = list(run_dir.rglob("eval"))
    if len(evals) == 1:
        return evals[0]
    if len(evals) > 1:
        evals.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return evals[0]
    raise FileNotFoundError(f"Nessuna cartella eval trovata in {run_dir}")

def load_data(eval_dir: Path):
    results_csv = list(eval_dir.rglob("results.csv"))[0]
    trace_csv   = list(eval_dir.rglob("twotier_trace_per_query.csv"))[0]
    summary_js  = list(eval_dir.rglob("twotier_trace_summary.json"))[0]

    results_df = pd.read_csv(results_csv)
    trace_df   = pd.read_csv(trace_csv)
    summary    = json.loads(Path(summary_js).read_text())

    return results_df, trace_df, summary, results_csv, trace_csv, summary_js

def analyze_trace(trace_df: pd.DataFrame):
    trace_df["low_triggered"] = trace_df["low_triggered"].astype(bool)
    trace_df["delta_ndcg@k"] = trace_df["delta_ndcg@k"].astype(float)
    trace_df["delta_mrr"] = trace_df["delta_mrr"].astype(float)
    trace_df["delta_recall@k"] = trace_df["delta_recall@k"].astype(float)
    trace_df["gains_rel"] = trace_df["gains_rel"].astype(int)

    called = trace_df[trace_df["low_triggered"] == True].copy()
    called["result_cat"] = np.where(
        called["delta_ndcg@k"] > 1e-9, "win",
        np.where(called["delta_ndcg@k"] < -1e-9, "loss", "tie")
    )

    win_rate  = (called["result_cat"] == "win").mean() if len(called) else 0.0
    loss_rate = (called["result_cat"] == "loss").mean() if len(called) else 0.0
    tie_rate  = (called["result_cat"] == "tie").mean() if len(called) else 0.0

    uplift_mean = called["delta_ndcg@k"].mean() if len(called) else 0.0
    recall_gain_mean = called["delta_recall@k"].mean() if len(called) else 0.0
    gains_rel_total = int(called["gains_rel"].sum()) if len(called) else 0

    best_help = called.sort_values("delta_ndcg@k", ascending=False).head(20)
    worst_hurt = called.sort_values("delta_ndcg@k", ascending=True).head(20)

    stats = {
        "n_queries_total": len(trace_df),
        "n_low_called": len(called),
        "pct_low_called": len(called) / max(1, len(trace_df)),
        "win_rate_when_called": win_rate,
        "loss_rate_when_called": loss_rate,
        "tie_rate_when_called": tie_rate,
        "uplift_mean_ndcg_when_called": uplift_mean,
        "recall_gain_mean_when_called": recall_gain_mean,
        "total_new_relevants_from_low": gains_rel_total,
    }

    return stats, best_help, worst_hurt

def compare_global(results_df: pd.DataFrame):
    base = results_df.set_index("name")
    if "High30_only" in base.index and "TwoTier_High30+Low70" in base.index:
        base_row = base.loc["High30_only"]
        two_row  = base.loc["TwoTier_High30+Low70"]
        diff = (two_row - base_row)[["ndcg_cut_10","recip_rank","map","P_10"]]
        return {
            "baseline_ndcg@10": float(base_row["ndcg_cut_10"]),
            "twotier_ndcg@10": float(two_row["ndcg_cut_10"]),
            "delta_ndcg@10": float(diff["ndcg_cut_10"]),
            "baseline_mrr": float(base_row["recip_rank"]),
            "twotier_mrr": float(two_row["recip_rank"]),
            "delta_mrr": float(diff["recip_rank"]),
            "baseline_P10": float(base_row["P_10"]),
            "twotier_P10": float(two_row["P_10"]),
            "delta_P10": float(diff["P_10"]),
        }
    return {}

def write_log(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def main():
    eval_dir = _resolve_eval_dir(RUN_DIR)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = eval_dir.parent / f"log_analysis.txt"

    results_df, trace_df, summary, results_csv, trace_csv, summary_js = load_data(eval_dir)
    stats, best_help, worst_hurt = analyze_trace(trace_df)
    deltas_global = compare_global(results_df)

    write_log(log_path, f"\n=== LOG ANALYSIS {timestamp} ===")
    write_log(log_path, f"Eval dir: {eval_dir}")
    write_log(log_path, f"Results CSV: {results_csv}")
    write_log(log_path, f"Trace CSV: {trace_csv}")
    write_log(log_path, f"Summary JSON: {summary_js}\n")

    write_log(log_path, "=== OVERALL IMPACT (GLOBAL METRICS) ===")
    for k,v in deltas_global.items():
        write_log(log_path, f"{k}: {v}")
    write_log(log_path, "")

    write_log(log_path, "=== GATING DIAGNOSTICS (PER-QUERY ANALYSIS) ===")
    for k,v in stats.items():
        write_log(log_path, f"{k}: {v}")
    write_log(log_path, "")

    write_log(log_path, "=== SUMMARY FROM EVAL (GROUND TRUTH FROM RUNTIME) ===")
    for k,v in summary.items():
        write_log(log_path, f"{k}: {v}")
    write_log(log_path, "")

    write_log(log_path, "=== TOP 20 HELPED QUERIES ===")
    write_log(log_path, best_help[["qid","delta_ndcg@k","delta_mrr","delta_recall@k","gains_rel"]].to_string(index=False))
    write_log(log_path, "")
    write_log(log_path, "=== TOP 20 HURT QUERIES ===")
    write_log(log_path, worst_hurt[["qid","delta_ndcg@k","delta_mrr","delta_recall@k","gains_rel"]].to_string(index=False))
    write_log(log_path, "")
    write_log(log_path, "=== END OF ANALYSIS ===\n")

    print(f"✅ Analisi completata. Log salvato in: {log_path}")

if __name__ == "__main__":
    main()
