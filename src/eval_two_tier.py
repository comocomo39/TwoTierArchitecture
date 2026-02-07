import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyterrier as pt
import faiss
import ir_datasets

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger
)
from .two_tier_utils import FaissIVFPQSearcher, TwoTier

# Config Init
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/tasb_two_tier.yaml")

TOPK = DATA["topk"]
CACHE = Path(PATHS.get("cache_dir", "cache"))
INDEX_ROOT = resolve_index_root(PATHS)
RUNS_DIR = Path(PATHS["runs_dir"])

# Nomi indici (Adatta se necessario)
TAG = DATA.get("dataset_tag", "msmarco")
IDX_HIGH30 = f"{TAG}_high_50pct_tasb_ivf_flat" 
IDX_LOW70  = f"{TAG}_low_50pct_tasb_ivf_flat"

run_dir = stamp_run_dir(RUNS_DIR, "eval_two_tier_comparison")
log = get_logger(run_dir)
out_dir = run_dir
out_dir.mkdir(parents=True, exist_ok=True)

# FAISS Loader Helper
def _load_faiss_index_factory(nprobe_val):
    def _loader(path):
        idx = faiss.read_index(str(path))
        if isinstance(idx, faiss.IndexIVF):
            idx.nprobe = nprobe_val
        return idx
    return _loader

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Argomento chiave per la velocità
    parser.add_argument("--nprobe", type=int, default=128, help="Nprobe per confronto equo (default 32)")
    args = parser.parse_args()

    log.info(f"=== Physical Evaluation: Two-Tier vs High-Only | nprobe={args.nprobe} ===")

    # 1. Dataset & Qrels
    small_ds = ir_datasets.load("msmarco-passage/dev/small")
    topics_df = pd.DataFrame([{"qid": str(q.query_id), "query": q.text} for q in small_ds.queries_iter()])
    qrels_df = pd.DataFrame([{"qid": str(q.query_id), "docno": str(q.doc_id), "label": q.relevance} for q in small_ds.qrels_iter()])
    
    # 2. Query Vectors (Cache)
    qvec_cache = CACHE / "qvec_irds:msmarco-passage_dev_recomputed.parquet"
    if not qvec_cache.exists(): raise FileNotFoundError("Cache query mancante")
    
    topics_cached = pd.read_parquet(qvec_cache)
    topics_df = topics_df.merge(topics_cached[["qid", "query_vec"]], on="qid", how="inner")
    log.info(f"Queries Loaded: {len(topics_df)}")

    # 3. Indici Fisici
    idx_high_dir = INDEX_ROOT / IDX_HIGH30
    idx_low_dir  = INDEX_ROOT / IDX_LOW70
    
    # Usiamo il loader che imposta nprobe
    loader = _load_faiss_index_factory(args.nprobe)
    
    retr_high = FaissIVFPQSearcher(idx_high_dir, topk=TOPK, index_loader=loader, nprobe=args.nprobe)
    retr_low  = FaissIVFPQSearcher(idx_low_dir,  topk=TOPK, index_loader=loader, nprobe=args.nprobe)

    # ------------------------------------------------------------------
    # RUN 1: TWO TIER (Physical)
    # ------------------------------------------------------------------
    two_tier_pipeline = TwoTier(
        retr_high=retr_high,
        retr_low=retr_low,
        mode="margin",
        margin=HP["margin_abs"], # 0.05 default
        final_topk=TOPK,
        log_stats=True
    )

    log.info("\n>>> Running Two-Tier (High + Low w/ Gating)...")
    t0 = time.perf_counter()
    res_two_tier = two_tier_pipeline.transform(topics_df)
    t1 = time.perf_counter()
    
    time_tt = (t1 - t0) * 1000
    msq_tt = time_tt / len(topics_df)
    stats_tt = two_tier_pipeline.get_stats()
    
    log.info(f"DONE Two-Tier. Time: {time_tt:.2f}ms | {msq_tt:.2f} ms/q | Acts: {stats_tt['activation_rate']*100:.1f}%")

    # ------------------------------------------------------------------
    # RUN 2: HIGH ONLY (Physical)
    # ------------------------------------------------------------------
    # Nota: Eseguiamo semplicemente il searcher 'retr_high' da solo.
    # Questo simula un sistema che ignora completamente il tier 2.
    
    log.info("\n>>> Running High-Only (Single Tier)...")
    t0 = time.perf_counter()
    res_high_only = retr_high.transform(topics_df)
    t1 = time.perf_counter()
    
    time_ho = (t1 - t0) * 1000
    msq_ho = time_ho / len(topics_df)
    
    log.info(f"DONE High-Only. Time: {time_ho:.2f}ms | {msq_ho:.2f} ms/q")

    # ------------------------------------------------------------------
    # EVALUATION & COMPARISON
    # ------------------------------------------------------------------
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    
    # Filtriamo qrels rilevanti per le query processate
    processed_qids = set(topics_df["qid"].unique())
    relevant = qrels_df[qrels_df["qid"].isin(processed_qids)]
    
    # Tabella Comparativa
    systems = [res_two_tier, res_high_only]
    names = ["Physical Two-Tier", "Physical High-Only"]
    
    eval_metrics = ["ndcg_cut_10", "recip_rank", "map"]
    
    log.info("\n=== COMPARATIVE RESULTS ===")
    eval_out = pt.Experiment(
        systems, topics_df, relevant, 
        eval_metrics=eval_metrics, names=names
    )
    
    # Aggiungiamo colonne di performance manuali al dataframe dei risultati
    # (pt.Experiment ritorna un DF, lo arricchiamo)
    perf_data = {
        "Physical Two-Tier": {"Time (ms)": time_tt, "ms/q": msq_tt, "Low Act %": stats_tt['activation_rate']*100},
        "Physical High-Only": {"Time (ms)": time_ho, "ms/q": msq_ho, "Low Act %": 0.0}
    }
    
    # Join manuale per visualizzazione
    final_rows = []
    for i, row in eval_out.iterrows():
        name = row["name"]
        p = perf_data.get(name, {})
        new_row = row.to_dict()
        new_row.update(p)
        final_rows.append(new_row)
        
    final_df = pd.DataFrame(final_rows)
    # Riordina colonne per leggibilità
    cols = ["name", "ms/q", "Low Act %", "ndcg_cut_10", "recip_rank", "map"]
    final_df = final_df[cols]
    
    # Stampa tabellare pulita
    print("\n" + final_df.to_string(index=False, float_format="{:.4f}".format))
    
    # Salva
    out_csv = run_dir / "comparison_results.csv"
    final_df.to_csv(out_csv, index=False)
    log.info(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    main()