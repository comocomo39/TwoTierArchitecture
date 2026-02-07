# src/evaluate_single_complete.py
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyterrier as pt
import faiss

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger
)

# Configurazione
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/tasb_two_tier.yaml")

SPLIT        = DATA["split"]
TOPK         = int(DATA["topk"])
NPROBE       = int(HP["nprobe"])
INDEX_ROOT   = resolve_index_root(PATHS)
IDX_COMPLETE = "msmarco_complete_tasb_ivf_flat"

run_dir = stamp_run_dir(PATHS["runs_dir"], "eval_single_complete")
log = get_logger(run_dir)

# Helper caricamento ID sicuro
def _load_ids_safe(path: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError:
        arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=str)

# =======================================================
# SEARCHER COMPLETO OTTIMIZZATO (Vettorializzato)
# =======================================================
class FastFaissIVFSearcher(pt.Transformer):
    """
    Versione OTTIMIZZATA: niente cicli for Python.
    Usa NumPy per formattare i risultati istantaneamente.
    """
    def __init__(self, index_dir: Path, topk: int = 1000, nprobe: int = 128):
        self.index_dir = Path(index_dir)
        self.topk = int(topk)
        
        # Carica indice
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = nprobe
            
        # Carica ID
        self.ids = _load_ids_safe(self.index_dir / "ids.npy")
        
        # Performance: Disabilita thread interni per test puliti single-thread
        # faiss.omp_set_num_threads(1) 

    def transform(self, df_queries: pd.DataFrame) -> pd.DataFrame:
        # Preparazione dati (Zero-copy cast)
        qids  = df_queries["qid"].astype(str).to_numpy()
        qvecs = np.ascontiguousarray(np.vstack(df_queries["query_vec"].values).astype(np.float32))
        
        # === RICERCA FAISS ===
        t0 = time.perf_counter()
        scores, idxs = self.index.search(qvecs, self.topk)
        
        # === FORMATTAZIONE VETTORIALE (Il trucco della velocità) ===
        # Invece del ciclo for, usiamo operazioni matriciali
        n_q, k = scores.shape
        scores_flat = scores.ravel()
        idxs_flat = idxs.ravel()
        
        # Maschera per risultati validi
        valid_mask = idxs_flat >= 0
        if not np.any(valid_mask):
            return pd.DataFrame(columns=["qid", "docno", "rank", "score"])

        # Costruiamo le colonne di output in un colpo solo (C-Speed)
        out_qids = np.repeat(qids, k)[valid_mask]
        out_docs = self.ids[idxs_flat[valid_mask]] # Lookup diretto nell'array ID
        out_scores = scores_flat[valid_mask]
        out_ranks = np.tile(np.arange(1, k + 1), n_q)[valid_mask]

        return pd.DataFrame({
            "qid": out_qids,
            "docno": out_docs,
            "rank": out_ranks,
            "score": out_scores,
        })

# =======================================================
# MAIN
# =======================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Importante: permetti di sovrascrivere nprobe da riga di comando
    parser.add_argument("--nprobe", type=int, default=NPROBE, help="Override nprobe")
    args = parser.parse_args()
    
    # 1. Init Index
    idx_dir = INDEX_ROOT / IDX_COMPLETE
    log.info(f"Loading Index: {idx_dir} (nprobe={args.nprobe})")
    
    retr = FastFaissIVFSearcher(idx_dir, topk=TOPK, nprobe=args.nprobe)
    
    # 2. Load Queries (Cache)
    dataset = pt.get_dataset(SPLIT)
    split_name_clean = DATA["split"].replace("/", "_")
    possible_filenames = [
        f"qvec_{split_name_clean}_recomputed.parquet",
        f"qvec_{split_name_clean}.parquet"
    ]
    CACHE = Path(PATHS.get("cache_dir", "cache"))
    found_qvec = None
    for fname in possible_filenames:
        if (CACHE / fname).exists():
            found_qvec = CACHE / fname; break
            
    if not found_qvec: raise FileNotFoundError("Query cache non trovata")
    
    log.info(f"Loading QVecs: {found_qvec}")
    queries_df = pd.read_parquet(found_qvec)
    queries_df["qid"] = queries_df["qid"].astype(str)
    
    # 3. Benchmark Tempo
    log.info("Starting Benchmark...")
    
    t0 = time.perf_counter()
    res = retr.transform(queries_df)
    t1 = time.perf_counter()
    
    total_ms = (t1 - t0) * 1000
    per_q = total_ms / len(queries_df)
    
    log.info("="*40)
    log.info(f"COMPLETE INDEX PERFORMANCE (Optimized)")
    log.info(f"Total Queries: {len(queries_df)}")
    log.info(f"Total Time   : {total_ms:.2f} ms")
    log.info(f"Per Query    : {per_q:.2f} ms/query")
    log.info("="*40)
    
    # 4. Evaluation Quality
    log.info("Evaluating Quality...")
    qrels = dataset.get_qrels()
    qrels["qid"] = qrels["qid"].astype(str)
    
    relevant = qrels[qrels["qid"].isin(queries_df["qid"])]
    
    eval_out = pt.Experiment(
        [res], queries_df, relevant, 
        eval_metrics=["ndcg_cut_10", "recip_rank"], 
        names=["Complete_Optimized"]
    )
    log.info(f"\n{eval_out}")

if __name__ == "__main__":
    main()