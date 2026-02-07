#!/usr/bin/env python3
# =======================================================
# src/eval_virtual_gridsearch.py
# Grid Search per Virtual Partitioning (Margin & Entropy)
# =======================================================
import argparse
import sys
import time
import math
import numpy as np
import pandas as pd
import faiss
import pyterrier as pt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import dai tuoi file esistenti
from common import (
    load_cfg, resolve_index_root, ensure_pt, get_logger, 
    load_quality_for_sample, stamp_run_dir
)
from two_tier_utils import _load_ids_safe, SharedMaskedTwoTier

# ==============================================================================
# CONFIGURAZIONE GRID SEARCH
# ==============================================================================
GRID_HIGH_SHARES   = [0.2, 0.3, 0.4, 0.5] 
GRID_MARGINS       = [0.1,0.2,0.5,1.0,1.5,2.0,2.5,3.0,3.5,5]
GRID_ENTROPY_THRS  = np.arange(0, 1.0, 0.10) # Aggiunte soglie entropia
MODES              = ["margin", "entropy"]           # Entrambe le modalità

# ==============================================================================
# Helper per il calcolo della maschera 
# ==============================================================================
def compute_split_masks(index, ids, qualities, high_share, min_docs=5):
    N = len(ids)
    list_id_of_vec = np.full(N, -1, dtype=np.int32)
    invlists = index.invlists
    
    rev_swig_ptr = faiss.rev_swig_ptr
    for list_no in range(index.nlist):
        ls = invlists.list_size(list_no)
        if ls == 0: continue
        ids_ptr = invlists.get_ids(list_no)
        c_ids = rev_swig_ptr(ids_ptr, ls)
        list_id_of_vec[c_ids] = list_no
        
    mask_high = np.zeros(N, dtype=bool)
    
    perm = np.argsort(list_id_of_vec)
    sorted_clusters = list_id_of_vec[perm]
    sorted_qual = qualities[perm]
    sorted_ids = perm 
    
    diff = np.where(np.diff(sorted_clusters) != 0)[0] + 1
    splits = np.split(np.arange(len(perm)), diff)
    
    high_indices_list = []
    
    for idxs in splits:
        if len(idxs) == 0: continue
        real_ids = sorted_ids[idxs]
        qs = sorted_qual[idxs]
        
        valid_mask = ~np.isnan(qs)
        if valid_mask.sum() < min_docs:
            continue
            
        valid_qs = qs[valid_mask]
        thr = np.quantile(valid_qs, 1.0 - high_share)
        
        is_high = (qs >= thr)
        high_indices_list.append(real_ids[is_high])

    if high_indices_list:
        all_high = np.concatenate(high_indices_list)
        mask_high[all_high] = True
    
    return mask_high, ~mask_high

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", type=str, default="msmarco_complete_tasb_ivf_flat")
    parser.add_argument("--qvec_path", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--sample_queries", type=int, default=6980)
    args = parser.parse_args()

    ensure_pt()
    
    # Config
    PATHS = load_cfg("configs/paths.yaml")
    DATA  = load_cfg("configs/dataset.yaml")
    HP    = load_cfg("configs/tasb_two_tier.yaml")
    
    INDEX_ROOT = resolve_index_root(PATHS)
    index_path = INDEX_ROOT / args.index_name
    CACHE_DIR  = Path(PATHS.get("cache_dir", "cache"))
    
    run_dir = stamp_run_dir(PATHS["runs_dir"], f"grid_virtual_{args.index_name}")
    log = get_logger(run_dir)
    
    # 1. Caricamento Dati
    index = faiss.read_index(str(index_path / "faiss.index"))
    ids = _load_ids_safe(index_path / "ids.npy")
    
    qual_id = DATA["dataset_id_quality"]
    qs_df = load_quality_for_sample(pd.Series(ids), qual_id, log)
    q_map = dict(zip(qs_df["docno"], qs_df["quality"]))
    qualities = np.array([q_map.get(d, np.nan) for d in ids], dtype=np.float32)
    
    # 2. Query & Qrels
    dataset = pt.get_dataset(DATA["split"])
    qrels = dataset.get_qrels()
    
    split_name_clean = DATA["split"].replace("/", "_")
    found_qvec = Path(args.qvec_path) if args.qvec_path else CACHE_DIR / f"qvec_{split_name_clean}_recomputed.parquet"
    
    topics_df = pd.read_parquet(found_qvec)
    topics_df["qid"] = topics_df["qid"].astype(str)
    qrels["qid"] = qrels["qid"].astype(str)
    relevant_qrels = qrels[qrels["qid"].isin(topics_df["qid"])].copy()
    
    # 3. Grid Search Loop
    grid_results = []
    
    for hs in GRID_HIGH_SHARES:
        log.info(f"--- [GRID] Computing Masks for High Share {hs:.2f} ---")
        mask_high, mask_low = compute_split_masks(index, ids, qualities, high_share=hs)
        
        # Inizializziamo tt (SharedMaskedTwoTier)
        tt = SharedMaskedTwoTier(
            index=index, docnos=ids, mask_high=mask_high, mask_low=mask_low,
            mode="margin", # placeholder
            margin=0.0,
            tau=HP.get("entropy_tau", 1.0), 
            entropy_threshold=0.0,
            topn_entropy=HP.get("entropy_topn", 10), 
            final_topk=int(DATA["topk"]),
            nprobe=args.nprobe, 
            topk_tier=int(DATA["topk"])
        )

        for mode in MODES:
            # Scegliamo quale lista di parametri usare
            thresholds = GRID_MARGINS if mode == "margin" else GRID_ENTROPY_THRS
            tt.mode = mode
            
            for val in thresholds:
                # Reset stats e aggiorna parametro specifico
                tt.reset_stats()
                if mode == "margin":
                    tt.margin = val
                    tt.entropy_threshold = 0.0 # disabilita l'altro
                else:
                    tt.entropy_threshold = val
                    tt.margin = 0.0 # disabilita l'altro

                log.info(f" > Run: Share={hs:.2f}, Mode={mode}, Val={val:.4f}")
                
                t_start = time.perf_counter()
                results = []
                mini_batch = 250 
                
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {executor.submit(tt.transform, topics_df.iloc[i:i+mini_batch].copy()): i 
                               for i in range(0, len(topics_df), mini_batch)}
                    for future in as_completed(futures):
                        results.append(future.result())
                
                res_all = pd.concat(results, ignore_index=True)
                t_end = time.perf_counter()
                
                # Eval
                eval_metrics = ["ndcg_cut_10", "recip_rank", "map"]
                exp_res = pt.Experiment([res_all], topics_df, relevant_qrels, 
                                        eval_metrics=eval_metrics, verbose=False).iloc[0]
                
                stats = tt.get_stats()
                avg_time_ms = ((t_end - t_start) * 1000.0) / len(topics_df)
                
                row = exp_res.to_dict()
                row.update({
                    "mode": mode,
                    "high_share": hs,
                    "margin": val if mode == "margin" else np.nan,
                    "entropy_thr": val if mode == "entropy" else np.nan,
                    "avg_time_ms_per_query": avg_time_ms,
                    "low_activation_rate": stats["rate"] * 100,
                })
                
                log.info(f"    -> nDCG@10: {row['ndcg_cut_10']:.4f} | Time: {avg_time_ms:.1f}ms | Rate: {row['low_activation_rate']:.1f}%")
                grid_results.append(row)

    # 4. Salvataggio e Ranking
    df = pd.DataFrame(grid_results)
    
    # Sorting per trovare il migliore (esempio pesato)
    df["rank_ndcg"]  = df["ndcg_cut_10"].rank(ascending=False)
    df["rank_speed"] = df["avg_time_ms_per_query"].rank(ascending=True)
    df["score_global"] = 0.5 * df["rank_ndcg"] + 0.5 * df["rank_speed"]
    
    df.sort_values("score_global", inplace=True)
    
    out_file_csv = run_dir / "grid_virtual_results.csv"
    df.to_csv(out_file_csv, index=False)
    log.info(f"🏆 Best: {df.iloc[0]['mode']} | Share: {df.iloc[0]['high_share']} | nDCG: {df.iloc[0]['ndcg_cut_10']:.4f}")
    log.info(f"[SAVE] Risultati salvati in {out_file_csv}")

if __name__ == "__main__":
    main()