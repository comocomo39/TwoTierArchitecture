import argparse
import sys
import time
import math
import numpy as np
import pandas as pd
import faiss
import pyterrier as pt
from pyterrier.measures import nDCG, RR, AP
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import dai tuoi file esistenti
from common import (
    load_cfg, resolve_index_root, ensure_pt, get_logger, 
    load_quality_for_sample, stamp_run_dir
)
from two_tier_utils import TwoTier, _load_ids_safe, SharedMaskedTwoTier

# ==============================================================================
# Helper per il calcolo della maschera (Simile a analyze_complete_qual)
# ==============================================================================
def compute_split_masks(index, ids, qualities, high_share, min_docs=5):
    """
    Ritorna due array di booleani (N,):
    - mask_high: True se il doc è nel top 'high_share' del suo cluster
    - mask_low:  True altrimenti
    """
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
    
    mask_low = ~mask_high
    return mask_high, mask_low

# ==============================================================================
# MAIN 
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_share", type=float, default=0.2)
    parser.add_argument("--index_name", type=str, default="msmarco_complete_tasb_ivf_flat", help="Nome indice completo (es. msmarco_complete...)")
    parser.add_argument("--gating_mode", type=str, default="margin")
    parser.add_argument("--qvec_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--nprobe", type=int, default=8192, help="NPROBE SANO (es. 32, 64)")
    args = parser.parse_args()

    ensure_pt()
    PATHS = load_cfg("configs/paths.yaml")
    DATA = load_cfg("configs/dataset.yaml")
    HP = load_cfg("configs/tasb_two_tier.yaml")
    
    INDEX_ROOT = resolve_index_root(PATHS)
    index_path = INDEX_ROOT / args.index_name
    run_dir = stamp_run_dir(PATHS["runs_dir"], f"dyn_split_{args.high_share}_{args.index_name}")
    log = get_logger(run_dir)
    
    log.info(f"Load Index: {index_path}")
    index = faiss.read_index(str(index_path / "faiss.index"))
    ids = _load_ids_safe(index_path / "ids.npy")
    
    qual_id = DATA["dataset_id_quality"] 
    qs_df = load_quality_for_sample(pd.Series(ids), qual_id, log)
    q_map = dict(zip(qs_df["docno"], qs_df["quality"]))
    qualities = np.array([q_map.get(d, np.nan) for d in ids], dtype=np.float32)
    
    log.info(f"Computing Splits (High={args.high_share})...")
    mask_high, mask_low = compute_split_masks(index, ids, qualities, args.high_share)
    log.info(f"High: {mask_high.sum():,} | Low: {mask_low.sum():,}")
    
    # Init Pipeline
    # Usiamo args.nprobe se passato, altrimenti config, altrimenti 32
    nprobe = args.nprobe if args.nprobe else HP.get("nprobe", 8192)
    
    log.info(f"Init SharedMaskedTwoTier with nprobe={nprobe}...")
    tt = SharedMaskedTwoTier(
        index=index, docnos=ids, mask_high=mask_high, mask_low=mask_low,
        mode=args.gating_mode, margin=HP.get("margin_abs", 0.5),
        tau=HP.get("entropy_tau", 1.0), entropy_threshold=HP.get("entropy_frac", 0.5),
        topn_entropy=HP.get("entropy_topn", 10), final_topk=1000,
        nprobe=nprobe, topk_tier=HP.get("topk", 100)
    )
    
    # Load Query
    dataset = pt.get_dataset(DATA["split"])
    possible_filenames = [
        f"qvec_{DATA['split'].replace('/', '_')}_recomputed.parquet",
        f"qvec_{DATA['split'].replace('/', '_')}.parquet"
    ]
    found_qvec = None
    if args.qvec_path: found_qvec = Path(args.qvec_path)
    else:
        for fname in possible_filenames:
            if (Path(PATHS.get("cache_dir", "cache")) / fname).exists():
                found_qvec = Path(PATHS.get("cache_dir", "cache")) / fname; break
    
    if not found_qvec: log.error("Cache query non trovata."); sys.exit(1)
    
    log.info(f"Load QVecs: {found_qvec}")
    topics = pd.read_parquet(found_qvec)
    topics["qid"] = topics["qid"].astype(str)
    
    # Run Batch (Sequenziale)
    log.info("Start Retrieval...")
    results = []
    total = len(topics)
    t0_global = time.perf_counter()
    
    for i in range(0, total, args.batch_size):
        batch = topics.iloc[i : i + args.batch_size].copy()
        t0 = time.perf_counter()
        res = tt.transform(batch)
        results.append(res)
        log.info(f"Batch {i//args.batch_size + 1}: {len(batch)}q in {time.perf_counter()-t0:.2f}s")
        
    res_all = pd.concat(results, ignore_index=True)
    log.info(f"Done in {time.perf_counter()-t0_global:.2f}s. Stats: {tt.get_stats()}")
    
    pt.io.write_results(res_all, str(run_dir / "run.txt"))
    
    # Eval
    log.info("Evaluating...")
    qrels = dataset.get_qrels(); qrels["qid"] = qrels["qid"].astype(str)
    res_all["qid"] = res_all["qid"].astype(str)
    rel_q = qrels[qrels["qid"].isin(res_all["qid"].unique())]
    
    if not rel_q.empty:
        eval_out = pt.Experiment([res_all], topics, rel_q, eval_metrics=[nDCG@10, RR@10, AP], names=["TwoTier"])
        log.info(f"\n{eval_out}")
    else:
        log.warning("No qrels overlap.")

if __name__ == "__main__":
    main()