# =======================================================
# src/eval_two_tier.py — Two-Tier Evaluation + Smart Grid Search (IRDS-based)
# =======================================================
import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyterrier as pt
import ir_datasets
import faiss
from itertools import product

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger,
)
from .two_tier_utils import FaissIVFPQSearcher, TwoTier

# -------------------- Config --------------------
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/tasb_two_tier.yaml")

TOPK         = DATA["topk"]
NPROBE       = HP["nprobe"]
NPROBE_REL   = HP["nprobe_rel"]

CACHE        = Path(PATHS.get("cache_dir", "cache"))
INDEX_ROOT   = resolve_index_root(PATHS)
RUNS_DIR     = Path(PATHS["runs_dir"])
run_dir      = stamp_run_dir(RUNS_DIR, "eval_two_tier_msmarco_dev_small")
log = get_logger(run_dir)
out_dir = run_dir
out_dir.mkdir(parents=True, exist_ok=True)

# =======================================================
# 1️⃣ Dataset & Query Cache
# =======================================================
log.info("[LOAD] Dataset: msmarco-passage/dev/small (ir_datasets)")
small_ds = ir_datasets.load("msmarco-passage/dev/small")

# ---- Query e qrels ----
topics_df = pd.DataFrame(
    [{"qid": str(q.query_id), "query": q.text} for q in small_ds.queries_iter()]
)
qrels_df = pd.DataFrame(
    [{"qid": str(q.query_id), "docno": str(q.doc_id), "label": q.relevance} for q in small_ds.qrels_iter()]
)
log.info(f"[LOAD] topics={len(topics_df)}  qrels={len(qrels_df)}")

# ---- Cache query_vec ----
Q_CACHE = CACHE / "qvec_irds:msmarco-passage_dev_recomputed.parquet"
if not Q_CACHE.exists():
    raise FileNotFoundError(f"❌ Cache mancante: {Q_CACHE}\n"
                            "Devi prima generare il file con le query encodate TAS-B.")
topics_cached = pd.read_parquet(Q_CACHE)
log.info(f"[CACHE] query_vec caricati da {Q_CACHE}")

# merge per sicurezza (associa qid→query_vec)
topics_df = topics_df.merge(topics_cached[["qid", "query_vec"]], on="qid", how="inner")
log.info(f"[MERGE] Query totali con vettore: {len(topics_df)}")

# =======================================================
# 2️⃣ Caricamento indici FAISS high/low
# =======================================================
TAG = DATA.get("dataset_tag", "msmarco")
index_high_dir = INDEX_ROOT / f"{TAG}_high_40pct_tasb_ivf_flat"
index_low_dir  = INDEX_ROOT / f"{TAG}_low_60pct_tasb_ivf_flat"

log.info(f"[LOAD] Indici: {index_high_dir.name} / {index_low_dir.name}")

def _load_index(path):
    idx = faiss.read_index(str(path))
    if isinstance(idx, faiss.IndexIVF):
        idx.nprobe = NPROBE
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = os.getenv("FAISS_FP16", "1") in ("1", "true", "True")
        idx = faiss.index_cpu_to_gpu(res, 0, idx, co)
        if isinstance(idx, faiss.IndexIVF):
            idx.nprobe = NPROBE
    return idx

retr_high = FaissIVFPQSearcher(index_high_dir, topk=TOPK, index_loader=_load_index)
retr_low  = FaissIVFPQSearcher(index_low_dir,  topk=TOPK, index_loader=_load_index)

# =======================================================
# 3️⃣ Smart Grid Search
# =======================================================
def grid_search_two_tier(retr_high, retr_low, topics_df, qrels_df, topk):
    """
    Ricerca intelligente:
    - 'margin' varia solo margin
    - 'entropy' varia solo entropy
    - 'margin_or_entropy' incrocia entrambi
    """
    
    margin_values   = [0.1,0.2,0.5,1.0,1.5,2.0,2.5,3.0,3.5,5]
    #margin_values   = np.arange(0.00, 3, 0.10).tolist()
    entropy_values = np.arange(0, 1.0, 0.10)

    modes           = ["margin", "entropy"]

    topics_df = topics_df.sample(n=1000, random_state=40)
    
    results = []
    total = sum(
        len(margin_values) if mode=="margin"
        else len(entropy_values) if mode=="entropy"
        else len(margin_values)*len(entropy_values)
        for mode in modes
    )
    log.info(f"[GRID] Avvio ricerca intelligente su {total} combinazioni...")

    for mode in modes:
        if mode == "margin":
            combos = [(m, None) for m in margin_values]
        elif mode == "entropy":
            combos = [(None, e) for e in entropy_values]
        else:  # margin_or_entropy
            margin_values   = [0.002]
            entropy_values = [0.0]
            combos = [(m, e) for m in margin_values for e in entropy_values]

        for (m, e) in combos:
            log.info(f"[GRID] mode={mode:<20} margin={m if m else '-'}  entropy={e if e else '-'}")

            two = TwoTier(
                retr_high=retr_high,
                retr_low=retr_low,
                mode=mode,
                margin=(m if m is not None else 0.05),
                margin_mode="absolute",
                tau=1,
                entropy_threshold=(e if e is not None else 0.8),
                topn_entropy=20,
                final_topk=topk,
                log_stats=True,
            )

            t0 = time.perf_counter()
            exp = pt.Experiment(
                [two], topics_df, qrels_df,
                eval_metrics=["ndcg_cut_10", "recip_rank", "map", "P_10", "recall_10"],
                names=[f"{mode}_m{m}_e{e}"], verbose=False
            )
            t1 = time.perf_counter()

            avg_time = (t1 - t0) / len(topics_df) * 1000.0
            stats = two.get_stats()

            row = exp.iloc[0].to_dict()
            row.update({
                "mode": mode,
                "margin": m,
                "entropy_thr": e,
                "avg_time_ms_per_query": avg_time,
                "low_activation_rate": stats.get("activation_rate", 0.0) * 100,
            })
            results.append(row)

    df = pd.DataFrame(results)

    # ranking metriche
    df["rank_ndcg"]   = df["ndcg_cut_10"].rank(ascending=False)
    df["rank_mrr"]    = df["recip_rank"].rank(ascending=False)
    df["rank_map"]    = df["map"].rank(ascending=False)
    df["rank_p10"]    = df["P_10"].rank(ascending=False)
    df["rank_recall"] = df["recall_10"].rank(ascending=False)
    df["rank_speed"]  = df["avg_time_ms_per_query"].rank(ascending=True)

    df["score_global"] = (
        0.35 * df["rank_ndcg"] +
        0.25 * df["rank_mrr"] +
        0.20 * df["rank_map"] +
        0.10 * df["rank_p10"] +
        0.05 * df["rank_recall"] +
        0.05 * df["rank_speed"]
    )

    df.sort_values("score_global", ascending=True, inplace=True)
    best = df.iloc[0]

    log.info(f"🏆 Migliore: mode={best['mode']}  margin={best['margin']}  entropy={best['entropy_thr']}")
    log.info(f"    nDCG@10={best['ndcg_cut_10']:.4f}  MRR={best['recip_rank']:.4f}  MAP={best['map']:.4f}")
    log.info(f"    P@10={best['P_10']:.4f}  Recall@10={best['recall_10']:.4f}")
    log.info(f"    tempo medio={best['avg_time_ms_per_query']:.2f} ms/query  attivazioni low-tier={best['low_activation_rate']:.2f}%")

    df.to_parquet(out_dir / "grid_results.parquet", index=False)
    df.to_csv(out_dir / "grid_results.csv", index=False)
    return df, best

# =======================================================
# 4️⃣ MAIN
# =======================================================
def main():
    log.info("=== Two-Tier Evaluation + Smart Grid Search ===")
    
    
    
    df_grid, best = grid_search_two_tier(retr_high, retr_low, topics_df, qrels_df, TOPK)

    two_best = TwoTier(
        retr_high=retr_high,
        retr_low=retr_low,
        mode=best["mode"],
        margin=best["margin"],
        margin_mode="absolute",
        tau=1.0,
        entropy_threshold=best["entropy_thr"],
        topn_entropy=10,
        final_topk=TOPK,
        log_stats=True,
    )

if __name__ == "__main__":
    main()
