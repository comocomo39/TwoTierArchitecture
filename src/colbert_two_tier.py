# src/colbert_two_tier.py
import os, time, glob, shutil, random
from pathlib import Path
import numpy as np, pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pyterrier as pt
import ir_datasets

# ColBERT (pyterrier_colbert)
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger,
    by_query_topk
)

# ========= Init & Config =========
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/colbert_two_tier.yaml")

INDEX_ROOT = resolve_index_root(PATHS)
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT         = DATA["split"]
SAMPLE_MODE   = DATA["sample_mode"]          # "sample" | "all_queries"
SAMPLE_TOPICS = DATA["sample_topics"]
NEG_PER_REL   = DATA["neg_per_rel"]
MAX_DOCS      = DATA["max_docs"]
HIGH_SHARE    = DATA["high_share"]
TOPK          = DATA["topk"]
DATASET_ID    = DATA["dataset_id_quality"]

# ColBERT / runtime
CHECKPOINT   = HP["checkpoint"]
NPROBE       = HP["nprobe"]
FAISS_DEPTH  = HP["faiss_depth"]
GPU_BUILD    = HP["gpu_build"]
RUN_DENSE    = HP["run_dense"]
RUN_NP       = HP["run_np"]  # placeholder off

# gating
MARGIN_ABS   = HP["margin_abs"]
ENTROPY_TAU  = HP["entropy_tau"]
ENTROPY_FRAC = HP["entropy_frac"]
ENTROPY_TOPN = HP["entropy_topn"]

run_dir = stamp_run_dir(PATHS["runs_dir"], "colbert_two_tier")
log = get_logger(run_dir)

# ========= Dataset & sample =========
ds = pt.get_dataset(SPLIT)
topics_all = ds.get_topics().astype({"qid":"str"})
qrels_all  = ds.get_qrels().astype({"qid":"str","docno":"str"})

if SAMPLE_MODE == "sample":
    topics = topics_all.sample(n=min(SAMPLE_TOPICS, len(topics_all)), random_state=42).sort_values("qid")
    qrels  = qrels_all[qrels_all["qid"].isin(topics["qid"])].copy()
else:
    topics = topics_all.sort_values("qid")
    qrels  = qrels_all.copy()

rel_docnos = set(qrels.loc[qrels["label"] > 0, "docno"])

def build_corpus_with_rel(dataset, must_have: set, max_docs: int | None, neg_per_rel: int = 3, seed: int = 42):
    random.seed(seed)
    rows_rel, rows_neg = [], []
    # rilevanti
    for r in dataset.get_corpus_iter():
        pid  = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
        if pid in must_have:
            text = (r.get("text") or "").strip()
            if text:
                rows_rel.append({"docno": pid, "text": text})
        if len(rows_rel) == len(must_have):
            break
    # negativi
    if max_docs is None:
        need_negs = len(must_have) * neg_per_rel
    else:
        need_negs = max(0, min(max_docs, len(must_have)*(1+neg_per_rel)) - len(rows_rel))
    if need_negs > 0:
        taken = {r["docno"] for r in rows_rel}
        for r in dataset.get_corpus_iter():
            pid  = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
            if pid in taken or pid in must_have:
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            rows_neg.append({"docno": pid, "text": text})
            if max_docs is not None and len(rows_neg) >= need_negs:
                break
    return pd.DataFrame(rows_rel + rows_neg)

_passages_max = MAX_DOCS if SAMPLE_MODE == "sample" else None
passages = build_corpus_with_rel(ds, rel_docnos, max_docs=_passages_max, neg_per_rel=NEG_PER_REL, seed=42)
passages.drop_duplicates("docno", inplace=True)
log.info(f"[subset] rel_included={len(rel_docnos & set(passages['docno']))}  total_subset={len(passages)}")

# ========= QualT5 split =========
from pyterrier_quality import QualCache

def _iter_cache_rows(qc):
    for rec in qc:
        try:
            docno = str(rec.get("docno")); qual = float(rec.get("quality"))
        except Exception:
            try:
                docno, qual = rec; docno = str(docno); qual = float(qual)
            except Exception:
                try:
                    docno = str(getattr(rec, "docno")); qual  = float(getattr(rec, "quality"))
                except Exception:
                    continue
        if docno is not None:
            yield docno, qual

def load_quality_for_sample(sample_docnos: pd.Series, dataset_id=DATASET_ID):
    wanted = set(sample_docnos.astype(str).tolist())
    try:
        qc = QualCache.from_url(f"hf:{dataset_id}@quantiles")
        rows = [(d,q) for d,q in _iter_cache_rows(qc) if d in wanted]
        df = pd.DataFrame(rows, columns=["docno","quality"])
        if not df.empty:
            log.info("[QualT5] @quantiles OK → [0,1] globali.")
            return df
        else:
            log.info("[QualT5] @quantiles OK ma nessuna sovrapposizione; Fallback RAW.")
    except Exception as e:
        log.info(f"[QualT5] @quantiles fallita ({e}). Fallback RAW).")
        try:
            for p in glob.glob(os.path.expanduser("~/.pyterrier/**/qt5-tiny.msmarco-passage.cache*"), recursive=True):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    qc = QualCache.from_url(f"hf:{dataset_id}")
    rows = [(d,float(q)) for d,q in _iter_cache_rows(qc) if d in wanted]
    if not rows:
        raise RuntimeError("QualT5: nessuna sovrapposizione sul sample.")
    df = (pd.DataFrame(rows, columns=["docno","quality_raw"])
            .sort_values(["quality_raw","docno"]).reset_index(drop=True))
    df["quality"] = df["quality_raw"].rank(pct=True, method="average")
    log.info("[QualT5] RAW → percentili [0,1] nel sample.")
    return df[["docno","quality"]]

qual = load_quality_for_sample(passages["docno"])
passages_q = passages.merge(qual, on="docno", how="inner")
cut = 1.0 - HIGH_SHARE
high = passages_q.loc[passages_q["quality"] >= cut, ["docno","text"]]
low  = passages_q.loc[passages_q["quality"] <  cut, ["docno","text"]]
log.info(f"[SPLIT] High={len(high)}  Low={len(low)}  matched={len(passages_q)}/{len(passages)}")

indexed_docnos = set(pd.concat([high["docno"], low["docno"]]))
qrels_cov = qrels[qrels["docno"].isin(indexed_docnos)].copy()
topics_cov = topics[topics["qid"].isin(set(qrels_cov["qid"]))].copy()
log.info(f"[coverage] queries con almeno un rilevante nel subset: {len(topics_cov)} / {len(topics)}")

# ========= ColBERT indexing =========
IDX_HIGH, IDX_LOW = "msmarco_high_70pct", "msmarco_low_30pct"

def ensure_colbert_index(df: pd.DataFrame, index_name: str, gpu: bool=False):
    idx_dir = Path(INDEX_ROOT) / index_name
    exists = idx_dir.exists() and any(idx_dir.rglob("*.faiss"))
    if exists:
        log.info(f"[ColBERT] Trovato indice '{index_name}'.")
        return
    log.info(f"[ColBERT] Indicizzo '{index_name}' ({len(df)} docs)...")
    try:
        indexer = ColBERTIndexer(CHECKPOINT, str(INDEX_ROOT), index_name,
                                 ids=True, chunksize=64, gpu=gpu, doc_maxlen=64)
    except TypeError:
        indexer = ColBERTIndexer(CHECKPOINT, str(INDEX_ROOT), index_name,
                                 ids=True, chunksize=64, gpu=gpu)
    indexer.index(df[["docno","text"]].to_dict("records"))

def detect_faiss_partitions(index_name: str) -> int:
    idx_dir = Path(INDEX_ROOT) / index_name
    files = list(idx_dir.glob("ivfpq.*.faiss"))
    if not files:
        return 100
    fname = files[0].name
    try:
        return int(fname.split(".")[1])
    except Exception:
        return 100

ensure_colbert_index(high, IDX_HIGH, gpu=GPU_BUILD)
ensure_colbert_index(low,  IDX_LOW,  gpu=GPU_BUILD)

part_high = detect_faiss_partitions(IDX_HIGH)
part_low  = detect_faiss_partitions(IDX_LOW)

pyt_high = ColBERTFactory(CHECKPOINT, str(INDEX_ROOT), IDX_HIGH,
                          faiss_partitions=part_high, memtype="mem", faisstype="mem", gpu=GPU_BUILD)
pyt_low  = ColBERTFactory(CHECKPOINT, str(INDEX_ROOT), IDX_LOW,
                          faiss_partitions=part_low,  memtype="mem", faisstype="mem", gpu=GPU_BUILD)

for f in (pyt_high, pyt_low):
    f.args.nprobe = NPROBE
    f.args.faiss_depth = FAISS_DEPTH

# ========= Query set (testuale) =========
qdf = topics_cov[["qid","query"]].copy()
log.info("[ENC] (ColBERT) end_to_end() con query testuali; no pre-encoding.")

def get_colbert_searchers(topk=TOPK):
    try:
        retrH = pyt_high.end_to_end() % topk
        retrL = pyt_low.end_to_end()  % topk
    except TypeError:
        retrH = pyt_high.end_to_end() >> by_query_topk(topk)
        retrL = pyt_low.end_to_end()  >> by_query_topk(topk)
    return retrH, retrL

# ========= TwoTier gating (coerente con TAS-B) =========
class TwoTier(pt.Transformer):
    def __init__(self, retr_high, retr_low, mode="margin_or_entropy",
                 margin=0.05, margin_mode="absolute",
                 tau=1.0, entropy_threshold=0.8, topn_entropy=10, final_topk=None,
                 log_stats=True, per_query_log=False, time_unit="ms"):
        self.retr_high, self.retr_low = retr_high, retr_low
        self.mode, self.margin = mode, float(margin)
        self.margin_mode = str(margin_mode).lower()
        self.tau, self.entropy_threshold = float(tau), float(entropy_threshold)
        self.topn_entropy, self.final_topk = int(topn_entropy), final_topk
        self.log_stats = bool(log_stats)
        self.per_query_log = bool(per_query_log)
        self.time_unit = "ms" if time_unit not in ("ms","s") else time_unit
        self._eps = 1e-12
        self.reset_stats()

    def reset_stats(self):
        self._q = 0
        self._low_acts = 0
        self._t_high = 0.0
        self._t_low  = 0.0
        self._t_merge= 0.0

    def _s(self, seconds):
        return seconds * (1000.0 if self.time_unit == "ms" else 1.0)

    def _gating_flags(self, run_h: pd.DataFrame) -> pd.Series:
        def need_low(group: pd.DataFrame) -> bool:
            s = group["score"].to_numpy(dtype=np.float32, copy=False)
            if len(s) == 0:
                return True
            # margin
            if len(s) < 2:
                need_m = True
            else:
                s_sorted = np.sort(s)[::-1]
                s1, s2 = float(s_sorted[0]), float(s_sorted[1])
                if self.margin_mode == "relative":
                    rel_gap = (s1 - s2) / (abs(s1) + self._eps)
                    need_m = rel_gap < self.margin
                else:
                    need_m = (s1 - s2) < self.margin
            # entropy
            k = min(self.topn_entropy, len(s))
            if k == 0:
                need_e = True
            else:
                top = np.sort(s)[-k:]
                z = top / max(self.tau, self._eps); z -= np.max(z)
                p = np.exp(z); p /= (p.sum() + self._eps)
                ent = float(-(p * np.log(p + self._eps)).sum())
                ent_max = float(np.log(k))
                thr = self.entropy_threshold * ent_max if 0 < self.entropy_threshold <= 1 else self.entropy_threshold
                need_e = ent >= thr
            if self.mode == "margin":  return need_m
            if self.mode == "entropy": return need_e
            if self.mode == "margin_or_entropy":  return need_m or need_e
            return (len(s) < 2) or (need_m and need_e)

        flags = run_h.groupby("qid", sort=False).apply(need_low)
        flags.name = "need_low"
        return flags

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.perf_counter()
        run_h = self.retr_high.transform(topics_df)
        t1 = time.perf_counter()

        flags = self._gating_flags(run_h)
        qids_low = set(flags[flags].index.astype(str))
        self._q = len(flags)

        if qids_low:
            need_low_df = topics_df[topics_df["qid"].astype(str).isin(qids_low)]
            run_l = self.retr_low.transform(need_low_df)
        else:
            run_l = None
        t2 = time.perf_counter()

        merged = run_h if run_l is None or len(run_l) == 0 else pd.concat([run_h, run_l], ignore_index=True)
        merged.sort_values(["qid","score"], ascending=[True, False], inplace=True)
        merged.drop_duplicates(subset=["qid","docno"], keep="first", inplace=True)
        if self.final_topk is not None:
            merged = merged.groupby("qid", group_keys=False).head(self.final_topk)
        merged["rank"] = merged.groupby("qid")["score"].rank(ascending=False, method="first").astype(np.int32)
        t3 = time.perf_counter()

        if self.log_stats:
            self._low_acts = len(qids_low)
            self._t_high   = (t1 - t0)
            self._t_low    = (t2 - t1)
            self._t_merge  = (t3 - t2)

        return merged[["qid","docno","rank","score"]]

    def get_stats(self):
        if self._q == 0:
            return dict(queries=0, low_activations=0, activation_rate=0.0,
                        **{f"avg_high_time_{self.time_unit}":0.0,
                           f"avg_low_time_{self.time_unit}":0.0,
                           f"avg_merge_time_{self.time_unit}":0.0})
        return dict(
            queries=self._q,
            low_activations=self._low_acts,
            activation_rate=(self._low_acts / self._q),
            **{f"avg_high_time_{self.time_unit}": self._s(self._t_high / self._q),
               f"avg_low_time_{self.time_unit}":  self._s(self._t_low  / self._q),
               f"avg_merge_time_{self.time_unit}":self._s(self._t_merge/ self._q)}
        )

# ========= Metriche & timing UNIFICATI =========
EVAL_METRICS = ["ndcg_cut_10", "recip_rank", "map", "P_10"]

def _warmup(retr, queries_df, n=5):
    try:
        _ = retr.transform(queries_df.head(n))
    except Exception:
        pass

def _time_retrieval(name, retr, queries_df):
    _warmup(retr, queries_df)
    t0 = time.perf_counter()
    _ = retr.transform(queries_df)
    t1 = time.perf_counter()
    total = (t1 - t0) * 1000.0
    perq  = total / max(1, len(queries_df))
    log.info(f"[{name}] retrieve_time = {total:8.1f} ms  ({perq:.2f} ms/query)")

def _eval_single(name, retr, queries_df, qrels_df):
    return pt.Experiment(
        [retr],
        queries_df,
        qrels_df,
        eval_metrics=EVAL_METRICS,
        names=[name],
        verbose=False
    )

# ========= Esperimenti (coerenti con TAS-B) =========
def run_suite(tag, retrH, retrL, queries_df, qrels_df):
    rows = []

    # baseline
    _time_retrieval(f"{tag} high_only", retrH, queries_df)
    rows.append(_eval_single(f"{tag}:high_only", retrH, queries_df, qrels_df))

    # margin_or_entropy
    two = TwoTier(retrH, retrL, mode="margin_or_entropy",
                  margin=MARGIN_ABS, margin_mode="absolute",
                  tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_or_entropy", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_or_entropy", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_or_entropy stats: {two.get_stats()}")

    # margin_and_entropy
    two = TwoTier(retrH, retrL, mode="margin_and_entropy",
                  margin=MARGIN_ABS, margin_mode="absolute",
                  tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_and_entropy", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_and_entropy", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_and_entropy stats: {two.get_stats()}")

    # entropy 0.9
    two = TwoTier(retrH, retrL, mode="entropy",
                  margin=MARGIN_ABS, margin_mode="absolute",
                  tau=ENTROPY_TAU, entropy_threshold=0.9,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.entropy0.9", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.entropy0.9", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.entropy0.9 stats: {two.get_stats()}")

    # margin abs
    two = TwoTier(retrH, retrL, mode="margin", margin_mode="absolute",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_abs", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_abs", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_abs stats: {two.get_stats()}")

    # margin rel
    two = TwoTier(retrH, retrL, mode="margin", margin_mode="relative",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_rel", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_rel", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_rel stats: {two.get_stats()}")

    return pd.concat(rows, ignore_index=True)

# ========= RUN =========
all_results = []
if RUN_DENSE:
    log.info("===== BACKEND: ColBERT (FAISS end_to_end), text queries =====")
    retrH, retrL = get_colbert_searchers(topk=TOPK)
    # ColBERT lavora su query testuali
    queries_df = qdf[["qid","query"]]
    res_dense = run_suite("ColBERT", retrH, retrL, queries_df, qrels_cov)
    all_results.append(res_dense)

if RUN_NP:
    log.info("===== BACKEND: ColBERT (exhaustive) =====")
    pass

if all_results:
    final = pd.concat(all_results, ignore_index=True)
    final.to_parquet(run_dir / "final.parquet", index=False)
    final.to_csv(run_dir / "final.csv", index=False)
    log.info("===== RIEPILOGO COMPLESSIVO salvato in runs =====")
