# src/tasb_two_tier.py
# ========= Imports base =========
import os, time, math, glob, shutil, random
from pathlib import Path
import numpy as np, pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pyterrier as pt
import ir_datasets
import pyterrier_dr as dr
import faiss
from pyterrier_dr.flex.core import FlexIndex

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger,
    by_query_topk, choose_nlist, adaptive_nprobe
)

# ========= Config =========
ensure_pt()

PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/tasb_two_tier.yaml")

INDEX_ROOT = resolve_index_root(PATHS)
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT = DATA["split"]
SAMPLE_MODE   = DATA["sample_mode"]           # "sample" | "all_queries"
SAMPLE_TOPICS = DATA["sample_topics"]
NEG_PER_REL   = DATA["neg_per_rel"]
MAX_DOCS      = DATA["max_docs"]
HIGH_SHARE    = DATA["high_share"]
TOPK          = DATA["topk"]
DATASET_ID    = DATA["dataset_id_quality"]

RUN_IVFPQ = HP["run_ivfpq"]; RUN_HNSW = HP["run_hnsw"]; RUN_NP = HP["run_np"]
IVF_NLIST = HP["ivf_nlist"]; PQ_M = HP["pq_m"]; PQ_NBITS = HP["pq_nbits"]
NPROBE = HP["nprobe"]; NPROBE_REL = HP["nprobe_rel"]
HNSW_EF_SEARCH = HP["hnsw_ef_search"]
MARGIN_ABS = HP["margin_abs"]; ENTROPY_TAU = HP["entropy_tau"]
ENTROPY_FRAC = HP["entropy_frac"]; ENTROPY_TOPN = HP["entropy_topn"]

run_dir = stamp_run_dir(PATHS["runs_dir"], "tasb_two_tier")
log = get_logger(run_dir)

# ========= Dataset & sample =========
ds = pt.get_dataset(SPLIT)
topics_all = ds.get_topics().astype({"qid":"str"})
qrels_all  = ds.get_qrels().astype({"qid":"str", "docno":"str"})

if SAMPLE_MODE == "sample":
    topics = topics_all.sample(n=min(SAMPLE_TOPICS, len(topics_all)), random_state=42).sort_values("qid")
    qrels  = qrels_all[qrels_all["qid"].isin(topics["qid"])].copy()
elif SAMPLE_MODE == "all_queries":
    topics = topics_all.sort_values("qid")
    qrels  = qrels_all.copy()
else:
    raise ValueError("SAMPLE_MODE deve essere 'sample' o 'all_queries'.")

rel_docnos = set(qrels.loc[qrels["label"] > 0, "docno"])

def build_corpus_with_rel(dataset, must_have: set, max_docs: int | None, neg_per_rel: int = 3, seed: int = 42):
    random.seed(seed)
    rows_rel, rows_neg = [], []
    for r in dataset.get_corpus_iter():
        pid  = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
        if pid in must_have:
            text = (r.get("text") or "").strip()
            if text:
                rows_rel.append({"docno": pid, "text": text})
        if len(rows_rel) == len(must_have):
            break
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

# ========= QualT5 & split =========
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
                    docno = str(getattr(rec, "docno"))
                    qual  = float(getattr(rec, "quality"))
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
high = passages_q.loc[passages_q["quality"] >= cut, ["docno","text"]].copy()
low  = passages_q.loc[passages_q["quality"] <  cut, ["docno","text"]].copy()
log.info(f"[SPLIT] High={len(high)}  Low={len(low)}  matched={len(passages_q)}/{len(passages)}")

indexed_docnos = set(pd.concat([high["docno"], low["docno"]]))
qrels_cov = qrels[qrels["docno"].isin(indexed_docnos)].copy()
topics_cov = topics[topics["qid"].isin(set(qrels_cov["qid"]))].copy()
log.info(f"[coverage] queries con almeno un rilevante nel subset: {len(topics_cov)} / {len(topics)}")

# ========= Encoders TAS-B =========
tasb = dr.TasB()
qenc = tasb.query_encoder()
denc = tasb.doc_encoder()

t0 = time.perf_counter()
qvec_df = qenc.transform(topics_cov[["qid","query"]].copy())
t1 = time.perf_counter()
log.info(f"[ENC] query encoding time = {(t1-t0)*1000:.1f} ms ({(t1-t0)*1000/len(qvec_df):.2f} ms/query)")

# ========= IVFPQ =========
IDX_HIGH_IVFPQ = "msmarco_high_70pct_tasb_ivfpq"
IDX_LOW_IVFPQ  = "msmarco_low_30pct_tasb_ivfpq"

def _encode_docs_numpy(df_docs: pd.DataFrame):
    out = denc.transform(df_docs[["docno","text"]].copy())
    vecs = np.vstack(out["doc_vec"].values).astype(np.float32, copy=False)
    ids  = out["docno"].astype(str).to_numpy()
    return ids, vecs

def _build_ivfpq_ip_index(embeddings: np.ndarray, nlist=None, m=32, nbits=8, train_cap=100_000, seed=42):
    embs = embeddings.astype(np.float32, copy=False)
    dim  = embs.shape[1]; n = embs.shape[0]
    nlist = int(nlist or choose_nlist(n))
    nlist = min(nlist, max(1, n//2))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(seed)
    train_sz = min(train_cap, n)
    train_idx = rng.choice(n, size=train_sz, replace=False)
    index.train(embs[train_idx]); index.add(embs)
    index.nprobe = adaptive_nprobe(NPROBE, TOPK, NPROBE_REL)
    return index

def _persist_faiss_index(index: faiss.Index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def _load_faiss_index(path: Path):
    idx = faiss.read_index(str(path))
    if isinstance(idx, faiss.IndexIVF):
        idx.nprobe = adaptive_nprobe(NPROBE, TOPK, NPROBE_REL)
    return idx

def ensure_ivfpq_index(df: pd.DataFrame, index_name: str):
    d = INDEX_ROOT / index_name
    faiss_path = d / "faiss.index"
    ids_path   = d / "ids.npy"
    emb_path   = d / "embeddings.npy"
    d.mkdir(parents=True, exist_ok=True)
    if faiss_path.exists() and ids_path.exists():
        log.info(f"[IVFPQ] Trovato '{index_name}'.")
        return
    log.info(f"[IVFPQ] Indicizzo '{index_name}' ({len(df)} docs)...")
    ids, embs = _encode_docs_numpy(df)
    index = _build_ivfpq_ip_index(embs, nlist=IVF_NLIST, m=PQ_M, nbits=PQ_NBITS)
    _persist_faiss_index(index, faiss_path)
    np.save(ids_path, np.asarray(ids, dtype='U64'))
    np.save(emb_path, embs.astype(np.float32, copy=False))
    log.info(f"[IVFPQ] Salvati index, ids, embeddings in {d}")

class FaissIVFPQSearcher(pt.Transformer):
    def __init__(self, index_dir: Path, topk=50, nprobe=None):
        super().__init__()
        self.topk = int(topk)
        self.index = _load_faiss_index(index_dir / "faiss.index")
        if isinstance(self.index, faiss.IndexIVF) and nprobe is not None:
            self.index.nprobe = int(nprobe)
        self.ids = np.load(index_dir / "ids.npy", allow_pickle=False)

    def transform(self, df_queries: pd.DataFrame) -> pd.DataFrame:
        qids  = df_queries["qid"].astype(str).to_numpy()
        qvecs = np.vstack(df_queries["query_vec"].values).astype(np.float32, copy=False)
        scores, idxs = self.index.search(qvecs, self.topk)
        out_qid, out_doc, out_rank, out_score = [], [], [], []
        for qi in range(len(qids)):
            valid = idxs[qi] >= 0
            if not np.any(valid):
                continue
            didxs = idxs[qi][valid]; scs = scores[qi][valid]
            k = len(didxs)
            out_qid.extend([qids[qi]]*k)
            out_doc.extend(self.ids[didxs].astype(str))
            out_rank.extend(np.arange(1, k+1, dtype=np.int32))
            out_score.extend(scs.astype(float))
        return pd.DataFrame({"qid": out_qid, "docno": out_doc, "rank": out_rank, "score": out_score})

def get_ivfpq_searchers(topk=TOPK):
    ensure_ivfpq_index(high, IDX_HIGH_IVFPQ)
    ensure_ivfpq_index(low,  IDX_LOW_IVFPQ)
    retrH = FaissIVFPQSearcher(INDEX_ROOT/IDX_HIGH_IVFPQ, topk=topk)
    retrL = FaissIVFPQSearcher(INDEX_ROOT/IDX_LOW_IVFPQ,  topk=topk)
    return retrH, retrL

# ========= HNSW =========
IDX_HIGH_HNSW = "msmarco_high_70pct_tasb_hnsw"
IDX_LOW_HNSW  = "msmarco_low_30pct_tasb_hnsw"

def ensure_hnsw_flexindex(df: pd.DataFrame, name: str):
    idx_dir = INDEX_ROOT / f"{name}.flex"
    if idx_dir.exists():
        log.info(f"[HNSW] Trovato '{name}.flex'")
        return
    log.info(f"[HNSW] Indicizzo '{name}' ({len(df)} docs)...")
    idx = FlexIndex(str(idx_dir), verbose=True)
    (denc >> idx).index(df[["docno","text"]].to_dict(orient="records"))

def _make_hnsw_retriever(idx: FlexIndex, topk: int, ef: int, topk_op):
    fn = idx.faiss_hnsw_retriever
    try:    return fn(k=topk, efSearch=ef)
    except TypeError: pass
    try:    return fn(k=topk, ef_search=ef)
    except TypeError: pass
    try:    return fn(k=topk, ef=ef)
    except TypeError: pass
    try:    return fn(k=topk)
    except TypeError: pass
    try:    return fn() >> topk_op
    except TypeError:
        raise RuntimeError("faiss_hnsw_retriever non accetta k/efSearch/ef_search/ef nella tua build.")

def get_hnsw_searchers(topk=TOPK, efSearch=HNSW_EF_SEARCH):
    ensure_hnsw_flexindex(high, IDX_HIGH_HNSW)
    ensure_hnsw_flexindex(low,  IDX_LOW_HNSW)
    idxH = FlexIndex(str(INDEX_ROOT/f"{IDX_HIGH_HNSW}.flex"), verbose=False)
    idxL = FlexIndex(str(INDEX_ROOT/f"{IDX_LOW_HNSW}.flex"),  verbose=False)
    rH = _make_hnsw_retriever(idxH, topk, efSearch, by_query_topk(topk))
    rL = _make_hnsw_retriever(idxL, topk, efSearch, by_query_topk(topk))
    return rH, rL

# ========= NP (exhaustive) =========
IDX_HIGH_NP = "msmarco_high_70pct_tasb_np"
IDX_LOW_NP  = "msmarco_low_30pct_tasb_np"

def ensure_np_flexindex(df: pd.DataFrame, name: str):
    idx_dir = INDEX_ROOT / f"{name}.flex"
    if idx_dir.exists():
        log.info(f"[NP] Trovato '{name}.flex'")
        return
    log.info(f"[NP] Indicizzo '{name}' ({len(df)} docs)...")
    idx = FlexIndex(str(idx_dir), verbose=True)
    (denc >> idx).index(df[["docno","text"]].to_dict(orient="records"))

def get_np_searchers(topk=TOPK):
    ensure_np_flexindex(high, IDX_HIGH_NP)
    ensure_np_flexindex(low,  IDX_LOW_NP)
    idxH = FlexIndex(str(INDEX_ROOT/f"{IDX_HIGH_NP}.flex"), verbose=False)
    idxL = FlexIndex(str(INDEX_ROOT/f"{IDX_LOW_NP}.flex"),  verbose=False)
    try:
        rH = idxH.np_retriever(k=topk); rL = idxL.np_retriever(k=topk)
    except TypeError:
        rH = idxH.np_retriever() >> by_query_topk(topk)
        rL = idxL.np_retriever() >> by_query_topk(topk)
    return rH, rL

# ========= TwoTier gating =========
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
                        avg_high_time_ms=0.0, avg_low_time_ms=0.0, avg_merge_time_ms=0.0)
        return dict(
            queries=self._q,
            low_activations=self._low_acts,
            activation_rate=(self._low_acts / self._q),
            avg_high_time_ms=self._s(self._t_high / self._q),
            avg_low_time_ms=self._s(self._t_low  / self._q),
            avg_merge_time_ms=self._s(self._t_merge/ self._q),
        )

# ========= Metriche & timing UNIFICATI =========
EVAL_METRICS = ["ndcg_cut_10","recip_rank","map","P_10"]

def _warmup(retr, queries_df, n=5):
    try:
        _ = retr.transform(queries_df.head(n))
    except Exception:
        pass

def _time_retrieval(name, retr, queries_df):
    _warmup(retr, queries_df)
    t0 = time.perf_counter(); _ = retr.transform(queries_df); t1 = time.perf_counter()
    total = (t1 - t0)*1000.0; perq = total / max(1, len(queries_df))
    log.info(f"[{name}] retrieve_time = {total:8.1f} ms  ({perq:.2f} ms/query)")

def _eval_single(name, retr, queries_df, qrels_df):
    return pt.Experiment([retr], queries_df, qrels_df,
                         eval_metrics=EVAL_METRICS, names=[name], verbose=False)

def run_suite(tag, retrH, retrL, queries_df, qrels_df):
    rows = []
    _time_retrieval(f"{tag} high_only", retrH, queries_df)
    rows.append(_eval_single(f"{tag}:high_only", retrH, queries_df, qrels_df))

    two = TwoTier(retrH, retrL, mode="margin_or_entropy",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_or_entropy", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_or_entropy", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_or_entropy stats: {two.get_stats()}")

    two = TwoTier(retrH, retrL, mode="margin_and_entropy",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_and_entropy", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_and_entropy", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_and_entropy stats: {two.get_stats()}")

    two = TwoTier(retrH, retrL, mode="entropy",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=0.9,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.entropy0.9", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.entropy0.9", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.entropy0.9 stats: {two.get_stats()}")

    two = TwoTier(retrH, retrL, mode="margin", margin_mode="absolute",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_abs", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_abs", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_abs stats: {two.get_stats()}")

    two = TwoTier(retrH, retrL, mode="margin", margin_mode="relative",
                  margin=MARGIN_ABS, tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                  topn_entropy=ENTROPY_TOPN, final_topk=TOPK, log_stats=True)
    _time_retrieval(f"{tag} two_tier.margin_rel", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_rel", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_rel stats: {two.get_stats()}")

    return pd.concat(rows, ignore_index=True)

# ========= RUN =========
all_results = []
queries_df = qvec_df[["qid","query_vec"]].copy()

if RUN_IVFPQ:
    log.info("===== BACKEND: IVFPQ (pre-encoded) =====")
    retrH, retrL = get_ivfpq_searchers(topk=TOPK)
    res_ivfpq = run_suite("IVFPQ", retrH, retrL, queries_df, qrels_cov)
    all_results.append(res_ivfpq)

if RUN_HNSW:
    log.info("===== BACKEND: HNSW (pre-encoded) =====")
    retrH, retrL = get_hnsw_searchers(topk=TOPK, efSearch=HNSW_EF_SEARCH)
    res_hnsw = run_suite("HNSW", retrH, retrL, queries_df, qrels_cov)
    all_results.append(res_hnsw)

if RUN_NP:
    log.info("===== BACKEND: NP (exhaustive, pre-encoded) =====")
    rH, rL = get_np_searchers(topk=TOPK)
    ident = pt.Transformer()
    retrH = ident >> rH; retrL = ident >> rL
    res_np = run_suite("NP", retrH, retrL, queries_df, qrels_cov)
    all_results.append(res_np)

if all_results:
    final = pd.concat(all_results, ignore_index=True)
    final.to_parquet(run_dir / "final.parquet", index=False)
    final.to_csv(run_dir / "final.csv", index=False)
    log.info("===== RIEPILOGO COMPLESSIVO salvato in runs =====")
