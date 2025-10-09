# src/tasb_single_index.py
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

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger,
    choose_nlist, adaptive_nprobe
)

# ========= Init & Config =========
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/tasb_single_index.yaml")

INDEX_ROOT = resolve_index_root(PATHS)
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# dataset & sampling
SPLIT         = DATA["split"]
SAMPLE_MODE   = DATA["sample_mode"]             # "sample" | "all_queries"
SAMPLE_TOPICS = DATA["sample_topics"]
NEG_PER_REL   = DATA["neg_per_rel"]
MAX_DOCS      = DATA["max_docs"]
HIGH_SHARE    = DATA["high_share"]
TOPK          = DATA["topk"]
DATASET_ID    = DATA["dataset_id_quality"]

# FAISS / gating
IVF_NLIST  = HP["ivf_nlist"]
PQ_M       = HP["pq_m"]
PQ_NBITS   = HP["pq_nbits"]
NPROBE     = HP["nprobe"]
NPROBE_REL = HP["nprobe_rel"]

MARGIN_ABS   = HP["margin_abs"]
ENTROPY_TAU  = HP["entropy_tau"]
ENTROPY_FRAC = HP["entropy_frac"]
ENTROPY_TOPN = HP["entropy_topn"]

# naming
IDX_ONE_IVFPQ = HP.get("index_name", "msmarco_all_tasb_ivfpq")
DOCNO_MAP_PATH = INDEX_ROOT / HP.get("docno_map", "one_ivfpq_map_docno.npy")
HIGH_IDS_PATH  = INDEX_ROOT / HP.get("high_ids",  "one_ivfpq_high_ids.npy")

run_dir = stamp_run_dir(PATHS["runs_dir"], "tasb_single_index")
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

# garantisco i doc rilevanti nel sottoinsieme
rel_docnos = set(qrels.loc[qrels["label"] > 0, "docno"])

def build_corpus_with_rel(dataset, must_have: set, max_docs: int | None, neg_per_rel: int = 3, seed: int = 42):
    random.seed(seed)
    rows_rel, rows_neg = [], []
    for r in dataset.get_corpus_iter():
        pid = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
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
            pid = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
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

# ========= QualT5: quality + split HIGH/LOW ordinato =========
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
                    docno = str(getattr(rec, "docno")); qual = float(getattr(rec, "quality"))
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

ordered_all = (passages_q[["docno","text","quality"]]
               .sort_values("quality", ascending=False)
               .reset_index(drop=True))
N = len(ordered_all)
H = int((ordered_all["quality"] >= (1.0 - HIGH_SHARE)).sum())
ordered_all["intid"] = np.arange(N, dtype=np.int64)

# salva mapping e confine HIGH
np.save(DOCNO_MAP_PATH, ordered_all["docno"].astype('U64').to_numpy())
np.save(HIGH_IDS_PATH,  np.arange(H, dtype=np.int64))
log.info(f"[split by quality] H={H}  N={N}  HIGH%={100.0*H/max(1,N):.2f}%")

# copertura per valutazione
indexed_docnos = set(ordered_all["docno"])
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

# ========= Costruzione indice IVFPQ unico =========
def _encode_docs_numpy_ids(df_docs: pd.DataFrame):
    out = denc.transform(df_docs[["docno","text"]].copy())
    vecs = np.vstack(out["doc_vec"].values).astype(np.float32, copy=False)
    ids  = df_docs["intid"].to_numpy(dtype=np.int64, copy=False)
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
    index.train(embs[train_idx])
    return index

def ensure_one_ivfpq_index(df_all: pd.DataFrame, index_name: str):
    d = INDEX_ROOT / index_name
    faiss_path = d / "faiss.index"
    d.mkdir(parents=True, exist_ok=True)
    if faiss_path.exists():
        log.info(f"[IVFPQ] Trovato indice unico '{index_name}'.")
        return
    log.info(f"[IVFPQ] Indicizzo indice unico '{index_name}' ({len(df_all)} docs)...")
    ids, embs = _encode_docs_numpy_ids(df_all)
    index = _build_ivfpq_ip_index(embs, nlist=IVF_NLIST, m=PQ_M, nbits=PQ_NBITS)
    index.nprobe = adaptive_nprobe(NPROBE, TOPK, NPROBE_REL)
    index.add_with_ids(embs, ids)  # ID contigui = intid
    faiss.write_index(index, str(faiss_path))
    log.info(f"[IVFPQ] Salvato indice unico in {faiss_path}")

def _load_faiss_index(path: Path):
    idx = faiss.read_index(str(path))
    if isinstance(idx, faiss.IndexIVF):
        idx.nprobe = adaptive_nprobe(NPROBE, TOPK, NPROBE_REL)
    return idx

# ========= Compat layer & Searcher unico =========
def _apply_bounds_numpy(D: np.ndarray, I: np.ndarray, low: int, high: int):
    mask = (I >= low) & (I < high)
    D = np.where(mask, D, -np.inf)
    order = np.argsort(-D, axis=1)
    row = np.arange(D.shape[0])[:, None]
    return D[row, order], I[row, order]

def _coarse_assign(index_ivf: faiss.Index, qvecs: np.ndarray, nprobe: int):
    qvecs = np.asarray(qvecs, dtype=np.float32, order='C')
    if not isinstance(index_ivf, faiss.IndexIVF):
        raise RuntimeError("Indice FAISS non è IVF; preassigned richiede IndexIVF.")
    quant = index_ivf.quantizer
    Dcent, Icent = quant.search(qvecs, int(nprobe))
    return Icent.astype(np.int64, copy=False), Dcent.astype(np.float32, copy=False)

class FaissIVFPQSearcherOne(pt.Transformer):
    """Indice unico. Baseline = HIGH-only [0, H)."""
    def __init__(self, index_dir: Path, topk=50, nprobe=None):
        super().__init__()
        self.topk = int(topk)
        self.faiss_index = _load_faiss_index(index_dir / "faiss.index")
        if isinstance(self.faiss_index, faiss.IndexIVF) and nprobe is not None:
            self.faiss_index.nprobe = int(nprobe)
        self.docnos = np.load(DOCNO_MAP_PATH, allow_pickle=False)
        if self.docnos.dtype.kind != 'U':
            self.docnos = np.asarray(self.docnos, dtype='U64')

    @staticmethod
    def _as_f32(a): return np.asarray(a, dtype=np.float32, order='C')
    @staticmethod
    def _as_i64(a): return np.asarray(a, dtype=np.int64,   order='C')

    def _search_preassigned_compat(self, qvecs, k, assign, cdis, bounds=None, nprobe=None):
        qvecs  = self._as_f32(qvecs); assign = self._as_i64(assign); cdis = self._as_f32(cdis)
        old_nprobe = None
        try:
            if isinstance(self.faiss_index, faiss.IndexIVF):
                old_nprobe = int(self.faiss_index.nprobe)
                want = int(nprobe) if nprobe is not None else int(assign.shape[1])
                if old_nprobe != want:
                    self.faiss_index.nprobe = want
            D, I = self.faiss_index.search_preassigned(qvecs, int(k), assign, cdis)
            if bounds is not None:
                lo, hi = int(bounds[0]), int(bounds[1])
                D, I = _apply_bounds_numpy(D, I, lo, hi)
            return D, I
        finally:
            if (old_nprobe is not None) and isinstance(self.faiss_index, faiss.IndexIVF):
                self.faiss_index.nprobe = old_nprobe

    def _search_with_selector(self, qvecs: np.ndarray, bounds=None, topk=None, nprobe=None):
        if topk is None:   topk = self.topk
        if nprobe is None: nprobe = self.faiss_index.nprobe if isinstance(self.faiss_index, faiss.IndexIVF) else 1
        qvecs = self._as_f32(qvecs)
        assign, cdis = _coarse_assign(self.faiss_index, qvecs, nprobe)
        old_nprobe = None
        try:
            if isinstance(self.faiss_index, faiss.IndexIVF):
                old_nprobe = int(self.faiss_index.nprobe)
                need_cols = assign.shape[1]
                if old_nprobe != need_cols:
                    self.faiss_index.nprobe = int(need_cols)
            D, I = self.faiss_index.search_preassigned(qvecs, int(topk), self._as_i64(assign), self._as_f32(cdis))
            if bounds is not None:
                D, I = _apply_bounds_numpy(D, I, int(bounds[0]), int(bounds[1]))
            return D, I
        finally:
            if (old_nprobe is not None) and isinstance(self.faiss_index, faiss.IndexIVF):
                self.faiss_index.nprobe = old_nprobe

    def _build_df(self, qids: np.ndarray, D: np.ndarray, I: np.ndarray) -> pd.DataFrame:
        rows = []
        for qi, qid in enumerate(qids):
            valid = I[qi] >= 0
            if not np.any(valid):
                continue
            didxs = I[qi][valid].astype(np.int64, copy=False)
            scs   = D[qi][valid].astype(float, copy=False)
            rows.extend((qid, self.docnos[d], r+1, s) for r, (d, s) in enumerate(zip(didxs, scs)))
        return pd.DataFrame(rows, columns=["qid","docno","rank","score"])

    def transform(self, qvec_df: pd.DataFrame) -> pd.DataFrame:
        qids  = qvec_df["qid"].astype(str).to_numpy()
        qvecs = self._as_f32(np.vstack(qvec_df["query_vec"].values))
        H     = len(np.load(HIGH_IDS_PATH, allow_pickle=False))
        D, I = self._search_with_selector(qvecs, bounds=(0, H), topk=self.topk)
        out = self._build_df(qids, D, I).sort_values(["qid","score"], ascending=[True, False])
        out["rank"] = out.groupby("qid")["score"].rank(ascending=False, method="first").astype(np.int32)
        return out[["qid","docno","rank","score"]]

# ========= Two-tier (coerente con TAS-B/ColBERT) =========
class TwoTierOneIndex(pt.Transformer):
    """High → gating → Low sullo stesso indice unico."""
    def __init__(self, searcher_one: FaissIVFPQSearcherOne,
                 mode="margin_or_entropy", margin=0.05, margin_mode="absolute",
                 tau=1.0, entropy_threshold=0.8, topn_entropy=10, final_topk=None,
                 nprobe_high=None, nprobe_low=None, max_codes_low=None,
                 log_stats=True, time_unit="ms"):
        super().__init__()
        self.S = searcher_one
        self.mode, self.margin, self.margin_mode = mode, float(margin), str(margin_mode).lower()
        self.tau, self.entropy_threshold, self.topn_entropy = float(tau), float(entropy_threshold), int(topn_entropy)
        self.final_topk = final_topk
        self._eps = 1e-12
        self.H = len(np.load(HIGH_IDS_PATH, allow_pickle=False))
        self.N = len(np.load(DOCNO_MAP_PATH, allow_pickle=False))
        self.nprobe_high = nprobe_high
        self.nprobe_low  = nprobe_low
        self.max_codes_low = max_codes_low
        self.log_stats = bool(log_stats)
        self.time_unit = "ms" if time_unit not in ("ms","s") else time_unit
        self._t_high = self._t_low = self._t_merge = 0.0
        self._q = self._low_acts = 0

    def _s(self, seconds):
        return seconds * (1000.0 if self.time_unit == "ms" else 1.0)

    def _need_low(self, s: np.ndarray) -> bool:
        if s.size == 0:
            return True
        # margin
        if s.size < 2:
            need_m = True
        else:
            s_sorted = np.sort(s)[::-1]
            s1, s2 = float(s_sorted[0]), float(s_sorted[1])
            if self.margin_mode == "relative":
                rel_gap = (s1 - s2) / (abs(s1) + self._eps)
                need_m = rel_gap < self.margin
            else:
                need_m = (s1 - s2) < self.margin
        # entropy (come negli altri file)
        k = int(min(self.topn_entropy, s.size))
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
        if self.mode == "margin_or_entropy":  return (need_m or need_e)
        # margin_and_entropy
        return (s.size < 2) or (need_m and need_e)

    def transform(self, qvec_df: pd.DataFrame) -> pd.DataFrame:
        qids  = qvec_df["qid"].astype(str).to_numpy()
        qvecs = np.vstack(qvec_df["query_vec"].values).astype(np.float32, copy=False)

        # HIGH
        t0 = time.perf_counter()
        nprobe_h = self.nprobe_high if self.nprobe_high is not None else \
                   (self.S.faiss_index.nprobe if isinstance(self.S.faiss_index, faiss.IndexIVF) else 1)
        assign_h, cdist_h = _coarse_assign(self.S.faiss_index, qvecs, nprobe=nprobe_h)
        D_h, I_h = self.S._search_preassigned_compat(qvecs, self.S.topk, assign_h, cdist_h,
                                                     bounds=(0, self.H), nprobe=nprobe_h)
        run_h = self.S._build_df(qids, D_h, I_h)
        t1 = time.perf_counter()

        flags = run_h.groupby("qid", sort=False)["score"].apply(
            lambda s: self._need_low(s.to_numpy(np.float32, copy=False))
        )
        qids_low = set(flags[flags].index.astype(str))
        self._q = len(flags)

        # LOW (solo query segnate)
        if qids_low:
            need_low_df = qvec_df[qvec_df["qid"].astype(str).isin(qids_low)]
            # riuso coarse assignment per coerenza
            idx_map = {qid:i for i, qid in enumerate(qvec_df["qid"].astype(str).to_numpy())}
            rows = [idx_map[q] for q in need_low_df["qid"].astype(str).to_numpy()]
            assign_l = assign_h[rows]; cdist_l = cdist_h[rows]
            D_l, I_l = self.S._search_preassigned_compat(
                np.vstack(need_low_df["query_vec"].values).astype(np.float32, copy=False),
                self.S.topk, assign_l, cdist_l, bounds=(self.H, self.N),
                nprobe=nprobe_h
            )
            run_l = self.S._build_df(need_low_df["qid"].astype(str).to_numpy(), D_l, I_l)
            merged = pd.concat([run_h, run_l], ignore_index=True)
        else:
            run_l = None
            merged = run_h
        t2 = time.perf_counter()

        # merge & topk
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

# ========= Factory =========
def get_one_ivfpq_searchers(topk=TOPK):
    ensure_one_ivfpq_index(ordered_all, IDX_ONE_IVFPQ)
    searcher_one = FaissIVFPQSearcherOne(INDEX_ROOT/IDX_ONE_IVFPQ, topk=topk)
    return searcher_one

# ========= Metriche & timing UNIFICATI =========
EVAL_METRICS = ["ndcg_cut_10","recip_rank","map","P_10"]

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
        queries_df[["qid","query_vec"]],
        qrels_df,
        eval_metrics=EVAL_METRICS,
        names=[name],
        verbose=False
    )

def run_suite(tag, searcher_one, queries_df, qrels_df):
    rows = []

    # baseline: HIGH only
    retrH = searcher_one
    _time_retrieval(f"{tag} high_only", retrH, queries_df)
    rows.append(_eval_single(f"{tag}:high_only", retrH, queries_df, qrels_df))

    # margin_or_entropy
    two = TwoTierOneIndex(searcher_one,
                          mode="margin_or_entropy",
                          margin=MARGIN_ABS, margin_mode="absolute",
                          tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                          topn_entropy=ENTROPY_TOPN, final_topk=TOPK,
                          nprobe_high=adaptive_nprobe(NPROBE, TOPK, NPROBE_REL))
    _time_retrieval(f"{tag} two_tier.margin_or_entropy", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_or_entropy", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_or_entropy stats: {two.get_stats()}")

    # margin_and_entropy
    two = TwoTierOneIndex(searcher_one,
                          mode="margin_and_entropy",
                          margin=MARGIN_ABS, margin_mode="absolute",
                          tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                          topn_entropy=ENTROPY_TOPN, final_topk=TOPK,
                          nprobe_high=adaptive_nprobe(NPROBE, TOPK, NPROBE_REL))
    _time_retrieval(f"{tag} two_tier.margin_and_entropy", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_and_entropy", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_and_entropy stats: {two.get_stats()}")

    # entropy 0.9
    two = TwoTierOneIndex(searcher_one,
                          mode="entropy",
                          margin=MARGIN_ABS, margin_mode="absolute",
                          tau=ENTROPY_TAU, entropy_threshold=0.9,
                          topn_entropy=ENTROPY_TOPN, final_topk=TOPK,
                          nprobe_high=adaptive_nprobe(NPROBE, TOPK, NPROBE_REL))
    _time_retrieval(f"{tag} two_tier.entropy0.9", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.entropy0.9", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.entropy0.9 stats: {two.get_stats()}")

    # margin abs
    two = TwoTierOneIndex(searcher_one,
                          mode="margin", margin_mode="absolute",
                          margin=MARGIN_ABS,
                          tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                          topn_entropy=ENTROPY_TOPN, final_topk=TOPK,
                          nprobe_high=adaptive_nprobe(NPROBE, TOPK, NPROBE_REL))
    _time_retrieval(f"{tag} two_tier.margin_abs", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_abs", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_abs stats: {two.get_stats()}")

    # margin rel
    two = TwoTierOneIndex(searcher_one,
                          mode="margin", margin_mode="relative",
                          margin=MARGIN_ABS,
                          tau=ENTROPY_TAU, entropy_threshold=ENTROPY_FRAC,
                          topn_entropy=ENTROPY_TOPN, final_topk=TOPK,
                          nprobe_high=adaptive_nprobe(NPROBE, TOPK, NPROBE_REL))
    _time_retrieval(f"{tag} two_tier.margin_rel", two, queries_df)
    rows.append(_eval_single(f"{tag}:two_tier.margin_rel", two, queries_df, qrels_df))
    log.info(f"[{tag}] two_tier.margin_rel stats: {two.get_stats()}")

    return pd.concat(rows, ignore_index=True)

# ========= RUN =========
log.info("===== BACKEND: IVFPQ (indice unico + preassigned compat) =====")
searcher = get_one_ivfpq_searchers(topk=TOPK)

queries_df = qvec_df[["qid","query_vec"]].copy()
final = run_suite("ONE-IVFPQ", searcher, queries_df, qrels_cov)

final.to_parquet(run_dir / "results.parquet", index=False)
final.to_csv(run_dir / "results.csv", index=False)
log.info("===== RISULTATI salvati in runs =====")
