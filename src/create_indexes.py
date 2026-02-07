# src/tasb_two_tier_index.py
# ===== Unified two-tier indexer (IVF-Flat | IVF-PQ), TAS-B enc, GPU optional =====
import os, gc, json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

import pyterrier as pt
import pyterrier_dr as dr
import faiss

from .common import (
    load_cfg, ensure_pt, resolve_index_root, stamp_run_dir, get_logger,
    choose_nlist, sanitize_split_tag,
    build_corpus_with_rel, build_corpus_all, load_quality_for_sample
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ========= Config =========
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")
HP    = load_cfg("configs/tasb_two_tier.yaml")

INDEX_ROOT = resolve_index_root(PATHS); INDEX_ROOT.mkdir(parents=True, exist_ok=True)
CACHE = Path(PATHS.get("cache_dir", "cache")); CACHE.mkdir(parents=True, exist_ok=True)

SPLIT        = DATA["split"]
SAMPLE_MODE  = DATA["sample_mode"]           # "sample" | "all_queries"
SAMPLE_TOPICS= DATA["sample_topics"]
NEG_PER_REL  = DATA["neg_per_rel"]
MAX_DOCS     = DATA["max_docs"]
HIGH_SHARE   = float(DATA["high_share"])
DATASET_ID   = DATA["dataset_id_quality"]

SPLIT_TAG = f"{sanitize_split_tag(SPLIT)}_hs{HIGH_SHARE}"
DOCVEC_CACHE = CACHE / "docvecs"; DOCVEC_CACHE.mkdir(parents=True, exist_ok=True)

# ---- Hyperparams / build ----
IVF_NLIST   = int(HP.get("ivf_nlist", 32768))
NPROBE      = int(HP.get("nprobe", 64))
TRAIN_CAP   = int(HP.get("train_cap", 200_000))
BUILD_CHUNK = int(HP.get("build_chunk", 100_000))
SHARDS      = int(HP.get("shards", 1))       # opzionale, 1 = no sharding

# PQ params (usati solo se ivf_quantizer == "pq")
PQ_M        = int(HP.get("pq_m", 16))
PQ_NBITS    = int(HP.get("pq_nbits", 8))

MODE        = HP.get("ivf_quantizer", "flat").lower()  # "flat" | "pq"
# === NUOVO FLAG ===
BUILD_MODE  = HP.get("build_mode", "split").lower()   # "split" | "complete"

# ---- Flags per non toccare YAML esistente ----
RUN_IVFFLAT = MODE == "flat"
RUN_IVFPQ   = MODE == "pq"
# === NUOVI FLAG ===
RUN_SPLIT   = BUILD_MODE == "split"
RUN_COMPLETE= BUILD_MODE == "complete"


run_dir = stamp_run_dir(PATHS["runs_dir"], f"build_ivf_{MODE}_{BUILD_MODE}_{SPLIT_TAG}")
log = get_logger(run_dir)

# ========= GPU helpers (FAISS) =========
def _env_bool(name: str, default: str = "1") -> bool:
    try:
        return bool(int(os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default).lower() in ("1","true","yes","y","on")

USE_GPU       = _env_bool("FAISS_GPU", "1") and faiss.get_num_gpus() > 0
_FAISS_DEVICE = int(os.getenv("FAISS_DEVICE", "0"))
_FAISS_FP16   = _env_bool("FAISS_FP16", "1")      # utile per IVFPQ/Flat add su GPU
_FAISS_SHARD  = _env_bool("FAISS_SHARD", "0")
_FAISS_TEMP_MB= int(os.getenv("FAISS_TEMP_MEM_MB", "2048"))

def _make_gpu_res():
    res = faiss.StandardGpuResources()
    try:
        res.setTempMemory(_FAISS_TEMP_MB * 1024 * 1024)
    except Exception:
        pass
    return res

def _cpu_to_gpu(index_cpu):
    if not USE_GPU:
        return index_cpu, None
    co = faiss.GpuClonerOptions()
    co.useFloat16 = _FAISS_FP16
    co.shard = _FAISS_SHARD
    res = _make_gpu_res()
    index_gpu = faiss.index_cpu_to_gpu(res, _FAISS_DEVICE, index_cpu, co)
    return index_gpu, res

def _gpu_to_cpu(index_gpu):
    try:
        return faiss.index_gpu_to_cpu(index_gpu)
    except Exception:
        return index_gpu

# ========= Dataset & sampling =========
ds = pt.get_dataset(SPLIT)
if SAMPLE_MODE == "all_queries":
    passages = build_corpus_all(ds, max_docs=None)
else:
    topics_all = ds.get_topics().astype({"qid": "str"})
    qrels_all  = ds.get_qrels().astype({"qid": "str", "docno": "str"})
    topics     = topics_all.sample(n=min(SAMPLE_TOPICS, len(topics_all)), random_state=42)
    qrels      = qrels_all[qrels_all["qid"].isin(topics["qid"])].copy()
    rel_docnos = set(qrels.loc[qrels["label"] > 0, "docno"])
    passages   = build_corpus_with_rel(ds, rel_docnos, max_docs=MAX_DOCS, neg_per_rel=NEG_PER_REL, seed=42)

passages.drop_duplicates("docno", inplace=True)
log.info(f"[subset] total_subset={len(passages)}")

# ========= Quality split cache =========
high_path = CACHE / f"high_{SPLIT_TAG}.parquet"
low_path  = CACHE / f"low_{SPLIT_TAG}.parquet"
if high_path.exists() and low_path.exists():
    high = pd.read_parquet(high_path)
    low  = pd.read_parquet(low_path)
    log.info(f"[SPLIT cache] High={len(high)} Low={len(low)} (cache hit)")
else:
    qual = load_quality_for_sample(passages["docno"], DATASET_ID, log)
    pq   = passages.merge(qual, on="docno", how="inner")
    cut  = 1.0 - HIGH_SHARE
    high = pq.loc[pq["quality"] >= cut, ["docno", "text"]].copy()
    low  = pq.loc[pq["quality"]  < cut, ["docno", "text"]].copy()
    high.to_parquet(high_path, index=False)
    low.to_parquet(low_path,  index=False)
    log.info(f"[SPLIT] High={len(high)} Low={len(low)}")
    del passages, pq, qual; gc.collect()

# ========= Encoders TAS-B (solo documenti) =========
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "true")
ENC_DEVICE = os.getenv("ENC_DEVICE", "cuda")
tasb = dr.TasB(device=ENC_DEVICE)
denc = tasb.doc_encoder(batch_size=int(os.getenv("DOCENC_BS", "32")))

# ========= Docvec memmap cache =========
def _sha256_iter_strs(strings_iter):
    h = hashlib.sha256()
    for s in strings_iter:
        if not isinstance(s, str): s = str(s)
        h.update(s.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()

def _docvec_cache_paths(cache_key: str):
    mm   = DOCVEC_CACHE / f"{cache_key}.f32.mmap"
    meta = DOCVEC_CACHE / f"{cache_key}.meta.json"
    return mm, meta

def _write_json(path: Path, obj: dict): path.write_text(json.dumps(obj, indent=2, sort_keys=True))
def _read_json(path: Path) -> dict: return json.loads(path.read_text())

def _ensure_docvec_memmap(df_docs: pd.DataFrame, cache_key: str):
    docs = df_docs.reset_index(drop=True)
    N = len(docs)
    mm_path, meta_path = _docvec_cache_paths(cache_key)

    if mm_path.exists() and meta_path.exists():
        try:
            meta = _read_json(meta_path)
            docno_hash_curr = _sha256_iter_strs(docs["docno"].astype(str).to_numpy())
            if meta.get("N")==N and meta.get("docno_hash")==docno_hash_curr and meta.get("dim", -1) > 0:
                log.info(f"[DOCVEC cache] hit: {cache_key}")
                return mm_path, meta
        except Exception:
            pass

    # build cache
    log.info(f"[DOCVEC cache] build: {cache_key} (N={N:,})")
    probe = denc.transform(docs.iloc[:4][["docno","text"]].copy())
    dim   = int(np.vstack(probe["doc_vec"].values).shape[1])

    bs = int(os.getenv("DOCENC_BS", 512))
    embs_mm = open_memmap(str(mm_path), mode='w+', dtype='float32', shape=(N, dim))
    wptr = 0
    from tqdm import tqdm
    for start in tqdm(range(0, N, bs), desc=f"[DOCVEC encode {cache_key}]"):
        end = min(start + bs, N)
        out  = denc.transform(docs.iloc[start:end][["docno","text"]].copy())
        vecs = np.vstack(out["doc_vec"].values).astype(np.float32, copy=False)
        embs_mm[wptr:wptr+len(vecs), :] = vecs
        wptr += len(vecs)
        del out, vecs; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    del embs_mm
    meta = {"N": N, "dim": dim, "docno_hash": _sha256_iter_strs(docs["docno"].astype(str).to_numpy())}
    _write_json(meta_path, meta)
    return mm_path, meta

# ========= FAISS factories =========
def make_ivf_index(dim: int, nlist: int, mode: str, pq_m: int, pq_nbits: int):
    mode = mode.lower()
    q = faiss.IndexFlatIP(dim)  # Inner Product; useremo L2-norm → IP ~ coseno
    if mode == "flat":
        return faiss.IndexIVFFlat(q, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
    if mode == "pq":
        # NB: firma corretta con metric
        return faiss.IndexIVFPQ(q, dim, int(nlist), int(pq_m), int(pq_nbits), faiss.METRIC_INNER_PRODUCT)
    raise ValueError(f"ivf_quantizer non supportato: {mode}")

def train_ivf(index_cpu, train_vecs: np.ndarray):
    # normalizza per usare IP come coseno
    #l2_normalize_inplace(train_vecs)
    if USE_GPU:
        idx_gpu, res = _cpu_to_gpu(index_cpu)
        idx_gpu.train(train_vecs)
        index_cpu = _gpu_to_cpu(idx_gpu); del idx_gpu, res
    else:
        index_cpu.train(train_vecs)
    return index_cpu

def add_in_chunks(index_cpu_or_gpu, vecs_mm, build_chunk: int, nprobe: int):
    # set nprobe dove supportato (IVF-Flat / IVFPQ)
    try: index_cpu_or_gpu.nprobe = nprobe
    except Exception: pass
    N, D = vecs_mm.shape
    buf = np.empty((build_chunk, D), dtype=np.float32, order="C")
    added = 0
    # add con buffer riutilizzabile per contenere i picchi RAM
    from tqdm import tqdm
    for s in tqdm(range(0, N, build_chunk), desc="[IVF add]"):
        e = min(N, s + build_chunk)
        sl = vecs_mm[s:e]
        buf[:e-s] = sl
        #l2 = np.linalg.norm(buf[:e-s], axis=1, keepdims=True); l2[l2==0] = 1.0
        #buf[:e-s] /= l2
        index_cpu_or_gpu.add(buf[:e-s])
        added += (e - s)
        if added == (e - s) or added % (build_chunk*10) == 0:
            gc.collect()
    return added

# ========= Build (singolo df → indice) =========
def build_one_index(df_docs: pd.DataFrame, index_name: str):
    outdir    = INDEX_ROOT / index_name
    faisspath = outdir / "faiss.index"
    idspath   = outdir / "ids.npy"
    outdir.mkdir(parents=True, exist_ok=True)

    if faisspath.exists() and idspath.exists():
        log.info(f"[SKIP] {index_name} già presente.")
        return

    docs = df_docs.reset_index(drop=True)[["docno","text"]].copy()
    ids_np = docs["docno"].astype(str).to_numpy()
    np.save(idspath, ids_np)

    # memmap vettori
    cache_key = f"{index_name}_{SPLIT_TAG}"
    mm_path, meta = _ensure_docvec_memmap(docs, cache_key)
    N, dim = int(meta["N"]), int(meta["dim"])
    vecs_mm = open_memmap(str(mm_path), mode='r', dtype='float32', shape=(N, dim))

    # sample per training
    take = min(TRAIN_CAP, N)
    rng = np.random.default_rng(42)
    tr_idx = rng.choice(N, size=take, replace=False)
    train_vecs = np.asarray(vecs_mm[tr_idx], dtype=np.float32, order="C")

    # costruisci indice
    nlist_in   = IVF_NLIST or choose_nlist(N)
    nlist_used = max(256, min(nlist_in, max(256, take // 40), max(256, N // 2)))
    index_cpu  = make_ivf_index(dim, nlist_used, MODE, PQ_M, PQ_NBITS)
    
    log.info(f"[BUILD] {index_name}  N={N:,} dim={dim} nlist={nlist_used} mode={MODE}")
    # train
    index_cpu  = train_ivf(index_cpu, train_vecs)

    # add su GPU se disponibile
    if USE_GPU:
        idx_gpu, res = _cpu_to_gpu(index_cpu)
        added = add_in_chunks(idx_gpu, vecs_mm, BUILD_CHUNK, NPROBE)
        index_cpu = _gpu_to_cpu(idx_gpu); del idx_gpu, res
    else:
        added = add_in_chunks(index_cpu, vecs_mm, BUILD_CHUNK, NPROBE)

    # salva
    faiss.write_index(index_cpu, str(faisspath))
    meta_out = {
        "mode": MODE, "nlist": int(nlist_used), "nprobe_default": int(NPROBE),
        "pq_m": int(PQ_M), "pq_nbits": int(PQ_NBITS) if MODE=="pq" else None,
        "train_cap": int(TRAIN_CAP), "build_chunk": int(BUILD_CHUNK),
        "count": int(N), "dim": int(dim)
    }
    json.dump(meta_out, open(outdir/"meta.json", "w"), indent=2)
    del index_cpu, vecs_mm; gc.collect()
    log.info(f"[DONE] {index_name}  (added={added:,}, dim={dim})")

# ========= Sharding opzionale (riduce picchi RAM) =========
def _slice_shard(df: pd.DataFrame, shards: int, shard_id: int) -> pd.DataFrame:
    if shards <= 1: return df.reset_index(drop=True)
    return df.iloc[shard_id::shards].reset_index(drop=True)

# ========= MAIN (MODIFICATO) =========
def main():
    TAG = DATA.get("dataset_tag", "msmarco")

    if RUN_SPLIT:
        log.info(f"Building in SPLIT mode (high/low) using MODE={MODE}")
        PCT = int(round(100 * HIGH_SHARE)); COMP = 100 - PCT
        base_high = f"{TAG}_high_{PCT}pct_tasb_ivf_{MODE}"
        base_low  = f"{TAG}_low_{COMP}pct_tasb_ivf_{MODE}"

        # HIGH
        if SHARDS <= 1:
            build_one_index(high, base_high)
        else:
            for sid in range(SHARDS):
                build_one_index(_slice_shard(high, SHARDS, sid), f"{base_high}.sh{sid:02d}-of-{SHARDS:02d}")

        # LOW
        if SHARDS <= 1:
            build_one_index(low, base_low)
        else:
            for sid in range(SHARDS):
                build_one_index(_slice_shard(low, SHARDS, sid), f"{base_low}.sh{sid:02d}-of-{SHARDS:02d}")
    
    elif RUN_COMPLETE:
        log.info(f"Building in COMPLETE mode (single index) using MODE={MODE}")
        # Combiniamo high e low per avere tutti i documenti
        all_docs = pd.concat([high, low], ignore_index=True).drop_duplicates(subset=["docno"]).reset_index(drop=True)
        log.info(f"Total documents for complete index: {len(all_docs):,}")
        
        # Nome indice riconoscibile
        base_complete = f"{TAG}_complete_tasb_ivf_{MODE}"
        
        if SHARDS <= 1:
            build_one_index(all_docs, base_complete)
        else:
            for sid in range(SHARDS):
                build_one_index(_slice_shard(all_docs, SHARDS, sid), f"{base_complete}.sh{sid:02d}-of-{SHARDS:02d}")
        
        del all_docs; gc.collect() # Pulisci il dataframe combinato
    
    else:
        log.error(f"BUILD_MODE '{BUILD_MODE}' not recognized. Use 'split' or 'complete'.")

    log.info("===== COSTRUZIONE INDICI COMPLETATA =====")

if __name__ == "__main__":
    main()