from pathlib import Path
import pandas as pd
from .config import INDEX_ROOT, CHECKPOINT
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory

def ensure_colbert_index(df: pd.DataFrame, index_name: str, gpu: bool=False):
    idx_dir = Path(INDEX_ROOT) / index_name
    exists = idx_dir.exists() and any(idx_dir.rglob("*.faiss"))
    if exists:
        print(f"[ColBERT] Found index '{index_name}'."); return
    print(f"[ColBERT] Indexing '{index_name}' ({len(df)} docs)...")
    try:
        indexer = ColBERTIndexer(CHECKPOINT, INDEX_ROOT, index_name,
                                 ids=True, chunksize=64, gpu=gpu, doc_maxlen=64)
    except TypeError:
        indexer = ColBERTIndexer(CHECKPOINT, INDEX_ROOT, index_name,
                                 ids=True, chunksize=64, gpu=gpu)
    indexer.index(df[["docno","text"]].to_dict("records"))

def detect_faiss_partitions(index_name: str) -> int:
    idx_dir = Path(INDEX_ROOT) / index_name
    files = list(idx_dir.glob("ivfpq.*.faiss"))
    if not files: return 100
    try:
        return int(files[0].name.split(".")[1])
    except Exception:
        return 100

def build_factories(idx_high, idx_low, nprobe, faiss_depth, gpu=False):
    part_high = detect_faiss_partitions(idx_high)
    part_low  = detect_faiss_partitions(idx_low)
    pyt_high = ColBERTFactory(CHECKPOINT, INDEX_ROOT, idx_high,
                              faiss_partitions=part_high, memtype="mem", faisstype="mem", gpu=gpu)
    pyt_low  = ColBERTFactory(CHECKPOINT, INDEX_ROOT, idx_low,
                              faiss_partitions=part_low, memtype="mem", faisstype="mem", gpu=gpu)
    for f in (pyt_high, pyt_low):
        f.args.nprobe = nprobe
        f.args.faiss_depth = faiss_depth
    return pyt_high.end_to_end(), pyt_low.end_to_end()
