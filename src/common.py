import os, math, time, random, glob, shutil
from pathlib import Path
import logging, yaml
import numpy as np, pandas as pd
import pyterrier as pt
from typing import Tuple
from pyterrier_quality import QualCache
import faiss

def sanitize_split_tag(s: str) -> str:
    # evita caratteri strani nei nomi file/percorsi
    return s.replace("/", "_").replace(":", "_")


def load_cfg(*paths):
    cfg = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f))
    return cfg

def resolve_index_root(paths_cfg):
    if paths_cfg.get("use_drive"):
        return Path(paths_cfg["drive_root"])
    return Path(paths_cfg["indices_dir"])

def stamp_run_dir(runs_dir: str, pipeline_name: str) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    rd = Path(runs_dir) / pipeline_name / ts
    rd.mkdir(parents=True, exist_ok=True)
    return rd

def get_logger(run_dir: Path):
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers): logger.removeHandler(h)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "logs.txt", encoding="utf-8"); fh.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

def ensure_pt():
    if not pt.started():
        pt.init()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(42)

def by_query_topk(k: int):
    try:
        return pt.apply.by_query(lambda df: df.nlargest(k, "score"))
    except Exception:
        return pt.apply.generic(
            lambda df: (df.sort_values(["qid","score"], ascending=[True, False])
                          .groupby("qid", group_keys=False).head(k)),
            requires=["qid","score"],
            produces=["qid","docno","score"]
        )

def choose_nlist(n_docs: int):
    return int(max(256, min(32768, 4 * math.sqrt(max(1, n_docs)))))


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

def build_corpus_all(dataset, max_docs: int | None = None):
    rows = []
    for i, r in enumerate(dataset.get_corpus_iter()):
        if max_docs is not None and i >= max_docs:
            break
        pid  = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
        text = (r.get("text") or "").strip()
        if not text:
            continue
        rows.append({"docno": pid, "text": text})
    return pd.DataFrame(rows)

def iter_quality_rows(qc):
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

def load_quality_for_sample(sample_docnos: pd.Series, dataset_id: str, log=None):
    wanted = set(sample_docnos.astype(str).tolist())
    def _log(msg): 
        if log: log.info(msg)
    try:
        qc = QualCache.from_url(f"hf:{dataset_id}@quantiles")
        rows = [(d,q) for d,q in iter_quality_rows(qc) if d in wanted]
        df = pd.DataFrame(rows, columns=["docno","quality"])
        if not df.empty:
            _log("[QualT5] @quantiles OK → [0,1] globali.")
            return df
        else:
            _log("[QualT5] @quantiles OK ma nessuna sovrapposizione; Fallback RAW.")
    except Exception as e:
        _log(f"[QualT5] @quantiles fallita ({e}). Fallback RAW).")
        try:
            #for p in glob.glob(os.path.expanduser("~/.pyterrier/**/qt5-tiny.msmarco-passage.cache*"), recursive=True):
            for p in glob.glob(os.path.expanduser("~/.pyterrier/**/qt5-base.msmarco-passage.cache*"), recursive=True):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    qc = QualCache.from_url(f"hf:{dataset_id}")
    rows = [(d,float(q)) for d,q in iter_quality_rows(qc) if d in wanted]
    if not rows:
        raise RuntimeError("QualT5: nessuna sovrapposizione sul sample.")
    df = (pd.DataFrame(rows, columns=["docno","quality_raw"])
            .sort_values(["quality_raw","docno"]).reset_index(drop=True))
    df["quality"] = df["quality_raw"].rank(pct=True, method="average")
    _log("[QualT5] RAW → percentili [0,1] nel sample.")
    return df[["docno","quality"]]