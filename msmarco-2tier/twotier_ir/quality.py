import os, glob, shutil
import pandas as pd
from pyterrier_quality import QualCache
from .config import DATASET_ID, HIGH_SHARE

def _iter_cache_rows(qc):
    for rec in qc:
        try:
            docno = str(rec.get("docno")); qual = float(rec.get("quality"))
        except Exception:
            try:
                docno, qual = rec; docno = str(docno); qual = float(qual)
            except Exception:
                try:
                    docno = str(getattr(rec, "docno")); qual = float(getattr(rec,"quality"))
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
            print("[QualT5] @quantiles OK → filtering-only.")
            return df
        else:
            print("[QualT5] @quantiles empty; fallback RAW.")
    except Exception as e:
        print(f"[QualT5] @quantiles failed ({e}); fallback RAW.")
        try:
            for p in glob.glob(os.path.expanduser("~/.pyterrier/**/qt5-tiny.msmarco-passage.cache*"), recursive=True):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    qc = QualCache.from_url(f"hf:{dataset_id}")
    rows = [(d,float(q)) for d,q in _iter_cache_rows(qc) if d in wanted]
    if not rows:
        raise RuntimeError("No overlap between sample and Qual cache.")
    df = pd.DataFrame(rows, columns=["docno","quality_raw"]).sort_values(["quality_raw","docno"]).reset_index(drop=True)
    df["quality"] = df["quality_raw"].rank(pct=True, method="average")
    return df[["docno","quality"]]

def split_high_low(passages_q: pd.DataFrame):
    cut = 1.0 - HIGH_SHARE
    high = passages_q.loc[passages_q["quality"] >= cut, ["docno","text"]]
    low  = passages_q.loc[passages_q["quality"] <  cut, ["docno","text"]]
    return high, low
