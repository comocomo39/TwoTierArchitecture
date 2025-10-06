import pandas as pd
import pyterrier as pt
from .config import PT_DATASET, SAMPLE_TOPICS

def load_topics_qrels():
    ds = pt.get_dataset(PT_DATASET)
    topics_all = ds.get_topics().astype({"qid":"str"})
    qrels_all  = ds.get_qrels().astype({"qid":"str","docno":"str"})
    topics = topics_all.sample(n=min(SAMPLE_TOPICS, len(topics_all)), random_state=42).sort_values("qid")
    qrels  = qrels_all[qrels_all["qid"].isin(topics["qid"])].copy()
    return ds, topics, qrels

def build_corpus_with_rel(dataset, must_have: set, max_docs: int, neg_per_rel: int = 3, seed: int = 42):
    import random
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

    need_negs = max(0, min(max_docs, len(must_have)*(1+neg_per_rel)) - len(rows_rel))
    if need_negs > 0:
        taken = set([r["docno"] for r in rows_rel])
        for r in dataset.get_corpus_iter():
            pid = str(r.get("doc_id") or r.get("docno") or r.get("docid"))
            if pid in taken or pid in must_have:
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            rows_neg.append({"docno": pid, "text": text})
            if len(rows_neg) >= need_negs:
                break
    df = pd.DataFrame(rows_rel + rows_neg).drop_duplicates("docno")
    return df
