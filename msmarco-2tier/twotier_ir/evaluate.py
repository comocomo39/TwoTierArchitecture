import pandas as pd
import pyterrier as pt

def coverage(topics, qrels, high, low):
    indexed_docnos = set(pd.concat([high["docno"], low["docno"]]))
    qrels_cov = qrels[qrels["docno"].isin(indexed_docnos)].copy()
    topics_cov = topics[topics["qid"].isin(set(qrels_cov["qid"]))].copy()
    print(f"[coverage] queries covered: {len(topics_cov)} / {len(topics)}")
    return topics_cov, qrels_cov

def run_experiment(pipelines, topics, qrels, names, metrics=None):
    metrics = metrics or ["ndcg_cut_10","recip_rank","map","P_10"]
    res = pt.Experiment(pipelines, topics, qrels, eval_metrics=metrics, names=names)
    print(res)
    return res
