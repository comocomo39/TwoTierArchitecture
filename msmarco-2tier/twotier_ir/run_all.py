import argparse, pandas as pd
from .utils import bootstrap
from .config import *
from .dataio import load_topics_qrels, build_corpus_with_rel
from .quality import load_quality_for_sample, split_high_low
from .indexing import ensure_colbert_index, build_factories
from .twotier import ColBERTTwoTier
from .evaluate import coverage, run_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU for ColBERT")
    parser.add_argument("--mode", default="all", choices=[
        "margin_or_entropy","margin_and_entropy","entropy","margin","all"
    ])
    args = parser.parse_args()

    bootstrap()

    # 1) dataset & sample
    ds, topics, qrels = load_topics_qrels()
    rel_docnos = set(qrels.loc[qrels["label"] > 0, "docno"])
    passages = build_corpus_with_rel(ds, rel_docnos, max_docs=MAX_DOCS, neg_per_rel=NEG_PER_REL, seed=42)
    print(f"[subset] rel_included={len(rel_docnos & set(passages['docno']))} total_subset={len(passages)}")

    # 2) quality & split
    qual = load_quality_for_sample(passages["docno"])
    passages_q = passages.merge(qual, on="docno", how="inner")
    high, low = split_high_low(passages_q)
    print(f"[SPLIT] High={len(high)} Low={len(low)}")

    # 3) indexing + factories
    ensure_colbert_index(high, IDX_HIGH, gpu=args.gpu)
    ensure_colbert_index(low,  IDX_LOW,  gpu=args.gpu)
    dense_high, dense_low = build_factories(IDX_HIGH, IDX_LOW, NPROBE, FAISS_DEPTH, gpu=args.gpu)

    # 4) pipelines two-tier
    def mk_tier(mode, final_topk=TOPK, margin_mode=MARGIN_MODE):
        return ColBERTTwoTier(dense_high, dense_low, mode=mode,
                              margin=MARGIN, tau=TAU, entropy_threshold=ENTR_TH,
                              topn_entropy=TOPN_ENTROPY, final_topk=final_topk,
                              log_stats=True, log_per_query=False, time_unit="ms",
                              margin_mode=margin_mode)

    tiers = []
    if args.mode in ("margin_or_entropy","all"): tiers.append(("two_tier(margin_or_entropy)", mk_tier("margin_or_entropy")))
    if args.mode in ("margin_and_entropy","all"): tiers.append(("two_tier(margin_and_entropy)", mk_tier("margin_and_entropy")))
    if args.mode in ("entropy","all"): tiers.append(("two_tier(entropy)", mk_tier("entropy")))
    if args.mode in ("margin","all"):
        tiers.append(("two_tier(margin_abs)", mk_tier("margin", margin_mode="absolute")))
        tiers.append(("two_tier(margin_rel)", mk_tier("margin", margin_mode="relative")))

    # 5) evaluation (sulle query coperte)
    topics_cov, qrels_cov = coverage(topics, qrels, high, low)
    pipelines = [dense_high] + [t for _, t in tiers]
    names     = ["ColBERT_high_only"] + [n for n, _ in tiers]
    res = run_experiment(pipelines, topics_cov, qrels_cov, names)
    print("Done.")

if __name__ == "__main__":
    main()
