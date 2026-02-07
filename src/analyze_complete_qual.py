#!/usr/bin/env python3
"""
analyze_qual.py

Analisi della distribuzione dei quality score (QualT5 Tiny vs Base) nei cluster IVF
di un indice già costruito con create_indexes.py.

Nuova versione:
- Usa un unico indice IVF "complete".
- Calcola, per una lista di low_share (= percentili), le soglie globali e
  le soglie cluster-specific sia per QualT5-Tiny che per QualT5-Base.
- Salva UNA SOLA immagine con 10 istogrammi:
    * 5 righe (low_share = 0.20, 0.40, 0.60, 0.80, 1.00)
    * 2 colonne (sinistra = Tiny, destra = Base)

Ogni subplot mostra:
- istogramma delle soglie cluster-specific per quel low_share
- retta verticale sulla soglia globale del corpus
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt

from common import (
    load_cfg,
    resolve_index_root,
    stamp_run_dir,
    get_logger,
    load_quality_for_sample,
)


# ----------------------------------------------------------------------
# Caricamento indice + ids
# ----------------------------------------------------------------------
def _load_index_and_ids(index_root: Path, index_name: str, log):
    """Carica indice FAISS IVF e ids.npy, facendo qualche check di base."""
    index_dir = index_root / index_name
    faiss_path = index_dir / "faiss.index"
    ids_path = index_dir / "ids.npy"
    meta_path = index_dir / "meta.json"

    if not faiss_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Indice '{index_name}' non trovato in {index_root} "
            f"(mancano faiss.index e/o ids.npy)"
        )

    log.info(f"[LOAD] Index dir: {index_dir}")
    index = faiss.read_index(str(faiss_path))

    # ids.npy è un array di oggetti (stringhe docno)
    ids = np.load(ids_path, allow_pickle=True).astype(str)

    if meta_path.exists():
        meta = json.load(open(meta_path, "r"))
        log.info(
            f"[META] mode={meta.get('mode')} nlist={meta.get('nlist')} "
            f"nprobe_default={meta.get('nprobe_default')} "
            f"N={meta.get('count')} dim={meta.get('dim')}"
        )
    else:
        meta = {}

    # Sanity check: ntotal deve coincidere con len(ids)
    if index.ntotal != len(ids):
        log.warning(
            f"[WARN] index.ntotal={index.ntotal} != len(ids)={len(ids)}. "
            "Proseguo comunque, ma controlla che l'indice sia coerente."
        )

    # Ci aspettiamo un IVF (IVF-Flat | IVFPQ), non HNSW o altro
    if not isinstance(index, faiss.IndexIVF):
        raise TypeError(
            f"Indice caricato non è un IndexIVF (trovato: {type(index)}). "
            "Questo script supporta solo indici IVF (IVF-Flat / IVFPQ)."
        )

    return index, ids, meta


# ----------------------------------------------------------------------
# Mapping vettore -> cluster IVF
# ----------------------------------------------------------------------
def _build_list_id_of_vec(ivf_index: faiss.IndexIVF, N: int, log) -> np.ndarray:
    """
    Costruisce un array list_id_of_vec di shape (N,) tale che
    list_id_of_vec[i] == id del cluster IVF a cui appartiene il vettore i.

    Si assume che l'indice sia stato costruito con add() su vettori in
    ordine 0..N-1, esattamente come in create_indexes.py.
    """
    invlists = ivf_index.invlists
    nlist = ivf_index.nlist

    list_id_of_vec = np.full(N, -1, dtype=np.int32)

    # faiss.rev_swig_ptr ci permette di convertire i pointer in array NumPy
    rev_swig_ptr = faiss.rev_swig_ptr

    assigned = 0
    for list_no in range(nlist):
        list_size = invlists.list_size(list_no)
        if list_size == 0:
            continue
        ids_ptr = invlists.get_ids(list_no)
        ids = rev_swig_ptr(ids_ptr, list_size)
        list_id_of_vec[ids] = list_no
        assigned += list_size

    # Check che tutti gli ID siano stati assegnati a qualche lista
    missing = int(np.sum(list_id_of_vec < 0))
    if missing > 0:
        log.warning(
            f"[WARN] {missing} vettori non assegnati a nessuna lista IVF "
            "(valore -1 in list_id_of_vec)."
        )

    log.info(
        f"[IVF] nlist={nlist}  assigned={assigned}  "
        f"missing={missing}"
    )
    return list_id_of_vec


# ----------------------------------------------------------------------
# Caricamento quality scores per un certo cache-id
# ----------------------------------------------------------------------
def _load_qualities_for_ids(docnos: np.ndarray, dataset_id_quality: str, log):
    """
    Usa load_quality_for_sample per ottenere i quality score (colonna 'quality')
    per tutti i docno presenti in docnos. Restituisce un array float32 di
    shape (len(docnos),) allineato a docnos, con NaN in caso di mancanza.
    """
    log.info(
        f"[Qual] Caricamento qualità da '{dataset_id_quality}' "
        f"per {len(docnos):,} doc."
    )
    df_q = load_quality_for_sample(docnos, dataset_id_quality, log)
    # df_q: colonna 'docno' (str), colonna 'quality' (float)
    qmap = dict(
        zip(
            df_q["docno"].astype(str).tolist(),
            df_q["quality"].astype(float).tolist(),
        )
    )

    qualities = np.empty(len(docnos), dtype=np.float32)
    missing = 0
    for i, d in enumerate(docnos.astype(str)):
        q = qmap.get(d)
        if q is None:
            qualities[i] = np.nan
            missing += 1
        else:
            qualities[i] = float(q)

    if missing > 0:
        log.warning(
            f"[Qual] Nessun quality score per {missing} doc; marcati come NaN."
        )

    return qualities


# ----------------------------------------------------------------------
# Soglie cluster-specific per un certo low_share
# ----------------------------------------------------------------------
def _compute_thresholds_per_cluster(
    list_id_of_vec: np.ndarray,
    qualities: np.ndarray,
    low_share: float,
    min_docs_per_cluster: int,
    log,
):
    """
    Calcola le soglie cluster-specific:
    - per ogni cluster c, prende i quality score dei doc in c
    - se il cluster ha almeno min_docs_per_cluster doc validi (non-NaN),
      calcola il percentile low_share (es. 0.20) come soglia per c.

    Restituisce:
    - thresholds: array delle soglie cluster-specific
    - rows: lista di dict con statistiche per CSV
    """
    assert 0.0 <= low_share <= 1.0, "low_share deve essere in [0,1]"

    cluster_thresholds = []
    rows = []

    nlist = int(list_id_of_vec.max()) + 1
    log.info(
        f"[THR] Calcolo soglie cluster-specific con low_share={low_share:.2f} "
        f"(~{int(low_share*100)}% low / {int((1-low_share)*100)}% high), "
        f"min_docs_per_cluster={min_docs_per_cluster}"
    )

    for c in range(nlist):
        idx = np.where(list_id_of_vec == c)[0]
        if idx.size == 0:
            continue

        qs = qualities[idx]
        qs = qs[~np.isnan(qs)]
        if qs.size < min_docs_per_cluster:
            continue

        thr_c = float(np.quantile(qs, low_share))
        cluster_thresholds.append(thr_c)

        rows.append(
            {
                "cluster": c,
                "size": int(qs.size),
                "threshold": thr_c,
                "low_share": low_share,
                "q_mean": float(qs.mean()),
                "q_std": float(qs.std()),
                "q_min": float(qs.min()),
                "q_max": float(qs.max()),
            }
        )

    thresholds = np.array(cluster_thresholds, dtype=np.float32)
    log.info(
        f"[THR] Soglie calcolate per {thresholds.size} cluster "
        f"(su {nlist} totali con almeno {min_docs_per_cluster} doc validi)."
    )

    return thresholds, rows


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Analizza la distribuzione dei quality score nei cluster IVF di un indice "
            "TAS-B (IVF-Flat / IVFPQ) e produce un'unica immagine con istogrammi "
            "Tiny vs Base per diversi low_share."
        )
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help=(
            "Nome della directory dell'indice sotto indices_dir. "
            "Se omesso, usa {dataset_tag}_complete_tasb_ivf_{ivf_quantizer} "
            "come in create_indexes.py."
        ),
    )
    parser.add_argument(
        "--min-docs-per-cluster",
        type=int,
        default=10,
        help="Numero minimo di documenti validi in un cluster per calcolare una soglia. Default: 10.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Numero di bin per l'istogramma delle soglie cluster-specific. Default: 100.",
    )

    args = parser.parse_args(argv)

    # ========= Config =========
    PATHS = load_cfg("configs/paths.yaml")
    DATA = load_cfg("configs/dataset.yaml")
    HP = load_cfg("configs/tasb_two_tier.yaml")

    INDEX_ROOT = resolve_index_root(PATHS)
    dataset_tag = DATA.get("dataset_tag", "msmarco")
    ivf_mode = HP.get("ivf_quantizer", "flat").lower()

    # ID qualità per Tiny e Base
    dataset_id_tiny = DATA.get("dataset_id_quality_tiny", DATA["dataset_id_quality"])
    dataset_id_base = DATA.get(
        "dataset_id_quality_base",
        dataset_id_tiny.replace("tiny", "base"),
    )

    # Nome indice di default (come in create_indexes.py, build_mode=complete)
    default_index_name = f"{dataset_tag}_complete_tasb_ivf_{ivf_mode}"
    index_name = args.index_name or default_index_name

    # Directory output per questa analisi
    run_dir = stamp_run_dir(
        PATHS["runs_dir"], f"analyze_ivf_qual_tiny_base_{ivf_mode}_{dataset_tag}"
    )
    log = get_logger(run_dir)
    log.info("===== ANALYZE IVF QUAL (Tiny vs Base) START =====")
    log.info(f"Index root : {INDEX_ROOT}")
    log.info(f"Index name : {index_name}")
    log.info(f"Dataset tag: {dataset_tag}")
    log.info(f"Qual Tiny  : {dataset_id_tiny}")
    log.info(f"Qual Base  : {dataset_id_base}")
    log.info(f"min_docs_per_cluster: {args.min_docs_per_cluster}")
    log.info(f"bins       : {args.bins}")

    # ========= Caricamento indice + ids =========
    index, ids, meta = _load_index_and_ids(INDEX_ROOT, index_name, log)
    N = len(ids)
    log.info(f"[DATA] N doc: {N:,}")

    # ========= list_id_of_vec =========
    list_id_of_vec = _build_list_id_of_vec(index, N, log)

    # ========= Quality Tiny & Base =========
    qualities_tiny = _load_qualities_for_ids(ids, dataset_id_tiny, log)
    qualities_base = _load_qualities_for_ids(ids, dataset_id_base, log)

    valid_tiny = qualities_tiny[~np.isnan(qualities_tiny)]
    valid_base = qualities_base[~np.isnan(qualities_base)]
    if valid_tiny.size == 0 or valid_base.size == 0:
        log.error("[Qual] Nessun quality score valido (Tiny o Base). Esco.")
        return 1

    # Lista di low_share che vogliamo plottare (5 righe)
    low_shares = [0.20, 0.30, 0.40,0.50, 0.60,0.70, 0.80]

    # ========= Figura con 5x2 subplot =========
    fig, axes = plt.subplots(
        nrows=len(low_shares),
        ncols=2,
        figsize=(20, 28),
        sharex=True,
    )

    # Forziamo gli assi in un array 2D anche se len(low_shares)==1
    axes = np.atleast_2d(axes)

    for row, low_share in enumerate(low_shares):
        # ----- Tiny (colonna 0) -----
        global_thr_tiny = float(np.quantile(valid_tiny, low_share))
        thr_tiny, _rows_tiny = _compute_thresholds_per_cluster(
            list_id_of_vec=list_id_of_vec,
            qualities=qualities_tiny,
            low_share=low_share,
            min_docs_per_cluster=args.min_docs_per_cluster,
            log=log,
        )

        ax_tiny = axes[row, 0]
        if thr_tiny.size > 0:
            ax_tiny.hist(thr_tiny, bins=args.bins)
            ax_tiny.axvline(
                global_thr_tiny,
                linestyle="--",
                linewidth=1.8,
                color="orange",
                label=f"Soglia globale = {global_thr_tiny:.3f}",
            )
        ax_tiny.set_ylabel(f"low={low_share:.2f}\n#cluster", fontsize=20)
        ax_tiny.legend(fontsize=20)
        if row == 0:
            ax_tiny.set_title("QualT5-Tiny", fontsize=20)

        # ----- Base (colonna 1) -----
        global_thr_base = float(np.quantile(valid_base, low_share))
        thr_base, _rows_base = _compute_thresholds_per_cluster(
            list_id_of_vec=list_id_of_vec,
            qualities=qualities_base,
            low_share=low_share,
            min_docs_per_cluster=args.min_docs_per_cluster,
            log=log,
        )

        ax_base = axes[row, 1]
        if thr_base.size > 0:
            ax_base.hist(thr_base, bins=args.bins)
            ax_base.axvline(
                global_thr_base,
                linestyle="--",
                linewidth=1.8,
                color="orange",
                label=f"Soglia globale = {global_thr_base:.3f}",
            )
        ax_base.legend(fontsize=20)
        if row == 0:
            ax_base.set_title("QualT5-Base", fontsize=20)

    # Label asse X solo sull'ultima riga
    for col in range(2):
        axes[-1, col].set_xlabel("Soglia di quality cluster-specific")

    fig.suptitle(
        "Distribuzione delle soglie cluster-specific (IVF)\n"
        "Confronto QualT5-Tiny vs QualT5-Base per diversi low_share",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_img = run_dir / "ivf_cluster_thresholds_tiny_vs_base_grid.png"
    fig.savefig(out_img, dpi=200)
    plt.close(fig)
    log.info(f"[PLOT] Figura multipla salvata in {out_img}")

    log.info("===== ANALYZE IVF QUAL (Tiny vs Base) DONE =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
