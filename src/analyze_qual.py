#!/usr/bin/env python3
"""
Simple histograms from Hugging Face quality caches (no CSVs).
Reads two dataset IDs (e.g., QualT5 Tiny and Base), pulls the quality
scores, and saves two basic histograms (plus an overlay) to
runs/analyze_qual/ .

Nuove flag:
- --use-quantiles : converte i punteggi in ranghi percentili [0,1] (appiattisce la distribuzione)
- --normalize-minmax: scala i punteggi in [0,1] (mantiene la forma della distribuzione)
- --counts        : plotta i conteggi (non la densità) ed esporta i conteggi per bin in CSV

Utilizzo
-----
python -m src.simple_qual_hist_hf \
  --tiny-hf pyterrier-quality/qt5-tiny.msmarco-passage.cache \
  --base-hf pyterrier-quality/qt5-base.msmarco-passage.cache \
  --tiny-label tiny --base-label base --bins 120 [--use-quantiles | --normalize-minmax] [--counts]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Evita problemi di permessi per la cache di matplotlib
mpl_cache_dir = os.environ.setdefault("MPLCONFIGDIR", "/workspace/.cache/matplotlib")
Path(mpl_cache_dir).mkdir(parents=True, exist_ok=True)

try:
    from pyterrier_quality import QualCache
except Exception as e:
    raise SystemExit("pyterrier-quality non installato. Esegui: pip install pyterrier-quality datasets huggingface_hub")

def load_qualities_from_hf(dataset_id: str, force_raw: bool = False) -> np.ndarray:
    """
    Carica i punteggi di qualità da un dataset Hugging Face.
    Tenta prima la vista '@quantiles', poi torna alla vista 'raw'.
    """
    # Se force_raw è True, carica solo la vista raw ("")
    views = [""] if force_raw else ["@quantiles", ""]
    
    for view in views:
        ds = f"hf:{dataset_id}{view}"
        try:
            qc = QualCache.from_url(ds)
            vals = []
            for rec in qc:  # qc è un iterabile di record
                q = None
                if isinstance(rec, dict):
                    q = rec.get("quality")
                elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    q = rec[1]  # Assume un formato (docid, score)
                
                if q is not None:
                    try:
                        val = float(q)
                        # Aggiungi questo controllo per filtrare i valori NaN
                        if not np.isnan(val):
                            vals.append(val)
                    except (ValueError, TypeError):
                        pass  # Ignora valori di qualità non numerici
            
            if vals:
                print(f"    -> Caricata vista '{view if view else 'raw'}'")
                return np.asarray(vals, dtype=np.float32)
        except Exception as e:
            print(f"    -> Impossibile caricare la vista '{view if view else 'raw'}': {e}")
            continue
            
    raise RuntimeError(f"Impossibile leggere i punteggi di qualità da {dataset_id}")

def to_percentiles(x: np.ndarray) -> np.ndarray:
    """Converte i punteggi in ranghi percentili [0,1] (scale-free)."""
    return pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=np.float32)

def normalize_min_max(x: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """Scala i punteggi nell'intervallo [0,1] usando un min e max globale."""
    if global_min == global_max:
        # Evita la divisione per zero se tutti i valori sono uguali
        return np.full_like(x, 0.5, dtype=np.float32)
    return ((x - global_min) / (global_max - global_min)).astype(np.float32)

def save_hist_csv(filepath: Path, edges: np.ndarray, hist_data: dict[str, np.ndarray]):
    """Salva i conteggi dei bin dell'istogramma in un file CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["bin_left", "bin_right"] + list(hist_data.keys())
    
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        
        # Combina i bordi e i conteggi di ogni istogramma
        rows = zip(
            edges[:-1],        # bin_left
            edges[1:],        # bin_right
            *hist_data.values() # count_label1, count_label2, ...
        )
        
        for row in rows:
            # Formatta la riga come [float, float, int, int, ...]
            w.writerow(
                [row[0], row[1]] + [int(count) for count in row[2:]]
            )

def plot_hist(q: np.ndarray, label: str, bins: int, out_png: Path, counts: bool = False, counts_csv: Path | None = None):
    """Genera e salva un singolo istogramma."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    
    hist, edges, _ = plt.hist(q, bins=bins, density=not counts, alpha=0.85, label=label)
    
    plt.xlabel("quality score")
    plt.ylabel("count" if counts else "density")
    plt.title(f"Histogram — {label}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    if counts and counts_csv is not None:
        save_hist_csv(counts_csv, edges, {f"count_{label}": hist})

def plot_overlay(a: np.ndarray, b: np.ndarray, labels: tuple[str, str], bins: int, out_png: Path, counts: bool = False, counts_csv: Path | None = None):
    """Genera e salva un istogramma sovrapposto."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    # Determina un range condiviso per un confronto equo
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    bin_range = (lo, hi)

    plt.figure(figsize=(9, 5))
    hist_a, edges, _ = plt.hist(a, bins=bins, range=bin_range, density=not counts, alpha=0.55, label=labels[0])
    hist_b, _, _ = plt.hist(b, bins=bins, range=bin_range, density=not counts, alpha=0.55, label=labels[1])
    
    plt.xlabel("quality score")
    plt.ylabel("count" if counts else "density")
    plt.title("Histogram overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    if counts and counts_csv is not None:
        hist_data = {
            f"count_{labels[0]}": hist_a,
            f"count_{labels[1]}": hist_b
        }
        save_hist_csv(counts_csv, edges, hist_data)

def main():
    ap = argparse.ArgumentParser(description="Genera istogrammi di quality score dai cache di Hugging Face.")
    ap.add_argument("--tiny-hf", type=str, required=True, help="ID dataset HF per Tiny (es. pyterrier-quality/qt5-tiny.msmarco-passage.cache)")
    ap.add_argument("--base-hf", type=str, required=True, help="ID dataset HF per Base (es. pyterrier-quality/qt5-base.msmarco-passage.cache)")
    ap.add_argument("--tiny-label", type=str, default="tiny")
    ap.add_argument("--base-label", type=str, default="base")
    ap.add_argument("--bins", type=int, default=120)
    
    ap.add_argument("--counts", action="store_true",
                    help="Plotta i conteggi e esporta il CSV per bin")
    ap.add_argument("--raw", action="store_true",
                    help="Forza il caricamento del cache RAW (salta @quantiles)")

    # Gruppo per trasformazioni mutuamente esclusive
    transform_group = ap.add_mutually_exclusive_group()
    transform_group.add_argument("--use-quantiles", action="store_true",
                                 help="Converte i punteggi RAW in ranghi percentili [0,1] (appiattisce la distribuzione)")
    transform_group.add_argument("--normalize-minmax", action="store_true",
                                 help="Normalizza i punteggi RAW in [0,1] (mantiene la forma della distribuzione)")
    
    args = ap.parse_args()

    outdir = Path("runs/analyze_qual")
    outdir.mkdir(parents=True, exist_ok=True)

    # Se vogliamo applicare una trasformazione (quantili o min-max),
    # dobbiamo forzare il caricamento dei dati grezzi.
    force_raw = args.raw or args.use_quantiles or args.normalize_minmax

    print(f"[LOAD] {args.tiny_hf}")
    qt = load_qualities_from_hf(args.tiny_hf, force_raw=force_raw)
    print(f"   -> n={qt.size}, mean={qt.mean():.6f}, min={qt.min():.6f}, max={qt.max():.6f}")

    print(f"[LOAD] {args.base_hf}")
    qb = load_qualities_from_hf(args.base_hf, force_raw=force_raw)
    print(f"   -> n={qb.size}, mean={qb.mean():.6f}, min={qb.min():.6f}, max={qb.max():.6f}")

    # Applica le trasformazioni
    if args.use_quantiles:
        print("[INFO] Conversione dei punteggi in ranghi percentili (quantili)...")
        qt = to_percentiles(qt)
        qb = to_percentiles(qb)
    elif args.normalize_minmax:
        print("[INFO] Normalizzazione dei punteggi in [0,1] (Min-Max)...")
        # Trova il min/max globale per un confronto equo
        all_scores = np.concatenate([qt, qb])
        g_min = all_scores.min()
        g_max = all_scores.max()
        print(f"   -> Scala globale: min={g_min:.6f}, max={g_max:.6f}")
        
        qt = normalize_min_max(qt, g_min, g_max)
        qb = normalize_min_max(qb, g_min, g_max)

    # Salva istogrammi singoli
    plot_hist(qt, args.tiny_label, args.bins, outdir / "hist_tiny.png",
              counts=args.counts, counts_csv=(outdir / "hist_tiny_counts.csv" if args.counts else None))
    
    plot_hist(qb, args.base_label, args.bins, outdir / "hist_base.png",
              counts=args.counts, counts_csv=(outdir / "hist_base_counts.csv" if args.counts else None))

    # Salva istogramma sovrapposto
    try:
        plot_overlay(qt, qb, (args.tiny_label, args.base_label), args.bins, outdir / "hist_overlay.png",
                     counts=args.counts, counts_csv=(outdir / "hist_overlay_counts.csv" if args.counts else None))
    except Exception as e:
        print(f"Impossibile generare l'overlay: {e}")

    print(f"✅ Istogrammi salvati in {outdir}")

if __name__ == "__main__":
    main()