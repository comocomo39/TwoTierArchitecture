#!/usr/bin/env python3
"""
Generate 8 trade-off plots (NDCG@10 vs Time, MRR vs Time) from Results.pdf tables.

Outputs:
- runs/plot_tradeoffs/<run_tag>/
    - TwoTier_nprobe=128_NDCG_vs_Time.png
    - TwoTier_nprobe=128_MRR_vs_Time.png
    - TwoTier_nprobe=8192_NDCG_vs_Time.png
    - TwoTier_nprobe=8192_MRR_vs_Time.png
    - VirtualPartition_nprobe=128_NDCG_vs_Time.png
    - VirtualPartition_nprobe=128_MRR_vs_Time.png
    - VirtualPartition_nprobe=8192_NDCG_vs_Time.png
    - VirtualPartition_nprobe=8192_MRR_vs_Time.png
    - <cfg>_points.csv  (one per cfg)

Usage
-----
python -m src.plot_tradeoffs --run-tag exp1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# Avoid matplotlib cache permission issues (common in containers)
mpl_cache_dir = os.environ.setdefault("MPLCONFIGDIR", "/workspace/.cache/matplotlib")
Path(mpl_cache_dir).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data (from Results.pdf)
# Each entry: (method, time_ms, ndcg10, mrr)
# -----------------------------
DATASETS: dict[str, list[tuple[str, float, float, float]]] = {
    "TwoTier_nprobe=128": [
        ("HighOnly (20%)", 2.3, 0.2222, 0.2011),
        ("HighOnly (30%)", 3.7, 0.2761, 0.2470),
        ("HighOnly (40%)", 4.7, 0.3115, 0.2767),
        ("HighOnly (50%)", 5.8, 0.3386, 0.3009),
        ("TwoTier (margin=0.5, 20%)", 6.3, 0.2896, 0.2596),
        ("TwoTier (margin=0.5, 30%)", 7.3, 0.3209, 0.2858),
        ("TwoTier (margin=0.5, 40%)", 7.9, 0.3421, 0.3032),
        ("TwoTier (margin=0.5, 50%)", 8.7, 0.3591, 0.3177),
        ("TwoTier (margin=1.0, 20%)", 8.1, 0.3229, 0.2882),
        ("TwoTier (margin=1.0, 30%)", 8.7, 0.3426, 0.3045),
        ("TwoTier (margin=1.0, 40%)", 9.2, 0.3561, 0.3149),
        ("TwoTier (margin=1.0, 50%)", 10.0, 0.3694, 0.3261),
        ("Complete Index (Baseline)", 11.2, 0.3895, 0.3415),
    ],
    "TwoTier_nprobe=8192": [
        ("HighOnly (20%)", 127.3, 0.2375, 0.2145),
        ("HighOnly (30%)", 184.6, 0.2946, 0.2629),
        ("HighOnly (40%)", 276.9, 0.3294, 0.2915),
        ("HighOnly (50%)", 313.6, 0.3589, 0.3172),
        ("TwoTier (margin=0.5, 20%)", 338.7, 0.3069, 0.2742),
        ("TwoTier (margin=0.5, 30%)", 385.3, 0.3405, 0.3024),
        ("TwoTier (margin=0.5, 40%)", 470.0, 0.3598, 0.3174),
        ("TwoTier (margin=0.5, 50%)", 477.0, 0.3794, 0.3337),
        ("TwoTier (margin=1.0, 20%)", 450.3, 0.3424, 0.3044),
        ("TwoTier (margin=1.0, 30%)", 479.0, 0.3632, 0.3218),
        ("TwoTier (margin=1.0, 40%)", 531.8, 0.3758, 0.3308),
        ("TwoTier (margin=1.0, 50%)", 532.3, 0.3909, 0.3431),
        ("Complete Index (Baseline)", 593.7, 0.4104, 0.3584),
    ],
    "VirtualPartition_nprobe=128": [
        ("HighOnly (20%)", 3.2, 0.1930, 0.1773),
        ("HighOnly (30%)", 4.5, 0.2449, 0.2222),
        ("HighOnly (40%)", 5.8, 0.2887, 0.2600),
        ("HighOnly (50%)", 7.4, 0.3171, 0.2834),
        ("TwoTier (margin=0.5, 20%)", 7.8, 0.2644, 0.2389),
        ("TwoTier (margin=0.5, 30%)", 8.8, 0.2989, 0.2689),
        ("TwoTier (margin=0.5, 40%)", 10.1, 0.3256, 0.2909),
        ("TwoTier (margin=0.5, 50%)", 10.6, 0.3443, 0.3066),
        ("TwoTier (margin=1.0, 20%)", 9.7, 0.3049, 0.2740),
        ("TwoTier (margin=1.0, 30%)", 10.7, 0.3267, 0.2919),
        ("TwoTier (margin=1.0, 40%)", 11.4, 0.3458, 0.3071),
        ("TwoTier (margin=1.0, 50%)", 12.3, 0.3591, 0.3180),
        ("Complete Index (Baseline)", 11.2, 0.3895, 0.3415),
    ],
    "VirtualPartition_nprobe=8192": [
        ("HighOnly (20%)", 164.4, 0.2063, 0.1889),
        ("HighOnly (30%)", 240.7, 0.2605, 0.2357),
        ("HighOnly (40%)", 309.3, 0.3061, 0.2746),
        ("HighOnly (50%)", 381.2, 0.3368, 0.2996),
        ("TwoTier (margin=0.5, 20%)", 412.9, 0.2809, 0.2530),
        ("TwoTier (margin=0.5, 30%)", 478.1, 0.3165, 0.2835),
        ("TwoTier (margin=0.5, 40%)", 529.3, 0.3448, 0.3067),
        ("TwoTier (margin=0.5, 50%)", 576.8, 0.3641, 0.3226),
        ("TwoTier (margin=1.0, 20%)", 519.9, 0.3215, 0.2879),
        ("TwoTier (margin=1.0, 30%)", 591.8, 0.3459, 0.3078),
        ("TwoTier (margin=1.0, 40%)", 617.1, 0.3656, 0.3233),
        ("TwoTier (margin=1.0, 50%)", 665.2, 0.3788, 0.3337),
        ("Complete Index (Baseline)", 593.7, 0.4104, 0.3584),
    ],
}


def build_df(config_key: str) -> pd.DataFrame:
    rows = DATASETS[config_key]
    df = pd.DataFrame(rows, columns=["method", "time_ms", "ndcg10", "mrr"])

    def family(m: str) -> str:
        if m.startswith("HighOnly"):
            return "HighOnly"
        if "margin=0.5" in m:
            return "TwoTier m=0.5"
        if "margin=1.0" in m:
            return "TwoTier m=1.0"
        if "Baseline" in m or "Complete Index" in m:
            return "Baseline"
        return "Other"

    def split_pct(m: str):
        for p in ("20%", "30%", "40%", "50%"):
            if p in m:
                return int(p.replace("%", ""))
        return None

    df["family"] = df["method"].apply(family)
    df["split"] = df["method"].apply(split_pct)
    return df


def plot_metric_vs_time(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_png: Path,
    annotate: bool = True,
    dpi: int = 150,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Sort for nicer lines/legends
    df_sorted = df.sort_values(["family", "split"], na_position="last")

    plt.figure(figsize=(6, 5))

    for fam, g in df_sorted.groupby("family", sort=False):
        # Baseline: one point
        if fam == "Baseline":
            plt.plot(g["time_ms"], g[metric], marker="o", linestyle="None", label=fam)
        else:
            plt.plot(g["time_ms"], g[metric], marker="o", label=fam)

        if annotate:
            for _, r in g.iterrows():
                tag = "Base" if fam == "Baseline" else (f"{int(r['split'])}%" if pd.notna(r["split"]) else "")
                if tag:
                    plt.annotate(
                        tag,
                        (r["time_ms"], r[metric]),
                        textcoords="offset points",
                        xytext=(5, -12),   # (x,y): 0 = centrato, y negativo = sotto il punto
                        ha="center",
                        va="top",
                    )

    plt.xlabel("Time (ms/query)")
    plt.ylabel("NDCG@10" if metric == "ndcg10" else "MRR")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate NDCG/MRR vs time plots (8 total).")
    ap.add_argument("--runs-dir", type=str, default="runs", help="Root runs directory (default: runs)")
    ap.add_argument("--subdir", type=str, default="plot_tradeoffs", help="Subfolder under runs/ (default: plot_tradeoffs)")
    ap.add_argument("--run-tag", type=str, default=None, help="Run tag folder name (default: timestamp)")
    ap.add_argument("--no-annotate", action="store_true", help="Disable point annotations (20/30/40/50, Base).")
    ap.add_argument("--dpi", type=int, default=150, help="PNG dpi (default: 150)")
    args = ap.parse_args()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.runs_dir) / args.subdir / run_tag
    outdir.mkdir(parents=True, exist_ok=True)

    configs = [
        "TwoTier_nprobe=128",
        "TwoTier_nprobe=8192",
        "VirtualPartition_nprobe=128",
        "VirtualPartition_nprobe=8192",
    ]

    for cfg in configs:
        df = build_df(cfg)

        # Save points used
        df.to_csv(outdir / f"{cfg}_points.csv", index=False)

        # NDCG@10 vs time
        plot_metric_vs_time(
            df=df,
            metric="ndcg10",
            title=f"{cfg}: NDCG@10 vs Time",
            out_png=outdir / f"{cfg}_NDCG_vs_Time.png",
            annotate=not args.no_annotate,
            dpi=args.dpi,
        )

        # MRR vs time
        plot_metric_vs_time(
            df=df,
            metric="mrr",
            title=f"{cfg}: MRR vs Time",
            out_png=outdir / f"{cfg}_MRR_vs_Time.png",
            annotate=not args.no_annotate,
            dpi=args.dpi,
        )

    print(f"✅ Saved 8 plots + CSVs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
