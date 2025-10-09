import os, math, time, random, glob, shutil
from pathlib import Path
import logging, yaml
import numpy as np, pandas as pd
import pyterrier as pt
from typing import Tuple

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

def adaptive_nprobe(base_nprobe: int, k: int, scale: float):
    return int(max(1, min(2048, round(base_nprobe * (1.0 + scale * math.log(max(2,k), 10))))))
