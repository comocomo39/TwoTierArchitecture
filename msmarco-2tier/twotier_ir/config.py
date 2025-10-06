from pathlib import Path
import os

# Parametri “ex-notebook” (puoi cambiare da CLI)
SAMPLE_TOPICS = int(os.getenv("SAMPLE_TOPICS", 1000))
NEG_PER_REL   = int(os.getenv("NEG_PER_REL", 3))
MAX_DOCS      = int(os.getenv("MAX_DOCS", 20_000))
HIGH_SHARE    = float(os.getenv("HIGH_SHARE", 0.70))
LOW_SHARE     = 1.0 - HIGH_SHARE
TOPK          = int(os.getenv("TOPK", 50))
FAISS_DEPTH   = int(os.getenv("FAISS_DEPTH", 500))
NPROBE        = int(os.getenv("NPROBE", 32))
MARGIN        = float(os.getenv("MARGIN", 0.05))
TAU           = float(os.getenv("TAU", 1.0))
ENTR_TH       = float(os.getenv("ENTR_TH", 0.8))
TOPN_ENTROPY  = int(os.getenv("TOPN_ENTROPY", 10))
MARGIN_MODE   = os.getenv("MARGIN_MODE", "absolute")  # or "relative"

# Dataset & indici
DATASET_ID = os.getenv("DATASET_ID", "pyterrier-quality/qt5-tiny.msmarco-passage.cache")
PT_DATASET = os.getenv("PT_DATASET", "irds:msmarco-passage/dev")

# Percorsi (predefiniti pensati per Docker)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app"))
DATA_ROOT    = Path(os.getenv("DATA_ROOT", "/app/data"))
INDEX_ROOT   = Path(os.getenv("INDEX_ROOT", "/app/indexes"))

INDEX_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ColBERT
CHECKPOINT = os.getenv("COLBERT_CHECKPOINT", "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip")
IDX_HIGH   = os.getenv("IDX_HIGH", "msmarco_high_70pct")
IDX_LOW    = os.getenv("IDX_LOW",  "msmarco_low_30pct")
