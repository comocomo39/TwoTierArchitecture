import os
import numpy as np
import pandas as pd
from pathlib import Path
import pyterrier as pt
import pyterrier_dr as dr
import faiss

from .common import (
    ensure_pt,
    load_cfg,
)

# 1. setup
ensure_pt()
PATHS = load_cfg("configs/paths.yaml")
DATA  = load_cfg("configs/dataset.yaml")

SPLIT = DATA["split"]  # tipo "msmarco-passage/dev"
CACHE = Path(PATHS.get("cache_dir", "cache"))
CACHE.mkdir(parents=True, exist_ok=True)

# device per l'encoder (stesso discorso dei doc)
ENC_DEVICE = os.getenv("ENC_DEVICE", "cuda")

# 2. carico dataset queries msmarco/dev
import ir_datasets as irds
ds_full = pt.get_dataset(SPLIT)

topics_all = ds_full.get_topics().astype({"qid":"str"}).copy()
# restringi a msmarco-passage/dev/small perché è quello che usi in eval
small_ds = irds.load("msmarco-passage/dev/small")
small_qids = {q.query_id for q in small_ds.queries_iter()}
topics_all = topics_all[topics_all["qid"].isin(small_qids)].copy()

# sanity: ci aspettiamo colonne ['qid','query']
assert "qid" in topics_all.columns
# a volte pyterrier chiama il testo "query", a volte "query_terrier"
# fallback robusto:
query_col = "query" if "query" in topics_all.columns else (
    "query_terrier" if "query_terrier" in topics_all.columns else None
)
if query_col is None:
    raise RuntimeError("Non trovo la colonna testo query in topics_all")

topics_all = topics_all[["qid", query_col]].rename(columns={query_col: "query_text"})
topics_all = topics_all.reset_index(drop=True)

# 3. encoder TAS-B per le query
tasb = dr.TasB(device=ENC_DEVICE)
qenc = tasb.query_encoder(batch_size=int(os.getenv("QENC_BS", "64")))

# 4. encodiamo in batch e normalizziamo L2
all_qvecs = []
all_qids = []
bs = int(os.getenv("QENC_BS", "64"))

for start in range(0, len(topics_all), bs):
    end = min(start + bs, len(topics_all))
    chunk = topics_all.iloc[start:end][["qid","query_text"]].copy()

    out = qenc.transform(chunk.rename(columns={"query_text":"query"}))
    # out contiene colonna "query_vec"
    vecs = np.vstack(out["query_vec"].values).astype(np.float32, copy=False)

    # L2 normalize (importantissimo per essere coerenti con la search IP=cosine)
    #faiss.normalize_L2(vecs)

    all_qids.extend(out["qid"].astype(str).tolist())
    all_qvecs.extend(list(vecs))  # una entry per riga

# 5. costruiamo df finale
qvec_df = pd.DataFrame({
    "qid": all_qids,
    "query_vec": all_qvecs,
})

# 6. salviamo parquet con un nome chiaro
out_path = CACHE / f"qvec_{SPLIT.replace('/','_')}_recomputed.parquet"
qvec_df.to_parquet(out_path, index=False)

print(f"[OK] Salvato {out_path} con {len(qvec_df)} query.")
