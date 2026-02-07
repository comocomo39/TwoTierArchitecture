# src/evaluate.py
import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyterrier as pt
import faiss

def _make_gpu_res():
    res = faiss.StandardGpuResources()
    try:
        res.setTempMemory(_FAISS_TEMP_MB * 1024 * 1024)
    except Exception:
        pass
    return res

def _cpu_to_gpu(index_cpu):
    if not USE_GPU:
        return index_cpu, None
    co = faiss.GpuClonerOptions()
    co.useFloat16 = _FAISS_FP16
    co.shard = _FAISS_SHARD
    res = _make_gpu_res()
    index_gpu = faiss.index_cpu_to_gpu(res, _FAISS_DEVICE, index_cpu, co)
    return index_gpu, res

def _gpu_to_cpu(index_gpu):
    try:
        return faiss.index_gpu_to_cpu(index_gpu)
    except Exception:
        return index_gpu

def _load_faiss_index(path: Path):
    idx = faiss.read_index(str(path))
    if isinstance(idx, faiss.IndexIVF):
        idx.nprobe = NPROBE
    if USE_GPU:
        gpu_idx, gpu_res = _cpu_to_gpu(idx)
        try:
            gpu_idx._gpu_res = gpu_res
        except Exception:
            pass
        idx = gpu_idx
        if isinstance(idx, faiss.IndexIVF):
            idx.nprobe = NPROBE
    return idx

def _load_ids_safe(path: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError:
        # vecchi file salvati come dtype=object -> serve allow_pickle=True
        arr = np.load(path, allow_pickle=True)
    # canonizza a stringhe “pure” (no object)
    return np.asarray(arr, dtype=str)

# ==============================================================================
# 1. SEARCHER FISICO OTTIMIZZATO (Vettorializzato)
# ==============================================================================
class FaissIVFPQSearcher(pt.Transformer):
    def __init__(self, index_dir, topk=1000, index_loader=None, nprobe=None):
        super().__init__()
        self.topk = int(topk)
        
        # Caricamento Indice
        if index_loader:
            self.index = index_loader(index_dir / "faiss.index")
        else:
            self.index = faiss.read_index(str(index_dir / "faiss.index"))
            
        # Caricamento ID
        self.ids = _load_ids_safe(index_dir / "ids.npy")
        
        # Override nprobe
        if isinstance(self.index, faiss.IndexIVF) and nprobe is not None:
            self.index.nprobe = int(nprobe)
            

    def transform(self, df_queries):
        # 1. Preparazione Vettori (Zero-copy)
        qids = df_queries["qid"].astype(str).values
        qvecs = np.ascontiguousarray(np.vstack(df_queries["query_vec"].values).astype(np.float32))
        n_q = len(qids)
        
        # 2. Ricerca FAISS
        scores, idxs = self.index.search(qvecs, self.topk)
        
        # 3. Formattazione Vettoriale (NO LOOP PYTHON)
        n_q, k = scores.shape
        scores_flat = scores.ravel()
        idxs_flat = idxs.ravel()
        
        # Maschera validità (-1 significa meno risultati del richiesto)
        valid_mask = idxs_flat >= 0
        if not np.any(valid_mask):
            return pd.DataFrame(columns=["qid", "docno", "rank", "score"])

        # Costruzione DataFrame istantanea
        return pd.DataFrame({
            "qid": np.repeat(qids, k)[valid_mask],
            "docno": self.ids[idxs_flat[valid_mask]],
            "rank": np.tile(np.arange(1, k + 1), n_q)[valid_mask],
            "score": scores_flat[valid_mask]
        })

# ==============================================================================
# 2. TWO TIER LOGIC (Standard)
# ==============================================================================
class TwoTier(pt.Transformer):
    def __init__(self, retr_high, retr_low, mode="margin_or_entropy",
                 margin=0.05, margin_mode="absolute",
                 tau=1.0, entropy_threshold=0.8, topn_entropy=10, final_topk=None,
                 log_stats=True, time_unit="ms"):
        self.retr_high, self.retr_low = retr_high, retr_low
        self.mode, self.margin = mode, float(margin)
        self.margin_mode = str(margin_mode).lower()
        self.tau, self.entropy_threshold = float(tau), float(entropy_threshold)
        self.topn_entropy, self.final_topk = int(topn_entropy), final_topk
        self.log_stats = bool(log_stats)
        self.time_unit = "ms" if time_unit not in ("ms","s") else time_unit
        self._eps = 1e-12
        self.reset_stats()

    def reset_stats(self):
        self._q = 0; self._low_acts = 0; self._t_high = 0.0; self._t_low = 0.0; self._t_merge = 0.0
    def _s(self, seconds): return seconds * (1000.0 if self.time_unit == "ms" else 1.0)

    def _gating_flags(self, run_h: pd.DataFrame) -> pd.Series:
        if run_h.empty: return pd.Series(dtype=bool)
        def need_low(group):
            s = group["score"].to_numpy(dtype=np.float32, copy=False)
            if len(s) < 2: return True
            if self.mode != "entropy":
                diff = s[0] - s[1]
                if self.margin_mode == "relative":
                    if (diff / (abs(s[0]) + 1e-9)) < self.margin: 
                        if self.mode == "margin": return True
                    elif self.mode == "margin": return False
                else:
                    if diff < self.margin:
                        if self.mode == "margin": return True
                    elif self.mode == "margin": return False
            k = min(self.topn_entropy, len(s))
            top = s[:k]
            z = top / max(self.tau, 1e-9); z -= np.max(z)
            p = np.exp(z); p /= (p.sum() + 1e-9)
            ent = -np.sum(p * np.log(p + 1e-9))
            thr = self.entropy_threshold * (np.log(k) if k > 1 else 1.0)
            return ent >= thr
        return run_h.groupby("qid", sort=False).apply(need_low)

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.perf_counter(); run_h = self.retr_high.transform(topics_df); t1 = time.perf_counter()
        
        # Gating
        flags = self._gating_flags(run_h)
        if flags.empty: qids_low = set()
        else: qids_low = set(flags[flags].index.astype(str))
        self._q += len(topics_df["qid"].unique())
        
        run_l = None
        if qids_low:
            need_low_df = topics_df[topics_df["qid"].astype(str).isin(qids_low)]
            run_l = self.retr_low.transform(need_low_df)
            
        t2 = time.perf_counter()
        
        # Merge
        if run_l is not None and not run_l.empty:
            merged = pd.concat([run_h, run_l], ignore_index=True)
            merged.sort_values(["qid","score"], ascending=[True, False], inplace=True)
        else:
            merged = run_h
            
        k = self.final_topk if self.final_topk is not None else len(merged)
        if not merged.empty:
            merged = merged.groupby("qid", group_keys=False).head(k)
            merged["rank"] = merged.groupby("qid")["score"].rank(ascending=False, method="first").astype(np.int32)
            
        t3 = time.perf_counter()
        if self.log_stats:
            self._low_acts += len(qids_low); self._t_high+=(t1-t0); self._t_low+=(t2-t1); self._t_merge+=(t3-t2)
        return merged[["qid","docno","rank","score"]]

    def get_stats(self):
        q = max(1, self._q)
        return dict(
            queries=self._q,
            low_activations=self._low_acts,
            activation_rate=(self._low_acts / q),
            avg_high_time_ms=self._s(self._t_high / q),
            avg_low_time_ms=self._s(self._t_low  / q),
            avg_merge_time_ms=self._s(self._t_merge/ q),
        )

class SharedMaskedTwoTier(pt.Transformer):
    """
    Two-Tier Virtuale (Standard OpenMP).
    Uniformato per usare il multithreading interno come l'indice fisico.
    """
    def __init__(self, index, docnos, mask_high, mask_low, 
                 mode="margin_or_entropy", margin=0.05, 
                 tau=1.0, entropy_threshold=0.5, topn_entropy=10, 
                 final_topk=1000, nprobe=None, topk_tier=100):
        
        super().__init__()
        self.index = index
        self.docnos = docnos
        self.topk_tier = topk_tier
        self.final_topk = final_topk
        
        # === UNIFORMITÀ: SBLOCCA I THREAD ===
        # Lasciamo che FAISS usi tutti i core disponibili, come nel metodo fisico.
        try:
            faiss.omp_set_num_threads(multiprocessing.cpu_count())
        except:
            pass
        
        self.quantizer = index.quantizer
        self.mode, self.margin = mode, margin
        self.tau, self.entropy_threshold = tau, entropy_threshold
        self.topn_entropy = topn_entropy
        
        try:
            import faiss.swigfaiss as swigfaiss
        except ImportError:
            swigfaiss = faiss
        
        self._fine_search_func = getattr(swigfaiss, "IndexIVF_search_preassigned", None)
        if self._fine_search_func is None:
            self._fine_search_func = swigfaiss.IndexIVF.search_preassigned

        if nprobe:
            self.index.nprobe = nprobe
            
        self.params_high = self._create_params_bitmap_safe(mask_high, nprobe)
        self.params_low  = self._create_params_bitmap_safe(mask_low, nprobe)
        self.reset_stats()
    
    def _create_params_bitmap_safe(self, mask_bool, nprobe):
        try:
            bitmap = np.packbits(mask_bool, bitorder='little')
        except TypeError:
            remainder = len(mask_bool) % 8
            if remainder > 0:
                pad = np.zeros(8 - remainder, dtype=bool)
                padded = np.concatenate([mask_bool, pad])
            else:
                padded = mask_bool
            padded = padded.reshape(-1, 8)[:, ::-1].ravel()
            bitmap = np.packbits(padded)
        sel = faiss.IDSelectorBitmap(len(mask_bool), faiss.swig_ptr(bitmap))
        params = faiss.SearchParametersIVF(sel=sel)
        if nprobe: params.nprobe = nprobe
        params._keep_alive_selector = sel 
        params._keep_alive_bitmap = bitmap 
        return params

    def reset_stats(self):
        self._q = 0; self._low_acts = 0
        self._t_coarse = 0.0; self._t_high = 0.0; self._t_low = 0.0; self._t_merge = 0.0

    def _gating_flags(self, run_h_df):
        if run_h_df.empty: return pd.Series(dtype=bool)
        def need_low(group):
            s = group["score"].values
            if len(s) < 2: return True
            if self.mode != "entropy":
                if (s[0] - s[1]) < self.margin:
                    if self.mode == "margin": return True
                elif self.mode == "margin": return False
            k = min(self.topn_entropy, len(s))
            top = s[:k]
            z = top / max(self.tau, 1e-9); z_max = z.max()
            z_exp = np.exp(z - z_max); p = z_exp / z_exp.sum()
            ent = -np.sum(p * np.log(p + 1e-9))
            thr = self.entropy_threshold * (np.log(k) if k > 1 else 1.0)
            return ent >= thr
        return run_h_df.groupby("qid", sort=False)[["score"]].apply(need_low)

    def _run_search_preassigned(self, n_q, qvecs, coarse_I, coarse_D, params):
        scores = np.empty((n_q, self.topk_tier), dtype=np.float32)
        idxs = np.empty((n_q, self.topk_tier), dtype=np.int64)
        self._fine_search_func(
            self.index, int(n_q), 
            faiss.swig_ptr(qvecs), 
            self.topk_tier,
            faiss.swig_ptr(coarse_I), 
            faiss.swig_ptr(coarse_D),
            faiss.swig_ptr(scores), 
            faiss.swig_ptr(idxs),
            False, params
        )
        return scores, idxs

    def _format_results(self, scores, idxs, qids):
        n_q, k = scores.shape
        valid_mask = idxs.ravel() >= 0
        if not np.any(valid_mask): return pd.DataFrame(columns=["qid", "docno", "rank", "score"])
        return pd.DataFrame({
            "qid": np.repeat(qids, k)[valid_mask],
            "docno": self.docnos[idxs.ravel()[valid_mask]],
            "rank": np.tile(np.arange(1, k + 1), n_q)[valid_mask],
            "score": scores.ravel()[valid_mask]
        })

    def transform(self, df_queries):
        import time
        qids = df_queries["qid"].astype(str).values
        qvecs = np.ascontiguousarray(np.vstack(df_queries["query_vec"].values).astype(np.float32))
        n_q = len(qids)
        t0 = time.perf_counter()
        
        nprobe = self.index.nprobe
        if self.params_high.nprobe > 0: nprobe = self.params_high.nprobe
        
        try:
            D_out, I_out = self.quantizer.search(qvecs, nprobe)
            coarse_D = np.ascontiguousarray(D_out, dtype=np.float32) # Distance
            coarse_I = np.ascontiguousarray(I_out, dtype=np.int64) # Cluster IDs
        except TypeError:
            coarse_D = np.empty((n_q, nprobe), dtype=np.float32)
            coarse_I = np.empty((n_q, nprobe), dtype=np.int64)
            self.quantizer.search(n_q, faiss.swig_ptr(qvecs), nprobe, faiss.swig_ptr(coarse_D), faiss.swig_ptr(coarse_I))

        t1 = time.perf_counter()
        self._t_coarse += (t1 - t0)
        
        sc_h, idx_h = self._run_search_preassigned(n_q, qvecs, coarse_I, coarse_D, self.params_high)
        run_h = self._format_results(sc_h, idx_h, qids)
        t2 = time.perf_counter()
        self._t_high += (t2 - t1)
        
        flags = self._gating_flags(run_h)
        if flags.empty: needed_qids = set()
        else: needed_qids = set(flags[flags].index)
        self._q += n_q; self._low_acts += len(needed_qids)
        
        run_l = None
        if len(needed_qids) > 0:
            is_low_idx = np.array([q in needed_qids for q in qids])
            sub_n = int(is_low_idx.sum())
            if sub_n > 0:
                sc_l, idx_l = self._run_search_preassigned(
                    sub_n, 
                    np.ascontiguousarray(qvecs[is_low_idx]), 
                    np.ascontiguousarray(coarse_I[is_low_idx]), 
                    np.ascontiguousarray(coarse_D[is_low_idx]), 
                    self.params_low
                )
                run_l = self._format_results(sc_l, idx_l, qids[is_low_idx])
        t3 = time.perf_counter()
        self._t_low += (t3 - t2)
        
        if run_l is not None and not run_l.empty:
            merged = pd.concat([run_h, run_l], ignore_index=True)
            merged.sort_values(["qid","score"], ascending=[True, False], inplace=True)
        else:
            merged = run_h
        if self.final_topk:
            merged = merged.groupby("qid").head(self.final_topk)
        if not merged.empty:
            merged["rank"] = merged.groupby("qid")["score"].rank(ascending=False, method="first").astype(int)
        
        t4 = time.perf_counter()
        self._t_merge += (t4 - t3)
        return merged[["qid", "docno", "rank", "score"]]
    
    def get_stats(self):
        return {
            "queries": self._q, "low_activations": self._low_acts,
            "rate": self._low_acts / max(1, self._q),
            "time_coarse_s": self._t_coarse, "time_high_s": self._t_high,
            "time_low_s": self._t_low, "time_merge_s": self._t_merge
        }