# ================== Two-tier (come la tua) ==================
import time
import pandas as pd
import numpy as np
import pyterrier as pt

class ColBERTTwoTier(pt.Transformer):
    """
    Two-tier ColBERT with (i) relative/absolute margin gating and
    (ii) lightweight logging of activation rate & timings.

    Parameters
    ----------
    retr_high, retr_low : pt.Transformer
        Pipelines end-to-end già pronte (es. ColBERTFactory(...).end_to_end()).
    mode : str
        'margin' | 'entropy' | 'margin_or_entropy' (default) | 'margin_and_entropy'.
    margin : float
        Soglia di confidenza per il criterio 'margin'.
        - Se margin_mode='absolute': usa gap = s1 - s2 (score assoluti).
        - Se margin_mode='relative': usa rel_gap = (s1 - s2) / (|s1| + eps).
    margin_mode : str
        'absolute' (default) o 'relative'.
    tau : float
        Temperatura per la softmax usata nell'entropia.
    entropy_threshold : float
        Se (0,1] è frazione di entropia massima log(k); se >1 è soglia assoluta.
    topn_entropy : int
        k massimo di documenti (per score) considerati per l'entropia.
    final_topk : int | None
        Taglio finale dopo merge. Se None non taglia.
    log_stats : bool
        Abilita raccolta statistiche aggregate (activation rate, tempi).
    log_per_query : bool
        Se True, salva un record per query con info di gating e tempi.
    time_unit : str
        'ms' (default) o 's' per i tempi riportati in get_stats()/per-query log.
    """

    def __init__(self, retr_high, retr_low, mode="margin_or_entropy",
                 margin=0.05, margin_mode="absolute",
                 tau=1.0, entropy_threshold=0.8, topn_entropy=10, final_topk=None,
                 log_stats=False, log_per_query=False, time_unit="ms"):
        self.retr_high, self.retr_low = retr_high, retr_low
        self.mode, self.margin = mode, float(margin)
        self.margin_mode = str(margin_mode).lower()
        self.tau, self.entropy_threshold = float(tau), float(entropy_threshold)
        self.topn_entropy, self.final_topk = int(topn_entropy), final_topk

        self.log_stats = bool(log_stats)
        self.log_per_query = bool(log_per_query)
        self.time_unit = time_unit if time_unit in ("ms", "s") else "ms"

        self._eps = 1e-12
        self.reset_stats()

    # ---------- Stats helpers ----------
    def reset_stats(self):
        self._q = 0
        self._low_acts = 0
        self._t_high_sum = 0.0
        self._t_low_sum = 0.0
        self._t_merge_sum = 0.0
        self._per_query = []  # opzionale

    def _scale(self, seconds):
        return seconds * (1000.0 if self.time_unit == "ms" else 1.0)

    def get_stats(self):
        if self._q == 0:
            return {
                "queries": 0, "low_activations": 0, "activation_rate": 0.0,
                f"avg_high_time_{self.time_unit}": 0.0,
                f"avg_low_time_{self.time_unit}": 0.0,
                f"avg_merge_time_{self.time_unit}": 0.0,
            }
        return {
            "queries": self._q,
            "low_activations": self._low_acts,
            "activation_rate": self._low_acts / self._q,
            f"avg_high_time_{self.time_unit}": self._scale(self._t_high_sum / self._q),
            f"avg_low_time_{self.time_unit}": self._scale(self._t_low_sum / self._q),
            f"avg_merge_time_{self.time_unit}": self._scale(self._t_merge_sum / self._q),
        }

    def get_per_query_log(self):
        """Ritorna la lista di dict per-query (se log_per_query=True)."""
        return list(self._per_query)

    # ---------- Gating criteria ----------
    def _need_low_margin(self, run_h):
        if len(run_h) < 2:
            return True, {"reason": "few_results"}

        top2 = run_h.nlargest(2, "score")["score"].values
        if len(top2) < 2:
            return True, {"reason": "few_results"}

        s1, s2 = float(top2[0]), float(top2[1])

        if self.margin_mode == "relative":
            rel_gap = (s1 - s2) / (abs(s1) + self._eps)
            need = rel_gap < self.margin
            return need, {"mode": "relative", "s1": s1, "s2": s2, "rel_gap": rel_gap, "threshold": self.margin}
        else:
            gap = s1 - s2
            need = gap < self.margin
            return need, {"mode": "absolute", "s1": s1, "s2": s2, "gap": gap, "threshold": self.margin}

    def _need_low_entropy(self, run_h):
        if len(run_h) == 0:
            return True, {"reason": "no_results", "k": 0, "entropy": None, "threshold": None}

        k = min(self.topn_entropy, len(run_h))
        # usa i top-k per score per stabilità
        s = run_h.nlargest(k, "score")["score"].values.astype(float)

        z = s / max(self.tau, self._eps)
        z = z - np.max(z)  # stabilità numerica
        p = np.exp(z)
        p = p / (p.sum() + self._eps)

        ent = float(-np.sum(p * np.log(p + self._eps)))
        ent_max = float(np.log(k))

        thr = self.entropy_threshold * ent_max if 0 < self.entropy_threshold <= 1 else self.entropy_threshold
        need = ent >= thr
        return need, {"k": k, "entropy": ent, "entropy_max": ent_max, "threshold": thr, "tau": self.tau}

    def _need_low(self, run_h):
        if self.mode == "margin":
            need_m, info_m = self._need_low_margin(run_h)
            return need_m, {"margin": info_m}

        if self.mode == "entropy":
            need_e, info_e = self._need_low_entropy(run_h)
            return need_e, {"entropy": info_e}

        need_m, info_m = self._need_low_margin(run_h)
        need_e, info_e = self._need_low_entropy(run_h)

        if self.mode == "margin_or_entropy":
            return (need_m or need_e), {"margin": info_m, "entropy": info_e}

        # default: margin_and_entropy
        if len(run_h) < 2:
            return True, {"margin": info_m, "entropy": info_e}
        return (need_m and need_e), {"margin": info_m, "entropy": info_e}

    # ---------- Merge & trim ----------
    @staticmethod
    def _merge_and_rerank(run_h, run_l):
        merged = (pd.concat([run_h, run_l], ignore_index=True)
                  if run_l is not None and len(run_l) > 0 else run_h.copy())
        merged = (merged.sort_values(["qid", "score"], ascending=[True, False])
                         .drop_duplicates(subset=["qid", "docno"], keep="first"))
        merged["rank"] = merged.groupby("qid")["score"].rank(ascending=False, method="first").astype(int)
        return merged[["qid", "docno", "rank", "score"]]

    def _maybe_trim(self, df):
        if self.final_topk is None:
            return df
        return (df.sort_values(["qid", "score"], ascending=[True, False])
                  .groupby("qid", group_keys=False).head(self.final_topk))

    # ---------- Main ----------
    def transform(self, topics_df):
        outs = []
        for _, row in topics_df.iterrows():
            q = row.to_frame().T
            qid = str(row.get("qid", ""))

            t0 = time.perf_counter()
            run_h = self.retr_high.transform(q).sort_values(["qid", "rank"])
            t1 = time.perf_counter()

            need_low, gate_info = self._need_low(run_h)
            run_l = None
            t2 = t1
            if need_low:
                run_l = self.retr_low.transform(q)
                t2 = time.perf_counter()

            merged = self._maybe_trim(self._merge_and_rerank(run_h, run_l))
            t3 = time.perf_counter()

            outs.append(merged)

            # ---- logging ----
            if self.log_stats:
                self._q += 1
                if need_low:
                    self._low_acts += 1
                self._t_high_sum += (t1 - t0)
                self._t_low_sum += (t2 - t1) if need_low else 0.0
                self._t_merge_sum += (t3 - t2)

            if self.log_per_query:
                self._per_query.append({
                    "qid": qid,
                    "need_low": bool(need_low),
                    "mode": self.mode,
                    "margin_mode": self.margin_mode,
                    "high_k": int(len(run_h)),
                    "low_k": int(len(run_l)) if run_l is not None else 0,
                    f"t_high_{self.time_unit}": self._scale(t1 - t0),
                    f"t_low_{self.time_unit}": self._scale((t2 - t1) if need_low else 0.0),
                    f"t_merge_{self.time_unit}": self._scale(t3 - t2),
                    **{f"gating_{k}": v for k, v in gate_info.items()},
                })

        return pd.concat(outs, ignore_index=True)