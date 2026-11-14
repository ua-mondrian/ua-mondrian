"""Статистика H1: Nadeau-Bengio corrected resampled t-test + Holm (протокол 2.7).

SE_corr = sqrt(s^2 * (1/10 + 1/4)) = sqrt(s^2 * 0.35); s^2 с делителем 9;
t = (mean - 0.01)/SE_corr, df=9, одностороннее p; двусторонний 95% ДИ.
Holm по ровно 20 конфигурациям; при s^2=0 -> p=NA и H1 не подтверждается.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

import config as C


def corrected_t(diffs: np.ndarray) -> dict:
    """Парные ΔF1 по 10 общим сплитам -> словарь статистик."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    mean = float(diffs.mean())
    s2 = float(diffs.var(ddof=1)) if n > 1 else np.nan  # делитель n-1 (=9 при n=10)
    se = float(np.sqrt(s2 * C.NB_SE_FACTOR)) if np.isfinite(s2) else np.nan
    if not np.isfinite(se) or se <= 0.0:
        return {"n": n, "mean": mean, "s2": s2, "se_corr": se,
                "ci_lo": np.nan, "ci_hi": np.nan, "t_stat": np.nan,
                "p_one_sided": np.nan, "complete": n == C.H1_DF + 1}
    t_stat = (mean - C.H1_DELTA0) / se
    p_one = float(sps.t.sf(t_stat, C.H1_DF))
    t_crit = float(sps.t.ppf(0.975, C.H1_DF))
    return {"n": n, "mean": mean, "s2": s2, "se_corr": se,
            "ci_lo": mean - t_crit * se, "ci_hi": mean + t_crit * se,
            "t_stat": float(t_stat), "p_one_sided": p_one,
            "complete": n == C.H1_DF + 1}


def holm_stepdown(pvals: dict[str, float | None]) -> dict[str, float | None]:
    """Holm (step-down Bonferroni); p=NA (None) учитывается как 1 только для Holm."""
    items = [(k, (1.0 if v is None or not np.isfinite(v) else float(v)))
             for k, v in pvals.items()]
    order = sorted(items, key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (key, p) in enumerate(order):
        adj = min(1.0, max(running_max, (m - i) * p))
        adjusted[key] = adj
        running_max = max(running_max, adj)
    # возвращаем NA там, где исходный p был NA
    return {k: (None if (pvals[k] is None or not np.isfinite(pvals[k])) else adjusted[k])
            for k in pvals}


def h1_verdict(stats_row: dict, holm_p: float | None,
               mean_delta_link: float | None) -> dict:
    """CONFIRMED только при одновременном: 10/10; SE>0; ДИ>0.01; Holm p<0.05;
    mean ΔLink <= 0 (primary)."""
    reasons = []
    ok = True
    if not stats_row.get("complete", False):
        ok, _ = False, reasons.append("N_c < 10 complete splits")
    if not (np.isfinite(stats_row["se_corr"]) and stats_row["se_corr"] > 0):
        ok, _ = False, reasons.append("SE_corr <= 0 или NA")
    if not (np.isfinite(stats_row["ci_lo"]) and stats_row["ci_lo"] > C.H1_DELTA0):
        ok, _ = False, reasons.append("нижняя граница 95% ДИ <= 0.01")
    if holm_p is None:
        ok, _ = False, reasons.append("Holm p NA (учтён как 1)")
    elif not holm_p < C.H1_ALPHA:
        ok, _ = False, reasons.append("Holm p >= 0.05")
    if mean_delta_link is None:
        ok, _ = False, reasons.append("mean ΔLink отсутствует")
    elif not mean_delta_link <= 0:
        ok, _ = False, reasons.append("mean ΔLink > 0")
    return {"verdict": "CONFIRMED" if ok else "NOT CONFIRMED",
            "reasons": reasons}
