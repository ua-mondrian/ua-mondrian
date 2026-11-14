"""Агрегация сырых журналов полного прогона (run_full.py) -> сводные таблицы + статистика H1.

Вход: logs/*.csv (сырые CSV, вручную не редактируются).
Выход (results/):
  ml_summary.csv       — Mean±SD / Median[IQR] ML-метрик по (dataset, k, method, classifier);
  privacy_summary.csv  — NCP/DM/DM_w/C*_AVG/листья по (dataset, k, method);
  attacks_summary.csv  — Linkage/SAI по (dataset, k, method, scenario);
  h1_pairs.csv         — сырые парные ΔF1 и ΔLink по 20 конфигурациям × 10 сплитов;
  h1_verdicts.csv      — Nadeau–Bengio + Holm + вердикт H1 (ровно 20 конфигураций);
  root_screen_summary.csv — PASS/FAIL диагностики корня по вариантам и сплитам;
  completeness.json    — полнота сетки (ожидается 10 сплитов на ячейку).

Правила протокола: NA не заменяются нулями/средними; вторичные метрики —
только описательно; H1 — единственный подтверждающий тест (2.7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import stats as st

LOG = C.LOG_DIR
RES = C.RESULT_DIR

H1_DATASETS = ["Adult", "Bank"]
H1_CLASSIFIERS = ["RF", "XGBoost"]
SPLITS = [(r, f) for r in C.REPEATS for f in C.FOLDS]  # 10 сплитов, фиксированный порядок

ML_KEYS = ["macro_f1", "accuracy", "roc_auc", "pr_auc"]
PRIV_KEYS = ["ncp", "dm", "dm_w", "c_avg"]
ATT_KEYS = ["linkage_risk", "overall_linkage_risk", "sai_gain",
            "unique_match_rate", "candidate_rate", "homogeneous_rate",
            "majority_accuracy", "mean_group_size", "coverage", "sai_coverage"]


def load_csv(name: str) -> pd.DataFrame:
    path = LOG / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "value" in df.columns:
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def pivot_metric(df: pd.DataFrame, metric: str, extra_group: str | None):
    """value_num одного измерения -> {(dataset, k, method, [classifier]): {split: value}}."""
    if df.empty:
        return {}
    sub = df[df["metric"] == metric]
    group_cols = ["dataset", "k", "method"] + ([extra_group] if extra_group else [])
    out: dict[tuple, dict[tuple, float]] = {}
    for key, g in sub.groupby(group_cols, sort=False):
        per_split = {}
        for _, row in g.iterrows():
            per_split[(int(row["repeat"]), int(row["fold"]))] = row["value_num"]
        out[key] = per_split
    return out


def agg_stats(values: list[float]) -> dict:
    """Mean±SD и Median[IQR] по описательным правилам; NA не заменяются."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan,
                "q25": np.nan, "q75": np.nan}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75))}


def summary_rows(df: pd.DataFrame, metrics: list[str], extra_group: str | None,
                 metric_col: str = "metric") -> list[dict]:
    """Mean±SD/Median[IQR] по каждому (ключ, metric) поверх сплитов."""
    if df.empty:
        return []
    group_cols = ["dataset", "k", "method"] + ([extra_group] if extra_group else [])
    rows = []
    for (key, g) in df.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, key))
        for metric in metrics:
            sub = g[g[metric_col] == metric]
            st_ = agg_stats(sub["value_num"].tolist())
            rows.append({**base, "metric": metric,
                         "n": st_["n"], "mean": st_["mean"], "sd": st_["sd"],
                         "median": st_["median"], "iqr_lo": st_["q25"],
                         "iqr_hi": st_["q75"]})
    return rows


def completeness(ml: pd.DataFrame) -> dict:
    """Полнота сетки: (dataset, k, method, classifier) × 10 сплитов, macro_f1."""
    expected = {}
    for ds in H1_DATASETS:
        for k in C.K_GRID:
            for m in C.METHODS_MAIN:
                for clf in H1_CLASSIFIERS:
                    expected[(ds, str(k), m, clf)] = set(SPLITS)
    have: dict[tuple, set] = {}
    if not ml.empty:
        sub = ml[(ml["metric"] == "macro_f1") & ml["method"].isin(C.METHODS_MAIN)]
        for _, row in sub.iterrows():
            have.setdefault((row["dataset"], row["k"], row["method"],
                             row["classifier"]), set()).add(
                (int(row["repeat"]), int(row["fold"])))
    missing = []
    for key, splits in expected.items():
        got = have.get(key, set())
        miss = sorted(splits - got)
        if miss:
            missing.append({"cell": list(key), "missing_splits": [list(s) for s in miss]})
    raw_b1 = {}
    if not ml.empty:
        for (ds, m), g in ml[ml["method"].isin(C.METHODS_ML_ONLY)].groupby(
                ["dataset", "method"]):
            raw_b1[f"{ds}/{m}"] = len(g[g["metric"] == "macro_f1"])
    return {"expected_cells": len(expected),
            "complete_cells": len(expected) - len(missing),
            "missing": missing, "raw_b1_splits": raw_b1}


def h1_analysis(ml: pd.DataFrame, att: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H1: ровно 20 конфигураций, парные ΔF1, Nadeau–Bengio, Holm, вердикт."""
    f1 = pivot_metric(ml, "macro_f1", "classifier")
    link = pivot_metric(att[att["scenario"] == "prosecutor"] if not att.empty else att,
                        "linkage_risk", None)

    def series(key) -> dict[tuple, float]:
        return f1.get(key, {})

    def link_series(key) -> dict[tuple, float]:
        return link.get(key, {})

    rows, pairs = [], []
    fam: dict[str, float | None] = {}
    stats_by_cfg: dict[str, dict] = {}
    for ds in H1_DATASETS:
        for clf in H1_CLASSIFIERS:
            for k in C.K_GRID:
                cfg = f"{ds}|{clf}|k={k}"
                local_key = (ds, str(k), "UA-Local", clf)
                b4_key = (ds, str(k), "B4", clf)
                s_loc, s_b4 = series(local_key), series(b4_key)
                common = sorted(set(s_loc) & set(s_b4), key=SPLITS.index)
                diffs = [s_loc[s] - s_b4[s] for s in common]
                for s, d in zip(common, diffs):
                    pairs.append({"config": cfg, "repeat": s[0], "fold": s[1],
                                  "delta_f1": d,
                                  "delta_link": link_series((ds, str(k), "UA-Local")).get(s, np.nan)
                                  - link_series((ds, str(k), "B4")).get(s, np.nan)})
                st_ = st.corrected_t(np.array(diffs, dtype=np.float64)) if diffs else \
                    {"n": 0, "mean": np.nan, "s2": np.nan, "se_corr": np.nan,
                     "ci_lo": np.nan, "ci_hi": np.nan, "t_stat": np.nan,
                     "p_one_sided": np.nan, "complete": False}
                stats_by_cfg[cfg] = st_
                fam[cfg] = st_["p_one_sided"] if np.isfinite(st_["p_one_sided"]) else None
    holm = st.holm_stepdown(fam)
    for cfg, st_ in stats_by_cfg.items():
        ds, clf, kv = cfg.split("|")
        k = int(kv.split("=")[1])
        dl = [p["delta_link"] for p in pairs if p["config"] == cfg
              and np.isfinite(p["delta_link"])]
        mean_dl = float(np.mean(dl)) if dl else None
        v = st.h1_verdict(st_, holm[cfg], mean_dl)
        rows.append({"dataset": ds, "classifier": clf, "k": k,
                     "n_complete": st_["n"], "mean_delta_f1": st_["mean"],
                     "s2": st_["s2"], "se_corr": st_["se_corr"],
                     "ci_lo": st_["ci_lo"], "ci_hi": st_["ci_hi"],
                     "t_stat": st_["t_stat"], "p_one_sided": st_["p_one_sided"],
                     "holm_p": holm[cfg], "mean_delta_link": mean_dl,
                     "verdict": v["verdict"], "reasons": ";".join(v["reasons"])})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def main() -> None:
    ml = load_csv("runs_ml.csv")
    priv = load_csv("runs_privacy.csv")
    att = load_csv("runs_attacks.csv")
    feas = load_csv("feasibility.csv")
    root = load_csv("root_screen.csv")

    # --- сводные описательные таблицы ---
    pd.DataFrame(summary_rows(ml, ML_KEYS, "classifier")).to_csv(
        RES / "ml_summary.csv", index=False)
    pd.DataFrame(summary_rows(priv, PRIV_KEYS, None)).to_csv(
        RES / "privacy_summary.csv", index=False)
    pd.DataFrame(summary_rows(att, ATT_KEYS, "scenario")).to_csv(
        RES / "attacks_summary.csv", index=False)

    # --- листья/выполнимость ---
    if not feas.empty:
        f = feas[feas["method"].isin(C.METHODS_MAIN)]
        leaves = f[f["leaves"] != "NA"]["leaves"].astype(float)
        infeasible = f[f["feasible"] == "False"]
        corr = load_csv("correctness.csv")
        pv_fail = 0
        if not corr.empty:
            pv = corr[corr["postvalidation_pass"] == "False"]
            pv_fail = len(pv)
        pd.DataFrame([{
            "n_anon_rows": len(f), "n_infeasible": len(infeasible),
            "min_leaves": float(leaves.min()) if len(leaves) else np.nan,
            "max_leaves": float(leaves.max()) if len(leaves) else np.nan,
            "postvalidate_fail": pv_fail,
        }]).to_csv(RES / "feasibility_overview.csv", index=False)

    # --- root-screen ---
    if not root.empty:
        rows = []
        for (ds, mode), g in root.groupby(["dataset", "mode"], sort=False):
            pass_rows = g[g["metric"] == "pass"]
            h_rows = g[g["metric"] == "h_b_sa"]
            rows.append({
                "dataset": ds, "mode": mode,
                "n_splits": len(pass_rows),
                "n_pass": int((pass_rows["value"] == "True").sum()),
                "h_b_sa_min": float(pd.to_numeric(h_rows["value_num"],
                                                  errors="coerce").min()),
                "h_b_sa_max": float(pd.to_numeric(h_rows["value_num"],
                                                  errors="coerce").max()),
            })
        pd.DataFrame(rows).to_csv(RES / "root_screen_summary.csv", index=False)

    # --- H1 ---
    verdicts, pairs = h1_analysis(ml, att)
    verdicts.to_csv(RES / "h1_verdicts.csv", index=False)
    pairs.to_csv(RES / "h1_pairs.csv", index=False)

    comp = completeness(ml)
    with open(RES / "completeness.json", "w", encoding="utf-8") as f:
        json.dump(comp, f, ensure_ascii=False, indent=2)

    # --- краткий вывод в stdout ---
    print(f"[agg] ячейки сетки: полные {comp['complete_cells']}/"
          f"{comp['expected_cells']}; отсутствующих ячеек: {len(comp['missing'])}")
    if not verdicts.empty:
        conf = verdicts[verdicts["verdict"] == "CONFIRMED"]
        print(f"[agg] H1 CONFIRMED: {len(conf)}/20 конфигураций")
        print(verdicts[["dataset", "classifier", "k", "n_complete",
                        "mean_delta_f1", "holm_p", "mean_delta_link",
                        "verdict"]].to_string(index=False))
    print(f"[agg] готово -> {RES}")


if __name__ == "__main__":
    main()
