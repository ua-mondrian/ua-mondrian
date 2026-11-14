"""Технический пилот UA-Mondrian перед полным прогоном (проверка реализуемости).

Один сплит (repeat=0, fold=1): Adult SA≠Y + Bank balance-SA-only, k=10, ℓ=2,
RF, alpha=beta=0.5. Методы: B2/B4/B4m/A1/A2/UA-Local + Raw/B1 (ML-ориентиры).
Adult SA=Y и CMC (A/B) — только root-screen. Secondary-атаки: проверка
детерминизма сидов (Journalist r=3, двойной прогон).

Запуск (из code/):  python run_pilot.py
Результаты: logs/*.csv + results/pilot_summary.json

Отличия журналов пилота от run_full.py: строки уровня дерева имеют classifier="RF"
(в run_full.py — "NA"); строки correctness для B4/B4m пилот не записывал (их дописал
run_full.py); диагностика корня для Adult SA=Y выполнялась на части D_test, в
run_full.py — на D_train (итог FAIL одинаков).
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attacks
import config as C
import core
import cv
import data as datalib
import metrics as mlib
import ml_eval
import methods_b4
import methods_ua
import preprocess
import root_screen

K = 10
ELL = 2
REPEAT, FOLD = 0, 1
CLASSIFIER = "RF"
ANON_METHODS = ["B2", "B4", "B4m", "A1", "A2", "UA-Local"]
ML_ONLY = ["Raw", "B1"]

LOG = C.LOG_DIR
RES = C.RESULT_DIR


def mem_mb() -> float:
    """Пиковая память процесса. На macOS ru_maxrss в байтах, на Linux — в KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def append_csv(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def build_dataset(name: str, data_dir, train_pos: np.ndarray):
    """Dataset + (для Bank) B_SA-binner, обученный ТОЛЬКО на D_train."""
    if name == "Adult":
        df = datalib.load_adult(data_dir)
        return df, datalib.adult_dataset(df, "SA_NEQ_Y")
    if name == "Bank":
        df = datalib.load_bank(data_dir)
        binner = datalib.QuantileBinner.fit(df["balance"].to_numpy(np.float64)[train_pos])
        return df, datalib.bank_dataset(df, binner)
    raise ValueError(name)


def stratification(name: str, df: pd.DataFrame) -> np.ndarray:
    """Метки стратификации (Y)."""
    if name == "Adult":
        return df[C.ADULT_Y].to_numpy(dtype=object)
    if name == "Bank":
        return df[C.BANK_Y].to_numpy(dtype=object)
    raise ValueError(name)


def evaluate_ml(pd_data, x_train, x_test, y_train, y_test, repeat, fold,
                classifier):
    """Обучение и оценка одного классификатора (seed по правилу сплита)."""
    seed = C.clf_seed(repeat, fold)
    if classifier == "B1":
        majority, proba_train = preprocess.train_zero_rule(y_train, len(pd_data.ds.labels))
        y_pred = np.full(len(y_test), majority, dtype=np.int64)
        proba = np.tile(proba_train, (len(y_test), 1))
    else:
        model = preprocess.make_classifier(classifier, seed)
        model.fit(x_train.to_numpy(), y_train)
        y_pred = model.predict(x_test.to_numpy()).astype(np.int64)
        proba = model.predict_proba(x_test.to_numpy())
    return ml_eval.evaluate_predictions(y_test, y_pred, proba,
                                        pd_data.ds.labels, pd_data.ds.positive)


def run_anon_method(pd_data, train_pos, test_pos, method, k, ell, repeat, fold,
                    classifier, dataset_name):
    """Полный цикл одного анонимизированного метода на сплите."""
    t0 = time.perf_counter()
    if method == "B4":
        root = methods_b4.run_b4(pd_data, train_pos, k, ell)
        leaves = [] if root is None else methods_b4.b4_leaves_core(root)
        route_fn = methods_b4.b4_route_core
    else:
        root = methods_ua.run_ua(pd_data, train_pos, k, ell, method)
        leaves = [] if root is None else methods_ua.leaves_of(root)
        route_fn = core.route_row
    t_tree = time.perf_counter() - t0
    feasible = root is not None
    row_base = {"dataset": dataset_name, "mode": pd_data.ds.extra.get("mode", "SA_NEQ_Y"),
                "repeat": repeat, "fold": fold, "k": k, "method": method,
                "classifier": classifier}
    out = {"row_base": row_base, "leaves": len(leaves), "feasible": feasible,
           "tree_s": t_tree, "ml": None, "privacy": None, "attacks": None,
           "postvalidation": None, "journal_id": C.JOURNAL_IDS[method]}

    if not feasible:
        out["feasibility_note"] = "Infeasible Constraints (root FAIL)"
        return out

    bsa_counts = methods_ua.leaf_bsa_counts(pd_data, leaves)

    # post-validation (независимая)
    pv_ok, pv_problems = core.post_validate(pd_data, train_pos, leaves, k, ell)
    out["postvalidation"] = {"pass": pv_ok, "problems": pv_problems[:10]}

    # метрики потерь
    w_dict = mlib.w_sens_weights(pd_data, beta=0.5)
    loss = mlib.loss_metrics(pd_data, leaves, k, w_dict)
    out["privacy"] = loss
    min_h = min(core.entropy_bits(c) for c in bsa_counts)

    # D_public (для журнала корректности) и D_ML
    d_public = preprocess.build_d_public(pd_data, leaves, bsa_counts)
    x_train, x_test = preprocess.build_d_ml(pd_data, leaves, root, route_fn,
                                            train_pos, test_pos)
    y = pd_data.y
    y_train, y_test = y[train_pos], y[test_pos]
    t_ml0 = time.perf_counter()
    ml = evaluate_ml(pd_data, x_train, x_test, y_train, y_test, repeat, fold, classifier)
    ml["runtime_s"] = time.perf_counter() - t_ml0
    out["ml"] = ml

    # primary-атака (Prosecutor, noise-free, J=QI, T_ext = D_train, rho=100%)
    ctx = attacks.AttackContext(pd_data, leaves, bsa_counts)
    base_key = attacks.scenario_key(attacks.SCENARIO_PRIMARY, repeat, fold,
                                    method, dataset_name)
    atk = ctx.run(list(pd_data.qi_order), train_pos, base_key)
    out["attacks"] = {attacks.SCENARIO_PRIMARY: atk}
    out["d_public_size"] = int(len(d_public))
    out["min_ec"] = loss["min_ec"]
    out["min_h_b_sa"] = float(min_h)
    out["runtime_s"] = time.perf_counter() - t0
    return out


def run_ml_only(pd_data, train_pos, test_pos, method, repeat, fold, classifier,
                dataset_name):
    """Raw (B0) и B1 (Zero-Rule): только ML utility."""
    t0 = time.perf_counter()
    y = pd_data.y
    if method == "Raw":
        x_train, x_test = preprocess.build_raw(pd_data, train_pos, test_pos)
    else:
        x_train = pd.DataFrame(np.zeros((len(train_pos), 1)))
        x_test = pd.DataFrame(np.zeros((len(test_pos), 1)))
    ml = evaluate_ml(pd_data, x_train, x_test, y[train_pos], y[test_pos],
                     repeat, fold, classifier)
    return {"row_base": {"dataset": dataset_name, "mode": "SA_NEQ_Y",
                         "repeat": repeat, "fold": fold, "k": "NA",
                         "method": method, "classifier": classifier},
            "ml": ml, "runtime_s": time.perf_counter() - t0,
            "journal_id": C.JOURNAL_IDS[method]}


def run_dataset_split(dataset_name: str, data_dir, repeat: int, fold: int,
                      k: int, ell: int, classifier: str, collect_det: bool = False):
    """Все методы на одном сплите одного датасета."""
    # 1) предварительный train_pos: Bank требует binner на train
    df_raw = (datalib.load_adult(data_dir) if dataset_name == "Adult"
              else datalib.load_bank(data_dir))
    if dataset_name == "Adult":
        y_str = df_raw[C.ADULT_Y].to_numpy(dtype=object)
        y_codes = np.array([{lab: i for i, lab in enumerate(C.ADULT_LABELS)}[v]
                            for v in y_str], dtype=np.int64)
    else:
        y_str = df_raw[C.BANK_Y].to_numpy(dtype=object)
        y_codes = np.array([{lab: i for i, lab in enumerate(C.BANK_LABELS)}[v]
                            for v in y_str], dtype=np.int64)
    row_id = df_raw["row_id"].to_numpy(np.int64)
    fold_of = cv.assign_folds(y_codes, row_id, repeat)
    # протокол: D_test = {row_id: fold_id == fold}; D_train — остальные
    test_pos = np.flatnonzero(fold_of == fold)       # D_test (~20%)
    train_pos = np.flatnonzero(fold_of != fold)      # D_train (~80%)

    df, ds = build_dataset(dataset_name, data_dir, train_pos)
    pd_data = core.prepare(ds, train_pos, beta=C.BETA)

    results = {"dataset": dataset_name, "repeat": repeat, "fold": fold,
               "n_train": int(len(train_pos)), "n_test": int(len(test_pos)),
               "methods": {}}
    for method in ANON_METHODS:
        results["methods"][method] = run_anon_method(
            pd_data, train_pos, test_pos, method, k, ell, repeat, fold,
            classifier, dataset_name)
    for method in ML_ONLY:
        results["methods"][method] = run_ml_only(
            pd_data, train_pos, test_pos, method, repeat, fold, classifier,
            dataset_name)
    # secondary: детерминизм сидов — Journalist r=3 на UA-Local (двойной прогон)
    if collect_det:
        method = "UA-Local"
        root = methods_ua.run_ua(pd_data, train_pos, k, ell, method)
        leaves = methods_ua.leaves_of(root)
        bsa_counts = methods_ua.leaf_bsa_counts(pd_data, leaves)
        ctx = attacks.AttackContext(pd_data, leaves, bsa_counts)
        j_pos = attacks.journalist_positions(
            pd_data, attacks.scenario_key("journalist_r3", repeat, fold,
                                          method, dataset_name),
            train_pos, test_pos)
        J3 = attacks.sort_j_by_rrep(pd_data, 3)
        r1 = ctx.run(J3, j_pos, attacks.scenario_key("journalist_r3", repeat, fold,
                                                     method, dataset_name))
        r2 = ctx.run(J3, j_pos, attacks.scenario_key("journalist_r3", repeat, fold,
                                                     method, dataset_name))
        results["det_journalist_r3"] = {
            "J": J3, "twice_identical": json.dumps(r1, sort_keys=True, default=str)
            == json.dumps(r2, sort_keys=True, default=str),
            "linkage_risk": r1["linkage_risk"], "coverage": r1["coverage"],
        }
    return results, pd_data, train_pos, test_pos


def write_logs(results: dict, ell: int) -> None:
    """CSV-логи: runs_ml / runs_privacy / runs_attacks / feasibility / correctness."""
    for name, res in results["methods"].items():
        base = res["row_base"]
        common = {**base, "ell": ell, "feasible": res.get("feasible", "NA")}
        if res.get("ml"):
            for metric, value in res["ml"].items():
                append_csv(LOG / "runs_ml.csv", {**common, "metric": metric,
                                                 "value": value,
                                                 "runtime_s": res["ml"].get("runtime_s", ""),
                                                 "mem_mb": "", "note": ""})
        if res.get("privacy"):
            for metric, value in res["privacy"].items():
                append_csv(LOG / "runs_privacy.csv", {**common, "metric": metric,
                                                      "value": value,
                                                      "runtime_s": round(res["runtime_s"], 3),
                                                      "mem_mb": round(mem_mb(), 1),
                                                      "note": f"leaves={res['leaves']}"})
        if res.get("attacks"):
            for scenario, atk in res["attacks"].items():
                for metric, value in atk.items():
                    append_csv(LOG / "runs_attacks.csv", {**common, "scenario": scenario,
                                                          "metric": metric, "value": value,
                                                          "runtime_s": "", "mem_mb": "",
                                                          "note": ""})
        if "B4" not in name and name not in ("Raw", "B1"):
            append_csv(LOG / "correctness.csv", {
                **common, "postvalidation_pass": res["postvalidation"]["pass"],
                "problems": ";".join(res["postvalidation"]["problems"])})
        append_csv(LOG / "feasibility.csv", {
            **common, "ell": ell, "planned": 1, "completed": 1,
            "leaves": res.get("leaves", "NA"), "min_ec": res.get("min_ec", "NA"),
            "min_h_b_sa": res.get("min_h_b_sa", "NA"), "suppression": 0,
            "runtime_s": round(res.get("runtime_s", res.get("tree_s", 0.0)), 3),
            "mem_mb": round(mem_mb(), 1),
            "note": res.get("feasibility_note", "")})


def main() -> None:
    t_start = time.perf_counter()
    C.LOG_DIR.mkdir(exist_ok=True, parents=True)
    C.RESULT_DIR.mkdir(exist_ok=True, parents=True)
    summary: dict = {"config": {"k": K, "ell": ELL, "classifier": CLASSIFIER,
                                "repeat": REPEAT, "fold": FOLD,
                                "alpha": C.ALPHA, "beta": C.BETA}}

    # --- основной сплит: Adult SA!=Y + Bank, детерминизм-проверка ---
    for dataset_name in ("Adult", "Bank"):
        t0 = time.perf_counter()
        res, pd_data, train_pos, test_pos = run_dataset_split(
            dataset_name, C.DATA_DIR, REPEAT, FOLD, K, ELL, CLASSIFIER,
            collect_det=True)
        res["wall_s"] = time.perf_counter() - t0
        res["dropped_missing"] = int(pd_data.ds.df.attrs.get("dropped_missing", 0))
        summary[dataset_name] = res
        write_logs(res, ELL)
        print(f"[pilot] {dataset_name}: {res['wall_s']:.1f}s, "
              f"train={res['n_train']}, test={res['n_test']}", flush=True)

    # --- воспроизводимость: повторный прогон тех же вычислений ---
    res2, _, _, _ = run_dataset_split(
        "Bank", C.DATA_DIR, REPEAT, FOLD, K, ELL, CLASSIFIER, collect_det=False)
    same = json.dumps(res2["methods"]["UA-Local"]["privacy"], sort_keys=True) == \
        json.dumps(summary["Bank"]["methods"]["UA-Local"]["privacy"], sort_keys=True)
    same_ml = json.dumps(res2["methods"]["UA-Local"]["ml"], sort_keys=True, default=str) == \
        json.dumps(summary["Bank"]["methods"]["UA-Local"]["ml"], sort_keys=True, default=str)
    summary["rerun_reproducibility"] = {
        "privacy_identical": same, "ml_identical": same_ml}

    # --- root-screen: Adult SA=Y + CMC A/B (на том же сплите) ---
    adult = datalib.load_adult(C.DATA_DIR)
    y_codes_adult = np.array([{lab: i for i, lab in enumerate(C.ADULT_LABELS)}[v]
                              for v in adult[C.ADULT_Y].to_numpy(dtype=object)])
    fold_of = cv.assign_folds(y_codes_adult, adult["row_id"].to_numpy(np.int64), REPEAT)
    train_pos = np.flatnonzero(fold_of == FOLD)
    sa_eq_y = datalib.adult_dataset(adult, "SA_EQ_Y")
    rs_adult = root_screen.root_screen(np.asarray(sa_eq_y.b_sa)[train_pos], K, ELL)
    summary["root_screen_Adult_SA_EQ_Y"] = rs_adult

    cmc = datalib.load_cmc(C.DATA_DIR)
    y_cmc = cmc[C.CMC_Y].to_numpy(np.int64).astype(int) - 1  # 0..2 коды страт
    fold_cmc = cv.assign_folds(y_cmc, cmc["row_id"].to_numpy(np.int64), REPEAT)
    train_cmc = np.flatnonzero(fold_cmc != FOLD)  # D_train: всё, что не тест
    for variant in ("A", "B"):
        ds_cmc = datalib.cmc_dataset(cmc, variant)
        bsa = np.asarray(ds_cmc.b_sa)[train_cmc]
        summary[f"root_screen_CMC_{variant}"] = root_screen.root_screen(bsa, K, ELL)

    # --- повтор: repeat=1 даёт другой сплит (проверка правила сидов) ---
    f0 = cv.assign_folds(y_codes_adult, adult["row_id"].to_numpy(np.int64), 0)
    f1 = cv.assign_folds(y_codes_adult, adult["row_id"].to_numpy(np.int64), 1)
    summary["split_seeds_differ"] = bool((f0 != f1).mean() > 0.5)

    # --- ручной пример DM_w (обязателен протоколом) ---
    summary["manual_dm_w"] = mlib.manual_dm_w_example()

    summary["total_wall_s"] = time.perf_counter() - t_start
    summary["peak_mem_mb"] = round(mem_mb(), 1)
    out_path = RES / "pilot_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[pilot] готово за {summary['total_wall_s']:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
