"""Полный эксперимент UA-Mondrian.

Сетка: {Adult SA≠Y, Bank} × 10 сплитов (repeat∈{0,1} × fold∈{1..5}) ×
k∈{5,10,25,50,100} × {B2, B4, B4m, A1, A2, UA-Global, UA-Local} × {RF, XGBoost}
+ Raw/B1 (по одному обучению на сплит, k не влияет; классификатор RF). Дерево
анонимизации строится один раз на (сплит, k, метод) и используется обоими
классификаторами: дерево детерминировано и от классификатора не зависит.

Диагностика корня: Adult SA=Y и CMC A/B — на каждом из 10 сплитов, на D_train.

Журналы — сырые CSV, вручную не редактируются. Строки уровня дерева
(runs_privacy / runs_attacks / feasibility / correctness / root_screen) пишутся
один раз на (сплит, k, метод) с classifier="NA": они от классификатора не зависят.
Строки runs_ml содержат конкретный классификатор.

Возобновляемость: атомарный элемент = (датасет, repeat, fold, k, метод); все его
строки пишутся после полного расчёта; готовые элементы пропускаются по каждому
файлу журнала отдельно, поэтому частично записанные элементы дозаполняются без
дубликатов. Для пересчёта с нуля журналы нужно сначала убрать из logs/.

Вторичные атаки (Journalist J=QI, Journalist r∈{3,5}, шум 5 % QI) — по флагу
--secondary (внутри основного прогона) или --secondary-only (после основного;
деревья пересчитываются).

Запуск (из code/):
    python run_full.py --repeats 0        # сплиты 0–4
    python run_full.py --repeats 1        # сплиты 5–9
    python run_full.py --secondary-only   # вторичные атаки
Фильтры --datasets/--folds/--ks/--methods/--classifiers позволяют запустить часть
сетки. На macOS xgboost требует libomp (см. config.set_xgboost_runtime_env).
"""

from __future__ import annotations

import argparse
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

ELL = C.ELL
ROOT_SCREEN_K = 10  # как в пилоте: |D_train|>=k тривиален, H-порог от k не зависит
SECONDARY = ["journalist", "journalist_r3", "journalist_r5", "noise5"]

LOG = C.LOG_DIR
RES = C.RESULT_DIR


def mem_mb() -> float:
    """Пиковая память процесса. На macOS ru_maxrss в байтах, на Linux — в KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def append_rows(path: Path, rows: list[dict]) -> None:
    """Атомарная дозапись блока строк (заголовок — только если файла нет)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def load_keys(path: Path, cols: list[str]) -> set[tuple]:
    """Множество готовых ключей из CSV (все значения — строки).

    keep_default_na=False: значения вроде "NA" в k не должны превращаться
    в NaN, иначе ключи Raw/B1 (k="NA") не совпадут при возобновлении.
    """
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return set()
    return set(tuple(r) for r in df[cols].itertuples(index=False, name=None))


def evaluate_model(pd_data, x_train, x_test, y_train, y_test, repeat, fold,
                   classifier):
    """Обучение и оценка одного классификатора (seed по правилу сплита)."""
    seed = C.clf_seed(repeat, fold)
    model = preprocess.make_classifier(classifier, seed)
    model.fit(x_train.to_numpy(), y_train)
    y_pred = model.predict(x_test.to_numpy()).astype(np.int64)
    proba = model.predict_proba(x_test.to_numpy())
    return ml_eval.evaluate_predictions(y_test, y_pred, proba,
                                        pd_data.ds.labels, pd_data.ds.positive)


def run_anon_item(pd_data, ds_name: str, mode: str, train_pos, test_pos,
                  repeat: int, fold: int, k: int, method: str,
                  missing_clf: list[str], need_priv: bool, need_att: bool,
                  need_feas: bool, need_corr: bool, missing_sec: list[str]) -> float:
    """Один анонимизированный (сплит, k, метод): дерево -> PV -> privacy ->
    primary-атака -> ML (по классификаторам) -> вторичные атаки -> запись строк.

    Записываются только недостающие группы строк (возобновляемость без
    дубликатов). Возвращает wall-time элемента.
    """
    t0 = time.perf_counter()
    if method == "B4":
        root = methods_b4.run_b4(pd_data, train_pos, k, ELL)
        leaves = [] if root is None else methods_b4.b4_leaves_core(root)
        route_fn = methods_b4.b4_route_core
    else:
        root = methods_ua.run_ua(pd_data, train_pos, k, ELL, method)
        leaves = [] if root is None else methods_ua.leaves_of(root)
        route_fn = core.route_row
    feasible = root is not None
    base = {"dataset": ds_name, "mode": mode, "repeat": repeat, "fold": fold,
            "k": k, "method": method, "classifier": "NA", "ell": ELL,
            "feasible": feasible}
    rows: dict[str, list[dict]] = {"privacy": [], "attacks": [],
                                   "correctness": [], "feasibility": [],
                                   "ml": []}

    if not feasible:
        if need_feas:  # только отсутствующая строка — без дубликатов
            rows["feasibility"].append({
                **base, "planned": 1, "completed": 1, "leaves": 0,
                "min_ec": "NA", "min_h_b_sa": "NA", "suppression": 0,
                "runtime_s": round(time.perf_counter() - t0, 3),
                "mem_mb": round(mem_mb(), 1),
                "note": "Infeasible Constraints (root FAIL)"})
    else:
        bsa_counts = methods_ua.leaf_bsa_counts(pd_data, leaves)
        pv_ok, pv_problems = core.post_validate(pd_data, train_pos, leaves,
                                                k, ELL)
        ctx = None
        if need_priv:
            w_dict = mlib.w_sens_weights(pd_data, beta=0.5)
            loss = mlib.loss_metrics(pd_data, leaves, k, w_dict)
            min_h = min(core.entropy_bits(c) for c in bsa_counts)
            for metric, value in loss.items():
                rows["privacy"].append({**base, "metric": metric, "value": value,
                                        "runtime_s": "", "mem_mb": "",
                                        "note": f"leaves={len(leaves)}"})
            ctx = attacks.AttackContext(pd_data, leaves, bsa_counts)
        if need_att:
            if ctx is None:
                ctx = attacks.AttackContext(pd_data, leaves, bsa_counts)
            atk = ctx.run(list(pd_data.qi_order), train_pos,
                          attacks.scenario_key(attacks.SCENARIO_PRIMARY,
                                               repeat, fold, method, ds_name))
            for metric, value in atk.items():
                rows["attacks"].append({**base, "scenario": "prosecutor",
                                        "metric": metric, "value": value,
                                        "runtime_s": "", "mem_mb": "", "note": ""})
        if need_feas:
            rows["feasibility"].append({
                **base, "planned": 1, "completed": 1, "leaves": len(leaves),
                "min_ec": loss["min_ec"] if need_priv else "NA",
                "min_h_b_sa": float(min(core.entropy_bits(c) for c in bsa_counts)),
                "suppression": 0, "runtime_s": "", "mem_mb": round(mem_mb(), 1),
                "note": ""})
        if need_corr:
            rows["correctness"].append({**base, "postvalidation_pass": pv_ok,
                                        "problems": ";".join(pv_problems)})
        if missing_clf:
            x_train, x_test = preprocess.build_d_ml(pd_data, leaves, root,
                                                    route_fn, train_pos, test_pos)
            t_ml0 = time.perf_counter()
            y_train, y_test = pd_data.y[train_pos], pd_data.y[test_pos]
            for clf in missing_clf:
                ml = evaluate_model(pd_data, x_train, x_test, y_train, y_test,
                                    repeat, fold, clf)
                ml["runtime_s"] = time.perf_counter() - t_ml0
                clf_base = {**base, "classifier": clf}
                for metric, value in ml.items():
                    rows["ml"].append({**clf_base, "metric": metric,
                                       "value": value,
                                       "runtime_s": ml.get("runtime_s", ""),
                                       "mem_mb": "", "note": ""})
        if missing_sec:  # вторичные атаки — после ML-оценки (протокол 2.6)
            if ctx is None:
                ctx = attacks.AttackContext(pd_data, leaves, bsa_counts)
            for scenario in SECONDARY:
                if scenario not in missing_sec:
                    continue
                key = attacks.scenario_key(scenario, repeat, fold, method,
                                           ds_name)
                if scenario == "noise5":
                    J = list(pd_data.qi_order)
                    override, _skipped = attacks.noise_value_override(
                        pd_data, key, J, train_pos)
                    res = ctx.run(J, train_pos, key, value_override=override)
                else:
                    r = {"journalist": None, "journalist_r3": 3,
                         "journalist_r5": 5}[scenario]
                    J = list(pd_data.qi_order) if r is None \
                        else attacks.sort_j_by_rrep(pd_data, r)
                    ext = attacks.journalist_positions(pd_data, key, train_pos,
                                                       test_pos)
                    res = ctx.run(J, ext, key)
                for metric, value in res.items():
                    rows["attacks"].append({**base, "scenario": scenario,
                                            "metric": metric, "value": value,
                                            "runtime_s": "", "mem_mb": "",
                                            "note": ""})

    wall = time.perf_counter() - t0
    for row in rows["feasibility"]:
        if row["runtime_s"] == "":
            row["runtime_s"] = round(wall, 3)
    append_rows(LOG / "runs_privacy.csv", rows["privacy"])
    append_rows(LOG / "runs_attacks.csv", rows["attacks"])
    append_rows(LOG / "correctness.csv", rows["correctness"])
    append_rows(LOG / "feasibility.csv", rows["feasibility"])
    append_rows(LOG / "runs_ml.csv", rows["ml"])
    return wall


def run_ml_only_item(pd_data, ds_name: str, mode: str, train_pos, test_pos,
                     repeat: int, fold: int, method: str) -> float:
    """Raw (B0) / B1 (Zero-Rule): одно обучение на сплит, RF (как в пилоте).

    B1 — явный мажоритарный класс (train_zero_rule), proba — частоты классов
    D_train; численно совпадает с пилотным B1 (RF на константном признаке
    давал те же предсказания и proba).
    """
    t0 = time.perf_counter()
    y_train, y_test = pd_data.y[train_pos], pd_data.y[test_pos]
    if method == "Raw":
        x_train, x_test = preprocess.build_raw(pd_data, train_pos, test_pos)
        ml = evaluate_model(pd_data, x_train, x_test, y_train, y_test,
                            repeat, fold, "RF")
    else:  # B1
        majority, proba_train = preprocess.train_zero_rule(y_train,
                                                           len(pd_data.ds.labels))
        y_pred = np.full(len(y_test), majority, dtype=np.int64)
        proba = np.tile(proba_train, (len(y_test), 1))
        ml = ml_eval.evaluate_predictions(y_test, y_pred, proba,
                                          pd_data.ds.labels, pd_data.ds.positive)
    # Raw/B1: 13 метрик без runtime_s (как в пилоте); время — только в столбце
    base = {"dataset": ds_name, "mode": mode, "repeat": repeat, "fold": fold,
            "k": "NA", "method": method, "classifier": "RF", "ell": ELL,
            "feasible": "NA"}
    rows = [{**base, "metric": m, "value": v, "runtime_s": "",
             "mem_mb": "", "note": ""} for m, v in ml.items()]
    append_rows(LOG / "runs_ml.csv", rows)
    return time.perf_counter() - t0


def write_root_screen(ds_name: str, variant_mode: str, repeat: int, fold: int,
                      b_sa_values: np.ndarray, done: set) -> None:
    """Root-screen PASS/FAIL -> root_screen.csv (один раз на сплит)."""
    key = (ds_name, variant_mode, str(repeat), str(fold))
    if key in done:
        return
    t0 = time.perf_counter()
    res = root_screen.root_screen(b_sa_values, ROOT_SCREEN_K, ELL)
    base = {"dataset": ds_name, "mode": variant_mode, "repeat": repeat,
            "fold": fold, "k": ROOT_SCREEN_K, "method": "root-screen",
            "classifier": "NA", "ell": ELL, "feasible": bool(res["pass"])}
    rows = []
    for metric in ("n_train", "h_b_sa", "h_threshold", "size_ok", "entropy_ok",
                   "pass"):
        note = res.get("reason", "") if metric == "pass" else ""
        rows.append({**base, "metric": metric, "value": res[metric],
                     "runtime_s": round(time.perf_counter() - t0, 3),
                     "mem_mb": round(mem_mb(), 1), "note": note})
    append_rows(LOG / "root_screen.csv", rows)


def main() -> None:
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser(description="Полный эксперимент UA-Mondrian")
    ap.add_argument("--datasets", nargs="+", default=["Adult", "Bank", "CMC"])
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--ks", nargs="+", type=int, default=list(C.K_GRID))
    ap.add_argument("--methods", nargs="+", default=list(C.METHODS_MAIN))
    ap.add_argument("--classifiers", nargs="+", default=["RF", "XGBoost"])
    ap.add_argument("--secondary", action="store_true",
                    help="вторичные атаки внутри основного прогона")
    ap.add_argument("--secondary-only", action="store_true",
                    help="только дозапуск вторичных атак")
    args = ap.parse_args()
    C.set_xgboost_runtime_env()
    LOG.mkdir(exist_ok=True, parents=True)
    RES.mkdir(exist_ok=True, parents=True)

    priv_cols = ["dataset", "mode", "repeat", "fold", "k", "method"]
    ml_cols = ["dataset", "repeat", "fold", "k", "method", "classifier"]
    priv_done = load_keys(LOG / "runs_privacy.csv", priv_cols)
    corr_done = load_keys(LOG / "correctness.csv", priv_cols)
    feas_done = load_keys(LOG / "feasibility.csv", priv_cols)
    ml_done = load_keys(LOG / "runs_ml.csv", ml_cols)
    root_done = load_keys(LOG / "root_screen.csv",
                          ["dataset", "mode", "repeat", "fold"])
    att_primary_done: set = set()
    sec_done: set = set()
    if (LOG / "runs_attacks.csv").exists():
        df = pd.read_csv(LOG / "runs_attacks.csv", dtype=str)
        if not df.empty and "scenario" in df.columns:
            prim = df[df["scenario"] == "prosecutor"]
            att_primary_done = set(
                tuple(r) for r in prim[priv_cols].itertuples(index=False, name=None))
            sec = df[df["scenario"] != "prosecutor"]
            sec_done = set(zip(sec["dataset"], sec["repeat"], sec["fold"],
                               sec["k"], sec["method"], sec["scenario"]))

    for ds_name in args.datasets:
        # ---------- CMC: только root-screen (два варианта) ----------
        if ds_name == "CMC":
            cmc = datalib.load_cmc(C.DATA_DIR)
            y_cmc = cmc[C.CMC_Y].to_numpy(np.int64).astype(int) - 1
            row_id_cmc = cmc["row_id"].to_numpy(np.int64)
            for repeat in args.repeats:
                fold_of = cv.assign_folds(y_cmc, row_id_cmc, repeat)
                for fold in args.folds:
                    train_pos = np.flatnonzero(fold_of != fold)  # D_train (80%)
                    for variant in ("A", "B"):
                        ds_cmc = datalib.cmc_dataset(cmc, variant)
                        bsa = np.asarray(ds_cmc.b_sa)[train_pos]
                        write_root_screen("CMC", f"CMC_{variant}", repeat, fold,
                                          bsa, root_done)
            print(f"[full] CMC root-screen: готово "
                  f"({time.perf_counter() - t_start:.0f}s)", flush=True)
            continue
        if ds_name not in ("Adult", "Bank"):
            raise ValueError(ds_name)

        # ---------- Adult / Bank ----------
        df_raw = (datalib.load_adult(C.DATA_DIR) if ds_name == "Adult"
                  else datalib.load_bank(C.DATA_DIR))
        y_col = C.ADULT_Y if ds_name == "Adult" else C.BANK_Y
        labels = C.ADULT_LABELS if ds_name == "Adult" else C.BANK_LABELS
        lab_idx = {lab: i for i, lab in enumerate(labels)}
        y_codes = np.array([lab_idx[v] for v in df_raw[y_col].to_numpy(dtype=object)],
                           dtype=np.int64)
        row_id = df_raw["row_id"].to_numpy(np.int64)

        for repeat in args.repeats:
            fold_of = cv.assign_folds(y_codes, row_id, repeat)
            for fold in args.folds:
                t_split = time.perf_counter()
                test_pos = np.flatnonzero(fold_of == fold)   # D_test (~20%)
                train_pos = np.flatnonzero(fold_of != fold)  # D_train (~80%)

                # root-screen Adult SA=Y (только диагностика, D_train)
                if ds_name == "Adult":
                    sa_eq_y = datalib.adult_dataset(df_raw, "SA_EQ_Y")
                    write_root_screen("Adult", "SA_EQ_Y", repeat, fold,
                                      np.asarray(sa_eq_y.b_sa)[train_pos],
                                      root_done)
                    ds = datalib.adult_dataset(df_raw, "SA_NEQ_Y")
                else:
                    binner = datalib.QuantileBinner.fit(
                        df_raw["balance"].to_numpy(np.float64)[train_pos])
                    ds = datalib.bank_dataset(df_raw, binner)
                pd_data = core.prepare(ds, train_pos, beta=C.BETA)
                mode = ds.extra.get("mode", "SA_NEQ_Y")

                # --- анонимизированные методы ---
                for k in args.ks:
                    for method in args.methods:
                        pkey = (ds_name, mode, str(repeat), str(fold), str(k),
                                method)
                        need_priv = pkey not in priv_done
                        need_att = pkey not in att_primary_done
                        need_feas = pkey not in feas_done
                        need_corr = pkey not in corr_done
                        missing_clf = [c for c in args.classifiers
                                       if (ds_name, str(repeat), str(fold),
                                           str(k), method, c) not in ml_done]
                        missing_sec = []
                        if args.secondary or args.secondary_only:
                            missing_sec = [sc for sc in SECONDARY
                                           if (ds_name, str(repeat), str(fold),
                                               str(k), method, sc) not in sec_done]
                        if not (need_priv or need_att or need_feas or need_corr
                                or missing_clf or missing_sec):
                            continue
                        wall = run_anon_item(pd_data, ds_name, mode, train_pos,
                                             test_pos, repeat, fold, k, method,
                                             missing_clf, need_priv, need_att,
                                             need_feas, need_corr, missing_sec)
                        print(f"[full] {ds_name} r{repeat} f{fold} k={k} "
                              f"{method}: {wall:.1f}s "
                              f"(priv={int(need_priv)} att={int(need_att)} "
                              f"ml={len(missing_clf)} sec={len(missing_sec)}) "
                              f"[сплит {time.perf_counter() - t_split:.0f}s, "
                              f"всего {time.perf_counter() - t_start:.0f}s]",
                              flush=True)

                # --- Raw / B1: одно обучение на сплит (RF, как в пилоте) ---
                for method in C.METHODS_ML_ONLY:
                    mkey = (ds_name, str(repeat), str(fold), "NA", method, "RF")
                    if mkey in ml_done:
                        continue
                    wall = run_ml_only_item(pd_data, ds_name, mode, train_pos,
                                            test_pos, repeat, fold, method)
                    print(f"[full] {ds_name} r{repeat} f{fold} {method}: "
                          f"{wall:.1f}s [всего "
                          f"{time.perf_counter() - t_start:.0f}s]", flush=True)

    total = time.perf_counter() - t_start
    print(f"[full] ГОТОВО за {total:.1f}s "
          f"({total / 3600:.2f} h), peak_mem={mem_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
