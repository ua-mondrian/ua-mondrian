"""Диагностика (не входит в протокол): почему B4 > Raw на Adult/RF и сколько сигнала несёт P_aux.
Те же сплиты, сиды и кодирование, что в run_full.py (build_raw)."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
import config as C, data as datalib, core, cv, preprocess, ml_eval
from sklearn.ensemble import RandomForestClassifier
C.set_xgboost_runtime_env()
out = []
for ds_name in ["Adult", "Bank"]:
    df_raw = datalib.load_adult(C.DATA_DIR) if ds_name == "Adult" else datalib.load_bank(C.DATA_DIR)
    y_col = C.ADULT_Y if ds_name == "Adult" else C.BANK_Y
    labels = C.ADULT_LABELS if ds_name == "Adult" else C.BANK_LABELS
    lab_idx = {l: i for i, l in enumerate(labels)}
    y = np.array([lab_idx[v] for v in df_raw[y_col].to_numpy(dtype=object)], dtype=np.int64)
    row_id = df_raw["row_id"].to_numpy(np.int64)
    for repeat in [0, 1]:
        fold_of = cv.assign_folds(y, row_id, repeat)
        for fold in [1, 2, 3, 4, 5]:
            t0 = time.time()
            te = np.flatnonzero(fold_of == fold); tr = np.flatnonzero(fold_of != fold)
            if ds_name == "Adult":
                ds = datalib.adult_dataset(df_raw, "SA_NEQ_Y")
            else:
                ds = datalib.bank_dataset(df_raw, datalib.QuantileBinner.fit(df_raw["balance"].to_numpy(np.float64)[tr]))
            pdd = core.prepare(ds, tr, beta=C.BETA)
            Xtr, Xte = preprocess.build_raw(pdd, tr, te)
            ytr, yte = pdd.y[tr], pdd.y[te]
            seed = C.clf_seed(repeat, fold)
            aux = set(ds.p_aux_num) | set(ds.p_aux_cat)
            def is_aux(c): return c.split("=")[0] in aux
            cols_all = list(Xtr.columns)
            variants = {
                "Raw_RF": (cols_all, lambda: preprocess.make_classifier("RF", seed)),
                "Raw_XGB": (cols_all, lambda: preprocess.make_classifier("XGBoost", seed)),
                "Raw_RF_msl5": (cols_all, lambda: RandomForestClassifier(200, min_samples_leaf=5, random_state=seed, n_jobs=-1)),
                "Raw_RF_msl20": (cols_all, lambda: RandomForestClassifier(200, min_samples_leaf=20, random_state=seed, n_jobs=-1)),
                "QIonly_RF": ([c for c in cols_all if not is_aux(c)], lambda: preprocess.make_classifier("RF", seed)),
                "PAUXonly_RF": ([c for c in cols_all if is_aux(c)], lambda: preprocess.make_classifier("RF", seed)),
                "PAUXonly_XGB": ([c for c in cols_all if is_aux(c)], lambda: preprocess.make_classifier("XGBoost", seed)),
            }
            if ds_name == "Adult":
                variants["Raw_RF_noEduNum"] = ([c for c in cols_all if c != "education-num"], lambda: preprocess.make_classifier("RF", seed))
            else:
                variants["Raw_RF_noDuration"] = ([c for c in cols_all if c != "duration"], lambda: preprocess.make_classifier("RF", seed))
                variants["Raw_XGB_noDuration"] = ([c for c in cols_all if c != "duration"], lambda: preprocess.make_classifier("XGBoost", seed))
            for name, (cols, mk) in variants.items():
                m = mk(); m.fit(Xtr[cols].to_numpy(), ytr)
                pred = m.predict(Xte[cols].to_numpy()).astype(np.int64); pr = m.predict_proba(Xte[cols].to_numpy())
                r = ml_eval.evaluate_predictions(yte, pred, pr, ds.labels, ds.positive)
                out.append(dict(dataset=ds_name, repeat=repeat, fold=fold, variant=name,
                                macro_f1=r["macro_f1"], roc_auc=r["roc_auc"], pr_auc=r["pr_auc"], accuracy=r["accuracy"]))
            pd.DataFrame(out).to_csv("results_posthoc/diag_raw_regularization.csv", index=False)
            print(ds_name, repeat, fold, f"{time.time()-t0:.0f}s", flush=True)
print("DONE")
