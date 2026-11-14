"""Разведочные расчёты (вне протокола): общие функции для
run_posthoc_alpha.py / run_posthoc_paux.py / run_posthoc_rrep.py.

Используются те же сплиты (cv.assign_folds), сиды классификаторов (C.clf_seed),
core.prepare(ds, train_pos, beta=C.BETA), кодирование D_ML
(preprocess.build_d_ml), гиперпараметры моделей и атака основного сценария, что в
run_full.py. Выходы пишутся в logs_posthoc/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attacks  # noqa: E402
import config as C  # noqa: E402
import core  # noqa: E402
import cv  # noqa: E402
import data as datalib  # noqa: E402
import metrics as mlib  # noqa: E402
import methods_b4  # noqa: E402
import methods_ua  # noqa: E402
import ml_eval  # noqa: E402
import preprocess  # noqa: E402

LOGP = C.CODE_DIR / 'logs_posthoc'
RESP = C.CODE_DIR / 'results_posthoc'
LOGP.mkdir(exist_ok=True)
RESP.mkdir(exist_ok=True)
ELL = C.ELL
ML_METRICS = ('macro_f1', 'accuracy', 'roc_auc', 'pr_auc')
ATT_METRICS = ('linkage_risk', 'sai_gain', 'coverage')


def iter_splits(ds_name: str, repeats=(0, 1), folds=(1, 2, 3, 4, 5), r_rep=None):
    """Генератор (repeat, fold, train_pos, test_pos, pd_data, pd_data_base) — как в run_full.main.
    r_rep: словарь замены R_rep (run_posthoc_rrep.py); pd_data_base — с протокольными R_rep."""
    df_raw = datalib.load_adult(C.DATA_DIR) if ds_name == 'Adult' else datalib.load_bank(C.DATA_DIR)
    y_col = C.ADULT_Y if ds_name == 'Adult' else C.BANK_Y
    labels = C.ADULT_LABELS if ds_name == 'Adult' else C.BANK_LABELS
    lab_idx = {lab: i for i, lab in enumerate(labels)}
    y_codes = np.array([lab_idx[v] for v in df_raw[y_col].to_numpy(dtype=object)], dtype=np.int64)
    row_id = df_raw['row_id'].to_numpy(np.int64)
    for repeat in repeats:
        fold_of = cv.assign_folds(y_codes, row_id, repeat)
        for fold in folds:
            test_pos = np.flatnonzero(fold_of == fold)
            train_pos = np.flatnonzero(fold_of != fold)
            if ds_name == 'Adult':
                ds = datalib.adult_dataset(df_raw, 'SA_NEQ_Y')
            else:
                binner = datalib.QuantileBinner.fit(df_raw['balance'].to_numpy(np.float64)[train_pos])
                ds = datalib.bank_dataset(df_raw, binner)
            base = core.prepare(ds, train_pos, beta=C.BETA)
            if r_rep is None:
                pdd = base
            else:
                import dataclasses
                ds2 = dataclasses.replace(ds, r_rep=dict(r_rep))
                pdd = core.prepare(ds2, train_pos, beta=C.BETA)
            yield repeat, fold, train_pos, test_pos, pdd, base


def build_tree(pd_data, train_pos, k, method: str, alpha: float | None = None):
    """B4 (отдельная реализация) или каркас UA; alpha задаёт UA-Local с другим α."""
    if method == 'B4':
        root = methods_b4.run_b4(pd_data, train_pos, k, ELL)
        leaves = [] if root is None else methods_b4.b4_leaves_core(root)
        return root, leaves, methods_b4.b4_route_core
    if alpha is not None:
        root = core.partition_tree(pd_data, train_pos, k, ELL, core.make_ua_order(alpha), 'local')
    else:
        root = methods_ua.run_ua(pd_data, train_pos, k, ELL, method)
    leaves = [] if root is None else methods_ua.leaves_of(root)
    return root, leaves, core.route_row


def tree_metrics(pd_eval, train_pos, leaves, k, ds_name, repeat, fold, method_key):
    """NCP/DM_w (протокольные веса pd_eval) + PostValidate + primary-атака."""
    bsa = methods_ua.leaf_bsa_counts(pd_eval, leaves)
    pv_ok, _ = core.post_validate(pd_eval, train_pos, leaves, k, ELL)
    loss = mlib.loss_metrics(pd_eval, leaves, k, mlib.w_sens_weights(pd_eval, beta=0.5))
    ctx = attacks.AttackContext(pd_eval, leaves, bsa)
    atk = ctx.run(list(pd_eval.qi_order), train_pos,
                  attacks.scenario_key(attacks.SCENARIO_PRIMARY, repeat, fold, method_key, ds_name))
    out = {'postvalidation_pass': float(pv_ok), 'leaves': float(len(leaves)),
           'ncp': loss['ncp'], 'dm_w': loss['dm_w']}
    out.update({m: atk[m] for m in ATT_METRICS})
    return out


def fit_eval(pd_data, x_train, x_test, train_pos, test_pos, repeat, fold, clf):
    seed = C.clf_seed(repeat, fold)
    model = preprocess.make_classifier(clf, seed)
    model.fit(x_train.to_numpy(), pd_data.y[train_pos])
    y_pred = model.predict(x_test.to_numpy()).astype(np.int64)
    proba = model.predict_proba(x_test.to_numpy())
    r = ml_eval.evaluate_predictions(pd_data.y[test_pos], y_pred, proba, pd_data.ds.labels, pd_data.ds.positive)
    return {m: r[m] for m in ML_METRICS}


def append(path: Path, rows: list[dict]):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)


def done_keys(path: Path, cols):
    if not path.exists():
        return set()
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    return set(tuple(r) for r in d[cols].itertuples(index=False, name=None))


_LOGS = {}
AUC_RTOL = 1e-5
AUC_DEV = [0.0]  # максимум отн. отклонения ROC/PR-AUC в контролях


def _log(name):
    if name not in _LOGS:
        _LOGS[name] = pd.read_csv(C.LOG_DIR / name)
    return _LOGS[name]


def control_check(ds_name, repeat, fold, k, method, tm: dict, ml: dict, rtol=1e-13, atol=1e-15):
    """Сверка воспроизведённого существующего метода с logs/*.csv. Возвращает max rel diff; AssertionError при расхождении."""
    worst = 0.0
    ml_log = _log('runs_ml.csv'); pr = _log('runs_privacy.csv'); at = _log('runs_attacks.csv')
    sel = lambda d: d[(d.dataset == ds_name) & (d.repeat == repeat) & (d.fold == fold) &
                      (d.k.astype(float) == k) & (d.method == method)]
    for clf, vals in ml.items():
        d = sel(ml_log); d = d[d.classifier == clf]
        for m, v in vals.items():
            ref = float(d[d.metric == m].value.iloc[0])
            rel = abs(v - ref) / max(abs(ref), 1e-300)
            if m in ('roc_auc', 'pr_auc'):
                # RF(n_jobs=-1) суммирует proba деревьев в недетерминированном порядке потоков:
                # ранги почти равных proba могут меняться -> AUC отличается на ~1e-7.
                assert rel < AUC_RTOL, (ds_name, k, method, clf, m, v, ref)
                AUC_DEV[0] = max(AUC_DEV[0], rel)
                continue
            assert np.isclose(v, ref, rtol=rtol, atol=atol), (ds_name, k, method, clf, m, v, ref)
            worst = max(worst, rel)
    if tm is None:
        return worst
    d = sel(pr)
    for m in ('ncp', 'dm_w'):
        ref = float(d[d.metric == m].value.iloc[0])
        assert np.isclose(tm[m], ref, rtol=rtol, atol=atol), (m, tm[m], ref)
        worst = max(worst, abs(tm[m] - ref) / max(abs(ref), 1e-300))
    d = sel(at); d = d[d.scenario == 'prosecutor']
    for m in ATT_METRICS:
        ref = float(d[d.metric == m].value.iloc[0])
        assert np.isclose(tm[m], ref, rtol=rtol, atol=atol), (m, tm[m], ref)
        worst = max(worst, abs(tm[m] - ref) / max(abs(ref), 1e-300))
    return worst


def now():
    return time.strftime('%H:%M:%S')
