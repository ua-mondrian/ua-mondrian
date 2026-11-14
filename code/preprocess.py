"""Препроцессинг: D_public, D_ML (Midpoint + иерархическое кодирование + P_aux), Raw.

Правила протокола 2.4:
  - категории: trim (сделано при загрузке), missing -> __MISSING__,
    unseen test -> __UNKNOWN__ -> корень обученной иерархии;
  - числовой QI: missing -> нижняя медиана D_train (индикатор только в D_ML),
    вне диапазона train — прижать к min/max_train;
  - D_ML: числовые QI* -> Midpoint (a+b)/2, категориальные QI* -> иерархическое
    кодирование label листа; P_aux — исходные значения (internal-only);
  - Y и SA/B_SA — не predictors; D_public и D_ML из одних листьев, но не смешиваются;
  - one-hot handle_unknown='ignore'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from core import Leaf, PreparedData
from data import ROOT_LABEL

MISSING = "__MISSING__"
UNKNOWN = "__UNKNOWN__"


def build_d_public(pd_data: PreparedData, leaves: list[Leaf],
                   bsa_counts: list[np.ndarray] | None = None) -> pd.DataFrame:
    """D_public = (QI*, B_SA) для D_train: по строке — обобщения её листа."""
    records = []
    for lf in leaves:
        for rid in lf.rows:
            rec: dict[str, object] = {"row_id": int(rid)}
            for attr in pd_data.qi_order:
                if attr in pd_data.numeric:
                    lo, hi = lf.num_ranges[attr]
                    rec[f"{attr}__lo"], rec[f"{attr}__hi"] = float(lo), float(hi)
                else:
                    rec[attr] = lf.cat_labels[attr]
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("row_id").reset_index(drop=True)
    pos = df["row_id"].to_numpy().astype(np.int64)
    df["b_sa"] = pd_data.b_sa[pos]
    return df


def _leaf_for_train(pd_data: PreparedData, leaves: list[Leaf]) -> dict[int, Leaf]:
    """row_id -> лист (для D_ML train-строк)."""
    out: dict[int, Leaf] = {}
    for lf in leaves:
        for rid in lf.rows:
            out[int(rid)] = lf
    return out


def _numeric_midpoint(pd_data: PreparedData, attr: str, value: float,
                      leaf_ranges: tuple[float, float]) -> float:
    lo, hi = leaf_ranges
    return (float(lo) + float(hi)) / 2.0


def build_d_ml(pd_data: PreparedData, leaves: list[Leaf], root,
               route_fn, train_pos: np.ndarray,
               test_pos: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D_ML для train (обобщение листа) и test (маршрут по дереву).

    root/route_fn: корень дерева и функция маршрута тест-строки (позиция -> лист).
    Возвращает (X_train, X_test) без Y (target добавляет вызывающий код).
    """
    ds = pd_data.ds
    train_map = _leaf_for_train(pd_data, leaves)
    # нижняя медиана D_train для числовых QI (правило missing)
    lower_median: dict[str, float] = {}
    for attr in pd_data.numeric:
        arr = pd_data.numeric[attr][train_pos]
        lower_median[attr] = float(np.sort(arr)[(len(arr) - 1) // 2])

    def rows_for(pos_array: np.ndarray, is_train: bool) -> list[dict]:
        recs: list[dict] = []
        for p in pos_array:
            rec: dict[str, object] = {"row_id": int(pd_data.ds.df["row_id"].iloc[p])}
            if is_train:
                lf = train_map[int(rec["row_id"])]
            else:
                lf = route_fn(root, pd_data, int(p))
            for attr in pd_data.qi_order:
                if attr in pd_data.numeric:
                    v = pd_data.numeric[attr][p]
                    lo, hi = lf.num_ranges[attr]
                    if np.isnan(v):
                        rec[attr] = lower_median[attr]
                        rec[f"{attr}__missing"] = 1.0  # индикатор только в D_ML
                    else:
                        lo_t, hi_t = pd_data.train_ranges[attr]
                        v = min(max(float(v), lo_t), hi_t)  # прижать к min/max_train
                        rec[attr] = (float(lo) + float(hi)) / 2.0
                else:
                    sv = pd_data.cat_uniq[attr][int(pd_data.cat_codes[attr][p])]
                    label = lf.cat_labels[attr]
                    if sv not in pd_data.hier[attr].parent:
                        label = ROOT_LABEL      # unseen test -> корень иерархии
                    elif sv not in pd_data.hier[attr].values_under(label):
                        label = ROOT_LABEL      # тест-значение вне label листа
                    rec[attr] = label
            for attr in ds.p_aux_num:
                rec[attr] = float(ds.df[attr].iloc[p])
            for attr in ds.p_aux_cat:
                v = ds.df[attr].iloc[p]
                rec[attr] = v if isinstance(v, str) else str(v)
            recs.append(rec)
        return recs

    x_train = pd.DataFrame.from_records(rows_for(train_pos, True))
    x_test = pd.DataFrame.from_records(rows_for(test_pos, False))

    # порядок one-hot категорий фиксируется по D_train
    cat_cols = [a for a in pd_data.qi_order if a in pd_data.cat_codes] + list(ds.p_aux_cat)
    num_cols = [a for a in pd_data.qi_order if a in pd_data.numeric] \
        + [f"{a}__missing" for a in pd_data.numeric] + list(ds.p_aux_num)
    for a in pd_data.numeric:
        col = f"{a}__missing"
        if col not in x_train.columns:
            x_train[col] = 0.0
        if col not in x_test.columns:
            x_test[col] = 0.0
    x_train[num_cols] = x_train[num_cols].astype(np.float64)
    x_test[num_cols] = x_test[num_cols].astype(np.float64)
    train_dummies = pd.get_dummies(x_train[cat_cols].astype(object), prefix_sep="=",
                                   dummy_na=False)
    test_dummies = pd.get_dummies(x_test[cat_cols].astype(object), prefix_sep="=",
                                  dummy_na=False)
    train_dummies, test_dummies = train_dummies.align(test_dummies, join="left",
                                                      axis=1, fill_value=0)
    x_train = pd.concat([x_train[num_cols], train_dummies.astype(np.float64)], axis=1)
    x_test = pd.concat([x_test[num_cols], test_dummies.astype(np.float64)], axis=1)
    return x_train, x_test


def build_raw(pd_data: PreparedData, train_pos: np.ndarray,
              test_pos: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B0 (Raw): исходные QI + P_aux без обобщения (Bank balance отсутствует как SA)."""
    ds = pd_data.ds

    def rows_for(pos_array: np.ndarray) -> list[dict]:
        recs: list[dict] = []
        for p in pos_array:
            rec: dict[str, object] = {}
            for attr in pd_data.qi_order:
                if attr in pd_data.numeric:
                    v = pd_data.numeric[attr][p]
                    if np.isnan(v):
                        arr = pd_data.numeric[attr][train_pos]
                        rec[attr] = float(np.sort(arr)[(len(arr) - 1) // 2])
                        rec[f"{attr}__missing"] = 1.0
                    else:
                        lo_t, hi_t = pd_data.train_ranges[attr]
                        rec[attr] = min(max(float(v), lo_t), hi_t)
                else:
                    rec[attr] = ds.df[attr].iloc[p]
            for attr in ds.p_aux_num:
                rec[attr] = float(ds.df[attr].iloc[p])
            for attr in ds.p_aux_cat:
                rec[attr] = ds.df[attr].iloc[p]
            recs.append(rec)
        return recs

    x_train = pd.DataFrame.from_records(rows_for(train_pos))
    x_test = pd.DataFrame.from_records(rows_for(test_pos))
    cat_cols = [a for a in pd_data.qi_order if a in pd_data.cat_codes] + list(ds.p_aux_cat)
    num_cols = [a for a in pd_data.qi_order if a in pd_data.numeric] \
        + [f"{a}__missing" for a in pd_data.numeric] + list(ds.p_aux_num)
    for a in pd_data.numeric:
        col = f"{a}__missing"
        if col not in x_train.columns:
            x_train[col] = 0.0
        if col not in x_test.columns:
            x_test[col] = 0.0
    x_train[num_cols] = x_train[num_cols].astype(np.float64)
    x_test[num_cols] = x_test[num_cols].astype(np.float64)
    tr_d = pd.get_dummies(x_train[cat_cols].astype(object), prefix_sep="=", dummy_na=False)
    te_d = pd.get_dummies(x_test[cat_cols].astype(object), prefix_sep="=", dummy_na=False)
    tr_d, te_d = tr_d.align(te_d, join="left", axis=1, fill_value=0)
    x_train = pd.concat([x_train[num_cols], tr_d.astype(np.float64)], axis=1)
    x_test = pd.concat([x_test[num_cols], te_d.astype(np.float64)], axis=1)
    return x_train, x_test


# ----------------------------------------------------------------------------
# Модели (гиперпараметры зафиксированы; seed по правилу сплита)
# ----------------------------------------------------------------------------
def make_classifier(classifier: str, seed: int):
    if classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=C.RF_PARAMS["n_estimators"],
                                      random_state=seed,
                                      n_jobs=C.RF_PARAMS["n_jobs"])
    if classifier == "XGBoost":
        C.set_xgboost_runtime_env()
        from xgboost import XGBClassifier
        params = dict(C.XGB_PARAMS)
        params["random_state"] = seed
        return XGBClassifier(**params)
    raise ValueError(classifier)


def train_zero_rule(y_train_codes: np.ndarray, n_labels: int):
    """B1: мажоритарный класс; proba — частоты классов D_train."""
    counts = np.bincount(y_train_codes, minlength=n_labels).astype(np.float64)
    majority = int(np.argmax(counts))  # ties -> меньший код (детерминировано)
    proba = counts / counts.sum()
    return majority, proba
