"""ML-оценка: Macro-F1 (primary), Accuracy, ROC-AUC/PR-AUC; NA-правила (протокол 2.5).

labels — заданный протоколом полный домен (Adult ["<=50K",">50K"],
Bank ["no","yes"]); zero_division=0; отсутствующий в test класс сохраняется
в знаменателе Macro-F1. Если AUC не определён -> NA с причиной, не 0.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def evaluate_predictions(y_test: np.ndarray, y_pred: np.ndarray,
                         proba: np.ndarray | None, labels: list[str],
                         positive: str) -> dict:
    """y_test/y_pred — коды классов (индексы labels); proba — (n, n_labels)."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    res: dict = {}
    p, r, f, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(labels))), zero_division=0)
    for i, lab in enumerate(labels):
        res[f"precision[{lab}]"] = float(p[i])
        res[f"recall[{lab}]"] = float(r[i])
        res[f"f1[{lab}]"] = float(f[i])
        res[f"support[{lab}]"] = int(support[i])
    res["macro_f1"] = float(f1_score(y_test, y_pred, labels=list(range(len(labels))),
                                     average="macro", zero_division=0))
    res["accuracy"] = float(accuracy_score(y_test, y_pred))
    # бинарные AUC: positive label задан протоколом
    pos_idx = labels.index(positive)
    classes_in_test = np.unique(y_test)
    if proba is None:
        res["roc_auc"] = np.nan
        res["pr_auc"] = np.nan
        res["auc_note"] = "predict_proba отсутствует"
    elif len(classes_in_test) < 2:
        res["roc_auc"] = np.nan
        res["pr_auc"] = np.nan
        res["auc_note"] = f"в y_test один класс: {labels[classes_in_test[0]]}"
    else:
        res["roc_auc"] = float(roc_auc_score(y_test, proba[:, pos_idx]))
        res["pr_auc"] = float(average_precision_score(y_test, proba[:, pos_idx]))
        res["auc_note"] = ""
    return res


def zero_rule_predict(proba: np.ndarray) -> np.ndarray:
    """B1: всегда мажоритарный класс."""
    return np.full(len(proba), int(np.argmax(proba[0])), dtype=np.int64)
