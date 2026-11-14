"""Назначение фолдов: Repeated Stratified 5-fold CV x2 строго по протоколу.

Правило (протокол, раздел «Протокол исключения утечки информации»):
  seed разбиения = 20261029 + 1000*repeat + fold;
  fold assignment для каждого repeat независимо внутри каждой страты Y:
  (1) отсортировать строки страты по row_id;
  (2) для fold=1 инициализировать numpy.random.Generator(PCG64(seed)) и вызвать
      permutation ОДИН раз для всей страты;
  (3) fold_id = stratum_position mod 5 + 1, где stratum_position — позиция
      row_id в этой перестановке (инверсия перестановки);
  для fold=2..5 используется сохранённое отображение row_id -> fold.
D_test = {row_id: fold_id == fold}.
"""

from __future__ import annotations

import numpy as np

import config as C


def assign_folds(y: np.ndarray, row_id: np.ndarray, repeat: int) -> np.ndarray:
    """fold_id (1..5) для каждой строки; y — стратификационные метки."""
    y = np.asarray(y)
    row_id = np.asarray(row_id)
    n = len(y)
    fold_of = np.zeros(n, dtype=np.int64)
    seed = C.split_seed(repeat, 1)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        idx = idx[np.argsort(row_id[idx], kind="stable")]  # сортировка по row_id
        rng = np.random.Generator(np.random.PCG64(seed))
        perm = rng.permutation(len(idx))  # ровно один вызов на всю страту
        positions = np.empty(len(idx), dtype=np.int64)
        positions[perm] = np.arange(len(idx))  # инверсия: позиция элемента в perm
        fold_of[idx] = positions % C.FOLDS[-1] + 1
    return fold_of


def train_test_masks(fold_of: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """D_test = {fold_id == fold}; D_train — остальные строки."""
    test_mask = fold_of == fold
    return ~test_mask, test_mask
