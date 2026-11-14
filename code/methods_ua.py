"""Каркас UA-Mondrian: B2, B4m, A1, A2, UA-Global, UA-Local.

B4m/A1/A2/UA различаются только score-функцией упорядочивания атрибутов
и областью MI (local/global); TrialSplit, GeneralizeNode, preprocessing,
логирование и post-validation общие (протокол 2.2).
"""

from __future__ import annotations

import numpy as np

import core
from core import PreparedData, TNode, collect_leaves


def run_ua(pd_data: PreparedData, train_pos: np.ndarray, k: int, ell: int,
           method: str) -> TNode | None:
    """Запуск метода каркаса UA; возвращает корень дерева или None (root FAIL).

    method:
      B2        — разрез по максимальному нормированному размаху;
      B4m       — raw IG в каркасе UA (local);
      A1        — alpha=1 (только нормированная локальная MI);
      A2        — alpha=0 (только w_sens);
      UA-Global — alpha=0.5, MI однократно на D_train;
      UA-Local  — alpha=0.5, MI пересчитывается в каждом узле.
    """
    if method == "B2":
        order_fn = core.make_b2_order(train_pos)
        mi_mode = "local"  # MI не используется; режим не влияет
    elif method == "B4m":
        order_fn = core.make_ig_order()
        mi_mode = "local"
    elif method == "A1":
        order_fn = core.make_ua_order(1.0)
        mi_mode = "local"
    elif method == "A2":
        order_fn = core.make_ua_order(0.0)
        mi_mode = "local"
    elif method == "UA-Global":
        order_fn = core.make_ua_order(0.5)
        mi_mode = "global"
    elif method == "UA-Local":
        order_fn = core.make_ua_order(0.5)
        mi_mode = "local"
    else:
        raise ValueError(method)
    return core.partition_tree(pd_data, train_pos, k, ell, order_fn, mi_mode)


def leaves_of(root: TNode) -> list:
    return collect_leaves(root)


def leaf_bsa_counts(pd_data: PreparedData, leaves: list) -> list[np.ndarray]:
    """bincount B_SA по каждому листу (для атак/PostValidate без пересчёта)."""
    out = []
    n_bsa = int(pd_data.b_sa.max()) + 1
    for lf in leaves:
        pos = lf.rows.astype(np.int64)
        out.append(np.bincount(pd_data.b_sa[pos], minlength=n_bsa).astype(np.float64))
    return out
