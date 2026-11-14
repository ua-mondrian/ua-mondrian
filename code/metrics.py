"""Метрики информационных потерь: NCP, DM, DM_w, C*_AVG (протокол 2.5).

Все значения вычисляются по D_public анонимизированных методов на D_train.
Bank balance в m и суммы не входит (balance — не QI). Raw получает NA.
"""

from __future__ import annotations

import numpy as np

from core import Leaf, PreparedData


def leaf_ncp_vector(pd_data: PreparedData, leaf: Leaf,
                    qi_order: list[str]) -> dict[str, float]:
    """NCP_{i,j} одного листа по каждому QI."""
    out: dict[str, float] = {}
    pos = leaf.rows.astype(np.int64)
    for attr in qi_order:
        if attr in pd_data.numeric:
            lo, hi = leaf.num_ranges[attr]
            lo_t, hi_t = pd_data.train_ranges[attr]
            span = hi_t - lo_t
            out[attr] = 0.0 if span <= 0 else float(hi - lo) / float(span)
        else:
            hier = pd_data.hier[attr]
            codes = pd_data.cat_codes[attr][pos]
            distinct = len(np.unique(codes))
            if distinct <= 1:
                out[attr] = 0.0  # все значения листа совпадают
            else:
                label = leaf.cat_labels[attr]
                out[attr] = hier.leaves_below(label) / float(hier.total_leaves())
    return out


def loss_metrics(pd_data: PreparedData, leaves: list[Leaf], k: int,
                 w_dict: dict[str, float]) -> dict:
    """NCP(D*), DM, DM_w, C*_AVG; w_dict — единые веса w_j (beta=0.5, D_train).

    DM_w: q=NCP; Z = sum(q*w); ŵ_i = (sum_j q_ij w_j)/Z при Z>0, иначе 1/M;
    DM_w = sum_i ŵ_i |EC_i|².
    """
    M = len(leaves)
    if M == 0:
        return {"M": 0, "ncp": np.nan, "dm": np.nan, "dm_w": np.nan,
                "c_avg": np.nan, "min_ec": np.nan, "max_ec": np.nan}
    qi_order = pd_data.qi_order
    m = len(qi_order)
    sizes = np.array([lf.size() for lf in leaves], dtype=np.float64)
    q = np.zeros((M, m), dtype=np.float64)
    for i, lf in enumerate(leaves):
        vec = leaf_ncp_vector(pd_data, lf, qi_order)
        q[i] = [vec[a] for a in qi_order]
    w = np.array([w_dict[a] for a in qi_order], dtype=np.float64)
    ncp = float(q.sum() / (M * m))
    dm = float((sizes ** 2).sum())
    weighted = q * w[None, :]
    z = float(weighted.sum())
    if z > 0:
        w_hat = weighted.sum(axis=1) / z
    else:
        w_hat = np.full(M, 1.0 / M)
    dm_w = float((w_hat * sizes ** 2).sum())
    n_total = int(sizes.sum())
    c_avg = (n_total / M) / k
    return {
        "M": M, "m": m, "ncp": ncp, "dm": dm, "dm_w": dm_w,
        "c_avg": float(c_avg), "min_ec": int(sizes.min()), "max_ec": int(sizes.max()),
    }


def manual_dm_w_example() -> dict[str, float]:
    """Ручной пример протокола: обязан воспроизводиться unit-тестом.

    M=2, |EC|=(2,3); два QI, w1=w2=0.5; q1=(0.2,0.4), q2=(0.6,0.0)
    -> L1=L2=0.30, Z=0.60, ŵ=(0.5,0.5); DM=13, DM_w=6.5.
    При Z=0 (все q=0): ŵ=(1/2,1/2), DM_w=6.5.
    """
    sizes = np.array([2.0, 3.0])
    q = np.array([[0.2, 0.4], [0.6, 0.0]])
    w = np.array([0.5, 0.5])
    dm = float((sizes ** 2).sum())
    weighted = q * w[None, :]
    z = weighted.sum()
    w_hat = weighted.sum(axis=1) / z if z > 0 else np.full(2, 1.0 / 2)
    dm_w = float((w_hat * sizes ** 2).sum())
    # вариант Z=0
    q0 = np.zeros_like(q)
    z0 = (q0 * w[None, :]).sum()
    w_hat0 = (q0 * w[None, :]).sum(axis=1) / z0 if z0 > 0 else np.full(2, 1.0 / 2)
    dm_w0 = float((w_hat0 * sizes ** 2).sum())
    return {"dm": dm, "dm_w": dm_w, "z": float(z), "dm_w_zero": dm_w0}


def w_sens_weights(pd_data: PreparedData, beta: float) -> dict[str, float]:
    """Единые для всех методов веса w_j = w_sens(A_j) при beta=0.5 (на D_train)."""
    return {a: beta * pd_data.dist[a] + (1.0 - beta) * pd_data.ds.r_rep[a]
            for a in pd_data.qi_order}
