"""Атака с точной (полуоткрытой) семантикой числового разреза.

Разрез «x <= t | x > t» записывает потомкам диапазоны (lo, t] и (t, hi]. В
attacks.AttackContext обе границы сравниваются как замкнутые (lo <= v <= hi), поэтому
запись со значением t считается совместимой и с правым поддеревом, хотя может
находиться только в левом. Атакующий знает правила обобщения (разд. 2.5 статьи), т. е.
знает, что нижняя граница правого потомка открыта. Здесь она проверяется строго
(lo < v); сценарии, сиды, метрики и выбор кандидатов наследуются от
attacks.AttackContext без изменений.

Результаты исходной (замкнутой) реализации — logs/runs_attacks.csv, исправленной —
logs_posthoc/halfopen/ (run_posthoc_attacks_halfopen.py). В статье используются вторые.
"""
from __future__ import annotations

import numpy as np

import attacks


def lo_open_flags(root, method: str, pd_data, leaves) -> list[dict]:
    """Флаги «нижняя граница открыта» по числовым QI для каждого листа (в порядке leaves).
    Лист сопоставляется по минимальному row_id (листья не пересекаются)."""
    num = list(pd_data.numeric)
    is_b4 = method == "B4"
    by_min: dict[int, dict] = {}
    stack = [(root, {a: False for a in num})]
    while stack:
        node, lo_open = stack.pop()
        leaf = (node.leaf is not None) if is_b4 else node.is_leaf
        if leaf:
            rows = node.leaf.rows if is_b4 else node.rows
            by_min[int(np.min(rows))] = lo_open
            continue
        attr = node.attr if is_b4 else node.split_attr
        kind = node.kind if is_b4 else node.split_kind
        lo_r = dict(lo_open)
        if kind == "num":
            lo_r[attr] = True
        stack.append((node.right, lo_r))
        stack.append((node.left, dict(lo_open)))
    assert len(by_min) == len(leaves), (len(by_min), len(leaves))
    return [by_min[int(np.min(lf.rows))] for lf in leaves]


class AttackContextHalfOpen(attacks.AttackContext):
    def __init__(self, pd_data, leaves, bsa_counts, lo_open: list[dict]):
        super().__init__(pd_data, leaves, bsa_counts)
        self.num_lo_open = {a: np.array([f[a] for f in lo_open], dtype=bool)
                            for a in pd_data.numeric}

    def _coverage_mask(self, p, J, override):
        pd_data = self.pd_data
        mask = np.ones(len(self.leaves), dtype=bool)
        for attr in J:
            if attr in pd_data.numeric:
                if override and (p, attr) in override:
                    v = float(override[(p, attr)])
                else:
                    v = float(pd_data.numeric[attr][p])
                if np.isnan(v):
                    return np.zeros(len(self.leaves), dtype=bool)
                lo = self.num_lo[attr]
                lower = np.where(self.num_lo_open[attr], lo < v, lo <= v)
                mask &= lower & (v <= self.num_hi[attr])
            else:
                if override and (p, attr) in override:
                    sv = str(override[(p, attr)])
                    ci = self.cat_code_index[attr].get(sv)
                    if ci is None:
                        return np.zeros(len(self.leaves), dtype=bool)
                else:
                    ci = int(pd_data.cat_codes[attr][p])
                mask &= self.cat_cover[attr][:, ci]
            if not mask.any():
                break
        return mask


def class_risk(pd_data, train_pos, leaves, prior) -> dict:
    """Классовая (partition) оценка: истинный класс эквивалентности; риск = среднее 1/|EC| = M/n."""
    n = len(train_pos)
    sizes = np.array([len(lf.rows) for lf in leaves], dtype=np.float64)
    n_bsa = len(prior)
    sai = 0.0
    for lf in leaves:
        b = pd_data.b_sa[lf.rows.astype(np.int64)]
        cnt = np.bincount(b, minlength=n_bsa).astype(np.float64)
        ph = cnt[b] / len(b)
        sai += float(np.maximum(0.0, ph - prior[b]).sum())
    assert int(sizes.sum()) == n
    return {"class_risk": len(leaves) / n, "class_sai_gain": sai / n}
