"""B4 — Mondrian с ранжированием атрибутов по информационному выигрышу
(упрощение InfoGain Mondrian, LeFevre et al., 2008) с общей проверкой энтропийного
ℓ-разнообразия.

Атрибуты узла упорядочиваются по IG_P(A_j) = H_P(Y) − H_P(Y|A_j); порог числового
разреза перебирается от нижней медианы, как в UA-Mondrian, а не выбирается по минимуму
взвешенной энтропии классов, как в исходном InfoGain Mondrian (разд. 2.3 статьи).

Отдельный модуль со своей traversal и собственными структурами: совпадение кода
с каркасом UA (methods_ua/core TrialSplit) исключено по построению. Общие с
протоколом компоненты: корневая проверка k/л по B_SA, иерархии, правило
стоп-условий, post-validation (core.post_validate), suppression=0, без random tie.

Псевдокод (по протоколу):
  1. If not Valid(P,B_SA,k,l): root -> Infeasible; child -> Internal Invariant Error.
  2. If |P| < 2k: Return {GeneralizeNode(P)}.
  3. IG[A_j] = H_P(Y) - H_P(Y|A_j)  (эквивалентно raw I(A_j;Y) в битах).
  4. Sort QI по IG убыв., затем fixed QI order.
  5-10. Перебор разрезов в детерминированном порядке; принимается первый,
        где оба ребёнка непусты и валидны.
  11. Если разрезов нет / у каждого QI одно значение: {GeneralizeNode(P)}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import core
from core import PreparedData, is_valid_node
from data import ROOT_LABEL


@dataclass
class B4Leaf:
    """Лист B4: row_id строк + обобщения (домены)."""
    rows: np.ndarray
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class B4Node:
    """Узел дерева B4 (для out-of-sample маршрута D_test)."""
    leaf: B4Leaf | None = None
    attr: str | None = None
    kind: str | None = None        # 'num' | 'cat'
    value: float | str | None = None
    left: "B4Node | None" = None
    right: "B4Node | None" = None


def _ig_scores(pd_data: PreparedData, pos: np.ndarray) -> dict[str, float]:
    """IG_P(A_j) = H_P(Y) - H_P(Y|A_j) = I_P(A_j;Y); собственная реализация B4."""
    y = pd_data.y[pos]
    _, y_counts = np.unique(y, return_counts=True)
    h_y = core.entropy_bits(y_counts)
    out = {}
    for attr in pd_data.qi_order:
        if attr in pd_data.cat_codes:
            vals = pd_data.cat_codes[attr][pos]
        else:
            vals = pd_data.numeric[attr][pos]
        uniq, inv = np.unique(vals, return_inverse=True)
        h_cond = 0.0
        for c in range(len(uniq)):
            mask = inv == c
            _, cnts = np.unique(y[mask], return_counts=True)
            h_cond += (mask.sum() / len(y)) * core.entropy_bits(cnts)
        out[attr] = h_y - h_cond
    return out


def _order_attrs(pd_data: PreparedData, scores: dict[str, float]) -> list[str]:
    idx = {a: i for i, a in enumerate(pd_data.qi_order)}
    return sorted(pd_data.qi_order, key=lambda a: (-scores[a], idx[a]))


def _numeric_split_points(values: np.ndarray, pos: np.ndarray) -> list[float]:
    """Медиана + midpoints соседних различных значений (правило протокола)."""
    order = np.lexsort((pos, values))
    v_sorted = values[order]
    points = [float(v_sorted[(len(v_sorted) - 1) // 2])]
    uniq = np.unique(values)
    for m in (uniq[:-1] + uniq[1:]) / 2.0:
        m = float(m)
        if all(m != p for p in points):
            points.append(m)
    return points


def _cat_branches(pd_data: PreparedData, attr: str, parent_label: str) -> list[str]:
    hier = pd_data.hier[attr]
    return sorted(
        b for b in pd_data.branch_codes[attr]
        if b != parent_label and _under(hier, b, parent_label)
    )


def _under(hier, node: str, ancestor: str) -> bool:
    if ancestor == ROOT_LABEL:
        return True
    cur = node
    while cur != ROOT_LABEL:
        if cur == ancestor:
            return True
        cur = hier.parent_of(cur)
    return False


def run_b4(pd_data: PreparedData, train_pos: np.ndarray, k: int,
           ell: int) -> B4Node | None:
    """B4. Возвращает корень или None (Infeasible Constraints)."""
    if not is_valid_node(pd_data.b_sa[train_pos], k, ell):
        return None
    root_ranges = {a: pd_data.train_ranges[a] for a in pd_data.qi_order if a in pd_data.numeric}
    root_labels = {}
    for attr in pd_data.qi_order:
        if attr in pd_data.cat_codes:
            codes = pd_data.cat_codes[attr][train_pos]
            chains = [pd_data.cat_chain_by_code[attr][int(c)] for c in np.unique(codes)]
            root_labels[attr] = core.cat_lca(chains)
    root = B4Node()
    stack = [(train_pos, root_ranges, root_labels, root)]
    row_ids = pd_data.ds.df["row_id"].to_numpy()
    while stack:
        pos, ranges, labels, slot = stack.pop()
        if not is_valid_node(pd_data.b_sa[pos], k, ell):
            raise AssertionError("B4 internal invariant error")
        placed = False
        if len(pos) >= 2 * k:
            scores = _ig_scores(pd_data, pos)
            for attr in _order_attrs(pd_data, scores):
                if attr in pd_data.cat_codes:
                    parent_label = labels[attr]
                    for branch in _cat_branches(pd_data, attr, parent_label):
                        mask = pd_data.branch_codes[attr][branch][pd_data.cat_codes[attr][pos]]
                        left, right = pos[mask], pos[~mask]
                        if not len(left) or not len(right):
                            continue
                        if is_valid_node(pd_data.b_sa[left], k, ell) and \
                           is_valid_node(pd_data.b_sa[right], k, ell):
                            l_l = dict(labels); l_l[attr] = branch
                            r_chains = [pd_data.cat_chain_by_code[attr][int(c)]
                                        for c in np.unique(pd_data.cat_codes[attr][right])]
                            l_r = dict(labels); l_r[attr] = core.cat_lca(r_chains)
                            slot.attr, slot.kind, slot.value = attr, "cat", branch
                            slot.left, slot.right = B4Node(), B4Node()
                            stack.append((right, dict(ranges), l_r, slot.right))
                            stack.append((left, dict(ranges), l_l, slot.left))
                            placed = True
                            break
                    if placed:
                        break
                else:
                    lo, hi = ranges[attr]
                    values = pd_data.numeric[attr]
                    for t in _numeric_split_points(values[pos], pos):
                        mask = values[pos] <= t
                        left, right = pos[mask], pos[~mask]
                        if not len(left) or not len(right):
                            continue
                        if is_valid_node(pd_data.b_sa[left], k, ell) and \
                           is_valid_node(pd_data.b_sa[right], k, ell):
                            r_l = dict(ranges); r_l[attr] = (lo, t)
                            r_r = dict(ranges); r_r[attr] = (t, hi)
                            slot.attr, slot.kind, slot.value = attr, "num", t
                            slot.left, slot.right = B4Node(), B4Node()
                            stack.append((right, r_r, labels, slot.right))
                            stack.append((left, r_l, labels, slot.left))
                            placed = True
                            break
                    if placed:
                        break
        if not placed:
            slot.leaf = B4Leaf(rows=row_ids[pos], ranges=dict(ranges), labels=dict(labels))
    return root


def b4_leaves(root: B4Node) -> list[B4Leaf]:
    out: list[B4Leaf] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.leaf is not None:
            out.append(node.leaf)
        else:
            assert node.left is not None and node.right is not None
            stack.append(node.right)
            stack.append(node.left)
    out.sort(key=lambda lf: int(lf.rows.min()))
    return out


def b4_leaves_core(root: B4Node) -> list:
    """Листья B4 в общем типе core.Leaf (для post-validate/метрик/D_ML)."""
    return [core.Leaf(rows=lf.rows, num_ranges=dict(lf.ranges),
                      cat_labels=dict(lf.labels)) for lf in b4_leaves(root)]


def b4_route_core(root: B4Node, pd_data: PreparedData, row_pos: int):
    """Маршрут тест-строки B4 с возвратом общего core.Leaf."""
    lf = b4_route(root, pd_data, row_pos)
    return core.Leaf(rows=lf.rows, num_ranges=dict(lf.ranges),
                     cat_labels=dict(lf.labels))


def b4_route(root: B4Node, pd_data: PreparedData, row_pos: int) -> B4Leaf:
    """Маршрут тест-строки по обученному дереву B4."""
    node = root
    while node.leaf is None:
        attr = node.attr
        assert node.kind is not None
        if node.kind == "num":
            go_left = pd_data.numeric[attr][row_pos] <= float(node.value)  # type: ignore[arg-type]
        else:
            sv = pd_data.cat_uniq[attr][int(pd_data.cat_codes[attr][row_pos])]
            go_left = sv in pd_data.hier[attr].values_under(str(node.value))  # type: ignore[arg-type]
        node = node.left if go_left else node.right  # type: ignore[assignment]
    return node.leaf  # type: ignore[return-value]
