"""Ядро UA-Mondrian: валидность, пробные разрезы, обобщение, рекурсия, PostValidate.

Детерминизм: никаких случайных чисел; все tie-break'и — точное сравнение float64
и затем заранее зафиксированный порядок QI (ds.qi).

Числовые кандидаты порога (по протоколу): сортировка по (значение, row_id);
первый кандидат — нижняя медиана x_(n/2); равные значения не разделяются;
следующие кандидаты — середины соседних различных значений в порядке возрастания.
Категориальные кандидаты: ветви обученной на D_train иерархии в лексикографическом
порядке кода ветви. Разрез принимается ТОЛЬКО если оба ребёнка валидны.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

import config as C
from data import ROOT_LABEL, CategoryHierarchy, Dataset

EPS_NORM = 1e-12  # константа нормировки MI из протокола


# ----------------------------------------------------------------------------
# Базовые величины
# ----------------------------------------------------------------------------
def entropy_bits(counts: np.ndarray) -> float:
    """H в битах по счётчикам категорий (float64)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def is_valid_node(b_sa_values: np.ndarray, k: int, ell: int) -> bool:
    """IsValidNode: |P| >= k И Entropy2(P[B_SA]) >= log2(l). Только B_SA."""
    if len(b_sa_values) < k:
        return False
    _, counts = np.unique(b_sa_values, return_counts=True)
    return entropy_bits(counts) >= np.log2(ell) - 1e-12


def mutual_information(a_codes: np.ndarray, y_codes: np.ndarray,
                       n_a: int, n_y: int) -> float:
    """I(A;Y) в битах: sum_{v,y} p(v,y) log2(p(v,y)/(p(v)p(y))).

    Нулевые частоты в log не подставляются (слагаемые с p(v,y)=0 пропускаются).
    """
    n = len(a_codes)
    if n == 0 or n_a <= 1 or n_y <= 1:
        return 0.0
    joint_idx = a_codes.astype(np.int64) * n_y + y_codes.astype(np.int64)
    joint = np.bincount(joint_idx, minlength=n_a * n_y).astype(np.float64)
    joint = joint.reshape(n_a, n_y)
    row = joint.sum(axis=1)   # счётчики p(v) * n
    col = joint.sum(axis=0)   # счётчики p(y) * n
    nz = joint > 0
    if not nz.any():
        return 0.0
    j = joint[nz]
    rv = np.repeat(row, n_y).reshape(n_a, n_y)[nz]
    cy = np.tile(col, n_a).reshape(n_a, n_y)[nz]
    terms = (j / n) * np.log2((j * n) / (rv * cy))
    return float(terms.sum())


def norm_entropy(values: np.ndarray) -> float:
    """Dist(A_j) = H_norm(A_j); при |V_Aj| <= 1 -> 0."""
    uniq = np.unique(values)
    if len(uniq) <= 1:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return entropy_bits(counts) / np.log2(len(uniq))


def cat_lca(chains: list[list[str]]) -> str:
    """Наименьший общий узел цепочек (лист -> ... -> ROOT): самый глубокий общий."""
    sets = [set(ch) for ch in chains]
    best, best_depth = ROOT_LABEL, -1
    for node in sets[0]:
        depth = len(chains[0]) - 1 - chains[0].index(node)
        if depth > best_depth and all(node in s for s in sets[1:]):
            best, best_depth = node, depth
    return best


# ----------------------------------------------------------------------------
# Подготовленные данные D_train
# ----------------------------------------------------------------------------
@dataclass
class PreparedData:
    """Предвычисленные массивы; используются только train-позиции."""
    ds: Dataset
    y: np.ndarray                      # int64 коды Y по ds.labels
    b_sa: np.ndarray                   # int64 коды B_SA
    qi_order: list[str]                # зафиксированный порядок QI
    numeric: dict[str, np.ndarray]     # attr -> float64 значения всех строк df
    cat_codes: dict[str, np.ndarray]   # attr -> int64 коды значений всех строк df
    cat_uniq: dict[str, list[str]]     # attr -> строковое значение кода
    hier: dict[str, CategoryHierarchy]
    train_ranges: dict[str, tuple[float, float]]
    dist: dict[str, float]             # Dist(A_j) на D_train
    w_sens: dict[str, float]           # w_sens(A_j) = beta*Dist + (1-beta)*R_rep
    cat_chain_by_code: dict[str, dict[int, list[str]]]  # attr: код -> цепочка (train)
    branch_codes: dict[str, dict[str, np.ndarray]]      # attr: ветвь -> bool mask по кодам


def prepare(ds: Dataset, train_pos: np.ndarray, beta: float) -> PreparedData:
    y_map = {lab: i for i, lab in enumerate(ds.labels)}
    y = np.array([y_map[v] for v in ds.df[ds.y_col].to_numpy(dtype=object)], dtype=np.int64)
    b_sa_codes = np.unique(ds.b_sa, return_inverse=True)[1].astype(np.int64)
    numeric, cat_codes, cat_uniq, hier = {}, {}, {}, {}
    chain_by_code, branch_codes = {}, {}
    train_ranges, dist, w_sens = {}, {}, {}
    cat_attrs = set(ds.categorical_qi)
    for attr in ds.qi:
        vals = ds.df[attr].to_numpy()
        if attr in cat_attrs:
            uniq, codes = np.unique(vals.astype(object), return_inverse=True)
            uniq_str = [str(v) for v in uniq]
            cat_codes[attr] = codes.astype(np.int64)
            cat_uniq[attr] = uniq_str
            h = CategoryHierarchy(attr, ds.hier_spec[attr], np.unique(vals[train_pos].astype(object)))
            hier[attr] = h
            chain_map = {}
            for i, sv in enumerate(uniq_str):
                chain_map[i] = h.chain.get(sv, [ROOT_LABEL])  # unseen (test) -> ROOT
            chain_by_code[attr] = chain_map
            # маска принадлежности кода значению ветви (по train-значениям)
            bm: dict[str, np.ndarray] = {}
            for branch in h.internal_nodes:
                under = h.values_under(branch)
                bm[branch] = np.array([sv in under for sv in uniq_str], dtype=bool)
            branch_codes[attr] = bm
            train_ranges[attr] = (0.0, 1.0)
            dist[attr] = norm_entropy(codes[train_pos])
        else:
            arr = vals.astype(np.float64)
            numeric[attr] = arr
            train_ranges[attr] = (float(arr[train_pos].min()), float(arr[train_pos].max()))
            dist[attr] = norm_entropy(arr[train_pos])
        r = ds.r_rep[attr]
        w_sens[attr] = beta * dist[attr] + (1.0 - beta) * r
    return PreparedData(
        ds=ds, y=y, b_sa=b_sa_codes, qi_order=list(ds.qi),
        numeric=numeric, cat_codes=cat_codes, cat_uniq=cat_uniq, hier=hier,
        train_ranges=train_ranges, dist=dist, w_sens=w_sens,
        cat_chain_by_code=chain_by_code, branch_codes=branch_codes,
    )


# ----------------------------------------------------------------------------
# Листья
# ----------------------------------------------------------------------------
@dataclass
class Leaf:
    rows: np.ndarray                            # row_id строк листа (int64)
    num_ranges: dict[str, tuple[float, float]]  # числовые QI: (lo, hi) домен листа
    cat_labels: dict[str, str]                  # категориальные QI: label иерархии

    def size(self) -> int:
        return int(len(self.rows))


def _inherit_ranges(ranges: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return {a: r for a, r in ranges.items()}


def _root_cat_labels(pd_data: PreparedData, pos: np.ndarray) -> dict[str, str]:
    labels: dict[str, str] = {}
    for attr in pd_data.qi_order:
        if attr in pd_data.cat_codes:
            codes = pd_data.cat_codes[attr][pos]
            chains = [pd_data.cat_chain_by_code[attr][int(c)] for c in np.unique(codes)]
            labels[attr] = cat_lca(chains)
    return labels


def root_leaf_ranges(pd_data: PreparedData, pos: np.ndarray) -> dict[str, tuple[float, float]]:
    """Диапазоны корня: для числовых QI — диапазон D_train (домен узла)."""
    return {a: pd_data.train_ranges[a] for a in pd_data.qi_order
            if a in pd_data.numeric}


# ----------------------------------------------------------------------------
# Кандидаты разрезов (перебор в детерминированном порядке)
# ----------------------------------------------------------------------------
def iter_numeric_splits(pd_data: PreparedData, attr: str, pos: np.ndarray) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    """Пороги: нижняя медиана, затем середины соседних различных значений по возрастанию.

    Порядок строк: (значение, row_id). Разрез: x <= t против x > t.
    Равные значения не разделяются (порог-медиана отправляет все равные влево;
    midpoint лежит строго между различными значениями).
    """
    values = pd_data.numeric[attr]
    v = values[pos]
    order = np.lexsort((pos, v))
    v_sorted = v[order]
    n = len(v_sorted)
    if n == 0:
        return
    cands: list[float] = [float(v_sorted[(n - 1) // 2])]  # нижняя медиана x_(n/2)
    uniq = np.unique(v)
    if len(uniq) > 1:
        for m in (uniq[:-1] + uniq[1:]) / 2.0:
            m = float(m)
            if all(m != c for c in cands):
                cands.append(m)
    for t in cands:
        left_mask = v <= t
        left = pos[left_mask]
        right = pos[~left_mask]
        if len(left) and len(right):
            yield t, left, right


def iter_categorical_splits(pd_data: PreparedData, attr: str, pos: np.ndarray,
                            parent_label: str) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Ветви иерархии, разделяющие значения узла; лексикографический порядок кода ветви."""
    codes = pd_data.cat_codes[attr][pos]
    bm = pd_data.branch_codes[attr]
    hier = pd_data.hier[attr]
    branches = [b for b in sorted(bm.keys())
                if b != parent_label and _is_descendant(hier, b, parent_label)]
    for branch in branches:
        mask_by_code = bm[branch]
        in_branch = mask_by_code[codes]
        if in_branch.all() or not in_branch.any():
            continue  # ветвь не разделяет значения узла на две непустые части
        left = pos[in_branch]
        right = pos[~in_branch]
        if len(left) and len(right):
            yield branch, left, right


def _is_descendant(hier: CategoryHierarchy, node: str, ancestor: str) -> bool:
    """node лежит в поддереве ancestor (включая случай ancestor == ROOT)."""
    if ancestor == ROOT_LABEL:
        return True
    cur = node
    while cur != ROOT_LABEL:
        if cur == ancestor:
            return True
        cur = hier.parent_of(cur)
    return False


# ----------------------------------------------------------------------------
# Score-функции
# ----------------------------------------------------------------------------
def node_mi_scores(pd_data: PreparedData, pos: np.ndarray,
                   attrs: list[str] | None = None) -> dict[str, float]:
    """Локальная I_P(A_j;Y) по атрибутам узла (raw MI, в битах)."""
    out: dict[str, float] = {}
    y = pd_data.y[pos]
    n_y = len(np.unique(y))
    for attr in (attrs or pd_data.qi_order):
        if attr in pd_data.cat_codes:
            codes_all = pd_data.cat_codes[attr][pos]
            uniq, inv = np.unique(codes_all, return_inverse=True)
            out[attr] = mutual_information(inv, y, len(uniq), n_y)
        else:
            vals = pd_data.numeric[attr][pos]
            uniq, inv = np.unique(vals, return_inverse=True)
            out[attr] = mutual_information(inv, y, len(uniq), n_y)
    return out


def normalize_mi(mi: dict[str, float]) -> dict[str, float]:
    """Ĩ = I / (Σ I + 1e-12); при Σ=0 все Ĩ = 0."""
    s = sum(mi.values())
    if s <= 0.0:
        return {a: 0.0 for a in mi}
    return {a: mi[a] / (s + EPS_NORM) for a in mi}


def global_mi_scores(pd_data: PreparedData, train_pos: np.ndarray) -> dict[str, float]:
    """Глобальный вариант: MI однократно на D_train до рекурсии."""
    return node_mi_scores(pd_data, train_pos)


def scores_ua(pd_data: PreparedData, mi_norm: dict[str, float], alpha: float) -> dict[str, float]:
    """Score = alpha*Ĩ - (1-alpha)*w_sens (w_sens зафиксирован на D_train)."""
    return {a: alpha * mi_norm[a] - (1.0 - alpha) * pd_data.w_sens[a] for a in mi_norm}


def scores_ig(mi_raw: dict[str, float]) -> dict[str, float]:
    """B4m: IG без нормировки и без w_sens."""
    return dict(mi_raw)


def sorted_attrs(pd_data: PreparedData, scores: dict[str, float]) -> list[str]:
    """Sort по Score по убыванию, затем зафиксированный порядок QI."""
    idx = {a: i for i, a in enumerate(pd_data.qi_order)}
    return sorted(pd_data.qi_order, key=lambda a: (-scores[a], idx[a]))


# --- фабрики упорядочивания атрибутов ---------------------------------------
def make_ua_order(alpha: float):
    """Score = alpha*Ĩ - (1-alpha)*w_sens; A1: alpha=1; A2: alpha=0."""
    def order(pd_data, pos, mi_raw, mi_norm):
        return sorted_attrs(pd_data, scores_ua(pd_data, mi_norm, alpha))
    return order


def make_ig_order():
    """B4m: упорядочивание по raw IG (без нормировки и весов)."""
    def order(pd_data, pos, mi_raw, mi_norm):
        return sorted_attrs(pd_data, scores_ig(mi_raw))
    return order


def make_b2_order(train_pos: np.ndarray):
    """B2 (Basic Mondrian): атрибут с максимальным нормированным размахом.

    Числовой: (max-min данных узла) / (max-min D_train); категориальный:
    (distinct узла - 1) / (distinct train - 1). Ties -> фиксированный порядок QI.
    """
    def order(pd_data, pos, mi_raw, mi_norm):
        w = {}
        for attr in pd_data.qi_order:
            if attr in pd_data.numeric:
                v = pd_data.numeric[attr][pos]
                lo_t, hi_t = pd_data.train_ranges[attr]
                span = hi_t - lo_t
                w[attr] = (float(v.max() - v.min()) / span) if span > 0 else 0.0
            else:
                codes = pd_data.cat_codes[attr][pos]
                d_node = len(np.unique(codes))
                d_train = pd_data.hier[attr].total_leaves()
                w[attr] = (d_node - 1) / (d_train - 1) if d_train > 1 else 0.0
        return sorted_attrs(pd_data, w)
    return order


# ----------------------------------------------------------------------------
# Рекурсия разбиения: строит дерево (нужно для out-of-sample маршрута D_test)
# ----------------------------------------------------------------------------
@dataclass
class TNode:
    is_leaf: bool
    rows: np.ndarray | None = None          # train row_id (только у листа)
    num_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    cat_labels: dict[str, str] = field(default_factory=dict)
    left: "TNode | None" = None
    right: "TNode | None" = None
    split_attr: str | None = None
    split_kind: str | None = None           # 'num' | 'cat'
    split_value: float | str | None = None  # порог или код ветви


def partition_tree(pd_data: PreparedData, train_pos: np.ndarray, k: int, ell: int,
                   score_order_fn, mi_mode: str) -> TNode | None:
    """Общий каркас разбиения (B2/B4m/A1/A2/UA-Global/UA-Local).

    score_order_fn(pd_data, pos, mi_raw, mi_norm) -> упорядоченный список атрибутов.
    Возвращает корень дерева или None (Infeasible Constraints: невалидный корень).
    Итеративное построение (глубина дерева в худшем случае O(n/k)).
    """
    if not is_valid_node(pd_data.b_sa[train_pos], k, ell):
        return None
    g_raw: dict[str, float] | None = None
    g_norm: dict[str, float] | None = None
    if mi_mode == "global":
        g_raw = node_mi_scores(pd_data, train_pos)
        g_norm = normalize_mi(g_raw)

    row_ids = pd_data.ds.df["row_id"].to_numpy()
    root = TNode(is_leaf=False)
    stack: list[tuple[np.ndarray, dict[str, tuple[float, float]], dict[str, str], TNode]] = [
        (train_pos, root_leaf_ranges(pd_data, train_pos),
         _root_cat_labels(pd_data, train_pos), root)
    ]
    while stack:
        pos, ranges, labels, slot = stack.pop()
        if not is_valid_node(pd_data.b_sa[pos], k, ell):
            raise AssertionError("Internal invariant error: невалидный узел в стеке")
        placed = False
        if len(pos) >= 2 * k:
            if mi_mode == "local":
                mi_raw = node_mi_scores(pd_data, pos)
                mi_norm = normalize_mi(mi_raw)
            else:
                mi_raw, mi_norm = g_raw, g_norm  # type: ignore[assignment]
            order = score_order_fn(pd_data, pos, mi_raw, mi_norm)
            for attr in order:
                if attr in pd_data.numeric:
                    lo, hi = ranges[attr]
                    for t, left, right in iter_numeric_splits(pd_data, attr, pos):
                        if is_valid_node(pd_data.b_sa[left], k, ell) and \
                           is_valid_node(pd_data.b_sa[right], k, ell):
                            r_l = _inherit_ranges(ranges); r_l[attr] = (lo, t)
                            r_r = _inherit_ranges(ranges); r_r[attr] = (t, hi)
                            slot.split_attr, slot.split_kind, slot.split_value = attr, "num", t
                            lc, rc = TNode(is_leaf=False), TNode(is_leaf=False)
                            slot.left, slot.right = lc, rc
                            stack.append((right, r_r, labels, rc))
                            stack.append((left, r_l, labels, lc))
                            placed = True
                            break
                    if placed:
                        break
                else:
                    parent_label = labels[attr]
                    for branch, left, right in iter_categorical_splits(
                            pd_data, attr, pos, parent_label):
                        if is_valid_node(pd_data.b_sa[left], k, ell) and \
                           is_valid_node(pd_data.b_sa[right], k, ell):
                            l_l = dict(labels); l_l[attr] = branch
                            r_chains = [pd_data.cat_chain_by_code[attr][int(c)]
                                        for c in np.unique(pd_data.cat_codes[attr][right])]
                            l_r = dict(labels); l_r[attr] = cat_lca(r_chains)
                            slot.split_attr, slot.split_kind = attr, "cat"
                            slot.split_value = branch
                            lc, rc = TNode(is_leaf=False), TNode(is_leaf=False)
                            slot.left, slot.right = lc, rc
                            stack.append((right, _inherit_ranges(ranges), l_r, rc))
                            stack.append((left, _inherit_ranges(ranges), l_l, lc))
                            placed = True
                            break
                    if placed:
                        break
        if not placed:
            slot.is_leaf = True
            slot.rows = row_ids[pos]
            slot.num_ranges = _inherit_ranges(ranges)
            slot.cat_labels = dict(labels)
    return root


def collect_leaves(root: TNode) -> list[Leaf]:
    """Листья дерева in-order (левое поддерево раньше правого)."""
    out: list[Leaf] = []
    stack: list[TNode] = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            out.append(Leaf(rows=node.rows, num_ranges=node.num_ranges,
                            cat_labels=node.cat_labels))
        else:
            stack.append(node.right)   # type: ignore[arg-type]
            stack.append(node.left)    # type: ignore[arg-type]
    out.sort(key=lambda lf: int(lf.rows.min()))
    return out


def route_row(root: TNode, pd_data: PreparedData, row_pos: int) -> TNode:
    """Маршрут строки (позиция в df) по обученному дереву — только D_test."""
    node = root
    while not node.is_leaf:
        attr = node.split_attr
        assert attr is not None and node.split_kind is not None
        if node.split_kind == "num":
            v = pd_data.numeric[attr][row_pos]
            go_left = v <= float(node.split_value)  # type: ignore[arg-type]
        else:
            sv = pd_data.cat_uniq[attr][int(pd_data.cat_codes[attr][row_pos])]
            under = pd_data.hier[attr].values_under(str(node.split_value))  # type: ignore[arg-type]
            go_left = sv in under  # unseen-значение уходит вправо
        node = node.left if go_left else node.right  # type: ignore[assignment]
    return node


def partition(pd_data: PreparedData, train_pos: np.ndarray, k: int, ell: int,
              score_order_fn, mi_mode: str) -> list[Leaf] | None:
    """Разбиение -> список листьев (in-order). None = Infeasible Constraints."""
    tree = partition_tree(pd_data, train_pos, k, ell, score_order_fn, mi_mode)
    return None if tree is None else collect_leaves(tree)


# ----------------------------------------------------------------------------
# PostValidate — независимая повторная проверка по D_public (без кеша алгоритма)
# ----------------------------------------------------------------------------
def post_validate(pd_data: PreparedData, train_pos: np.ndarray,
                  leaves: list[Leaf], k: int, ell: int) -> tuple[bool, list[str]]:
    """PASS только при всех проверках; пересчёт по записям, не по кешу."""
    problems: list[str] = []
    train_set = pd_data.ds.df["row_id"].to_numpy()[train_pos]
    # 1) покрытие: дизъюнктные листья точно покрывают корень, без дубликатов и потерь
    all_rows = np.concatenate([lf.rows for lf in leaves]) if leaves else np.array([], dtype=np.int64)
    if len(all_rows) != len(np.unique(all_rows)):
        problems.append("дубликаты строк между листьями")
    if len(all_rows) != len(train_set):
        problems.append(f"потеря записей: {len(train_set)} -> {len(all_rows)}")
    if not np.array_equal(np.sort(all_rows), np.sort(train_set)):
        problems.append("множество строк листьев != D_train")
    # 2) каждый лист удовлетворяет IsValidNode по B_SA
    #    row_id в data.py назначается как arange(len(df)), т.е. row_id == позиция строки
    for i, lf in enumerate(leaves):
        pos = lf.rows.astype(np.int64)
        if not is_valid_node(pd_data.b_sa[pos], k, ell):
            problems.append(f"лист {i}: не удовлетворяет IsValidNode "
                            f"(n={lf.size()}, H={entropy_bits(np.bincount(pd_data.b_sa[pos])):.4f})")
        # 3) сохранённые диапазоны воспроизводят обобщение
        for attr, (lo, hi) in lf.num_ranges.items():
            vals = pd_data.numeric[attr][pos]
            if not ((vals >= lo - 1e-9) & (vals <= hi + 1e-9)).all():
                problems.append(f"лист {i}: значение {attr} вне диапазона [{lo},{hi}]")
            lo_t, hi_t = pd_data.train_ranges[attr]
            if lo < lo_t - 1e-9 or hi > hi_t + 1e-9:
                problems.append(f"лист {i}: диапазон {attr} шире диапазона D_train")
        for attr, label in lf.cat_labels.items():
            hier = pd_data.hier[attr]
            under = hier.values_under(label)
            codes = pd_data.cat_codes[attr][pos]
            for c in np.unique(codes):
                sv = pd_data.cat_uniq[attr][int(c)]
                if sv not in hier.parent:
                    continue  # unseen-значения допустимы только в D_ML,test
                if sv not in under:
                    problems.append(f"лист {i}: значение '{sv}' {attr} не покрыто label '{label}'")
    return (len(problems) == 0), problems
