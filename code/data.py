"""Загрузка наборов данных, роли атрибутов, B_SA и иерархии категорий.

Все преобразования обучаются строго на D_train; D_test не участвует ни в
построении иерархий, ни в квантовании B_SA, ни в выборе порогов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------
# Иерархия категориального атрибута
# --------------------------------------------------------------------------
ROOT_LABEL = "__ROOT__"


class CategoryHierarchy:
    """Детерминированное дерево обобщения, обученное на D_train.

    Узел адресуется полным путём (например ``workclass/private_self``);
    листья — исходные значения атрибута. Обобщение поднимает значение на
    один уровень вверх по своей ветви, самый общий узел — ``__ROOT__``.
    """

    def __init__(self, attr: str, spec, train_values):
        self.attr = attr
        self.parent: dict[str, str] = {ROOT_LABEL: ROOT_LABEL}
        self.depth: dict[str, int] = {ROOT_LABEL: 0}
        self.order: dict[str, int] = {ROOT_LABEL: 0}
        self.children: dict[str, list[str]] = {ROOT_LABEL: []}
        self._counter = 0
        for group in spec:
            if isinstance(group, tuple):
                self._add_group(ROOT_LABEL, group)   # (имя, [значения/подгруппы])
            else:
                for g in group:                      # список групп
                    self._add_group(ROOT_LABEL, g)

        present = set(train_values)
        covered: set[str] = set()
        for value in sorted(present):
            if value not in self.parent:
                self._attach_value(value, ROOT_LABEL)
            covered.add(value)
        self.train_values = sorted(present)
        self.covered_values = sorted(covered)
        # цепочка «лист → корень» для каждого значения
        self.chain: dict[str, list[str]] = {v: self.ancestors(v) for v in self.train_values}
        # лексикографически упорядоченный список внутренних узлов (pre-order)
        self.internal_nodes = [
            lab for lab in self.parent
            if self.children.get(lab) and lab != ROOT_LABEL
        ]
        self.internal_nodes.sort(key=lambda lab: self.order[lab])

    # -- построение --------------------------------------------------------
    def _new_label(self, parent: str, name: str) -> str:
        label = name if parent == ROOT_LABEL else f"{parent}/{name}"
        self._counter += 1
        self.parent[label] = parent
        self.depth[label] = self.depth[parent] + 1
        self.order[label] = self._counter
        self.children[label] = []
        self.children[parent].append(label)
        return label

    def _add_group(self, parent: str, group) -> None:
        name, members = group
        label = self._new_label(parent, name)
        for member in members:
            if isinstance(member, tuple):
                self._add_group(label, member)
            else:
                self._attach_value(member, label)

    def _attach_value(self, value: str, parent: str) -> None:
        if value in self.parent:  # значение уже объявлено в спецификации
            return
        self.parent[value] = parent
        self.depth[value] = self.depth[parent] + 1
        self.order[value] = len(self.order) + 1
        self.children[value] = []
        self.children[parent].append(value)

    # -- запросы -----------------------------------------------------------
    def ancestors(self, value: str) -> list[str]:
        chain = [value]
        node = value
        while node != ROOT_LABEL:
            node = self.parent[node]
            chain.append(node)
        return chain

    def parent_of(self, label: str) -> str:
        return self.parent.get(label, ROOT_LABEL)

    def is_ancestor_or_self(self, label: str, value: str) -> bool:
        """Лежит ли исходное значение value внутри обобщения label."""
        if value not in self.parent:
            return False
        node = value
        while True:
            if node == label:
                return True
            if node == ROOT_LABEL:
                return False
            node = self.parent[node]

    def map_test_value(self, value: str) -> str:
        """Неизвестное тестовое значение → корень обученной иерархии."""
        return value if value in self.parent else ROOT_LABEL

    def candidates(self) -> list[str]:
        return list(self.internal_nodes)

    def values_under(self, label: str) -> set[str]:
        """Исходные train-значения, покрытые узлом label (обученная иерархия)."""
        return {v for v in self.train_values
                if label in self.chain.get(v, [ROOT_LABEL])}

    def leaves_below(self, label: str) -> int:
        """Число train-листьев иерархии под label (для NCP категорий)."""
        return len(self.values_under(label))

    def total_leaves(self) -> int:
        """|V_train| — общее число train-значений."""
        return len(self.train_values)


# --------------------------------------------------------------------------
# Подготовленный датасет
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    df: pd.DataFrame
    y_col: str
    labels: list[str]
    positive: str
    sa_col: str
    b_sa: np.ndarray
    b_sa_kind: str
    numeric_qi: list[str]
    categorical_qi: list[str]
    p_aux_num: list[str]
    p_aux_cat: list[str]
    r_rep: dict[str, float]
    hier_spec: dict
    qi_order: list[str] = field(default_factory=list)  # зафиксированный порядок перебора QI
    extra: dict = field(default_factory=dict)

    @property
    def qi(self) -> list[str]:
        """Зафиксированный порядок QI из протокола (для ties и перебора)."""
        if self.qi_order:
            return list(self.qi_order)
        return list(self.numeric_qi) + list(self.categorical_qi)

    @property
    def n(self) -> int:
        return len(self.df)


def _strip_obj(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def load_adult(data_dir) -> pd.DataFrame:
    frames = []
    for fname, skip in (("adult.data", 0), ("adult.test", 1)):
        path = data_dir / fname
        frame = pd.read_csv(
            path, header=None, names=C.ADULT_COLUMNS, sep=",",
            skipinitialspace=True, skiprows=skip, na_values=["?"],
            keep_default_na=False, dtype=str,
        )
        frame["_source"] = fname
        frame["_sourcerow"] = np.arange(len(frame))
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = _strip_obj(df)
    df["income"] = df["income"].str.rstrip(".")
    # заранее заданное правило очистки: строки с любым «?» удаляются целиком,
    # Y при очистке не используется
    keep = ~df[C.ADULT_COLUMNS].isna().any(axis=1)
    dropped = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    for col in ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"):
        df[col] = df[col].astype(np.float64)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df.attrs["dropped_missing"] = dropped
    df.attrs["raw_rows"] = 48842
    return df


def load_bank(data_dir) -> pd.DataFrame:
    path = data_dir / "bank-full.csv"
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    df = _strip_obj(df)
    for col in ("age", "balance", "day", "duration", "campaign", "pdays", "previous"):
        df[col] = df[col].astype(np.float64)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df.attrs["raw_rows"] = len(df)
    return df


def load_cmc(data_dir) -> pd.DataFrame:
    path = data_dir / "cmc.data"
    df = pd.read_csv(path, header=None, names=C.CMC_COLUMNS, dtype=str, keep_default_na=False)
    df = _strip_obj(df)
    for col in C.CMC_COLUMNS:
        df[col] = df[col].astype(np.float64)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df.attrs["raw_rows"] = len(df)
    return df


# --------------------------------------------------------------------------
# B_SA: квантильные категории balance по D_train
# --------------------------------------------------------------------------
@dataclass
class QuantileBinner:
    edges: np.ndarray          # inner edges
    lo: float
    hi: float

    @classmethod
    def fit(cls, values: np.ndarray, n_bins: int = C.BANK_Q4) -> "QuantileBinner":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(values, qs, method="linear")
        inner = np.unique(edges[1:-1])
        return cls(edges=inner, lo=float(values.min()), hi=float(values.max()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.edges.size == 0:
            return np.zeros(len(values), dtype=np.int64)
        clipped = np.clip(values.astype(np.float64), self.lo, self.hi)
        return np.searchsorted(self.edges, clipped, side="right").astype(np.int64)

    def n_categories(self) -> int:
        return int(self.edges.size) + 1


# --------------------------------------------------------------------------
# Роли признаков по режимам
# --------------------------------------------------------------------------
def adult_dataset(df: pd.DataFrame, mode: str = "SA_NEQ_Y") -> Dataset:
    hier_spec = {a: C.ADULT_HIER[a] for a in C.ADULT_HIER}
    if mode == "SA_NEQ_Y":
        sa_col, b_sa = "occupation", df["occupation"].to_numpy(dtype=object)
        r_rep = {a: C.R_REP_ADULT[a] for a in C.ADULT_QI}
        numeric_qi = list(C.ADULT_NUMERIC)
        categorical_qi = [a for a in C.ADULT_QI if a not in numeric_qi]
        p_aux_num = list(C.ADULT_P_AUX)
        p_aux_cat: list[str] = []
        labels, positive = list(C.ADULT_LABELS), C.ADULT_POSITIVE
    elif mode == "SA_EQ_Y":
        sa_col, b_sa = "income", df["income"].to_numpy(dtype=object)
        r_rep = {a: C.R_REP_ADULT[a] for a in C.ADULT_QI}
        numeric_qi, categorical_qi = [], []
        p_aux_num, p_aux_cat = [], []
        labels, positive = list(C.ADULT_LABELS), C.ADULT_POSITIVE
    else:
        raise ValueError(mode)
    return Dataset(
        name=f"Adult:{mode}", df=df, y_col=C.ADULT_Y, labels=labels, positive=positive,
        sa_col=sa_col, b_sa=b_sa, b_sa_kind="categorical",
        numeric_qi=numeric_qi, categorical_qi=categorical_qi,
        p_aux_num=p_aux_num, p_aux_cat=p_aux_cat, r_rep=r_rep,
        hier_spec=hier_spec, qi_order=list(C.ADULT_QI), extra={"mode": mode},
    )


def bank_dataset(df: pd.DataFrame, binner: QuantileBinner) -> Dataset:
    b_sa = binner.transform(df[C.BANK_SA].to_numpy(dtype=np.float64))
    r_rep = {a: C.R_REP_BANK[a] for a in C.BANK_QI}
    return Dataset(
        name="Bank", df=df, y_col=C.BANK_Y, labels=list(C.BANK_LABELS), positive=C.BANK_POSITIVE,
        sa_col=C.BANK_SA, b_sa=b_sa, b_sa_kind="quantile4",
        numeric_qi=list(C.BANK_NUMERIC),
        categorical_qi=[a for a in C.BANK_QI if a not in C.BANK_NUMERIC],
        p_aux_num=["day", "duration", "campaign", "pdays", "previous"],
        p_aux_cat=["contact", "month", "poutcome"],
        r_rep=r_rep, hier_spec={a: C.BANK_HIER[a] for a in C.BANK_HIER},
        qi_order=list(C.BANK_QI), extra={"binner": binner},
    )


def cmc_dataset(df: pd.DataFrame, variant: str) -> Dataset:
    if variant == "A":
        qi = list(C.CMC_A_QI); sa = C.CMC_A_SA
    elif variant == "B":
        qi = list(C.CMC_B_QI); sa = C.CMC_B_SA
    else:
        raise ValueError(variant)
    return Dataset(
        name=f"CMC:{variant}", df=df, y_col=C.CMC_Y,
        labels=["1", "2", "3"], positive="1", sa_col=sa,
        b_sa=df[sa].astype(np.int64).to_numpy(dtype=object), b_sa_kind="raw",
        numeric_qi=list(qi), categorical_qi=[], p_aux_num=[], p_aux_cat=[],
        r_rep={a: 0.5 for a in qi}, hier_spec={}, qi_order=list(qi),
        extra={"variant": variant},
    )


def build_hierarchies(ds: Dataset, train_idx: np.ndarray) -> dict[str, CategoryHierarchy]:
    """Иерархии обучаются только на D_train."""
    out: dict[str, CategoryHierarchy] = {}
    for attr in ds.categorical_qi:
        values = np.unique(ds.df[attr].to_numpy(dtype=object)[train_idx])
        out[attr] = CategoryHierarchy(attr, ds.hier_spec[attr], values)
    return out


def check_hierarchy_coverage(ds: Dataset, hiers: dict[str, CategoryHierarchy]) -> list[str]:
    problems = []
    for attr, hier in hiers.items():
        vals = set(ds.df[attr].unique())
        missing = sorted(v for v in vals if v not in hier.parent)
        if missing:
            problems.append(f"{ds.name}:{attr} значения без ветви: {missing}")
    return problems


def y_codes(ds: Dataset) -> np.ndarray:
    mapping = {lab: i for i, lab in enumerate(ds.labels)}
    vals = ds.df[ds.y_col].to_numpy(dtype=object)
    unknown = sorted({v for v in vals if v not in mapping})
    if unknown:
        raise ValueError(f"{ds.name}: метки вне заданного протоколом домена: {unknown}")
    return np.array([mapping[v] for v in vals], dtype=np.int64)
