"""Эмпирические атаки: Record Linkage и SAI (протокол 2.6, без D_test при обучении).

Атакующий знает QI*, B_SA, J ⊆ QI, T_ext и правила обобщения, но не соответствия
записей. Точный SA — только у доверенного оценщика. Для Bank T_ext никогда не
содержит balance как QI; B_SA — лишь публикуемый SA, не linkage key.

Seeds: scenario seed = первые 8 байт SHA-256 UTF-8
"UA-v1.0|attack|{scenario}|{repeat}|{fold}|{method}|{dataset}", big-endian
uint64 -> PCG64; record-tie seed = та же строка + "|{row_id}".
Строка «UA-v1.0|attack|…» — фиксированный идентификатор правила сидов: её изменение
изменит все результаты атак.
Выбор из списков, отсортированных по row_id.
"""

from __future__ import annotations

import hashlib
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np

from core import Leaf, PreparedData

SCENARIO_PRIMARY = "prosecutor"
SCENARIOS_JOURNALIST = {"journalist": None, "journalist_r3": 3, "journalist_r5": 5}
SCENARIO_NOISE5 = "noise5"
NOISE_FRACTION = 0.05


def scenario_key(scenario: str, repeat: int, fold: int, method: str, dataset: str) -> str:
    return f"UA-v1.0|attack|{scenario}|{repeat}|{fold}|{method}|{dataset}"


def make_rng(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed_u64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.Generator(np.random.PCG64(seed_u64))


def record_tie_rng(base_key: str, row_id: int) -> np.random.Generator:
    return make_rng(base_key + f"|{row_id}")


def round_half_even(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


# ----------------------------------------------------------------------------
# Атака Record Linkage + SAI по списку внешних записей
# ----------------------------------------------------------------------------
class AttackContext:
    """Кеш D_public для атаки: матрицы покрытия по листьям (векторный поиск M)."""

    def __init__(self, pd_data: PreparedData, leaves: list[Leaf],
                 bsa_counts: list[np.ndarray]):
        self.pd_data = pd_data
        self.leaves = leaves
        self.leaf_row_sorted = [np.sort(lf.rows) for lf in leaves]
        self.leaf_row_sets = [set(r.tolist()) for r in self.leaf_row_sorted]
        self.leaf_sizes = np.array([len(r) for r in self.leaf_row_sorted], dtype=np.int64)
        published: set[int] = set()
        for s in self.leaf_row_sets:
            published |= s
        self.published_rows = published
        n_bsa = int(pd_data.b_sa.max()) + 1
        pc = np.bincount(pd_data.b_sa, minlength=n_bsa).astype(np.float64)
        self.prior = pc / pc.sum()
        self.bsa_matrix = np.array(bsa_counts, dtype=np.float64)  # (leaves, n_bsa)
        # числовые домены листьев
        self.num_lo: dict[str, np.ndarray] = {}
        self.num_hi: dict[str, np.ndarray] = {}
        for attr in pd_data.numeric:
            self.num_lo[attr] = np.array([lf.num_ranges[attr][0] for lf in leaves],
                                         dtype=np.float64)
            self.num_hi[attr] = np.array([lf.num_ranges[attr][1] for lf in leaves],
                                         dtype=np.float64)
        # категориальное покрытие: leaf x code -> значение покрыто label листа
        self.cat_cover: dict[str, np.ndarray] = {}
        # значение -> код в cat_uniq (для noise-подмен)
        self.cat_code_index: dict[str, dict[str, int]] = {}
        for attr in pd_data.cat_codes:
            hier = pd_data.hier[attr]
            uniq_str = pd_data.cat_uniq[attr]
            mat = np.zeros((len(leaves), len(uniq_str)), dtype=bool)
            for li, lf in enumerate(leaves):
                under = hier.values_under(lf.cat_labels[attr])
                for ci, sv in enumerate(uniq_str):
                    if sv in under:
                        mat[li, ci] = True
            self.cat_cover[attr] = mat
            self.cat_code_index[attr] = {sv: ci for ci, sv in enumerate(uniq_str)}

    def _coverage_mask(self, p: int, J: list[str], override: dict | None) -> np.ndarray:
        """Булева маска совместимых листьев для записи p (векторно).

        Noise-подмены идут тем же векторным путём: подменённое значение
        подставляется до сравнения, без Python-цикла по листьям для каждой записи.
        """
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
                mask &= (self.num_lo[attr] <= v) & (v <= self.num_hi[attr])
            else:
                if override and (p, attr) in override:
                    sv = str(override[(p, attr)])
                    ci = self.cat_code_index[attr].get(sv)
                    if ci is None:  # подмена вне домена — не покрывает ни один лист
                        return np.zeros(len(self.leaves), dtype=bool)
                else:
                    ci = int(pd_data.cat_codes[attr][p])
                mask &= self.cat_cover[attr][:, ci]
            if not mask.any():
                break
        return mask

    def run(self, J: list[str], ext_pos: np.ndarray,
            base_key: str, value_override: dict | None = None) -> dict:
        """M(x,D*) = {r* : ∀A_j∈J, x[A_j] ∈ g(r*[A_j])}; метрики протокола."""
        pd_data = self.pd_data
        ext_pos = np.asarray(ext_pos)
        ext_pos = ext_pos[np.argsort(pd_data.ds.df["row_id"].to_numpy()[ext_pos], kind="stable")]
        row_ids = pd_data.ds.df["row_id"].to_numpy()

        n_cov = cand = unique = homogeneous = majority_acc = 0
        risk_sum = overall_risk_sum = sai_sum = 0.0
        sai_n = sizes_sum = 0
        chosen_hash: list[int] = []

        for p in ext_pos:
            row_id = int(row_ids[p])
            mask = self._coverage_mask(p, J, value_override)
            m_size = int(self.leaf_sizes[mask].sum())
            true_published = row_id in self.published_rows
            if true_published:
                n_cov += 1
            if m_size == 0:
                continue  # пустой M: успех 0; из SAI не исключается из Coverage
            cand += 1
            counts = self.bsa_matrix[mask].sum(axis=0)
            true_in_m = true_published and self._true_in_m(row_id, mask)
            if true_published:
                risk_sum += (1.0 / m_size) if true_in_m else 0.0
                sizes_sum += m_size
                unique += 1 if m_size == 1 else 0
                b_x = int(pd_data.b_sa[p])
                p_hat = counts / m_size
                sai_sum += max(0.0, float(p_hat[b_x] - self.prior[b_x]))
                sai_n += 1
                if np.count_nonzero(counts) == 1:
                    homogeneous += 1
                majority_acc += 1 if int(np.argmax(counts)) == b_x else 0
                if m_size > 1:
                    merged = np.sort(self._merged_rows(mask))
                    rng = record_tie_rng(base_key, row_id)
                    chosen_hash.append(int(merged[rng.integers(0, m_size)]))
            # OverallLinkageRisk использует |T_all|: I(true∈M)/|M| (0 для неопубликованных)
            overall_risk_sum += (1.0 / m_size) if true_in_m else 0.0

        n_all = len(ext_pos)
        t_cov = n_cov
        return {
            "n_t_all": n_all,
            "n_t_cov": t_cov,
            "coverage": (t_cov / n_all) if n_all else np.nan,
            "candidate_rate": (cand / t_cov) if t_cov else np.nan,
            "unique_match_rate": (unique / t_cov) if t_cov else np.nan,
            "linkage_risk": (risk_sum / t_cov) if t_cov else np.nan,
            "overall_linkage_risk": (overall_risk_sum / n_all) if n_all else np.nan,
            "sai_gain": (sai_sum / sai_n) if sai_n else np.nan,
            "sai_coverage": (sai_n / n_all) if n_all else np.nan,
            "homogeneous_rate": (homogeneous / t_cov) if t_cov else np.nan,
            "majority_accuracy": (majority_acc / t_cov) if t_cov else np.nan,
            "mean_group_size": (sizes_sum / t_cov) if t_cov else np.nan,
            "chosen_candidates_hash": hash(tuple(chosen_hash)) & 0xFFFFFFFF,
        }

    def _true_in_m(self, row_id: int, mask: np.ndarray) -> bool:
        """Опубликованная true-запись лежит в M (только для trusted evaluator)."""
        idxs = np.flatnonzero(mask)
        for li in idxs:
            if row_id in self.leaf_row_sets[int(li)]:
                return True
        return False

    def _merged_rows(self, mask: np.ndarray) -> np.ndarray:
        idxs = np.flatnonzero(mask)
        return np.concatenate([self.leaf_row_sorted[int(i)] for i in idxs])


# ----------------------------------------------------------------------------
# Сценарии
# ----------------------------------------------------------------------------
def journalist_positions(pd_data: PreparedData, base_key: str,
                         train_pos: np.ndarray, test_pos: np.ndarray) -> np.ndarray:
    """Journalist: n_J = floor(|D_train|/2) записей D_train (permutation PCG64
    по списку, отсортированному по row_id) плюс весь D_test как внешний реестр."""
    n_j = int(len(train_pos) // 2)
    order = np.argsort(pd_data.ds.df["row_id"].to_numpy()[train_pos], kind="stable")
    rng = make_rng(base_key)
    perm = rng.permutation(len(train_pos))
    chosen = train_pos[order][perm[:n_j]]
    return np.concatenate([chosen, test_pos])


def noise_value_override(pd_data: PreparedData, base_key: str, J: list[str],
                         ext_pos: np.ndarray) -> tuple[dict, list[str]]:
    """5% noise QI: count = round_half_even(0.05*|T_ext|*|J|) пар (row_id, A_j)
    в fixed order; выбор PCG64 без замены; categorical replacement — равновероятно
    из D_train domain кроме исходного; numeric — из отсортированных distinct
    D_train значений кроме исходного; singleton domain пропускается."""
    rng = make_rng(base_key)
    ext_pos = np.asarray(ext_pos)
    ext_pos = ext_pos[np.argsort(pd_data.ds.df["row_id"].to_numpy()[ext_pos], kind="stable")]
    pairs: list[tuple[int, str]] = []
    for p in ext_pos:
        for attr in J:
            pairs.append((int(p), attr))
    count = round_half_even(NOISE_FRACTION * len(ext_pos) * len(J))
    count = min(count, len(pairs))
    idx = rng.choice(len(pairs), size=count, replace=False) if count else np.array([], dtype=int)
    skipped: list[str] = []
    override: dict[tuple[int, str], float | str] = {}
    for i in idx:
        p, attr = pairs[int(i)]
        if attr in pd_data.numeric:
            train_vals = np.unique(pd_data.numeric[attr])
            train_vals = train_vals[~np.isnan(train_vals)]
            cur = pd_data.numeric[attr][p]
            pool = train_vals[train_vals != cur]
            if len(pool) == 0:
                skipped.append(f"{attr}@{p}:singleton")
                continue
            override[(p, attr)] = float(pool[rng.integers(0, len(pool))])
        else:
            hier = pd_data.hier[attr]
            pool = [v for v in hier.train_values
                    if v != pd_data.cat_uniq[attr][int(pd_data.cat_codes[attr][p])]]
            if len(pool) == 0:
                skipped.append(f"{attr}@{p}:singleton")
                continue
            override[(p, attr)] = pool[int(rng.integers(0, len(pool)))]
    return override, skipped


def sort_j_by_rrep(pd_data: PreparedData, r: int) -> list[str]:
    """J с r атрибутами: по убыванию R_rep, ties — фиксированный порядок QI."""
    idx = {a: i for i, a in enumerate(pd_data.qi_order)}
    ordered = sorted(pd_data.qi_order,
                     key=lambda a: (-pd_data.ds.r_rep[a], idx[a]))
    return ordered[:r]
