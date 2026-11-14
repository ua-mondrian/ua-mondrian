"""Модульные тесты реализации протокола UA-Mondrian (группы проверок 1–14 из протокола).

Запуск (из code/):  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attacks
import config as C
import core
import cv
import data as datalib
import metrics as mlib
import methods_b4
import methods_ua
import ml_eval
import preprocess
import root_screen
from data import ROOT_LABEL


# ----------------------------------------------------------------------------
# Фикстуры: мини-данные в формате Adult / Bank
# ----------------------------------------------------------------------------
def make_adult_df(n_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(2 * n_per_class):
        income = "<=50K" if i < n_per_class else ">50K"
        age = int(rng.integers(20, 60)) + (5 if income == ">50K" else 0)
        rows.append({
            "age": float(age), "workclass": "Private" if i % 3 else "State-gov",
            "fnlwgt": float(1000 + i), "education": "HS-grad" if i % 2 else "Bachelors",
            "education-num": float(9 if i % 2 else 13),
            "marital-status": "Never-married" if i % 2 else "Married-civ-spouse",
            "occupation": "Sales" if income == "<=50K" else "Exec-managerial",
            "relationship": "Not-in-family" if i % 2 else "Wife",
            "race": "White", "sex": "Male" if i % 2 else "Female",
            "capital-gain": 0.0, "capital-loss": 0.0,
            "hours-per-week": float(40 + (i % 5)),
            "native-country": "United-States", "income": income,
        })
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def make_bank_df(n_per_class: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(2 * n_per_class):
        y = "no" if i < n_per_class else "yes"
        rows.append({
            "age": float(30 + (i % 30)), "job": ["admin.", "blue-collar", "retired"][i % 3],
            "marital": "married" if i % 2 else "single",
            "education": ["secondary", "tertiary", "primary"][i % 3],
            "default": "no", "housing": "yes" if i % 2 else "no",
            "loan": "no" if i % 3 else "yes",
            "balance": float(500 * (1 if i < n_per_class else 3) + (i % 7) * 10),
            "contact": "cellular", "day": float(5 + i % 20), "month": "may",
            "duration": float(100 + i % 50), "campaign": float(1 + i % 3),
            "pdays": -1.0, "previous": 0.0, "poutcome": "unknown", "y": y,
        })
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df


@pytest.fixture
def adult_pd():
    df = make_adult_df()
    ds = datalib.adult_dataset(df, "SA_NEQ_Y")
    train_pos = np.arange(len(df))
    return core.prepare(ds, train_pos, beta=0.5), ds, train_pos


@pytest.fixture
def bank_pd():
    df = make_bank_df()
    binner = datalib.QuantileBinner.fit(df["balance"].to_numpy(np.float64))
    ds = datalib.bank_dataset(df, binner)
    train_pos = np.arange(len(df))
    return core.prepare(ds, train_pos, beta=0.5), ds, train_pos


# ----------------------------------------------------------------------------
# (5) MI edge tests
# ----------------------------------------------------------------------------
def test_mi_edges():
    # пустой P
    assert core.mutual_information(np.array([], dtype=np.int64),
                                   np.array([], dtype=np.int64), 0, 2) == 0.0
    # |V_Y| = 1 -> MI = 0
    assert core.mutual_information(np.array([0, 1, 2]), np.array([0, 0, 0]), 3, 1) == 0.0
    # константный A_j -> MI = 0
    assert core.mutual_information(np.array([0, 0, 0]), np.array([0, 1, 0]), 1, 2) == 0.0
    # все MI = 0 (Y ортогонален A) -> нормировка даёт все нули
    a = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 1, 0, 1, 0, 1])
    mi = {"x": core.mutual_information(a, y, 3, 2)}
    norm = core.normalize_mi(mi)
    assert all(abs(v) < 1e-12 for v in norm.values())
    # часть MI = 0
    y2 = np.array([0, 0, 0, 1, 1, 1])
    mi2 = {"x": core.mutual_information(a, y2, 3, 2), "z": 0.0}
    norm2 = core.normalize_mi(mi2)
    assert norm2["x"] > 0.5 and norm2["z"] == 0.0
    # NaN/inf запрещены
    assert all(np.isfinite(v) for v in norm2.values())
    # |V_Aj| <= 1 -> Dist = 0
    assert core.norm_entropy(np.array([7.0, 7.0, 7.0])) == 0.0
    # энтропия ровно log2(l) на границе -> валидно
    assert core.entropy_bits(np.array([2, 2])) == pytest.approx(1.0)
    assert core.is_valid_node(np.array([0, 0, 1, 1]), k=2, ell=2)


# ----------------------------------------------------------------------------
# (1)-(3) валидность, медиана, повторы, отклонение разрезов
# ----------------------------------------------------------------------------
def test_numeric_candidates_lower_median_and_repeats():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    pos = np.arange(4)
    class FakePD:  # минимальный стаб
        numeric = {"a": vals}
    cands = core.iter_numeric_splits(FakePD(), "a", pos)
    t0, left, right = next(cands)
    assert t0 == 2.0  # нижняя медиана x_(n/2) для чётного n
    assert list(left) == [0, 1] and list(right) == [2, 3]
    ts = [t for t, _, _ in cands]
    assert 1.5 in ts and 2.5 in ts and 3.5 in ts  # середины соседних различных


def test_equal_values_not_split_single_value_generalizes():
    # все QI константны -> единственный кандидат не даёт двух непустых детей
    df = make_adult_df(40)
    for a in C.ADULT_QI:
        if a in C.ADULT_NUMERIC:
            df[a] = 42.0
        else:
            df[a] = "Const" if a != "native-country" else "United-States"
    ds = datalib.adult_dataset(df, "SA_NEQ_Y")
    ds.b_sa = np.array([0, 1] * (len(df) // 2))  # валидный корень
    train_pos = np.arange(len(df))
    pd_data = core.prepare(ds, train_pos, beta=0.5)
    leaves = core.partition(pd_data, train_pos, k=10, ell=2,
                            score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    assert leaves is not None and len(leaves) == 1
    # атрибут с единственным значением не даёт разрезов
    pos = np.arange(6)
    vals = np.array([5.0] * 6)
    class FakePD:
        numeric = {"a": vals}
    assert list(core.iter_numeric_splits(FakePD(), "a", pos)) == []


def test_impossible_split_publishes_no_invalid_leaf(adult_pd):
    """Разрез принимается только при валидности ОБОИХ детей; иначе обобщение."""
    pd_data, ds, train_pos = adult_pd
    # сделаем B_SA сильно несбалансированной и малой группой, чтобы дети были невалидны
    k, ell = 10, 2
    leaves = core.partition(pd_data, train_pos, k=k, ell=ell,
                            score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    assert leaves is not None
    for lf in leaves:
        pos = lf.rows.astype(np.int64)
        assert core.is_valid_node(pd_data.b_sa[pos], k, ell)
    ok, problems = core.post_validate(pd_data, train_pos, leaves, k, ell)
    assert ok, problems


def test_child_invalid_rejects_split(adult_pd):
    """Если хотя бы один ребёнок невалиден, разрез не принимается в этой позиции."""
    pd_data, ds, train_pos = adult_pd
    k, ell = 10, 2
    # большой k: дети медианного разреза часто невалидны; алгоритм обязан
    # перебрать следующие кандидаты/атрибуты, но никогда не принять невалидный
    k = 60
    leaves = core.partition(pd_data, train_pos, k=k, ell=ell,
                            score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    assert leaves is not None
    for lf in leaves:
        pos = lf.rows.astype(np.int64)
        assert core.is_valid_node(pd_data.b_sa[pos], k, ell)


def test_root_fail_infeasible(adult_pd):
    pd_data, ds, train_pos = adult_pd
    leaves = core.partition(pd_data, train_pos, k=10_000, ell=2,
                            score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    assert leaves is None  # Infeasible Constraints


# ----------------------------------------------------------------------------
# (7) покрытие, повторяемость
# ----------------------------------------------------------------------------
def test_partition_covers_train_exactly_and_is_deterministic(adult_pd):
    pd_data, ds, train_pos = adult_pd
    l1 = core.partition(pd_data, train_pos, k=10, ell=2,
                        score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    l2 = core.partition(pd_data, train_pos, k=10, ell=2,
                        score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    rows1 = np.sort(np.concatenate([lf.rows for lf in l1]))
    rows2 = np.sort(np.concatenate([lf.rows for lf in l2]))
    assert np.array_equal(rows1, rows2)
    assert np.array_equal(rows1, np.sort(pd_data.ds.df["row_id"].to_numpy()))
    assert len(np.unique(rows1)) == len(rows1)


# ----------------------------------------------------------------------------
# (8) повреждённый лист отклоняется PostValidate
# ----------------------------------------------------------------------------
def test_postvalidate_rejects_damaged_leaf(adult_pd):
    pd_data, ds, train_pos = adult_pd
    leaves = core.partition(pd_data, train_pos, k=10, ell=2,
                            score_order_fn=core.make_ua_order(0.5), mi_mode="local")
    ok, _ = core.post_validate(pd_data, train_pos, leaves, 10, 2)
    assert ok
    # повреждение: сдвиг диапазона числового QI у первого листа
    bad = [core.Leaf(rows=leaves[0].rows,
                     num_ranges={a: (lo - 5.0, hi) for a, (lo, hi) in
                                 leaves[0].num_ranges.items()},
                     cat_labels=dict(leaves[0].cat_labels))]
    ok_bad, problems = core.post_validate(pd_data, train_pos[:0], bad, 10, 2)
    assert not ok_bad and any("диапазон" in p for p in problems)


# ----------------------------------------------------------------------------
# (4) детерминированные ties
# ----------------------------------------------------------------------------
def test_ties_use_fixed_qi_order(adult_pd):
    pd_data, ds, train_pos = adult_pd
    scores = {a: 0.5 for a in pd_data.qi_order}
    assert core.sorted_attrs(pd_data, scores) == list(pd_data.qi_order)
    scores2 = {a: (0.0 if a == "age" else 1.0) for a in pd_data.qi_order}
    assert core.sorted_attrs(pd_data, scores2)[0] != "age" or scores2["age"] == 1.0


# ----------------------------------------------------------------------------
# (12) ручной пример DM_w
# ----------------------------------------------------------------------------
def test_manual_dm_w_example():
    ex = mlib.manual_dm_w_example()
    assert ex["dm"] == pytest.approx(13.0)
    assert ex["dm_w"] == pytest.approx(6.5)
    assert ex["z"] == pytest.approx(0.6)
    assert ex["dm_w_zero"] == pytest.approx(6.5)


# ----------------------------------------------------------------------------
# (13) B4/B4m различимы; A3 отсутствует
# ----------------------------------------------------------------------------
def test_b4_b4m_distinct_and_a3_absent(adult_pd):
    pd_data, ds, train_pos = adult_pd
    assert C.JOURNAL_IDS["B4"] != C.JOURNAL_IDS["B4m"]
    assert "A3" not in C.JOURNAL_IDS
    assert "A3" not in C.METHODS_MAIN
    # оба работают и дают валидные релизы (независимые реализации)
    root_b4 = methods_b4.run_b4(pd_data, train_pos, k=10, ell=2)
    leaves_b4 = methods_b4.b4_leaves_core(root_b4)
    root_b4m = methods_ua.run_ua(pd_data, train_pos, k=10, ell=2, method="B4m")
    leaves_b4m = methods_ua.leaves_of(root_b4m)
    ok_b4, prob_b4 = core.post_validate(pd_data, train_pos, leaves_b4, 10, 2)
    ok_b4m, prob_b4m = core.post_validate(pd_data, train_pos, leaves_b4m, 10, 2)
    assert ok_b4, prob_b4
    assert ok_b4m, prob_b4m


# ----------------------------------------------------------------------------
# (6) unknown/missing/out-of-range
# ----------------------------------------------------------------------------
def test_preprocessing_unknown_missing_out_of_range():
    df = make_adult_df(40)  # 80 строк
    # unseen-значение test-строки: подмена ДО prepare (код попадёт в cat_uniq,
    # но не в иерархию, обученную на train-строках)
    df.loc[79, "workclass"] = "__UNSEEN_VALUE__"
    ds = datalib.adult_dataset(df, "SA_NEQ_Y")
    # train = чётные позиции: B_SA (occupation) сбалансирована -> корень валиден
    train_pos = np.arange(0, 80, 2)
    pd_data = core.prepare(ds, train_pos, beta=0.5)
    root = methods_ua.run_ua(pd_data, train_pos, k=10, ell=2, method="UA-Local")
    leaves = methods_ua.leaves_of(root)
    test_pos = np.array([61, 71, 79])
    x_train, x_test = preprocess.build_d_ml(pd_data, leaves, root, core.route_row,
                                            train_pos, test_pos)
    col_root = f"workclass={ROOT_LABEL}"
    assert col_root in x_test.columns
    # unseen-строка 79 -> только корневая колонка; остальные workclass-колонки 0
    assert x_test[col_root].iloc[2] == 1.0
    others = [c for c in x_test.columns if c.startswith("workclass=") and c != col_root]
    assert all(x_test[o].iloc[2] == 0.0 for o in others)
    # правило unseen: train-значение остаётся собой, неизвестное -> корень
    hier_wc = pd_data.hier["workclass"]
    assert hier_wc.map_test_value("Private") == "Private"
    assert hier_wc.map_test_value("__UNSEEN_VALUE__") == ROOT_LABEL

    # out-of-range numeric: прижатие к min/max train (маршрут в крайний лист)
    attr_n = "age"
    lo_t, hi_t = pd_data.train_ranges[attr_n]
    pd_data.numeric[attr_n][71] = hi_t + 1000.0
    x_train2, x_test2 = preprocess.build_d_ml(pd_data, leaves, root, core.route_row,
                                              train_pos, test_pos)
    assert x_test2[attr_n].max() <= x_train2[attr_n].max() + 1e-9
    assert x_test2[attr_n].min() >= x_train2[attr_n].min() - 1e-9

    # missing numeric: нижняя медиана D_train + индикатор (только D_ML)
    pd_data.numeric[attr_n][61] = np.nan
    x_train3, x_test3 = preprocess.build_d_ml(pd_data, leaves, root, core.route_row,
                                              train_pos, test_pos)
    assert x_test3[f"{attr_n}__missing"].sum() == 1
    arr = pd_data.numeric[attr_n][train_pos]
    lower_median = float(np.sort(arr)[(len(arr) - 1) // 2])
    assert x_test3[attr_n].iloc[0] == pytest.approx(lower_median, abs=1e-9)
    assert x_test3[f"{attr_n}__missing"].iloc[0] == 1.0


# ----------------------------------------------------------------------------
# (9) Bank role test: balance только SA
# ----------------------------------------------------------------------------
def test_bank_balance_role(bank_pd):
    pd_data, ds, train_pos = bank_pd
    # balance отсутствует в QI/P_aux
    assert "balance" not in pd_data.qi_order
    assert "balance" not in ds.p_aux_num and "balance" not in ds.p_aux_cat
    root = methods_ua.run_ua(pd_data, train_pos, k=10, ell=2, method="UA-Local")
    leaves = methods_ua.leaves_of(root)
    ok, problems = core.post_validate(pd_data, train_pos, leaves, 10, 2)
    assert ok, problems
    test_pos = np.arange(len(ds.df))
    x_train, x_test = preprocess.build_d_ml(pd_data, leaves, root, core.route_row,
                                            train_pos, test_pos)
    rx_train, rx_test = preprocess.build_raw(pd_data, train_pos, test_pos)
    for frame in (x_train, x_test, rx_train, rx_test):
        assert not any(c.startswith("balance") for c in frame.columns)
    # D_public: B_SA присутствует, QI* без balance
    d_public = preprocess.build_d_public(pd_data, leaves)
    assert "b_sa" in d_public.columns
    assert not any(c.startswith("balance") for c in d_public.columns)
    # J атакующего = QI — balance не входит
    ctx = attacks.AttackContext(pd_data, leaves,
                                methods_ua.leaf_bsa_counts(pd_data, leaves))
    atk = ctx.run(list(pd_data.qi_order), train_pos,
                  attacks.scenario_key("prosecutor", 0, 1, "UA-Local", "Bank"))
    assert atk["coverage"] == 1.0  # primary: Coverage=1 на сплите


# ----------------------------------------------------------------------------
# (10) Coverage=0 -> FAIL correctness
# ----------------------------------------------------------------------------
def test_coverage_zero_is_correctness_fail(adult_pd):
    pd_data, ds, train_pos = adult_pd
    # D_public по сбалансированной части train (15 Sales + 15 Exec);
    # атакуем строки вне релиза
    small_train = np.concatenate([np.arange(0, 30, 2), np.arange(40, 70, 2)])
    root = methods_ua.run_ua(pd_data, small_train, k=10, ell=2, method="UA-Local")
    leaves = methods_ua.leaves_of(root)
    ctx = attacks.AttackContext(pd_data, leaves,
                                methods_ua.leaf_bsa_counts(pd_data, leaves))
    outside = np.arange(1, 80, 2)  # нечётные записи не опубликованы
    atk = ctx.run(list(pd_data.qi_order), outside,
                  attacks.scenario_key("prosecutor", 0, 1, "UA-Local", "Adult"))
    assert atk["coverage"] == 0.0
    # в терминах протокола это FAIL attack-harness/correctness, а не статистика
    assert atk["n_t_cov"] == 0


# ----------------------------------------------------------------------------
# (11) labels / zero_division / AUC-NA
# ----------------------------------------------------------------------------
def test_ml_eval_labels_zero_division_auc_na():
    labels = ["<=50K", ">50K"]
    # отсутствующий в test класс сохраняется в знаменателе; zero_division=0
    y_test = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 1])
    res = ml_eval.evaluate_predictions(y_test, y_pred, None, labels, ">50K")
    assert res["f1[>50K]"] == 0.0            # zero_division=0
    assert res["support[>50K]"] == 0
    # класс 0: precision=1, recall=2/3 -> f1=0.8; класс 1: f1=0; macro = 0.4
    assert res["macro_f1"] == pytest.approx(0.4)
    # proba отсутствует -> AUC NA с причиной, не 0
    assert np.isnan(res["roc_auc"]) and "predict_proba" in res["auc_note"]
    # один класс в y_test при наличии proba -> AUC NA с причиной
    proba_one = np.array([[0.9, 0.1], [0.8, 0.2]])
    res_one = ml_eval.evaluate_predictions(np.array([0, 0]), np.array([0, 0]),
                                           proba_one, labels, ">50K")
    assert np.isnan(res_one["roc_auc"]) and "один класс" in res_one["auc_note"]
    # нормальный случай с proba
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    res2 = ml_eval.evaluate_predictions(np.array([0, 0, 1]), np.array([0, 0, 1]),
                                        proba, labels, ">50K")
    assert 0.0 <= res2["roc_auc"] <= 1.0


# ----------------------------------------------------------------------------
# (14) Root-screen Adult SA=Y и CMC PASS/FAIL
# ----------------------------------------------------------------------------
def test_root_screen_pass_fail():
    # пограничный PASS: n>=k и H ровно log2(l)=1
    b = np.array([0] * 8 + [1] * 8)
    rs = root_screen.root_screen(b, k=10, ell=2)
    assert rs["pass"] and rs["h_b_sa"] == pytest.approx(1.0)
    # FAIL по энтропии
    b2 = np.array([0] * 15 + [1])
    rs2 = root_screen.root_screen(b2, k=10, ell=2)
    assert not rs2["pass"] and not rs2["entropy_ok"]
    # FAIL по размеру
    rs3 = root_screen.root_screen(b[:8], k=10, ell=2)
    assert not rs3["pass"] and not rs3["size_ok"]


def test_root_screen_adult_sa_eq_y_and_cmc():
    df = make_adult_df(40)
    ds = datalib.adult_dataset(df, "SA_EQ_Y")
    # income сбалансирован -> H = 1 -> PASS при k<=40
    rs = root_screen.root_screen(np.asarray(ds.b_sa), k=10, ell=2)
    assert rs["pass"]
    # CMC-подобные данные: крайне редкий SA -> FAIL
    rng = np.random.default_rng(7)
    cmc_rows = []
    for i in range(40):
        cmc_rows.append({
            "wife_age": float(25 + i % 15), "wife_edu": float(i % 4 + 1),
            "husband_edu": float(i % 3 + 1), "num_children": float(i % 5),
            "wife_religion": float(i % 2), "wife_working": float(i % 2),
            "husband_occ": float(i % 4 + 1), "sol_index": float(i % 4),
            "media_exposure": 1.0 if rng.random() < 0.05 else 0.0,
            "contraceptive_method": float(i % 3 + 1),
        })
    cmc_df = pd.DataFrame(cmc_rows)
    cmc_df["row_id"] = np.arange(len(cmc_df), dtype=np.int64)
    ds2 = datalib.cmc_dataset(cmc_df, "A")
    bsa = np.asarray(ds2.b_sa)
    assert set(np.unique(bsa)) == {0.0, 1.0}
    rs2 = root_screen.root_screen(bsa, k=10, ell=2)
    assert not rs2["pass"]  # H(медиаэкспозиция ~5%) < log2(2)


# ----------------------------------------------------------------------------
# (доп) сиды атак: SHA-256 -> PCG64 детерминизм
# ----------------------------------------------------------------------------
def test_attack_seed_determinism(adult_pd):
    pd_data, ds, train_pos = adult_pd
    root = methods_ua.run_ua(pd_data, train_pos, k=10, ell=2, method="UA-Local")
    leaves = methods_ua.leaves_of(root)
    ctx = attacks.AttackContext(pd_data, leaves,
                                methods_ua.leaf_bsa_counts(pd_data, leaves))
    key = attacks.scenario_key("journalist_r3", 0, 1, "UA-Local", "Adult")
    J3 = attacks.sort_j_by_rrep(pd_data, 3)
    r1 = ctx.run(J3, train_pos, key)
    r2 = ctx.run(J3, train_pos, key)
    assert json_dump(r1) == json_dump(r2)
    assert r1["coverage"] > 0


def json_dump(x) -> str:
    import json
    return json.dumps(x, sort_keys=True, default=str)


# ----------------------------------------------------------------------------
# (доп) Journalist T_ext: n_J = floor(|D_train|/2) + весь D_test
# ----------------------------------------------------------------------------
def test_journalist_ext_size(adult_pd):
    pd_data, ds, train_pos = adult_pd
    test_pos = np.array([0, 1, 2, 3])  # условный D_test
    ext = attacks.journalist_positions(pd_data, "UA-v1.0|attack|journalist|0|1|B2|Adult",
                                       train_pos, test_pos)
    assert len(ext) == len(train_pos) // 2 + len(test_pos)


# ----------------------------------------------------------------------------
# (доп) B_SA квантование Bank: границы только по D_train, clip, searchsorted
# ----------------------------------------------------------------------------
def test_bank_binner_train_only_and_clip():
    df = make_bank_df(40)
    train_pos = np.arange(50)  # условный train
    binner = datalib.QuantileBinner.fit(df["balance"].to_numpy(np.float64)[train_pos])
    b_all = binner.transform(df["balance"].to_numpy(np.float64))
    assert binner.n_categories() <= 4
    assert set(np.unique(b_all)) <= set(range(binner.n_categories()))
    # test вне диапазона train прижимается к краям
    b_out = binner.transform(np.array([binner.lo - 100.0, binner.hi + 100.0]))
    assert b_out[0] == 0 and b_out[1] == binner.n_categories() - 1
