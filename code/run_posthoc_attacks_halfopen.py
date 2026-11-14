"""Разведочный пересчёт атак (вне протокола) с точной (полуоткрытой) семантикой
числового разреза + классовая оценка риска 1/|EC|. Деревья строятся теми же функциями,
что run_full.py / run_posthoc_*.py; контроль: замкнутый режим (attacks.AttackContext)
воспроизводит logs/runs_attacks.csv (основной сценарий — все конфигурации; вторичные — сплит r0 f1),
logs_posthoc/alpha_grid.csv и rrep_variants.csv (LinkageRisk/SAI-gain).
Запуск (из code/): python run_posthoc_attacks_halfopen.py <mode> <dataset> <repeat> <fold>
mode ∈ {main, alpha, rrep}; выход — logs_posthoc/halfopen/<mode>_<ds>_<r>_<f>.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_common as H  # noqa: E402
import attacks  # noqa: E402
import attacks_halfopen as AH  # noqa: E402
import config as C  # noqa: E402
import methods_ua  # noqa: E402

OUTD = H.LOGP / 'halfopen'
OUTD.mkdir(exist_ok=True)
METHODS = ['B2', 'B4', 'B4m', 'A1', 'A2', 'UA-Global', 'UA-Local']
KS = [5, 10, 25, 50, 100]
SECONDARY = ['journalist', 'journalist_r3', 'journalist_r5', 'noise5']
CMP = ('linkage_risk', 'sai_gain', 'coverage', 'n_t_cov', 'mean_group_size')
RTOL = 1e-12


def scen_args(pdd, scenario, key, tr, te):
    if scenario == 'prosecutor':
        return list(pdd.qi_order), tr, None
    if scenario == 'noise5':
        J = list(pdd.qi_order)
        override, _ = attacks.noise_value_override(pdd, key, J, tr)
        return J, tr, override
    r = {'journalist': None, 'journalist_r3': 3, 'journalist_r5': 5}[scenario]
    J = list(pdd.qi_order) if r is None else attacks.sort_j_by_rrep(pdd, r)
    return J, attacks.journalist_positions(pdd, key, tr, te), None


def check(ref: dict, got: dict, tag):
    for m, v in ref.items():
        g = got[m]
        assert np.isclose(g, v, rtol=RTOL, atol=1e-15), (tag, m, g, v)


_REF = {}


def ref_main(ds, repeat, fold, k, method, scenario):
    if 'main' not in _REF:
        a = pd.read_csv(C.LOG_DIR / 'runs_attacks.csv')
        a = a[a.metric.isin(CMP)]
        _REF['main'] = {key: dict(zip(g.metric, g.value)) for key, g in
                        a.groupby(['dataset', 'repeat', 'fold', 'k', 'method', 'scenario'])}
    return _REF['main'][(ds, repeat, fold, k, method, scenario)]


def ref_post(name, ds, repeat, fold, k, method):
    if name not in _REF:
        a = pd.read_csv(H.LOGP / f'{name}.csv')
        a = a[a.metric.isin(('linkage_risk', 'sai_gain', 'coverage'))]
        _REF[name] = {key: dict(zip(g.metric, g.value)) for key, g in
                      a.groupby(['dataset', 'repeat', 'fold', 'k', 'method'])}
    return _REF[name][(ds, repeat, fold, k, method)]


def one_tree(pd_att, tr, te, leaves, root, method_for_walk, ds, repeat, fold, k, key_method,
             scenarios, verify_closed_for, ref_fn):
    bsa = methods_ua.leaf_bsa_counts(pd_att, leaves)
    flags = AH.lo_open_flags(root, method_for_walk, pd_att, leaves)
    ctx_c = attacks.AttackContext(pd_att, leaves, bsa)
    ctx_h = AH.AttackContextHalfOpen(pd_att, leaves, bsa, flags)
    cr = AH.class_risk(pd_att, tr, leaves, ctx_c.prior)
    rows = []
    for sc in scenarios:
        key = attacks.scenario_key(sc, repeat, fold, key_method, ds)
        J, ext, ov = scen_args(pd_att, sc, key, tr, te)
        if sc in verify_closed_for:
            rc = ctx_c.run(J, ext, key, value_override=ov)
            check(ref_fn(sc), rc, (ds, repeat, fold, k, key_method, sc))
        rh = ctx_h.run(J, ext, key, value_override=ov)
        for m, v in rh.items():
            rows.append(dict(dataset=ds, repeat=repeat, fold=fold, k=k, method=key_method,
                             scenario=sc, semantics='halfopen', metric=m, value=v))
        if sc == 'prosecutor':
            for m, v in cr.items():
                rows.append(dict(dataset=ds, repeat=repeat, fold=fold, k=k, method=key_method,
                                 scenario=sc, semantics='class', metric=m, value=v))
            rows.append(dict(dataset=ds, repeat=repeat, fold=fold, k=k, method=key_method,
                             scenario=sc, semantics='class', metric='leaves', value=len(leaves)))
    return rows


def main(mode, ds, repeat, fold):
    out = OUTD / f'{mode}_{ds}_{repeat}_{fold}.csv'
    if out.exists():
        print('exists', out); return
    t0 = time.perf_counter(); rows = []
    if mode == 'rrep':
        proto = dict(C.R_REP_ADULT if ds == 'Adult' else C.R_REP_BANK)
        SWAP = {0.8: 0.2, 0.2: 0.8, 0.5: 0.5}
        variants = {'UA-Rrep-equal': {a: 0.5 for a in proto},
                    'UA-Rrep-swap': {a: SWAP[v] for a, v in proto.items()}}
        for meth, rr in variants.items():
            for rp, fd, tr, te, pdd, base in H.iter_splits(ds, repeats=(repeat,), folds=(fold,), r_rep=rr):
                for k in KS:
                    root, leaves, _ = H.build_tree(pdd, tr, k, 'UA-Local')
                    rows += one_tree(base, tr, te, leaves, root, 'UA-Local', ds, rp, fd, k, meth,
                                     ['prosecutor'], ['prosecutor'],
                                     lambda sc: ref_post('rrep_variants', ds, rp, fd, k, meth))
    else:
        for rp, fd, tr, te, pdd, base in H.iter_splits(ds, repeats=(repeat,), folds=(fold,)):
            for k in KS:
                if mode == 'main':
                    for m in METHODS:
                        root, leaves, _ = H.build_tree(pdd, tr, k, m)
                        ver = ['prosecutor'] + (SECONDARY if (rp, fd) == (0, 1) else [])
                        rows += one_tree(pdd, tr, te, leaves, root, m, ds, rp, fd, k, m,
                                         ['prosecutor'] + SECONDARY, ver,
                                         lambda sc, m=m, k=k: ref_main(ds, rp, fd, k, m, sc))
                else:
                    for a in [0.25, 0.65, 0.8, 0.9]:
                        meth = f'UA-a{a}'
                        root, leaves, _ = H.build_tree(pdd, tr, k, meth, alpha=a)
                        rows += one_tree(pdd, tr, te, leaves, root, meth, ds, rp, fd, k, meth,
                                         ['prosecutor'], ['prosecutor'],
                                         lambda sc, meth=meth, k=k: ref_post('alpha_grid', ds, rp, fd, k, meth))
                print(f'{mode} {ds} r{repeat} f{fold} k={k}: {time.perf_counter()-t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out.with_suffix('.tmp'), index=False)
    out.with_suffix('.tmp').rename(out)
    print(f'DONE {mode} {ds} r{repeat} f{fold}: {len(rows)} rows, {time.perf_counter()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
