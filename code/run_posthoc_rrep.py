"""Разведочный расчёт (вне протокола): чувствительность UA-Local к экспертным весам R_rep.
Варианты: equal — R_rep = 0,5 у всех QI; swap — High ↔ Low (0,8 ↔ 0,2), Medium без изменений.
w_sens пересчитывается core.prepare с новыми R_rep; NCP/DM_w и атака считаются с
протокольными весами (для сопоставимости с B4). 2 варианта × 5 k × 10 сплитов × 2 датасета.
Также: ρ Спирмена Score на корне (вариант против протокольного UA-Local).
Контроль: путь с заменой R_rep на протокольные значения, UA-Local, r0 f1 k = 10, сверка с logs.
Выход: logs_posthoc/rrep_variants.csv, logs_posthoc/rrep_root_rho.csv.
Запуск из code/: python run_posthoc_rrep.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr
import posthoc_common as H
import config as C
import core
import preprocess
OUT = H.LOGP / 'rrep_variants.csv'; RHO = H.LOGP / 'rrep_root_rho.csv'
KEYS = ['dataset', 'repeat', 'fold', 'k', 'method']
done = H.done_keys(OUT, KEYS)
SWAP = {0.8: 0.2, 0.2: 0.8, 0.5: 0.5}
t0 = time.perf_counter(); n = 0
for ds_name in ['Adult', 'Bank']:
    proto = dict(C.R_REP_ADULT if ds_name == 'Adult' else C.R_REP_BANK)
    variants = {'UA-Rrep-equal': {a: 0.5 for a in proto}, 'UA-Rrep-swap': {a: SWAP[v] for a, v in proto.items()}}
    # контроль пути с заменой R_rep
    for repeat, fold, tr, te, pdd, base in H.iter_splits(ds_name, repeats=(0,), folds=(1,), r_rep=proto):
        root, leaves, rf = H.build_tree(pdd, tr, 10, 'UA-Local')
        tm = H.tree_metrics(base, tr, leaves, 10, ds_name, 0, 1, 'UA-Local')
        xtr, xte = preprocess.build_d_ml(pdd, leaves, root, rf, tr, te)
        ml = {c: H.fit_eval(pdd, xtr, xte, tr, te, 0, 1, c) for c in ('RF', 'XGBoost')}
        print(f'[{H.now()}] CONTROL {ds_name} R_rep path: OK, max rel diff {H.control_check(ds_name, 0, 1, 10, "UA-Local", tm, ml):.2e}, AUC max rel dev {H.AUC_DEV[0]:.1e}', flush=True)
    for meth, rr in variants.items():
        for repeat, fold, tr, te, pdd, base in H.iter_splits(ds_name, r_rep=rr):
            mn = core.normalize_mi(core.node_mi_scores(base, tr))
            s0 = core.scores_ua(base, mn, 0.5); s1 = core.scores_ua(pdd, mn, 0.5)
            q = base.qi_order
            H.append(RHO, [dict(dataset=ds_name, repeat=repeat, fold=fold, method=meth,
                                rho=spearmanr([s0[a] for a in q], [s1[a] for a in q])[0],
                                top3_proto='|'.join(core.sorted_attrs(base, s0)[:3]),
                                top3_variant='|'.join(core.sorted_attrs(pdd, s1)[:3]))]) if (ds_name, str(repeat), str(fold), '5', meth) not in done else None
            for k in [5, 10, 25, 50, 100]:
                if (ds_name, str(repeat), str(fold), str(k), meth) in done:
                    continue
                ti = time.perf_counter()
                root, leaves, rf = H.build_tree(pdd, tr, k, 'UA-Local')
                br = dict(dataset=ds_name, repeat=repeat, fold=fold, k=k, method=meth)
                tm = H.tree_metrics(base, tr, leaves, k, ds_name, repeat, fold, meth)
                rows = [dict(br, classifier='NA', metric=m, value=v) for m, v in tm.items()]
                xtr, xte = preprocess.build_d_ml(pdd, leaves, root, rf, tr, te)
                for c in ('RF', 'XGBoost'):
                    rows += [dict(br, classifier=c, metric=m, value=v)
                             for m, v in H.fit_eval(pdd, xtr, xte, tr, te, repeat, fold, c).items()]
                H.append(OUT, rows); n += 1
                print(f'[{H.now()}] {ds_name} {meth} r{repeat} f{fold} k={k}: {time.perf_counter()-ti:.1f}s '
                      f'[{n} items, {time.perf_counter()-t0:.0f}s]', flush=True)
print(f'[{H.now()}] DONE rrep, {n} items, {time.perf_counter()-t0:.0f}s', flush=True)
