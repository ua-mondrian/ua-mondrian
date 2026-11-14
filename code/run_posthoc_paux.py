"""Разведочный расчёт (вне протокола): чувствительность ML-оценки к составу D_ML.
B4 и UA-Local, все k, 10 сплитов, RF и XGBoost. Дерево строится один раз, затем
обучаются варианты D_ML: QIonly (только QI*, без P_aux); Adult — без education-num;
Bank — без duration. Полный D_ML (контроль) — на (repeat 0, fold 1) при всех k,
сверка с logs/runs_ml.csv (rtol 1e-13); в остальных сплитах полный D_ML берётся из логов.
H1 в этих режимах не проверяется. Выход: logs_posthoc/paux_variants.csv.
Запуск из code/: python run_posthoc_paux.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_common as H
import preprocess
OUT = H.LOGP / 'paux_variants.csv'
KEYS = ['dataset', 'repeat', 'fold', 'k', 'method']
done = H.done_keys(OUT, KEYS)
EXTRA = {'Adult': ('noEduNum', ['education-num']), 'Bank': ('noDuration', ['duration'])}


t0 = time.perf_counter(); n = 0; worst = 0.0
for ds_name in ['Adult', 'Bank']:
    vname, vcols = EXTRA[ds_name]
    for repeat, fold, tr, te, pdd, base in H.iter_splits(ds_name):
        for k in [5, 10, 25, 50, 100]:
            for meth in ['B4', 'UA-Local']:
                if (ds_name, str(repeat), str(fold), str(k), meth) in done:
                    continue
                ti = time.perf_counter()
                root, leaves, rf = H.build_tree(pdd, tr, k, meth)
                xtr, xte = preprocess.build_d_ml(pdd, leaves, root, rf, tr, te)
                br = dict(dataset=ds_name, repeat=repeat, fold=fold, k=k, method=meth)
                rows = []
                if repeat == 0 and fold == 1:
                    ml = {c: H.fit_eval(pdd, xtr, xte, tr, te, repeat, fold, c) for c in ('RF', 'XGBoost')}
                    worst = max(worst, H.control_check(ds_name, repeat, fold, k, meth, None, ml))
                    for c, vals in ml.items():
                        rows += [dict(br, classifier=c, variant='full', metric=m, value=v) for m, v in vals.items()]
                qcols = list(pdd.ds.p_aux_num) + [c for c in xtr.columns for a in pdd.ds.p_aux_cat if c.startswith(f'{a}=')]
                for variant, cols in (('QIonly', qcols), (vname, vcols)):
                    assert all(c in xtr.columns for c in cols), cols
                    a_tr, a_te = xtr.drop(columns=cols), xte.drop(columns=cols)
                    for c in ('RF', 'XGBoost'):
                        rows += [dict(br, classifier=c, variant=variant, metric=m, value=v)
                                 for m, v in H.fit_eval(pdd, a_tr, a_te, tr, te, repeat, fold, c).items()]
                H.append(OUT, rows); n += 1
                print(f'[{H.now()}] {ds_name} r{repeat} f{fold} k={k} {meth}: {time.perf_counter()-ti:.1f}s '
                      f'[{n} items, {time.perf_counter()-t0:.0f}s, control max rel diff {worst:.1e}]', flush=True)
print(f'[{H.now()}] DONE paux, {n} items, {time.perf_counter()-t0:.0f}s, control OK (max rel diff {worst:.2e}, AUC max rel dev {H.AUC_DEV[0]:.1e})', flush=True)
