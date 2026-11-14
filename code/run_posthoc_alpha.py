"""Разведочный расчёт (вне протокола): α-сетка для UA-Local (локальная MI, β = 0,5).
α ∈ {0,25; 0,65; 0,8; 0,9}; α = 0 / 0,5 / 1 уже есть в logs (A2 / UA-Local / A1).
2 датасета × 10 сплитов × 5 k; RF и XGBoost; primary-атака; NCP/DM_w.
Контроль: на (repeat 0, fold 1, k = 10) UA-Local при α = 0,5 тем же путём кода
сверяется с logs/*.csv (rtol 1e-13). Выход: logs_posthoc/alpha_grid.csv.
Запуск из code/: python run_posthoc_alpha.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_common as H
import preprocess
ALPHAS = [0.25, 0.65, 0.8, 0.9]
OUT = H.LOGP / 'alpha_grid.csv'
KEYS = ['dataset', 'repeat', 'fold', 'k', 'method']
done = H.done_keys(OUT, KEYS)
t0 = time.perf_counter(); n = 0
for ds_name in ['Adult', 'Bank']:
    for repeat, fold, tr, te, pdd, base in H.iter_splits(ds_name):
        if repeat == 0 and fold == 1:
            root, leaves, rf = H.build_tree(pdd, tr, 10, 'UA-Local', alpha=0.5)
            tm = H.tree_metrics(pdd, tr, leaves, 10, ds_name, 0, 1, 'UA-Local')
            xtr, xte = preprocess.build_d_ml(pdd, leaves, root, rf, tr, te)
            ml = {c: H.fit_eval(pdd, xtr, xte, tr, te, 0, 1, c) for c in ('RF', 'XGBoost')}
            w = H.control_check(ds_name, 0, 1, 10, 'UA-Local', tm, ml)
            print(f'[{H.now()}] CONTROL {ds_name} UA-Local α=0.5 r0 f1 k=10: OK, max rel diff {w:.2e}, AUC max rel dev {H.AUC_DEV[0]:.1e}', flush=True)
        for k in [5, 10, 25, 50, 100]:
            for a in ALPHAS:
                meth = f'UA-a{a}'
                if (ds_name, str(repeat), str(fold), str(k), meth) in done:
                    continue
                ti = time.perf_counter()
                root, leaves, rf = H.build_tree(pdd, tr, k, meth, alpha=a)
                base_row = dict(dataset=ds_name, repeat=repeat, fold=fold, k=k, method=meth, alpha=a)
                rows = []
                tm = H.tree_metrics(pdd, tr, leaves, k, ds_name, repeat, fold, meth)
                rows += [dict(base_row, classifier='NA', metric=m, value=v) for m, v in tm.items()]
                xtr, xte = preprocess.build_d_ml(pdd, leaves, root, rf, tr, te)
                for c in ('RF', 'XGBoost'):
                    rows += [dict(base_row, classifier=c, metric=m, value=v)
                             for m, v in H.fit_eval(pdd, xtr, xte, tr, te, repeat, fold, c).items()]
                H.append(OUT, rows); n += 1
                print(f'[{H.now()}] {ds_name} r{repeat} f{fold} k={k} α={a}: {time.perf_counter()-ti:.1f}s '
                      f'[{n} items, {time.perf_counter()-t0:.0f}s]', flush=True)
print(f'[{H.now()}] DONE alpha grid, {n} items, {time.perf_counter()-t0:.0f}s', flush=True)
