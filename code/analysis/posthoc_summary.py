"""Сводки разведочных расчётов (α-сетка, варианты P_aux и R_rep) -> results_posthoc/*.csv. Парные разности с B4 по 10 сплитам;
ДИ Nadeau–Bengio (stats.corrected_t: берутся только ci_lo/ci_hi). Запуск из code/."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd
from stats import corrected_t
R = Path('results_posthoc'); L = Path('logs_posthoc')
ml = pd.read_csv('logs/runs_ml.csv'); ml = ml[ml.metric == 'macro_f1'].dropna(subset=['k'])
ml['k'] = ml.k.astype(int)
at = pd.read_csv('logs/runs_attacks.csv'); at = at[(at.scenario == 'prosecutor') & at.metric.isin(['linkage_risk', 'sai_gain'])]
pr = pd.read_csv('logs/runs_privacy.csv'); pr = pr[pr.metric == 'ncp']
KEY = ['dataset', 'k', 'repeat', 'fold']

def proto(method):
    """Длинная таблица протокольного метода: F1_RF, F1_XGB, LR, SAI, NCP по сплитам."""
    a = ml[ml.method == method].pivot_table(index=KEY, columns='classifier', values='value')
    a.columns = ['F1_' + ('XGB' if c == 'XGBoost' else c) for c in a.columns]
    b = at[at.method == method].groupby(KEY + ['metric']).value.first().unstack('metric')
    b = b.rename(columns={'linkage_risk': 'LR', 'sai_gain': 'SAI'})
    c = pr[pr.method == method].groupby(KEY).value.first().rename('NCP')
    return a.join(b).join(c)

def posthoc(path, extra_key=()):
    d = pd.read_csv(path)
    d['classifier'] = d['classifier'].fillna('NA')  # строка "NA" читается pandas как NaN
    d['col'] = np.where(d.classifier == 'NA', d.metric, 'F1_' + d.classifier.replace({'XGBoost': 'XGB'}))
    d = d[d.col.isin(['F1_RF', 'F1_XGB', 'linkage_risk', 'sai_gain', 'ncp']) &
          ((d.classifier == 'NA') | (d.metric == 'macro_f1'))]
    w = d.pivot_table(index=['method', *extra_key, *KEY], columns='col', values='value')
    return w.rename(columns={'linkage_risk': 'LR', 'sai_gain': 'SAI', 'ncp': 'NCP'})

B4 = proto('B4')

def compare(df, label_cols):
    rows = []
    for lab, g in df.groupby(label_cols):
        g = g.droplevel(label_cols) if isinstance(g.index, pd.MultiIndex) else g
        for (ds, k), h in g.groupby(level=[0, 1]):
            b = B4.loc[h.index]
            assert len(h) == 10, (lab, ds, k, len(h))
            r = dict(zip(label_cols, lab if isinstance(lab, tuple) else (lab,)))
            r.update(dataset=ds, k=k)
            for m in ['F1_RF', 'F1_XGB', 'LR', 'SAI', 'NCP']:
                if m not in h: continue
                r[m] = h[m].mean()
                diff = (h[m] - b[m]).to_numpy()
                if m.startswith('F1'):
                    c = corrected_t(diff); r[f'd{m}'] = diff.mean(); r[f'd{m}_lo'] = c['ci_lo']; r[f'd{m}_hi'] = c['ci_hi']
                else:
                    rel = diff / b[m].to_numpy(); c = corrected_t(rel)
                    r[f'd{m}%'] = 100 * rel.mean(); r[f'd{m}%_lo'] = 100 * c['ci_lo']; r[f'd{m}%_hi'] = 100 * c['ci_hi']
            rows.append(r)
    return pd.DataFrame(rows)

out = {}
if (L / 'alpha_grid.csv').exists() and 'noalpha' not in sys.argv:
    g = posthoc(L / 'alpha_grid.csv')
    parts = [g]
    for meth, a in (('A2', 0.0), ('UA-Local', 0.5), ('A1', 1.0)):
        p = proto(meth); p['method'] = f'UA-a{a}'; parts.append(p.set_index('method', append=True).reorder_levels(['method', *KEY]))
    g = pd.concat(parts).sort_index()
    s = compare(g, ['method']); s['alpha'] = s.method.str[4:].astype(float)
    s = s.sort_values(['dataset', 'k', 'alpha']); s.to_csv(R / 'alpha_grid_summary.csv', index=False); out['alpha'] = s
if (L / 'rrep_variants.csv').exists():
    g = posthoc(L / 'rrep_variants.csv')
    p = proto('UA-Local'); p['method'] = 'UA-Local(proto)'
    g = pd.concat([g, p.set_index('method', append=True).reorder_levels(['method', *KEY])]).sort_index()
    s = compare(g, ['method']); s.to_csv(R / 'rrep_summary.csv', index=False); out['rrep'] = s
    rho = pd.read_csv(L / 'rrep_root_rho.csv')
    rho.groupby(['dataset', 'method']).agg(rho_min=('rho', 'min'), rho_max=('rho', 'max'),
                                           top3_variant=('top3_variant', lambda x: sorted(set(x))),
                                           top3_proto=('top3_proto', lambda x: sorted(set(x)))).to_csv(R / 'rrep_root_rho_summary.csv')
if (L / 'paux_variants.csv').exists():
    d = pd.read_csv(L / 'paux_variants.csv'); d = d[d.metric == 'macro_f1']
    full = ml[ml.method.isin(['B4', 'UA-Local'])].assign(variant='full')
    d = pd.concat([d, full[d.columns.intersection(full.columns)]])
    d = d[~((d.variant == 'full') & d.duplicated(['dataset', 'repeat', 'fold', 'k', 'method', 'classifier', 'variant'], keep='last'))]
    w = d.pivot_table(index=['dataset', 'variant', 'classifier', 'k', 'repeat', 'fold'], columns='method', values='value')
    rows = []
    for (ds, v, c, k), h in w.groupby(level=[0, 1, 2, 3]):
        assert len(h) == 10, (ds, v, c, k, len(h))
        diff = (h['UA-Local'] - h['B4']).to_numpy(); ct = corrected_t(diff)
        rows.append(dict(dataset=ds, variant=v, classifier=c, k=k, B4=h['B4'].mean(), UA_Local=h['UA-Local'].mean(),
                         dF1=diff.mean(), dF1_lo=ct['ci_lo'], dF1_hi=ct['ci_hi'], ua_wins=int((diff > 0).sum())))
    s = pd.DataFrame(rows); s.to_csv(R / 'paux_summary.csv', index=False); out['paux'] = s
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for k, v in out.items():
    print('==', k); print(v.round(4).to_string(index=False))
