"""Сводки разведочного пересчёта атак с точной семантикой разреза (вне протокола).
Вход: logs_posthoc/halfopen/{main,alpha,rrep}_*.csv (run_posthoc_attacks_halfopen.py).
Выход (results_posthoc/): attacks_halfopen_summary.csv (формат results/attacks_summary.csv),
risk_ci_halfopen.csv (UA-Local vs B4; относит. разность, NB-ДИ, Holm внутри датасета:
семейство halfopen {LR, SAI}×5k и отдельно семейство class {1/|EC|, SAI по классу}×5k),
h1_dlink_halfopen.csv, alpha_grid_summary_halfopen.csv, rrep_summary_halfopen.csv.
Запуск из code/: python analysis/halfopen_summary.py"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd, scipy.stats as sps
from scipy.stats import spearmanr
import config as C
from stats import corrected_t, holm_stepdown

L = Path('logs_posthoc/halfopen'); R = Path('results_posthoc')
KEY = ['dataset', 'k', 'repeat', 'fold']
def load(mode):
    fs = sorted(L.glob(f'{mode}_*.csv'))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), len(fs)
main, nf = load('main'); assert nf == 20, nf
assert main.groupby(['dataset', 'method', 'k', 'scenario', 'semantics', 'metric']).size().eq(10).all()

# 1. summary в формате attacks_summary.csv
def q(x, p): return float(np.quantile(x, p))
rows = []
for (ds, k, m, sc, met), g in main.groupby(['dataset', 'k', 'method', 'scenario', 'metric']):
    v = g.value.to_numpy(float)
    rows.append(dict(dataset=ds, k=k, method=m, scenario=sc, metric=met, n=len(v), mean=v.mean(),
                     sd=v.std(ddof=1), median=np.median(v), iqr_lo=q(v, .25), iqr_hi=q(v, .75)))
S = pd.DataFrame(rows); S.to_csv(R / 'attacks_halfopen_summary.csv', index=False)
W = main[main.scenario == 'prosecutor'].pivot_table(index=['method', *KEY], columns='metric', values='value')

# 2. UA-Local vs B4 — относительная разность, NB-ДИ, p против 0, Holm
tcrit = float(sps.t.ppf(0.975, C.H1_DF))
def p0(x):
    se = np.sqrt(x.var(ddof=1) * C.NB_SE_FACTOR); return float(2 * sps.t.sf(abs(x.mean() / se), C.H1_DF))
def rel(a, b, met, ds):
    rr = []
    for k in C.K_GRID:
        x = W.loc[a].xs((ds, k), level=[0, 1])[met].sort_index().to_numpy()
        y = W.loc[b].xs((ds, k), level=[0, 1])[met].sort_index().to_numpy()
        r = (x - y) / y; c = corrected_t(r)
        rr.append(dict(dataset=ds, metric=met, k=k, A=a, B=b, mean_A=x.mean(), mean_B=y.mean(),
                       diff=(x - y).mean(), rel_pct=100 * r.mean(), rel_lo=100 * c['ci_lo'],
                       rel_hi=100 * c['ci_hi'], p_two=p0(r), a_lower=int((x < y).sum())))
    return rr
ci = []
for ds in ['Adult', 'Bank']:
    for fam in (('linkage_risk', 'sai_gain'), ('class_risk', 'class_sai_gain')):
        part = pd.DataFrame(sum([rel('UA-Local', 'B4', m, ds) for m in fam], []))
        part['holm_p'] = pd.Series(holm_stepdown({i: part.loc[i, 'p_two'] for i in part.index}))
        part['family'] = '+'.join(fam); ci.append(part)
CI = pd.concat(ci, ignore_index=True); CI.to_csv(R / 'risk_ci_halfopen.csv', index=False)

# 3. H1: mean ΔLink (UA-Local − B4) по primary
h1 = []
for ds in ['Adult', 'Bank']:
    for k in C.K_GRID:
        x = W.loc['UA-Local'].xs((ds, k), level=[0, 1])['linkage_risk']
        y = W.loc['B4'].xs((ds, k), level=[0, 1])['linkage_risk']
        old = pd.read_csv('results/h1_pairs.csv')
        h1.append(dict(dataset=ds, k=k, mean_dlink_halfopen=(x - y).mean()))
H1 = pd.DataFrame(h1); H1.to_csv(R / 'h1_dlink_halfopen.csv', index=False)

# 4. α-сетка и варианты R_rep: LinkageRisk/SAI-gain — по полуоткрытой семантике,
#    Macro-F1 — из alpha_grid_summary.csv / rrep_summary.csv (от атаки не зависит)
def rel_post(mode, oldfile):
    d, n = load(mode); assert n == 20, (mode, n)
    d = d[(d.scenario == 'prosecutor') & (d.semantics == 'halfopen') & d.metric.isin(['linkage_risk', 'sai_gain'])]
    w = d.pivot_table(index=['method', *KEY], columns='metric', values='value')
    extra = {'alpha': {'UA-a0.0': 'A2', 'UA-a0.5': 'UA-Local', 'UA-a1.0': 'A1'},
             'rrep': {'UA-Local(proto)': 'UA-Local'}}[mode]
    old = pd.read_csv(R / oldfile); out = []
    for _, r in old.iterrows():
        src = extra.get(r.method)
        g = (W.loc[src] if src else w.loc[r.method]).xs((r.dataset, r.k), level=[0, 1]).sort_index()
        b = W.loc['B4'].xs((r.dataset, r.k), level=[0, 1]).sort_index()
        r = r.copy()
        for met, col in (('linkage_risk', 'LR'), ('sai_gain', 'SAI')):
            x, y = g[met].to_numpy(), b[met].to_numpy(); rr = (x - y) / y; c = corrected_t(rr)
            r[col] = x.mean(); r[f'd{col}%'] = 100 * rr.mean(); r[f'd{col}%_lo'] = 100 * c['ci_lo']; r[f'd{col}%_hi'] = 100 * c['ci_hi']
        out.append(r)
    o = pd.DataFrame(out); o.to_csv(R / oldfile.replace('.csv', '_halfopen.csv'), index=False); return o
AL = rel_post('alpha', 'alpha_grid_summary.csv'); RR = rel_post('rrep', 'rrep_summary.csv')

# 5. Утверждения текста
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('== Таблица 4 (halfopen + class)')
print(CI.round(4).to_string(index=False))
print('== H1 ΔLink'); print(H1.round(5).to_string(index=False))
old_at = pd.read_csv('results/attacks_summary.csv'); old_at = old_at[old_at.scenario == 'prosecutor']
pm = S[(S.scenario == 'prosecutor')]
ratio = pm[pm.metric == 'linkage_risk'].merge(old_at[old_at.metric == 'linkage_risk'], on=['dataset', 'k', 'method'])
ratio['r'] = ratio.mean_x / ratio.mean_y
print('== ratio halfopen/closed LR by dataset:', ratio.groupby('dataset').r.agg(['min', 'max']).round(3).to_dict())
cls = pm[pm.metric == 'class_risk'].merge(pm[pm.metric == 'linkage_risk'], on=['dataset', 'k', 'method'])
cls['share'] = cls.mean_y / cls.mean_x
print('== LR/class share range:', cls.groupby('dataset').share.agg(['min', 'max']).round(3).to_dict())
anon = ['B2', 'B4', 'B4m', 'A1', 'A2', 'UA-Global', 'UA-Local']
for met in ['linkage_risk', 'sai_gain', 'class_risk']:
    t = pm[pm.metric == met].pivot_table(index=['dataset', 'k'], columns='method', values='mean')[anon]
    print(f'== {met}: argmin / argmax by (dataset,k)')
    print(pd.DataFrame({'min': t.idxmin(1), 'max': t.idxmax(1)}).to_string())
print('== max SD (B4/UA-Local, LR,SAI):', S[(S.scenario == 'prosecutor') & S.method.isin(['B4', 'UA-Local']) & S.metric.isin(['linkage_risk', 'sai_gain'])].groupby('dataset').sd.max().to_dict())
print('== unique_match_rate max / candidate_rate min:', pm[pm.metric == 'unique_match_rate']['mean'].max(), pm[pm.metric == 'candidate_rate']['mean'].min())
print('== coverage primary min:', main[(main.scenario == 'prosecutor') & (main.metric == 'coverage')].value.min())
b4m = pm[pm.metric.isin(['linkage_risk', 'sai_gain'])].pivot_table(index=['dataset', 'k', 'metric'], columns='method', values='mean')
print('== max |B4m−B4|, |A1−B4|:', float((b4m['B4m'] - b4m['B4']).abs().max()), float((b4m['A1'] - b4m['B4']).abs().max()))
for ds in ['Adult', 'Bank']:
    for met in ['linkage_risk', 'sai_gain']:
        rr = pd.DataFrame(rel('A2', 'A1', met, ds))
        print(f'== A2 vs A1 {ds} {met}:', rr[['k', 'rel_pct', 'rel_lo', 'rel_hi']].round(1).values.tolist())
# вторичные сценарии
sec = S[S.metric == 'linkage_risk'].pivot_table(index=['scenario', 'dataset', 'k'], columns='method', values='mean')[anon]
cov = S[(S.metric == 'coverage')].groupby('scenario')['mean'].agg(['min', 'max'])
print('== coverage by scenario'); print(cov.round(4).to_string())
for sc in ['journalist', 'journalist_r3', 'journalist_r5', 'noise5']:
    rhos = [spearmanr(sec.loc[('prosecutor', ds, k)], sec.loc[(sc, ds, k)])[0] for ds in ['Adult', 'Bank'] for k in C.K_GRID]
    rd = {ds: [spearmanr(sec.loc[('prosecutor', ds, k)], sec.loc[(sc, ds, k)])[0] for k in C.K_GRID] for ds in ['Adult', 'Bank']}
    d = sec.loc[sc]; dd = (d['UA-Local'] - d['B4'])
    print(f'== {sc}: rho min/max {min(rhos):.2f}/{max(rhos):.2f}', {k: np.round(v, 2).tolist() for k, v in rd.items()},
          '| UA−B4 Adult:', dd.loc['Adult'].round(5).tolist(), '| Bank:', dd.loc['Bank'].round(5).tolist())
print('== Adult k=10 r3: B4', round(sec.loc[('journalist_r3', 'Adult', 10), 'B4'], 5), 'UA', round(sec.loc[('journalist_r3', 'Adult', 10), 'UA-Local'], 5))
print('== journalist vs primary (Adult k=10 B4):', round(sec.loc[('journalist', 'Adult', 10), 'B4'], 5), round(sec.loc[('prosecutor', 'Adult', 10), 'B4'], 5))
print('== alpha grid (halfopen):'); print(AL[['method', 'dataset', 'k', 'dF1_RF', 'dF1_XGB', 'dLR%', 'dLR%_lo', 'dLR%_hi', 'dSAI%']].round(4).to_string(index=False))
print('== rrep (halfopen):'); print(RR[['method', 'dataset', 'k', 'dF1_RF', 'dF1_XGB', 'dLR%', 'dLR%_lo', 'dLR%_hi', 'dSAI%']].round(4).to_string(index=False))
