"""Разведочный расчёт (вне протокола): относительная разность риска
UA-Local против B4, основной сценарий (prosecutor), исходная (замкнутая) семантика интервалов, LinkageRisk и SAI-gain.
r_s = (UA-Local_s − B4_s)/B4_s по 10 сплитам; SE = sqrt(s²·0,35) (Nadeau–Bengio),
95 % ДИ r̄ ± t(0,975; 9)·SE; p двусторонний против 0 (t = r̄/SE, df = 9) -> p_two/holm_p;
Holm по 10 сравнениям внутри датасета (2 метрики × 5 k).
Для сравнения также p по абсолютной разности (t = mean(UA−B4)/SE_abs) -> p_two_abs/holm_p_abs.
Запуск из code/: python analysis/risk_ci_posthoc.py"""
import sys
sys.path.insert(0, '.')
import numpy as np, pandas as pd, scipy.stats as sps
import config as C
from stats import corrected_t, holm_stepdown

a = pd.read_csv('logs/runs_attacks.csv')
a = a[(a.scenario == 'prosecutor') & a.metric.isin(['linkage_risk', 'sai_gain'])
      & a.method.isin(['B4', 'UA-Local'])]
key = ['dataset', 'metric', 'k', 'repeat', 'fold', 'method']
g = a.groupby(key)['value']
assert (g.nunique() == 1).all(), 'значения атаки различаются между классификаторами'
w = g.first().unstack('method').reset_index()
tcrit = float(sps.t.ppf(0.975, C.H1_DF))


def p_vs_zero(x):
    se = np.sqrt(x.var(ddof=1) * C.NB_SE_FACTOR)
    return float(2 * sps.t.sf(abs(x.mean() / se), C.H1_DF))


rows = []
for (ds, met, k), d in w.groupby(['dataset', 'metric', 'k']):
    d = d.sort_values(['repeat', 'fold'])
    assert len(d) == 10
    b4, ua = d['B4'].to_numpy(), d['UA-Local'].to_numpy()
    diff = ua - b4
    r = diff / b4
    cd = corrected_t(diff); cr = corrected_t(r)   # берём только ci_lo/ci_hi
    rows.append(dict(dataset=ds, metric=met, k=int(k), method='UA-Local', n=10,
                     B4=b4.mean(), UA_Local=ua.mean(), diff=diff.mean(),
                     ci_lo=cd['ci_lo'], ci_hi=cd['ci_hi'],
                     rel_pct=100 * r.mean(), rel_lo=100 * cr['ci_lo'], rel_hi=100 * cr['ci_hi'],
                     p_two=p_vs_zero(r), p_two_abs=p_vs_zero(diff),
                     wins_lower=int((ua < b4).sum())))
out = pd.DataFrame(rows)
for pc, hc in (('p_two', 'holm_p'), ('p_two_abs', 'holm_p_abs')):
    out[hc] = np.nan
    for ds, d in out.groupby('dataset'):
        for i, v in holm_stepdown({i: d.loc[i, pc] for i in d.index}).items():
            out.loc[i, hc] = v
out = out.sort_values(['dataset', 'metric', 'k']).reset_index(drop=True)
out.to_csv('results_posthoc/risk_ci_nb.csv', index=False)

pd.set_option('display.width', 220)
print(out[['dataset', 'metric', 'k', 'B4', 'UA_Local', 'diff', 'rel_pct', 'rel_lo', 'rel_hi',
           'wins_lower', 'holm_p', 'holm_p_abs']].round(4).to_string())
