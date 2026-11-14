"""Разведочный расчёт: размахи Ĩ (локальная MI на корне) и w_sens по QI на корне дерева,
10 сплитов; α*, при котором α·range(Ĩ) = (1−α)·range(w_sens). Запуск из code/."""
import sys
sys.path.insert(0, '.')
import numpy as np
import config as C, data as datalib, core, cv
for ds_name in ['Adult', 'Bank']:
    df_raw = datalib.load_adult(C.DATA_DIR) if ds_name == 'Adult' else datalib.load_bank(C.DATA_DIR)
    labels = C.ADULT_LABELS if ds_name == 'Adult' else C.BANK_LABELS
    y_col = C.ADULT_Y if ds_name == 'Adult' else C.BANK_Y
    li = {l: i for i, l in enumerate(labels)}
    y = np.array([li[v] for v in df_raw[y_col].to_numpy(dtype=object)])
    res = []
    for rep in (0, 1):
        fo = cv.assign_folds(y, df_raw['row_id'].to_numpy(np.int64), rep)
        for f in (1, 2, 3, 4, 5):
            tr = np.flatnonzero(fo != f)
            ds = datalib.adult_dataset(df_raw, 'SA_NEQ_Y') if ds_name == 'Adult' else \
                datalib.bank_dataset(df_raw, datalib.QuantileBinner.fit(df_raw['balance'].to_numpy(np.float64)[tr]))
            p = core.prepare(ds, tr, beta=C.BETA)
            mn = core.normalize_mi(core.node_mi_scores(p, tr))
            w = np.array([p.w_sens[a] for a in p.qi_order]); i = np.array([mn[a] for a in p.qi_order])
            rw, ri = w.max() - w.min(), i.max() - i.min()
            res.append((w.min(), w.max(), i.min(), i.max(), rw / (rw + ri)))
    r = np.array(res)
    print(ds_name, 'w_sens %.3f–%.3f' % (r[:, 0].min(), r[:, 1].max()), 'Ĩ %.3f–%.3f' % (r[:, 2].min(), r[:, 3].max()),
          'alpha* %.3f–%.3f (mean %.3f)' % (r[:, 4].min(), r[:, 4].max(), r[:, 4].mean()))
