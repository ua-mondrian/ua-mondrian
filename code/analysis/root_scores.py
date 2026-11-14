import sys; sys.path.insert(0, ".")
import numpy as np, config as C, data as datalib, core, cv
from scipy.stats import spearmanr
for ds_name in ["Adult","Bank"]:
    df_raw = datalib.load_adult(C.DATA_DIR) if ds_name=="Adult" else datalib.load_bank(C.DATA_DIR)
    labels = C.ADULT_LABELS if ds_name=="Adult" else C.BANK_LABELS
    y_col = C.ADULT_Y if ds_name=="Adult" else C.BANK_Y
    li={l:i for i,l in enumerate(labels)}
    y=np.array([li[v] for v in df_raw[y_col].to_numpy(dtype=object)])
    orders=set(); wr=[]; ir=[]; rUA_A2=[]; rUA_IG=[]
    for rep in (0,1):
        fo=cv.assign_folds(y, df_raw["row_id"].to_numpy(np.int64), rep)
        for f in (1,2,3,4,5):
            tr=np.flatnonzero(fo!=f)
            ds=datalib.adult_dataset(df_raw,"SA_NEQ_Y") if ds_name=="Adult" else datalib.bank_dataset(df_raw, datalib.QuantileBinner.fit(df_raw["balance"].to_numpy(np.float64)[tr]))
            p=core.prepare(ds,tr,beta=0.5); mi=core.node_mi_scores(p,tr); mn=core.normalize_mi(mi)
            ua=core.scores_ua(p,mn,0.5); a2=core.scores_ua(p,mn,0.0)
            orders.add(tuple(core.sorted_attrs(p,ua)[:3]))
            w=np.array([p.w_sens[a] for a in p.qi_order]); i=np.array([mn[a] for a in p.qi_order])
            wr.append((w.min(),w.max())); ir.append((i.min(),i.max()))
            q=p.qi_order
            rUA_A2.append(spearmanr([ua[a] for a in q],[a2[a] for a in q])[0]); rUA_IG.append(spearmanr([ua[a] for a in q],[mi[a] for a in q])[0])
    wr=np.array(wr); ir=np.array(ir)
    print(ds_name,"top3 UA:",orders)
    print(" w_sens range min..max over splits:",wr[:,0].min().round(3),wr[:,1].max().round(3)," Inorm:",ir[:,0].min().round(3),ir[:,1].max().round(3))
    print(" spearman UA~A2",np.round([min(rUA_A2),max(rUA_A2)],2)," UA~IG",np.round([min(rUA_IG),max(rUA_IG)],2))
