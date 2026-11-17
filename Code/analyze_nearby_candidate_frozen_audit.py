"""Post hoc proximity and source-candidate diagnostics; never train or change inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import itertools
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from analyze_v5_constant_baseline_audit import ROOT, V5, SPLITS, NEURAL, SEEDS, prediction_path

B = 20000
# The internal protocol note is not distributed; the same rules are stated in
# supplementary methods S2 and S5 and in manuscript/supplementary_tables/README.md.
PROTOCOL = ROOT/'docs/protocols/nearby_candidate_subset_protocol_20261116.md'
HASHES = {}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def track(p):
    p = Path(p)
    HASHES[str(p.relative_to(ROOT)).replace('\\', '/')] = digest(p)
    return p


def read(p):
    return pd.read_csv(track(p))


def score(y, pred, w):
    err = pred-y
    mse = np.dot(w, err**2)
    var = np.dot(w, (y-np.dot(w, y))**2)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.dot(w, abs(err))),
                r2=float(1-mse/var) if len(y)>1 and var>0 else np.nan,
                bias=float(np.dot(w, err)))


def classify(lo, hi):
    return 'A_lower_error' if hi<0 else 'A_higher_error' if lo>0 else 'unresolved'


def load(split):
    test = read(V5/'p3_protocol'/split/'test_records.csv').sort_values('record_id').set_index('record_id')
    train = read(V5/'p3_protocol'/split/'train_records.csv').sort_values('record_id').set_index('record_id')
    assert test.index.is_unique and train.index.is_unique and not set(test.index)&set(train.index)
    def aligned(p):
        f=read(p).sort_values('record_id').set_index('record_id')
        assert f.index.equals(test.index), str(p)
        np.testing.assert_allclose(f[['lon','lat','log_dep']],test[['lon','lat','log_dep']],rtol=0,atol=1e-12)
        assert np.isfinite(f.prediction_log_dep).all()
        return f.prediction_log_dep.to_numpy(float)
    matrices={m:np.column_stack([aligned(prediction_path(split,m,s)) for s in SEEDS]) for m in NEURAL}
    matrices['PDE-MAP']=aligned(V5/'pde_map'/f"{'p3' if split.startswith('outer_') else 'p3c'}_formal_{split}"/'test_predictions.csv')[:,None]
    gp=V5/'p3_runs'/f"{'p3b' if split.startswith('outer_') else 'p3c'}_{split}_b3_gp_seed42"/split/'B3_gp_rbf/test_predictions.csv'
    if split.startswith('outer_'):
        matrices['GP/RBF']=aligned(gp)[:,None]
    matrices['Train-mean constant']=np.full((len(test),1),float(train.log_dep.mean()))
    return train,test,matrices


def components(full):
    xy=np.radians(full[['lon','lat']].to_numpy(float))
    lon,lat=xy[:,0],xy[:,1]
    a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    distances=2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))
    adjacency=distances<=3
    visited=set(); labels=np.empty(len(full),dtype=int)
    for i in range(len(full)):
        if i in visited: continue
        stack=[i]; members=[];visited.add(i)
        while stack:
            j=stack.pop();members.append(j)
            for k in np.flatnonzero(adjacency[j]):
                if int(k) not in visited: visited.add(int(k));stack.append(int(k))
        labels[members]=int(full.index[members].min())
    return labels,distances


def finish(out, frames, extra):
    out.mkdir()
    for name,frame in frames.items(): frame.to_csv(out/name,index=False)
    for rel,h in HASHES.items(): assert digest(ROOT/rel)==h,rel
    manifest=dict(no_training=True,no_input_changes=True,protocol=str(PROTOCOL.relative_to(ROOT)),
                  source_sha256=HASHES,**extra)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(extra,ensure_ascii=False,indent=2),flush=True)


def near(out):
    train,test,_=load(next(iter(SPLITS)))
    full=pd.concat([train,test]).sort_index()
    assert len(full)==317 and full.index.is_unique
    labels,dist=components(full)
    gseries=pd.Series(labels,index=full.index)
    memberships=full.reset_index()[['record_id','lon','lat','dep','log_dep']]
    memberships['component_3km']=labels
    groups=[]
    for group in np.unique(labels):
        ix=np.flatnonzero(labels==group); f=full.iloc[ix]
        groups.append(dict(component_3km=int(group),n=len(f),n_coordinates=len(f[['lon','lat']].drop_duplicates()),
                           record_ids=';'.join(map(str,f.index)),n_flux_values=f.dep.nunique(),
                           max_pair_distance_km=float(dist[np.ix_(ix,ix)].max())))
    groups=pd.DataFrame(groups)
    assert len(groups[groups.n>1])==10 and groups.loc[groups.n>1,'n'].sum()==28
    old=read(V5/'estimand_site_audit_20261114/metrics_both_estimands.csv')
    oldci=read(V5/'estimand_site_audit_20261114/paired_model_contrasts.csv')
    rng=np.random.default_rng(20261117)
    metrics=[];contrasts=[];splitcounts=[];validation=[]
    for split in SPLITS:
        train,test,matrices=load(split)
        assert set(train.index)|set(test.index)==set(full.index)
        np.testing.assert_allclose(test[['lon','lat','log_dep']],full.loc[test.index,['lon','lat','log_dep']],rtol=0,atol=1e-12)
        ti=full.index.get_indexer(test.index);tr=full.index.get_indexer(train.index)
        shared=set(gseries.loc[test.index])&set(gseries.loc[train.index])
        cross=dist[np.ix_(ti,tr)]
        invnear=np.unique(gseries.loc[test.index],return_inverse=True)[1]
        invcoord=np.unique(test[['lon','lat']].to_numpy(),axis=0,return_inverse=True)[1]
        splitcounts.append(dict(split=split,n_test=len(test),n_coordinate_groups=len(np.unique(invcoord)),
            n_nearby_groups=len(np.unique(invnear)),n_cross_split_components=len(shared),
            n_cross_split_pairs_3km=int((cross<=3).sum()),nearest_cross_split_km=float(cross.min())))
        y=test.log_dep.to_numpy(float);n=len(y);points={}
        schemes=[('record_iid',np.arange(n),False),('coordinate_cluster',invcoord,False),
                 ('nearby_cluster',invnear,False),('nearby_equal',invnear,True)]
        for scheme,inv,equal in schemes:
            g=len(np.unique(inv));sizes=np.bincount(inv);grouping=np.eye(g)[inv].T
            w=1/(g*sizes[inv]) if equal else np.full(n,1/n)
            draws=rng.multinomial(g,np.full(g,1/g),size=B)
            weights=draws/g if equal else draws/(draws@sizes)[:,None]
            expanded=weights[0,inv]/sizes[inv] if equal else weights[0,inv]
            assert abs(expanded.sum()-1)<1e-12
            boots={}
            for model,p in matrices.items():
                for estimand in ['ensemble','mean_seed_metric']:
                    pp=p.mean(axis=1,keepdims=True) if estimand=='ensemble' else p
                    vals=[score(y,pp[:,j],w) for j in range(pp.shape[1])]
                    m={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
                    points[estimand,model]=m
                    metrics.append(dict(split=split,scheme=scheme,estimand=estimand,model=model,**m))
                    for j in [0]:
                        v=score(y,pp[:,j],w)
                        np.testing.assert_allclose([v['rmse'],v['mae'],v['r2']],
                            [np.sqrt(mean_squared_error(y,pp[:,j],sample_weight=w)),mean_absolute_error(y,pp[:,j],sample_weight=w),r2_score(y,pp[:,j],sample_weight=w)],atol=1e-12)
                    if not equal:
                        prior=old[(old.split==split)&(old.weighting=='record_equal')&(old.estimand==estimand)&(old.model==model)].iloc[0]
                        np.testing.assert_allclose([m[k] for k in m],[prior[k] for k in m],atol=1e-11,rtol=0)
                    sq=grouping@((pp-y[:,None])**2);ab=grouping@abs(pp-y[:,None])
                    if equal: sq=sq/sizes[:,None];ab=ab/sizes[:,None]
                    br=np.sqrt(weights@sq).mean(axis=1);ba=(weights@ab).mean(axis=1)
                    boots[estimand,model,'rmse']=br;boots[estimand,model,'mae']=ba
                    expanded_scores=[score(y,pp[:,j],expanded) for j in range(pp.shape[1])]
                    np.testing.assert_allclose([br[0],ba[0]],[np.mean([v[k] for v in expanded_scores]) for k in ['rmse','mae']],atol=1e-12)
            for a,b in itertools.combinations(matrices,2):
                for estimand in ['ensemble','mean_seed_metric']:
                    for k in ['rmse','mae']:
                        d=boots[estimand,a,k]-boots[estimand,b,k];lo,hi=np.quantile(d,[.025,.975])
                        contrasts.append(dict(split=split,scheme=scheme,estimand=estimand,A=a,B=b,metric=k,
                            delta=points[estimand,a][k]-points[estimand,b][k],ci95_low=lo,ci95_high=hi,classification=classify(lo,hi)))
            validation.append(dict(split=split,scheme=scheme,weighted_sklearn_and_first_draw_checks=True))
        print('Completed near: '+split,flush=True)
    contrasts=pd.DataFrame(contrasts);metrics=pd.DataFrame(metrics)
    keys=['split','estimand','A','B','metric']
    primary=contrasts[contrasts.scheme=='coordinate_cluster'][keys+['classification','delta']]
    changes=contrasts[contrasts.scheme.isin(['nearby_cluster','nearby_equal'])].merge(primary,on=keys,suffixes=('','_coordinate'))
    changes['classification_changed']=changes.classification!=changes.classification_coordinate
    changes['point_direction_changed']=np.sign(changes.delta)!=np.sign(changes.delta_coordinate)
    prior=oldci[oldci.scheme=='coordinate_cluster'][keys+['classification']]
    mc=primary.merge(prior,on=keys,suffixes=('','_old'))
    mc=mc[mc.classification!=mc.classification_old]
    # Monte Carlo differences in a boundary CI must be shown, not attributed to proximity.
    winners=metrics[metrics.estimand=='ensemble'].loc[lambda f:f.groupby(['split','scheme']).rmse.idxmin(),['split','scheme','model','rmse']]
    finish(out,{'memberships.csv':memberships,'groups.csv':groups,'split_counts.csv':pd.DataFrame(splitcounts),
        'metrics.csv':metrics,'paired_contrasts.csv':contrasts,'changes_vs_coordinate.csv':changes,
        'coordinate_mc_changes_vs_old.csv':mc,'ensemble_rmse_winners.csv':winners,'validation_checks.csv':pd.DataFrame(validation)},
        dict(non_singleton_groups=10,rows_in_non_singleton_groups=28,n_bootstrap=B,rng_seed=20261117,
             cross_split_components=int(pd.DataFrame(splitcounts).n_cross_split_components.sum()),
             n_contrasts=len(contrasts),n_changed_classifications=int(changes.classification_changed.sum()),
             n_coordinate_monte_carlo_changes=len(mc),
             uncertainty='Conditional on fixed fits; percentile nominal 95%, unadjusted; seeds not resampled'))


def subset(out):
    status=read(ROOT/'Data/metadata/p2_identity_reaudit_20261116/record_candidate_status_317.csv').set_index('record_id')
    assert len(status)==317 and status.index.is_unique
    flags={'exact_flux93':status.combined_exact_coordinate_value_rows_0p1percent>0,
           'tolerance_flux98':status.combined_coordinate_value_rows_0p1percent>0}
    assert [int(v.sum()) for v in flags.values()]==[93,98]
    metricrows=[];counts=[];contrasts=[];members=[]
    for split in SPLITS:
        train,test,matrices=load(split)
        np.testing.assert_allclose(test[['lon','lat']],status.loc[test.index,['lon','lat']],rtol=0,atol=1e-12)
        yall=test.log_dep.to_numpy(float)
        for definition,flag in flags.items():
            mask=flag.loc[test.index].to_numpy()
            for selection,keep in [('candidate',mask),('other',~mask)]:
                f=test.loc[keep];y=yall[keep];n=len(y)
                if n==0: raise ValueError(f'Empty subset: {split} {definition} {selection}')
                nc=len(f[['lon','lat']].drop_duplicates())
                counts.append(dict(split=split,definition=definition,selection=selection,n=n,n_coordinates=nc,
                    fraction=n/len(test),log_min=float(y.min()),log_max=float(y.max()),log_mean=float(y.mean()),
                    log_sd=float(y.std(ddof=1)) if n>1 else np.nan))
                for rid in f.index: members.append(dict(split=split,definition=definition,selection=selection,record_id=int(rid)))
                for estimand in ['ensemble','mean_seed_metric']:
                    points={}
                    for model,p in matrices.items():
                        pp=p[keep];pp=pp.mean(axis=1,keepdims=True) if estimand=='ensemble' else pp
                        vals=[score(y,pp[:,j],np.full(n,1/n)) for j in range(pp.shape[1])]
                        m={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
                        points[model]=m
                        metricrows.append(dict(split=split,definition=definition,selection=selection,estimand=estimand,model=model,n=n,**m))
                    for comparator in matrices:
                        if comparator=='Wind-PINN':continue
                        contrasts.append(dict(split=split,definition=definition,selection=selection,estimand=estimand,
                            comparator=comparator,**{'delta_'+k:points['Wind-PINN'][k]-points[comparator][k] for k in ['rmse','mae']}))
        print('Completed subset: '+split,flush=True)
    finish(out,{'subset_counts.csv':pd.DataFrame(counts),'metrics.csv':pd.DataFrame(metricrows),
                'wind_contrasts.csv':pd.DataFrame(contrasts),'subset_memberships.csv':pd.DataFrame(members)},
           dict(exact_candidates=93,tolerance_candidates=98,min_subset_records=int(pd.DataFrame(counts).n.min()),
                inference='Descriptive only; no subset significance tests, archive labels, or clean-data claims'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['near','subset']);args=parser.parse_args()
    out=V5/('nearby_group_audit_20261116' if args.task=='near' else 'source_candidate_subset_audit_20261116')
    if out.exists():raise FileExistsError(out)
    track(PROTOCOL);track(Path(__file__));track(ROOT/'Code/analyze_v5_constant_baseline_audit.py')
    for name in ['main.tex','cas-refs.bib','main.pdf','main_anon.pdf']: track(ROOT/'manuscript'/name)
    near(out) if args.task=='near' else subset(out)
