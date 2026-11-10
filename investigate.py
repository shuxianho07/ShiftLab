"""Task-group and sample-size ablations; see STUDY_PLAN.md for fixed design."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from benchmark import MLP, TASKS, sample, targets

GROUPS = {'all': tuple(range(6)), 'risk': (1,2,5), 'direction': (0,3,4), 'mean_volatility': (0,1)}
SPLITS = ('iid','scale','heavy_tail','correlated')

def dataset(seed, maximum, validation, test, width):
    streams = np.random.SeedSequence([20261110, seed]).spawn(6)
    train=sample(np.random.default_rng(streams[0]),maximum,width,'iid')
    val=sample(np.random.default_rng(streams[1]),validation,width,'iid')
    tests={name:sample(np.random.default_rng(stream),test,width,name) for name,stream in zip(SPLITS,streams[2:])}
    return train,val,tests

def count_params(inputs, hidden, outputs):
    return (inputs+1)*hidden+(hidden+1)*outputs

def matched_width(inputs, hidden, tasks):
    goal=count_params(inputs,hidden,tasks)
    return min(range(1,hidden+1),key=lambda h:abs(tasks*count_params(inputs,h,1)-goal))

def paired_summary(rows):
    summaries=[]
    for n in sorted({r['train'] for r in rows}):
        for split in SPLITS:
            for group in GROUPS:
                selected=[r for r in rows if r['train']==n and r['split']==split and r['group']==group]
                for reference in ('independent','matched','ridge'):
                    differences=[];shared=[];base=[]
                    for seed in sorted({r['seed'] for r in selected}):
                        a=np.mean([r['nrmse'] for r in selected if r['seed']==seed and r['method']=='shared'])
                        b=np.mean([r['nrmse'] for r in selected if r['seed']==seed and r['method']==reference])
                        shared.append(a);base.append(b);differences.append(b-a)
                    summaries.append(dict(train=n,split=split,group=group,reference=reference,
                        shared_nrmse=float(np.mean(shared)),baseline_nrmse=float(np.mean(base)),
                        gain=float(np.mean(differences)),gain_seed_sd=float(np.std(differences,ddof=1)) if len(differences)>1 else 0,
                        positive_seeds=sum(d>0 for d in differences),seeds=len(differences)))
    return summaries

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),indent=2))
    rows=[];budgets=[];hashes=[]
    for seed in a.seeds:
        pool,vx,tests=dataset(seed,max(a.sizes),a.validation,a.test,a.width)
        py=targets(pool);reference_sd=np.maximum(py.std(0),1e-8)
        for name,x in {'training_pool':pool,'validation':vx,**tests}.items():
            hashes.append(dict(seed=seed,split=name,shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        for n in a.sizes:
            x=pool[:n];mu=py[:n].mean(0);sd=np.maximum(py[:n].std(0),1e-8)
            y=(py[:n]-mu)/sd;vy=(targets(vx)-mu)/sd
            independent_cache={}
            def independent(width, task):
                key=(width,task)
                if key not in independent_cache:
                    start=time.perf_counter()
                    model=MLP(a.width,width,1,seed*100+task).fit(x,y[:,task:task+1],a.steps,128,seed,(vx,vy[:,task:task+1]))
                    independent_cache[key]=model
                    budgets.append(dict(seed=seed,train=n,kind='independent',tasks=[TASKS[task]],hidden=width,
                        parameters=sum(p.size for p in model.p),optimizer_steps=a.steps,
                        best_step=model.best_step,training_seconds=time.perf_counter()-start))
                return independent_cache[key]
            design=np.column_stack([x,np.ones(n)])
            penalty=np.eye(a.width+1)*.001;penalty[-1,-1]=0
            ridge=np.linalg.solve(design.T@design+penalty,design.T@y)
            for group,ids in GROUPS.items():
                ids=list(ids);start=time.perf_counter()
                shared=MLP(a.width,a.hidden,len(ids),seed).fit(x,y[:,ids],a.steps,128,seed,(vx,vy[:,ids]))
                budgets.append(dict(seed=seed,train=n,kind='shared',tasks=[TASKS[j] for j in ids],hidden=a.hidden,
                    parameters=sum(p.size for p in shared.p),optimizer_steps=a.steps,best_step=shared.best_step,
                    training_seconds=time.perf_counter()-start))
                width=matched_width(a.width,a.hidden,len(ids))
                full=[independent(a.hidden,j) for j in ids];matched=[independent(width,j) for j in ids]
                for split,xx in tests.items():
                    truth=targets(xx)[:,ids]
                    preds={'shared':shared.predict(xx),
                        'independent':np.column_stack([m.predict(xx)[:,0] for m in full]),
                        'matched':np.column_stack([m.predict(xx)[:,0] for m in matched]),
                        'ridge':(np.column_stack([xx,np.ones(len(xx))])@ridge)[:,ids]}
                    for method,pred in preds.items():
                        raw=pred*sd[ids]+mu[ids];error=raw-truth
                        for k,j in enumerate(ids):
                            rmse=float(np.sqrt(np.mean(error[:,k]**2)))
                            rows.append(dict(seed=seed,train=n,split=split,group=group,task=TASKS[j],method=method,
                                rmse=rmse,nrmse=rmse/reference_sd[j],mae=float(np.mean(abs(error[:,k])))))
            print(f'Completed seed={seed}, train={n}',flush=True)
    summary=paired_summary(rows)
    write_csv(out/'metrics.csv',rows);write_csv(out/'summary.csv',summary)
    (out/'budgets.json').write_text(json.dumps(budgets,indent=2))
    (out/'split_hashes.json').write_text(json.dumps(hashes,indent=2))
    lines=['# Task-sharing investigation','',
        'Positive gain = baseline error minus shared-model error; larger is better for sharing. Scores average the same tasks within seed before aggregation. SD is across paired seed differences, not a confidence interval.','',
        '## IID data-efficiency results','','| Training windows | Group | Baseline | Shared NRMSE | Baseline NRMSE | Gain | Gain SD | Positive seeds |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['split']=='iid':
            lines.append(f"| {r['train']} | {r['group']} | {r['reference']} | {r['shared_nrmse']:.4f} | {r['baseline_nrmse']:.4f} | {r['gain']:+.4f} | {r['gain_seed_sd']:.4f} | {r['positive_seeds']}/{r['seeds']} |")
    lines += ['', 'All shifted-distribution comparisons are in summary.csv; task-level raw and normalized errors are in metrics.csv.',
        'Each fit normalizes targets using only its own training subset. Reporting uses the maximum training pool standard deviation as a common reference across sizes; that reporting reference never enters training. Validation and test data are independent and fixed within seed. See STUDY_PLAN.md for the full design.',
        'Parameter counts are approximately matched in the matched baseline; compute is not matched. These exploratory results do not prove that task similarity causes transfer or that a selected group will generalize to a fresh benchmark.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from analyze_investigation import analyze
    analyze(out)
    print(f'Wrote {len(rows)} task metrics and {len(summary)} paired comparisons to {out}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',type=int,nargs='+',default=[256,1024,4096]);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--width',type=int,default=16);p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--validation',type=int,default=1024)
    p.add_argument('--test',type=int,default=2048);p.add_argument('--output',default='investigation')
    a=p.parse_args()
    if min(*a.sizes,a.hidden,a.steps,a.validation,a.test)<1 or a.width<2 or len(set(a.sizes))!=len(a.sizes) or len(set(a.seeds))!=len(a.seeds):
        p.error('Positive sizes, width >= 2, and unique sizes/seeds required')
    run(a)
