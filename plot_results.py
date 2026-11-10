"""Optional plot: pip install matplotlib, then python plot_results.py."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results'); rows=list(csv.DictReader((root/'metrics.csv').open()))
splits=['iid','scale','heavy_tail','correlated']
methods=['train_mean','ridge','shared_mlp','independent_mlp','matched_independent_mlp']
seeds=sorted({r['seed'] for r in rows})
fig,ax=plt.subplots(figsize=(12,6))
for i,method in enumerate(methods):
    vals=[];errs=[]
    for split in splits:
        scores=[np.mean([float(r['normalized_rmse']) for r in rows if r['method']==method and r['split']==split and r['seed']==s]) for s in seeds]
        vals.append(np.mean(scores));errs.append(np.std(scores,ddof=1) if len(scores)>1 else 0)
    ax.bar(np.arange(4)+(i-2)*.16,vals,.16,label=method,yerr=errs,capsize=2)
ax.set_xticks(range(4),['IID','Scale shift','Heavy tails','Correlated'])
ax.set_ylabel('Mean task-normalized RMSE (lower is better)')
ax.set_title('ShiftLab: multi-task learning under distribution shift')
ax.legend(ncol=2);ax.spines[['top','right']].set_visible(False)
fig.text(.02,.015,'Error bars: seed SD. Parameter matching is approximate; training compute is not matched.',fontsize=9)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(root/'comparison.png',dpi=160)
