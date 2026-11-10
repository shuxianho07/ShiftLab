"""Plot fixed-design study: pip install matplotlib; python plot_investigation.py."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from investigate import GROUPS

out=Path('investigation');rows=list(csv.DictReader((out/'summary.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,reference in zip(axes,['independent','matched']):
    for group in GROUPS:
        selected=sorted([r for r in rows if r['split']=='iid' and r['reference']==reference and r['group']==group],key=lambda r:int(r['train']))
        ax.errorbar([int(r['train']) for r in selected],[float(r['gain']) for r in selected],
                    yerr=[float(r['gain_seed_sd']) for r in selected],marker='o',capsize=3,label=group)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xscale('log',base=2)
    ax.set_xlabel('Training windows');ax.set_title('Sharing vs '+('width-64 specialists' if reference=='independent' else 'parameter-matched specialists'))
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Baseline NRMSE − shared NRMSE (positive favors sharing)')
axes[1].legend();fig.suptitle('ShiftLab: task sharing and data efficiency (IID)')
fig.text(.02,.015,'Error bars: SD of paired differences across five seeds. Compute is not matched; groups contain different tasks.',fontsize=9)
fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(out/'learning_curves.png',dpi=160)
