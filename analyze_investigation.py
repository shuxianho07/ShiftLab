"""Generate matched-task group ablations and a readable findings report."""
import csv
from pathlib import Path
import numpy as np

def analyze(out):
    from investigate import GROUPS, SPLITS, write_csv
    from benchmark import TASKS
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    summaries=list(csv.DictReader((out/'summary.csv').open()))
    comparisons=[]
    for n in sorted({int(r['train']) for r in rows}):
        for split in SPLITS:
            for group,ids in GROUPS.items():
                if group=='all':continue
                tasks=[TASKS[j] for j in ids];gains=[]
                all_scores=[];group_scores=[]
                for seed in sorted({r['seed'] for r in rows}):
                    def score(name):
                        return np.mean([float(r['nrmse']) for r in rows if int(r['train'])==n and r['seed']==seed and r['split']==split and r['method']=='shared' and r['group']==name and r['task'] in tasks])
                    all_score=score('all');group_score=score(group)
                    all_scores.append(all_score);group_scores.append(group_score);gains.append(all_score-group_score)
                comparisons.append(dict(train=n,split=split,group=group,
                    all_six_same_tasks_nrmse=float(np.mean(all_scores)),group_nrmse=float(np.mean(group_scores)),
                    gain=float(np.mean(gains)),gain_seed_sd=float(np.std(gains,ddof=1)) if len(gains)>1 else 0,
                    positive_seeds=sum(g>0 for g in gains),seeds=len(gains)))
    write_csv(out/'group_ablation.csv',comparisons)
    sizes=sorted({int(r['train']) for r in rows});small,large=sizes[0],sizes[-1]
    def result(n,group,reference,split='iid'):
        return next(r for r in summaries if int(r['train'])==n and r['group']==group and r['reference']==reference and r['split']==split)
    risk=result(large,'risk','matched');all_tasks=result(large,'all','matched');small_all=result(small,'all','matched')
    improvement=100*float(risk['gain'])/float(risk['baseline_nrmse'])
    lines=['# Findings: when does sharing help?','',
        f'The fixed design evaluated {len(sizes)} training sizes, four task groups, four test distributions, and {risk["seeds"]} seeds. See STUDY_PLAN.md for the exact design.',
        '', '## Main observations','',
        f'- At {small:,} training windows, the all-task shared model had higher IID error than parameter-matched specialists ({float(small_all["shared_nrmse"]):.4f} versus {float(small_all["baseline_nrmse"]):.4f}); no small-data sharing advantage was observed in that comparison.',
        f'- At {large:,} windows, the all-task shared model beat parameter-matched specialists ({float(all_tasks["shared_nrmse"]):.4f} versus {float(all_tasks["baseline_nrmse"]):.4f}).',
        f'- At {large:,} windows, the risk-group shared model reduced IID error by {improvement:.1f}% against its parameter-matched specialists ({float(risk["shared_nrmse"]):.4f} versus {float(risk["baseline_nrmse"]):.4f}), with positive paired gains in {risk["positive_seeds"]}/{risk["seeds"]} seeds.',
        '- Full-width independent specialists still outperform these shared models on the largest IID dataset. The result depends on the resource comparison.',
        '', '## Does narrowing the task group help?','',
        'This compares each group against the all-six model on exactly the same target tasks. Positive gain favors the narrower group. Both use hidden width 64; output counts and total parameter counts differ.', '',
        '| Training windows | Group | All-six model on same tasks | Group model | Gain | Gain SD | Positive seeds |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['split']=='iid':
            lines.append(f"| {r['train']} | {r['group']} | {r['all_six_same_tasks_nrmse']:.4f} | {r['group_nrmse']:.4f} | {r['gain']:+.4f} | {r['gain_seed_sd']:.4f} | {r['positive_seeds']}/{r['seeds']} |")
    lines += ['', '## Interpretation','',
        'These results suggest that, under this architecture and optimization budget, task sharing can use a limited parameter budget effectively after sufficient training data is available. They do not support a blanket claim that multi-task learning helps most when data is scarce.',
        'Narrower groups improve the largest-data IID results on their own targets relative to the all-six model. This is evidence of group-dependent interference in this setup, not proof of its mechanism: head count, loss scaling, checkpoint selection, and capacity allocation also change.',
        'Simple linear baselines remain strong on some tasks and shifts. Report task-level errors rather than presenting the best aggregate as universal superiority.',
        '', '## Limits','',
        'Five seeds and a small synthetic task set support exploratory observations. Seed SD is not a confidence interval. Parameter matching is approximate, and training compute is not matched. Architecture and optimizer settings were held fixed rather than tuned equally per baseline. Test-based follow-up hypotheses require a fresh confirmatory evaluation.']
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':analyze(Path('investigation'))
