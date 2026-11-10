# Findings: when does sharing help?

The fixed design evaluated 3 training sizes, four task groups, four test distributions, and 5 seeds. See STUDY_PLAN.md for the exact design.

## Main observations

- At 256 training windows, the all-task shared model had higher IID error than parameter-matched specialists (0.5242 versus 0.5013); no small-data sharing advantage was observed in that comparison.
- At 4,096 windows, the all-task shared model beat parameter-matched specialists (0.4016 versus 0.4167).
- At 4,096 windows, the risk-group shared model reduced IID error by 6.9% against its parameter-matched specialists (0.5505 versus 0.5912), with positive paired gains in 5/5 seeds.
- Full-width independent specialists still outperform these shared models on the largest IID dataset. The result depends on the resource comparison.

## Does narrowing the task group help?

This compares each group against the all-six model on exactly the same target tasks. Positive gain favors the narrower group. Both use hidden width 64; output counts and total parameter counts differ.

| Training windows | Group | All-six model on same tasks | Group model | Gain | Gain SD | Positive seeds |
|---:|---|---:|---:|---:|---:|---:|
| 256 | risk | 0.7798 | 0.7841 | -0.0043 | 0.0089 | 1/5 |
| 256 | direction | 0.2685 | 0.2640 | +0.0046 | 0.0052 | 4/5 |
| 256 | mean_volatility | 0.5820 | 0.5782 | +0.0038 | 0.0049 | 4/5 |
| 1024 | risk | 0.6564 | 0.6602 | -0.0038 | 0.0076 | 1/5 |
| 1024 | direction | 0.2487 | 0.2355 | +0.0131 | 0.0069 | 5/5 |
| 1024 | mean_volatility | 0.4931 | 0.4744 | +0.0187 | 0.0138 | 5/5 |
| 4096 | risk | 0.5788 | 0.5505 | +0.0283 | 0.0172 | 5/5 |
| 4096 | direction | 0.2243 | 0.2050 | +0.0193 | 0.0051 | 5/5 |
| 4096 | mean_volatility | 0.4247 | 0.3780 | +0.0467 | 0.0195 | 5/5 |

## Interpretation

These results suggest that, under this architecture and optimization budget, task sharing can use a limited parameter budget effectively after sufficient training data is available. They do not support a blanket claim that multi-task learning helps most when data is scarce.
Narrower groups improve the largest-data IID results on their own targets relative to the all-six model. This is evidence of group-dependent interference in this setup, not proof of its mechanism: head count, loss scaling, checkpoint selection, and capacity allocation also change.
Simple linear baselines remain strong on some tasks and shifts. Report task-level errors rather than presenting the best aggregate as universal superiority.

## Limits

Five seeds and a small synthetic task set support exploratory observations. Seed SD is not a confidence interval. Parameter matching is approximate, and training compute is not matched. Architecture and optimizer settings were held fixed rather than tuned equally per baseline. Test-based follow-up hypotheses require a fresh confirmatory evaluation.
