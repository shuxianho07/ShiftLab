# Follow-up investigation: task sharing and data efficiency

## Fixed design

This plan is specified before running the follow-up experiment. Earlier benchmark results motivated the question, so this is exploratory follow-up research, not an independent confirmatory study.

- Training sample sizes: 256, 1,024, 4,096, nested within each seed.
- Five seeds: 0, 1, 2, 3, 4.
- Training/validation/test generators use independent random streams with a new study namespace (20261110), distinct from the original benchmark.
- Validation: 1,024 IID windows per seed. Test: 2,048 per distribution per seed, fixed across all models and training sizes.
- Each model normalizes targets using its own training subset. For comparable reporting across sizes, normalized RMSE divides raw RMSE by the maximum training-pool target standard deviation per seed. This reporting reference does not enter model fitting; no validation/test labels enter preprocessing.
- Report raw RMSE as well as normalized RMSE to make normalization explicit.
- Shared groups: all six; risk (volatility/drawdown/tail loss); direction (mean/trend/positive fraction); mean+volatility.
- Baselines: per-task width-64 MLPs; approximate aggregate-parameter-matched independent MLPs for each group; ridge regression.
- Every network uses the original MLP, Adam learning rate 0.003, 1,200 steps, batch size 128, validation checks every 50 steps.
- Primary outcome: IID per-task normalized RMSE, comparing each shared group against both independent baselines on the exact same tasks and samples. Positive transfer = lower error than that baseline. Compare means across tasks within seed, then paired differences across seeds.
- Secondary outcomes: per-task transfer and robustness on scale, heavy-tail, and correlated splits.
- Uncertainty: paired-seed sample standard deviations; no significance or population confidence claims. Tasks and test rows are not treated as independent training replications.
- Record exact parameter counts, optimizer updates, timings, split hashes, selected steps, environment, configuration, and all task-level errors.
- Do not choose the best group on test data and report it as a validated winner. Model/group selection after this study requires a fresh final test.

## Interpretation limits

Independent networks have a full hidden representation per task; shared heads and validation selection interact with task count. Approximate parameter matching does not match optimizer updates or FLOPs. Finite optimization and shared-loss scaling can affect apparent transfer. The synthetic functions are not market prediction tasks.
