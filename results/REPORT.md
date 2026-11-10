# Experiment results

Mean task-normalized RMSE (lower is better). Standard deviation across seeds; not a confidence interval.

| Distribution | Method | Mean | Seed SD |
|---|---|---:|---:|
| iid | train_mean | 1.0042 | 0.0120 |
| iid | ridge | 0.4642 | 0.0065 |
| iid | shared_mlp | 0.4056 | 0.0062 |
| iid | independent_mlp | 0.3512 | 0.0096 |
| iid | matched_independent_mlp | 0.4250 | 0.0077 |
| scale | train_mean | 3.7058 | 0.0220 |
| scale | ridge | 2.9308 | 0.0318 |
| scale | shared_mlp | 2.7218 | 0.0194 |
| scale | independent_mlp | 2.5167 | 0.0541 |
| scale | matched_independent_mlp | 2.8418 | 0.0482 |
| heavy_tail | train_mean | 2.6603 | 0.0284 |
| heavy_tail | ridge | 1.9756 | 0.0281 |
| heavy_tail | shared_mlp | 1.8832 | 0.0103 |
| heavy_tail | independent_mlp | 1.7314 | 0.0168 |
| heavy_tail | matched_independent_mlp | 1.9422 | 0.0275 |
| correlated | train_mean | 4.0593 | 0.0245 |
| correlated | ridge | 2.1855 | 0.0218 |
| correlated | shared_mlp | 2.4689 | 0.0117 |
| correlated | independent_mlp | 2.3075 | 0.0121 |
| correlated | matched_independent_mlp | 2.5434 | 0.0582 |

Synthetic function approximation only. These results do not establish trading performance, novelty, or publication readiness.
Models select their best checkpoint using an independent IID validation split. Normalization uses training samples only.
The matched independent baseline approximates the shared model total parameter budget; training compute is not matched. See budgets.json for counts and timings.
The equal-width independent baseline has more total parameters. Results across methods do not establish a causal effect of representation sharing.
