# Task-sharing investigation

Positive gain = baseline error minus shared-model error; larger is better for sharing. Scores average the same tasks within seed before aggregation. SD is across paired seed differences, not a confidence interval.

## IID data-efficiency results

| Training windows | Group | Baseline | Shared NRMSE | Baseline NRMSE | Gain | Gain SD | Positive seeds |
|---:|---|---|---:|---:|---:|---:|---:|
| 256 | all | independent | 0.5242 | 0.5099 | -0.0143 | 0.0041 | 0/5 |
| 256 | all | matched | 0.5242 | 0.5013 | -0.0228 | 0.0031 | 0/5 |
| 256 | all | ridge | 0.5242 | 0.4743 | -0.0499 | 0.0031 | 0/5 |
| 256 | risk | independent | 0.7841 | 0.7753 | -0.0089 | 0.0068 | 1/5 |
| 256 | risk | matched | 0.7841 | 0.7685 | -0.0156 | 0.0062 | 0/5 |
| 256 | risk | ridge | 0.7841 | 0.7788 | -0.0053 | 0.0073 | 1/5 |
| 256 | direction | independent | 0.2640 | 0.2445 | -0.0195 | 0.0072 | 0/5 |
| 256 | direction | matched | 0.2640 | 0.2436 | -0.0203 | 0.0051 | 0/5 |
| 256 | direction | ridge | 0.2640 | 0.1697 | -0.0942 | 0.0055 | 0/5 |
| 256 | mean_volatility | independent | 0.5782 | 0.5713 | -0.0069 | 0.0098 | 2/5 |
| 256 | mean_volatility | matched | 0.5782 | 0.5675 | -0.0107 | 0.0058 | 0/5 |
| 256 | mean_volatility | ridge | 0.5782 | 0.5144 | -0.0638 | 0.0080 | 0/5 |
| 1024 | all | independent | 0.4525 | 0.4340 | -0.0185 | 0.0093 | 0/5 |
| 1024 | all | matched | 0.4525 | 0.4432 | -0.0093 | 0.0044 | 0/5 |
| 1024 | all | ridge | 0.4525 | 0.4601 | +0.0075 | 0.0067 | 4/5 |
| 1024 | risk | independent | 0.6602 | 0.6572 | -0.0031 | 0.0211 | 3/5 |
| 1024 | risk | matched | 0.6602 | 0.6583 | -0.0019 | 0.0159 | 3/5 |
| 1024 | risk | ridge | 0.6602 | 0.7542 | +0.0940 | 0.0180 | 5/5 |
| 1024 | direction | independent | 0.2355 | 0.2108 | -0.0247 | 0.0050 | 0/5 |
| 1024 | direction | matched | 0.2355 | 0.2134 | -0.0222 | 0.0022 | 0/5 |
| 1024 | direction | ridge | 0.2355 | 0.1659 | -0.0696 | 0.0039 | 0/5 |
| 1024 | mean_volatility | independent | 0.4744 | 0.4819 | +0.0075 | 0.0297 | 2/5 |
| 1024 | mean_volatility | matched | 0.4744 | 0.4801 | +0.0057 | 0.0248 | 2/5 |
| 1024 | mean_volatility | ridge | 0.4744 | 0.4970 | +0.0226 | 0.0143 | 5/5 |
| 4096 | all | independent | 0.4016 | 0.3478 | -0.0538 | 0.0082 | 0/5 |
| 4096 | all | matched | 0.4016 | 0.4167 | +0.0151 | 0.0038 | 5/5 |
| 4096 | all | ridge | 0.4016 | 0.4580 | +0.0564 | 0.0055 | 5/5 |
| 4096 | risk | independent | 0.5505 | 0.5056 | -0.0449 | 0.0159 | 0/5 |
| 4096 | risk | matched | 0.5505 | 0.5912 | +0.0407 | 0.0110 | 5/5 |
| 4096 | risk | ridge | 0.5505 | 0.7508 | +0.2003 | 0.0074 | 5/5 |
| 4096 | direction | independent | 0.2050 | 0.1901 | -0.0149 | 0.0096 | 0/5 |
| 4096 | direction | matched | 0.2050 | 0.2000 | -0.0050 | 0.0072 | 1/5 |
| 4096 | direction | ridge | 0.2050 | 0.1652 | -0.0398 | 0.0038 | 0/5 |
| 4096 | mean_volatility | independent | 0.3780 | 0.3646 | -0.0134 | 0.0230 | 1/5 |
| 4096 | mean_volatility | matched | 0.3780 | 0.4214 | +0.0434 | 0.0126 | 5/5 |
| 4096 | mean_volatility | ridge | 0.3780 | 0.4946 | +0.1166 | 0.0169 | 5/5 |

All shifted-distribution comparisons are in summary.csv; task-level raw and normalized errors are in metrics.csv.
Each fit normalizes targets using only its own training subset. Reporting uses the maximum training pool standard deviation as a common reference across sizes; that reporting reference never enters training. Validation and test data are independent and fixed within seed. See STUDY_PLAN.md for the full design.
Parameter counts are approximately matched in the matched baseline; compute is not matched. These exploratory results do not prove that task similarity causes transfer or that a selected group will generalize to a fresh benchmark.
