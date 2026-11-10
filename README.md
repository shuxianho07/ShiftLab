# ShiftLab

**Multi-task learning under distribution shift for financial time-series functions.**

ShiftLab is a standalone machine-learning research prototype. It asks whether a shared neural representation improves data-driven approximation of related numerical functions, and how those models fail when input distributions change.

The project demonstrates dataset generation, neural training, controlled evaluation, baseline construction, checkpointing, and inference. It is designed to support original follow-up experiments rather than to reproduce a particular lab's project.

## Research design

Six reference functions operate on synthetic 16-element return windows: mean, population standard deviation, maximum drawdown, linear trend slope, positive-value fraction, and average loss over the lowest floor(20%) observations. The last quantity is a finite-window tail statistic, not a calibrated estimate of population expected shortfall. Drawdown compounds returns scaled by 0.05, including initial capital in the high-water mark. Labels are computed exactly; there are no LLM-generated labels or external market data.

### Models and controls

| Model | Purpose |
|---|---|
| Training mean | Constant baseline |
| Ridge regression | Linear baseline |
| Shared MLP | One representation, six outputs |
| Independent MLPs | Separate models with the same hidden width as the shared model |
| Parameter-matched independent MLPs | Separate models with approximately the same aggregate parameter count as the shared model |

The MLP has one tanh hidden layer and is trained with Adam. Gradients are implemented explicitly in NumPy and checked against finite differences. The default shared model has 1,478 parameters; the matched collection uses six width-14 models with 1,518 parameters. Parameter matching does not match training compute.

### Splits and leakage controls

- Training: 4,096 IID uniform windows.
- Validation: 1,024 independent IID windows; select the lowest validation-loss checkpoint, checking every 50 steps and at the final step.
- Testing: 2,048 windows each from IID, larger-support uniform, clipped heavy-tailed, and correlated distributions.
- Independent random-number streams keep training, validation, and tests separate.
- Target normalization uses training statistics only.
- Three training seeds; per-task MAE and training-standard-deviation-normalized RMSE.
- Task scores are averaged within each seed before computing mean and seed standard deviation. This is not a confidence interval.

Training budgets, wall times, checkpoint-selection histories, configuration, and Python/NumPy/platform versions are saved alongside the results.

## Run

Python 3.10+; no GPU or API key required:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python benchmark.py --output results
```

To explore data efficiency, run independent configurations into separate directories:

```sh
OPENBLAS_NUM_THREADS=1 python benchmark.py --train 256 --output results_256
OPENBLAS_NUM_THREADS=1 python benchmark.py --train 1024 --output results_1024
```

These extra data-efficiency experiments are not included in the default reported results. Keep all test distributions fixed when comparing training-set sizes, as the runner does using independent random streams.

### Load a model for inference

Save a JSON array of windows to `windows.json` (each row must contain 16 finite numbers):

```sh
python predict.py --checkpoint results/models_seed_0.npz --input windows.json
python predict.py --checkpoint results/models_seed_0.npz --input windows.json --method matched_independent_mlp
```

Model outputs are unconstrained approximations; they may violate mathematical bounds outside the training distribution. Saved checkpoints use NumPy arrays without pickle.

## Results and interpretation

See [experiment report](results/REPORT.md), [task-level metrics](results/metrics.csv), [model budgets](results/budgets.json), and [training histories](results/training_history.json).

Exact numerical algorithms remain the appropriate practical tool for computing these known functions. Learned approximation is the experimental subject, not a replacement advocated by this project. Results do not measure stock forecasting, trading profitability, or performance on real market observations. Financial terminology describes the synthetic label functions only.

Limitations include a small fixed task set, one architecture, three seeds, approximate parameter matching, and unmatched training compute. Shift distributions change multiple properties at once. Using test findings to revise the architecture would require a fresh final test suite. No novelty or publication claim is made.

## Follow-up investigation: when does sharing help?

The completed study varies training-set size (256 / 1,024 / 4,096) and four task groups across five seeds. It compares shared networks against both equal-width and approximately parameter-matched specialists, plus ridge regression. Each fit uses only its own training subset for normalization; evaluation uses a common training-pool reference scale. Nested training samples and independent, fixed validation/test splits make the data-efficiency comparisons paired.

```sh
OPENBLAS_NUM_THREADS=1 python investigate.py
# Optional figures; requires matplotlib:
python plot_investigation.py
```

Read the [plain-language findings](investigation/FINDINGS.md), [fixed experiment plan](STUDY_PLAN.md), [follow-up report](investigation/REPORT.md), and [paired comparisons](investigation/summary.csv). The [learning curves](investigation/learning_curves.png) show seed variability. These follow-up test splits use a new random namespace, separate from the original benchmark. This study is exploratory; the groups were chosen before its runs, but the question was motivated by earlier results.

No follow-up checkpoints are saved; the exact configuration, source, environment, split hashes, model budgets, and full metrics support rerunning the study. Original benchmark checkpoints remain available under `results/`.

## Research extensions

1. Add compute-matched optimization budgets to the completed data-efficiency study.
2. Compare permutation-invariant representations for symmetric functions, while retaining order-sensitive representations for drawdown and trend.
3. Use a fresh benchmark to validate hypotheses suggested by the completed task-group ablations.
4. Test low-rank adaptation of shared representations on new functions.
5. Introduce a separate controlled shift-validation suite for robust model selection.
