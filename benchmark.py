"""Reproducible neural function approximation under input distribution shift."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np

TASKS = ('mean', 'volatility', 'max_drawdown', 'trend', 'positive_fraction', 'tail_loss')

def targets(x):
    """Functions over bounded synthetic return windows; no market prediction."""
    wealth = np.cumprod(1 + x * .05, axis=1)
    # Include initial capital in the high-water mark.
    peak = np.maximum.accumulate(np.column_stack([np.ones(len(x)), wealth]), axis=1)[:, 1:]
    time = np.linspace(-1, 1, x.shape[1])
    return np.column_stack([x.mean(1), x.std(1), (1 - wealth / peak).max(1),
                            x @ time / (time @ time), (x > 0).mean(1),
                            -np.sort(x, axis=1)[:, :max(1, x.shape[1] // 5)].mean(1)])

def sample(rng, n, width, split):
    if split == 'iid': return rng.uniform(-1, 1, (n, width))
    if split == 'scale': return rng.uniform(-2, 2, (n, width))
    if split == 'heavy_tail': return np.clip(rng.standard_t(3, (n, width)) / np.sqrt(3), -4, 4)
    if split == 'correlated':
        z = rng.normal(0, np.sqrt(1 - .8**2), (n, width))
        z[:, 0] = rng.normal(size=n)
        for j in range(1, width): z[:, j] += .8 * z[:, j - 1]
        return z
    raise ValueError(split)

class MLP:
    """Two-layer neural network with explicit gradients and Adam optimization."""
    def __init__(self, inputs, hidden, outputs, seed):
        rng = np.random.default_rng(seed)
        self.p = [rng.normal(0, np.sqrt(2 / inputs), (inputs, hidden)), np.zeros(hidden),
                  rng.normal(0, np.sqrt(1 / hidden), (hidden, outputs)), np.zeros(outputs)]

    def predict(self, x):
        a, b, c, d = self.p
        return np.tanh(x @ a + b) @ c + d

    def loss_grad(self, x, y):
        a, b, c, d = self.p
        h = np.tanh(x @ a + b)
        error = h @ c + d - y
        g = 2 * error / error.size
        dh = (g @ c.T) * (1 - h * h)
        return np.mean(error**2), [x.T @ dh, dh.sum(0), h.T @ g, g.sum(0)]

    def fit(self, x, y, steps, batch, seed, validation=None):
        rng = np.random.default_rng(seed)
        m, v = [np.zeros_like(p) for p in self.p], [np.zeros_like(p) for p in self.p]
        best_loss = float('inf')
        best_weights = None
        self.history = []
        self.best_step = steps
        for step in range(1, steps + 1):
            idx = rng.integers(len(x), size=min(batch, len(x)))
            _, grads = self.loss_grad(x[idx], y[idx])
            for j, g in enumerate(grads):
                m[j] = .9 * m[j] + .1 * g
                v[j] = .999 * v[j] + .001 * g * g
                self.p[j] -= .003 * (m[j] / (1 - .9**step)) / (np.sqrt(v[j] / (1 - .999**step)) + 1e-8)
            if validation is not None and (step % 50 == 0 or step == steps):
                vx, vy = validation
                loss = float(np.mean((self.predict(vx) - vy)**2))
                self.history.append(dict(step=step, validation_mse=loss))
                if loss < best_loss:
                    best_loss = loss
                    best_weights = [p.copy() for p in self.p]
                    self.best_step = step
        if best_weights is not None:
            self.p = best_weights
        return self


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows, budgets, histories = [], [], []
    for seed in args.seeds:
        streams = np.random.SeedSequence(seed).spawn(6)
        x = sample(np.random.default_rng(streams[0]), args.train, args.width, 'iid'); y = targets(x)
        vx = sample(np.random.default_rng(streams[1]), args.validation, args.width, 'iid')
        vy = targets(vx)
        mu, sd = y.mean(0), np.maximum(y.std(0), 1e-8)
        yn = (y - mu) / sd
        vyn = (vy - mu) / sd
        start = time.perf_counter()
        shared = MLP(args.width, args.hidden, len(TASKS), seed).fit(
            x, yn, args.steps, 128, seed, (vx, vyn))
        shared_seconds = time.perf_counter() - start
        # Match aggregate parameter count, including biases, to the shared model.
        shared_count = sum(p.size for p in shared.p)
        candidates = range(1, args.hidden + 1)
        matched_width = min(candidates, key=lambda h: abs(len(TASKS) * ((args.width + 2)*h + 1) - shared_count))
        families = {}
        for method, width in [('independent_mlp', args.hidden), ('matched_independent_mlp', matched_width)]:
            start = time.perf_counter()
            families[method] = [MLP(args.width, width, 1, seed+j).fit(
                x, yn[:, j:j+1], args.steps, 128, seed, (vx, vyn[:, j:j+1])) for j in range(len(TASKS))]
            budgets.append(dict(seed=seed, method=method, hidden=width,
                parameters=sum(p.size for m in families[method] for p in m.p),
                optimizer_steps=args.steps * len(TASKS), training_seconds=time.perf_counter()-start))
        budgets.append(dict(seed=seed, method='shared_mlp', hidden=args.hidden,
            parameters=shared_count, optimizer_steps=args.steps, training_seconds=shared_seconds))
        for method, models in {'shared_mlp':[shared], **families}.items():
            for j,m in enumerate(models):
                histories.append(dict(seed=seed, method=method, model=j, best_step=m.best_step,
                                      validation=m.history))
        design = np.column_stack([x, np.ones(len(x))])
        penalty = np.eye(args.width + 1) * .001; penalty[-1, -1] = 0
        ridge = np.linalg.solve(design.T @ design + penalty, design.T @ yn)
        np.savez(out / f'models_seed_{seed}.npz', mean=mu, std=sd, ridge=ridge,
                 **{f'shared_{j}':p for j,p in enumerate(shared.p)},
                 **{f'{name}_{t}_{j}':p for name,models in families.items() for t,m in enumerate(models) for j,p in enumerate(m.p)})
        for stream, split in zip(streams[2:], ['iid', 'scale', 'heavy_tail', 'correlated']):
            xx = sample(np.random.default_rng(stream), args.test, args.width, split); yy = targets(xx)
            predictions = {'shared_mlp': shared.predict(xx) * sd + mu,
                           **{name:np.column_stack([m.predict(xx)[:, 0] for m in models])*sd+mu for name,models in families.items()},
                           'ridge': np.column_stack([xx, np.ones(len(xx))]) @ ridge * sd + mu,
                           'train_mean': np.broadcast_to(mu, yy.shape)}
            for method, pred in predictions.items():
                for j, task in enumerate(TASKS):
                    rows.append(dict(seed=seed, split=split, method=method, task=task,
                                     mae=float(np.mean(abs(pred[:,j]-yy[:,j]))),
                                     normalized_rmse=float(np.sqrt(np.mean((pred[:,j]-yy[:,j])**2))/sd[j])))
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'training_history.json').write_text(json.dumps(histories, indent=2))
    (out/'budgets.json').write_text(json.dumps(budgets, indent=2))
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()), indent=2))
    (out/'config.json').write_text(json.dumps(vars(args), indent=2))
    summary = []
    for split in ['iid', 'scale', 'heavy_tail', 'correlated']:
        for method in ['train_mean', 'ridge', 'shared_mlp', 'independent_mlp', 'matched_independent_mlp']:
            # Average tasks within seed first; uncertainty is across independent runs.
            scores = [np.mean([r['normalized_rmse'] for r in rows if r['seed']==s and r['split']==split and r['method']==method]) for s in args.seeds]
            summary.append((split, method, np.mean(scores), np.std(scores, ddof=1) if len(scores)>1 else 0))
    lines = ['# Experiment results', '', 'Mean task-normalized RMSE (lower is better). Standard deviation across seeds; not a confidence interval.', '',
             '| Distribution | Method | Mean | Seed SD |', '|---|---|---:|---:|']
    lines += [f'| {s} | {m} | {a:.4f} | {b:.4f} |' for s,m,a,b in summary]
    lines += ['', 'Synthetic function approximation only. These results do not establish trading performance, novelty, or publication readiness.',
              'Models select their best checkpoint using an independent IID validation split. Normalization uses training samples only.',
              'The matched independent baseline approximates the shared model total parameter budget; training compute is not matched. See budgets.json for counts and timings.',
              'The equal-width independent baseline has more total parameters. Results across methods do not establish a causal effect of representation sharing.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results');p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--train',type=int,default=4096);p.add_argument('--test',type=int,default=2048)
    p.add_argument('--width',type=int,default=16);p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--validation',type=int,default=1024)
    a=p.parse_args()
    if min(a.train,a.test,a.hidden,a.steps,a.validation)<1 or a.width<2 or len(set(a.seeds))!=len(a.seeds):p.error('Positive sizes and unique seeds required')
    run(a)
