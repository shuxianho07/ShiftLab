"""Load saved checkpoints and evaluate JSON arrays of synthetic return windows."""
import argparse
import json
import numpy as np
from benchmark import TASKS

METHODS = ('shared_mlp', 'independent_mlp', 'matched_independent_mlp', 'ridge', 'train_mean')

def infer(weights, windows, method='shared_mlp'):
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    x = np.asarray(windows, dtype=float)
    width = weights['ridge'].shape[0] - 1
    if x.ndim != 2 or not len(x) or x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError(f'Expected finite, nonempty array of shape [samples, {width}]')
    def forward(prefix):
        a, b, c, d = [weights[f'{prefix}_{j}'] for j in range(4)]
        return np.tanh(x @ a + b) @ c + d
    if method == 'shared_mlp': y = forward('shared')
    elif method == 'ridge': y = np.column_stack([x, np.ones(len(x))]) @ weights['ridge']
    elif method == 'train_mean': y = np.zeros((len(x), len(TASKS)))
    else: y = np.column_stack([forward(f'{method}_{t}')[:, 0] for t in range(len(TASKS))])
    return y * weights['std'] + weights['mean']

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True)
    p.add_argument('--method',choices=METHODS,default='shared_mlp')
    a=p.parse_args()
    with open(a.input) as f: windows=json.load(f)
    with np.load(a.checkpoint,allow_pickle=False) as weights: y=infer(weights,windows,a.method)
    print(json.dumps([dict(zip(TASKS,map(float,row))) for row in y],indent=2))
