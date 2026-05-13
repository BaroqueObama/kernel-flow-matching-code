"""Generate Figure 9: sample comparison (4 families x 4 methods)."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

data = json.load(open('results/metrics/sample_comparison_data.json'))
entries = data['families']
m = data['m']

family_names = {'gaussian_mixture': 'Gaussian\nMixture', 'rings': 'Rings',
                'moons': 'Moons', 'spirals': 'Spirals'}

n_families = len(entries)
fig, axes = plt.subplots(n_families, 4, figsize=(10, 2.5 * n_families))

columns = ['Ground Truth', f'Support ($m={m}$)', 'Plug-in', 'Learned ICFM']
colors = ['#FF9800', '#4CAF50', '#F44336', '#2196F3']
markers = ['.', 'x', '.', '.']
sizes = [4, 30, 4, 4]
alphas = [0.3, 0.9, 0.5, 0.5]

for row, entry in enumerate(entries):
    family = entry['family']
    target = np.array(entry['target'])
    support = np.array(entry['support'])
    plugin = np.array(entry['plugin'])
    learned = np.array(entry['learned'])

    # Compute shared axis limits from ground truth (with padding)
    all_pts = target
    pad = 0.15
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    xrange = xmax - xmin
    yrange = ymax - ymin
    span = max(xrange, yrange)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 * (1 + pad)
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    datasets = [target, support, plugin, learned]

    for col, (ds, color, marker, sz, alpha) in enumerate(
            zip(datasets, colors, markers, sizes, alphas)):
        ax = axes[row, col]
        ax.scatter(ds[:, 0], ds[:, 1], c=color, s=sz, alpha=alpha, marker=marker)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(columns[col], fontsize=12)
        if col == 0:
            ax.set_ylabel(family_names.get(family, family), fontsize=11)

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/sample_comparison_4x4.pdf', bbox_inches='tight')
plt.savefig('results/figures/sample_comparison_4x4.png', bbox_inches='tight', dpi=150)
print('Saved to results/figures/sample_comparison_4x4.{pdf,png}')
