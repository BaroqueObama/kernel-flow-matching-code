"""Generate LOFO generalization figure."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

d = json.load(open('results/metrics/lofo_generalization.json'))

families = ['gaussian_mixture', 'rings', 'moons', 'spirals']
family_labels = {'gaussian_mixture': 'GMM', 'rings': 'Rings', 'moons': 'Moons', 'spirals': 'Spirals'}

held_out_mmds = []
plugin_mmds = []
for fam in families:
    entry = d[f'lofo_no_{fam}'][0]
    held_out_mmds.append(entry['families'][fam]['mmd_mean'])
    plugin_mmds.append(d['plugin_baseline'][fam]['mmd_mean'])

ratios = [h / p for h, p in zip(held_out_mmds, plugin_mmds)]

fig, ax = plt.subplots(figsize=(6, 3.5))

x = np.arange(len(families))
width = 0.35

ax.bar(x - width/2, plugin_mmds, width, color='#F44336', edgecolor='black', linewidth=0.5, label='Plug-in')
ax.bar(x + width/2, held_out_mmds, width, color='#FF9800', edgecolor='black', linewidth=0.5, label='LOFO (held-out)')

ax.set_xticks(x)
ax.set_xticklabels([family_labels[f] for f in families], fontsize=13)
ax.set_ylabel('MMD$^2$ (log scale)', fontsize=14)
ax.set_yscale('log')
ax.legend(fontsize=11, loc='upper right')
ax.tick_params(labelsize=11)
ax.set_ylim(bottom=1e-4, top=1.0)
ax.set_title('Leave-one-family-out generalization', fontsize=15)

for i, (hm, pm, ratio) in enumerate(zip(held_out_mmds, plugin_mmds, ratios)):
    if ratio > 2:
        ax.text(i + width/2, hm * 0.5, f'{ratio:.0f}$\\times$', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    else:
        ax.text(i + width/2, hm * 0.5, f'{ratio:.1f}$\\times$', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/lofo_generalization.pdf', bbox_inches='tight')
plt.savefig('results/figures/lofo_generalization.png', bbox_inches='tight', dpi=150)
print('Saved to results/figures/lofo_generalization.{pdf,png}')
