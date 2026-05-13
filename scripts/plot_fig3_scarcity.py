"""Generate Figure 3: Support scarcity on DINOv2+PCA ImageNet (d=16)."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

d = json.load(open('results/metrics/fig3_scarcity_data.json'))
d16 = d['d_16']

ms = sorted([int(m) for m in d16['learned'].keys()])
learned_means = [d16['learned'][str(m)]['mmd_mean'] for m in ms]
learned_stds = [d16['learned'][str(m)]['mmd_std'] for m in ms]

plugin_ms = sorted([int(m) for m in d16['plugin'].keys()])
plugin_means = [d16['plugin'][str(m)]['mmd_mean'] for m in plugin_ms]

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(plugin_ms, plugin_means, 'D--', color='#F44336', markersize=7, linewidth=1.5, label='Plug-in')
ax.errorbar(ms, learned_means, yerr=learned_stds, fmt='o-', color='#2196F3',
            markersize=7, linewidth=1.5, capsize=3, label='Learned (OFF)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Support size $m$', fontsize=14)
ax.set_ylabel('MMD$^2$', fontsize=14)
ax.set_xticks([1, 5, 10, 25, 50])
ax.set_xticklabels(['1', '5', '10', '25', '50'])
ax.legend(fontsize=11, loc='upper right')
ax.tick_params(labelsize=11)

from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/imagenet_support_scarcity.pdf', bbox_inches='tight')
plt.savefig('results/figures/imagenet_support_scarcity.png', bbox_inches='tight', dpi=150)
print('Saved to results/figures/imagenet_support_scarcity.{pdf,png}')
