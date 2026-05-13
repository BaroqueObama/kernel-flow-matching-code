"""Figure 2: what learning corrects -- learned kernel behavior and support scarcity."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

kernels = json.load(open('results/metrics/learned_kernels_shells.json'))

timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
t_keys = ['t_0.1', 't_0.3', 't_0.5', 't_0.7', 't_0.9']

nw_neffs = [kernels['d_8'][tk]['nw_neff_mean'] for tk in t_keys]
n_heads = len(kernels['d_8']['t_0.5']['learned_neff_per_head'])

for h_idx in range(n_heads):
    head_neffs = [kernels['d_8'][tk]['learned_neff_per_head'][h_idx][0] for tk in t_keys]
    ax1.plot(timesteps, head_neffs, 'o-', color='#2196F3', markersize=5,
             linewidth=1.5, label='4 Learned heads' if h_idx == 0 else None)

ax1.plot(timesteps, nw_neffs, 'D--', color='#F44336', markersize=7, linewidth=2, label='NW (isotropic)')

ax1.axhline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_xlabel("Flow time $t$", fontsize=14)
ax1.set_ylabel("$n_{\\mathrm{eff}}$", fontsize=14)
ax1.set_yscale("log")
ax1.set_ylim(0.8, 55)
ax1.legend(fontsize=11, loc='lower left')
ax1.tick_params(labelsize=11)
ax1.set_title("(a) Shells $d\\!=\\!8$: learned heads resist collapse", fontsize=14)

scarcity = json.load(open('results/metrics/fig3_scarcity_data.json'))
d16 = scarcity['d_16']

ms = sorted([int(m) for m in d16['learned'].keys()])
learned_means = [d16['learned'][str(m)]['mmd_mean'] for m in ms]
learned_stds = [d16['learned'][str(m)]['mmd_std'] for m in ms]

plugin_ms = sorted([int(m) for m in d16['plugin'].keys()])
plugin_means = [d16['plugin'][str(m)]['mmd_mean'] for m in plugin_ms]

ax2.plot(plugin_ms, plugin_means, 'D--', color='#F44336', markersize=7, linewidth=1.5, label='Plug-in')
ax2.errorbar(ms, learned_means, yerr=learned_stds, fmt='o-', color='#2196F3',
             markersize=7, linewidth=1.5, capsize=3, label='Learned (OFF)')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Support size $m$', fontsize=13)
ax2.set_ylabel('MMD$^2$', fontsize=13)
ax2.set_xticks([1, 5, 10, 25, 50])
ax2.set_xticklabels(['1', '5', '10', '25', '50'])
ax2.legend(fontsize=11, loc='upper right')
ax2.tick_params(labelsize=11)
ax2.set_title('(b) ImageNet $d\\!=\\!16$: learned wins at small $m$', fontsize=14)

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/fig2_what_learning_corrects.pdf', bbox_inches='tight')
plt.savefig('results/figures/fig2_what_learning_corrects.png', bbox_inches='tight', dpi=150)
print('Saved to results/figures/fig2_what_learning_corrects.{pdf,png}')
