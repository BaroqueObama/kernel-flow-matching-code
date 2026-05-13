"""Figure 2 (mismatch variant): failure-mode independence and learned kernel analysis."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

labels = ['GMM\n$d\\!=\\!16$', 'Shells\n$d\\!=\\!8$', 'Shells\n$d\\!=\\!32$', 'ImageNet\n$d\\!=\\!16$', 'Whitened\nImageNet']
neffs = [1.0, 6.5, 8.0, 13.4, 1.05]
ratios = [0.91, 11.2, 6.4, 0.99, 0.74]

x = np.arange(len(labels))
width = 0.35

bars1 = ax1.bar(x - width/2, neffs, width, color='#90CAF9', edgecolor='black', linewidth=0.5, label='$n_{\\mathrm{eff}}$')
bars2 = ax1.bar(x + width/2, ratios, width, color='#EF9A9A', edgecolor='black', linewidth=0.5, label='Learned/plug-in')

ax1.axhline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Value", fontsize=12)
ax1.set_yscale("log")
ax1.set_ylim(0.5, 20)
ax1.legend(fontsize=9, loc='upper left')
ax1.tick_params(axis='y', labelsize=11)
ax1.set_title("(a) Independence of failure modes", fontsize=12)

kernels = json.load(open('results/metrics/learned_kernels_shells.json'))

timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
t_keys = ['t_0.1', 't_0.3', 't_0.5', 't_0.7', 't_0.9']

nw_neffs = [kernels['d_8'][tk]['nw_neff_mean'] for tk in t_keys]

n_heads = len(kernels['d_8']['t_0.5']['learned_neff_per_head'])
head_colors = ['#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']
for h_idx in range(n_heads):
    head_neffs = [kernels['d_8'][tk]['learned_neff_per_head'][h_idx][0] for tk in t_keys]
    ax2.plot(timesteps, head_neffs, 'o-', color=head_colors[h_idx], markersize=5,
             linewidth=1.2, alpha=0.7, label=f'Learned (Head {h_idx+1})' if h_idx < 2 else None)

ax2.plot(timesteps, nw_neffs, 'D--', color='#F44336', markersize=7, linewidth=2, label='NW (isotropic)')

# Dummy artists so heads 3-4 appear in legend without re-plotting
ax2.plot([], [], 'o-', color=head_colors[2], markersize=5, linewidth=1.2, alpha=0.7, label='Learned (Head 3)')
ax2.plot([], [], 'o-', color=head_colors[3], markersize=5, linewidth=1.2, alpha=0.7, label='Learned (Head 4)')

ax2.axhline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xlabel("Flow time $t$", fontsize=14)
ax2.set_ylabel("$n_{\\mathrm{eff}}$", fontsize=14)
ax2.set_yscale("log")
ax2.set_ylim(0.8, 55)
ax2.legend(fontsize=8, loc='lower left', ncol=2)
ax2.tick_params(labelsize=11)
ax2.set_title("(b) Shells $d\\!=\\!8$: learned heads resist collapse", fontsize=12)

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/mismatch_scatter.pdf', bbox_inches='tight')
plt.savefig('results/figures/mismatch_scatter.png', bbox_inches='tight', dpi=150)
print('Saved to results/figures/mismatch_scatter.{pdf,png}')
