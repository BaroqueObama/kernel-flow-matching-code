"""Appendix figures: collapse/mismatch, support sweep, and IP-Adapter null analysis."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

TITLE = 13
LABEL = 12
TICK = 10
LEGEND = 10
BLUE = '#2196F3'
RED = '#F44336'
GREEN = '#4CAF50'

def save(fig, name):
    from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(f'results/figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'results/figures/{name}.png', bbox_inches='tight', dpi=150)
    print(f'  Saved {name}')
    plt.close(fig)

print('1. ImageNet transition + Shells (combined)')
it = json.load(open('results/metrics/imagenet_test_eval.json'))
ip_baselines = json.load(open('results/metrics/imagenet_plugin_baselines.json'))
a3 = json.load(open('results/metrics/a3_shells_merged.json'))

groups = defaultdict(list)
for e in it:
    groups[(e['d'], e.get('use_exact_head', False))].append(e['mmd_mean'])
plugin_by_d = {e['d']: e['mmd_mean'] for e in ip_baselines if e.get('support_size') == 50}
dims = [8, 16, 32, 64]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

# Left: ImageNet transition
on_means, off_means, plug_means = [], [], []
for d in dims:
    on_vals = [v for v in groups.get((d, True), []) if v < 0.05]
    off_vals = [v for v in groups.get((d, False), []) if v < 0.05]
    on_means.append(np.mean(on_vals) if on_vals else np.nan)
    off_means.append(np.mean(off_vals) if off_vals else np.nan)
    plug_means.append(plugin_by_d.get(d, np.nan))

ax1.plot(dims, plug_means, 's--', color=RED, markersize=6, linewidth=1.5, label='Plug-in')
ax1.plot(dims, off_means, 'o-', color=BLUE, markersize=6, linewidth=1.5, label='Learned (OFF)')
ax1.plot(dims, on_means, 'D-', color='#FF9800', markersize=6, linewidth=1.5, label='Learned (ON)')
ax1.annotate('3/5 diverge', xy=(64, on_means[3] if not np.isnan(on_means[3]) else 0.02),
             xytext=(42, 0.018), fontsize=9, color='black',
             arrowprops=dict(arrowstyle='->', color='black'))
ax1.set_xscale('log', base=2)
ax1.set_xticks(dims)
ax1.set_xticklabels([str(d) for d in dims], fontsize=TICK)
ax1.set_xlabel('Dimension $d$', fontsize=LABEL)
ax1.set_ylabel('MMD$^2$', fontsize=LABEL)
ax1.tick_params(labelsize=TICK)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_title('(a) ImageNet: method comparison', fontsize=TITLE)

# Right: Shells bar chart
dims_shells = [entry['d'] for entry in a3['comparison_table']]
off_mmds = [entry['off_mmd'] for entry in a3['comparison_table']]
plugin_mmds = [entry['plugin_mmd'] for entry in a3['comparison_table']]

x = np.arange(len(dims_shells))
width = 0.35
ax2.bar(x - width/2, plugin_mmds, width, color=RED, edgecolor='black', linewidth=0.5, label='Plug-in')
ax2.bar(x + width/2, off_mmds, width, color=BLUE, edgecolor='black', linewidth=0.5, label='Learned (OFF)')
ax2.set_xticks(x)
ax2.set_xticklabels([f'$d={d}$' for d in dims_shells], fontsize=TICK)
ax2.set_ylabel('MMD$^2$', fontsize=LABEL)
ax2.set_yscale('log')
ax2.set_ylim(top=0.03)
ax2.legend(fontsize=9, loc='upper right')
ax2.tick_params(labelsize=TICK)
ax2.set_title('(b) Shells: learned vs. plug-in', fontsize=TITLE)
for i, (pm, lm) in enumerate(zip(plugin_mmds, off_mmds)):
    ax2.text(i - width/2, pm * 0.5, f'{pm/lm:.0f}$\\times$', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
save(fig, 'app_collapse_mismatch')


print('2. Support sweep (GMMs + Shells)')
c1 = json.load(open('results/metrics/c1_plugin_support_sweep_v2.json'))
wandb = json.load(open('results/metrics/wandb_all_runs.json'))
p3 = json.load(open('results/metrics/p3_shells_support_sweep.json'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

# Left: R^2 GMMs
plugin_ms = [5, 10, 25, 50, 100, 250, 500]
plugin_mmds_c1 = [c1[f'plugin_m{m}']['mmd_mean'] for m in plugin_ms if f'plugin_m{m}' in c1]
plugin_ms = [m for m in plugin_ms if f'plugin_m{m}' in c1]
support_runs = {5: 'b1_main_r2_support_size5_s42', 10: 'b1_main_r2_support_size10_s42',
                25: 'b1_main_r2_support_size25_s42', 50: 'b1_main_r2_s42',
                100: 'b1_main_r2_support_size100_s42', 250: 'b1_main_r2_support_size250_s42',
                500: 'b1_main_r2_support_size500_s42'}
learned_ms = sorted(support_runs.keys())
learned_mmds = [wandb[support_runs[m]]['summary']['val/mmd'] for m in learned_ms]

ax1.plot(plugin_ms, plugin_mmds_c1, 'D--', color=RED, markersize=6, linewidth=1.5, label='Plug-in')
ax1.plot(learned_ms, learned_mmds, 'o-', color=BLUE, markersize=6, linewidth=1.5, label='Learned (ON)')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Support size $m$', fontsize=LABEL)
ax1.set_ylabel('MMD$^2$', fontsize=LABEL)
ax1.legend(fontsize=9)
ax1.tick_params(labelsize=TICK)
ax1.set_title('(a) $\\mathbb{R}^2$ GMMs: support sweep', fontsize=TITLE)

# Right: Shells
ms_p3 = sorted([int(m) for m in p3['learned'].keys()])
learned_p3 = [p3['learned'][str(m)]['mmd_mean'] for m in ms_p3]
plugin_p3 = [p3['plugin'][str(m)]['mmd_mean'] for m in ms_p3]

ax2.plot(ms_p3, plugin_p3, 'D--', color=RED, markersize=6, linewidth=1.5, label='Plug-in')
ax2.plot(ms_p3, learned_p3, 'o-', color=BLUE, markersize=6, linewidth=1.5, label='Learned (OFF)')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Support size $m$', fontsize=LABEL)
ax2.set_ylabel('MMD$^2$', fontsize=LABEL)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=TICK)
ax2.set_title('(b) Shells $d=8$: support sweep', fontsize=TITLE)

plt.tight_layout()
save(fig, 'app_support_sweep')


print('3. IP-Adapter null analysis')
ip = json.load(open('results/metrics/ip_adapter_nw_correlation.json'))
null_data = json.load(open('results/metrics/ipadapter_null_analysis.json'))

timesteps = ip['timesteps']
all_rhos_by_t = {}
for t in timesteps:
    data = ip['results_by_timestep'][str(t)]
    all_rhos_by_t[t] = [d['per_row_spearman_mean'] for d in data]

rhos_t1 = all_rhos_by_t[1]
mean_rhos = [np.mean(all_rhos_by_t[t]) for t in timesteps]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))

IP_TITLE = 14
IP_LABEL = 13
IP_TICK = 11
IP_LEGEND = 11

# (a) Histogram of per-head rho at t=1
ax1.hist(rhos_t1, bins=30, color=BLUE, edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.axvline(np.mean(rhos_t1), color=RED, linewidth=2, linestyle='-', label=f'Mean = {np.mean(rhos_t1):.2f}')
ax1.axvline(0, color='gray', linewidth=1, linestyle='--', label='Permutation null')
ax1.set_xlabel('Spearman $\\rho$', fontsize=IP_LABEL)
ax1.set_ylabel('Count', fontsize=IP_LABEL)
ax1.legend(fontsize=IP_LEGEND, loc='upper left')
ax1.tick_params(labelsize=IP_TICK)
ax1.set_title('(a) Per-head $\\rho$ at $t=1$', fontsize=IP_TITLE)

# (b) CDF
sorted_rhos = np.sort(rhos_t1)
cdf = np.arange(1, len(sorted_rhos)+1) / len(sorted_rhos)
ax2.plot(sorted_rhos, cdf, color=BLUE, linewidth=2)
ax2.axvline(0.9, color='gray', linewidth=1, linestyle='--')
frac_above = sum(1 for r in rhos_t1 if r > 0.9) / len(rhos_t1)
ax2.axhline(1 - frac_above, color=RED, linewidth=1, linestyle=':', label=f'{frac_above*100:.0f}% above 0.9')
ax2.set_xlabel('Spearman $\\rho$', fontsize=IP_LABEL)
ax2.set_ylabel('CDF', fontsize=IP_LABEL)
ax2.legend(fontsize=IP_LEGEND)
ax2.tick_params(labelsize=IP_TICK)
ax2.set_title('(b) CDF at $t=1$', fontsize=IP_TITLE)

# (c) Mean rho vs timestep with null baselines
ax3.plot(timesteps, mean_rhos, 'o-', color=BLUE, markersize=6, linewidth=2, label='Observed $\\rho$')
ax3.fill_between(timesteps, [0]*len(timesteps), mean_rhos, alpha=0.1, color=BLUE)
ax3.axhline(1.0, color='gray', linewidth=1, linestyle='--', label='Same-NN ceiling')
ax3.axhline(0.0, color='gray', linewidth=1, linestyle=':', label='Permutation null')
ax3.set_xlim(1050, -50)
ax3.set_xlabel('Diffusion timestep $t$', fontsize=IP_LABEL)
ax3.set_ylabel('Mean Spearman $\\rho$', fontsize=IP_LABEL)
ax3.legend(fontsize=IP_LEGEND, loc='lower right')
ax3.tick_params(labelsize=IP_TICK)
ax3.annotate('denoising $\\rightarrow$', xy=(900, 0.05), fontsize=11, color='black')
ax3.set_title('(c) $\\rho$ vs. denoising with null baselines', fontsize=IP_TITLE)

plt.tight_layout()
save(fig, 'ipadapter_null_analysis')

print('\nDone!')
