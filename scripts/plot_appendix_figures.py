"""Appendix figures: ImageNet transition, shells comparison, whitened control, support sweep, attention."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

TITLE = 14
LABEL = 13
TICK = 11
LEGEND = 11
BLUE = '#2196F3'
RED = '#F44336'
GREEN = '#4CAF50'


def save(fig, name):
    from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(f'results/figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'results/figures/{name}.png', bbox_inches='tight', dpi=150)
    print(f'  Saved {name}')
    plt.close(fig)


print('1. ImageNet transition curve')
it = json.load(open('results/metrics/imagenet_test_eval.json'))
ip = json.load(open('results/metrics/imagenet_plugin_baselines.json'))

groups = defaultdict(list)
for e in it:
    groups[(e['d'], e.get('use_exact_head', False))].append(e['mmd_mean'])

dims = [8, 16, 32, 64]
plugin_mmds = {e['d']: e['mmd_mean'] for e in ip if e.get('support_size') == 50}

fig, ax = plt.subplots(figsize=(5, 3.5))
# Exclude diverged runs (mmd > 0.05) from means
on_means, off_means, plug_means = [], [], []
for d in dims:
    on_vals = [v for v in groups.get((d, True), []) if v < 0.05]
    off_vals = [v for v in groups.get((d, False), []) if v < 0.05]
    on_means.append(np.mean(on_vals) if on_vals else np.nan)
    off_means.append(np.mean(off_vals) if off_vals else np.nan)
    plug_means.append(plugin_mmds.get(d, np.nan))

ax.plot(dims, plug_means, 's--', color=GREEN, markersize=7, linewidth=1.5, label='Plug-in')
ax.plot(dims, off_means, 'o-', color=BLUE, markersize=7, linewidth=1.5, label='Learned (OFF)')
ax.plot(dims, on_means, 'D-', color=RED, markersize=7, linewidth=1.5, label='Learned (ON)')
ax.annotate('3/5 diverge', xy=(64, on_means[3] if not np.isnan(on_means[3]) else 0.02),
            xytext=(45, 0.018), fontsize=10, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED))

ax.set_xscale('log', base=2)
ax.set_xticks(dims)
ax.set_xticklabels([str(d) for d in dims], fontsize=TICK)
ax.set_xlabel('Dimension $d$', fontsize=LABEL)
ax.set_ylabel('MMD$^2$', fontsize=LABEL)
ax.tick_params(labelsize=TICK)
ax.legend(fontsize=LEGEND, loc='upper left')
ax.set_title('ImageNet DINOv2+PCA: method comparison vs. $d$', fontsize=TITLE)
save(fig, 'imagenet_transition_curve')


print('2. Shells vs GMM')
a3 = json.load(open('results/metrics/a3_shells_merged.json'))

fig, ax = plt.subplots(figsize=(5, 4))
dims_shells = []
off_mmds = []
plugin_mmds_shells = []
for entry in a3['comparison_table']:
    dims_shells.append(entry['d'])
    off_mmds.append(entry['off_mmd'])
    plugin_mmds_shells.append(entry['plugin_mmd'])

x = np.arange(len(dims_shells))
width = 0.35
ax.bar(x - width/2, plugin_mmds_shells, width, color=RED, edgecolor='black', linewidth=0.5, label='Plug-in')
ax.bar(x + width/2, off_mmds, width, color=BLUE, edgecolor='black', linewidth=0.5, label='Learned (OFF)')
ax.set_xticks(x)
ax.set_xticklabels([f'$d={d}$' for d in dims_shells], fontsize=TICK)
ax.set_ylabel('MMD$^2$', fontsize=LABEL)
ax.set_yscale('log')
ax.legend(fontsize=LEGEND, loc='lower right')
ax.tick_params(labelsize=TICK)
ax.set_ylim(top=0.03)
ax.set_title('Spherical shells: learned vs. plug-in', fontsize=TITLE)
for i, (pm, lm) in enumerate(zip(plugin_mmds_shells, off_mmds)):
    ratio = pm / lm
    ax.text(i - width/2, pm * 0.5, f'{ratio:.0f}$\\times$', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
save(fig, 'a3_shells_vs_gmm')


print('3. Whitened ImageNet')
wh = json.load(open('results/metrics/whitened_imagenet_nw_entropy.json'))
gs = wh['graded_sweep']

alphas = [e['alpha'] for e in gs]
neffs = [e['n_eff'] for e in gs]
plugin_mmds_wh = [e['plugin_mmd'] for e in gs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

ax1.plot(alphas, neffs, 'o-', color=BLUE, markersize=7, linewidth=2)
ax1.set_xlabel('Whitening strength $\\alpha$', fontsize=LABEL)
ax1.set_ylabel('$n_{\\mathrm{eff}}$', fontsize=LABEL)
ax1.tick_params(labelsize=TICK)
ax1.set_title('(a) $n_{\\mathrm{eff}}$ drops with whitening', fontsize=TITLE)

ax2.plot(alphas, plugin_mmds_wh, 'D-', color=RED, markersize=7, linewidth=2)
ax2.set_xlabel('Whitening strength $\\alpha$', fontsize=LABEL)
ax2.set_ylabel('Plug-in MMD$^2$', fontsize=LABEL)
ax2.tick_params(labelsize=TICK)
ax2.set_title('(b) Plug-in quality unchanged', fontsize=TITLE)
ymin, ymax = min(plugin_mmds_wh), max(plugin_mmds_wh)
ax2.set_ylim(ymin * 0.8, ymax * 1.3)

plt.tight_layout()
save(fig, 'whitened_imagenet_comparison')


print('4. Support size sweep')
c1 = json.load(open('results/metrics/c1_plugin_support_sweep_v2.json'))
wandb = json.load(open('results/metrics/wandb_all_runs.json'))

plugin_ms_c1 = [5, 10, 25, 50, 100]
plugin_mmds_c1 = [c1[f'plugin_m{m}']['mmd_mean'] for m in plugin_ms_c1]

support_runs = {
    5: 'b1_main_r2_support_size5_s42',
    10: 'b1_main_r2_support_size10_s42',
    25: 'b1_main_r2_support_size25_s42',
    50: 'b1_main_r2_s42',
    100: 'b1_main_r2_support_size100_s42',
    250: 'b1_main_r2_support_size250_s42',
    500: 'b1_main_r2_support_size500_s42',
}
learned_ms_c1 = sorted(support_runs.keys())
learned_mmds_c1 = [wandb[support_runs[m]]['summary']['val/mmd'] for m in learned_ms_c1]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(plugin_ms_c1, plugin_mmds_c1, 'D--', color=RED, markersize=7, linewidth=1.5, label='Plug-in')
ax.plot(learned_ms_c1, learned_mmds_c1, 'o-', color=BLUE, markersize=7, linewidth=1.5, label='Learned (ON)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Support size $m$', fontsize=LABEL)
ax.set_ylabel('MMD$^2$', fontsize=LABEL)
ax.set_xticks([5, 10, 25, 50, 100, 250, 500])
ax.set_xticklabels(['5', '10', '25', '50', '100', '250', '500'], fontsize=9)
ax.legend(fontsize=LEGEND)
ax.tick_params(labelsize=TICK)
ax.set_title('$\\mathbb{R}^2$ GMM: learned vs. plug-in support sweep', fontsize=TITLE)
save(fig, 'c1_support_size_learned_vs_plugin')


print('5. Shells support sweep')
p3 = json.load(open('results/metrics/p3_shells_support_sweep.json'))

ms_p3 = sorted([int(m) for m in p3['learned'].keys()])
learned_p3 = [p3['learned'][str(m)]['mmd_mean'] for m in ms_p3]
plugin_p3 = [p3['plugin'][str(m)]['mmd_mean'] for m in ms_p3]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(ms_p3, plugin_p3, 'D--', color=RED, markersize=7, linewidth=1.5, label='Plug-in')
ax.plot(ms_p3, learned_p3, 'o-', color=BLUE, markersize=7, linewidth=1.5, label='Learned (OFF)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Support size $m$', fontsize=LABEL)
ax.set_ylabel('MMD$^2$', fontsize=LABEL)
ax.legend(fontsize=LEGEND)
ax.tick_params(labelsize=TICK)
ax.set_title('Shells $d=8$: support sweep', fontsize=TITLE)
save(fig, 'p3_shells_support_sweep')


print('6. Attention analysis')
g2 = json.load(open('results/metrics/g2_g3_attention_combined.json'))

target = None
for entry in g2:
    cfg = entry['config_summary']
    if isinstance(cfg, dict) and cfg.get('d') == 8 and cfg.get('n_heads') == 8:
        target = entry
        break

if target is not None:
    entropy = target['entropy']
    # Group by layer and head: each entry has time_centers, entropy_mean, entropy_std
    heads_layer0 = {}
    for key, vals in entropy.items():
        parts = key.split('_')  # L0_H0
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        if layer == 0:
            heads_layer0[head] = vals

    n_heads = len(heads_layer0)
    time_centers = heads_layer0[0]['time_centers']

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_heads))

    for h_idx in sorted(heads_layer0.keys()):
        vals = heads_layer0[h_idx]
        ax.plot(time_centers, vals['entropy_mean'], '-', color=colors[h_idx],
                linewidth=1.5, label=f'Head {h_idx}')

    ax.set_xlabel('Flow time $t$', fontsize=LABEL)
    ax.set_ylabel('Attention entropy (nats)', fontsize=LABEL)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.tick_params(labelsize=TICK)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_title('$d=8$, $H=8$: per-head entropy vs. flow time (layer 0)', fontsize=TITLE)
    save(fig, 'g2_g3_attention_d=8_H=8_dk=16')
else:
    print('  WARNING: d=8 H=8 entry not found in attention data')


print('\nDone! All appendix figures regenerated.')
