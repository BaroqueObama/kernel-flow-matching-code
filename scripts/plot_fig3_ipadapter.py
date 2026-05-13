"""Generate Figure 3: IP-Adapter attention weights vs NW kernel weights."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

ip = json.load(open('results/metrics/ip_adapter_nw_correlation.json'))

timesteps = ip['timesteps']  # [981, 761, 501, 261, 1]

spearmans = []
pearsons = []
fracs = []
for t in timesteps:
    data = ip['results_by_timestep'][str(t)]
    sp = [d['per_row_spearman_mean'] for d in data]
    pe = [d['raveled_pearson_r'] for d in data]
    spearmans.append(np.mean(sp))
    pearsons.append(np.mean(pe))
    fracs.append(sum(1 for s in sp if s > 0.9) / len(sp) * 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

# Left: correlation vs timestep
ax1.plot(timesteps, spearmans, 'o-', color='#2196F3', markersize=7, linewidth=2, label='Spearman $\\rho$')
ax1.plot(timesteps, pearsons, 'D--', color='#F44336', markersize=7, linewidth=2, label='Pearson $r$')
ax1.set_xlabel('Diffusion timestep $t$', fontsize=14)
ax1.set_ylabel('Correlation', fontsize=14)
ax1.set_xlim(1050, -50)  # reverse axis: denoising goes right
ax1.set_ylim(0.3, 1.0)
ax1.legend(fontsize=11, loc='lower right')
ax1.tick_params(labelsize=11)
ax1.annotate('denoising $\\rightarrow$', xy=(700, 0.35), fontsize=12, color='black', ha='center')
ax1.set_title('(a) IP-Adapter attention $\\approx$ NW weights', fontsize=14)

# Right: fraction of heads with rho > 0.9
colors = ['#90CAF9', '#64B5F6', '#42A5F5', '#1E88E5', '#1565C0']
bars = ax2.bar(range(len(timesteps)), fracs, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(timesteps)))
ax2.set_xticklabels([str(t) for t in timesteps])
ax2.set_xlabel('Diffusion timestep $t$', fontsize=14)
ax2.set_ylabel('Heads with $\\rho > 0.9$ (%)', fontsize=14)
ax2.tick_params(labelsize=11)
ax2.set_ylim(0, 68)
ax2.set_title('(b) Fraction of NW-like heads', fontsize=14)
# Add percentage labels on bars
for i, (bar, frac) in enumerate(zip(bars, fracs)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{frac:.0f}%', ha='center', fontsize=12)

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/ipadapter_nw_correlation.pdf', bbox_inches='tight')
plt.savefig('results/figures/ipadapter_nw_correlation.png', bbox_inches='tight', dpi=150)
print('Saved to results/figures/ipadapter_nw_correlation.{pdf,png}')
