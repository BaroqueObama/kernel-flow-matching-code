"""Generate IP-Adapter NW correlation figure for the paper."""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

with open("results/metrics/ip_adapter_nw_correlation.json") as f:
    data = json.load(f)

timesteps = data["timesteps"]
results = data["results_by_timestep"]

# Aggregate per-timestep: mean Spearman rho across all heads/images
ts_rho = {}
ts_pearson = {}
ts_frac_above_09 = {}

for ts_str, entries in results.items():
    ts = int(ts_str)
    rhos = [e["per_row_spearman_mean"] for e in entries]
    pearsons = [e["raveled_pearson_r"] for e in entries]
    ts_rho[ts] = np.mean(rhos)
    ts_pearson[ts] = np.mean(pearsons)
    ts_frac_above_09[ts] = np.mean([1.0 if r > 0.9 else 0.0 for r in rhos])

sorted_ts = sorted(ts_rho.keys(), reverse=True)
rho_vals = [ts_rho[t] for t in sorted_ts]
pearson_vals = [ts_pearson[t] for t in sorted_ts]
frac_vals = [ts_frac_above_09[t] for t in sorted_ts]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

# Panel (a): Spearman rho vs timestep (denoising direction: high t = noisy, low t = clean)
ax = axes[0]
ax.plot(sorted_ts, rho_vals, 'o-', color='#2166ac', linewidth=2, markersize=7, label=r'Spearman $\rho$')
ax.plot(sorted_ts, pearson_vals, 's--', color='#b2182b', linewidth=2, markersize=6, label=r'Pearson $r$')
ax.set_xlabel('Diffusion timestep $t$', fontsize=11)
ax.set_ylabel('Correlation', fontsize=11)
ax.set_title(r'IP-Adapter attention $\approx$ NW weights', fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.set_ylim(0.3, 1.0)
ax.invert_xaxis()
ax.annotate(r'$\rho = 0.89$', xy=(1, ts_rho[1]), xytext=(150, 0.75),
            arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.5),
            fontsize=11, color='#2166ac', fontweight='bold')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.text(800, 0.52, 'chance', fontsize=8, color='gray')
ax.annotate('', xy=(100, 0.35), xytext=(900, 0.35),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
ax.text(500, 0.36, 'denoising', fontsize=8, color='gray', ha='center')

# Panel (b): fraction of heads with rho > 0.9 (bar chart)
ax = axes[1]
bars = ax.bar(range(len(sorted_ts)), [f * 100 for f in frac_vals],
              color=['#deebf7', '#9ecae1', '#4292c6', '#2171b5', '#084594'],
              edgecolor='#333333', linewidth=0.5)
ax.set_xticks(range(len(sorted_ts)))
ax.set_xticklabels([str(t) for t in sorted_ts], fontsize=9)
ax.set_xlabel('Diffusion timestep $t$', fontsize=11)
ax.set_ylabel(r'Heads with $\rho > 0.9$ (%)', fontsize=11)
ax.set_title('Fraction of NW-like heads', fontsize=12)
ax.set_ylim(0, 100)
for i, (bar, frac) in enumerate(zip(bars, frac_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{frac*100:.0f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig('results/figures/ipadapter_nw_correlation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('results/figures/ipadapter_nw_correlation.png', bbox_inches='tight', dpi=300)
print("Saved ipadapter_nw_correlation.pdf")
