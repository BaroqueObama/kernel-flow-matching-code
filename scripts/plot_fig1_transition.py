"""Figure 1: dimensionality transition -- exact-head benefit, kernel collapse, multi-head scaling."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

with open("results/metrics/h4_transition_merged.json") as f:
    trans = json.load(f)

with open("results/metrics/a1_multiseed_merged.json") as f:
    a1 = json.load(f)

with open("results/metrics/nw_weight_entropy_by_dim.json") as f:
    neff_data = json.load(f)

with open("results/metrics/g1_scaling_results.json") as f:
    g1 = json.load(f)

dims = [2, 4, 8, 16]
benefits = []
for entry in trans["combined_table"]:
    if entry["d"] in [2, 4]:
        benefits.append(entry["relative_benefit"] * 100)

for d in [8, 16]:
    on_mmd = a1["summary"][f"{d}_ON"]["mmd_mean"]
    off_mmd = a1["summary"][f"{d}_OFF"]["mmd_mean"]
    benefits.append((off_mmd - on_mmd) / off_mmd * 100)

# Error propagation: benefit = (OFF - ON)/OFF * 100
import numpy as np
benefit_errs = []
for entry in trans["combined_table"]:
    if entry["d"] in [2, 4]:
        on_std = entry.get("on_mmd_std", 0)
        off_std = entry.get("off_mmd_std", 0)
        on_m = entry["on_mmd"]
        off_m = entry["off_mmd"]
        if off_m > 0 and (on_std > 0 or off_std > 0):
            err = np.sqrt((on_std/off_m)**2 + (on_m*off_std/off_m**2)**2) * 100
        else:
            err = 0
        benefit_errs.append(err)
for d in [8, 16]:
    on_m = a1["summary"][f"{d}_ON"]["mmd_mean"]
    on_std = a1["summary"][f"{d}_ON"]["mmd_std"]
    off_m = a1["summary"][f"{d}_OFF"]["mmd_mean"]
    off_std = a1["summary"][f"{d}_OFF"]["mmd_std"]
    err = np.sqrt((on_std/off_m)**2 + (on_m*off_std/off_m**2)**2) * 100
    benefit_errs.append(err)

p_values = {2: 0.033, 4: 0.026}
if "8" in a1["t_tests"]:
    p_values[8] = a1["t_tests"]["8"]["mmd_p_value"]
if "16" in a1["t_tests"]:
    p_values[16] = a1["t_tests"]["16"]["mmd_p_value"]
significant = [p_values.get(d, 1.0) < 0.05 for d in dims]

# n_eff at t=0.56, averaged over tasks (no per-task std available)
neff_dims = [2, 4, 8, 16, 32, 64]
t_idx = 5  # t=0.56
neff_values = [neff_data["by_dim"][str(d)]["effective_sample_size"][t_idx] for d in neff_dims]

H_values = [1, 2, 4, 8, 16]
alphas = [g1["n_heads_results"][str(h)]["alpha"] for h in H_values]
alpha_cis = [g1["n_heads_results"][str(h)]["alpha_ci"] for h in H_values]
dk_values = [g1["n_heads_results"][str(h)]["d_k"] for h in H_values]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 11

colors = ["#2196F3" if sig else "#BBDEFB" for sig in significant]
ax1.bar(range(len(dims)), benefits, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
ax1.set_xticks(range(len(dims)))
ax1.set_xticklabels([f"$d={d}$" for d in dims], fontsize=TICK_SIZE)
ax1.set_ylabel("Exact head benefit (%)", fontsize=LABEL_SIZE)
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.set_ylim(-5, 35)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2196F3", edgecolor="black", label="$p < 0.05$"),
                   Patch(facecolor="#BBDEFB", edgecolor="black", label="n.s.")]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=11)
ax1.set_title("(a) Exact head benefit", fontsize=TITLE_SIZE)

ax2.plot(neff_dims, neff_values, "o-", color="#E91E63", markersize=7, linewidth=2)
ax2.axhline(1, color="gray", linewidth=1, linestyle="--", label="1-NN ($n_{\\mathrm{eff}}=1$)")
ax2.set_xlabel("Dimension $d$", fontsize=LABEL_SIZE)
ax2.set_ylabel("$n_{\\mathrm{eff}}$ at $t = 0.56$", fontsize=LABEL_SIZE)
ax2.set_xscale("log", base=2)
ax2.set_xticks(neff_dims)
ax2.set_xticklabels([f"${d}$" for d in neff_dims], fontsize=TICK_SIZE)
ax2.set_yscale("log")
ax2.set_ylim(0.8, 12)
ax2.tick_params(axis='y', labelsize=TICK_SIZE)
ax2.legend(loc="upper right", fontsize=11)
ax2.set_title("(b) Kernel collapse", fontsize=TITLE_SIZE)

yerr_low = [alphas[i] - alpha_cis[i][0] for i in range(len(H_values))]
yerr_high = [alpha_cis[i][1] - alphas[i] for i in range(len(H_values))]
ax3.errorbar(H_values, alphas, yerr=[yerr_low, yerr_high], fmt="s-", color="#4CAF50",
             markersize=7, linewidth=2, capsize=3)
ax3.set_xlabel("Number of heads $H$", fontsize=LABEL_SIZE)
ax3.set_ylabel("Scaling exponent $\\alpha$", fontsize=LABEL_SIZE)
ax3.set_xscale("log", base=2)
ax3.set_xticks(H_values)
ax3.set_xticklabels([f"${h}$" for h in H_values], fontsize=TICK_SIZE)
ax3.tick_params(axis='y', labelsize=TICK_SIZE)
ax3.set_ylim(0.45, 0.72)
ax3.set_title("(c) Multi-head scaling", fontsize=TITLE_SIZE)

plt.tight_layout()
from pathlib import Path; Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("results/figures/a1_multiseed_transition.pdf", bbox_inches="tight")
plt.savefig("results/figures/a1_multiseed_transition.png", bbox_inches="tight", dpi=150)
print("Saved to results/figures/a1_multiseed_transition.{pdf,png}")
