"""
Generates publication-quality figures for the BEE26 mathematical model:
"To Gossip or Not To Gossip: The Evolution of Cooperation Under Horizontal Gene Transfer"

Strict Design Rules Enforced:
1. NO text anywhere inside the graph data area (no ax.text inside axes).
   Data area contains ONLY data curves and background shading.
2. Single unified legend placed ONCE outside in multi-panel figures.
3. Legends in single plots placed strictly outside the axes area.
4. Consistent font, line styles, and color palette across all figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from models.simulation import (
    base_model_ode,
    independent_plasmid_ode,
    dependent_plasmid_ode,
    synergy_model_ode,
    simulate_system,
    parameter_sweep
)

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------
# Global Typography & Style Configuration
# -------------------------------------------------------------
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.color": "#999999"
})

# -------------------------------------------------------------
# Strictly Unified Color Palette
# -------------------------------------------------------------
COLOR_GC = "#1a5fb4"        # Generous Cooperator (Cobalt Blue)
COLOR_GD = "#e66100"        # Generous Defector (Amber Orange)
COLOR_SC = "#26a269"        # Secretive Cooperator (Forest Green)
COLOR_SD = "#c01c28"        # Secretive Defector (Crimson Red)
COLOR_SC_PLUS = "#8e24aa"   # Plasmid-bearing SC (Vibrant Purple)
COLOR_SD_PLUS = "#681313"   # Plasmid-bearing SD (Deep Maroon)

# Shading tones
SHADE_SD = "#ffebee"        # Soft Red
SHADE_COEX = "#fff9c4"      # Soft Amber
SHADE_SC_PLUS = "#f3e5f5"   # Soft Purple
SHADE_GC = "#e3f2fd"        # Soft Blue

# Simulation parameters
fc = 0.2
fs = 0.1
alpha = 0.1
ratios = np.linspace(0.0, 2.0, 101)


# =============================================================
# 1. Base Model Bifurcation (Slide 10)
# =============================================================
print("Generating: slide10_base_model_result.png")
y0_base = [0.25, 0.25, 0.25, 0.25]
results_base = parameter_sweep(
    base_model_ode,
    ratios,
    lambda r: (alpha, fc, r * fc),
    y0_base,
    t_span=(0, 200)
)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(ratios, results_base[:, 0], label="GC (Generous Cooperator)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios, results_base[:, 1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios, results_base[:, 2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios, results_base[:, 3], label="SD (Secretive Defector)", color=COLOR_SD, linestyle="-", linewidth=3.0)

ax.set_title("Base Model — Steady State vs $f_s / f_c$ (Slide 10)", pad=14)
ax.set_xlabel(r"Ratio of HGT Cost to Cooperation Cost ($f_s / f_c$)")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)

# Legend placed outside on the right - ZERO text in data area!
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide10_base_model_result.png")
plt.close()


# =============================================================
# 2. Base Model Trajectory
# =============================================================
print("Generating: base_model_trajectory.png")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
t_eval, y_traj_base = simulate_system(base_model_ode, (0, 100), y0_base, args=(alpha, fc, fs))
ax.plot(t_eval, y_traj_base[0], label="GC (Generous Cooperator)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(t_eval, y_traj_base[1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(t_eval, y_traj_base[2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(t_eval, y_traj_base[3], label="SD (Secretive Defector)", color=COLOR_SD, linestyle="-", linewidth=3.0)

ax.set_title("Base Model — Population Trajectory over Time ($f_s=0.1, f_c=0.2, \\alpha=0.1$)", pad=14)
ax.set_xlabel("Time ($t$, generations)")
ax.set_ylabel("Population Fraction ($x_i$)")
ax.set_xlim(0, 100)
ax.set_ylim(-0.02, 1.04)

ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/base_model_trajectory.png")
plt.close()


# =============================================================
# 3. Independent Beneficial Plasmid Model (Slide 15)
# =============================================================
print("Generating: slide15_independent_plasmid_result.png")
fplus = 0.3
y0_indep = [1/8.0] * 8
results_indep = parameter_sweep(
    independent_plasmid_ode,
    ratios,
    lambda r: (alpha, fc, r * fc, fplus),
    y0_indep,
    t_span=(0, 200)
)

fig, ax = plt.subplots(figsize=(8.8, 4.8))
total_sd = results_indep[:, 6] + results_indep[:, 7]
sd_plus = results_indep[:, 6]
total_gc = results_indep[:, 0] + results_indep[:, 1]
total_gd = results_indep[:, 2] + results_indep[:, 3]
total_sc = results_indep[:, 4] + results_indep[:, 5]

ax.plot(ratios, total_gc, label="GC Total ($GC_+ + GC_-$)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios, total_gd, label="GD Total ($GD_+ + GD_-$)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios, total_sc, label="SC Total ($SC_+ + SC_-$)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios, total_sd, label=r"SD Total ($SD_+ + SD_-$)", color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.plot(ratios, sd_plus, label=r"$SD_+$ (Plasmid Defector)", color=COLOR_SD_PLUS, linestyle="--", linewidth=2.0)

ax.set_title("Independent Beneficial Plasmid Model — Steady State (Slide 15)", pad=14)
ax.set_xlabel(r"Ratio of HGT Cost to Cooperation Cost ($f_s / f_c$)")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)

ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide15_independent_plasmid_result.png")
plt.close()


# =============================================================
# 4. Independent Beneficial Plasmid Trajectory
# =============================================================
print("Generating: independent_plasmid_trajectory.png")
fig, ax = plt.subplots(figsize=(8.8, 4.8))
t_eval, y_traj_indep = simulate_system(independent_plasmid_ode, (0, 100), y0_indep, args=(alpha, fc, fs, fplus))
ax.plot(t_eval, y_traj_indep[0] + y_traj_indep[1], label="GC Total ($GC_+ + GC_-$)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(t_eval, y_traj_indep[2] + y_traj_indep[3], label="GD Total ($GD_+ + GD_-$)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(t_eval, y_traj_indep[4] + y_traj_indep[5], label="SC Total ($SC_+ + SC_-$)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(t_eval, y_traj_indep[6], label="$SD_+$ (Plasmid Defector)", color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.plot(t_eval, y_traj_indep[7], label="$SD_-$ (Plasmid-Free Defector)", color=COLOR_SD_PLUS, linestyle=":", linewidth=2.0)

ax.set_title("Independent Beneficial Plasmid Model — Trajectory over Time", pad=14)
ax.set_xlabel("Time ($t$, generations)")
ax.set_ylabel("Population Fraction ($x_i$)")
ax.set_xlim(0, 100)
ax.set_ylim(-0.02, 1.04)

ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/independent_plasmid_trajectory.png")
plt.close()


# =============================================================
# 5. Dependent Beneficial Plasmid Model (Slide 20)
# =============================================================
print("Generating: slide20_dependent_plasmid_result.png")
ratios_fplus = np.linspace(0.0, 2.0, 101)
y0_dep = [1/6.0] * 6
results_dep = parameter_sweep(
    dependent_plasmid_ode,
    ratios_fplus,
    lambda r: (alpha, fc, fs, r * fc),
    y0_dep,
    t_span=(0, 150)
)

fig, ax = plt.subplots(figsize=(8.8, 4.8))
# Shaded background zones (no text inside data area!)
ax.axvspan(0.0, 0.95, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.95, 1.05, color=SHADE_COEX, alpha=0.6)
ax.axvspan(1.05, 2.0, color=SHADE_SC_PLUS, alpha=0.35)

ax.plot(ratios_fplus, results_dep[:, 0], label="GC ($GC_+$)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios_fplus, results_dep[:, 2], label="GD", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios_fplus, results_dep[:, 3], label=r"$SC_+$ (Beneficial Cooperator)", color=COLOR_SC_PLUS, linestyle="-", linewidth=3.0)
ax.plot(ratios_fplus, results_dep[:, 5], label="SD (Secretive Defector)", color=COLOR_SD, linestyle="-", linewidth=3.0)

ax.set_title("Dependent Beneficial Plasmid Model — Steady State vs $f_+ / f_c$ (Slide 20)", pad=14)
ax.set_xlabel(r"Ratio of Plasmid Benefit to Cooperation Cost ($f_+ / f_c$)")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)

ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide20_dependent_plasmid_result.png")
plt.close()


# =============================================================
# 6. Dependent Beneficial Plasmid Trajectory (1x2 Subplots)
# =============================================================
print("Generating: dependent_plasmid_trajectory.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

# Subcritical
t_eval, y_sub = simulate_system(dependent_plasmid_ode, (0, 120), y0_dep, args=(alpha, fc, fs, 0.7 * fc))
ax1.plot(t_eval, y_sub[0], color=COLOR_GC, linestyle="--", linewidth=2.0)
ax1.plot(t_eval, y_sub[2], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax1.plot(t_eval, y_sub[3], color=COLOR_SC_PLUS, linestyle="-", linewidth=2.0)
ax1.plot(t_eval, y_sub[5], color=COLOR_SD, linestyle="-", linewidth=3.0)
ax1.set_title("Subcritical Regime: $f_+ / f_c = 0.7$ (SD Wins)", pad=10)
ax1.set_xlabel("Time ($t$, generations)")
ax1.set_ylabel("Population Fraction ($x_i$)")
ax1.set_xlim(0, 120)
ax1.set_ylim(-0.02, 1.04)

# Supercritical
t_eval, y_super = simulate_system(dependent_plasmid_ode, (0, 120), y0_dep, args=(alpha, fc, fs, 1.5 * fc))
ax2.plot(t_eval, y_super[0], color=COLOR_GC, linestyle="--", linewidth=2.0)
ax2.plot(t_eval, y_super[2], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax2.plot(t_eval, y_super[3], color=COLOR_SC_PLUS, linestyle="-", linewidth=3.0)
ax2.plot(t_eval, y_super[5], color=COLOR_SD, linestyle="-", linewidth=2.0)
ax2.set_title("Supercritical Regime: $f_+ / f_c = 1.5$ ($SC_+$ Fixation)", pad=10)
ax2.set_xlabel("Time ($t$, generations)")
ax2.set_xlim(0, 120)

# SINGLE shared legend placed ONCE at top outside the data area
legend_elements = [
    Line2D([0], [0], color=COLOR_GC, linestyle="--", linewidth=2.0, label="GC"),
    Line2D([0], [0], color=COLOR_GD, linestyle="-.", linewidth=2.0, label="GD"),
    Line2D([0], [0], color=COLOR_SC_PLUS, linestyle="-", linewidth=2.5, label="$SC_+$"),
    Line2D([0], [0], color=COLOR_SD, linestyle="-", linewidth=2.5, label="SD")
]
fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, framealpha=0.95, edgecolor="#cccccc")

plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.savefig("figures/dependent_plasmid_trajectory.png")
plt.close()


# =============================================================
# 7. Plasmid-Chromosome Synergy Model (Slide 24)
# =============================================================
print("Generating: slide24_synergy_model_result.png")
ratios_fpc = np.linspace(0.0, 2.0, 101)
y0_synergy = [0.25, 0.25, 0.25, 0.25]
results_synergy = parameter_sweep(
    synergy_model_ode,
    ratios_fpc,
    lambda r: (alpha, fc, fs, r * (fs + fc)),
    y0_synergy,
    t_span=(0, 50)
)

fig, ax = plt.subplots(figsize=(8.8, 4.8))
# Shaded background zones (no text inside data area!)
ax.axvspan(0.0, 0.65, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.65, 0.95, color=SHADE_COEX, alpha=0.6)
ax.axvspan(0.95, 2.0, color=SHADE_GC, alpha=0.35)

ax.plot(ratios_fpc, results_synergy[:, 0], label="GC (Generous Cooperator)", color=COLOR_GC, linestyle="-", linewidth=3.0)
ax.plot(ratios_fpc, results_synergy[:, 1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios_fpc, results_synergy[:, 2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios_fpc, results_synergy[:, 3], label="SD (Secretive Defector)", color=COLOR_SD, linestyle="-", linewidth=3.0)

ax.set_title(r"Plasmid–Chromosome Synergy Model — Steady State (Slide 24)", pad=14)
ax.set_xlabel(r"Synergy Parameter $\Gamma = f_{pc} / (f_s + f_c)$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)

ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide24_synergy_model_result.png")
plt.close()


# =============================================================
# 8. Synergy Model Trajectory (1x2 Subplots)
# =============================================================
print("Generating: synergy_model_trajectory.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

# Low Synergy
t_eval, y_low = simulate_system(synergy_model_ode, (0, 100), y0_synergy, args=(alpha, fc, fs, 0.5 * (fs + fc)))
ax1.plot(t_eval, y_low[0], color=COLOR_GC, linestyle="--", linewidth=2.0)
ax1.plot(t_eval, y_low[1], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax1.plot(t_eval, y_low[2], color=COLOR_SC, linestyle=":", linewidth=2.0)
ax1.plot(t_eval, y_low[3], color=COLOR_SD, linestyle="-", linewidth=3.0)
ax1.set_title(r"Low Synergy: $\Gamma = 0.5$ (SD Fixation)", pad=10)
ax1.set_xlabel("Time ($t$, generations)")
ax1.set_ylabel("Population Fraction ($x_i$)")
ax1.set_xlim(0, 100)
ax1.set_ylim(-0.02, 1.04)

# High Synergy
t_eval, y_high = simulate_system(synergy_model_ode, (0, 100), y0_synergy, args=(alpha, fc, fs, 1.5 * (fs + fc)))
ax2.plot(t_eval, y_high[0], color=COLOR_GC, linestyle="-", linewidth=3.0)
ax2.plot(t_eval, y_high[1], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax2.plot(t_eval, y_high[2], color=COLOR_SC, linestyle=":", linewidth=2.0)
ax2.plot(t_eval, y_high[3], color=COLOR_SD, linestyle="-", linewidth=2.0)
ax2.set_title(r"High Synergy: $\Gamma = 1.5$ (GC Fixation — HGT & Cooperation Win!)", pad=10)
ax2.set_xlabel("Time ($t$, generations)")
ax2.set_xlim(0, 100)

# SINGLE shared legend placed ONCE at top outside data area
legend_elements_syn = [
    Line2D([0], [0], color=COLOR_GC, linestyle="-", linewidth=2.5, label="GC"),
    Line2D([0], [0], color=COLOR_GD, linestyle="-.", linewidth=2.0, label="GD"),
    Line2D([0], [0], color=COLOR_SC, linestyle=":", linewidth=2.0, label="SC"),
    Line2D([0], [0], color=COLOR_SD, linestyle="-", linewidth=2.5, label="SD")
]
fig.legend(handles=legend_elements_syn, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, framealpha=0.95, edgecolor="#cccccc")

plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.savefig("figures/synergy_model_trajectory.png")
plt.close()


# =============================================================
# 9. Master Comparison Overview (2x2 Grid)
# =============================================================
print("Generating: model_comparison_overview.png")
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey=True)

# Panel (a): Base Model
ax = axes[0, 0]
ax.plot(ratios, results_base[:, 0], color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios, results_base[:, 1], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios, results_base[:, 2], color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios, results_base[:, 3], color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.axvspan(0.0, 2.0, color=SHADE_SD, alpha=0.35)
ax.set_title("(a) Base Model: Costly HGT & Cooperation", pad=10)
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)

# Panel (b): Independent Beneficial Plasmid
ax = axes[0, 1]
ax.plot(ratios, total_gc, color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios, total_gd, color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios, total_sc, color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios, total_sd, color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.plot(ratios, sd_plus, color=COLOR_SD_PLUS, linestyle="--", linewidth=1.8)
ax.axvspan(0.0, 2.0, color=SHADE_SD, alpha=0.35)
ax.set_title("(b) Independent Beneficial Plasmid Model", pad=10)
ax.set_xlabel(r"$f_s / f_c$")
ax.set_xlim(0.0, 2.0)

# Panel (c): Dependent Beneficial Plasmid
ax = axes[1, 0]
ax.axvspan(0.0, 0.95, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.95, 1.05, color=SHADE_COEX, alpha=0.6)
ax.axvspan(1.05, 2.0, color=SHADE_SC_PLUS, alpha=0.35)
ax.plot(ratios_fplus, results_dep[:, 0], color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios_fplus, results_dep[:, 2], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios_fplus, results_dep[:, 3], color=COLOR_SC_PLUS, linestyle="-", linewidth=3.0)
ax.plot(ratios_fplus, results_dep[:, 5], color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.set_title("(c) Dependent Beneficial Plasmid Model", pad=10)
ax.set_xlabel(r"$f_+ / f_c$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)

# Panel (d): Plasmid-Chromosome Synergy
ax = axes[1, 1]
ax.axvspan(0.0, 0.65, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.65, 0.95, color=SHADE_COEX, alpha=0.6)
ax.axvspan(0.95, 2.0, color=SHADE_GC, alpha=0.35)
ax.plot(ratios_fpc, results_synergy[:, 0], color=COLOR_GC, linestyle="-", linewidth=3.0)
ax.plot(ratios_fpc, results_synergy[:, 1], color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios_fpc, results_synergy[:, 2], color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios_fpc, results_synergy[:, 3], color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.set_title(r"(d) Plasmid–Chromosome Synergy Model", pad=10)
ax.set_xlabel(r"$\Gamma = f_{pc} / (f_s + f_c)$")
ax.set_xlim(0.0, 2.0)

# Single shared legend placed ONCE at top of the entire figure!
shared_legend = [
    Line2D([0], [0], color=COLOR_GC, linestyle="-", linewidth=2.5, label="GC (Generous Cooperator)"),
    Line2D([0], [0], color=COLOR_GD, linestyle="-.", linewidth=2.0, label="GD (Generous Defector)"),
    Line2D([0], [0], color=COLOR_SC, linestyle=":", linewidth=2.0, label="SC (Secretive Cooperator)"),
    Line2D([0], [0], color=COLOR_SD, linestyle="-", linewidth=2.8, label="SD (Secretive Defector)"),
    Line2D([0], [0], color=COLOR_SC_PLUS, linestyle="-", linewidth=2.8, label=r"$SC_+$ (Beneficial Plasmid Cooperator)"),
    Line2D([0], [0], color=COLOR_SD_PLUS, linestyle="--", linewidth=2.0, label=r"$SD_+$ (Plasmid Defector)")
]
fig.legend(handles=shared_legend, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, framealpha=0.95, edgecolor="#cccccc")

plt.suptitle("Evolutionary Dynamics of Horizontal Gene Transfer: 4-Model Comparison", fontsize=15, y=1.02)
plt.subplots_adjust(top=0.88, hspace=0.30, wspace=0.18)
plt.savefig("figures/model_comparison_overview.png", bbox_inches="tight")
plt.close()

print("All 9 figures generated strictly following the no-text-in-data-area rule!")
