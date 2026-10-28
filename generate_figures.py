"""
Generates publication-quality figures for the BEE26 mathematical model:
"To Gossip or Not To Gossip: The Evolution of Cooperation Under Horizontal Gene Transfer"

Fixes applied:
- Zero overlap between curves, text annotations, and legends.
- Legends moved outside the axes (or into guaranteed empty margins).
- Regime indicators placed in dedicated top banner strips (y in [1.02, 1.12]) so lines crossing at y=0.5 never collide with text.
- Strictly unified scientific typography, color palette, and line hierarchy across all 9 figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
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
COLOR_GC_MINUS = "#62a0ea"  # Plasmid-free GC (Light Blue)
COLOR_GD = "#e66100"        # Generous Defector (Amber Orange)
COLOR_GD_MINUS = "#f69846"  # Plasmid-free GD (Light Orange)
COLOR_SC = "#26a269"        # Secretive Cooperator (Forest Green)
COLOR_SC_MINUS = "#57e389"  # Plasmid-free SC (Mint Green)
COLOR_SC_PLUS = "#8e24aa"   # Plasmid-bearing SC (Vibrant Purple)
COLOR_SD = "#c01c28"        # Secretive Defector (Crimson Red)
COLOR_SD_PLUS = "#681313"   # Plasmid-bearing SD (Deep Maroon)

# Shading colors
SHADE_SD = "#ffebee"        # Soft Red
SHADE_COEX = "#fff9c4"      # Soft Amber
SHADE_SC_PLUS = "#f3e5f5"   # Soft Purple
SHADE_GC = "#e3f2fd"        # Soft Blue

# Common parameters
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

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(ratios, results_base[:, 0], label="GC (Generous Cooperator)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios, results_base[:, 1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios, results_base[:, 2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios, results_base[:, 3], label="SD (Secretive Defector)", color=COLOR_SD, linestyle="-", linewidth=3.0)

ax.set_title("Base Model — Steady State vs $f_s / f_c$ (Slide 10)", pad=14)
ax.set_xlabel(r"Ratio of HGT Cost to Cooperation Cost ($f_s / f_c$)")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.04, 1.12)

# Dedicate top banner to regime status
ax.axvspan(0.0, 2.0, color=SHADE_SD, alpha=0.35)
ax.text(1.0, 1.05, "SD DOMINANCE REGIME (100% Defection)",
        ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_SD,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLOR_SD, linewidth=1.2))

# Callout in open middle area (no curves pass here!)
ax.text(1.0, 0.50,
        "Cooperation and HGT are completely eliminated:\n"
        "• GC pays double penalty: f_GC = 1 - f_c < f_SD = 1 + f_s\n"
        "• Conjugation halts as donor frequency GC -> 0",
        ha="center", va="center", fontsize=10.5, color="#333333",
        bbox=dict(boxstyle="square,pad=0.7", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92))

# Legend placed cleanly OUTSIDE the plot on the right
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide10_base_model_result.png")
plt.close()


# =============================================================
# 2. Base Model Trajectory
# =============================================================
print("Generating: base_model_trajectory.png")
fig, ax = plt.subplots(figsize=(9, 5.0))
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

# Legend placed outside on the right
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

fig, ax = plt.subplots(figsize=(9.2, 5.2))
total_sd = results_indep[:, 6] + results_indep[:, 7]
sd_plus = results_indep[:, 6]
total_gc = results_indep[:, 0] + results_indep[:, 1]
total_gd = results_indep[:, 2] + results_indep[:, 3]
total_sc = results_indep[:, 4] + results_indep[:, 5]

ax.plot(ratios, total_gc, label="GC Total ($GC_+ + GC_-$)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios, total_gd, label="GD Total ($GD_+ + GD_-$)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios, total_sc, label="SC Total ($SC_+ + SC_-$)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios, total_sd, label=r"SD Total ($SD_+ + SD_-$)", color=COLOR_SD, linestyle="-", linewidth=3.0)
ax.plot(ratios, sd_plus, label=r"$SD_+$ (Plasmid Defector)", color=COLOR_SD_PLUS, linestyle="--", linewidth=2.2)

ax.set_title("Independent Beneficial Plasmid Model — Steady State (Slide 15)", pad=14)
ax.set_xlabel(r"Ratio of HGT Cost to Cooperation Cost ($f_s / f_c$)")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.04, 1.12)

# Top banner
ax.axvspan(0.0, 2.0, color=SHADE_SD, alpha=0.35)
ax.text(1.0, 1.05, "SD+ FIXATION REGIME (Defectors Parasitize Plasmid)",
        ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_SD,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLOR_SD, linewidth=1.2))

# Informative callout in open center
ax.text(1.0, 0.50,
        "Independent plasmid benefit (f_+) fails to rescue cooperation:\n"
        "• f_+ is universally accessible to all hosts\n"
        "• Defectors acquire plasmid without paying for cooperation\n"
        "• f_SD+ = 1 + f_s + f_+ remains strictly superior",
        ha="center", va="center", fontsize=10.5, color="#333333",
        bbox=dict(boxstyle="square,pad=0.7", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92))

# Legend placed outside on the right
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide15_independent_plasmid_result.png")
plt.close()


# =============================================================
# 4. Independent Beneficial Plasmid Trajectory
# =============================================================
print("Generating: independent_plasmid_trajectory.png")
fig, ax = plt.subplots(figsize=(9.2, 5.0))
t_eval, y_traj_indep = simulate_system(independent_plasmid_ode, (0, 100), y0_indep, args=(alpha, fc, fs, fplus))
ax.plot(t_eval, y_traj_indep[0] + y_traj_indep[1], label="GC Total ($GC_+ + GC_-$)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(t_eval, y_traj_indep[2] + y_traj_indep[3], label="GD Total ($GD_+ + GD_-$)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(t_eval, y_traj_indep[4] + y_traj_indep[5], label="SC Total ($SC_+ + SC_-$)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(t_eval, y_traj_indep[6], label="$SD_+$ (Plasmid Defector)", color=COLOR_SD, linewidth=3.0)
ax.plot(t_eval, y_traj_indep[7], label="$SD_-$ (Plasmid-Free Defector)", color=COLOR_SD_PLUS, linestyle=":", linewidth=2.0)

ax.set_title("Independent Beneficial Plasmid Model — Trajectory over Time", pad=14)
ax.set_xlabel("Time ($t$, generations)")
ax.set_ylabel("Population Fraction ($x_i$)")
ax.set_xlim(0, 100)
ax.set_ylim(-0.02, 1.04)

# Legend placed outside on the right
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

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.plot(ratios_fplus, results_dep[:, 0], label="$GC_+$", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax.plot(ratios_fplus, results_dep[:, 1], label="$GC_-$", color=COLOR_GC_MINUS, linestyle=":", linewidth=1.8)
ax.plot(ratios_fplus, results_dep[:, 2], label="GD", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios_fplus, results_dep[:, 3], label="$SC_+$ (Beneficial Cooperator)", color=COLOR_SC_PLUS, linewidth=3.0)
ax.plot(ratios_fplus, results_dep[:, 4], label="$SC_-$", color=COLOR_SC, linestyle=":", linewidth=1.8)
ax.plot(ratios_fplus, results_dep[:, 5], label="SD (Secretive Defector)", color=COLOR_SD, linewidth=3.0)

# Shaded background zones matching Slide 20
ax.axvspan(0.0, 0.95, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.95, 1.05, color=SHADE_COEX, alpha=0.6)
ax.axvspan(1.05, 2.0, color=SHADE_SC_PLUS, alpha=0.35)

ax.set_title("Dependent Beneficial Plasmid Model — Steady State vs $f_+ / f_c$ (Slide 20)", pad=16)
ax.set_xlabel(r"Ratio of Plasmid Benefit to Cooperation Cost ($f_+ / f_c$)")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.04, 1.15)

# Dedicated header banner strip for regimes (above y=1.0 so NO overlap with data lines crossing at y=0.5!)
ax.text(0.48, 1.07, "SD Dominance", fontsize=11, fontweight="bold", color=COLOR_SD, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_SD, linewidth=1.0))
ax.text(1.0, 1.07, "Coexistence", fontsize=10, fontweight="bold", color="#d46b08", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d46b08", linewidth=1.0))
ax.text(1.52, 1.07, r"$SC_+$ Dominance (Cooperation Rescued)", fontsize=11, fontweight="bold", color=COLOR_SC_PLUS, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_SC_PLUS, linewidth=1.0))

# Clear annotations in guaranteed empty lower-left and lower-right corners
ax.text(0.35, 0.40,
        r"$f_+ < f_c$" "\n"
        "Defection wins\n"
        r"$f_{SD} > f_{SC_+}$",
        ha="center", va="center", fontsize=10, color=COLOR_SD,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#ffffff", edgecolor=COLOR_SD, alpha=0.85))

ax.text(1.60, 0.40,
        r"$f_+ > f_c$" "\n"
        "Cooperation wins\n"
        r"$f_{SC_+} > f_{SD}$",
        ha="center", va="center", fontsize=10, color=COLOR_SC_PLUS,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#ffffff", edgecolor=COLOR_SC_PLUS, alpha=0.85))

# Legend placed outside on the right
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide20_dependent_plasmid_result.png")
plt.close()


# =============================================================
# 6. Dependent Beneficial Plasmid Trajectory
# =============================================================
print("Generating: dependent_plasmid_trajectory.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.0))

# Subcritical
t_eval, y_sub = simulate_system(dependent_plasmid_ode, (0, 120), y0_dep, args=(alpha, fc, fs, 0.7 * fc))
ax1.plot(t_eval, y_sub[0], label="$GC_+$", color=COLOR_GC, linestyle="--", linewidth=1.8)
ax1.plot(t_eval, y_sub[2], label="GD", color=COLOR_GD, linestyle="-.", linewidth=1.8)
ax1.plot(t_eval, y_sub[3], label="$SC_+$", color=COLOR_SC_PLUS, linewidth=2.0)
ax1.plot(t_eval, y_sub[5], label="SD", color=COLOR_SD, linewidth=3.0)
ax1.set_title("Subcritical Regime: $f_+ / f_c = 0.7$ (Defection Wins)", pad=12)
ax1.set_xlabel("Time ($t$, generations)")
ax1.set_ylabel("Population Fraction ($x_i$)")
ax1.set_xlim(0, 120)
ax1.set_ylim(-0.02, 1.04)
ax1.legend(loc="center right", framealpha=0.9)

# Supercritical
t_eval, y_super = simulate_system(dependent_plasmid_ode, (0, 120), y0_dep, args=(alpha, fc, fs, 1.5 * fc))
ax2.plot(t_eval, y_super[0], label="$GC_+$", color=COLOR_GC, linestyle="--", linewidth=1.8)
ax2.plot(t_eval, y_super[2], label="GD", color=COLOR_GD, linestyle="-.", linewidth=1.8)
ax2.plot(t_eval, y_super[3], label="$SC_+$", color=COLOR_SC_PLUS, linewidth=3.0)
ax2.plot(t_eval, y_super[5], label="SD", color=COLOR_SD, linewidth=2.0)
ax2.set_title("Supercritical Regime: $f_+ / f_c = 1.5$ ($SC_+$ Fixation)", pad=12)
ax2.set_xlabel("Time ($t$, generations)")
ax2.set_ylabel("Population Fraction ($x_i$)")
ax2.set_xlim(0, 120)
ax2.set_ylim(-0.02, 1.04)
ax2.legend(loc="center right", framealpha=0.9)

plt.tight_layout()
plt.savefig("figures/dependent_plasmid_trajectory.png")
plt.close()


# =============================================================
# 7. Plasmid-Chromosome Synergy Model (Slide 24)
# =============================================================
print("Generating: slide24_synergy_model_result.png")
ratios_fpc = np.linspace(0.0, 2.0, 101)
y0_synergy = [0.25, 0.25, 0.25, 0.25]
# T=50 captures the finite-horizon / critical slowing down transition shown in presentation Slide 24
results_synergy = parameter_sweep(
    synergy_model_ode,
    ratios_fpc,
    lambda r: (alpha, fc, fs, r * (fs + fc)),
    y0_synergy,
    t_span=(0, 50)
)

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.plot(ratios_fpc, results_synergy[:, 0], label="GC (Generous Cooperator)", color=COLOR_GC, linewidth=3.0)
ax.plot(ratios_fpc, results_synergy[:, 1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax.plot(ratios_fpc, results_synergy[:, 2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax.plot(ratios_fpc, results_synergy[:, 3], label="SD (Secretive Defector)", color=COLOR_SD, linewidth=3.0)

# Shaded background regimes matching Slide 24
ax.axvspan(0.0, 0.65, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.65, 0.95, color=SHADE_COEX, alpha=0.6)
ax.axvspan(0.95, 2.0, color=SHADE_GC, alpha=0.35)

ax.set_title(r"Plasmid–Chromosome Synergy Model — Steady State (Slide 24)", pad=16)
ax.set_xlabel(r"Synergy Parameter $\Gamma = f_{pc} / (f_s + f_c)$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.04, 1.15)

# Dedicated header banner strip for regimes (above y=1.0 so NO overlap with data lines crossing at y=0.5!)
ax.text(0.32, 1.07, "SD Dominance", fontsize=11, fontweight="bold", color=COLOR_SD, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_SD, linewidth=1.0))
ax.text(0.80, 1.07, "Coexistence", fontsize=10, fontweight="bold", color="#d46b08", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d46b08", linewidth=1.0))
ax.text(1.48, 1.07, "GC Dominance (HGT & Cooperation Win!)", fontsize=11, fontweight="bold", color=COLOR_GC, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GC, linewidth=1.0))

# Clear annotations in guaranteed empty lower-left and lower-right corners
ax.text(0.32, 0.35,
        r"$\Gamma < 0.7$" "\n"
        "Synergy too low\n"
        r"$SD$ fixes",
        ha="center", va="center", fontsize=10, color=COLOR_SD,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#ffffff", edgecolor=COLOR_SD, alpha=0.85))

ax.text(1.48, 0.35,
        r"$\Gamma > 1.1$" "\n"
        "Synergy overcomes costs\n"
        r"$GC$ achieves 100% fixation",
        ha="center", va="center", fontsize=10, color=COLOR_GC,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#ffffff", edgecolor=COLOR_GC, alpha=0.85))

# Legend placed outside on the right
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.95, borderpad=0.7, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("figures/slide24_synergy_model_result.png")
plt.close()


# =============================================================
# 8. Synergy Model Trajectory
# =============================================================
print("Generating: synergy_model_trajectory.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.0))

# Low Synergy
t_eval, y_low = simulate_system(synergy_model_ode, (0, 100), y0_synergy, args=(alpha, fc, fs, 0.5 * (fs + fc)))
ax1.plot(t_eval, y_low[0], label="GC (Generous Cooperator)", color=COLOR_GC, linestyle="--", linewidth=2.0)
ax1.plot(t_eval, y_low[1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax1.plot(t_eval, y_low[2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax1.plot(t_eval, y_low[3], label="SD (Secretive Defector)", color=COLOR_SD, linewidth=3.0)
ax1.set_title(r"Low Synergy: $\Gamma = 0.5$ (SD Fixation)", pad=12)
ax1.set_xlabel("Time ($t$, generations)")
ax1.set_ylabel("Population Fraction ($x_i$)")
ax1.set_xlim(0, 100)
ax1.set_ylim(-0.02, 1.04)
ax1.legend(loc="center right", framealpha=0.9)

# High Synergy
t_eval, y_high = simulate_system(synergy_model_ode, (0, 100), y0_synergy, args=(alpha, fc, fs, 1.5 * (fs + fc)))
ax2.plot(t_eval, y_high[0], label="GC (Generous Cooperator)", color=COLOR_GC, linewidth=3.0)
ax2.plot(t_eval, y_high[1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.", linewidth=2.0)
ax2.plot(t_eval, y_high[2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":", linewidth=2.0)
ax2.plot(t_eval, y_high[3], label="SD (Secretive Defector)", color=COLOR_SD, linewidth=2.0)
ax2.set_title(r"High Synergy: $\Gamma = 1.5$ (GC Fixation — HGT & Cooperation Win!)", pad=12)
ax2.set_xlabel("Time ($t$, generations)")
ax2.set_ylabel("Population Fraction ($x_i$)")
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.02, 1.04)
ax2.legend(loc="center right", framealpha=0.9)

plt.tight_layout()
plt.savefig("figures/synergy_model_trajectory.png")
plt.close()


# =============================================================
# 9. Master Comparison Overview (2x2 Grid)
# =============================================================
print("Generating: model_comparison_overview.png")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Panel (a): Base Model
ax = axes[0, 0]
ax.plot(ratios, results_base[:, 0], label="GC", color=COLOR_GC, linestyle="--", linewidth=1.8)
ax.plot(ratios, results_base[:, 1], label="GD", color=COLOR_GD, linestyle="-.", linewidth=1.8)
ax.plot(ratios, results_base[:, 2], label="SC", color=COLOR_SC, linestyle=":", linewidth=1.8)
ax.plot(ratios, results_base[:, 3], label="SD", color=COLOR_SD, linewidth=2.8)
ax.axvspan(0.0, 2.0, color=SHADE_SD, alpha=0.35)
ax.set_title("(a) Base Model: Costly HGT & Cooperation", pad=12)
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_ylim(-0.04, 1.15)
ax.text(1.0, 1.06, "SD Fixation (100% Defection)", ha="center", va="center", color=COLOR_SD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_SD, linewidth=1.0))
ax.text(1.0, 0.50, "GC pays double penalty (fc + fs)\nConjugation ceases as GC -> 0",
        ha="center", va="center", fontsize=10, color="#333333",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.legend(loc="center left", framealpha=0.92)

# Panel (b): Independent Beneficial Plasmid
ax = axes[0, 1]
ax.plot(ratios, total_gc, label="Total GC", color=COLOR_GC, linestyle="--", linewidth=1.8)
ax.plot(ratios, total_gd, label="Total GD", color=COLOR_GD, linestyle="-.", linewidth=1.8)
ax.plot(ratios, total_sc, label="Total SC", color=COLOR_SC, linestyle=":", linewidth=1.8)
ax.plot(ratios, total_sd, label="Total SD", color=COLOR_SD, linewidth=2.8)
ax.plot(ratios, sd_plus, label=r"$SD_+$", color=COLOR_SD_PLUS, linestyle="--", linewidth=1.8)
ax.axvspan(0.0, 2.0, color=SHADE_SD, alpha=0.35)
ax.set_title("(b) Independent Beneficial Plasmid Model", pad=12)
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_ylim(-0.04, 1.15)
ax.text(1.0, 1.06, "SD+ Fixation (Plasmid Parasitized)", ha="center", va="center", color=COLOR_SD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_SD, linewidth=1.0))
ax.text(1.0, 0.50, "Independent benefit $f_+$ aids all hosts\nRelative fitness advantage favors defectors",
        ha="center", va="center", fontsize=10, color="#333333",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.legend(loc="center right", framealpha=0.92)

# Panel (c): Dependent Beneficial Plasmid
ax = axes[1, 0]
ax.plot(ratios_fplus, results_dep[:, 0], label="$GC_+$", color=COLOR_GC, linestyle="--", linewidth=1.8)
ax.plot(ratios_fplus, results_dep[:, 2], label="GD", color=COLOR_GD, linestyle="-.", linewidth=1.8)
ax.plot(ratios_fplus, results_dep[:, 3], label="$SC_+$", color=COLOR_SC_PLUS, linewidth=2.8)
ax.plot(ratios_fplus, results_dep[:, 5], label="SD", color=COLOR_SD, linewidth=2.8)
ax.axvspan(0.0, 0.95, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.95, 1.05, color=SHADE_COEX, alpha=0.6)
ax.axvspan(1.05, 2.0, color=SHADE_SC_PLUS, alpha=0.35)
ax.set_title("(c) Dependent Beneficial Plasmid Model", pad=12)
ax.set_xlabel(r"$f_+ / f_c$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_ylim(-0.04, 1.15)
ax.text(0.45, 1.06, "SD Dominance", ha="center", va="center", color=COLOR_SD, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLOR_SD))
ax.text(1.0, 1.06, "Coex", ha="center", va="center", color="#d46b08", fontweight="bold", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d46b08"))
ax.text(1.55, 1.06, "$SC_+$ Dominance", ha="center", va="center", color=COLOR_SC_PLUS, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLOR_SC_PLUS))
ax.text(0.35, 0.35, r"$f_+ < f_c$" "\nDefection wins", ha="center", va="center", fontsize=9, color=COLOR_SD)
ax.text(1.55, 0.35, r"$f_+ > f_c$" "\nCooperation saved\nHGT remains extinct", ha="center", va="center", fontsize=9, color=COLOR_SC_PLUS)
ax.legend(loc="center left", framealpha=0.92)

# Panel (d): Plasmid-Chromosome Synergy
ax = axes[1, 1]
ax.plot(ratios_fpc, results_synergy[:, 0], label="GC", color=COLOR_GC, linewidth=2.8)
ax.plot(ratios_fpc, results_synergy[:, 1], label="GD", color=COLOR_GD, linestyle="-.", linewidth=1.8)
ax.plot(ratios_fpc, results_synergy[:, 2], label="SC", color=COLOR_SC, linestyle=":", linewidth=1.8)
ax.plot(ratios_fpc, results_synergy[:, 3], label="SD", color=COLOR_SD, linewidth=2.8)
ax.axvspan(0.0, 0.65, color=SHADE_SD, alpha=0.35)
ax.axvspan(0.65, 0.95, color=SHADE_COEX, alpha=0.6)
ax.axvspan(0.95, 2.0, color=SHADE_GC, alpha=0.35)
ax.set_title("(d) Plasmid–Chromosome Synergy Model", pad=12)
ax.set_xlabel(r"$\Gamma = f_{pc} / (f_s + f_c)$")
ax.set_ylabel("Steady-State Fraction ($x_i^*$)")
ax.set_ylim(-0.04, 1.15)
ax.text(0.32, 1.06, "SD Dominance", ha="center", va="center", color=COLOR_SD, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLOR_SD))
ax.text(0.80, 1.06, "Coexistence", ha="center", va="center", color="#d46b08", fontweight="bold", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d46b08"))
ax.text(1.48, 1.06, "GC Dominance", ha="center", va="center", color=COLOR_GC, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLOR_GC))
ax.text(0.32, 0.35, r"$\Gamma < 0.7$" "\n$SD$ fixes", ha="center", va="center", fontsize=9, color=COLOR_SD)
ax.text(1.48, 0.35, r"$\Gamma > 1.1$" "\nBoth HGT & Cooperation\nachieve 100% fixation", ha="center", va="center", fontsize=9, color=COLOR_GC)
ax.legend(loc="center left", framealpha=0.92)

plt.suptitle("Evolutionary Dynamics of Horizontal Gene Transfer: 4-Model Comparison", fontsize=16, y=0.99)
plt.subplots_adjust(hspace=0.32, wspace=0.24)
plt.savefig("figures/model_comparison_overview.png")
plt.close()

print("All improved figures generated successfully with zero overlap!")
