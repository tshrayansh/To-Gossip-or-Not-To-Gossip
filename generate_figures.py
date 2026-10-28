"""
Generates publication-quality figures for the BEE26 mathematical model:
"To Gossip or Not To Gossip: The Evolution of Cooperation Under Horizontal Gene Transfer"
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

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Set clean scientific plotting style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "lines.linewidth": 2.5
})

# Standard color palette matching slides
COLOR_GC = "#1f77b4"  # Blue
COLOR_GD = "#ff7f0e"  # Orange
COLOR_SC = "#2ca02c"  # Green
COLOR_SD = "#d62728"  # Red
COLOR_SC_PLUS = "#9467bd" # Purple
COLOR_SD_PLUS = "#8c564b" # Brown / Dark Red

print("Generating Figure 1: Base Model Result (Slide 10)...")
# -------------------------------------------------------------
# 1. Base Model Bifurcation (Slide 10)
# -------------------------------------------------------------
fc = 0.2
alpha = 0.1
ratios = np.linspace(0.0, 2.0, 101)
y0_base = [0.25, 0.25, 0.25, 0.25]

results_base = parameter_sweep(
    base_model_ode,
    ratios,
    lambda r: (alpha, fc, r * fc),
    y0_base,
    t_span=(0, 250)
)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(ratios, results_base[:, 0], label="GC", color=COLOR_GC, linestyle="--")
ax.plot(ratios, results_base[:, 1], label="GD", color=COLOR_GD, linestyle="-.")
ax.plot(ratios, results_base[:, 2], label="SC", color=COLOR_SC, linestyle=":")
ax.plot(ratios, results_base[:, 3], label="SD", color=COLOR_SD, linestyle="-", linewidth=3)

ax.set_title("Base Model — Steady State vs $f_s / f_c$ (Slide 10)", fontweight="bold", pad=12)
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-state fraction")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)
ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="center left", framealpha=0.9)

# Highlight SD dominance
ax.text(1.0, 0.5, "SD Dominance\n(Cooperation & HGT Extinct)",
        ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_SD,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor=COLOR_SD, alpha=0.8))

plt.tight_layout()
plt.savefig("figures/slide10_base_model_result.png")
plt.close()

# Trajectory for Base Model
fig, ax = plt.subplots(figsize=(8, 5))
t_eval, y_traj = simulate_system(base_model_ode, (0, 100), y0_base, args=(alpha, fc, 0.1))
ax.plot(t_eval, y_traj[0], label="GC (Generous Cooperator)", color=COLOR_GC)
ax.plot(t_eval, y_traj[1], label="GD (Generous Defector)", color=COLOR_GD)
ax.plot(t_eval, y_traj[2], label="SC (Secretive Cooperator)", color=COLOR_SC)
ax.plot(t_eval, y_traj[3], label="SD (Secretive Defector)", color=COLOR_SD, linewidth=3)
ax.set_title("Base Model — Population Trajectory over Time ($f_s=0.1, f_c=0.2, \\alpha=0.1$)", fontweight="bold")
ax.set_xlabel("Time ($t$)")
ax.set_ylabel("Population fraction")
ax.set_xlim(0, 100)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig("figures/base_model_trajectory.png")
plt.close()


print("Generating Figure 2: Independent Beneficial Plasmid Model (Slide 15)...")
# -------------------------------------------------------------
# 2. Independent Beneficial Plasmid Model (Slide 15)
# -------------------------------------------------------------
fplus = 0.3
y0_indep = [1/8.0] * 8

results_indep = parameter_sweep(
    independent_plasmid_ode,
    ratios,
    lambda r: (alpha, fc, r * fc, fplus),
    y0_indep,
    t_span=(0, 250)
)

fig, ax = plt.subplots(figsize=(8, 5.5))
# Combine plasmid-bearing and plasmid-free secretive defectors
sd_total = results_indep[:, 6] + results_indep[:, 7]
sd_plus = results_indep[:, 6]
sd_minus = results_indep[:, 7]
gc_total = results_indep[:, 0] + results_indep[:, 1]
gd_total = results_indep[:, 2] + results_indep[:, 3]
sc_total = results_indep[:, 4] + results_indep[:, 5]

ax.plot(ratios, gc_total, label="GC ($GC_+ + GC_-$)", color=COLOR_GC, linestyle="--")
ax.plot(ratios, gd_total, label="GD ($GD_+ + GD_-$)", color=COLOR_GD, linestyle="-.")
ax.plot(ratios, sc_total, label="SC ($SC_+ + SC_-$)", color=COLOR_SC, linestyle=":")
ax.plot(ratios, sd_total, label=r"$SD_+ + SD_-$ (Total SD)", color=COLOR_SD, linewidth=3)
ax.plot(ratios, sd_plus, label=r"$SD_+$ (Plasmid-bearing SD)", color=COLOR_SD_PLUS, linestyle="-", linewidth=2)

ax.set_title("Independent Beneficial Plasmid Model — Steady State (Slide 15)", fontweight="bold", pad=12)
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-state fraction")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="center right", framealpha=0.9)

ax.text(0.6, 0.45, "SD Dominance ($SD_+$ Wins)\nIndependent plasmid benefit\nfails to promote cooperation",
        ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_SD,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor=COLOR_SD, alpha=0.8))

plt.tight_layout()
plt.savefig("figures/slide15_independent_plasmid_result.png")
plt.close()

# Trajectory for Independent Plasmid
fig, ax = plt.subplots(figsize=(8, 5))
t_eval, y_traj_indep = simulate_system(independent_plasmid_ode, (0, 100), y0_indep, args=(alpha, fc, 0.1, fplus))
ax.plot(t_eval, y_traj_indep[0] + y_traj_indep[1], label="GC (Generous Cooperators)", color=COLOR_GC)
ax.plot(t_eval, y_traj_indep[2] + y_traj_indep[3], label="GD (Generous Defectors)", color=COLOR_GD)
ax.plot(t_eval, y_traj_indep[4] + y_traj_indep[5], label="SC (Secretive Cooperators)", color=COLOR_SC)
ax.plot(t_eval, y_traj_indep[6], label="$SD_+$ (Plasmid-bearing Defector)", color=COLOR_SD, linewidth=2.5)
ax.plot(t_eval, y_traj_indep[7], label="$SD_-$ (Plasmid-free Defector)", color=COLOR_SD_PLUS, linestyle="--")
ax.set_title("Independent Beneficial Plasmid Model — Trajectory over Time", fontweight="bold")
ax.set_xlabel("Time ($t$)")
ax.set_ylabel("Population fraction")
ax.set_xlim(0, 100)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig("figures/independent_plasmid_trajectory.png")
plt.close()


print("Generating Figure 3: Dependent Beneficial Plasmid Model (Slide 20)...")
# -------------------------------------------------------------
# 3. Dependent Beneficial Plasmid Model (Slide 20)
# -------------------------------------------------------------
fs = 0.1
ratios_fplus = np.linspace(0.0, 2.0, 101)
y0_dep = [1/6.0] * 6

results_dep = parameter_sweep(
    dependent_plasmid_ode,
    ratios_fplus,
    lambda r: (alpha, fc, fs, r * fc),
    y0_dep,
    t_span=(0, 150)
)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(ratios_fplus, results_dep[:, 0], label="$GC_+$", color=COLOR_GC, linestyle="--")
ax.plot(ratios_fplus, results_dep[:, 1], label="$GC_-$", color=COLOR_GC, linestyle=":")
ax.plot(ratios_fplus, results_dep[:, 2], label="GD", color=COLOR_GD, linestyle="-.")
ax.plot(ratios_fplus, results_dep[:, 3], label="$SC_+$", color=COLOR_SC_PLUS, linewidth=3)
ax.plot(ratios_fplus, results_dep[:, 4], label="$SC_-$", color=COLOR_SC, linestyle=":")
ax.plot(ratios_fplus, results_dep[:, 5], label="SD", color=COLOR_SD, linewidth=3)

# Shaded regions matching Slide 20
ax.axvspan(0.0, 0.95, color="#ffcdd2", alpha=0.25, label="SD Regime")
ax.axvspan(0.95, 1.05, color="#fff9c4", alpha=0.4, label="Coexistence")
ax.axvspan(1.05, 2.0, color="#e1bee7", alpha=0.25, label="$SC_+$ Regime")

ax.set_title("Dependent Beneficial Plasmid Model — Steady State vs $f_+ / f_c$ (Slide 20)", fontweight="bold", pad=12)
ax.set_xlabel(r"$f_+ / f_c$")
ax.set_ylabel("Steady-state fraction")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), framealpha=0.9)

ax.text(0.45, 0.75, "SD Dominance", fontsize=13, fontweight="bold", color=COLOR_SD, ha="center")
ax.text(1.0, 0.45, "Coexistence\nBoundary", fontsize=11, fontweight="bold", color="#f57f17", ha="center")
ax.text(1.55, 0.75, "$SC_+$ Dominance\n(Cooperation Rescued,\nHGT Suppressed)", fontsize=13, fontweight="bold", color=COLOR_SC_PLUS, ha="center")

plt.tight_layout()
plt.savefig("figures/slide20_dependent_plasmid_result.png")
plt.close()

# Trajectory for Dependent Plasmid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
t_eval, y_sub = simulate_system(dependent_plasmid_ode, (0, 120), y0_dep, args=(alpha, fc, fs, 0.7 * fc))
ax1.plot(t_eval, y_sub[0], label="$GC_+$", color=COLOR_GC, linestyle="--")
ax1.plot(t_eval, y_sub[2], label="GD", color=COLOR_GD, linestyle="-.")
ax1.plot(t_eval, y_sub[3], label="$SC_+$", color=COLOR_SC_PLUS)
ax1.plot(t_eval, y_sub[5], label="SD", color=COLOR_SD, linewidth=2.5)
ax1.set_title("Subcritical: $f_+ / f_c = 0.7$ (Defection Wins)")
ax1.set_xlabel("Time ($t$)")
ax1.set_ylabel("Population fraction")
ax1.set_xlim(0, 120)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend()

t_eval, y_super = simulate_system(dependent_plasmid_ode, (0, 120), y0_dep, args=(alpha, fc, fs, 1.5 * fc))
ax2.plot(t_eval, y_super[0], label="$GC_+$", color=COLOR_GC, linestyle="--")
ax2.plot(t_eval, y_super[2], label="GD", color=COLOR_GD, linestyle="-.")
ax2.plot(t_eval, y_super[3], label="$SC_+$", color=COLOR_SC_PLUS, linewidth=2.5)
ax2.plot(t_eval, y_super[5], label="SD", color=COLOR_SD)
ax2.set_title("Supercritical: $f_+ / f_c = 1.5$ ($SC_+$ Cooperation Wins)")
ax2.set_xlabel("Time ($t$)")
ax2.set_ylabel("Population fraction")
ax2.set_xlim(0, 120)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("figures/dependent_plasmid_trajectory.png")
plt.close()


print("Generating Figure 4: Plasmid-Chromosome Synergy Model (Slide 24)...")
# -------------------------------------------------------------
# 4. Plasmid-Chromosome Synergy Model (Slide 24)
# -------------------------------------------------------------
ratios_fpc = np.linspace(0.0, 2.0, 101)
y0_synergy = [0.25, 0.25, 0.25, 0.25]
# T=50 captures the finite-horizon / critical slowing down transition seen on Slide 24
results_synergy = parameter_sweep(
    synergy_model_ode,
    ratios_fpc,
    lambda r: (alpha, fc, fs, r * (fs + fc)),
    y0_synergy,
    t_span=(0, 50)
)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(ratios_fpc, results_synergy[:, 0], label="GC", color=COLOR_GC, linewidth=3)
ax.plot(ratios_fpc, results_synergy[:, 1], label="GD", color=COLOR_GD, linestyle="-.")
ax.plot(ratios_fpc, results_synergy[:, 2], label="SC", color=COLOR_SC, linestyle=":")
ax.plot(ratios_fpc, results_synergy[:, 3], label="SD", color=COLOR_SD, linewidth=3)

# Shading matching Slide 24
ax.axvspan(0.0, 0.65, color="#ffcdd2", alpha=0.25)
ax.axvspan(0.65, 0.95, color="#fff9c4", alpha=0.45)
ax.axvspan(0.95, 2.0, color="#bbdefb", alpha=0.25)

ax.set_title("Plasmid–Chromosome Synergy Model — Steady State (Slide 24)", fontweight="bold", pad=12)
ax.set_xlabel(r"$f_{pc} / (f_s + f_c)$")
ax.set_ylabel("Steady-state fraction")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), framealpha=0.9)

ax.text(0.3, 0.5, "SD", fontsize=15, fontweight="bold", color=COLOR_SD, ha="center")
ax.text(0.8, 0.5, "Coexistence", fontsize=14, fontweight="bold", color="#f57f17", ha="center")
ax.text(1.5, 0.5, "GC", fontsize=15, fontweight="bold", color=COLOR_GC, ha="center")

plt.tight_layout()
plt.savefig("figures/slide24_synergy_model_result.png")
plt.close()

# Trajectory for Synergy Model
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
t_eval, y_low = simulate_system(synergy_model_ode, (0, 100), y0_synergy, args=(alpha, fc, fs, 0.5 * (fs + fc)))
ax1.plot(t_eval, y_low[0], label="GC (Generous Cooperator)", color=COLOR_GC)
ax1.plot(t_eval, y_low[1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.")
ax1.plot(t_eval, y_low[2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":")
ax1.plot(t_eval, y_low[3], label="SD (Secretive Defector)", color=COLOR_SD, linewidth=2.5)
ax1.set_title("Low Synergy: $f_{pc}/(f_s+f_c) = 0.5$ (SD Fixation)")
ax1.set_xlabel("Time ($t$)")
ax1.set_ylabel("Population fraction")
ax1.set_xlim(0, 100)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend()

t_eval, y_high = simulate_system(synergy_model_ode, (0, 100), y0_synergy, args=(alpha, fc, fs, 1.5 * (fs + fc)))
ax2.plot(t_eval, y_high[0], label="GC (Generous Cooperator)", color=COLOR_GC, linewidth=2.5)
ax2.plot(t_eval, y_high[1], label="GD (Generous Defector)", color=COLOR_GD, linestyle="-.")
ax2.plot(t_eval, y_high[2], label="SC (Secretive Cooperator)", color=COLOR_SC, linestyle=":")
ax2.plot(t_eval, y_high[3], label="SD (Secretive Defector)", color=COLOR_SD)
ax2.set_title("High Synergy: $f_{pc}/(f_s+f_c) = 1.5$ (GC Fixation — HGT & Cooperation Win!)")
ax2.set_xlabel("Time ($t$)")
ax2.set_ylabel("Population fraction")
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("figures/synergy_model_trajectory.png")
plt.close()


print("Generating Figure 5: Master Comparison Overview...")
# -------------------------------------------------------------
# 5. Composite Comparison Overview (All 4 Models)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Base Model
ax = axes[0, 0]
ax.plot(ratios, results_base[:, 0], label="GC", color=COLOR_GC, linestyle="--")
ax.plot(ratios, results_base[:, 1], label="GD", color=COLOR_GD, linestyle="-.")
ax.plot(ratios, results_base[:, 2], label="SC", color=COLOR_SC, linestyle=":")
ax.plot(ratios, results_base[:, 3], label="SD", color=COLOR_SD, linewidth=3)
ax.set_title("(a) Base Model: Costly HGT & Cooperation", fontweight="bold")
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-state fraction")
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.text(1.0, 0.5, "SD Wins (100% Defection)", ha="center", va="center", color=COLOR_SD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee", edgecolor=COLOR_SD, alpha=0.8))
ax.legend(loc="center left")

# Subplot 2: Independent Beneficial Plasmid
ax = axes[0, 1]
ax.plot(ratios, gc_total, label="GC ($GC_+ + GC_-$)", color=COLOR_GC, linestyle="--")
ax.plot(ratios, gd_total, label="GD ($GD_+ + GD_-$)", color=COLOR_GD, linestyle="-.")
ax.plot(ratios, sc_total, label="SC ($SC_+ + SC_-$)", color=COLOR_SC, linestyle=":")
ax.plot(ratios, sd_total, label=r"Total SD ($SD_+ + SD_-$)", color=COLOR_SD, linewidth=3)
ax.set_title("(b) Independent Beneficial Plasmid Model", fontweight="bold")
ax.set_xlabel(r"$f_s / f_c$")
ax.set_ylabel("Steady-state fraction")
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.text(1.0, 0.5, "SD Wins ($SD_+$ Domination)", ha="center", va="center", color=COLOR_SD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee", edgecolor=COLOR_SD, alpha=0.8))
ax.legend(loc="center right")

# Subplot 3: Dependent Beneficial Plasmid
ax = axes[1, 0]
ax.plot(ratios_fplus, results_dep[:, 0], label="$GC_+$", color=COLOR_GC, linestyle="--")
ax.plot(ratios_fplus, results_dep[:, 2], label="GD", color=COLOR_GD, linestyle="-.")
ax.plot(ratios_fplus, results_dep[:, 3], label="$SC_+$", color=COLOR_SC_PLUS, linewidth=3)
ax.plot(ratios_fplus, results_dep[:, 5], label="SD", color=COLOR_SD, linewidth=3)
ax.axvspan(0.0, 0.95, color="#ffcdd2", alpha=0.25)
ax.axvspan(0.95, 1.05, color="#fff9c4", alpha=0.4)
ax.axvspan(1.05, 2.0, color="#e1bee7", alpha=0.25)
ax.set_title("(c) Dependent Beneficial Plasmid Model", fontweight="bold")
ax.set_xlabel(r"$f_+ / f_c$")
ax.set_ylabel("Steady-state fraction")
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.text(0.4, 0.5, "SD", ha="center", va="center", color=COLOR_SD, fontweight="bold")
ax.text(1.0, 0.5, "Coex", ha="center", va="center", color="#f57f17", fontweight="bold")
ax.text(1.6, 0.5, "$SC_+$ Wins", ha="center", va="center", color=COLOR_SC_PLUS, fontweight="bold")
ax.legend(loc="center left")

# Subplot 4: Synergy Model
ax = axes[1, 1]
ax.plot(ratios_fpc, results_synergy[:, 0], label="GC", color=COLOR_GC, linewidth=3)
ax.plot(ratios_fpc, results_synergy[:, 1], label="GD", color=COLOR_GD, linestyle="-.")
ax.plot(ratios_fpc, results_synergy[:, 2], label="SC", color=COLOR_SC, linestyle=":")
ax.plot(ratios_fpc, results_synergy[:, 3], label="SD", color=COLOR_SD, linewidth=3)
ax.axvspan(0.0, 0.65, color="#ffcdd2", alpha=0.25)
ax.axvspan(0.65, 0.95, color="#fff9c4", alpha=0.45)
ax.axvspan(0.95, 2.0, color="#bbdefb", alpha=0.25)
ax.set_title("(d) Plasmid–Chromosome Synergy Model", fontweight="bold")
ax.set_xlabel(r"$f_{pc} / (f_s + f_c)$")
ax.set_ylabel("Steady-state fraction")
ax.set_ylim(-0.02, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.text(0.3, 0.5, "SD", ha="center", va="center", color=COLOR_SD, fontweight="bold")
ax.text(0.8, 0.5, "Coex", ha="center", va="center", color="#f57f17", fontweight="bold")
ax.text(1.5, 0.5, "GC Wins", ha="center", va="center", color=COLOR_GC, fontweight="bold")
ax.legend(loc="center left")

plt.suptitle("Evolutionary Dynamics of Horizontal Gene Transfer: 4-Model Comparison", fontweight="bold", fontsize=17, y=0.99)
plt.tight_layout()
plt.savefig("figures/model_comparison_overview.png")
plt.close()

print("All figures generated successfully in figures/")
