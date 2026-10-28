"""
Mathematical models and numerical solvers for:
"To Gossip or Not To Gossip: The Evolution of Cooperation Under Horizontal Gene Transfer"
BEE 2026, The Institute of Mathematical Sciences (IMSc)
Authors: Ananya, Shrayansh, Yogesh
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =====================================================================
# 1. Base Model (Slides 7-10)
# Strategies: GC, GD, SC, SD
# =====================================================================

def base_model_ode(t, y, alpha, fc, fs):
    """
    Replicator-conjugation dynamics for Base Model.
    y = [GC, GD, SC, SD]
    
    Fitness:
        f_GC = 1 - fc
        f_GD = 1
        f_SC = 1 + fs - fc
        f_SD = 1 + fs
    Mean fitness:
        f_bar = 1 - fc*(GC + SC) + fs*(SC + SD)
    Conjugation events:
        GC + SD -> GC + SC (rate alpha*GC*SD)
        GC + GD -> 2GC    (rate alpha*GC*GD)
    """
    GC, GD, SC, SD = y
    f_GC = 1.0 - fc
    f_GD = 1.0
    f_SC = 1.0 + fs - fc
    f_SD = 1.0 + fs
    
    f_bar = 1.0 - fc * (GC + SC) + fs * (SC + SD)
    
    dGC = (f_GC - f_bar) * GC + alpha * GC * GD
    dGD = (f_GD - f_bar) * GD - alpha * GC * GD
    dSC = (f_SC - f_bar) * SC + alpha * GC * SD
    dSD = (f_SD - f_bar) * SD - alpha * GC * SD
    
    return [dGC, dGD, dSC, dSD]


# =====================================================================
# 2. Independent Beneficial Plasmid Model (Slides 11-15)
# Strategies: GC+, GC-, GD+, GD-, SC+, SC-, SD+, SD- (8 types)
# =====================================================================

def independent_plasmid_ode(t, y, alpha, fc, fs, fplus):
    """
    Replicator-conjugation dynamics for Independent Beneficial Plasmid Model.
    y = [GC+, GC-, GD+, GD-, SC+, SC-, SD+, SD-]
    
    Every plasmid carrier (+) gets independent benefit fplus.
    """
    GC_p, GC_m, GD_p, GD_m, SC_p, SC_m, SD_p, SD_m = y
    
    f_GC_p = 1.0 - fc + fplus
    f_GC_m = 1.0 - fc
    f_GD_p = 1.0 + fplus
    f_GD_m = 1.0
    f_SC_p = 1.0 + fs - fc + fplus
    f_SC_m = 1.0 + fs - fc
    f_SD_p = 1.0 + fs + fplus
    f_SD_m = 1.0 + fs
    
    f_bar = (1.0 - fc * (GC_p + GC_m + SC_p + SC_m)
             + fs * (SC_p + SC_m + SD_p + SD_m)
             + fplus * (GC_p + GD_p + SC_p + SD_p))
    
    # Conjugation terms from Slide 14
    c_GC_p = alpha * (GC_p * GC_m + GC_p * GD_p + GC_p * GD_m + 2.0 * GC_m * GD_p)
    c_GC_m = alpha * (GC_m * GD_m - GC_p * GC_m - GC_m * GD_p)
    c_GD_p = alpha * (GD_p * GD_m - GC_p * GD_p - GC_m * GD_p)
    c_GD_m = alpha * (-GC_p * GD_m - GC_m * GD_m - GD_p * GD_m)
    c_SC_p = alpha * (GC_p * SC_m + GC_p * SD_p + GC_p * SD_m + GC_m * SD_p + GD_p * SC_m)
    c_SC_m = alpha * (GC_m * SD_m - GC_p * SC_m - GD_p * SC_m)
    c_SD_p = alpha * (GD_p * SD_m - GC_p * SD_p - GC_m * SD_p)
    c_SD_m = alpha * (-GD_p * SD_m - GC_p * SD_m - GC_m * SD_m)
    
    dGC_p = (f_GC_p - f_bar) * GC_p + c_GC_p
    dGC_m = (f_GC_m - f_bar) * GC_m + c_GC_m
    dGD_p = (f_GD_p - f_bar) * GD_p + c_GD_p
    dGD_m = (f_GD_m - f_bar) * GD_m + c_GD_m
    dSC_p = (f_SC_p - f_bar) * SC_p + c_SC_p
    dSC_m = (f_SC_m - f_bar) * SC_m + c_SC_m
    dSD_p = (f_SD_p - f_bar) * SD_p + c_SD_p
    dSD_m = (f_SD_m - f_bar) * SD_m + c_SD_m
    
    return [dGC_p, dGC_m, dGD_p, dGD_m, dSC_p, dSC_m, dSD_p, dSD_m]


# =====================================================================
# 3. Dependent Beneficial Plasmid Model (Slides 16-20)
# Strategies: GC+, GC-, GD, SC+, SC-, SD (6 types)
# =====================================================================

def dependent_plasmid_ode(t, y, alpha, fc, fs, fplus):
    """
    Replicator-conjugation dynamics for Dependent Beneficial Plasmid Model.
    y = [GC+, GC-, GD, SC+, SC-, SD]
    
    Plasmid benefit fplus is conditional: received only by cooperators (GC+, SC+).
    """
    GC_p, GC_m, GD, SC_p, SC_m, SD = y
    
    f_GC_p = 1.0 - fc + fplus
    f_GC_m = 1.0 - fc
    f_GD   = 1.0
    f_SC_p = 1.0 + fs - fc + fplus
    f_SC_m = 1.0 + fs - fc
    f_SD   = 1.0 + fs
    
    f_bar = (1.0 - fc * (GC_p + GC_m + SC_p + SC_m)
             + fs * (SC_p + SC_m + SD)
             + fplus * (GC_p + SC_p))
    
    c_GC_p = alpha * (GC_p * GD + GC_p * GC_m)
    c_GC_m = alpha * (GC_m * GD + GC_m * SD - GC_p * GC_m)
    c_GD   = -alpha * (GC_p * GD + GC_m * GD)
    c_SC_p = alpha * (GC_p * SD + SC_m * GC_p)
    c_SC_m = -alpha * (SC_m * GC_p)
    c_SD   = -alpha * (GC_p * SD + GC_m * SD)
    
    dGC_p = (f_GC_p - f_bar) * GC_p + c_GC_p
    dGC_m = (f_GC_m - f_bar) * GC_m + c_GC_m
    dGD   = (f_GD   - f_bar) * GD   + c_GD
    dSC_p = (f_SC_p - f_bar) * SC_p + c_SC_p
    dSC_m = (f_SC_m - f_bar) * SC_m + c_SC_m
    dSD   = (f_SD   - f_bar) * SD   + c_SD
    
    return [dGC_p, dGC_m, dGD, dSC_p, dSC_m, dSD]


# =====================================================================
# 4. Plasmid-Chromosome Synergy Model (Slides 21-24)
# Strategies: GC, GD, SC, SD (4 types)
# =====================================================================

def synergy_model_ode(t, y, alpha, fc, fs, fpc):
    """
    Replicator-conjugation dynamics for Plasmid-Chromosome Synergy Model.
    y = [GC, GD, SC, SD]
    
    Synergy benefit fpc is assigned to Generous Cooperators (GC),
    which possess both chromosomal HGT machinery and the cooperative plasmid.
    """
    GC, GD, SC, SD = y
    
    f_GC = 1.0 - fc + fpc
    f_GD = 1.0
    f_SC = 1.0 + fs - fc
    f_SD = 1.0 + fs
    
    f_bar = 1.0 - fc * (GC + SC) + fs * (SC + SD) + fpc * GC
    
    dGC = (f_GC - f_bar) * GC + alpha * GC * GD
    dGD = (f_GD - f_bar) * GD - alpha * GC * GD
    dSC = (f_SC - f_bar) * SC + alpha * GC * SD
    dSD = (f_SD - f_bar) * SD - alpha * GC * SD
    
    return [dGC, dGD, dSC, dSD]


# =====================================================================
# Solvers and Simulation Helpers
# =====================================================================

def simulate_system(ode_func, t_span, y0, args, t_eval=None, method='RK45'):
    """Integrates an ODE system and normalizes population fractions."""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(ode_func, t_span, y0, args=args, t_eval=t_eval, method=method, atol=1e-9, rtol=1e-8)
    # Ensure positive frequencies summing to 1
    y_norm = np.maximum(0.0, sol.y)
    y_norm /= np.sum(y_norm, axis=0, keepdims=True)
    return sol.t, y_norm


def parameter_sweep(ode_func, sweep_vals, param_setter, y0, t_span=(0, 200), method='RK45'):
    """
    Sweeps a parameter value and returns steady-state fractions.
    param_setter is a function: sweep_val -> tuple of args for ode_func
    """
    num_types = len(y0)
    results = np.zeros((len(sweep_vals), num_types))
    
    for idx, val in enumerate(sweep_vals):
        args = param_setter(val)
        sol = solve_ivp(ode_func, t_span, y0, args=args, method=method, atol=1e-8, rtol=1e-7)
        final_state = np.maximum(0.0, sol.y[:, -1])
        final_state /= np.sum(final_state)
        results[idx] = final_state
        
    return results

