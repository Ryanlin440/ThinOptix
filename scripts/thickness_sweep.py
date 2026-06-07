#!/usr/bin/env python3
"""
thickness_sweep.py

Sweeps each layer thickness one at a time (fixing the other two) and records
how T/F/G color (CIE Lab) and Tvis change.

Edit the MATERIALS and REFERENCE THICKNESSES sections below before running.
Output goes to:
  output/csv/sweep_<layer>.csv
  output/png/sweep_<layer>.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optical_coating_tool import (
    load_nk, interp_nk, calculate_tfg,
    cie_lab, make_tvis_weights, tvis_fast,
    _find, WL_START, WL_END, WL_STEP,
)

# ── EDIT THESE ────────────────────────────────────────────────────────────────
# Material CSV filenames — must exist in data/materials/ or the working folder.
GLASS_FILE = "glass-industr-2018-mat.csv"
L1_FILE    = "L1-Al-mat.csv"     # <-- change to your Al or SiAl filename
L2_FILE    = "L2-Ag-mat.csv"     # Ag (already in data/materials/)
L3_FILE    = "L3-Al-mat.csv"     # <-- change to your Al or SiAl filename

GLASS_MM = 3.0   # glass substrate thickness in mm

# Reference (fixed) thicknesses used when a layer is NOT being swept (nm)
D1_REF = 40.0   # L1 reference thickness
D2_REF = 15.0   # L2 (Ag) reference thickness
D3_REF = 40.0   # L3 reference thickness

# Sweep range for each layer (nm)
SWEEP = np.arange(5, 101, 5)    # 5, 10, 15, … 100 nm
# ── END EDIT ──────────────────────────────────────────────────────────────────

# Color-spec limits (from lowe_design.py) — drawn as dashed lines on plots
LIMITS = {
    "T_a": (-3.0,  0.0),
    "T_b": (-3.0,  3.0),
    "G_a": (-5.0,  0.0),
    "G_b": (-15.0, -5.0),
    "F_a": (-4.0,  4.0),
    "F_b": (-15.0, -4.0),
}

_OUTDIR_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "csv")
_OUTDIR_PNG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "png")
os.makedirs(_OUTDIR_CSV, exist_ok=True)
os.makedirs(_OUTDIR_PNG, exist_ok=True)


def load_materials():
    wl = np.arange(WL_START, WL_END + 0.1, WL_STEP)
    glass_nk = interp_nk(load_nk(_find(GLASS_FILE)), wl)
    L1_nk    = interp_nk(load_nk(_find(L1_FILE)),    wl)
    L2_nk    = interp_nk(load_nk(_find(L2_FILE)),    wl)
    L3_nk    = interp_nk(load_nk(_find(L3_FILE)),    wl)
    tvis_w   = make_tvis_weights(wl)
    return wl, glass_nk, L1_nk, L2_nk, L3_nk, tvis_w


def run_sweep(layer: str, wl, glass_nk, L1_nk, L2_nk, L3_nk, tvis_w) -> pd.DataFrame:
    rows = []
    for val in SWEEP:
        d1 = val if layer == "d1" else D1_REF
        d2 = val if layer == "d2" else D2_REF
        d3 = val if layer == "d3" else D3_REF

        df = calculate_tfg(wl, glass_nk, L1_nk, L2_nk, L3_nk, d1, d2, d3, GLASS_MM)

        TY, _, Ta, Tb = cie_lab(wl, df["T_percent"].values)
        FY, _, Fa, Fb = cie_lab(wl, df["F_percent"].values)
        GY, _, Ga, Gb = cie_lab(wl, df["G_percent"].values)
        tvis = tvis_fast(df["T_percent"].values, tvis_w)

        rows.append(dict(
            thickness_nm=val,
            d1=d1, d2=d2, d3=d3,
            Tvis=tvis,
            T_Y=TY, T_a=Ta, T_b=Tb,
            F_Y=FY, F_a=Fa, F_b=Fb,
            G_Y=GY, G_a=Ga, G_b=Gb,
        ))
        print(f"  {layer}={val:5.0f} nm  Tvis={tvis:5.2f}%  T_a*={Ta:6.2f}  T_b*={Tb:6.2f}")

    return pd.DataFrame(rows)


def plot_sweep(results: pd.DataFrame, layer: str, label: str):
    x = results["thickness_nm"].values
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"Thickness sweep — {layer} ({label})\n"
        f"Fixed: d1={D1_REF if layer!='d1' else 'swept'} nm, "
        f"d2={D2_REF if layer!='d2' else 'swept'} nm, "
        f"d3={D3_REF if layer!='d3' else 'swept'} nm",
        fontsize=11,
    )

    specs = [
        ("T", "T_Y", "T_a", "T_b", "#1f77b4"),
        ("F", "F_Y", "F_a", "F_b", "#ff7f0e"),
        ("G", "G_Y", "G_a", "G_b", "#2ca02c"),
    ]

    for col, (spec, ykey, akey, bkey, color) in enumerate(specs):
        for row, (metric, ylabel) in enumerate([(ykey, "Y"), (akey, "a*"), (bkey, "b*")]):
            ax = axes[row][col]
            ax.plot(x, results[metric].values, color=color, lw=1.8)
            ax.set_ylabel(f"{spec}  {ylabel}")
            ax.grid(True, alpha=0.3)

            # draw spec limit lines
            limit_key = f"{spec}_{ylabel[0]}"   # e.g. "T_a", "F_b"
            if limit_key in LIMITS and ylabel != "Y":
                lo, hi = LIMITS[limit_key]
                ax.axhline(lo, color="red",  ls="--", lw=0.9, label=f"limit {lo}")
                ax.axhline(hi, color="red",  ls="--", lw=0.9, label=f"limit {hi}")
                ax.legend(fontsize=7)

    # bottom row: Tvis across all columns (merged into col 0, hide others)
    for ax in axes[2][1:]:
        ax.set_visible(False)
    axes[2][0].clear()
    axes[2][0].plot(x, results["Tvis"].values, color="black", lw=1.8)
    axes[2][0].set_ylabel("Tvis (%)")
    axes[2][0].set_xlabel(f"{layer} (nm)")
    axes[2][0].grid(True, alpha=0.3)

    for col in range(3):
        axes[2][col].set_xlabel(f"{layer} (nm)")

    plt.tight_layout()
    out = os.path.join(_OUTDIR_PNG, f"sweep_{layer}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → plot saved: {out}")


def main():
    print("Loading materials…")
    wl, glass_nk, L1_nk, L2_nk, L3_nk, tvis_w = load_materials()

    layer_labels = {
        "d1": f"L1 — {os.path.basename(L1_FILE)}",
        "d2": f"L2 — {os.path.basename(L2_FILE)} (Ag)",
        "d3": f"L3 — {os.path.basename(L3_FILE)}",
    }

    for layer, label in layer_labels.items():
        print(f"\nSweeping {label}…")
        results = run_sweep(layer, wl, glass_nk, L1_nk, L2_nk, L3_nk, tvis_w)

        csv_path = os.path.join(_OUTDIR_CSV, f"sweep_{layer}.csv")
        results.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  → data  saved: {csv_path}")

        plot_sweep(results, layer, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
