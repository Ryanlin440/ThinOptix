#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_coating_tool.py  —  simplified WVASE replacement

Stack layout:  air | L3 | L2 | L1 | glass | air
  T = transmission (film-side incidence, normal)
  F = film-side reflectance   (incident from air above L3)
  G = glass-side reflectance  (incident from air below glass)

The coating layers are treated coherently via transfer-matrix method.
The thick glass substrate is treated incoherently to avoid false interference fringes.

Modes:
  1  Specify d1, d2, d3 → compute T/F/G spectra + CIE Lab colour
  2  Optimise d1, d3 to maximise Tvis with L2 (Ag) fixed at 15 nm
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import colour
from colour import SpectralDistribution, SpectralShape
from colour.colorimetry import sd_to_XYZ
from colour.models import XYZ_to_Lab
from scipy.optimize import differential_evolution, minimize

# ── Configuration ─────────────────────────────────────────────────────────────

_HERE   = os.path.dirname(os.path.abspath(__file__))
_STACK  = os.path.join(_HERE, "data", "materials")
_OUTDIR     = os.path.join(_HERE, "output")
_OUTDIR_CSV = os.path.join(_OUTDIR, "csv")

DEFAULTS: dict = dict(
    glass    = "glass-industr-2018-mat.csv",
    L1       = "L1-Si3N4-mat.csv",
    L2       = "L2-Ag-mat.csv",
    L3       = "L3-top-ZrO2-mat.csv",
    d1       = 45.0,
    d2       = 15.0,
    d3       = 50.0,
    glass_mm = 3.0,
    out      = "result",
)

WL_START, WL_END, WL_STEP = 300.0, 2500.0, 10.0
L2_FIXED_NM = 15.0          # Ag layer fixed thickness for optimisation

ILLUMINANT = "D65"           # CIE illuminant for colour and Tvis calculation
CMF_NAME   = "CIE 1931 2 Degree Standard Observer"
VIS_SHAPE  = SpectralShape(380, 780, 1)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _find(name: str) -> str:
    for folder in (os.getcwd(), _HERE, _STACK):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"'{name}' not found.\n"
        f"  Searched: {os.getcwd()}\n"
        f"            {_HERE}\n"
        f"            {_STACK}"
    )


def _ask_file(label: str, default: str) -> str:
    s = input(f"  {label} [{default}]: ").strip()
    return _find(s or default)


def _ask_float(label: str, default: float) -> float:
    s = input(f"  {label} [{default}]: ").strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except ValueError:
        print(f"    ← invalid input, using {default}")
        return float(default)


def _ask(label: str, default: str) -> str:
    s = input(f"  {label} [{default}]: ").strip()
    return s or default


# ── n,k loading ───────────────────────────────────────────────────────────────

def load_nk(path: str) -> pd.DataFrame:
    """Load a material CSV with columns: wavelength (or NK), n, k."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lo = {c.lower(): c for c in df.columns}

    wl_col = (
        lo.get("wavelength") or lo.get("wavelength_nm")
        or lo.get("nk") or df.columns[0]
    )
    for key in ("n", "k"):
        if key not in lo:
            raise ValueError(f"{path}: missing '{key}' column. Found: {list(df.columns)}")

    out = df[[wl_col, lo["n"], lo["k"]]].copy()
    out.columns = ["wavelength", "n", "k"]
    out = (
        out.apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("wavelength")
        .reset_index(drop=True)
    )
    if out["wavelength"].max() < 20:      # microns → nm
        out["wavelength"] *= 1000.0
    return out


def interp_nk(df: pd.DataFrame, wl: np.ndarray) -> np.ndarray:
    """Interpolate n,k to the working wavelength grid. Returns complex n̂ = n − ik."""
    n = np.interp(wl, df["wavelength"].values, df["n"].values)
    k = np.interp(wl, df["wavelength"].values, df["k"].values)
    return (n - 1j * k).astype(complex)


# ── Transfer matrix (coherent, normal incidence) ──────────────────────────────

def _tmm(n0: complex, ns: complex, layers_n: list, layers_d: list, wl: float):
    """Coherent characteristic-matrix method at normal incidence.

    Returns (R, T) as intensity fractions for a stack:  n0 | layers | ns
    """
    M = np.eye(2, dtype=complex)
    for ni, di in zip(layers_n, layers_d):
        delta = 2.0 * np.pi * ni * di / wl
        c, s = np.cos(delta), np.sin(delta)
        M = M @ np.array(
            [[c,         1j * s / ni],
             [1j * ni * s,         c]],
            dtype=complex,
        )
    D = n0 * M[0, 0] + n0 * ns * M[0, 1] + M[1, 0] + ns * M[1, 1]
    r = (n0 * M[0, 0] + n0 * ns * M[0, 1] - M[1, 0] - ns * M[1, 1]) / D
    t = 2.0 * n0 / D
    R = max(0.0, float(abs(r) ** 2))
    T = max(0.0, float(abs(t) ** 2 * np.real(ns) / np.real(n0)))
    return R, T


def _bare(n0: complex, ns: complex):
    """Fresnel intensity R, T at a bare interface n0 → ns (normal incidence)."""
    r = (n0 - ns) / (n0 + ns)
    t = 2.0 * n0 / (n0 + ns)
    R = float(abs(r) ** 2)
    T = float(abs(t) ** 2 * np.real(ns) / np.real(n0))
    return R, T


def _xglass(ng: complex, mm: float, wl_nm: float) -> float:
    """Single-pass internal transmittance of a thick glass plate.

    ng = n − ik  ⟹  imag(ng) = −k, so k = −imag(ng).
    """
    k = max(0.0, -float(np.imag(ng)))
    alpha = 4.0 * np.pi * k / (wl_nm * 1e-9)
    return float(np.exp(-alpha * mm * 1e-3))


# ── T / F / G calculation ─────────────────────────────────────────────────────

def calculate_tfg(
    wl:       np.ndarray,
    glass_nk: np.ndarray,
    L1_nk:    np.ndarray,
    L2_nk:    np.ndarray,
    L3_nk:    np.ndarray,
    d1:       float,
    d2:       float,
    d3:       float,
    glass_mm: float,
) -> pd.DataFrame:
    """Compute T/F/G spectra for the coated-glass system.

    Returns a DataFrame with columns:
      wavelength, T_percent, F_percent, G_percent
    """
    air = complex(1.0)
    T_out, F_out, G_out = [], [], []

    for i, wl_i in enumerate(wl):
        ng, n1, n2, n3 = glass_nk[i], L1_nk[i], L2_nk[i], L3_nk[i]
        x = _xglass(ng, glass_mm, wl_i)

        # ── Film-side: air → L3→L2→L1 → glass ──────────────────────────────
        Rf, Tf = _tmm(air, ng, [n3, n2, n1], [d3, d2, d1], wl_i)

        # Coating viewed from glass side: glass → L1→L2→L3 → air
        Rb, Tb = _tmm(ng, air, [n1, n2, n3], [d1, d2, d3], wl_i)

        # Bare bottom glass surface: glass → air
        Rgb, Tgb = _bare(ng, air)

        # Incoherent multiple reflections (coating ↔ bottom glass surface)
        dF = max(1.0 - Rb * Rgb * x**2, 1e-12)
        T_i = Tf * x * Tgb / dF
        F_i = Rf + Tf * Rgb * x**2 * Tb / dF

        # ── Glass-side: air → glass → L1→L2→L3 → air ───────────────────────
        Rag, Tag = _bare(air, ng)           # bottom glass surface: air → glass
        # Rag == Rgb by symmetry at normal incidence; Tgb already computed

        dG = max(1.0 - Rag * Rb * x**2, 1e-12)
        G_i = Rag + Tag * Rb * x**2 * Tgb / dG

        T_out.append(T_i)
        F_out.append(F_i)
        G_out.append(G_i)

    return pd.DataFrame({
        "wavelength": wl,
        "T_percent":  np.clip(T_out, 0.0, 1.0) * 100.0,
        "F_percent":  np.clip(F_out, 0.0, 1.0) * 100.0,
        "G_percent":  np.clip(G_out, 0.0, 1.0) * 100.0,
    })


# ── CIE colour ────────────────────────────────────────────────────────────────

def _build_cmf_cache():
    cmfs = colour.MSDS_CMFS[CMF_NAME].copy().align(VIS_SHAPE)
    ill  = colour.SDS_ILLUMINANTS[ILLUMINANT].copy().align(VIS_SHAPE)
    xy_w = colour.CCS_ILLUMINANTS[CMF_NAME][ILLUMINANT]
    return cmfs, ill, xy_w

_cmfs, _ill, _xy_w = _build_cmf_cache()


def make_tvis_weights(wl_grid: np.ndarray) -> np.ndarray:
    """Precompute per-wavelength weights w so that Tvis% = 100 · dot(T_frac, w).

    Uses CIE y̅(λ) · S(λ) normalised to sum = 1 on the working wavelength grid.
    Only wavelengths in 380–780 nm contribute (CMF is zero outside).
    """
    wl_vis = _cmfs.wavelengths                           # 380–780 nm at 1 nm
    yS = _cmfs.values[:, 1] * _ill.values.squeeze()     # y̅ · S
    w  = np.interp(wl_grid, wl_vis, yS, left=0.0, right=0.0)
    total = w.sum()
    return w / max(total, 1e-12)


def tvis_fast(T_percent: np.ndarray, weights: np.ndarray) -> float:
    """Fast photopic transmittance using precomputed CIE weights."""
    return float(100.0 * np.dot(T_percent / 100.0, weights))


def cie_lab(wl_nm: np.ndarray, values_percent: np.ndarray) -> tuple[float, float, float, float]:
    """Compute (Y, L*, a*, b*) for a spectrum given in percent.

    Uses ASTM E308 method with the configured illuminant and CMF.
    Y is the CIE photopic transmittance/reflectance (0–100 scale).
    """
    v = np.clip(values_percent / 100.0, 0.0, None)
    sd = SpectralDistribution(dict(zip(wl_nm.tolist(), v.tolist())))
    sd = sd.copy().align(VIS_SHAPE)

    XYZ   = sd_to_XYZ(sd,                            cmfs=_cmfs, illuminant=_ill, k=None, method="ASTM E308")
    XYZ_w = sd_to_XYZ(colour.sd_ones(VIS_SHAPE),     cmfs=_cmfs, illuminant=_ill, k=None, method="ASTM E308")

    XYZ_norm = (XYZ / XYZ_w[1]) * 100.0
    Lab = XYZ_to_Lab(XYZ_norm / 100.0, _xy_w)

    return float(XYZ_norm[1]), float(Lab[0]), float(Lab[1]), float(Lab[2])


# ── Optimisation ──────────────────────────────────────────────────────────────

def optimise_thicknesses(
    wl:       np.ndarray,
    glass_nk: np.ndarray,
    L1_nk:    np.ndarray,
    L2_nk:    np.ndarray,
    L3_nk:    np.ndarray,
    glass_mm: float,
    d1_bounds: tuple[float, float] = (5.0, 200.0),
    d3_bounds: tuple[float, float] = (5.0, 200.0),
) -> dict:
    """Maximise Tvis by varying d1 (L1) and d3 (L3) with d2 (Ag) = L2_FIXED_NM.

    Strategy:
      1. Differential evolution (global) — finds basin of attraction.
      2. L-BFGS-B (local polish) — refines to high accuracy.

    Returns dict: d1, d2, d3 (nm) and Tvis (%).
    """
    tvis_w = make_tvis_weights(wl)

    def neg_tvis(x):
        d1, d3 = float(x[0]), float(x[1])
        out = calculate_tfg(
            wl, glass_nk, L1_nk, L2_nk, L3_nk,
            d1, L2_FIXED_NM, d3, glass_mm,
        )
        return -tvis_fast(out["T_percent"].values, tvis_w)

    bounds = [d1_bounds, d3_bounds]

    print("  Phase 1: global search (differential evolution)…", flush=True)
    de = differential_evolution(
        neg_tvis,
        bounds=bounds,
        seed=42,
        maxiter=300,
        popsize=15,
        tol=5e-5,
        workers=1,
        disp=False,
        polish=False,
    )
    print(f"    global best: d1={de.x[0]:.1f}  d3={de.x[1]:.1f}  Tvis={-de.fun:.3f}%")

    print("  Phase 2: local polish (L-BFGS-B)…", flush=True)
    pol = minimize(
        neg_tvis,
        x0=de.x,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    )

    best = pol if pol.fun <= de.fun else de
    d1_opt, d3_opt = best.x
    return {
        "d1":   float(d1_opt),
        "d2":   L2_FIXED_NM,
        "d3":   float(d3_opt),
        "Tvis": float(-best.fun),
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def print_color_table(df: pd.DataFrame) -> list[dict]:
    """Compute and print CIE Lab for each spectrum column. Returns list of row dicts."""
    wl = df["wavelength"].values
    spec_cols = [c for c in df.columns if c != "wavelength"]

    print(f"\n  CIE Lab  [{ILLUMINANT}, {CMF_NAME}]")
    header = f"  {'Spectrum':<14}{'Y':>8}{'L*':>8}{'a*':>8}{'b*':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    rows = []
    for col in spec_cols:
        tmp = df[["wavelength", col]].dropna()
        Y, L, a, b = cie_lab(tmp["wavelength"].values, tmp[col].values)
        print(f"  {col:<14}{Y:8.2f}{L:8.2f}{a:8.2f}{b:8.2f}")
        rows.append({"spectrum": col, "Y": round(Y, 4),
                     "L*": round(L, 4), "a*": round(a, 4), "b*": round(b, 4)})
    return rows


def save_results(out_base: str, spec_df: pd.DataFrame, color_rows: list[dict]):
    """Save spectral TFG CSV and colour Lab CSV."""
    spec_path  = out_base + "_TFG.csv"
    color_path = out_base + "_Lab.csv"
    spec_df.to_csv(spec_path,  index=False, float_format="%.6f")
    pd.DataFrame(color_rows).to_csv(color_path, index=False, float_format="%.4f")
    print(f"\n  Saved spectral CSV : {spec_path}")
    print(f"  Saved colour CSV   : {color_path}")


def print_spectrum_summary(df: pd.DataFrame, tvis_w: np.ndarray):
    """Print T/F/G values at key wavelengths plus Tvis."""
    wl   = df["wavelength"].values
    T    = df["T_percent"].values
    F    = df["F_percent"].values
    G    = df["G_percent"].values
    tvis = tvis_fast(T, tvis_w)

    print(f"\n  Tvis (CIE Y, {ILLUMINANT}) = {tvis:.2f}%")
    print(f"\n  {'λ (nm)':>8}  {'T %':>7}  {'F %':>7}  {'G %':>7}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    key_wl = [380, 400, 450, 500, 550, 600, 650, 700, 750, 780]
    for target in key_wl:
        idx = np.argmin(abs(wl - target))
        print(f"  {wl[idx]:>8.0f}  {T[idx]:7.2f}  {F[idx]:7.2f}  {G[idx]:7.2f}")


# ── Material loading (interactive) ────────────────────────────────────────────

def _load_materials():
    """Ask user for material files and interpolate to the working wavelength grid."""
    print("\n── Material files (press Enter to accept defaults) ──────────────────")
    glass_file = _ask_file("Glass substrate CSV", DEFAULTS["glass"])
    L1_file    = _ask_file("L1 material CSV    ", DEFAULTS["L1"])
    L2_file    = _ask_file("L2 material CSV    ", DEFAULTS["L2"])
    L3_file    = _ask_file("L3 material CSV    ", DEFAULTS["L3"])

    wl = np.arange(WL_START, WL_END + 0.1, WL_STEP)

    print("\n  Loading material data…", end=" ", flush=True)
    glass_nk = interp_nk(load_nk(glass_file), wl)
    L1_nk    = interp_nk(load_nk(L1_file),    wl)
    L2_nk    = interp_nk(load_nk(L2_file),    wl)
    L3_nk    = interp_nk(load_nk(L3_file),    wl)
    print("done.")

    return wl, glass_nk, L1_nk, L2_nk, L3_nk


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_OUTDIR_CSV, exist_ok=True)
    print()
    print("═" * 60)
    print("  Optical Coating Design Tool  (WVASE simplified replacement)")
    print(f"  Stack: air | L3 | L2 | L1 | glass | air")
    print(f"  Colour / Tvis: illuminant {ILLUMINANT}, {CMF_NAME}")
    print("═" * 60)

    print("\nMode:")
    print("  1  Calculate T/F/G + colour for given thicknesses")
    print("  2  Optimise L1/L3 thicknesses to maximise Tvis")
    print("     (L2 Ag fixed at 15 nm)")
    mode = input("\n  Your choice [1]: ").strip() or "1"

    if mode not in ("1", "2"):
        print("Unknown mode. Exiting.")
        return

    wl, glass_nk, L1_nk, L2_nk, L3_nk = _load_materials()
    tvis_w = make_tvis_weights(wl)

    print("\n── Glass ────────────────────────────────────────────────────────────")
    glass_mm = _ask_float("Glass substrate thickness (mm)", DEFAULTS["glass_mm"])

    # ── Mode 1: single calculation ────────────────────────────────────────────
    if mode == "1":
        print("\n── Layer thicknesses (nm) ───────────────────────────────────────────")
        d1 = _ask_float("L1 thickness (nm)", DEFAULTS["d1"])
        d2 = _ask_float("L2 thickness (nm)", DEFAULTS["d2"])
        d3 = _ask_float("L3 thickness (nm)", DEFAULTS["d3"])

        print("\n── Output ───────────────────────────────────────────────────────────")
        out_base = _ask("Output filename base", DEFAULTS["out"])
        out_path = os.path.join(_OUTDIR_CSV, out_base)

        print(f"\n  Calculating (d1={d1:.1f} nm, d2={d2:.1f} nm, d3={d3:.1f} nm)…",
              flush=True)
        df = calculate_tfg(wl, glass_nk, L1_nk, L2_nk, L3_nk, d1, d2, d3, glass_mm)

        print_spectrum_summary(df, tvis_w)
        color_rows = print_color_table(df)
        save_results(out_path, df, color_rows)

    # ── Mode 2: optimisation ──────────────────────────────────────────────────
    else:
        print("\n── Optimisation bounds (nm) ─────────────────────────────────────────")
        d1_lo = _ask_float("L1 min (nm)", 5.0)
        d1_hi = _ask_float("L1 max (nm)", 200.0)
        d3_lo = _ask_float("L3 min (nm)", 5.0)
        d3_hi = _ask_float("L3 max (nm)", 200.0)

        print("\n── Output ───────────────────────────────────────────────────────────")
        out_base = _ask("Output filename base", DEFAULTS["out"])
        out_path = os.path.join(_OUTDIR_CSV, out_base)

        print(f"\n  Optimising d1, d3  (L2 Ag = {L2_FIXED_NM:.0f} nm fixed)…\n")
        result = optimise_thicknesses(
            wl, glass_nk, L1_nk, L2_nk, L3_nk, glass_mm,
            d1_bounds=(d1_lo, d1_hi),
            d3_bounds=(d3_lo, d3_hi),
        )

        d1_opt = result["d1"]
        d2_opt = result["d2"]
        d3_opt = result["d3"]
        tvis   = result["Tvis"]

        print(f"\n  ┌─ Optimised thicknesses ──────────────────────┐")
        print(f"  │  L1 = {d1_opt:7.2f} nm                          │")
        print(f"  │  L2 = {d2_opt:7.2f} nm  (Ag, fixed)             │")
        print(f"  │  L3 = {d3_opt:7.2f} nm                          │")
        print(f"  │  Tvis ({ILLUMINANT}) = {tvis:6.2f}%                    │")
        print(f"  └──────────────────────────────────────────────┘")

        print("\n  Computing spectra at optimal thicknesses…", flush=True)
        df = calculate_tfg(
            wl, glass_nk, L1_nk, L2_nk, L3_nk,
            d1_opt, d2_opt, d3_opt, glass_mm,
        )

        print_spectrum_summary(df, tvis_w)
        color_rows = print_color_table(df)
        save_results(out_path, df, color_rows)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
