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
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
_STACK      = os.path.join(_HERE, "data", "materials")
_DATA       = os.path.join(_HERE, "data")
_OUTDIR     = os.path.join(_HERE, "output")
_OUTDIR_CSV = os.path.join(_OUTDIR, "csv")
_OUTDIR_PNG = os.path.join(_OUTDIR, "png")

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

# ── IGU / NFRC constants ───────────────────────────────────────────────────────
_SIGMA      = 5.670374419e-8     # W/m²/K⁴  Stefan–Boltzmann
_H_OUT_SHGC = 13.0              # W/m²K  exterior film, SHGC boundary conditions
_H_IN_SHGC  = 7.7               # W/m²K  interior film, SHGC boundary conditions
_K_GLASS    = 1.0               # W/m/K  glass thermal conductivity

_GAS_K = {                       # thermal conductivities [W/m/K]
    "argon": 0.0160, "ar": 0.0160,
    "air":   0.0240,
    "krypton": 0.0095, "kr": 0.0095,
}

IGU_DEFAULTS = dict(
    solar_csv  = "E891AM1_5.csv",
    glass2_csv = "glass-clear.csv",
    glass1_mm  = 3.0,
    glass2_mm  = 4.0,
    gap_mm     = 12.0,
    gas        = "90%Ar+10%air",
    eps_normal = 0.02,
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _find(name: str) -> str:
    for folder in (os.getcwd(), _HERE, _DATA, _STACK):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"'{name}' not found.\n"
        f"  Searched: {os.getcwd()}\n"
        f"            {_HERE}\n"
        f"            {_DATA}\n"
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


# ── Spectra plot ──────────────────────────────────────────────────────────────

def plot_tfg_spectra(spec_df: pd.DataFrame, out_png: str, title: str = "") -> None:
    """Save a T/F/G vs wavelength PNG."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    wl = spec_df["wavelength"].values
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wl, spec_df["T_percent"].values, label="T (transmission)",   color="#1f77b4", lw=1.8)
    ax.plot(wl, spec_df["F_percent"].values, label="F (film-side R)",    color="#ff7f0e", lw=1.8)
    ax.plot(wl, spec_df["G_percent"].values, label="G (glass-side R)",   color="#2ca02c", lw=1.8)
    ax.axvspan(380, 780, color="gold", alpha=0.10, label="Visible (380–780 nm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("T / R (%)")
    ax.set_xlim(wl[0], wl[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved spectra plot     : {out_png}")


# ── IGU / NFRC functions ──────────────────────────────────────────────────────

def _u_si_to_ip(u: float) -> float:
    return u * 0.1761101838


def _emissivity_n2h(e_n: float) -> float:
    """Normal emissivity → hemispherical via bisection of the NFRC polynomial."""
    def poly(eh):
        return (-0.000393556 + 0.77769*eh + 0.36499*eh**2 + 5.10459*eh**3
                - 31.73344*eh**4 + 96.50648*eh**5 - 186.00584*eh**6
                + 232.53437*eh**7 - 167.73313*eh**8 + 51.35161*eh**9)
    e_n = float(np.clip(e_n, 1e-6, 0.999999))
    lo, hi = 1e-6, 0.999999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (poly(lo) - e_n) * (poly(mid) - e_n) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _load_solar(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load solar irradiance CSV → (wavelength_nm, irradiance)."""
    df = pd.read_csv(path)
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    wl_col, e_col = (nums[0], nums[1]) if len(nums) >= 2 else (df.columns[0], df.columns[1])
    wl = df[wl_col].to_numpy(float)
    E  = df[e_col].to_numpy(float)
    mask = np.isfinite(wl) & np.isfinite(E)
    wl, E = wl[mask], E[mask]
    if wl.max() < 10:       # µm → nm
        wl *= 1000.0
    return wl, E


def _clear_glass_TR(wl_nm: np.ndarray, n: np.ndarray, k: np.ndarray,
                    t_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Incoherent T and R_eff for a plain glass slab."""
    wl_m = wl_nm * 1e-9
    nc = n + 1j * k
    r = (nc - 1) / (nc + 1)
    R = np.abs(r) ** 2
    alpha = 4 * np.pi * k / wl_m
    x = np.exp(-alpha * t_m)
    T     = np.clip(((1 - R)**2 * x / (1 - R**2 * x**2)).astype(float), 0, 1)
    R_eff = np.clip((R * (1 + (1 - R)**2 * x**2 / (1 - R**2 * x**2))).astype(float), 0, 1)
    return T, R_eff


def _igu_optics(T1, R1_front, R1_back, T2, R2):
    """Incoherent two-pane IGU optics (gap multiple reflections)."""
    den = np.where(np.abs(1 - R1_back * R2) < 1e-9, 1e-9, 1 - R1_back * R2)
    T = np.clip(T1 * T2 / den,               0, 1)
    R = np.clip(R1_front + T1**2 * R2 / den, 0, 1)
    A = np.clip(1 - T - R,                   0, 1)
    return T, R, A


def _weighted_avg(wl, y, wl_min, wl_max, w_wl, w):
    m = (wl >= wl_min) & (wl <= wl_max)
    wl2, y2 = wl[m], y[m]
    if wl2.size < 2:
        return float("nan")
    w2 = np.interp(wl2, w_wl, w, left=0.0, right=0.0)
    num = float(np.trapezoid(y2 * w2, wl2))
    den = float(np.trapezoid(w2, wl2))
    return num / den if den > 0 else float("nan")


def _cie_ybar_5nm() -> tuple[np.ndarray, np.ndarray]:
    wl = np.arange(380, 781, 5, dtype=float)
    ybar = np.array([
        0.000040,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,
        0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.091000,0.112600,
        0.139000,0.169300,0.208000,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,
        0.862000,0.914850,0.954000,0.980300,0.995000,1.000000,0.995000,0.978600,0.952000,0.915400,
        0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,
        0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,
        0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,
        0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,
        0.000015
    ], dtype=float)
    return wl, ybar


def _parse_gas(spec: str) -> dict[str, float]:
    """Parse a gas mix string like '90%Ar+10%air' into {name: fraction}."""
    s = (spec or "").strip().lower().replace(" ", "")
    if not s:
        return {"argon": 1.0}

    def norm(g: str) -> str:
        return {"ar": "argon", "kr": "krypton"}.get(g, g)

    if s in _GAS_K:
        return {norm(s): 1.0}

    mix: dict[str, float] = {}
    for p in re.split(r"[+;,]", s):
        m = re.match(r"^([0-9]*\.?[0-9]+)%([a-z]+)$", p)
        if m:
            mix[norm(m.group(2))] = mix.get(norm(m.group(2)), 0.0) + float(m.group(1)) / 100
            continue
        m = re.match(r"^([a-z]+):([0-9]*\.?[0-9]+)$", p)
        if m:
            mix[norm(m.group(1))] = mix.get(norm(m.group(1)), 0.0) + float(m.group(2))
            continue
        m = re.match(r"^([a-z]+)([0-9]*\.?[0-9]+)$", p)
        if m:
            mix[norm(m.group(1))] = mix.get(norm(m.group(1)), 0.0) + float(m.group(2))
    if not mix:
        return {"argon": 1.0}
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()} if total > 0 else {"argon": 1.0}


def _k_gas(spec: str) -> float:
    return sum(_GAS_K.get(g, _GAS_K["argon"]) * x for g, x in _parse_gas(spec).items())


def _h_rad_gap(eps2_h: float, eps3_h: float, Tm: float) -> float:
    den = max(1 / eps2_h + 1 / eps3_h - 1, 1e-9)
    return 4 * _SIGMA * Tm**3 / den


def _calc_u(t1_m, t2_m, gap_m, gas, eps2_n, eps3_h=0.89,
            h_in=8.0, h_out=25.0, Nu_eff=1.135, Tm=293.15):
    eps2_h = _emissivity_n2h(eps2_n)
    kg = _k_gas(gas)
    h_c = Nu_eff * kg / gap_m
    h_r = _h_rad_gap(eps2_h, eps3_h, Tm)
    h_gap = h_c + h_r
    R = (1/h_out + t1_m/_K_GLASS + 1/h_gap + t2_m/_K_GLASS + 1/h_in)
    U = 1.0 / R
    return U, h_gap, eps2_h, kg


def _inward_frac(side: str, t_m: float, gap_m: float, h_gap: float,
                 h_in: float, h_out: float) -> float:
    R_half = (t_m / 2) / _K_GLASS
    Rg     = t_m / _K_GLASS
    Ri, Ro = 1 / h_in, 1 / h_out
    Rgap   = 1 / h_gap
    if side == "outer":
        R_out = R_half + Ro
        R_in  = R_half + Rgap + Rg + Ri
    else:
        R_in  = R_half + Ri
        R_out = R_half + Rgap + Rg + Ro
    return R_out / (R_in + R_out)


# ── Combined IGU interactive step ─────────────────────────────────────────────

def run_igu_calculation(spec_df: pd.DataFrame, wl: np.ndarray, out_base: str) -> None:
    """Prompt for IGU parameters, compute NFRC-style performance, print and save."""
    print("\n── IGU / NFRC Performance ──────────────────────────────────────────────")
    run = input("  Run IGU/NFRC calculation? [y/N]: ").strip().lower()
    if run not in ("y", "yes"):
        return

    print("\n  Data files:")
    solar_path  = _ask_file("  Solar spectrum CSV", IGU_DEFAULTS["solar_csv"])
    glass2_path = _ask_file("  Clear glass CSV   ", IGU_DEFAULTS["glass2_csv"])

    print("\n  IGU configuration:")
    glass1_mm = _ask_float("  Coated pane thickness  (mm)", IGU_DEFAULTS["glass1_mm"])
    glass2_mm = _ask_float("  Clear pane thickness   (mm)", IGU_DEFAULTS["glass2_mm"])
    gap_mm    = _ask_float("  Gap thickness          (mm)", IGU_DEFAULTS["gap_mm"])
    gas_spec  = _ask(      "  Gap gas spec              ", IGU_DEFAULTS["gas"])
    eps_n     = _ask_float("  Coating normal emissivity ", IGU_DEFAULTS["eps_normal"])
    eps_n     = float(np.clip(eps_n, 1e-6, 0.999999))

    # ── coated pane spectra (T=transmission, F=film-side R, G=glass-side R) ──
    T1    = spec_df["T_percent"].values / 100.0
    R1_f  = spec_df["F_percent"].values / 100.0   # film-side R (faces gap)
    R1_g  = spec_df["G_percent"].values / 100.0   # glass-side R (faces outside)

    # ── clear glass pane ──────────────────────────────────────────────────────
    g2_df = load_nk(glass2_path)
    n2 = np.interp(wl, g2_df["wavelength"].values, g2_df["n"].values)
    k2 = np.interp(wl, g2_df["wavelength"].values, g2_df["k"].values)
    T2, R2 = _clear_glass_TR(wl, n2, k2, glass2_mm * 1e-3)

    # coating on surface #2 (inner face of outer pane, gap side)
    # outside → R1_g; gap side → R1_f
    T_iu, R_iu, A_iu = _igu_optics(T1, R1_g, R1_f, T2, R2)

    # ── solar weighting ────────────────────────────────────────────────────────
    sw_wl, sw_E = _load_solar(solar_path)
    Tsol = _weighted_avg(wl, T_iu, 300, 2500, sw_wl, sw_E)
    Rsol = _weighted_avg(wl, R_iu, 300, 2500, sw_wl, sw_E)
    Asol = 1.0 - Tsol - Rsol

    # ── visible (IGU) ─────────────────────────────────────────────────────────
    v_wl, v_yb = _cie_ybar_5nm()
    VT_iu = _weighted_avg(wl, T_iu, 380, 780, v_wl, v_yb)

    # ── thermal: U-value ──────────────────────────────────────────────────────
    U_si, h_gap, eps2_h, kg = _calc_u(
        glass1_mm * 1e-3, glass2_mm * 1e-3, gap_mm * 1e-3,
        gas_spec, eps_n,
    )
    U_ip = _u_si_to_ip(U_si)

    # ── SHGC (approximate inward-fraction method) ─────────────────────────────
    f_o  = _inward_frac("outer", glass1_mm * 1e-3, gap_mm * 1e-3, h_gap, _H_IN_SHGC, _H_OUT_SHGC)
    f_i  = _inward_frac("inner", glass1_mm * 1e-3, gap_mm * 1e-3, h_gap, _H_IN_SHGC, _H_OUT_SHGC)
    SHGC = Tsol + Asol * 0.99 * f_o + Asol * 0.01 * f_i
    LSG  = VT_iu / SHGC if SHGC > 0 else float("nan")

    # ── monolithic pane: CIE Lab + absorption ─────────────────────────────────
    TY, _, Ta, Tb = cie_lab(wl, spec_df["T_percent"].values)
    FY, _, Fa, Fb = cie_lab(wl, spec_df["F_percent"].values)
    GY, _, Ga, Gb = cie_lab(wl, spec_df["G_percent"].values)
    absorb_pct = 100.0 - TY - FY   # T + F + Abs = 100% for single pane

    # ── print combined summary ─────────────────────────────────────────────────
    W = 62
    bar = "─" * W
    print()
    print(f"  ┌{bar}┐")
    print(f"  │  {'Monolithic coated glass':<{W-2}}│")
    print(f"  ├{'─'*16}┬{'─'*11}┬{'─'*11}┬{'─'*11}┬{'─'*(W-52)}┤")
    print(f"  │  {'Spectrum':<14}│{'Y':>10} │{'a*':>10} │{'b*':>10} │{'':<{W-52}}│")
    print(f"  ├{'─'*16}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*(W-52)}┤")
    print(f"  │  {'T':<14}│{TY:10.2f} │{Ta:10.2f} │{Tb:10.2f} │{'':<{W-52}}│")
    print(f"  │  {'G':<14}│{GY:10.2f} │{Ga:10.2f} │{Gb:10.2f} │{'':<{W-52}}│")
    print(f"  │  {'F':<14}│{FY:10.2f} │{Fa:10.2f} │{Fb:10.2f} │{'':<{W-52}}│")
    print(f"  ├{'─'*16}┴{'─'*11}┴{'─'*11}┴{'─'*11}┴{'─'*(W-52)}┤")
    print(f"  │  Absorption  (= 100% − T_Y − F_Y) = {absorb_pct:6.2f}%{'':<{W-43}}│")
    print(f"  │  Emissivity  (normal)               εn = {eps_n:.3f}{'':<{W-47}}│")
    print(f"  ├{bar}┤")
    print(f"  │  IGU: {glass1_mm:.0f} mm coated / {gap_mm:.0f} mm [{gas_spec}] / {glass2_mm:.0f} mm clear", end="")
    print(" " * max(0, W - len(f"  IGU: {glass1_mm:.0f} mm coated / {gap_mm:.0f} mm [{gas_spec}] / {glass2_mm:.0f} mm clear") + 2) + "│")
    print(f"  ├{bar}┤")
    print(f"  │  Tvis          {VT_iu*100:6.1f}%{'':<{W-23}}│")
    print(f"  │  Tsol          {Tsol*100:6.1f}%{'':<{W-23}}│")
    print(f"  │  U-value       {U_si:6.3f} W/m²K  ({U_ip:.3f} BTU/hr·ft²·°F){'':<{W-47}}│")
    print(f"  │  SHGC          {SHGC*100:6.1f}%{'':<{W-23}}│")
    print(f"  │  LSG           {LSG:6.2f}{'':<{W-22}}│")
    print(f"  └{bar}┘")
    print()

    # ── save IGU CSV ──────────────────────────────────────────────────────────
    igue_path = out_base + "_IGU.csv"
    pd.DataFrame({
        "param": ["T_Y","T_a*","T_b*","G_Y","G_a*","G_b*","F_Y","F_a*","F_b*",
                  "absorption_%","eps_normal",
                  "Tvis_IGU_%","Tsol_%","U_SI","U_IP","SHGC_%","LSG",
                  "gas_k_eff","eps2_hemi"],
        "value": [TY, Ta, Tb, GY, Ga, Gb, FY, Fa, Fb,
                  absorb_pct, eps_n,
                  VT_iu*100, Tsol*100, U_si, U_ip, SHGC*100, LSG,
                  kg, eps2_h],
    }).to_csv(igue_path, index=False, float_format="%.4f")
    print(f"  Saved IGU results CSV  : {igue_path}")

    # ── spectra PNG ───────────────────────────────────────────────────────────
    os.makedirs(_OUTDIR_PNG, exist_ok=True)
    png_name = os.path.basename(out_base) + "_TFG.png"
    png_path = os.path.join(_OUTDIR_PNG, png_name)
    plot_tfg_spectra(spec_df, png_path)


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
        run_igu_calculation(df, wl, out_path)

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
        run_igu_calculation(df, wl, out_path)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
