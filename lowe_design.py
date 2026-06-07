#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lowe_design.py  —  Low-E coating grid search

Stack:  air | L3 | L2(Ag) | L1 | glass | air
  T = transmission (normal incidence, film side)
  F = film-side reflectance   (incident from air above L3)
  G = glass-side reflectance  (incident from air below glass)

For each Ag thickness in [10, 12.5, 15] nm, exhaustively search:
  d1 (L1) ∈ [1, 100] nm  — 1 nm step
  d3 (L3) ∈ [1, 100] nm  — 1 nm step

Design constraints (all must be satisfied):
  Tvis (Y)     → maximise
  Fvis (Y)     < 12%
  Gvis (Y)     < 12%
  Ta*  ∈ [-3,  0]
  Tb*  ∈ [-3,  3]
  Ga*  ∈ [-5,  0]
  Gb*  ∈ [-15, -5]
  Fa*  ∈ [-4,  4]
  Fb*  ∈ [-15, -4]

Outputs per Ag thickness:
  lowe_Ag<d2>nm_passing.csv      — every passing design sorted by Tvis ↓
  lowe_Ag<d2>nm_top<N>_colors.png — color patches for the top N designs
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import colour
from colour import SpectralShape
from colour.models import XYZ_to_Lab


# ── Configuration — edit these to change the search ───────────────────────────

_HERE   = os.path.dirname(os.path.abspath(__file__))
_STACK  = os.path.join(_HERE, "data", "materials")
_OUTDIR     = os.path.join(_HERE, "output")
_OUTDIR_CSV = os.path.join(_OUTDIR, "csv")
_OUTDIR_PNG = os.path.join(_OUTDIR, "png")

GLASS_FILE = os.path.join(_STACK, "glass-industr-2018-mat.csv")
L1_FILE    = os.path.join(_STACK, "L1-Si3N4-mat.csv")
L2_FILE    = os.path.join(_STACK, "L2-Ag-mat.csv")
L3_FILE    = os.path.join(_STACK, "L3-top-ZrO2-mat.csv")

GLASS_MM        = 3.0                      # glass substrate thickness (mm)
AG_THICKNESSES  = [10.0, 12.5, 15.0]      # Ag layer fixed values (nm)
D1_RANGE        = np.arange(1.0, 101.0)   # L1 search range 1–100 nm
D3_RANGE        = np.arange(1.0, 101.0)   # L3 search range 1–100 nm
TOP_N           = 5                        # designs to visualise per Ag thickness

WL_START, WL_END, WL_STEP = 300.0, 2500.0, 10.0

ILLUMINANT = "D65"
CMF_NAME   = "CIE 1931 2 Degree Standard Observer"
VIS_SHAPE  = SpectralShape(380, 780, 1)

# ── Design constraints ────────────────────────────────────────────────────────
FVIS_MAX = 12.0
GVIS_MAX = 12.0
TA_MIN, TA_MAX = -3.0,  0.0
TB_MIN, TB_MAX = -3.0,  3.0
GA_MIN, GA_MAX = -5.0,  0.0
GB_MIN, GB_MAX = -15.0, -5.0
FA_MIN, FA_MAX = -4.0,  4.0
FB_MIN, FB_MAX = -15.0, -4.0


# ── n,k loading ───────────────────────────────────────────────────────────────

def load_nk(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lo = {c.lower(): c for c in df.columns}
    wl_col = (
        lo.get("wavelength") or lo.get("wavelength_nm")
        or lo.get("nk") or df.columns[0]
    )
    for key in ("n", "k"):
        if key not in lo:
            raise ValueError(f"{path}: missing '{key}' column")
    out = df[[wl_col, lo["n"], lo["k"]]].copy()
    out.columns = ["wavelength", "n", "k"]
    out = (
        out.apply(pd.to_numeric, errors="coerce")
        .dropna().sort_values("wavelength").reset_index(drop=True)
    )
    if out["wavelength"].max() < 20:
        out["wavelength"] *= 1000.0
    return out


def interp_nk(df: pd.DataFrame, wl: np.ndarray) -> np.ndarray:
    n = np.interp(wl, df["wavelength"].values, df["n"].values)
    k = np.interp(wl, df["wavelength"].values, df["k"].values)
    return (n - 1j * k).astype(complex)


# ── Vectorised transfer matrix ────────────────────────────────────────────────

def _tmm_grid(n0: complex, ns: complex,
              n_layers: list, d_layers: list, wl: float):
    """Characteristic-matrix TMM at normal incidence.

    d_layers entries may be 2-D arrays (for d1/d3 grids).
    Broadcasting propagates the grid shape through the matrix product,
    producing R, T arrays whose shape matches the broadcast of all d_layers.

    Order: n0 | layer[0] | layer[1] | … | ns
    """
    M00 = np.ones(1, dtype=complex)
    M01 = np.zeros(1, dtype=complex)
    M10 = np.zeros(1, dtype=complex)
    M11 = np.ones(1, dtype=complex)

    for ni, di in zip(n_layers, d_layers):
        delta = (2.0 * np.pi / wl) * ni * di
        c = np.cos(delta)
        s = np.sin(delta)
        n00 = M00 * c        + M01 * (1j * ni * s)
        n01 = M00 * (1j * s / ni) + M01 * c
        n10 = M10 * c        + M11 * (1j * ni * s)
        n11 = M10 * (1j * s / ni) + M11 * c
        M00, M01, M10, M11 = n00, n01, n10, n11

    D = n0 * M00 + n0 * ns * M01 + M10 + ns * M11
    r = (n0 * M00 + n0 * ns * M01 - M10 - ns * M11) / D
    t = 2.0 * n0 / D
    R = np.maximum(0.0, (r * np.conj(r)).real)
    T = np.maximum(0.0, (t * np.conj(t)).real * np.real(ns) / np.real(n0))
    return R, T


def _bare(n0: complex, ns: complex):
    r = (n0 - ns) / (n0 + ns)
    t = 2.0 * n0 / (n0 + ns)
    return float(abs(r) ** 2), float(abs(t) ** 2 * np.real(ns) / np.real(n0))


def _xglass(ng: complex, mm: float, wl_nm: float) -> float:
    k = max(0.0, -float(np.imag(ng)))       # ng = n − ik, imag = −k
    alpha = 4.0 * np.pi * k / (wl_nm * 1e-9)
    return float(np.exp(-alpha * mm * 1e-3))


# ── Grid T/F/G computation ────────────────────────────────────────────────────

def calculate_tfg_grid(
    wl:       np.ndarray,
    glass_nk: np.ndarray,
    L1_nk:    np.ndarray,
    L2_nk:    np.ndarray,
    L3_nk:    np.ndarray,
    d1_arr:   np.ndarray,      # shape (N1,)
    d2:       float,
    d3_arr:   np.ndarray,      # shape (N3,)
    glass_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute T/F/G for every (d1, d3) combination.

    Returns T, F, G each of shape (N1, N3, NW) in percent.

    Strategy: loop over wavelengths (221 iterations), vectorise the
    N1×N3 = 10 000 thickness combinations at each step using 2-D
    broadcast arrays.  Peak extra memory is a few ×(N1×N3) complex arrays
    (< 1 MB), while the TMM inner-loop stays in pure NumPy.
    """
    N1, N3, NW = len(d1_arr), len(d3_arr), len(wl)

    # 2-D broadcast grids — shape (N1, 1) and (1, N3)
    d1_g = d1_arr[:, np.newaxis].astype(complex)
    d3_g = d3_arr[np.newaxis, :].astype(complex)

    T_spec = np.empty((N1, N3, NW), dtype=np.float32)
    F_spec = np.empty((N1, N3, NW), dtype=np.float32)
    G_spec = np.empty((N1, N3, NW), dtype=np.float32)

    air = complex(1.0)

    for i, wl_i in enumerate(wl):
        ng, n1, n2, n3 = glass_nk[i], L1_nk[i], L2_nk[i], L3_nk[i]
        x = _xglass(ng, glass_mm, wl_i)

        # Film-side:  air → L3 → L2 → L1 → glass
        Rf, Tf = _tmm_grid(air, ng, [n3, n2, n1], [d3_g, d2, d1_g], wl_i)
        # Coating from glass side: glass → L1 → L2 → L3 → air
        Rb, Tb = _tmm_grid(ng, air, [n1, n2, n3], [d1_g, d2, d3_g], wl_i)

        # Bare bottom glass surface (glass → air)
        Rgb, Tgb = _bare(ng, air)
        # Bare bottom glass surface (air → glass)
        Rag, Tag = _bare(air, ng)          # Rag == Rgb at normal incidence

        dF = np.maximum(1.0 - Rb * Rgb * x**2, 1e-12)
        T_i = np.clip(Tf * x * Tgb / dF, 0.0, 1.0)
        F_i = np.clip(Rf + Tf * Rgb * x**2 * Tb / dF, 0.0, 1.0)

        dG = np.maximum(1.0 - Rag * Rb * x**2, 1e-12)
        G_i = np.clip(Rag + Tag * Rb * x**2 * Tgb / dG, 0.0, 1.0)

        T_spec[:, :, i] = (T_i * 100).astype(np.float32)
        F_spec[:, :, i] = (F_i * 100).astype(np.float32)
        G_spec[:, :, i] = (G_i * 100).astype(np.float32)

    return T_spec, F_spec, G_spec


# ── CIE colour (vectorised) ───────────────────────────────────────────────────

def _build_xyz_weights(wl_grid: np.ndarray) -> np.ndarray:
    """Return (NW, 3) weight matrix W such that XYZ = (spec_frac @ W) * 100.

    W is normalised so that a perfect white (spec_frac = 1) gives Y = 1.0
    (i.e. XYZ[1] = 1.0  →  Y% = 100).
    """
    cmfs = colour.MSDS_CMFS[CMF_NAME].copy().align(VIS_SHAPE)
    ill  = colour.SDS_ILLUMINANTS[ILLUMINANT].copy().align(VIS_SHAPE)

    wl_vis = cmfs.wavelengths                          # 380–780 nm
    xyz_bar = cmfs.values                              # (401, 3)
    S = ill.values.squeeze()                           # (401,)

    xyzS = xyz_bar * S[:, np.newaxis]                 # (401, 3)
    w = np.zeros((len(wl_grid), 3))
    for j in range(3):
        w[:, j] = np.interp(wl_grid, wl_vis, xyzS[:, j], left=0.0, right=0.0)

    denom = w[:, 1].sum()                             # = ∑ y̅·S  (Y norm)
    return w / max(denom, 1e-12)


def _xyz_to_lab_vec(XYZ_frac: np.ndarray, XYZ_n_frac: np.ndarray) -> np.ndarray:
    """Vectorised CIE Lab from normalised XYZ (Y=1 for white).

    XYZ_frac  : (…, 3)  —  XYZ / 100 of the sample
    XYZ_n_frac: (3,)    —  XYZ / 100 for the white point (illuminant)
    Returns Lab (…, 3)  —  L* 0-100, a* and b* unbounded.
    """
    xyz_r = XYZ_frac / XYZ_n_frac          # (…, 3)  normalised to white

    delta = 6.0 / 29.0
    f = np.where(
        xyz_r > delta**3,
        np.cbrt(np.maximum(xyz_r, 0.0)),
        xyz_r / (3.0 * delta**2) + 4.0 / 29.0,
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def compute_color_grid(
    spec: np.ndarray,      # (N1, N3, NW) in percent
    xyz_w: np.ndarray,     # (NW, 3) weight matrix from _build_xyz_weights
    XYZ_n: np.ndarray,     # (3,) white-point XYZ fraction
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CIE XYZ (fraction) and Lab for every grid point.

    Returns:
      XYZ : (N1, N3, 3)  — XYZ fractions (Y = 1.0 for 100%)
      Lab : (N1, N3, 3)  — L*, a*, b*
    """
    N1, N3, NW = spec.shape
    spec_flat = spec.reshape(-1, NW).astype(np.float64) / 100.0   # (N1*N3, NW)
    XYZ_flat  = spec_flat @ xyz_w                                   # (N1*N3, 3)
    Lab_flat  = _xyz_to_lab_vec(XYZ_flat, XYZ_n)                   # (N1*N3, 3)
    return XYZ_flat.reshape(N1, N3, 3), Lab_flat.reshape(N1, N3, 3)


# ── Lab → display RGB ─────────────────────────────────────────────────────────

def lab_to_display_rgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    """Convert CIE L*a*b* (D65) to gamma-corrected sRGB [0, 1] for display.

    Values outside the sRGB gamut are clipped.
    """
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def f_inv(t: float) -> float:
        return t**3 if t > delta else 3.0 * delta**2 * (t - 4.0 / 29.0)

    # XYZ (D65 white = 0.95047, 1.00000, 1.08883)
    X = 0.95047 * f_inv(fx)
    Y = 1.00000 * f_inv(fy)
    Z = 1.08883 * f_inv(fz)

    # XYZ → linear sRGB (IEC 61966-2-1)
    R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * c ** (1.0 / 2.4) - 0.055

    return (
        float(np.clip(gamma(R), 0.0, 1.0)),
        float(np.clip(gamma(G), 0.0, 1.0)),
        float(np.clip(gamma(B), 0.0, 1.0)),
    )


# ── Constraint check ──────────────────────────────────────────────────────────

def check_constraints(
    T_Y: np.ndarray, F_Y: np.ndarray, G_Y: np.ndarray,
    T_lab: np.ndarray, F_lab: np.ndarray, G_lab: np.ndarray,
) -> np.ndarray:
    """Return boolean mask (N1, N3) — True where all constraints are met."""
    Ta = T_lab[..., 1];  Tb = T_lab[..., 2]
    Fa = F_lab[..., 1];  Fb = F_lab[..., 2]
    Ga = G_lab[..., 1];  Gb = G_lab[..., 2]

    mask = (
        (F_Y < FVIS_MAX) &
        (G_Y < GVIS_MAX) &
        (Ta >= TA_MIN) & (Ta <= TA_MAX) &
        (Tb >= TB_MIN) & (Tb <= TB_MAX) &
        (Ga >= GA_MIN) & (Ga <= GA_MAX) &
        (Gb >= GB_MIN) & (Gb <= GB_MAX) &
        (Fa >= FA_MIN) & (Fa <= FA_MAX) &
        (Fb >= FB_MIN) & (Fb <= FB_MAX)
    )
    return mask


# ── Color-patch figure ────────────────────────────────────────────────────────

def save_color_figure(
    designs: pd.DataFrame,
    d2: float,
    top_n: int,
    out_path: str,
) -> None:
    """Save a PNG with color patches (T / F / G) for the top_n designs."""
    rows = min(top_n, len(designs))
    if rows == 0:
        return

    fig_w = 7.0
    patch_h = 1.8
    label_h = 0.7
    margin_top = 0.5
    fig_h = rows * (patch_h + label_h) + margin_top

    fig, axes = plt.subplots(
        rows, 3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [patch_h + label_h] * rows},
    )
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["T  (transmission)", "F  (film-side refl.)", "G  (glass-side refl.)"]
    fig.suptitle(
        f"Top {rows} designs — Ag L2 = {d2:.1f} nm  |  illuminant {ILLUMINANT}",
        fontsize=10, y=1.0,
    )

    for row_idx, (_, row) in enumerate(designs.head(rows).iterrows()):
        for col_idx, ch in enumerate(("T", "F", "G")):
            ax = axes[row_idx, col_idx]
            L = row[f"{ch}_L"]
            a = row[f"{ch}_a"]
            b = row[f"{ch}_b"]
            Y = row[f"{ch}_Y"]
            rgb = lab_to_display_rgb(L, a, b)

            # Color rectangle
            rect = mpatches.FancyBboxPatch(
                (0.05, 0.35), 0.90, 0.60,
                boxstyle="round,pad=0.02",
                linewidth=0.6, edgecolor="#333333",
                facecolor=rgb,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(rect)

            # Label text
            label = (
                f"Y={Y:.1f}%\n"
                f"L*={L:.1f}\n"
                f"a*={a:.1f}  b*={b:.1f}"
            )
            ax.text(
                0.50, 0.12, label,
                ha="center", va="bottom",
                transform=ax.transAxes,
                fontsize=7.5, family="monospace",
            )

            # Column header (first row only)
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=8, pad=4)

            # Row header
            if col_idx == 0:
                ax.set_ylabel(
                    f"d1={row['d1']:.0f}  d3={row['d3']:.0f} nm\nTvis={row['T_Y']:.1f}%",
                    fontsize=7.5, labelpad=4,
                )

            ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {out_path}")


# ── 3-D surface figures ───────────────────────────────────────────────────────

def save_3d_tvis_figure(
    d1_arr: np.ndarray,
    d3_arr: np.ndarray,
    T_Y: np.ndarray,
    mask: np.ndarray,
    best_d1: float,
    best_d3: float,
    best_tvis: float,
    d2: float,
    out_path: str,
) -> None:
    """3-D Tvis surface with all passing designs and the best design highlighted."""
    D1, D3 = np.meshgrid(d1_arr, d3_arr, indexing="ij")
    i1, i3 = np.where(mask)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(D1, D3, T_Y, cmap="viridis", alpha=0.82,
                           linewidth=0, antialiased=True, rstride=2, cstride=2)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.50, pad=0.08)
    cbar.set_label("Tvis (%)", fontsize=9)

    if len(i1):
        ax.scatter(d1_arr[i1], d3_arr[i3], T_Y[i1, i3],
                   color="tomato", s=14, depthshade=True,
                   label=f"passing  (n={len(i1)})", zorder=5)

    ax.scatter([best_d1], [best_d3], [best_tvis],
               color="gold", s=200, marker="*", depthshade=False,
               edgecolors="k", linewidths=0.7,
               label=f"best  Tvis = {best_tvis:.2f}%\nd1={best_d1:.0f} nm  d3={best_d3:.0f} nm",
               zorder=10)

    ax.set_xlabel("d1 – L1 (nm)", labelpad=9, fontsize=9)
    ax.set_ylabel("d3 – L3 (nm)", labelpad=9, fontsize=9)
    ax.set_zlabel("Tvis (%)",      labelpad=9, fontsize=9)
    ax.set_title(f"Visible Transmission  |  Ag L2 = {d2:.1f} nm", fontsize=11, pad=14)
    ax.view_init(elev=28, azim=225)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.75)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {out_path}")


def save_3d_color_figure(
    d1_arr: np.ndarray,
    d3_arr: np.ndarray,
    T_lab: np.ndarray,
    F_lab: np.ndarray,
    G_lab: np.ndarray,
    mask: np.ndarray,
    d2: float,
    out_path: str,
) -> None:
    """2 × 3 grid of 3-D surfaces for Ta*, Tb*, Fa*, Fb*, Ga*, Gb*.

    Translucent horizontal planes mark the spec min/max limits.
    Red dots show where designs pass ALL constraints simultaneously.
    """
    D1, D3 = np.meshgrid(d1_arr, d3_arr, indexing="ij")
    i1, i3 = np.where(mask)

    panels = [
        ("Ta*", T_lab[..., 1], TA_MIN, TA_MAX, "PiYG"),
        ("Tb*", T_lab[..., 2], TB_MIN, TB_MAX, "PuOr"),
        ("Fa*", F_lab[..., 1], FA_MIN, FA_MAX, "PiYG"),
        ("Fb*", F_lab[..., 2], FB_MIN, FB_MAX, "PuOr"),
        ("Ga*", G_lab[..., 1], GA_MIN, GA_MAX, "PiYG"),
        ("Gb*", G_lab[..., 2], GB_MIN, GB_MAX, "PuOr"),
    ]

    fig = plt.figure(figsize=(22, 13))
    fig.suptitle(
        f"Color Parameter Landscape  |  Ag L2 = {d2:.1f} nm\n"
        "Blue plane = spec lower limit  |  Red plane = spec upper limit  "
        "|  Red dots = passing designs",
        fontsize=11,
    )

    # Corner grids for limit planes (shared across subplots)
    _xx = np.array([[d1_arr[0], d1_arr[-1]], [d1_arr[0], d1_arr[-1]]])
    _yy = np.array([[d3_arr[0], d3_arr[0]], [d3_arr[-1], d3_arr[-1]]])

    for idx, (label, Z, zmin, zmax, cmap) in enumerate(panels):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")

        surf = ax.plot_surface(D1, D3, Z, cmap=cmap, alpha=0.78,
                               linewidth=0, antialiased=True, rstride=2, cstride=2)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.42, pad=0.08)
        cbar.set_label(label, fontsize=8)

        # Spec-limit planes
        for zlim, col in [(zmin, "steelblue"), (zmax, "firebrick")]:
            zz = np.full_like(_xx, float(zlim))
            ax.plot_surface(_xx, _yy, zz, color=col, alpha=0.20, linewidth=0)

        # Passing designs (all constraints met)
        if len(i1):
            ax.scatter(d1_arr[i1], d3_arr[i3], Z[i1, i3],
                       color="red", s=7, depthshade=False, zorder=5)

        ax.set_xlabel("d1 – L1 (nm)", labelpad=5, fontsize=7)
        ax.set_ylabel("d3 – L3 (nm)", labelpad=5, fontsize=7)
        ax.set_zlabel(label,           labelpad=5, fontsize=7)
        ax.set_title(f"{label}   spec [{zmin}, {zmax}]", fontsize=9, pad=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=28, azim=225)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {out_path}")


def print_trends_report(
    d1_arr: np.ndarray,
    d3_arr: np.ndarray,
    T_Y: np.ndarray,
    T_lab: np.ndarray,
    F_lab: np.ndarray,
    G_lab: np.ndarray,
    mask: np.ndarray,
    d2: float,
) -> None:
    """Compute Pearson correlations and print tuning guidance."""
    N1, N3 = T_Y.shape
    d1_flat = np.repeat(d1_arr[:, np.newaxis], N3, axis=1).ravel()
    d3_flat = np.repeat(d3_arr[np.newaxis, :], N1, axis=0).ravel()

    metrics = [
        ("T_Y",  T_Y.ravel(),           None,    None),
        ("Ta*",  T_lab[..., 1].ravel(), TA_MIN,  TA_MAX),
        ("Tb*",  T_lab[..., 2].ravel(), TB_MIN,  TB_MAX),
        ("Fa*",  F_lab[..., 1].ravel(), FA_MIN,  FA_MAX),
        ("Fb*",  F_lab[..., 2].ravel(), FB_MIN,  FB_MAX),
        ("Ga*",  G_lab[..., 1].ravel(), GA_MIN,  GA_MAX),
        ("Gb*",  G_lab[..., 2].ravel(), GB_MIN,  GB_MAX),
    ]

    sep = "  " + "─" * 56
    print(f"\n{sep}")
    print(f"  TREND ANALYSIS  —  Ag L2 = {d2:.1f} nm")
    print(sep)

    # ── Full-grid Pearson correlations
    print(f"\n  Full-grid Pearson ρ (all {N1*N3:,} thickness combinations):")
    print(f"    {'Metric':>5}   ρ(d1-L1)   ρ(d3-L3)   dominant layer")
    print("    " + "─" * 46)
    corrs: dict[str, tuple[float, float]] = {}
    for name, vals, _, _ in metrics:
        r1 = float(np.corrcoef(d1_flat, vals)[0, 1])
        r3 = float(np.corrcoef(d3_flat, vals)[0, 1])
        corrs[name] = (r1, r3)
        dom = "L1 (d1)" if abs(r1) >= abs(r3) else "L3 (d3)"
        print(f"    {name:>5}    {r1:+.3f}      {r3:+.3f}      {dom}")

    # ── Passing-design statistics
    n_pass = int(mask.sum())
    n_total = N1 * N3
    print(f"\n  Passing designs: {n_pass:,} / {n_total:,}  ({100*n_pass/n_total:.1f}% of grid)")

    if n_pass == 0:
        print("  No passing designs — see correlation table for guidance.\n")
        return

    i1_p, i3_p = np.where(mask)
    d1_p = d1_arr[i1_p];  d3_p = d3_arr[i3_p]
    T_p  = T_Y[i1_p, i3_p]
    bi   = int(np.argmax(T_p))

    print(f"\n  Thickness range of passing designs:")
    print(f"    d1 (L1 Si₃N₄) : {d1_p.min():.0f} – {d1_p.max():.0f} nm"
          f"   mean {d1_p.mean():.1f} nm   σ {d1_p.std():.1f} nm")
    print(f"    d3 (L3 ZrO₂)  : {d3_p.min():.0f} – {d3_p.max():.0f} nm"
          f"   mean {d3_p.mean():.1f} nm   σ {d3_p.std():.1f} nm")
    print(f"\n  Tvis across passing designs:")
    print(f"    Best : {T_p.max():.2f}%  at  d1 = {d1_p[bi]:.0f} nm, d3 = {d3_p[bi]:.0f} nm")
    print(f"    Mean : {T_p.mean():.2f}%     range [{T_p.min():.2f}%,  {T_p.max():.2f}%]")

    # ── Directional guidance
    _up   = "increases ↑"
    _down = "decreases ↓"
    print(f"\n  Tuning guidance (direction = full-grid trend):")
    print(f"    {'Metric':>5}   thicker d1 (L1) →      thicker d3 (L3) →      spec")
    print("    " + "─" * 66)
    for name, _, lo, hi in metrics:
        r1, r3 = corrs[name]
        spec_str = f"[{lo}, {hi}]" if lo is not None else "—"
        print(f"    {name:>5}   {(_up if r1>0 else _down):<20}   "
              f"{(_up if r3>0 else _down):<20}   {spec_str}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_OUTDIR_CSV, exist_ok=True)
    os.makedirs(_OUTDIR_PNG, exist_ok=True)
    print()
    print("═" * 60)
    print("  Low-E Coating Grid Search")
    print(f"  d1 (L1): {int(D1_RANGE[0])}–{int(D1_RANGE[-1])} nm  │"
          f"  d3 (L3): {int(D3_RANGE[0])}–{int(D3_RANGE[-1])} nm")
    print(f"  Ag L2 values: {AG_THICKNESSES} nm")
    print(f"  Grid size: {len(D1_RANGE)}×{len(D3_RANGE)} = {len(D1_RANGE)*len(D3_RANGE):,} per Ag")
    print(f"  Illuminant: {ILLUMINANT}  │  {CMF_NAME}")
    print("═" * 60)

    # ── Load materials
    print("\nLoading material n,k data…", end=" ", flush=True)
    wl = np.arange(WL_START, WL_END + 0.1, WL_STEP)
    glass_nk = interp_nk(load_nk(GLASS_FILE), wl)
    L1_nk    = interp_nk(load_nk(L1_FILE),    wl)
    L2_nk    = interp_nk(load_nk(L2_FILE),    wl)
    L3_nk    = interp_nk(load_nk(L3_FILE),    wl)
    print("done.")

    # ── Precompute CIE colour weights
    print("Precomputing CIE colour weights…", end=" ", flush=True)
    xyz_w = _build_xyz_weights(wl)                    # (NW, 3), normalised
    XYZ_n = xyz_w.sum(axis=0)                         # white-point XYZ fractions
    print(f"done  (D65 XYZ_n = [{XYZ_n[0]*100:.2f}, {XYZ_n[1]*100:.2f}, {XYZ_n[2]*100:.2f}])")

    # ── Constraint-summary header
    print()
    print("Constraints:")
    print(f"  Fvis < {FVIS_MAX}%  |  Gvis < {GVIS_MAX}%")
    print(f"  Ta* ∈ [{TA_MIN},{TA_MAX}]  Tb* ∈ [{TB_MIN},{TB_MAX}]")
    print(f"  Ga* ∈ [{GA_MIN},{GA_MAX}]  Gb* ∈ [{GB_MIN},{GB_MAX}]")
    print(f"  Fa* ∈ [{FA_MIN},{FA_MAX}]  Fb* ∈ [{FB_MIN},{FB_MAX}]")
    print()

    all_summary_rows = []   # for a combined summary CSV

    for d2 in AG_THICKNESSES:
        tag = f"Ag{d2:.1f}nm".replace(".", "p")
        print(f"── Ag L2 = {d2:.1f} nm ─────────────────────────────────────────")

        # ── Compute T/F/G spectra for the full grid
        t0 = time.perf_counter()
        print(f"  Computing T/F/G for {len(D1_RANGE)}×{len(D3_RANGE)} grid…", end=" ", flush=True)
        T_spec, F_spec, G_spec = calculate_tfg_grid(
            wl, glass_nk, L1_nk, L2_nk, L3_nk,
            D1_RANGE, d2, D3_RANGE, GLASS_MM,
        )
        print(f"done  ({time.perf_counter()-t0:.1f}s)")

        # ── CIE colour for all grid points
        print("  Computing CIE XYZ + Lab…", end=" ", flush=True)
        _, T_lab = compute_color_grid(T_spec, xyz_w, XYZ_n)
        _, F_lab = compute_color_grid(F_spec, xyz_w, XYZ_n)
        _, G_lab = compute_color_grid(G_spec, xyz_w, XYZ_n)
        print("done.")

        # Y (photopic) is just the L* channel mapped back: Y = XYZ[1]*100
        # But easier: recompute Y directly from spectra @ xyz_w[:, 1]
        N1, N3, NW = T_spec.shape
        yw = xyz_w[:, 1]                                # (NW,) — Y-channel weight
        T_Y = (T_spec.reshape(N1*N3, NW) / 100.0) @ yw * 100   # %
        F_Y = (F_spec.reshape(N1*N3, NW) / 100.0) @ yw * 100
        G_Y = (G_spec.reshape(N1*N3, NW) / 100.0) @ yw * 100
        T_Y = T_Y.reshape(N1, N3)
        F_Y = F_Y.reshape(N1, N3)
        G_Y = G_Y.reshape(N1, N3)

        # ── Apply constraints
        mask = check_constraints(T_Y, F_Y, G_Y, T_lab, F_lab, G_lab)
        n_pass = int(mask.sum())
        print(f"  Passing designs: {n_pass} / {N1*N3:,}")

        if n_pass == 0:
            print("  No passing designs — skipping output.\n")
            continue

        # ── Collect passing designs into a DataFrame
        i1_idx, i3_idx = np.where(mask)        # indices into D1_RANGE, D3_RANGE

        rows = []
        for i1, i3 in zip(i1_idx, i3_idx):
            rows.append({
                "d1":  D1_RANGE[i1],
                "d2":  d2,
                "d3":  D3_RANGE[i3],
                "T_Y": round(float(T_Y[i1, i3]), 4),
                "T_L": round(float(T_lab[i1, i3, 0]), 4),
                "T_a": round(float(T_lab[i1, i3, 1]), 4),
                "T_b": round(float(T_lab[i1, i3, 2]), 4),
                "F_Y": round(float(F_Y[i1, i3]), 4),
                "F_L": round(float(F_lab[i1, i3, 0]), 4),
                "F_a": round(float(F_lab[i1, i3, 1]), 4),
                "F_b": round(float(F_lab[i1, i3, 2]), 4),
                "G_Y": round(float(G_Y[i1, i3]), 4),
                "G_L": round(float(G_lab[i1, i3, 0]), 4),
                "G_a": round(float(G_lab[i1, i3, 1]), 4),
                "G_b": round(float(G_lab[i1, i3, 2]), 4),
            })

        df = pd.DataFrame(rows).sort_values("T_Y", ascending=False).reset_index(drop=True)

        # ── Print top N to console
        print(f"\n  Top {min(TOP_N, n_pass)} designs (sorted by Tvis ↓):")
        header = (
            f"  {'d1':>5} {'d3':>5}  "
            f"{'T_Y':>6} {'T_L':>6} {'T_a':>6} {'T_b':>6}  "
            f"{'F_Y':>6} {'F_a':>6} {'F_b':>6}  "
            f"{'G_Y':>6} {'G_a':>6} {'G_b':>6}"
        )
        print(header)
        print("  " + "─" * (len(header) - 2))
        for _, r in df.head(TOP_N).iterrows():
            print(
                f"  {r.d1:5.0f} {r.d3:5.0f}  "
                f"{r.T_Y:6.2f} {r.T_L:6.2f} {r.T_a:6.2f} {r.T_b:6.2f}  "
                f"{r.F_Y:6.2f} {r.F_a:6.2f} {r.F_b:6.2f}  "
                f"{r.G_Y:6.2f} {r.G_a:6.2f} {r.G_b:6.2f}"
            )

        # ── Save CSV
        csv_path = os.path.join(_OUTDIR_CSV, f"lowe_{tag}_passing.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n  Saved : {csv_path}  ({n_pass} rows)")

        # ── Save color-patch PNG
        png_path = os.path.join(_OUTDIR_PNG, f"lowe_{tag}_top{TOP_N}_colors.png")
        save_color_figure(df, d2, TOP_N, png_path)

        # ── Best design
        best = df.iloc[0]

        # ── 3-D Tvis figure (best design highlighted)
        save_3d_tvis_figure(
            D1_RANGE, D3_RANGE, T_Y, mask,
            best["d1"], best["d3"], best["T_Y"], d2,
            os.path.join(_OUTDIR_PNG, f"lowe_{tag}_3d_Tvis.png"),
        )

        # ── 3-D colour-parameter figure (Ta*, Tb*, Fa*, Fb*, Ga*, Gb*)
        save_3d_color_figure(
            D1_RANGE, D3_RANGE, T_lab, F_lab, G_lab, mask, d2,
            os.path.join(_OUTDIR_PNG, f"lowe_{tag}_3d_color.png"),
        )

        # ── Trend analysis report
        print_trends_report(D1_RANGE, D3_RANGE, T_Y, T_lab, F_lab, G_lab, mask, d2)

        all_summary_rows.append(dict(best))

        print()

    # ── Cross-Ag comparison
    if all_summary_rows:
        print("─" * 60)
        print("Best design per Ag thickness:")
        summary = pd.DataFrame(all_summary_rows)
        print(summary[["d2", "d1", "d3", "T_Y", "T_a", "T_b",
                        "F_Y", "F_a", "F_b", "G_Y", "G_a", "G_b"]].to_string(index=False))
        summary.to_csv(os.path.join(_OUTDIR_CSV, "lowe_best_per_Ag.csv"), index=False, float_format="%.4f")
        print(f"\nSaved : {os.path.join(_OUTDIR_CSV, 'lowe_best_per_Ag.csv')}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
