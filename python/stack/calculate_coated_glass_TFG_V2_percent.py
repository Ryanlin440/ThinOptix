#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_coated_glass_TFG.py

Calculate coated-glass spectra:
  T_percent = transmission through coated glass (%)
  F_percent = film-side reflection (%)  (incident from coating/top side)
  G_percent = glass-side reflection (%) (incident from glass-substrate side)

Default stack definition:
  Glass substrate / L1 / L2 / L3 / air
So for film-side incidence, light sees:
  air -> L3 -> L2 -> L1 -> glass substrate -> air

Input material CSV files need columns:
  wavelength or NK, n, k

Output:
  coated_glass_TFG.csv with wavelength from 300 to 2500 nm, 10 nm interval.
  Output is percent only.

Note:
  The coating layers are calculated coherently by transfer matrix.
  The thick glass substrate is treated incoherently, so it avoids unrealistic
  dense interference fringes from a 3 mm glass substrate.
"""

from __future__ import annotations
from scipy.optimize import minimize
import os
import glob
import numpy as np
import pandas as pd


# -----------------------------
# User defaults
# -----------------------------
DEFAULT_GLASS_FILE = "glass-industr-2018-mat.csv"
DEFAULT_L1_FILE = "L1-Si3N4-mat.csv"
DEFAULT_L2_FILE = "L2-Ag-mat.csv"
DEFAULT_L3_FILE = "L3-top-ZrO2-mat.csv"
DEFAULT_OUTPUT_FILE = "coated_glass_TFG.csv"

WL_START_NM = 300.0
WL_END_NM = 2500.0
WL_STEP_NM = 10.0


# -----------------------------
# File helpers
# -----------------------------
def script_folder() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def find_file(filename: str) -> str:
    """
    Find file in:
      1. current working folder
      2. script folder
    """
    if os.path.isfile(filename):
        return os.path.abspath(filename)

    p2 = os.path.join(script_folder(), filename)
    if os.path.isfile(p2):
        return p2

    raise FileNotFoundError(
        f"Cannot find '{filename}'. Put it in the same folder as this script, "
        f"or run Python from the folder containing the file."
    )


def ask_file(prompt: str, default_name: str) -> str:
    ans = input(f"{prompt} [default: {default_name}]: ").strip()
    name = ans if ans else default_name
    return find_file(name)


def ask_float(prompt: str, default_value: float) -> float:
    ans = input(f"{prompt} [default: {default_value}]: ").strip()
    if ans == "":
        return float(default_value)
    try:
        return float(ans)
    except ValueError:
        print(f"Invalid input; using default {default_value}")
        return float(default_value)


# -----------------------------
# Load n,k data
# -----------------------------
def load_nk_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    if "wavelength" in lower:
        wl_col = lower["wavelength"]
    elif "wavelength_nm" in lower:
        wl_col = lower["wavelength_nm"]
    elif "nk" in lower:
        wl_col = lower["nk"]
    else:
        wl_col = df.columns[0]

    if "n" not in lower or "k" not in lower:
        raise ValueError(f"{path} must have n and k columns. Found columns: {df.columns.tolist()}")

    out = df[[wl_col, lower["n"], lower["k"]]].copy()
    out.columns = ["wavelength", "n", "k"]
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    out = out.sort_values("wavelength")

    # If wavelength appears to be in microns, convert to nm.
    if out["wavelength"].max() < 20:
        out["wavelength"] *= 1000.0

    return out


def interp_nk(df: pd.DataFrame, wl_nm: np.ndarray) -> np.ndarray:
    n = np.interp(wl_nm, df["wavelength"], df["n"])
    k = np.interp(wl_nm, df["wavelength"], df["k"])
    return n - 1j * k  # use n - i*k for absorbing media with this transfer-matrix convention


# -----------------------------
# Transfer matrix at normal incidence
# -----------------------------
def stack_rt(n0: complex, ns: complex, layer_n: list[complex], layer_d_nm: list[float], wl_nm: float):
    """
    Coherent characteristic-matrix calculation at normal incidence.

    Stack:
      medium n0 / layer 1 / layer 2 / ... / medium ns

    Returns:
      R, T, r, t

    T uses admittance correction: T = |t|^2 * Re(ns) / Re(n0)
    """
    M = np.identity(2, dtype=complex)

    for n_layer, d_nm in zip(layer_n, layer_d_nm):
        delta = 2.0 * np.pi * n_layer * d_nm / wl_nm
        eta = n_layer  # normal incidence, non-magnetic material
        m = np.array(
            [
                [np.cos(delta), 1j * np.sin(delta) / eta],
                [1j * eta * np.sin(delta), np.cos(delta)],
            ],
            dtype=complex,
        )
        M = M @ m

    eta0 = n0
    etas = ns

    denom = eta0 * M[0, 0] + eta0 * etas * M[0, 1] + M[1, 0] + etas * M[1, 1]
    r = (eta0 * M[0, 0] + eta0 * etas * M[0, 1] - M[1, 0] - etas * M[1, 1]) / denom
    t = 2.0 * eta0 / denom

    R = float(np.abs(r) ** 2)
    T = float(np.abs(t) ** 2 * (np.real(etas) / np.real(eta0)))

    # Small numerical cleanup
    R = max(0.0, min(5.0, R))
    T = max(0.0, min(5.0, T))
    return R, T, r, t


def bare_interface_rt(n0: complex, ns: complex):
    """Intensity R,T for a bare interface n0 -> ns at normal incidence."""
    r = (n0 - ns) / (n0 + ns)
    t = 2 * n0 / (n0 + ns)
    R = float(np.abs(r) ** 2)
    T = float(np.abs(t) ** 2 * (np.real(ns) / np.real(n0)))
    return R, T


def glass_internal_transmission(n_glass_complex: complex, glass_thickness_mm: float, wl_nm: float) -> float:
    """
    One-pass absorption inside thick glass:
      x = exp(-alpha*d)
      alpha = 4*pi*k/lambda
    """
    k = max(0.0, float(np.imag(n_glass_complex)))
    wl_m = wl_nm * 1e-9
    d_m = glass_thickness_mm * 1e-3
    alpha = 4.0 * np.pi * k / wl_m
    return float(np.exp(-alpha * d_m))

def compute_Tvis(out_df):
    """
    Approximate visible transmission using a photopic weighting.
    Assumes wavelength column is in nm and T_percent is in %.
    """

    wl = out_df["wavelength"].values
    T = out_df["T_percent"].values / 100.0

    # Approximate eye sensitivity curve centered at 555 nm
    V = np.exp(-0.5 * ((wl - 555.0) / 60.0) ** 2)

    return 100.0 * np.sum(T * V) / np.sum(V)
# -----------------------------
# Coated glass T/F/G calculation
# -----------------------------
def calculate_tfg(
    wl_nm: np.ndarray,
    glass_nk: np.ndarray,
    L1_nk: np.ndarray,
    L2_nk: np.ndarray,
    L3_nk: np.ndarray,
    d1_nm: float,
    d2_nm: float,
    d3_nm: float,
    glass_thickness_mm: float,
) -> pd.DataFrame:

    air = 1.0 + 0j

    T_list = []
    F_list = []
    G_list = []

    for i, wl in enumerate(wl_nm):
        ng = glass_nk[i]
        n1 = L1_nk[i]
        n2 = L2_nk[i]
        n3 = L3_nk[i]

        x = glass_internal_transmission(ng, glass_thickness_mm, wl)

        # Film-side incidence:
        # air -> L3 -> L2 -> L1 -> glass
        R_coat_f, T_coat_f, _, _ = stack_rt(
            air, ng,
            layer_n=[n3, n2, n1],
            layer_d_nm=[d3_nm, d2_nm, d1_nm],
            wl_nm=wl,
        )

        # Reflection/transmission from glass side of the coating:
        # glass -> L1 -> L2 -> L3 -> air
        R_coat_back, T_coat_back, _, _ = stack_rt(
            ng, air,
            layer_n=[n1, n2, n3],
            layer_d_nm=[d1_nm, d2_nm, d3_nm],
            wl_nm=wl,
        )

        # Bare back surface of glass: glass -> air
        R_glass_to_air, T_glass_to_air = bare_interface_rt(ng, air)

        # Incoherent combination through thick glass substrate
        denom_f = 1.0 - R_coat_back * R_glass_to_air * (x ** 2)
        if abs(denom_f) < 1e-12:
            denom_f = 1e-12

        T_film_total = T_coat_f * x * T_glass_to_air / denom_f
        R_film_total = R_coat_f + (T_coat_f * R_glass_to_air * (x ** 2) * T_coat_back) / denom_f

        # Glass-side incidence:
        # air -> glass substrate -> L1 -> L2 -> L3 -> air
        R_air_to_glass, T_air_to_glass = bare_interface_rt(air, ng)
        R_glass_back_to_air, T_glass_back_to_air = bare_interface_rt(ng, air)

        # At the coating side, from inside glass to coating/air:
        R_stack_from_glass = R_coat_back
        T_stack_from_glass = T_coat_back

        denom_g = 1.0 - R_glass_back_to_air * R_stack_from_glass * (x ** 2)
        if abs(denom_g) < 1e-12:
            denom_g = 1e-12

        T_glass_total = T_air_to_glass * x * T_stack_from_glass / denom_g
        R_glass_total = R_air_to_glass + (
            T_air_to_glass * R_stack_from_glass * (x ** 2) * T_glass_back_to_air
        ) / denom_g

        T_list.append(T_film_total)
        F_list.append(R_film_total)
        G_list.append(R_glass_total)

    out = pd.DataFrame(
        {
            "wavelength": wl_nm,
            "T_percent": np.clip(T_list, 0, 1) * 100.0,
            "F_percent": np.clip(F_list, 0, 1) * 100.0,
            "G_percent": np.clip(G_list, 0, 1) * 100.0,
        }
    )
    return out
def objective(x, wl, glass_nk, L1_nk, L2_nk, L3_nk, glass_thickness):
    d1, d3 = x

    out = calculate_tfg(
        wl_nm=wl,
        glass_nk=glass_nk,
        L1_nk=L1_nk,
        L2_nk=L2_nk,
        L3_nk=L3_nk,
        d1_nm=float(d1),
        d2_nm=15.0,
        d3_nm=float(d3),
        glass_thickness_mm=glass_thickness,
    )

    Tvis = compute_Tvis(out)

    return -Tvis  # maximize Tvis

def main():
    print("")
    print("=== Coated Glass T/F/G Calculator ===")
    print("Material CSV columns should be: wavelength or NK, n, k")
    print("Layer order is: glass substrate / L1 / L2 / L3 / air")
    print("Film-side reflection F means incident from air/top coating side.")
    print("Glass-side reflection G means incident from glass-substrate side.")
    print("")

    glass_file = ask_file("Glass substrate n,k CSV", DEFAULT_GLASS_FILE)
    L1_file = ask_file("L1 material n,k CSV", DEFAULT_L1_FILE)
    L2_file = ask_file("L2 material n,k CSV", DEFAULT_L2_FILE)
    L3_file = ask_file("L3 material n,k CSV", DEFAULT_L3_FILE)

    print("")
    d1 = ask_float("Enter L1 thickness in nm", 45.0)
    d2 = ask_float("Enter L2 thickness in nm", 15.0)
    d3 = ask_float("Enter L3 thickness in nm", 50.0)
    glass_thickness = ask_float("Enter glass substrate thickness in mm", 3.0)

    out_name = input(f"Output CSV filename [default: {DEFAULT_OUTPUT_FILE}]: ").strip() or DEFAULT_OUTPUT_FILE
    out_path = os.path.join(os.getcwd(), out_name)

    wl = np.arange(WL_START_NM, WL_END_NM + 0.1, WL_STEP_NM)

    glass_df = load_nk_csv(glass_file)
    L1_df = load_nk_csv(L1_file)
    L2_df = load_nk_csv(L2_file)
    L3_df = load_nk_csv(L3_file)

    glass_nk = interp_nk(glass_df, wl)
    L1_nk = interp_nk(L1_df, wl)
    L2_nk = interp_nk(L2_df, wl)
    L3_nk = interp_nk(L3_df, wl)
    
    print("\nOptimizing thicknesses...\n")

    result = minimize(
        objective,
        x0=[60.0, 60.0],  # initial guess
        bounds=[(20.0, 100.0), (20.0, 100.0)],
        args=(
            wl,
            glass_nk,
            L1_nk,
            L2_nk,
            L3_nk,
            glass_thickness,
        ),
        method="L-BFGS-B",
    )

    best_d1, best_d3 = result.x
    best_Tvis = -result.fun

    print(f"Best L1 thickness = {best_d1:.2f} nm")
    print(f"Best L3 thickness = {best_d3:.2f} nm")
    print(f"Best Tvis = {best_Tvis:.2f}%")
    out = calculate_tfg(
        wl_nm=wl,
        glass_nk=glass_nk,
        L1_nk=L1_nk,
        L2_nk=L2_nk,
        L3_nk=L3_nk,
        d1_nm=float(best_d1),
        d2_nm=15.0,
        d3_nm=float(best_d3),
        glass_thickness_mm=glass_thickness,
    )
    # Add absorption checks in percent. For absorbing coatings, T + R is usually < 100%.
    # out["A_from_film_percent"] = 100.0 - out["T_percent"] - out["F_percent"]
    # out["A_from_glass_percent"] = 100.0 - out["T_percent"] - out["G_percent"]

    out.to_csv(out_path, index=False)

    print("")
    print("Calculation finished.")
    print(f"Output saved to: {out_path}")
    print("")
    print("First 5 rows:")
    print(out.head().to_string(index=False))
    print("")
    print("Note: output values are percentages. For absorbing Ag films, T + R is normally less than 100%; absorption = 100 - T - R.")


if __name__ == "__main__":
    main()
