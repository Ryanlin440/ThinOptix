import os
import glob
import numpy as np
import pandas as pd
import colour
from colour import SpectralDistribution, SpectralShape
from colour.colorimetry import sd_to_XYZ
from colour.models import XYZ_to_Lab


def ylab_one_spectrum(wl_nm, values_percent, start_nm=380, end_nm=780, interval_nm=1):
    """Return Y, L*, a*, b* for one spectrum (%).

    The input spectrum values are assumed to be in percent.
    Example: 55 means 55%.
    """

    shape = SpectralShape(start_nm, end_nm, interval_nm)

    cmfs_name = "CIE 1931 2 Degree Standard Observer"
    illuminant_name = "C"

    cmfs = colour.MSDS_CMFS[cmfs_name].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name].copy().align(shape)
    xy_w = colour.CCS_ILLUMINANTS[cmfs_name][illuminant_name]

    wl_nm = np.asarray(wl_nm, dtype=float)
    v = np.asarray(values_percent, dtype=float)

    # Convert percent to fraction.
    v = np.clip(v / 100.0, 0.0, None)

    sd = SpectralDistribution(dict(zip(wl_nm, v)))
    sd = sd.copy().align(shape)

    XYZ = sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant, k=None, method="ASTM E308")
    XYZ_w = sd_to_XYZ(colour.sd_ones(shape), cmfs=cmfs, illuminant=illuminant, k=None, method="ASTM E308")

    XYZ = (XYZ / XYZ_w[1]) * 100.0
    Lab = XYZ_to_Lab(XYZ / 100.0, xy_w)

    return float(XYZ[1]), float(Lab[0]), float(Lab[1]), float(Lab[2])


def load_spectra_from_csv(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load spectra from CSV.

    Required:
        one wavelength column named wavelength, wl, wavelength_nm, nm, or lambda

    New in v4:
        The spectrum columns do NOT need to be named T, F, G.
        The program will calculate all numeric columns after the wavelength column.
        Example columns:
            wavelength, 71-1, 71-3, 71-5
    """

    # utf-8-sig removes the invisible BOM character sometimes added by Excel.
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Clean column names.
    df.columns = [str(c).strip() for c in df.columns]

    # Find wavelength column more flexibly.
    possible_wl_names = ["wavelength", "wl", "wavelength_nm", "nm", "lambda"]
    wl_col = None

    for c in df.columns:
        if c.strip().lower() in possible_wl_names:
            wl_col = c
            break

    if wl_col is None:
        raise ValueError(
            "CSV file must have a wavelength column. "
            f"Accepted names: {possible_wl_names}. Found: {df.columns.tolist()}"
        )

    # Rename wavelength column to standard name.
    if wl_col != "wavelength":
        df = df.rename(columns={wl_col: "wavelength"})

    # Convert wavelength to numeric and remove bad rows.
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df = df.dropna(subset=["wavelength"])

    # All columns except wavelength are candidate spectrum columns.
    candidate_cols = [c for c in df.columns if c != "wavelength"]

    spectrum_cols = []
    for c in candidate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() > 0:
            spectrum_cols.append(c)

    if not spectrum_cols:
        raise ValueError(
            "No spectrum columns found. "
            "The CSV should have wavelength plus at least one numeric data column."
        )

    # Keep wavelength and spectrum columns.
    df = df[["wavelength"] + spectrum_cols].copy()

    return df, spectrum_cols


if __name__ == "__main__":

    # ---- choose spectra file interactively ----
    # Lists all "*TFG.csv" files in the same folder as this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tfg_files = sorted(glob.glob(os.path.join(script_dir, "*TFG.csv")))

    if not tfg_files:
        raise FileNotFoundError(f"No *TFG.csv files found in: {script_dir}")

    print("")
    print("Select spectra file (*TFG.csv):")
    for i, p in enumerate(tfg_files, start=1):
        print(f"  {i}) {os.path.basename(p)}")

    choice = input(f"Your choice (1-{len(tfg_files)}) [default 1]: ").strip() or "1"

    try:
        idx = int(choice)
        if idx < 1 or idx > len(tfg_files):
            idx = 1
    except ValueError:
        idx = 1

    csv_file = tfg_files[idx - 1]

    df, spectrum_cols = load_spectra_from_csv(csv_file)
    wl = df["wavelength"].to_numpy()

    # ---- Output in requested table format, tab-separated and Excel-friendly ----
    print("")
    print(f"File: {os.path.basename(csv_file)}")
    print(f"Spectrum columns found: {', '.join(spectrum_cols)}")
    print("")
    print("color\tY\tL*\ta*\tb*")

    results = []

    for col in spectrum_cols:
        # Drop rows where this spectrum has missing values.
        temp = df[["wavelength", col]].dropna()
        Y, L, a, b = ylab_one_spectrum(temp["wavelength"].to_numpy(), temp[col].to_numpy())

        print(f"{col}\t{Y:.2f}\t{L:.2f}\t{a:.2f}\t{b:.2f}")

        results.append({
            "color": col,
            "Y": round(Y, 2),
            "L*": round(L, 2),
            "a*": round(a, 2),
            "b*": round(b, 2),
        })

    # ---- Also save output as CSV ----
    out_path = os.path.splitext(csv_file)[0] + "_Lab_output.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("")
    print(f"Saved output CSV: {out_path}")
