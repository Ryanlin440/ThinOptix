# ThinOptix

ThinOptix is a Python workflow for modelling and optimising low-emissivity (Low-E) coated glass stacks. The project evaluates spectral transmission and reflectance for a multilayer stack using a transfer-matrix approach, then computes visible performance metrics such as photopic transmittance and CIE Lab colour coordinates.

The core stack model is:

air | L3 | L2 (Ag) | L1 | glass | air

The project is used to:

- compute T / F / G spectra for a coated-glass design
- search for thickness combinations that meet transmission and colour constraints
- compare different silver layer thicknesses and material datasets
- generate plots and summary CSV files for candidate designs

---

## Project purpose

This repository is built around optical coating design rather than a general-purpose app. It is intended for research and engineering exploration of low-E coating stacks where the design goal is often:

- maximise visible transmission (Tvis)
- keep visible reflectance on both sides under limits
- maintain colour coordinates within acceptable ranges
- evaluate multiple Ag thicknesses and material candidates

The code implements a thin-film optical model and uses numerical search over thickness combinations to identify feasible designs.

---

## Main files

### Root scripts

- `lowe_design.py` — primary exhaustive grid search for low-E coating designs.
  - Reads material files and experiment files.
  - Evaluates every L1/L3 thickness combination for selected Ag thicknesses.
  - Applies design constraints and saves passing designs.
  - Generates design-space and 3D visualisation plots.

- `optical_coating_tool.py` — general optical coating utility and optimizer.
  - Computes T/F/G spectra for a single stack.
  - Converts spectra to CIE Lab values.
  - Implements thickness optimization for L1/L3 with Ag fixed at 15 nm.
  - Includes optional IGU / NFRC-style performance calculations.

### Scripts folder

- `scripts/thickness_sweep.py` — sweeps one layer thickness at a time to study sensitivity.
- `scripts/calculate_coated_glass_TFG_V2_percent.py` — older or alternative coated-glass calculation script.

### Colour analysis utilities

- `color_analysis/colorLAB_TFG_csv_v4_table_output.py` — processes TFG/Lab CSV outputs to build table-based summaries and plots.

---

## Repository structure

```text
ThinOptix/
├── README.md
├── lowe_design.py
├── optical_coating_tool.py
├── data/
│   ├── experiments/
│   └── materials/
├── output/
│   ├── csv/
│   └── png/
│       ├── 3d_surfaces/
│       ├── color_patches/
│       └── design_space/
├── scripts/
│   ├── calculate_coated_glass_TFG_V2_percent.py
│   ├── thickness_sweep.py
│   └── ...
├── color_analysis/
│   ├── colorLAB_TFG_csv_v4_table_output.py
│   ├── *.csv
│   └── ...
└── ...
```

---

## Data format

### Material files

Material CSVs live in `data/materials/` and should contain at least:

- wavelength column (or `NK`)
- `n` column
- `k` column

Examples include files such as:

- `glass-industr-2018-mat.csv`
- `L1-Si3N4-mat.csv`
- `L2-Ag-mat.csv`
- `L3-top-ZrO2-mat.csv`

The loader code accepts wavelength values in either nm or microns and converts if needed.

### Experiment files

Experiment CSVs in `data/experiments/` represent optical data for the Ag layer or similar material variants used in the grid-search workflow. These files are loaded by `lowe_design.py` and used as the L2 layer in the search.

---

## Setup

This project uses Python 3 and the following packages:

```bash
pip install numpy pandas scipy matplotlib colour
```

A typical virtual environment setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy matplotlib colour
```

---

## Typical workflows

### 1) Run the full Low-E design search

From the project root:

```bash
python lowe_design.py
```

This script:

- loads material files
- evaluates a 1 nm thickness grid for L1 and L3
- evaluates Ag thicknesses at 10, 12.5 and 15 nm
- checks the optical constraints
- writes passing designs to `output/csv/`
- produces design-space and colour patch plots in `output/png/`

Expected output files include patterns such as:

- `output/csv/lowe_Ag10p0nm_passing.csv`
- `output/csv/best_designs_all_experiments.csv`
- `output/png/design_space/*.png`
- `output/png/color_patches/*.png`

### 2) Run the interactive optical coating tool

```bash
python optical_coating_tool.py
```

This script is more interactive and can:

- evaluate one specific stack (d1, d2, d3)
- compute spectral T/F/G data and CIE Lab values
- optimise d1 and d3 for a fixed Ag layer thickness
- optionally run an IGU / NFRC-style performance estimate

### 3) Sweep a single layer thickness

```bash
python scripts/thickness_sweep.py
```

This is useful when you want to see how changing one layer affects T/F/G colour and Tvis while holding the other layers fixed.

### 4) Compute a coated glass spectrum directly

```bash
python scripts/calculate_coated_glass_TFG_V2_percent.py
```

This older script asks for file names and thicknesses interactively and writes a spectral CSV containing percent transmission and reflection values.

---

## What the model calculates

The core calculations model a stack normal-incidence transfer matrix with a thick glass substrate treated incoherently to avoid unrealistic fringe effects from the substrate thickness.

For each wavelength:

- T = transmitted fraction through the coated glass
- F = film-side reflectance
- G = glass-side reflectance

Those spectra are then integrated against CIE photopic weighting to estimate visible performance metrics such as:

- Tvis
- Y, L*, a*, b* in CIE Lab space

Design checks are applied using the thresholds defined in `lowe_design.py`, including visible reflectance limits and colour-coordinate constraints.

---

## Output files

### CSV outputs

The project writes output CSV files in `output/csv/`, including:

- passing designs from each experiment and Ag thickness
- best design summary across experiments
- spectral T/F/G results
- Lab colour summaries

### Plot outputs

The project writes PNG files under `output/png/`:

- `3d_surfaces/` for 3D Tvis and colour surfaces
- `color_patches/` for candidate design colour patches
- `design_space/` for 2D thickness-search maps

---

## Notes and practical guidance

- The project is highly parameterised, so the main knobs are material CSVs, the Ag thickness list, and the search ranges defined near the top of `lowe_design.py`.
- For experiments and material comparisons, the workflow is easiest to use when all input data are placed under `data/materials/` and `data/experiments/` with consistent wavelength and optical property formatting.
- If you are exploring a new coating, start with `optical_coating_tool.py` for a single-stack run, then switch to `lowe_design.py` for the exhaustive design sweep.

---

## Summary

ThinOptix is a research-oriented optical design toolkit for low-E coatings. It combines thin-film transfer-matrix physics, photometric colour calculations, and optimization/search routines to identify coating stacks that satisfy visible-light transmission and reflectance requirements.
