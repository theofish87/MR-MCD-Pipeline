# MR-MCD-Pipeline
NV Centre Magnetometry - MC Dropout Uncertainty Quantification


# MR-MCD-Pipeline

**NV Centre Magnetometry — MC Dropout Uncertainty Quantification**

**Author:** Yue Yu, Shirsopratim Chattopadhyay, TU Dresden / ct.qmat


---

## Overview

This repository contains the full pipeline for validating and calibrating
MC Dropout-based uncertainty quantification in NV centre magnetometry reconstruction.

The pipeline is based on the uPINN model by Dubois et al. (2022), extended with
Monte Carlo Dropout and Fixed BatchNorm for uncertainty estimation.

**Reference:**
> A.E.E. Dubois et al., *Phys. Rev. Applied* **18**, 064076 (2022)
> "Untrained Physically Informed Neural Network for Image Reconstruction of Magnetic Field Sources"

---

## Pipeline Overview

```
Step 1 — Ubermag Simulation (Local PC)
    ↓
    Generate synthetic magnetic domain data with known ground truth
    Output: Bz_NV_*.npy (input field), Mz_true_*.npy (ground truth)
    Upload to Google Drive

Step 2 — MC Model Reconstruction (Google Colab)
    ↓
    Feed Bz_NV into MC Dropout uPINN
    T = 100 stochastic forward passes
    Output: M_predicted (mean), M_std (uncertainty), M_true_converted

Step 3 — Calibration Analysis (Google Colab)
    ↓
    z-score, Reliability Diagram, ECE
    Three masks: No mask / With mask / Background only
    Reference: Chuan Guo et al., ICML 2017
```

---

## Repository Structure

```
MR-MCD-Pipeline/
│
├── ubermag/                      # Step 1: Micromagnetic simulation scripts
│   ├── Neel_Domainwall.ipynb     # Néel domain wall simulation
│   └── C1_Skyrmion.ipynb         # Skyrmion simulation
│
├── model/                        # MC Dropout uPINN model code
│   ├── 2D/
│   │   ├── Magnetisation/
│   │   │   ├── Generator.py      # CNN architecture + Dropout2d(p=0.1)
│   │   │   ├── Train.py          # MC Dropout inference (T=100)
│   │   │   ├── Propagator.py     # FFT-based forward model
│   │   │   └── ...
│   │   └── ...
│   └── requirements.txt
│
├── reconstruction/               # Step 2: Reconstruction notebooks (Google Colab)
│   ├── Neel_DW.ipynb             # Néel domain wall reconstruction
│   └── Rec_Skyrmion.ipynb        # Skyrmion reconstruction
│
├── calibration/                  # Step 3: Calibration analysis (Google Colab)
│   └── calibration_analysis.ipynb
│
└── README.md
```

---

## Our Modifications to the Original uPINN

| File | Modification |
|------|-------------|
| `Generator.py` | Added `Dropout2d(p=0.1)` after conv4 |
| `Train.py` | Added `enable_mc_dropout()` — freezes BatchNorm, keeps Dropout active |
| `Train.py` | Added MC Dropout inference loop (T=100 forward passes) |
| `Train.py` | `extract_results()` now outputs MC mean, std, and variance |

---

## Simulation Cases

| Case | Type | Domain Wall Width | Status |
|------|------|-------------------|--------|
| Néel Domain Wall | Single centered Néel DW | δ ≈ 13.6 nm | ✅ Complete |
| Skyrmion | Single centered Néel skyrmion | r₀ = 80 nm | ✅ Complete |

**Material parameters (Co/Pt system):**
- Ms = 5.8×10⁵ A/m
- A = 1.5×10⁻¹¹ J/m
- D = 3.0×10⁻³ J/m² (interfacial DMI, Néel type)
- K = 0.8×10⁶ J/m³ (uniaxial anisotropy, easy OOP)

---

## Calibration Analysis

Based on: Chuan Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

### Method
1. **Sigma Floor** — `σ = max(σ_min, σ*)` to prevent division by zero
2. **z-score** — `z = (M_predicted - M_true) / σ` per pixel
3. **Three masks:**
   - Mask A: No mask (all pixels)
   - Mask B: With mask — `|M_true| > threshold` ← most important
   - Mask C: Background only
4. **Reliability Diagram** — σ_group vs ε_group (ideal: diagonal y=x)
5. **ECE** — `Σ (N_b/N) |ε_group - σ_group|`

### Current Results (Néel DW)

| Metric | Value |
|--------|-------|
| M_std mean | 0.0039 μB nm² |
| Actual error (ε) | ~3.6 μB nm² |
| Ratio σ/ε | ~0.002 (model severely overconfident) |
| ECE | 3.62 |

---

## Data Storage

Simulation data (`.npy` files) are stored on Google Drive, not in this repository.

```
Google Drive: Colab Notebooks/Ubermag_Model_Verification/
├── ubermag_data/          ← Simulation outputs (uploaded from local PC)
│   ├── neel_dw/
│   └── skyrmion/
├── uberesult/             ← Reconstruction results
│   ├── neel_dw/
│   └── skyrmion/
└── calibration/           ← Calibration analysis results
    ├── neel_dw/
    ├── skyrmion/
    └── pooled/
```

---

## Requirements

```
Python 3.8+
torch
numpy
matplotlib
scipy
ubermag (for Step 1, local PC only)
discretisedfield
micromagneticmodel
oommfc
```

Install dependencies:
```bash
pip install -r model/requirements.txt
```

---

## References

1. A.E.E. Dubois et al., *Phys. Rev. Applied* **18**, 064076 (2022)
2. Y. Gal & Z. Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
3. C. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
4. J.-P. Tetienne et al., *Nat. Commun.* **6**, 6733 (2015)
