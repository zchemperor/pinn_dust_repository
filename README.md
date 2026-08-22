# PMIP4-Informed Wind-PINN for Sparse LGM Dust Deposition Reconstruction

This repository contains the code and raw input data needed to reproduce the PMIP4-informed Wind-PINN analysis of sparse Last Glacial Maximum (LGM) dust deposition records on the sphere.

The study reconstructs log dust deposition from 317 DIRTMAP LGM records while using annual 850 hPa horizontal winds from PMIP4-CMIP6 LGM simulations as a physics-based regularizing prior. The primary model, **Wind-PINN**, couples the observations to a steady spherical advection-diffusion residual with a constant, learned effective diffusivity. The repository also implements matched statistical, neural, and numerical baselines.

> **Scope and interpretation.** PMIP4 winds are climate-model output, not observations of LGM winds and not a validated dust-transport mechanism. They are used only as a regularizing input. The results support regional and conditional comparisons, not universal extrapolation skill or a direct physical reconstruction of the LGM dust cycle.

## Main findings

The frozen spatial-validation experiments show that Wind-PINN outperforms its matched No-PDE MLP in all three outer confirmation blocks. Its mean held-out RMSE is lower than GP/RBF in North America and East Asia, with a modest margin in East Asia. GP/RBF performs better in tropical Atlantic-Africa, and strict large-region stress tests show negative mean R-squared values. PDE-MAP also performs competitively in some regions. Thus, neural physics regularization is not universally preferable and the reported method has a clear extrapolation boundary.

## Repository layout

```text
Code/                         Reproducible preprocessing, training, validation, and analysis scripts
Data/original_data/           Original study inputs, including DIRTMAP tables
Data/external_raw/            Raw PMIP4-CMIP6 LGM ua/va NetCDF files (20 files)
Figures/                      Locally generated analysis figures (not intended for upload)
Data/processed_data/          Locally generated preprocessing products (not intended for upload)
Data/model_results/           Locally generated split files, predictions, metrics, and analyses (not intended for upload)
Data/trained_models/          Locally generated model checkpoints (not intended for upload)
```

`Data/external_raw/` contains AWI-ESM-1-1-LR (used as primary forcing and for temporal robustness), MPI-ESM1-2-LR (for model‑forcing sensitivity), and MIROC-ES2L (for model‑forcing sensitivity). These raw NetCDF files are large (individual model outputs can reach tens of gigabytes) and need to be downloaded from the ESGF (Earth System Grid Federation) distributed data nodes.
