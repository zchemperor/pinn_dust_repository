# PMIP4-Informed Wind-PINN for Sparse LGM Dust Deposition Reconstruction

This repository contains the code, the raw input data, and the supplementary evidence tables for the PMIP4-informed Wind-PINN analysis of sparse Last Glacial Maximum (LGM) dust deposition records on the sphere.

The study reconstructs log dust deposition from 317 DIRTMAP LGM records while using annual 850 hPa horizontal winds from PMIP4-CMIP6 LGM simulations as a mathematical regularizing prior. The primary model, **Wind-PINN**, couples the observations to a spherical differential expression of standardized log-deposition with a constant, learned numerical smoothing coefficient. The repository also implements matched statistical, neural, and numerical baselines.

> **Scope and interpretation.** PMIP4 winds are climate-model output, not observations of LGM transport and not a validated dust-transport mechanism. They enter a penalty term only; they are never additional deposition measurements. The results support conditional, retrospective comparisons, not extrapolation skill, a validated dust cycle, an identified atmospheric diffusivity, or calibrated predictive uncertainty.

## Main findings

Across six retrospective spatial holdouts (three regional blocks and three broad leave-region tests), the differential penalty lowered regional error relative to the matched No-PDE network in all three regional tests, under both the mean of five per-seed metrics and the metric of the five-seed mean prediction. This comparison tests the penalty as a whole; the matched zero-wind network tests the contribution of the wind terms.

Absolute reconstruction skill remained limited. Primary-forcing Wind-PINN ensembles had negative R² in all six tests, and in every test an available ensemble or deterministic comparator had a lower point-estimate RMSE. Adding wind did not produce a uniform benefit: relative to zero wind, RMSE was lower in North America and northern extrapolation, but zero-wind ensembles performed better in the tropical regional holdout and in the southern and tropical extrapolation tests. In the tropical regional test the two five-seed summaries rank the models differently, and both are reported.

A restricted spatial-coefficient variant (Relative-D) offered no consistent gain across the six tests. The PDE-constrained MAP baseline remained competitive in several tests. All comparisons are reported after a fixed budget of 150,000 Adam updates with five seeds; finite training does not establish convergence. Machine-readable evidence, including the per-record source audit, is in the supplementary tables described below.

## Repository layout

```text
Code/                 Preprocessing, training, validation, analysis, and export scripts
Data/original_data/   Original study inputs, including the LGM and Holocene DIRTMAP tables
Data/metadata/        Per-record source audit: candidate links, row mapping, source inventory
Data/model_results/   Audit outputs: identity status, proximity groups, candidate subsets,
                      coefficient and residual diagnostics
manuscript/           Supplementary methods source and the 24 supplementary tables (S1-S24)
scripts/              Builders for the per-record identity tables
```

Some audit scripts name an internal protocol note under `docs/` as their input. Those working notes are not distributed with this repository; the corresponding rules are stated in the supplementary methods and in `manuscript/supplementary_tables/README.md`.

`Data/external_raw/` (raw PMIP4-CMIP6 NetCDF), `Data/processed_data/`, `Figures/`, and `Data/trained_models/` are **not** part of this release. The raw forcing files are large model outputs and must be downloaded from the ESGF distributed data nodes; `Data/external_raw/` covers AWI-ESM-1-1-LR (primary forcing and temporal robustness), MPI-ESM1-2-LR, and MIROC-ES2L.

## Supplementary evidence for the submitted manuscript

`manuscript/supplementary_methods.tex` is the source of the supplementary methods document (sections S1-S6). `manuscript/supplementary_tables/` holds the 24 tables referred to in the manuscript, together with a README and a manifest containing row counts, column names, and SHA-256 checksums. The file names below are stable; the S-numbers are the ones used in the manuscript.

| Supplementary section | Content | Tables |
|---|---|---|
| S1 | Input rows, units, and target definition | S6 |
| S2 | Source-candidate rules and coverage | S1, S2, S14, S15 |
| S3 | Single-candidate flux-ratio diagnostic | S3 |
| S4 | Historical ages and ambiguous cases | S4, S5 |
| S5 | Record dependence and candidate-subset evaluation | S7-S13, S16 |
| S6 | Coefficient optimization and independent residual evaluation | S17-S24 |

Two conventions are easy to conflate and are kept separate throughout:

- **S2 coverage** counts distinct input rows with an exact-coordinate *and* flux-agreement candidate (93 rows), or with a coordinate-window candidate (98 rows).
- **S3** uses a different population: one distinct core-identifier/LGM-value candidate at an exact coordinate, without requiring flux agreement (118 pairs).

Source-candidate fields describe entries in the historical source tables. They are **not** verified identities, ages, archives, or particle-size fractions of the current 317 inputs, and no candidate metadata was transferred to a current row. Every input status remains unresolved with respect to row lineage.

## Reproducing the audit

The scripts under `Code/` and `scripts/` are the exact archived sources used to produce these tables. They were written against the full local project tree and expect the project-relative paths shown above; where an input lies outside this release (for example, cached predictions under `Data/model_results/*/predictions/` or fitted checkpoints under `Data/trained_models/`), the corresponding script documents that path and will report it as missing. They are provided for inspection and for reconstruction of the tables from the included inputs, not as a turnkey pipeline.

Reading the supplementary tables requires nothing beyond a CSV reader. `manuscript/supplementary_tables/manifest.json` lists the SHA-256 of every table, and `manuscript/supplementary_tables/README.md` defines the identifiers, units, missing-value conventions, and the sign convention of each reported contrast.

## License and data provenance

The LGM input table derives from the DIRTMAP compilation distributed with the upstream study; the DIRTMAP version 1 and 2 source tables are archived by NOAA/NCEI, and Supporting Information Table S1 of Lambert et al. (2015) was recovered separately. Those third-party publisher files and database packages are not redistributed here; the supplementary README records the citations and source locations. See `LICENSE` for the terms of this repository.
