# Variable-diffusivity PINN dust reconstruction

This repository contains code and data products associated with the manuscript
on variable-diffusivity physics-informed neural networks for sparse global dust
deposition reconstruction.

The study uses a spherical advection-diffusion constraint and a dual-output PINN
to estimate both the reconstructed dust deposition field `u(x)` and an effective
spatially varying diffusivity field `D(x)`. The empirical case studies include
Holocene and Last Glacial Maximum dust-deposition reconstructions.

## Repository structure

```text
Code/                 Python scripts and notebooks for preprocessing, training,
                      post-processing, diagnostics, and figure generation
Data/
  original_data/      Input data files used by the workflow
  processed_data/     Processed data tables used by diagnostics and examples
  model_results/      Model outputs and figure-source data included in this release
  trained_models/     Model checkpoints, if included in a separate release
Figures/              Generated figure files, if produced by local scripts
examples/             Minimal reproduction example for manuscript Fig. 6
```

## Main code files

- `Code/preprocess.ipynb`: preprocessing of empirical, simulated, wind, and grid data.
- `Code/pinn_simulated.ipynb`: PINN experiments on simulated benchmark fields.
- `Code/pinn_empirical.ipynb`: PINN experiments on empirical Holocene and LGM data.
- `Code/sensitivity_pinn_empirical.ipynb`: PDE-weight sensitivity experiments.
- `Code/kriging.ipynb`: kriging baseline processing and comparison.
- `Code/compute_pde_residuals.py`: PDE residual and boundary-condition diagnostics.
- `Code/null_test_random_D.py`: random-D null-test diagnostics.
- `Code/correlation_D_wind.py`: inferred `D(x)` and wind-profile diagnostics.
- `Code/generate_paper_figures.py`: manuscript figure assembly from stored outputs.
- `Code/style_jcp.py`: shared plotting style and figure-saving utilities.

## Data products included in this release

This release includes the data products needed to run the minimal reproduction
example for manuscript Fig. 6:

```text
Data/model_results/df_pinn_empirical_Holocene_global_grid.csv
Data/model_results/df_pinn_empirical_LGM_global_grid.csv
Data/processed_data/df_wind.csv
```

The `Data/original_data/` folder contains source data files used by the broader
workflow. Full training outputs, model checkpoints, ablation outputs, and some
intermediate diagnostic tables may be distributed separately if they are not
included in a specific repository release. Public source datasets are described
and cited in the manuscript.

## Environment

Install the Python dependencies listed in `requirements.txt`. A clean virtual
environment or conda environment is recommended.

```bash
pip install -r requirements.txt
```

Some geospatial packages, especially `geopandas` and `cartopy`, may be easier to
install through conda on some systems.

## Minimal reproduction example

Run the verified minimal example from the repository root:

```bash
python examples/minimal_reproduction.py
```

This regenerates the two source panels used in manuscript Fig. 6:

```text
examples/output/CORRELATION_ZONAL_PROFILE_D_WIND.png
examples/output/CORRELATION_SCATTER_D_WIND.png
```

The example imports the original plotting functions from
`Code/correlation_D_wind.py`, so the generated figures use the same filtering,
correlation calculation, and plotting style as the manuscript figures.

## Full workflow scripts

The notebooks and scripts in `Code/` document the broader preprocessing,
training, post-processing, and diagnostic workflow used for the manuscript.
Some scripts require additional files that are not part of the minimal example,
such as full training outputs, ablation outputs, model checkpoints, or
intermediate diagnostic tables. Before running those scripts, check that the
corresponding files in `Data/`, `Data/trained_models/`, or `Ablation_Result/`
are available.

## Notes

- The repository is organized for manuscript code review and lightweight
  reproducibility checking.
- The scripts use relative paths based on the repository root and the `Code/`
  directory.
- The verified minimal example writes figures to `examples/output/`.
- Main figure scripts may write figures to `Figures/`.
- Random-seed and checkpoint-dependent results require the corresponding stored
  model outputs or checkpoints.

## Citation

If using this repository, please cite the associated manuscript once published.
