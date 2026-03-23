# NEON Data Process

This repository processes NEON site-level hourly soil respiration (SR) flux data with a Bayesian state-space model in PyMC.
The HPC workflow is designed for one independent fit per site-year, which maps naturally to SLURM job arrays.

## Input Data

Expected file layout:

`./data/${sitecode}/${sitecode}_${YYYY}_hourly_gC_allpos.csv`

Each CSV should contain at least:
- `startDateTime`
- `horizontalPosition` (plot/chamber ID)
- `flux_gC`

## Repository Structure

- `main/data_io.py`
  - `load_neon_site_year(...)`: load and preprocess one site-year (recommended for HPC jobs).
  - `load_neon_sitedata(...)`: load all available years for one site.
  - Creates a full hourly x plot grid and keeps missing plot-hours as `NaN`.
- `main/state_space_model.py`
  - Defines the PyMC state-space model.
  - Uses 4 chains and supports `cores` to parallelize chain sampling on CPU.
- `Data_process.py`
  - CLI entry point for fitting one site-year and saving posterior outputs.
- `scripts/submit_array.slurm`
  - SLURM array script to run many site-year fits in parallel.
- `task_list.example.txt`
  - Example task list (`SITE YEAR` per line) for SLURM arrays.
- `environment.yml`
  - Conda environment for reproducible setup.
- `example_code/`
  - Reference notebook/script used during model development.

## Model + Missing Data Strategy

For each site-year:
- Data are cleaned and aggregated to hourly values by plot.
- A complete hourly timeline is built across all observed plots.
- Missing flux values remain missing and are excluded from likelihood evaluation.
- The latent site process combines:
  - seasonal harmonics (annual + second harmonic),
  - AR(1) temporal dynamics,
  - plot random effects,
  - hour-of-day effects.

This setup handles irregular missingness across both time and plots without dropping the full time axis.

## Local Run (Single Site-Year)

From repo root:

```bash
python Data_process.py --site JERC --year 2020 --data-dir data --output-dir results --seed 42 --cores 4
```

Common options:
- `--site`: site code (for example `JERC`)
- `--year`: year to fit
- `--data-dir`: base data directory (default: `data`)
- `--output-dir`: base output directory (default: `results`)
- `--seed`: random seed
- `--cores`: CPU workers for PyMC chains (default: `min(4, os.cpu_count())`)

## HPC Run (SLURM Job Array)

1. Create task file:
   - Copy `task_list.example.txt` to `task_list.txt`
   - Keep one task per line: `SITE YEAR`
2. Edit `scripts/submit_array.slurm`:
   - Set `#SBATCH --array=1-N` to match number of lines in `task_list.txt`
   - Adjust `--time`, `--mem`, and other site-specific directives
   - Confirm your environment activation lines (module/conda/micromamba) are correct for your cluster
3. Submit:

```bash
sbatch scripts/submit_array.slurm
```

The script maps each `SLURM_ARRAY_TASK_ID` to one line in `task_list.txt` and runs:

`python Data_process.py --site ... --year ... --data-dir ... --output-dir ... --seed ... --cores 4`

## Threading and Performance Notes

To avoid BLAS oversubscription when running multi-chain sampling:
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

These are set in `scripts/submit_array.slurm` by default (with override support).

Primary scalability strategy:
- Parallelize across site-years with SLURM arrays.
- Parallelize chains inside each task with `--cpus-per-task=4` and `--cores 4`.

## Outputs

Each run writes to:

`results/${site}/${year}/`

Main output files:
- `${site}_${year}_posterior.nc` (full ArviZ InferenceData)
- `${site}_${year}_bayesian_site_hourly_soil_co2_estimate_AR1.csv`
- `${site}_${year}_bayesian_plot_offsets_AR1.csv`
- `${site}_${year}_bayesian_hour_effects_AR1.csv`
- `${site}_${year}_bayesian_scalar_parameters_AR1.csv`
- `${site}_${year}_arviz_summary.csv`

## Environment Setup

Create and activate environment:

```bash
conda env create -f environment.yml
conda activate pymc_py311 (on puma)
```

## Quick Checklist Before Large Submission

- Test one job first (`SITE YEAR`) and check runtime/memory.
- Verify convergence metrics (`r_hat`, ESS) in `${site}_${year}_arviz_summary.csv`.
- Scale `#SBATCH --array`, `--time`, and `--mem` after profiling one or two representative years.


