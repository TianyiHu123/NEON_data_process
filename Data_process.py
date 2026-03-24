"""
CLI driver for NEON site-year soil respiration state-space fits (PyMC).
Use on a single machine or under SLURM (one process per site-year).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from main.data_io import load_neon_site_year
from main.state_space_model import State_space_model


def _datetime_series_aligned_to_mu(data_all: pd.DataFrame) -> pd.Series:
    """One timestamp per time_idx, same order as model time dimension."""
    u = data_all.sort_values("time_idx").drop_duplicates("time_idx", keep="first")
    return u.sort_values("time_idx")["datetime_hour"]


def save_outputs(
    idata: az.InferenceData,
    data_all: pd.DataFrame,
    all_plots: list,
    site: str,
    year: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_path = out_dir / f"{site}_{year}_posterior.nc"
    idata.to_netcdf(str(nc_path))
    print(f"Wrote {nc_path}")

    datetime_hour = _datetime_series_aligned_to_mu(data_all).values

    mu_post = idata.posterior["mu_t"].mean(dim=("chain", "draw")).values
    hdi_mu = az.hdi(idata.posterior["mu_t"], hdi_prob=0.95)
    hdi_arr = np.asarray(hdi_mu)
    if hdi_arr.ndim == 2 and hdi_arr.shape[-1] == 2:
        mu_low, mu_high = hdi_arr[:, 0], hdi_arr[:, 1]
    elif hdi_arr.ndim == 2 and hdi_arr.shape[0] == 2:
        mu_low, mu_high = hdi_arr[0], hdi_arr[1]
    else:
        stacked = np.asarray(hdi_mu).reshape(-1, 2)
        mu_low, mu_high = stacked[:, 0], stacked[:, 1]

    if len(datetime_hour) != len(mu_post):
        raise ValueError(
            f"Time axis mismatch: datetime_hour {len(datetime_hour)} vs mu_t {len(mu_post)}"
        )

    site_result = pd.DataFrame(
        {
            "datetime_hour": datetime_hour,
            "mu_post": mu_post,
            "mu_low": mu_low,
            "mu_high": mu_high,
        }
    )
    site_result["site_flux_est"] = np.exp(site_result["mu_post"])
    site_result["site_flux_low"] = np.exp(site_result["mu_low"])
    site_result["site_flux_high"] = np.exp(site_result["mu_high"])

    hourly_path = out_dir / f"{site}_{year}_bayesian_site_hourly_soil_co2_estimate_AR1.csv"
    site_result.to_csv(hourly_path, index=False)
    print(f"Wrote {hourly_path}")

    plot_offset_post = idata.posterior["plot_offset"].mean(dim=("chain", "draw")).values
    plot_offset_table = pd.DataFrame(
        {"plotID": all_plots, "posterior_plot_offset": plot_offset_post}
    )
    plot_path = out_dir / f"{site}_{year}_bayesian_plot_offsets_AR1.csv"
    plot_offset_table.to_csv(plot_path, index=False)
    print(f"Wrote {plot_path}")

    hour_effect_post = idata.posterior["hour_effect"].mean(dim=("chain", "draw")).values
    hour_effect_table = pd.DataFrame(
        {"hour_of_day": np.arange(24), "posterior_hour_effect": hour_effect_post}
    )
    hour_path = out_dir / f"{site}_{year}_bayesian_hour_effects_AR1.csv"
    hour_effect_table.to_csv(hour_path, index=False)
    print(f"Wrote {hour_path}")

    rho_post = idata.posterior["rho"].mean(dim=("chain", "draw")).values.item()
    sigma_proc_post = idata.posterior["sigma_proc"].mean(dim=("chain", "draw")).values.item()
    sigma_plot_post = idata.posterior["sigma_plot"].mean(dim=("chain", "draw")).values.item()
    sigma_hour_post = idata.posterior["sigma_hour"].mean(dim=("chain", "draw")).values.item()

    scalar_summary = pd.DataFrame(
        {
            "parameter": ["rho", "sigma_proc", "sigma_plot", "sigma_hour"],
            "posterior_mean": [rho_post, sigma_proc_post, sigma_plot_post, sigma_hour_post],
        }
    )
    scalar_path = out_dir / f"{site}_{year}_bayesian_scalar_parameters_AR1.csv"
    scalar_summary.to_csv(scalar_path, index=False)
    print(f"Wrote {scalar_path}")

    summary = az.summary(
        idata,
        var_names=[
            "beta0",
            "beta_sin_1",
            "beta_cos_1",
            "beta_sin_2",
            "beta_cos_2",
            "rho",
            "sigma_proc",
            "sigma_plot",
            "sigma_hour",
            "sigma_obs",
        ],
        round_to=3,
    )
    summary_path = out_dir / f"{site}_{year}_arviz_summary.csv"
    summary.to_csv(summary_path)
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit NEON hourly soil CO2 flux state-space model for one site-year."
    )
    parser.add_argument("--site", required=True, help="NEON site code, e.g. JERC")
    parser.add_argument("--year", type=int, required=True, help="Calendar year, e.g. 2020")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing ./<site>/<site>_<year>_hourly_gC_allpos.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory; writes to <output-dir>/<site>/<year>/",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for PyMC sampling")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Parallel workers for MCMC chains (default: min(4, CPU count))",
    )
    args = parser.parse_args()

    neon_data = load_neon_site_year(args.site, args.year, args.data_dir)
    data_all = neon_data[args.year]
    data_obs = neon_data[f"obs_{args.year}"]

    all_plots = sorted(data_all["horizontalPosition"].dropna().unique().tolist())

    idata = State_space_model(data_all, data_obs, args.seed, cores=args.cores)

    out_dir = Path(args.output_dir) / args.site / str(args.year)
    save_outputs(idata, data_all, all_plots, args.site, args.year, out_dir)


if __name__ == "__main__":
    main()
