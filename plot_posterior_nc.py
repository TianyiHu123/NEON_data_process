"""
Plot posterior summaries for NEON soil CO2 flux state-space model (flux scale).

Uses streaming accumulation over draws (see main.posterior_flux) to avoid large tensors.
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from main.data_io import load_neon_site_year
from main.posterior_flux import (
    accumulate_chain_mean_flux,
    sigma_obs_posterior_mean,
    time_hour_arrays_from_data,
)

matplotlib.use("Agg")  # non-interactive backend; safe on headless nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare posterior of state-space model with hourly soil CO2 flux for one site-year."
    )
    parser.add_argument("--site", required=True, help="NEON site code, e.g. JERC")
    parser.add_argument("--year", type=int, required=True, help="Calendar year, e.g. 2020")
    args = parser.parse_args()

    original_flux = "./data"
    posterior_file = f"./results/{args.site}/{args.year}/{args.site}_{args.year}_posterior.nc"

    neon_data = load_neon_site_year(args.site, args.year, original_flux)
    data_all = neon_data[args.year].copy()
    data_obs = neon_data[f"obs_{args.year}"]
    n_plot = int(data_all["plot_idx"].nunique())
    del neon_data, data_obs

    ds = xr.open_dataset(posterior_file, group="posterior", chunks={"draw": 1})
    nt = ds.sizes["time"]
    time_array, hour_idx = time_hour_arrays_from_data(data_all, nt)

    stats = accumulate_chain_mean_flux(ds, hour_idx)
    sigma_obs_mean = sigma_obs_posterior_mean(ds)
    n_chain = ds.sizes["chain"]
    ds.close()

    mean_mu = stats["mean_flux_mu"]
    mean_sph = stats["mean_flux_sph"]
    mean_sp = stats["mean_flux_sp"]
    mean_sh = stats["mean_flux_sh"]
    mean_eta_sph = stats["mean_eta_sph"]

    data_plot = data_all.drop(
        columns=["datetime_hour", "horizontalPosition", "hour_of_day", "time_idx", "y"],
        errors="ignore",
    )

    cmap = cm.get_cmap("tab10", max(n_plot, 3))
    color_plots = [mcolors.to_hex(cmap(i)) for i in range(n_plot)]
    color_post = "black"

    # --- Main 4-panel figure (per-chain mean flux) ---
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    ax = axes[0]
    for ch in range(n_chain):
        label_model = "mean flux E[exp(mu_t)]" if ch == 0 else "_nolegend_"
        ax.plot(time_array, mean_mu[ch], color=color_post, alpha=0.5, linewidth=1.5, label=label_model)
    for p in range(n_plot):
        label_obs = f"obs_p{p + 1}"
        ax.plot(
            time_array,
            data_plot["flux_gC"][data_plot["plot_idx"] == p].values,
            color=color_plots[p],
            alpha=0.5,
            linewidth=0.5,
            linestyle="--",
            label=label_obs,
        )
    ax.set_ylabel("SR latent site \n (gC/m²/s)")
    ax.grid()
    ax.legend(loc="upper left")

    ax = axes[1]
    for p in range(n_plot):
        for ch in range(n_chain):
            label_model = "E[exp(mu+plot+hour)]" if ch == 0 and p == 0 else "_nolegend_"
            ax.plot(
                time_array,
                mean_sph[ch, :, p],
                color=color_post,
                alpha=0.5,
                linewidth=1.5,
                label=label_model,
            )
        label_obs = f"obs_p{p + 1}"
        ax.plot(
            time_array,
            data_plot["flux_gC"][data_plot["plot_idx"] == p],
            color=color_plots[p],
            alpha=0.5,
            linewidth=0.5,
            linestyle="--",
            label=label_obs,
        )
    ax.set_ylabel("SR + plot + hour \n (gC/m²/s)")
    ax.grid()
    ax.legend(loc="upper left")

    ax = axes[2]
    for p in range(n_plot):
        for ch in range(n_chain):
            label_model = "E[exp(mu+plot)]" if ch == 0 and p == 0 else "_nolegend_"
            ax.plot(
                time_array,
                mean_sp[ch, :, p],
                color=color_post,
                alpha=0.5,
                linewidth=1.5,
                label=label_model,
            )
        label_obs = f"obs_p{p + 1}"
        ax.plot(
            time_array,
            data_plot["flux_gC"][data_plot["plot_idx"] == p],
            color=color_plots[p],
            alpha=0.5,
            linewidth=0.5,
            linestyle="--",
            label=label_obs,
        )
    ax.set_ylabel("SR + plot \n (gC/m²/s)")
    ax.grid()
    ax.legend(loc="upper left")

    ax = axes[3]
    for ch in range(n_chain):
        label_model = "E[exp(mu+hour)]" if ch == 0 else "_nolegend_"
        ax.plot(time_array, mean_sh[ch], color=color_post, alpha=0.5, linewidth=1.5, label=label_model)
    for p in range(n_plot):
        label_obs = f"obs_p{p + 1}"
        ax.plot(
            time_array,
            data_plot["flux_gC"][data_plot["plot_idx"] == p],
            color=color_plots[p],
            alpha=0.5,
            linewidth=0.5,
            linestyle="--",
            label=label_obs,
        )
    ax.set_ylabel("SR + hour \n (gC/m²/s)")
    ax.grid()
    ax.legend(loc="upper left")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    figure_path = f"./results/{args.site}/{args.year}/SR_flux_posterior.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- sigma_obs band (log-scale SD; band = exp(mean_eta ± mean_sigma_obs)) ---
    mean_flux_pooled = mean_sph.mean(axis=0)
    mean_eta_pooled = mean_eta_sph.mean(axis=0)

    fig2, axes2 = plt.subplots(n_plot, 1, figsize=(8, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes2 = [axes2]

    for p in range(n_plot):
        axp = axes2[p]
        sig = float(sigma_obs_mean[p])
        lower = np.exp(mean_eta_pooled[:, p] - sig)
        upper = np.exp(mean_eta_pooled[:, p] + sig)
        axp.fill_between(time_array, lower, upper, color="0.85", label="exp(mean_eta ± sigma_obs) band")
        axp.plot(
            time_array,
            mean_flux_pooled[:, p],
            color=color_post,
            linewidth=1.5,
            label="mean chain-mean flux (pooled)",
        )
        axp.plot(
            time_array,
            data_plot["flux_gC"][data_plot["plot_idx"] == p].values,
            color=color_plots[p],
            linewidth=0.6,
            linestyle="--",
            alpha=0.7,
            label="obs",
        )
        axp.set_ylabel(f"plot {p + 1}\n(gC/m²/s)")
        axp.grid()
        axp.legend(loc="upper left", fontsize=8)

    axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig2.suptitle(
        "sigma_obs is on log-flux scale; center = mean of per-chain E[exp(eta)]; band uses mean_eta ± sigma_obs",
        fontsize=9,
        y=1.002,
    )
    fig2.tight_layout()
    sigma_path = f"./results/{args.site}/{args.year}/SR_flux_posterior_sigma_obs.png"
    fig2.savefig(sigma_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
