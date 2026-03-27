"""
Plot prior and posterior (chain by chain) of each parameters in state space model.

"""

import argparse
import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from main.data_io import load_neon_site_year

matplotlib.use("Agg")  # non-interactive backend; safe on headless nodes


def normal_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def halfnormal_pdf(x, sigma):
    pdf = np.zeros_like(x, dtype=float)
    mask = x >= 0.0
    if np.any(mask):
        xx = x[mask]
        pdf[mask] = np.sqrt(2.0 / np.pi) * np.exp(-(xx * xx) / (2.0 * sigma * sigma)) / sigma
    return pdf


def uniform_pdf(x, lower, upper):
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lower) & (x <= upper)
    width = upper - lower
    if width > 0.0:
        pdf[mask] = 1.0 / width
    return pdf


def parameter_prior_pdf(param_name, x, beta0_mu):
    if param_name == "beta0":
        return normal_pdf(x, mu=beta0_mu, sigma=2.0)
    if param_name in ("beta_sin_1", "beta_cos_1"):
        return normal_pdf(x, mu=0.0, sigma=1.0)
    if param_name in ("beta_sin_2", "beta_cos_2"):
        return normal_pdf(x, mu=0.0, sigma=0.5)
    if param_name == "rho":
        return uniform_pdf(x, lower=-0.99, upper=0.99)
    if param_name == "sigma_proc":
        return halfnormal_pdf(x, sigma=0.2)
    if param_name == "sigma_plot":
        return halfnormal_pdf(x, sigma=0.5)
    if param_name == "sigma_hour":
        return halfnormal_pdf(x, sigma=0.3)
    if param_name in ("plot_offset_raw", "hour_effect_raw"):
        return normal_pdf(x, mu=0.0, sigma=1.0)
    if param_name == "sigma_obs":
        return halfnormal_pdf(x, sigma=0.5)
    raise ValueError(f"Unknown parameter for prior PDF: {param_name}")


def get_scalar_chain_draw(ds, var_name):
    arr = np.asarray(ds[var_name].values)
    if arr.ndim != 2:
        raise ValueError(f"{var_name} expected 2D (chain, draw), got shape {arr.shape}")
    return arr


def get_vector_chain_draw(ds, var_name):
    da = ds[var_name]
    extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
    if len(extra_dims) != 1:
        raise ValueError(
            f"{var_name} expected one extra dim besides chain/draw, got dims {da.dims}"
        )
    elem_dim = extra_dims[0]
    arr = np.asarray(da.transpose("chain", "draw", elem_dim).values)
    return arr, elem_dim


def x_grid_from_samples(sample_list):
    all_values = np.concatenate(sample_list)
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return np.linspace(-1.0, 1.0, 200)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    span = high - low
    if span <= 0.0:
        span = max(abs(low), 1.0) * 0.2
    margin = 0.1 * span
    return np.linspace(low - margin, high + margin, 400)


def plot_chain_hist_with_prior(ax, chain_draw, param_name, beta0_mu, chain_colors, bins=40):
    n_chain = chain_draw.shape[0]
    samples = []
    for ch in range(n_chain):
        vals = chain_draw[ch]
        vals = vals[np.isfinite(vals)]
        samples.append(vals)

    x = x_grid_from_samples(samples)
    y_prior = parameter_prior_pdf(param_name, x, beta0_mu=beta0_mu)

    for ch, vals in enumerate(samples):
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            color=chain_colors[ch],
            label=f"chain {ch}",
        )
    ax.plot(x, y_prior, color="black", linewidth=1.6, label="prior")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot prior vs posterior PDFs for state-space model parameters."
    )
    parser.add_argument("--site", required=True, help="NEON site code, e.g. JERC")
    parser.add_argument("--year", type=int, required=True, help="Calendar year, e.g. 2020")
    args = parser.parse_args()

    original_flux = "./data"
    posterior_file = f"./results/{args.site}/{args.year}/{args.site}_{args.year}_posterior.nc"
    output_dir = f"./results/{args.site}/{args.year}"
    os.makedirs(output_dir, exist_ok=True)

    neon_data = load_neon_site_year(args.site, args.year, original_flux)
    data_obs = neon_data[f"obs_{args.year}"]
    beta0_mu = float(np.nanmean(data_obs["y"].values))

    ds = xr.open_dataset(posterior_file, group="posterior", chunks=None)

    parameter_1d = [
        "beta0",
        "beta_sin_1",
        "beta_cos_1",
        "beta_sin_2",
        "beta_cos_2",
        "rho",
        "sigma_proc",
        "sigma_plot",
        "sigma_hour",
    ]
    parameter_2d = ["plot_offset_raw", "hour_effect_raw", "sigma_obs"]

    n_chain = int(ds.sizes["chain"])
    cmap = cm.get_cmap("tab10", max(n_chain, 3))
    chain_colors = [mcolors.to_hex(cmap(i)) for i in range(n_chain)]

    fig1, axes1 = plt.subplots(len(parameter_1d), 1, figsize=(9, 2.2 * len(parameter_1d)))
    if len(parameter_1d) == 1:
        axes1 = [axes1]

    for i, pname in enumerate(parameter_1d):
        ax = axes1[i]
        draws = get_scalar_chain_draw(ds, pname)
        plot_chain_hist_with_prior(
            ax=ax,
            chain_draw=draws,
            param_name=pname,
            beta0_mu=beta0_mu,
            chain_colors=chain_colors,
        )
        ax.set_ylabel(pname)
        if i == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes1[-1].set_xlabel("parameter value")
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/params_prior_posterior_1d.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    for pname in parameter_2d:
        vec_draws, elem_dim = get_vector_chain_draw(ds, pname)
        n_elem = vec_draws.shape[2]
        fig, axes = plt.subplots(n_elem, 1, figsize=(9, max(2.0 * n_elem, 3.0)))
        if n_elem == 1:
            axes = [axes]

        for idx in range(n_elem):
            ax = axes[idx]
            plot_chain_hist_with_prior(
                ax=ax,
                chain_draw=vec_draws[:, :, idx],
                param_name=pname,
                beta0_mu=beta0_mu,
                chain_colors=chain_colors,
            )
            ax.set_ylabel(f"{pname}[{idx}]")
            if idx == 0:
                ax.legend(loc="upper right", ncol=2, fontsize=8)
        axes[-1].set_xlabel("parameter value")
        fig.suptitle(f"{pname} by {elem_dim}", y=1.0)
        fig.tight_layout()
        fig.savefig(
            f"{output_dir}/params_prior_posterior_{pname}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    ds.close()