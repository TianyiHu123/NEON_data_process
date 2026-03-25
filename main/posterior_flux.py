"""
Streaming accumulation of posterior mean flux on the original (flux) scale.

For each draw, the model is additive on log-flux: eta = mu_t + plot + hour (as appropriate).
We accumulate sum(exp(eta)) and sum(eta) per chain without materializing (chain, draw, time, plot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def time_hour_arrays_from_data(data_all: pd.DataFrame, n_time_expected: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    One timestamp and hour-of-day index per model time step, aligned with ``time_idx`` order.
    """
    u = (
        data_all.sort_values("time_idx")
        .drop_duplicates("time_idx", keep="first")
        .sort_values("time_idx")
    )
    if len(u) != n_time_expected:
        raise ValueError(
            f"Expected {n_time_expected} unique time_idx values from data, got {len(u)}"
        )
    time_array = pd.DatetimeIndex(u["datetime_hour"])
    hour_idx = u["datetime_hour"].dt.hour.to_numpy(dtype=np.int64)
    return time_array, hour_idx


def accumulate_chain_mean_flux(
    ds: xr.Dataset,
    hour_idx: np.ndarray,
    dtype=np.float64,
) -> dict[str, np.ndarray]:
    """
    Per chain, mean over draws of exp(eta) for each panel, and mean eta for the full prediction (sph).

    Returns arrays with shapes:
      mean_flux_mu: (n_chain, n_time)
      mean_flux_sph, mean_eta_sph: (n_chain, n_time, n_plot)
      mean_flux_sp: (n_chain, n_time, n_plot)
      mean_flux_sh: (n_chain, n_time)
    """
    n_chain = ds.sizes["chain"]
    n_draw = ds.sizes["draw"]
    nt = len(hour_idx)
    n_plot = ds.sizes["plot"]

    hour_idx = np.asarray(hour_idx, dtype=np.int64)

    acc_mu = np.zeros((n_chain, nt), dtype=dtype)
    acc_sph = np.zeros((n_chain, nt, n_plot), dtype=dtype)
    acc_sp = np.zeros((n_chain, nt, n_plot), dtype=dtype)
    acc_sh = np.zeros((n_chain, nt), dtype=dtype)
    acc_eta_sph = np.zeros((n_chain, nt, n_plot), dtype=dtype)

    for c in range(n_chain):
        for d in range(n_draw):
            mu = ds["mu_t"].isel(chain=c, draw=d).values
            po = ds["plot_offset"].isel(chain=c, draw=d).values
            he = ds["hour_effect"].isel(chain=c, draw=d).values
            if mu.shape[0] != nt:
                raise ValueError(f"mu_t time length {mu.shape[0]} != hour_idx length {nt}")

            h_of_t = he[hour_idx]

            log_sph = mu[:, None] + po[None, :] + h_of_t[:, None]
            log_sp = mu[:, None] + po[None, :]
            log_sh = mu + h_of_t

            acc_mu[c] += np.exp(mu)
            acc_sph[c] += np.exp(log_sph)
            acc_sp[c] += np.exp(log_sp)
            acc_sh[c] += np.exp(log_sh)
            acc_eta_sph[c] += log_sph

    inv = 1.0 / float(n_draw)
    return {
        "mean_flux_mu": acc_mu * inv,
        "mean_flux_sph": acc_sph * inv,
        "mean_flux_sp": acc_sp * inv,
        "mean_flux_sh": acc_sh * inv,
        "mean_eta_sph": acc_eta_sph * inv,
    }


def sigma_obs_posterior_mean(ds: xr.Dataset) -> np.ndarray:
    """Posterior mean of sigma_obs per plot (log-scale observation SD). Shape (n_plot,)."""
    return ds["sigma_obs"].mean(dim=("chain", "draw")).values
