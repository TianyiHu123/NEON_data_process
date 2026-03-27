"""
Streaming accumulation of posterior mean flux on the original (flux) scale.

For each draw, the model is additive on log-flux: eta = mu_t + plot + hour (as appropriate).
We compute per-chain means over draws without materializing a full (chain, draw, time, plot)
xarray tensor. For each chain, NumPy broadcasting builds (draw, time, plot) log-flux arrays,
then sum/mean over the draw axis (materializing exp(...) for that 3D block per chain).
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


def _chain_slice_draw_first(
    da: xr.DataArray,
    chain: int,
    second_dim: str,
) -> np.ndarray:
    """``isel(chain=...)`` then ``(draw, second_dim)`` row-major for vectorized ops."""
    sub = da.isel(chain=chain)
    dims = list(sub.dims)
    if "draw" not in dims:
        raise ValueError(f"Expected dimension 'draw' on {getattr(da, 'name', 'var')}, got {dims}")
    if second_dim not in sub.dims:
        others = [d for d in dims if d != "draw"]
        if len(others) != 1:
            raise ValueError(
                f"Cannot align {getattr(da, 'name', 'var')}: want second_dim={second_dim!r}, got dims {dims}"
            )
        second_dim = others[0]
    return sub.transpose("draw", second_dim).values


def accumulate_chain_mean_flux(
    ds: xr.Dataset,
    hour_idx: np.ndarray,
    dtype=np.float64,
) -> dict[str, np.ndarray]:
    """
    Per chain, mean over draws of exp(eta) for each panel, and mean eta for the full prediction (sph).

    Vectorized over draws: for each chain, loads all draws at once, applies broadcasting on
    (draw, time, plot), then sums over draw. Avoids Python per-draw loops while keeping memory
    bounded to O(n_draw * n_time * n_plot) per chain (not full chain x draw x time x plot).
    Callers should open the posterior dataset without ``chunks={"draw": 1}`` so each
    ``(chain, draw, ...)`` slice is read contiguously (e.g. ``chunks=None``).

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

    inv = 1.0 / float(n_draw)

    for c in range(n_chain):
        mu = _chain_slice_draw_first(ds["mu_t"], c, "time")
        po = _chain_slice_draw_first(ds["plot_offset"], c, "plot")
        he = _chain_slice_draw_first(ds["hour_effect"], c, "hour_of_day")

        if mu.shape != (n_draw, nt):
            raise ValueError(
                f"mu_t slice shape {mu.shape} != (n_draw={n_draw}, nt={nt}); check time vs hour_idx"
            )
        if po.shape != (n_draw, n_plot):
            raise ValueError(f"plot_offset slice shape {po.shape} != (n_draw, n_plot)")
        if he.shape[0] != n_draw:
            raise ValueError(f"hour_effect draw dimension {he.shape[0]} != n_draw={n_draw}")

        h_of_t = he[:, hour_idx]

        log_sph = mu[:, :, None] + po[:, None, :] + h_of_t[:, :, None]
        log_sp = mu[:, :, None] + po[:, None, :]
        log_sh = mu + h_of_t

        acc_mu[c] = np.exp(mu).sum(axis=0) * inv
        acc_sph[c] = np.exp(log_sph).sum(axis=0) * inv
        acc_sp[c] = np.exp(log_sp).sum(axis=0) * inv
        acc_sh[c] = np.exp(log_sh).sum(axis=0) * inv
        acc_eta_sph[c] = log_sph.sum(axis=0) * inv

    return {
        "mean_flux_mu": acc_mu,
        "mean_flux_sph": acc_sph,
        "mean_flux_sp": acc_sp,
        "mean_flux_sh": acc_sh,
        "mean_eta_sph": acc_eta_sph,
    }


def sigma_obs_posterior_mean(ds: xr.Dataset) -> np.ndarray:
    """Posterior mean of sigma_obs per plot (log-scale observation SD). Shape (n_plot,)."""
    return ds["sigma_obs"].mean(dim=("chain", "draw")).values
