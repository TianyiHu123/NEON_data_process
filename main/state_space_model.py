import os

# With multiple MCMC chains (cores > 1), cap BLAS threads in the environment so
# workers do not oversubscribe the node, e.g. OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
# OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1 (see scripts/submit_array.slurm).

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def State_space_model(data_all, data_filtered, RANDOM_SEED, cores=None):
    
    timearray = np.unique(data_all[['time_idx']].values)
    coords = {
    "time": timearray,
    "plot": sorted(data_all['horizontalPosition'].dropna().unique()),
    "hour_of_day": np.arange(24),
    "obs": np.arange(len(data_filtered)),
    }

    y_obs_lnflux = data_filtered['y'].values
    t_input      = data_filtered["time_idx"].astype(int).values
    plot_input   = data_filtered["plot_idx"].astype(int).values
    hour_input   = data_filtered["hour_of_day"].astype(int).values

    annual_period = 24.0 * 365.0
    annual_sin_1 = np.sin(2 * np.pi * timearray / annual_period)
    annual_cos_1 = np.cos(2 * np.pi * timearray / annual_period)
    annual_sin_2 = np.sin(4 * np.pi * timearray / annual_period)
    annual_cos_2 = np.cos(4 * np.pi * timearray / annual_period)

    with pm.Model(coords=coords) as model:

        beta0 = pm.Normal("beta0", mu=np.nanmean(y_obs_lnflux), sigma=2.0)

        # Annual seasonal harmonics
        beta_sin_1 = pm.Normal("beta_sin_1", mu=0.0, sigma=1.0)
        beta_cos_1 = pm.Normal("beta_cos_1", mu=0.0, sigma=1.0)
        beta_sin_2 = pm.Normal("beta_sin_2", mu=0.0, sigma=0.5)
        beta_cos_2 = pm.Normal("beta_cos_2", mu=0.0, sigma=0.5)

        # Constrain to (-1, 1) for stability
        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)
        sigma_proc = pm.HalfNormal("sigma_proc", sigma=0.2)

        mu_ar = pm.AR("mu_ar",
                      rho=rho,
                      sigma=sigma_proc,
                      constant=False,
                      init_dist=pm.Normal.dist(0.0, sigma=0.5),
                      dims="time",
                      )
        
        mu_t = pm.Deterministic("mu_t",
                                 beta0 +
                                 beta_sin_1 * annual_sin_1 +
                                 beta_cos_1 * annual_cos_1 +
                                 beta_sin_2 * annual_sin_2 +
                                 beta_cos_2 * annual_cos_2 + 
                                 mu_ar,
                                 dims="time",)
        
        sigma_plot      = pm.HalfNormal("sigma_plot", sigma=0.5)
        plot_offset_raw = pm.Normal("plot_offset_raw", mu=0.0, sigma=1.0, dims="plot")
        plot_offset = pm.Deterministic("plot_offset",
                                        sigma_plot * (plot_offset_raw - pm.math.mean(plot_offset_raw)),
                                        dims="plot",)

        sigma_hour      = pm.HalfNormal("sigma_hour", sigma=0.3)
        hour_effect_raw = pm.Normal("hour_effect_raw", mu=0.0, sigma=1.0, dims="hour_of_day")
        hour_effect = pm.Deterministic("hour_effect",
                                        sigma_hour * (hour_effect_raw - pm.math.mean(hour_effect_raw)),
                                        dims="hour_of_day",)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5, dims="plot")

        mu_obs = mu_t[t_input] + plot_offset[plot_input] + hour_effect[hour_input]

        y_like = pm.Normal("y_like",
                           mu=mu_obs,
                           sigma=sigma_obs[plot_input],
                           observed=y_obs_lnflux,
                           dims="obs",)

        if cores is None:
            cores = min(4, os.cpu_count() or 1)
        idata = pm.sample(
            draws=1000,
            tune=2000,
            chains=4,
            cores=cores,
            target_accept=0.95,
            max_treedepth=15,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    return idata