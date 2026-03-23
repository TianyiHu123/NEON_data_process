from pathlib import Path
import numpy as np
import pandas as pd

NEON_variables = ["startDateTime","horizontalPosition", "flux_gC"]

def load_neon_sitedata(sitename, data_dir, Neon_var=None):

    # get data
    Neon_dir = Path(data_dir) / sitename
    files = sorted(Neon_dir.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"No Neon data found in {Neon_dir}")
    
    # NEON data format: {sitename}_{yyyy}_hourly_gC_allpos.csv
    # Inquire year range of data
    years = []
    for f in files:
        fname = f.name
        # Expected format: {sitename}_{yyyy}_hourly_gC_allpos.csv
        parts = fname.split("_")
        if len(parts) >= 3 and parts[0] == sitename and parts[1].isdigit() and parts[2] == "hourly":
            years.append(int(parts[1]))
    if not years:
        raise ValueError(f"No properly named NEON files found in {Neon_dir} for site {sitename}")
    year_min, year_max = min(years), max(years)
    n_years = year_max - year_min + 1
    print(f"Data available for years {year_min} to {year_max} at site {sitename}")

    # read in each years data and append to dictionary of years
    if Neon_var is None:
        Neon_var = NEON_variables
    neon_data = {}
    for y in range(year_min, year_max+1, 1):

        # Read relevant columns from each file and store by year in neon_data
        fname = f"{sitename}_{y}_hourly_gC_allpos.csv"
        file_path = Neon_dir / fname
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist. Skipping year {y}.")
            continue
        df = pd.read_csv(file_path, usecols=[col for col in Neon_var if col in pd.read_csv(file_path, nrows=0).columns])
        df['startDateTime'] = pd.to_datetime(df['startDateTime'], utc=True, errors="coerce")
        df['flux_gC'] = pd.to_numeric(df['flux_gC'], errors="coerce")
        df['flux_gC'] = df['flux_gC'].replace([np.inf, -np.inf], np.nan)
        # Remove rows with unusable time or missing plot ID
        # Keep NaN flux values for now so we can preserve missing plot-hours
        df = df.dropna(subset=['startDateTime', 'horizontalPosition'])
        df.loc[df['flux_gC'] <= 0, 'flux_gC'] = np.nan

        # Round to hour to ensure consistent hourly timestamps
        df["datetime_hour"] = df['startDateTime'].dt.floor("h")

        # If there are duplicate measurements within the same plot-hour,
        # average them. mean() skips NaNs by default; groups with all-NaN remain NaN.
        df = df.groupby(["datetime_hour", 'horizontalPosition'], as_index=False)['flux_gC'].mean()
        # Sort for stable downstream behavior
        df = df.sort_values(["datetime_hour", 'flux_gC']).reset_index(drop=True)

        time_start = df["datetime_hour"].min()
        time_end   = df["datetime_hour"].max()

        all_times = pd.date_range(start=time_start, end=time_end, freq="h", tz="UTC")
        all_plots = sorted(df['horizontalPosition'].dropna().unique())

        full_index = pd.MultiIndex.from_product(
            [all_times, all_plots], names=["datetime_hour", 'horizontalPosition']
        )

        df = (
            df.set_index(["datetime_hour", 'horizontalPosition'])
                    .reindex(full_index)
                    .reset_index()
        )

        # Create hour-of-day and integer indices on the FULL grid
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        plot_to_idx = {p: i for i, p in enumerate(all_plots)}

        df["hour_of_day"] = df["datetime_hour"].dt.hour
        df["time_idx"]    = df["datetime_hour"].map(time_to_idx)
        df["plot_idx"]    = df['horizontalPosition'].map(plot_to_idx)

        df["y"] = np.log(df['flux_gC'])

        print(f'For year {y}')
        print(f"Number of hourly time steps in full grid: {len(all_times)}")
        print(f"Number of plots: {len(all_plots)}")
        print(f"Plots found: {all_plots}")
        print(f"Total plot-hour rows in full grid: {len(df)}")
        print(f"Missing flux rows in full grid: {df['flux_gC'].isna().sum()}")

        obs_mask = df["y"].notna().values
        # df_obs is the observed subset used by the likelihood
        df_obs = df.loc[obs_mask].copy().reset_index(drop=True)

        neon_data[y] = df
        neon_data['obs_'+str(y)] = df_obs

    return neon_data, [year_min, year_max]