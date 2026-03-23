import main.data_io as dataio
import numpy as np
import pandas as pd


if __name__ == '__main__':

    
    Neon_data, year_range = dataio.load_neon_sitedata(sitename='JERC', data_dir='./data', Neon_var=None)

    print(Neon_data[2020].columns)
    print(Neon_data[2020][['time_idx','plot_idx']])
    n_times = Neon_data[2020][['time_idx']].values
    print(np.unique(n_times))
    print(Neon_data['obs_2020'].columns)
    print(Neon_data['obs_2020'][['time_idx','plot_idx','hour_of_day','flux_gC','y']])