#%%
import sys
from pathlib import Path

SRC = Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics\src")
sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simulation.brownian_motion import brownian_motion

# %%
class pricePATH:

    def geometric_bm(self, s_0: float, vol: float, t_n: float, rate: float, n_all: int, n_sims: int):
        
        bm = brownian_motion(n_all, t_n, n_sims)
        path_gbm = pd.DataFrame()
        self.path_gbm = path_gbm
        bms = bm.simulate()
        #col = range(1, n_sims + 1, 1)
        for i in range(n_sims):
            path_i = s_0 * np.exp(vol*bms.iloc[:, i].to_numpy() + (rate - 0.5*(vol**2))*np.linspace(0, t_n, n_all))
            self.path_gbm = pd.concat([self.path_gbm, pd.DataFrame({i+1 :path_i})], axis = 1)

        return self.path_gbm

# %%
