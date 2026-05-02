#%%
import sys
from pathlib import Path

SRC = Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics\src")
sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib.pyplot as plt
from simulation.pricing_paths import pricePATH

#%%
test = pricePATH()

prices = test.geometric_bm(1000, 0.2, 20, 0.1, 1000, 5)
print(prices)

#%%
plt.figure(figsize=(18, 9))
plt.plot(prices)
plt.show()

#%%
t = 20; n = 1000000; sig = 0.5; alph = 0.1
s_t = test.geometric_bm(1000, vol = sig, t_n = t, rate = alph, n_all = n, n_sims=1)
log_ret = np.log(s_t.iloc[:-1, 0].to_numpy() / s_t.iloc[1:, 0].to_numpy())

#%%
var_est = (1/t) * sum(log_ret ** 2)
print(f"{var_est:.5f}")                     # Sum of the squared log returns over the period 0-T
var = sig ** 2
print(f"{var:.5f}")                       # Variance
# %%
