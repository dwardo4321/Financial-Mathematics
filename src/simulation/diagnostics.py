# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics")))

# %%
from src.simulation.brownian_motion import path_generator
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd

# %%
# 1 --------------------- Quadratic variation for a random walk --------------------------------------------------------
def qv_random_walk(n: int, loc: float, scale: float):
    x_i = np.random.normal(loc, scale, size=n)
    m_i = []; qv_i = []
    m_i.append(x_i[0]); qv_i.append(x_i[0] ** 2)
    for i in range(len(x_i)):
        m_i.append(x_i[i] + m_i[i])
        qv_i.append((x_i[i] ** 2) + qv_i[i])
    out = pd.DataFrame({"Random Walk": m_i, "Quadratic Variation": qv_i})

    return out


# print(qv_random_walk(1000, 0, 0.1, plot=True))

# %%
# 2 --------------------- Quadratic variation of Brownian motion -------------------------------------------------------
def qv_brownian_motion(scale: float, t, n: int, num_gens: int, correlated: tuple[bool, np.ndarray] = (False, None)):
    # ------------------------------------------------
    if correlated[0]:
        path = path_generator(n = n, t = t, num_gens = num_gens, correlated =(True, correlated[1]))
    else:
        path = path_generator(n = n, t = t, num_gens = num_gens)
    # ------------------------------------------------
    qv_bm = np.zeros((n-1, num_gens))
    for j in range(num_gens):
        qv_bm[:, j] = np.cumsum((np.diff(path.iloc[:, j]))**2, axis = 0)
    qv_bm = pd.DataFrame(qv_bm, columns=np.arange(1, num_gens+1, 1))

    return qv_bm

# %%
#print(qv_brownian_motion(0.1, 2, 1000, 1, plot=True))