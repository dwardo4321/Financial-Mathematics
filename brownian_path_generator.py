import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd
from numpy.ma.core import cumsum

def path_generator(n: int, t: float, scale: float, num_gens: int, plot=False, correlated: tuple[object, np.ndarray] = (False, None)):
    dt = t / (n - 1)
    w_t = []

    for j in range(num_gens):
        norm_rv = np.random.normal(0, scale * np.sqrt(dt), (n - 1, num_gens))
        out = pd.DataFrame(np.vstack((np.zeros((1, num_gens)), np.cumsum(norm_rv, axis=0))),
                           columns=np.arange(1, num_gens + 1, 1))

    if correlated[0]:
        l_matrix = np.linalg.cholesky(correlated[1])
        correlated_out = l_matrix * correlated[1]
        z = np.random.normal(0, 1, size=(n - 1, num_gens))
        dw = np.sqrt(dt) * (z @ l_matrix.T) * scale
        out = pd.DataFrame(np.vstack([np.zeros((1, num_gens)), np.cumsum(dw, axis=0)]),
                           columns=np.arange(1, num_gens + 1, 1))

    if plot:
        plt.figure(figsize=(10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("pink")
        plt.plot(out)
        plt.xlabel('t', fontsize=15)
        plt.ylabel('W(t)', fontsize=15)
        plt.grid()
        return out, plt.show()
    else:
        return out
