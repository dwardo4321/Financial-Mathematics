import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd

def path_generator(n: int, t: float, sig: float, num_gens: int, plot=False, correlated: tuple[object, np.ndarray] = (False, None)):
    dt = t / (n - 1)
    w_t = []

    for j in range(num_gens):
        norm_rv = np.random.normal(0, sig * np.sqrt(dt), (n-1, num_gens))
        out = pd.DataFrame(np.vstack((np.zeros((1, num_gens)), np.cumsum(norm_rv, axis=0))), columns=np.arange(1, num_gens+1, 1))

    if correlated[0]:
        l_matrix = np.linalg.cholesky(correlated[1])
        correlated_out = l_matrix * correlated[1]
        z = np.random.normal(0, 1, size=(n-1, num_gens))
        dw = np.sqrt(dt) * (z @ l_matrix.T) * sig
        out = pd.DataFrame(np.vstack([np.zeros((1,num_gens)), np.cumsum(dw, axis=0)]), columns=np.arange(1, num_gens+1, 1))

    if plot:
        plt.figure(figsize = (10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("pink")
        plt.plot(out)
        plt.xlabel('t', fontsize = 15)
        plt.ylabel('W(t)', fontsize=15)
        plt.grid()
        return out.head(), plt.show()
    else:
        return out.head()

print(path_generator(1000, 5, 0.8, 3, plot=False))  # Ex 1

corr1 = np.array([[1.00, 0.70, 0.20, -0.10, 0.35],
                 [0.70, 1.00, 0.25,  0.05, 0.40],
                 [0.20, 0.25, 1.00,  0.60, 0.15],
                 [-0.10, 0.05, 0.60, 1.00, 0.10],
                 [0.35, 0.40, 0.15,  0.10, 1.00]])

print(path_generator(1000, 5, 0.8, corr1.shape[0], plot=True, correlated=(True, corr1)))  # Ex 2

dim = 50
corr2 = np.full((dim, dim), 0.75)
np.fill_diagonal(corr2, 1)

print(path_generator(1000, 5, 0.8, dim, plot=True, correlated=(True, corr2)))  # Ex 3

