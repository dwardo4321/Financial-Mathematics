import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

N = 100000
norm_rv1 = np.random.normal(0, 1.5, N-1)
norm_rv2 = np.random.normal(0, 1.5, N-1)

W_T1 = np.concatenate(([0], np.cumsum(norm_rv1)))[:, None]
W_T2 = np.concatenate(([0], np.cumsum(norm_rv2)))[:, None]

phi = 0.95
W_T3 = phi * W_T1 + np.sqrt(1 - phi**2) * W_T2

plt.figure(figsize = (19, 10))
plt.plot(W_T1, color = 'b')
plt.plot(W_T2, color = 'r')
plt.plot(W_T3, color = 'black')
plt.show()