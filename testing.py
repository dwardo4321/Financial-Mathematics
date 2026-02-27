import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

N = 100000
norm_rv1 = np.random.normal(0, 100, N-1)
norm_rv2 = np.random.normal(0, 100, N-1)

W_T1 = np.concatenate(([0], np.cumsum(norm_rv1)))[:, None]
W_T2 = np.concatenate(([0], np.cumsum(norm_rv2)))[:, None]