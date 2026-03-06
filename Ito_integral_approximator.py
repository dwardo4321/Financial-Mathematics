import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd

def ito_integral_approx(t: float, sig: float, partitions: int, func):
    time_dat = np.arange(0, t, t/partitions)
    dw_t = np.random.normal(0, sig * np.sqrt(time_dat), len(time_dat))
    x=0

    for i in range(len(time_dat)):
        x += dw_t[i] * func(time_dat[i])
    return x

# print(ito_integral_approx(5, 0.3, 100, lambda j: np.exp(j)))

