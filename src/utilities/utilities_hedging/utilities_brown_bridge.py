import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

def brownian_bridge(num_timesteps: int, num_paths: int, T: float, plot: bool = False):
    dt = T / (num_timesteps - 1)
    t = np.linspace(0, T, num_timesteps)

    dw = np.sqrt(dt) * np.random.randn(num_timesteps - 1, num_paths)
    w = np.vstack([np.zeros((1, num_paths)), np.cumsum(dw, axis=0)])

    bridge = w - (t[:, None] / T) *w[-1, :]
    out = pd.DataFrame(bridge, index=t)

    if plot:
        plt.figure(figsize=(12, 7.5), facecolor = "grey")
        plt.plot(bridge)
        plt.xlabel("Time (t)", fontsize = 12)
        plt.ylabel("W(t)", fontsize=12)
        plt.grid()
        return out, plt.show()
    else:
        return out

