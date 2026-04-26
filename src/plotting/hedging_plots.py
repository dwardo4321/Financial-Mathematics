
# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics")))

#%%
def delta_engine_plotter(dim, cols, t_0n, out, nrow, ncol, x_label: str = "Time"):
    fig = plt.figure(figsize=dim, facecolor='darkgrey')

    for i, data in enumerate(cols):
        sub = fig.add_subplot(nrow, ncol, i + 1)
        sub.plot(t_0n, out[data].values)
        sub.set_xlabel(x_label, fontsize=12)
        sub.set_ylabel(data, fontsize=12)
        sub.grid()
    fig.tight_layout()
    plt.show(block=False)
    return fig

#%%
def discretisation_bridge_monitoring_plot(monte_carlo_prices):
    cols1 = ["Bridge Correction Price", "Naive Monitoring Price"]
    cols2 = ["Bridge Error", "Naive Error"]
    fig1 = delta_engine_plotter((12, 4), cols1, monte_carlo_prices.index[:-1], monte_carlo_prices.iloc[:-1,], 1, 2, x_label="n lengths")
    fig2 = delta_engine_plotter((12, 4), cols2, monte_carlo_prices["dt"].tolist()[:-1], monte_carlo_prices.iloc[:-1,], 1, 2, x_label="dt")
    
    return fig1, fig2
# %%
