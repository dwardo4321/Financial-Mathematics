
# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
class engine_plotter:

    def delta_engine_plotter(self, dim, cols, t_0n, out, nrow, ncol, x_label: str = "Time"):
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
