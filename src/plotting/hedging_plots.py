
# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics")))

#%%
class plotter_hedging:

    def discretisation_bridge_monitoring_plot(self, monte_carlo_prices):
        cols1 = ["Bridge Correction Price", "Naive Monitoring Price"]
        cols2 = ["Bridge Error", "Naive Error"]
        fig1 = self.delta_engine_plotter((12, 4), cols1, monte_carlo_prices.index[:-1], monte_carlo_prices.iloc[:-1,], 1, 2, x_label="n lengths")
        fig2 = self.delta_engine_plotter((12, 4), cols2, monte_carlo_prices["dt"].tolist()[:-1], monte_carlo_prices.iloc[:-1,], 1, 2, x_label="dt")
        
        return fig1, fig2
# %%
