#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics\src")))
                            
from risk.vol_misspec_arb import volatility_misspec_arbitrage

#%%
vmp = volatility_misspec_arbitrage(1000, 1005, 0.2, 0.05, 2, 0.05, 100000, False, 0.05)
vmp.volatility_misspecification_pricer()
vmp.vmp_plot()

# %%
