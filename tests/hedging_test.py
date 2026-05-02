#%%
from hedging.delta_hedging import delta_hedge_engine

#%%
test = delta_hedge_engine(s_0 = 980, strike = 985, sig = 0.1, rate = 0.09, T = 3, alpha = 0.05, n = 10000, risk_neutral_pricing = False)
test.bsm_put_call_pricer()
test.plot_bsm_pc()
