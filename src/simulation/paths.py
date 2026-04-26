import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd


def geometric_bm(s_0: float, sd: float, t_n: float, rate: float, n_all: int, n_sims: int):
    bms = path_generator(n_all, t_n, n_sims)
    paths = pd.DataFrame()
    #col = range(1, n_sims + 1, 1)
    for i in range(n_sims):
        path_i = s_0 * np.exp(sd*bms.iloc[:, i].to_numpy() + (rate - 0.5*(sd**2))*np.linspace(0, t_n, n_all))
        paths = pd.concat([paths, pd.DataFrame({i+1 :path_i})], axis = 1)

    return paths

#test = geometric_bm(1000, 0.2, 20, 0.1, 1000, 5)
#print(np.log(test))
#test_1 = test.iloc[:,0].to_numpy()
#print(test_1[:, None][10, 0])
#plt.figure(figsize=(18, 5))
#plt.plot(test)
#plt.show()

# t = 20; n = 1000000; sig = 0.5; alph = 0.1
# s_t = geometric_bm(1000, sd = sig, t_n = t, ret = alph, n_all = n)
# log_ret = np.log(s_t.iloc[:-1, 0].to_numpy() / s_t.iloc[1:, 0].to_numpy())

# var_est = (1/t) * sum(log_ret ** 2)
# print(f"{var_est:.5f}")                     # Sum of the squared log returns over the period 0-T
# var = sig ** 2
# print(f"{var:.5f}")                       # Variance

