import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd

def path_generator(n: int, t: float, sig: float, num_gens: int, plot=False):
    dt = t / (n - 1)
    w_t = []

    for j in range(num_gens):
        norm_rv = np.random.normal(0, sig * np.sqrt(dt), n-1)
        path = np.concatenate(([0], np.cumsum(norm_rv)))
        w_t.append(path)
    out = np.column_stack(w_t)
    out = pd.DataFrame(out)
    if plot:
        plt.figure(figsize = (10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("pink")
        plt.plot(out)
        plt.xlabel('t', fontsize = 15)
        plt.ylabel('W(t)', fontsize=15)
        plt.grid()
        return out.head(), plt.show()
    else:
        return out.head()

print(path_generator(1000, 5, 0.8, 5, plot=True))

