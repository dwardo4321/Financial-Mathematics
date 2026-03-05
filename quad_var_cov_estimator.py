import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd


# 1 --------------------- Quadratic variation for a random walk --------------------------------------------------------
def qv_random_walk(n: int, loc: float, scale: float, plot=False):
    x_i = np.random.normal(loc, scale, size=n)
    m_i = []; qv_i = []
    m_i.append(x_i[0]); qv_i.append(x_i[0] ** 2)
    for i in range(len(x_i)):
        m_i.append(x_i[i] + m_i[i])
        qv_i.append((x_i[i] ** 2) + qv_i[i])
    out = pd.DataFrame({"Random Walk": m_i, "Quadratic Variation": qv_i})
    # ------------------------------------------------
    if plot:
        fig = plt.figure(figsize=(18, 5), facecolor="darkgrey")
        y_labels = ["M_k", "Cumulative Value"]
        titles = ["Symmetric Random Walk", "Quadratic Variation vs Theoretical Variance"]

        for i, data in enumerate(out.columns):
            sub = fig.add_subplot(1, 2, i + 1)
            plt.gca().set_facecolor("pink")
            sub.plot(out.index, out[data])
            if data == "Quadratic Variation":
                sub.plot(out.index, (scale ** 2) * out.index, linestyle='--')
                sub.legend(["Quadratic Variation", "Theoretical Variance"], edgecolor='navy')
            sub.set_ylabel(y_labels[i], fontsize=10)
            sub.set_xlabel('k', fontsize=10)
            sub.set_title(titles[i], fontsize=12)
            plt.grid()
            plt.tight_layout()
        return out.head(), plt.show()
    else:
        return out.head()


# print(qv_random_walk(1000, 0, 0.1, plot=True))

# 2 --------------------- Quadratic variation of Brownian motion -------------------------------------------------------
def qv_brownian_motion(scale: float, t, n: int, num_gens: int, plot=False, correlated: tuple[bool, np.ndarray] = (False, None)):
    # ------------------------------------------------
    import brownian_path_generator as bpg
    if correlated[0]:
        path = bpg.path_generator(n, t, scale, num_gens, plot=False, correlated =(True, correlated[1]))
    else:
        path = bpg.path_generator(n, t, scale, num_gens, plot=False)
    # ------------------------------------------------
    qv_bm = np.zeros((n-1, num_gens))
    for j in range(num_gens):
        qv_bm[:, j] = np.cumsum((np.diff(path.iloc[:, j]))**2, axis = 0)
    qv_bm = pd.DataFrame(qv_bm, columns=np.arange(1, num_gens+1, 1))
    # ------------------------------------------------
    if plot:
        dt = t/(n-1); x = np.arange(0, t, dt)
        plt.figure(figsize=(10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("pink")
        plt.plot(x, qv_bm.values)
        plt.xlabel('t', fontsize=15)
        plt.ylabel('Quadratic Variation', fontsize=15)
        plt.grid()
        return qv_bm, plt.show()
    else:
        return qv_bm

# print(qv_brownian_motion(0.1, 2, 1000, 1, plot=True))

# 3 --------------------- Realized volatility from GBM -----------------------------------------------------------------
# This one reduces to the QV of a BM
def geometric_bm(s_0: float, sig: float, t: float, alph: float, n: int):
    import brownian_path_generator as bpg
    bms = bpg.path_generator(n, t, sig, 1)
    price = s_0 * np.exp(sig*bms.iloc[:, 0].to_numpy() + (alph - 0.5*(sig**2))*np.linspace(0, t, n))
    out = pd.DataFrame({"Time": np.linspace(0, t, n), "Price": price}).set_index("Time")
    return out

#test = geometric_bm(1000, 0.2, 20, 0.1, 100000)
#print(test)
#plt.figure(figsize=(18, 5))
#plt.plot(test)
#plt.show()
# 4 --------------------- Quadratic covariation / cross-variation ------------------------------------------------------
