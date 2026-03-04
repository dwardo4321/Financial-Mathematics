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
def qv_brownian_motion(scale: float, t, n: int, num_gens: int, plot=False):
    import brownian_path_generator as bpg
    path = bpg.path_generator(n, t, scale, num_gens, plot=False)

    for j in range(len(path.columns))
    for i in range(len(path.index)-1):


    if plot:
        plt.figure(figsize=(10, 10))
        plt.plot(qv_bm.index, qv_bm.values)
        return qv_bm, plt.show()
    else:
        return qv_bm

print(qv_brownian_motion(0.1, 2, 1000, 3))
# 3 --------------------- Realized volatility from GBM -------------------------------------------------------

# 4 --------------------- Quadratic covariation / cross-variation ------------------------------------------------------
