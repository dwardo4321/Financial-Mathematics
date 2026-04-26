import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def qv_random_walk_plot(out, scale: float): 
    fig = plt.figure(figsize=(18, 5), facecolor="darkgrey")
    y_labels = ["M_k", "Cumulative Value"]
    titles = ["Symmetric Random Walk", "Quadratic Variation vs Theoretical Variance"]

    for i, data in enumerate(out.columns):
        sub = fig.add_subplot(1, 2, i + 1)
        plt.gca().set_facecolor("white")
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

def qv_brownian_motion_plot(qv_bm, t, n):
    dt = t/(n-1); x = np.arange(0, t, dt)
    plt.figure(figsize=(10, 5), facecolor='darkgrey')
    plt.gca().set_facecolor("white")
    plt.plot(x, qv_bm.values)
    plt.xlabel('t', fontsize=15)
    plt.ylabel('Quadratic Variation', fontsize=15)
    plt.grid()

    return plt.show()