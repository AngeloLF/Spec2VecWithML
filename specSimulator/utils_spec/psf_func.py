import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from time import time


def simpleLinear(x, a):

    return a






### For MOFFAT :

def moffat2d_timbre(gamma, alpha):

    return gamma * 6


@njit(["float32[:,:](int32[:,:], int32[:,:], float32, float32, float32, float32, float32)",
       "float32[:,:](float32[:,:], float32[:,:], float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def moffat2d_jit(x, y, amplitude, x_c, y_c, gamma, alpha):

    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a





### For GAUSSIAN

def gaussian2d_timbre(std):

    return std * 5


@njit(["float32[:,:](int32[:,:], int32[:,:], float32, float32, float32, float32)",
       "float32[:,:](float32[:,:], float32[:,:], float32, float32, float32, float32)"], fastmath=True, cache=True)
def gaussian2d_jit(x, y, amplitude, x_c, y_c, std):

    xc = x - x_c
    yc = y - y_c

    rr_ss = (xc * xc + yc * yc) / (np.float32(2.0) * std * std)

    a = np.exp(-rr_ss)
    norm = np.float32(2.0) * np.pi * std * std
    a *= amplitude / norm

    return a





### For test function:

def gsa(x, y, x_c, y_c, timbre_size, c="r", ls="-", label=None):

    xmin = max(0,          int(x_c - timbre_size))
    xmax = min(np.size(x), int(x_c + timbre_size))
    ymin = max(0,          int(y_c - timbre_size))
    ymax = min(np.size(y), int(y_c + timbre_size))
    
    X = np.array([xmin, xmax, xmax, xmin, xmin])
    Y = np.array([ymin, ymin, ymax, ymax, ymin])

    plt.plot(X, Y, c=c, ls=ls, label=label)



if __name__ == "__main__":

    plt.figure(figsize=(9, 9))

    nbFunc = 3
    x_c, y_c = 64, 64
    amplitude = 10000

    x = np.arange(128, dtype="int32")
    y = np.arange(128, dtype="int32")
    xx, yy = np.meshgrid(x, y)


    # moffat :
    graph_num = 0
    gammas = [3.0, 6.0, 9.0]
    alpha = 2.0

    for i, gamma in enumerate(gammas):
        plt.subplot(nbFunc, len(gammas), graph_num + i + 1)
        timbre_size = moffat2d_timbre(gamma, alpha)
        plt.title(f"arg of {gamma}")
        func = moffat2d_jit(xx, yy, amplitude, x_c, y_c, gamma, alpha)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)


    # moffat alpha :
    graph_num = len(gammas)
    gamma = 3.0
    alphas = [1.5, 2.0, 2.5]

    for i, alpha in enumerate(alphas):
        plt.subplot(nbFunc, len(alphas), graph_num + i + 1)
        timbre_size = moffat2d_timbre(gamma, alpha)
        plt.title(f"arg of alpha = {alpha}")
        func = moffat2d_jit(xx, yy, amplitude, x_c, y_c, gamma, alpha)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)


    # gaussian :
    stds = [3.0, 6.0, 9.0]
    graph_num = len(gammas) + len(alphas)

    for i, std in enumerate(stds):
        plt.subplot(nbFunc, len(stds), graph_num + i + 1)
        timbre_size = gaussian2d_timbre(std)
        plt.title(f"arg of {std}")
        func = gaussian2d_jit(xx, yy, amplitude, x_c, y_c, std)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)

    plt.tight_layout()
    plt.show()









