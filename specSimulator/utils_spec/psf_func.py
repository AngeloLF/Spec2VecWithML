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





### For GAUSSIAN :

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





### For MOFFAT NOT ALIGNED :

def moffat2dNA_timbre(gamma, alpha, desaligned):

    return gamma * 7


@njit(["float32[:,:](int32[:,:], int32[:,:], float32, float32, float32, float32, float32, float32)",
       "float32[:,:](float32[:,:], float32[:,:], float32, float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def moffat2dNA_jit(x, y, amplitude, x_c, y_c, gamma, alpha, desaligned):

    moffat_a = moffat2d_jit(x, y, amplitude, x_c, y_c, gamma, alpha)
    moffat_na = moffat2d_jit(x, y, amplitude, x_c+desaligned, y_c+desaligned, gamma, alpha)

    return moffat_a / 2 + moffat_na / 2





### For GAUSSIAN NOT ALIGNED :

def gaussian2dNA_timbre(std, desaligned):

    return std * 6


@njit(["float32[:,:](int32[:,:], int32[:,:], float32, float32, float32, float32, float32)",
       "float32[:,:](float32[:,:], float32[:,:], float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def gaussian2dNA_jit(x, y, amplitude, x_c, y_c, std, desaligned):

    gauss_a = gaussian2d_jit(x, y, amplitude, x_c, y_c, std)
    gauss_na = gaussian2d_jit(x, y, amplitude, x_c+desaligned, y_c+desaligned, std)

    return gauss_a / 2 + gauss_na / 2





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

    nbFunc = 5
    x_c, y_c = 64, 64
    amplitude = 10000

    x = np.arange(128, dtype="int32")
    y = np.arange(128, dtype="int32")
    xx, yy = np.meshgrid(x, y)
    totArgs = 0


    # moffat :
    graph_num = 0
    gammas = [3.0, 6.0, 9.0]
    totArgs += len(gammas)
    alpha = 2.0

    for i, gamma in enumerate(gammas):
        plt.subplot(nbFunc, len(gammas), graph_num + i + 1)
        timbre_size = moffat2d_timbre(gamma, alpha)
        plt.title(f"Moffat gamma={gamma} alpha={alpha}")
        func = moffat2d_jit(xx, yy, amplitude, x_c, y_c, gamma, alpha)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)


    # moffat alpha :
    graph_num = totArgs
    alphas = [1.5, 2.0, 2.5]
    totArgs += len(alphas)
    gamma = 3.0

    for i, alpha in enumerate(alphas):
        plt.subplot(nbFunc, len(alphas), graph_num + i + 1)
        timbre_size = moffat2d_timbre(gamma, alpha)
        plt.title(f"Moffat gamma={gamma} alpha={alpha}")
        func = moffat2d_jit(xx, yy, amplitude, x_c, y_c, gamma, alpha)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)


    # gaussian :
    stds = [3.0, 6.0, 9.0]
    graph_num = totArgs
    totArgs += len(stds)

    for i, std in enumerate(stds):
        plt.subplot(nbFunc, len(stds), graph_num + i + 1)
        timbre_size = gaussian2d_timbre(std)
        plt.title(f"Gaussian std={std}")
        func = gaussian2d_jit(xx, yy, amplitude, x_c, y_c, std)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)


    # moffat na :
    graph_num = totArgs
    desaligneds = [1.0, 2.0, 4.0]
    totArgs += len(desaligneds)
    gamma = 3.0
    alpha = 2.0

    for i, desaligned in enumerate(desaligneds):
        plt.subplot(nbFunc, len(desaligneds), graph_num + i + 1)
        timbre_size = moffat2dNA_timbre(gamma, alpha, desaligned)
        plt.title(f"MoffatNA gamma={gamma} alpha={alpha} desaligned={desaligned}")
        func = moffat2dNA_jit(xx, yy, amplitude, x_c, y_c, gamma, alpha, desaligned)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)


    # gaussian na :
    graph_num = totArgs
    desaligneds = [1.0, 2.0, 4.0]
    totArgs += len(desaligneds)
    std = 3.0

    for i, desaligned in enumerate(desaligneds):
        plt.subplot(nbFunc, len(stds), graph_num + i + 1)
        timbre_size = gaussian2dNA_timbre(std, desaligned)
        plt.title(f"GaussianNA std={std} desaligned={desaligned}")
        func = gaussian2dNA_jit(xx, yy, amplitude, x_c, y_c, std, desaligned)
        plt.imshow(np.log10(func+1))
        plt.xlabel(f"Flux : {np.sum(func)}")
        gsa(x, y, x_c, y_c, timbre_size)

    plt.tight_layout()
    plt.show()









