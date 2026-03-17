import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from time import time
import coloralf as c
from scipy.interpolate import RegularGridInterpolator
import sys



def simpleLinear(x, a):

    return a

def simpleX(x, a):

    return x






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
    gauss_na = gaussian2d_jit(x, y, amplitude, x_c+desaligned*std, y_c+desaligned*std, std)

    return gauss_a / 2 + gauss_na / 2







### For STARDICE PSF

def stardice_timbre(lambdas):

    return 24

def make_interpolator(x, y, z, method='linear'):
    
    interp = RegularGridInterpolator((x, y), z.T, method=method, bounds_error=False, fill_value=0.0)
    
    def interpolate(xi, yi):
        xi, yi = np.asarray(xi), np.asarray(yi)
        shape = np.broadcast_shapes(xi.shape, yi.shape)
        points = np.column_stack([xi.ravel(), yi.ravel()])
        return interp(points).reshape(shape)
    
    return interpolate


def get_stardice_psf(files=["stardice_psf_cube_order1.npz", "stardice_psf_cube_order2.npz"]):

    # for order 1
    w = np.arange(300, 1100, 1)
    data_order1 = np.load(f"specSimulator/datafile/psfs/{files[0]}")
    m_order1 = data_order1["cube"]
    dx_order1 = data_order1["dx"]
    dy_order1 = data_order1["dy"]
    theta0 = data_order1["angle"]
    mfunc_order1 = list()

    for wi, mi in zip(w, m_order1):

        x = np.arange(0, mi.shape[1]) - mi.shape[1] / 2
        y = np.arange(0, mi.shape[0]) - mi.shape[0] / 2

        mfunc_order1.append(make_interpolator(x, y, mi))

    # for order 2
    w = np.arange(300, 1100, 1)
    data_order2 = np.load(f"specSimulator/datafile/psfs/{files[1]}")
    m_order2 = data_order2["cube"]
    dx_order2 = data_order2["dx"]
    dy_order2 = data_order2["dy"]
    mfunc_order2 = list()

    for wi, mi in zip(w, m_order2):

        x = np.arange(0, mi.shape[1]) - mi.shape[1] / 2
        y = np.arange(0, mi.shape[0]) - mi.shape[0] / 2

        mfunc_order2.append(make_interpolator(x, y, mi))

    mfunc = [None, mfunc_order1, mfunc_order2]
    dx = [None, dx_order1, dx_order2]
    dy = [None, dy_order1, dy_order2]

    if "debug-psf" in sys.argv:

        theta0deg = theta0 / 180 * np.pi
        angle_voulu = 0.

        dtheta = theta0deg - angle_voulu

        xt = np.linspace(-24, 24, 500)
        yt = np.linspace(-24, 24, 500)
        xxt, yyt = np.meshgrid(xt, yt)

        nxxt = xxt*np.cos(dtheta) - yyt*np.sin(dtheta)
        nyyt = xxt*np.sin(dtheta) + yyt*np.cos(dtheta)

        zt = mfunc_order1[800-300](xxt, yyt)
        nzt = mfunc_order1[800-300](nxxt, nyyt)

        plt.figure(figsize=(18, 6))

        plt.subplot(131)
        plt.imshow(np.log10(m_order1[800-300]+1), cmap="gray", origin="lower")

        plt.subplot(132)
        plt.imshow(np.log10(zt+1), cmap="gray", origin="lower")

        plt.subplot(133)
        plt.imshow(np.log10(nzt+1), cmap="gray", origin="lower")
        plt.show()


    def stardice_psf(order, angle, x, y, amplitude, x_c, y_c, lambdas):

        if order == 0:

            return moffat2d_jit(x, y, amplitude, x_c, y_c, 3.0, 2.0)
        
        elif lambdas < 300 or lambdas >= 1100:
            if "debug" in sys.argv:
                print(f"Full zeros because lambdas < 300 or lambdas >= 1100")
            return np.zeros_like(x)
        else:

            dtheta = theta0/180*np.pi - angle/180*np.pi
            x_rot = (x-x_c+dx[order][lambdas-300])*np.cos(dtheta) - (y-y_c+dy[order][lambdas-300])*np.sin(dtheta)
            y_rot = (x-x_c+dx[order][lambdas-300])*np.sin(dtheta) + (y-y_c+dy[order][lambdas-300])*np.cos(dtheta)

            if "debug" in sys.argv and lambdas >= 600:
                print(f"Sum of stardice psf at lambdas={lambdas} : {np.sum(mfunc[order][lambdas-300](x_rot, y_rot) * amplitude)}")
            return mfunc[order][lambdas-300](x_rot, y_rot) * amplitude

    return stardice_psf


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









