import numpy as np
import os, sys
from scipy import interpolate
from astropy.modeling import models, fitting
import warnings
import matplotlib.pyplot as plt
import coloralf as c







def get_theta0(x0, y0, CCD_size, CCD_PIXEL2ARCSEC, a=0.0, return_Cp=False):
    
    """
    ---

    Params:
    - x0, y0 : int or float of the order 0
    - CCD_SIZE : list with de size X and y of the CCD
    - CCD_PIXEL2ARCSEC : 
    - a : angle of the dispersion (°)
    """

    arcsec2rad = 1 / 180 * np.pi / 3600
    pix2rad = CCD_PIXEL2ARCSEC * arcsec2rad
    xc, yc = CCD_size[0]/2, CCD_size[1]/2
    arad = a * np.pi / 180


    if a == 0.0:
        
        xp = xc
        yp = y0

    else:

        xp = xc*np.cos(arad)**2 + x0*np.sin(arad)**2 + (yc-y0)*np.sin(arad)*np.cos(arad)
        yp = (xc - xp) / np.tan(arad) + yc


    if isinstance(x0, (int, float)):
        signOf_S0_C = -1.0 if x0 < xp else 1.0 
        signOf_Cp_C = -1.0 if yp < yc else 1.0
    else:
        signOf_S0_C = np.ones_like(x0)
        signOf_S0_C[x0 < xp] *= -1

        signOf_Cp_C = np.ones_like(x0)
        signOf_Cp_C[yp < yc] *= -1


    d_S0_C = signOf_S0_C * np.sqrt((x0-xp)**2 + (y0-yp)**2)
    d_Cp_C = signOf_Cp_C * np.sqrt((xp-xc)**2 + (yp-yc)**2)

    ksi = d_Cp_C * pix2rad
    theta0 = d_S0_C * pix2rad * np.cos(ksi)


    if return_Cp : return theta0, (xp, yp)
    else : return theta0


def get_theta0(x0, y0, CCD_size, CCD_PIXEL2ARCSEC, a=0.0):
    """ Return the incident angle on the disperser in radians, with respect to the disperser normal and the X axis.

    Parameters
    ----------
    x0: float, tuple, list
        The order 0 position in the full non-rotated image.

    Returns
    -------
    theta0: float
        The incident angle in radians

    Examples
    --------
    >>> get_theta0((parameters.CCD_IMSIZE/2,parameters.CCD_IMSIZE/2))
    0.0
    >>> get_theta0(parameters.CCD_IMSIZE/2)
    0.0
    """
    pix_to_rad = CCD_PIXEL2ARCSEC * np.pi / (180. * 3600.)
    if isinstance(x0, (list, tuple, np.ndarray)):
        return (x0[0] - CCD_size[0] / 2) * pix_to_rad
    else:
        return (x0 - CCD_size[0] / 2) * pix_to_rad



def test_theta0(CCD_size=[1024, 128], CCD_PIXEL2ARCSEC=0.401, CCD_PIXEL2MM=24e-3, DISTANCE2CCD=55.45):

    sx, sy = CCD_size
    N = 1000
    cmap_name = "coolwarm"

    if "a" in sys.argv:

        a = np.linspace(0.0, 5.0, N)
        S0 = [256, 96]

        x = np.ones(N) * S0[0]
        y = np.ones(N) * S0[1]

    elif "s" in sys.argv:

        S0s = [[128, 64], [128, 96], [896, 96], [896, 32], [128, 32], [128, 64]]
        tN = int(N / (len(S0s)-1))
        N = tN * (len(S0s)-1)

        x, y = np.array([]), np.array([])

        for i in range(len(S0s)-1):

            x0i, y0i = S0s[i]
            x1i, y1i = S0s[i+1]

            x = np.append(x, np.linspace(x0i, x1i, tN))
            y = np.append(y, np.linspace(y0i, y1i, tN))

        a = np.ones(N) * 0.0

    else:

        raise Exception(f"Select a or s in argv ...")



    theta0 = np.zeros(N)
    xp = np.zeros(N)
    yp = np.zeros(N)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / (N - 1)) for i in range(N)]

    plt.figure()

    for i in range(N):

        theta0[i], (xpi, ypi) = get_theta0(x[i], y[i], CCD_size, CCD_PIXEL2ARCSEC, a[i], return_Cp=True)
        xp[i], yp[i] = xpi, ypi

    plt.subplot(211)
    plt.scatter(x, y, color=colors, marker="*")
    plt.scatter(xp, yp, color=colors)
    plt.scatter(sx/2, sy/2, marker="+", color="k")
    plt.xlabel(f"CCD axis X")
    plt.ylabel(f"CCD axis Y")
    
    plt.subplot(224)
    plt.scatter(a, theta0, color=colors)
    plt.xlabel(f"angle")
    plt.ylabel(f"theta0")

    plt.subplot(223)
    plt.scatter(np.arange(np.size(theta0)), theta0, color=colors)
    plt.xlabel(f"along points ...")
 
    plt.show()



def test_theta0_2d(CCD_size=[1024, 128], CCD_PIXEL2ARCSEC=0.401, CCD_PIXEL2MM=24e-3, DISTANCE2CCD=55.45):

    sx, sy = CCD_size
    x = np.arange(sx)
    y = np.arange(sy)
    xx, yy = np.meshgrid(x, y)

    a = 23.0

    plt.figure()

    theta0 = get_theta0(xx, yy, CCD_size, CCD_PIXEL2ARCSEC, a) * 180 / np.pi * 3600
    theta0_spec = (xx - CCD_size[0]/2) * CCD_PIXEL2ARCSEC

    plt.imshow(theta0-theta0_spec, cmap="coolwarm")
    plt.colorbar(label="theta0 (arcsec)") 
    plt.show()








class MyDisperser():

    def __init__(self, hparameters, As, lambdas, x0):

        self.hp = hparameters

        # Load transmission
        filename = f"{self.hp.DISPERSER_DIR}/{self.hp.DISPERSER}/transmission.txt"
        a = np.loadtxt(filename)
        l, t, _ = a.T
        self.transmission = interpolate.interp1d(l, t, bounds_error=False, fill_value=0.)

        # Load Ratio order
        filename = f"{self.hp.DISPERSER_DIR}/{self.hp.DISPERSER}/ratio_order_2over1.txt"
        a = np.loadtxt(filename)
        if a.T.shape[0] == 2 : l, t = a.T
        else : l, t, e = a.T

        ### test stardice weirdos
        if np.sum(t < 0) > 0:
            print(f"{c.y}WARNING : when load disperser, ratio_order_2over1 containt negative values -> cut to 0{c.d}")
            t[t < 0] = 0

        self.ratio_order_2over1 = interpolate.interp1d(l, t, bounds_error=False, kind="linear", fill_value="extrapolate")

        filename = f"{self.hp.DISPERSER_DIR}/{self.hp.DISPERSER}/ratio_order_3over2.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            if a.T.shape[0] == 2 : l, t = a.T
            else : l, t, e = a.T
            self.ratio_order_3over2 = interpolate.interp1d(l, t, bounds_error=False, kind="linear", fill_value="extrapolate")
            self.ratio_order_3over1 = interpolate.interp1d(l, self.ratio_order_3over2(l)*self.ratio_order_2over1(l), bounds_error=False, kind="linear", fill_value="extrapolate")
        else:
            self.ratio_order_3over2 = None
            self.ratio_order_3over1 = None     
            
        # Load grating
        filedef = "hologram_grooves_per_mm.txt"

        if filedef in os.listdir(f"{self.hp.DISPERSER_DIR}/{self.hp.DISPERSER}"):

            print(f"{c.y}INFO [load_disperser.py] : use {filedef} for load the grating of {self.hp.DISPERSER}{c.d}")
            filename = f"{self.hp.DISPERSER_DIR}/{self.hp.DISPERSER}/{filedef}"
            a = np.loadtxt(filename)
            self.N_x, self.N_y, N_data = a.T
            self.rebin()
            self.N_interp = interpolate.CloughTocher2DInterpolator((self.N_x, self.N_y), N_data)
            self.N_fit = self.fit_poly2d(self.N_x, self.N_y, N_data, order=2)

        else:
            # use N.txt
            print(f"{c.y}INFO [load_disperser.py] : use N.txt for load the grating of {self.hp.DISPERSER}{c.d}")
            filename = f"{self.hp.DISPERSER_DIR}/{self.hp.DISPERSER}/N.txt"
            N, N_err = np.loadtxt(filename)
            self.N_x = np.arange(0, self.hp.SIM_NX).astype(float)
            self.N_y = np.arange(0, self.hp.SIM_NY).astype(float)
            self.rebin()
            self.N_interp = lambda x, y: N
            self.N_fit = lambda x, y: N

        # make distance_along_disp_axis:
        self.dist_along_disp_axis = list()
        for order, A in enumerate(As):
            if A != 0.0 : self.dist_along_disp_axis.append(self.grating_lambda_to_pixel(lambdas, x0=x0, order=order))
            else : self.dist_along_disp_axis.append(None)



    def rebin(self):

        if self.hp.CCD_REBIN > 1:

            self.N_x /= self.hp.CCD_REBIN
            self.N_y /= self.hp.CCD_REBIN



    def grating_lambda_to_pixel(self, lambdas, x0, alpha=0.0, order=1):

        lambdas = np.copy(lambdas)
        # theta0 = (x0[0] - self.hp.CCD_IMSIZE / 2) * self.hp.CCD_PIXEL2ARCSEC * self.hp.CCD_ARCSEC2RADIANS
        # theta0 = (x0[0]) * hparameters.CCD_PIXEL2ARCSEC * hparameters.CCD_ARCSEC2RADIANS
        theta0 = get_theta0(x0[0], x0[1], [self.hp.SIM_NX, self.hp.SIM_NY], self.hp.CCD_PIXEL2ARCSEC, a=alpha)

        theta = np.arcsin(np.clip(order * lambdas * 1e-6 * self.N(x0) + np.sin(theta0),-1, 1))
        deltaX = self.hp.DISTANCE2CCD * (np.tan(theta) - np.tan(theta0)) / self.hp.CCD_PIXEL2MM

        print(f"distance 2 ccd : {self.hp.DISTANCE2CCD }")

        return deltaX


    def N(self, x):

        if x[0] < np.min(self.N_x) or x[0] > np.max(self.N_x) or x[1] < np.min(self.N_y) or x[1] > np.max(self.N_y) : N = float(self.N_fit(*x))
        else : N = int(self.N_interp(*x))

        return N

    def fit_poly2d(self, x, y, z, order):
    
        p_init = models.Polynomial2D(degree=order)
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            p = fit_p(p_init, x, y, z)
            return p



if __name__ == "__main__":

    if "2d" in sys.argv:
        test_theta0_2d()
    else:
        test_theta0()
