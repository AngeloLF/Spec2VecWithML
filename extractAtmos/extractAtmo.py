import numpy as np
import matplotlib.pyplot as plt
import sys, json, os, shutil
import coloralf as c
# import alftool as alf
from scipy import interpolate
from time import time
from copy import deepcopy
from types import MethodType
from scipy.ndimage import zoom

sys.path.append('./models/')
from get_argv import get_argv

# IMPORTATION SPECTRACTOR
spectractor_version = "Spectractor" 
for argv in sys.argv:
    if "=" in argv and argv.split("=")[0] == "specver":
        spectractor_version = argv.split("=")[1]
sys.path.append(f"./{spectractor_version}")
from spectractor import parameters
from spectractor.extractor.spectrum import Spectrum
from spectractor.fit.fit_spectrogram import SpectrogramFitWorkspace, run_spectrogram_minimisation
from spectractor.fit.fit_spectrum import SpectrumFitWorkspace, run_spectrum_minimisation





def showHeader(header):

    keys = list(header.keys())
    print(f"\nHEADER : ")

    for k in keys:
        print(f"{c.lr}{k}{c.d} : {header[k]}")






def recupAtmosFromParams(w, file_json, wanted_labels=["vaod", "ozone", "pwv", "d_ccd"]):

    data = dict()

    for l, v, e in zip(w.params.labels, w.params.values, w.params.err):
        
        for wl in wanted_labels:
            if wl.lower() in l.lower():
                data[wl] = [float(v), float(e)]

    with open(file_json, "w") as f:
        json.dump(data, f)




def printdebug(msg, debug, color=c.m):

    if debug:
        print(f"{color}DEBUG : {msg}{c.d}")




def extractOne(Args, num_str, path="./results/output_simu", atmoParamFolder="atmos_params_fit", mode="spec_size"):

    parameters.DISPLAY = False
    debug = False

    ### Importation hparams & variable params
    with open(f"{path}/{Args.test}/hparams.json", "r") as fjson:
        hp = json.load(fjson)

    if Args.model == "true":
        predFolder = "spectrum"
        saveFolder = "true"

    elif Args.model == "spectractorfile":
        predFolder = "Spectractor"
        saveFolder = "Spectractor"

    else:
        predFolder = f"pred_{Args.fullname}"
        saveFolder = f"pred_{Args.fullname}"

    file_json = f"{path}/{Args.test}/{atmoParamFolder}/{saveFolder}/atmos_params_{num_str}_spectrum.json"
    if atmoParamFolder not in os.listdir(f"{path}/{Args.test}"):
        try:
            os.mkdir(f"{path}/{Args.test}/{atmoParamFolder}")
        except:
            print(f"WARNING [extractAtmos.py] : mkdir of {path}/{Args.test}/{atmoParamFolder} not work")
    if saveFolder not in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}"):
        try:
            os.mkdir(f"{path}/{Args.test}/{atmoParamFolder}/{saveFolder}")
        except:
            print(f"WARNING [extractAtmos.py] : mkdir of {path}/{Args.test}/{atmoParamFolder}/{saveFolder} not work")


    ### EXTRACTION with the spectrum
    c.fg(f"INFO [extractAtmo.py] : Begin Spectrum Minimisation for {Args.test}/{predFolder}/spectrum_{num_str}.npy ...")
    if f"images_{num_str}_spectrum.fits" in os.listdir(f"{path}/{Args.test}/spectrum_fits"):
        file_name = f"{path}/{Args.test}/spectrum_fits/images_{num_str}_spectrum.fits"
        spec = Spectrum(file_name, fast_load=True)

        if "debug" in sys.argv:
            parameters.DEBUG = True
            parameters.VERBOSE = True
            parameters.DISPLAY = True # oskour ...
            debug = True

        if predFolder is not None and predFolder != "Spectractor": # need to change data / lambdas_binw / err / cov_matrix

            if not "Spectractor" in predFolder:
                spec.header["D2CCD"] = hp["DISTANCE2CCD"]
                spec.header["PIXSHIFT"] = 0.0
                print(f"Change d2ccd and set PIXSHIFT at 0")

            spec.convert_from_flam_to_ADUrate()
            
            x = np.arange(300, 1100).astype(float)
            y = np.load(f"{path}/{Args.test}/{predFolder}/spectrum_{num_str}.npy")
            scale = np.size(x) / np.size(spec.lambdas)

            ## All to SIMU size
            if mode == "simu_size":
                finterp_err = interpolate.interp1d(spec.lambdas, spec.err, kind='linear', bounds_error=False, fill_value=0.0)
                spec.lambdas = x
                spec.data = y / spec.gain / spec.expo
                spec.lambdas_binwidths = np.gradient(spec.lambdas)
                spec.err = finterp_err(x)
                spec.err[spec.err == 0.] = np.min(spec.err[spec.err > 0.])
                spec.cov_matrix = np.diag(spec.err * spec.err) # zoom(spec.cov_matrix, (scale, scale))

            ## All to SPECTRACTOR size
            elif mode == "spec_size":
                finterp = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=0.0)
                spec.data = finterp(spec.lambdas) / spec.gain / spec.expo / scale

            else:
                raise Exception(f"Mode {mode} d'ont exist (choose in `spec_size`, `simu_size`)")


            spec.convert_from_ADUrate_to_flam()

            printdebug(f"Go SpectrumFitWorkspace...", debug)
            w = SpectrumFitWorkspace(spec, atmgrid_file_name="", verbose=debug, plot=debug, live_fit=False, fit_angstrom_exponent=True)
            if mode == "simu_size":
                w.simulation.alf_sim_lambdas = np.arange(300, 1100).astype(float)
            printdebug(f"End of SpectrumFitWorkspace", debug)

        else:

            printdebug(f"Go SpectrumFitWorkspace...", debug)
            w = SpectrumFitWorkspace(spec, atmgrid_file_name="", verbose=debug, plot=debug, live_fit=False, fit_angstrom_exponent=True)
            printdebug(f"End of SpectrumFitWorkspace", debug)




        w.filename = ""
        if "date_obs" not in dir(w.spectrum):
            w.spectrum.date_obs = "2017-05-31T02:53:52.356"
            print(f"Info [extractAtmos.py] : DATE-OBS not given")

        printdebug(f"Go run_spectrum_minimisation...", debug)
        run_spectrum_minimisation(w, method="newton", with_line_search=False)
        printdebug(f"End of run_spectrum_minimisation", debug)

        recupAtmosFromParams(w, file_json)
    else:
        print(f"Info [extractAtmos.py] : fits {path}/{Args.test}/spectrum_fits/images_{num_str}_spectrum.fits not exist [skip this one]")






if __name__ == "__main__":

    """
    For extract atmos with spectractor minimisation !
    From pred spectrum or :
        * true : using true spectrum from simulation
        * spectractorfile : using spectrum.fits product of spectractor extraction (different from pred_Spectractor_x_x_0e+00 -> 2 interpolation)
    """

    # arguments needed
    path = "./results/output_simu"
    atmoParamFolder = "atmos_params_fit"
    Args = get_argv(sys.argv[1:], prog="extract_atmo")
    makeonly = None

    # multiple cpu ?
    nrange = None
    for arg in sys.argv[1:]:
        if arg[:6] == "range=" : nrange = arg[6:]
        if arg[:9] == "makeonly=" : makeonly = int(arg[9:])

    nums_str = np.sort([fspectrum.split("_")[1][:-4] for fspectrum in os.listdir(f"{path}/{Args.test}/spectrum")])

    # build partition
    if nrange is None:
        partition = [None]*len(nums_str)
    else:
        nbegin, nsimu = nrange.split("_")
        partition = np.arange(int(nbegin), int(nbegin) + int(nsimu))

    t0 = time()
    
    for n in nums_str:

        if (partition[0] is None or int(n) in partition) and (makeonly is None or makeonly == int(n)):

            extractOne(Args, n, path=path, atmoParamFolder=atmoParamFolder, mode="simu_size")

    tf = time() - t0
    print(f"Finish in {tf/60:.1f} min [{tf/len(partition):.1f} sec/extraction]")





