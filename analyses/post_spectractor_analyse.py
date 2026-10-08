import matplotlib.pyplot as plt
import pickle, json, os, sys, shutil
from astropy.io import fits
import numpy as np
from scipy import interpolate
from time import time
import coloralf as c
import astropy.coordinates as AC
from astropy import units as u
import traceback, json




def printinfo(msg, color=c.g, ret=0):

    tabulation = '\n'*ret
    print(f"{tabulation}{color}INFO [apply_spectractor.py] {msg}{c.d}")




# tel = "auxtel"
# test = "8"
# seed = "413"
# score = "chi2"

# ARGV : test seed tel num
if len(sys.argv) < 5:
    raise Exception(f"Number of argv is not 4 (test, seed, tel, score)")
else:
    test, seed, tel, score = sys.argv[1:5]

nb_show_min = 5
nb_show_max = 5


# test folder
testfolder = f"test{test}{tel}-{seed}"
testdir = f"./results/output_simu/{testfolder}"
remember = f"./results/analyse/{score}/pred_Spectractor_x_x_0e+00/{testfolder}/remember_score_classic.txt"

with open(remember, "r") as f:
    data = f.read().split("\n")

print(f"{c.g}Top {nb_show_min} :{c.d}")
for d in data[:nb_show_min]:
    print(f"    - {d}")

print(f"{c.r}Last {nb_show_max} :{c.d}")
for d in data[-nb_show_max:]:
    print(f"    - {d}")



num = input(f"Num to analyse (empty for cancel) : ")


if num != "":

    spectractor_debug = input(f"Spectractor debug (y/n or empty) : ")
    spectractor_verbose = input(f"Spectractor verbose (y/n or empty) : ")


    # IMPORTATION SPECTRACTOR
    spectractor_version = "Spectractor" 
    for argv in sys.argv:
        if "=" in argv and argv.split("=")[0] == "specver":
            spectractor_version = argv.split("=")[1]
    sys.path.append(f"./{spectractor_version}")
    from spectractor.extractor import extractor
    from spectractor.extractor.spectrum import Spectrum
    from spectractor import parameters
    from spectractor.simulation.adr import hadec2zdpar

    # IMPORTATION SPECSIMULATOR
    sys.path.append(f"./specSimulator")
    from specsimulator import SpecSimulator
    import hparams
    import utils_spec.psf_func as pf




    # post_spectractor_analyse_folder
    psaf = f"./results/analyse/post_spectractor_analyse"
    savefile = f"{testdir}/image_fits/images_{num}.fits"
    spectrum_save = f"{psaf}/spectrum_fits"

    if "post_spectractor_analyse" not in os.listdir(f"./results/analyse"):
        os.mkdir(psaf)
        os.mkdir(spectrum_save)




    ### Importation hparams & variable params
    with open(f"{testdir}/hparams.json", "r") as fjson:
        hp = json.load(fjson)
    vp = np.load(f"{testdir}/vparams.npz")

    gain, ron = hp["CCD_GAIN"], hp["cparams"]["CCD_READ_OUT_NOISE"]

    rebin = hp["CCD_REBIN"]
    if hp["telescope"].lower() in ["auxtel", "auxtelqn"]:
        rebin *= 2




    ### Select config
    if f"{hp['telescope'].lower()}.ini" in os.listdir(f"./{spectractor_version}/config/"):
        configName = hp['telescope'].lower()
    elif "TEL_NAME" in hp.keys():
        configName = hp["TEL_NAME"]
    elif f"{hp['telescope'].lower()}" == "auxtelqn":
        configName = "auxtel"
    else:
        configName = hp['telescope'].lower()
    config = f"./{spectractor_version}/config/{configName}.ini"
    print(f"Loading config : {config}")




    # DEBUG
    if spectractor_debug in ["yes", "y"]:
        print(f"Debug mode for spectractor ...")
        parameters.DEBUG = True

    # VERBOSE
    if spectractor_verbose in ["yes", "y"]:
        print(f"Verbose mode for spectractor ...")
        parameters.VERBOSE = True




    print(f"{c.g}Begin extraction of {c.ti}{savefile}{c.d}")
    xt = np.arange(hp["LAMBDA_MIN"], hp["LAMBDA_MAX"], hp["LAMBDA_STEP"])
    yt = np.load(f"{testdir}/spectrum/spectrum_{num}.npy")

    t0 = time()

    try:

        # EXTRACTION
        spectrum = extractor.Spectractor(savefile, spectrum_save, guess=[64*rebin, 512*rebin], target_label=vp["TARGET"][int(num)], disperser_label=hp["DISPERSER"], config=config)
        
        # Need to interpolate to have the same lambdas as simulation
        finterp = interpolate.interp1d(spectrum.lambdas, spectrum.data, kind='linear', bounds_error=False, fill_value=0.0)
        finterp_err = interpolate.interp1d(spectrum.lambdas, spectrum.err, kind='linear', bounds_error=False, fill_value=0.0)
        finterp_no = interpolate.interp1d(spectrum.lambdas, spectrum.data_next_order, kind='linear', bounds_error=False, fill_value=0.0)
        finterp_err_no = interpolate.interp1d(spectrum.lambdas, spectrum.err_next_order, kind='linear', bounds_error=False, fill_value=0.0)

        spectrum.lambdas = xt
        spectrum.data = finterp(xt)
        spectrum.err = finterp_err(xt)
        spectrum.cov_matrix = np.diag(spectrum.err ** 2) # need to fake a new cov matrix as well ...
        spectrum.data_next_order = finterp_no(xt)
        spectrum.err_next_order = finterp_err_no(xt)
        spectrum.lambdas_binwidths = np.gradient(spectrum.lambdas)

        spectrum.convert_from_flam_to_ADUrate()

        # Extract of spectractor extraction, and convert ADU/s to e-
        xp = spectrum.lambdas
        yp = spectrum.data * gain * spectrum.expo
        yperr = spectrum.err * gain * spectrum.expo

    except Exception as e:

        # Spectractor failed
        printinfo(traceback.format_exc(), color=c.lk)
        printinfo(f"Exception : {e}", color=c.r)
        yp = np.zeros_like(xt) * np.nan
        yperr = np.zeros_like(xt) * np.nan
        printinfo(f"Make nan yt ....", color=c.g)

    ftime = time() - t0
    print(f"Extraction time : {ftime:.1f} sec")

    # save spectrum
    np.save(f"{spectrum_save}/spectrum_{num}.npy", yp)
    np.save(f"{spectrum_save}/spectrumerr_{num}.npy", yperr)

    # show result
    plt.title(f"Extraction of {spectrum_save}/spectrum_{num}.npy with Spectractor")
    plt.plot(xt, yt , c='g', label='Spectrum to predict')
    plt.errorbar(xt, yp, yerr=yperr, c='r', label="Spectractor extraction")
    plt.legend()
    plt.show()
















