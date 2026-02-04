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



def printinfo(msg, color=c.g, ret=0):

    tabulation = '\n'*ret

    print(f"{tabulation}{color}INFO [apply_spectractor.py] {msg}{c.d}")



def apply_spectractor(testname, pathtest="./results/output_simu", makeonly=None, maxloop=1, specver="Spectractor", debug=False, spectractor_debug=None, partition=None):
    """
    Apply Spectractor

    Param:
        - testname [str] : name of the test folder
        - pathtest [str] : path of the test folder
        - makeonly [int or None] : number of the simulation is we want only one apply
        - maxloop [int] : number of try for spectractor's extraction
        - specver [str] : the version of spectractor (need to have the spectractor folder)
        - debug [bool] : debug of apply_spectractor itself
        - spectractor_debug [None or str] : None, "debug" or "verbose" for spectractor. "debug" turn on parameters.DEBUG and parameters.VERBOSE
        - partition [list of int | numpy of int] : indice for apply a sub-list of images
    """



    ### Folders
    testdir = f"{pathtest}/{testname}"
    # folder name for spectractor (`x` and `0e+00` is to have de same format as ML models)
    pred = f"pred_Spectractor_x_x_0e+00"



    ### Importation hparams & variable params
    with open(f"{testdir}/hparams.json", "r") as fjson:
        hp = json.load(fjson)
    vp = np.load(f"{testdir}/vparams.npz")



    ### find images
    # folder for spectractor & ML models is differente for big images (like auxtel ...)
    if hp["telescope"] in ["auxtel", "auxtelqn"]:
        folder_image = "imageOrigin"
        printinfo(f"INFO : image folder bascule to `imageOrigin`")
    else:
        folder_image = "image"
    images = np.sort([f"{testdir}/{folder_image}/{fimage}" for fimage in os.listdir(f"{testdir}/{folder_image}")])
    


    ### Select config
    if f"{hp['telescope'].lower()}.ini" in os.listdir(f"./{specver}/config/"):
        configName = hp['telescope'].lower()
    elif "TEL_NAME" in hp.keys():
        configName = hp["TEL_NAME"]
    elif f"{hp['telescope'].lower()}" == "auxtelqn":
        configName = "auxtel"
    else:
        configName = hp['telescope'].lower()
    config = f"./{specver}/config/{configName}.ini"
    printinfo(f"Loading config : {config}")



    ### Recup values from simulation
    gain, ron = hp["CCD_GAIN"], hp["cparams"]["CCD_READ_OUT_NOISE"]
    rebin = hp["CCD_REBIN"]
    if hp["telescope"].lower() in ["auxtel", "auxtelqn"]:
        rebin *= 2
    
    

    ### header for CTIO
    header = fits.Header()

    header["DISPERSER"] = hp["DISPERSER"]
    header["EXPTIME"] = hp["cparams"]["EXPOSURE"]
    header['OUTTEMP'] = hp["cparams"]["ATM_TEMPERATURE"]
    header['OUTPRESS'] = hp["OBS_PRESSURE"]
    header['OUTHUM'] = hp["cparams"]["ATM_HUMIDITY"]
    header['XLENGTH'] = hp["SIM_NX"]
    header['YLENGTH'] = hp["SIM_NX"]
    header['XPIXSIZE'] = hp["CCD_PIXEL2ARCSEC"]
    header['GTGAIN11'], header['GTGAIN12'], header['GTGAIN21'], header['GTGAIN22'] = gain, gain, gain, gain
    header['GTRON11'], header['GTRON12'], header['GTRON21'], header['GTRON22'] = ron, ron, ron, ron

    header["FILTER"] = "empty"
    header["FILTERS"] = f"dia {hp['DISPERSER']}"
    header["FILTER1"] = "dia"
    header['FILTER2'] = hp["DISPERSER"]

    header["GRATING"] = hp["DISPERSER"]
    header["BUNIT"] = "adu"
    header["OBS-ELEV"] = 2662.99616375123
    header["OBS-LAT"] =  -30.2446389756252

    header["LINSPOS"] = hp["DISTANCE2CCD"] - 115
    if hp["DISPERSER"] == "holo4_003":
        header["LINSPOS"] -= 4
    printinfo(f"Distance CCD in hparams : {hp['DISTANCE2CCD']}")

    header["DATE-OBS"] = "2017-05-31T02:53:52.356"
    header["DATE"] = "2017-05-31T02:55:54"
    header['RA'] = "12:53:8.10"
    header['DEC'] = "-18:33:27.78"
    header['HA'] = 28.2 # "01:52:42.92"

    header["AIRTEMP"] = hp["cparams"]["ATM_TEMPERATURE"]
    header["HUMIDITY"] = hp["cparams"]["ATM_HUMIDITY"]
    header["PRESSURE"] = hp["OBS_PRESSURE"]

    lat = AC.Latitude(hp["OBS_LATITUDE"], unit=u.deg)
    dec = hp["cparams"]["ADR_DEC"]
    ha  = hp["cparams"]["ADR_HOUR_ANGLE"]
    _, par_angle = hadec2zdpar(ha, dec, lat.degree, deg=True)
    header["ROTPA"] = 270 - hp["OBS_CAMERA_ROTATION"] # hp["OBS_CAMERA_ROTATION"] - hp["OBS_CAMERA_RA_FLIP_SIGN"] * par_angle

    printinfo(f"ROTPA et par_angle : {header['ROTPA']}, {par_angle}")

    xt = np.arange(hp["LAMBDA_MIN"], hp["LAMBDA_MAX"], hp["LAMBDA_STEP"])



    # create new folders for spectractor, if partition is not given (in this case, folder need to be created before)
    if partition is None and makeonly is None:
        printinfo(f"No partition, reset folders")
        for fold in ["image_fits", "spectrum_fits", "spectractor_exceptions", pred]:
            if fold in os.listdir(testdir) : shutil.rmtree(f"{testdir}/{fold}")
            os.mkdir(f"{testdir}/{fold}")
    else:
        printinfo(f"Partition : {partition}")



    # save exe. times
    times = np.zeros(hp["nsimu"])
    nbsok = np.zeros(hp["nsimu"]).astype(bool)
    dictOfExceptionSpectractor = dict()



    # iteration on simu
    for n in range(hp["nsimu"]):

        if (makeonly is None or makeonly == n) and (partition is None or n in partition):


            ### extract unique values
            header["TARGET"] = vp["TARGET"][n]
            header["AIRMASS"] = vp["ATM_AIRMASS"][n]
            header["ANGLE"] = vp["ROTATION_ANGLE"][n]

            header["AMSTART"] = vp["ATM_AIRMASS"][n]
            header["AMEND"] = vp["ATM_AIRMASS"][n]
            header["OBJECT"] = vp["TARGET"][n]

            printinfo(f"Make {n}, with angle {header['ANGLE']}", ret=8)
            


            ### Initialise empty image
            data = np.zeros((hp["SIM_NX"], hp["SIM_NX"])) + hp["cparams"]["BACKGROUND_LEVEL"] * hp["cparams"]["EXPOSURE"]

            # add noise
            d = np.copy(data).astype(np.float32) * gain
            noisy = np.random.poisson(d).astype(np.float32)
            noisy += np.random.normal(scale=hp["cparams"]["CCD_READ_OUT_NOISE"]*np.ones_like(noisy)).astype(np.float32)
            data = noisy / gain
            min_noz = np.min(data[data > 0])
            data[data <= 0] = min_noz



            ### integration de la simulation
            bande = int((hp["SIM_NX"] - hp["SIM_NY"])/2)
            data[bande:bande+hp["SIM_NY"], :] = np.load(images[n])
            true_name = images[n].split("_")[-1][:-4]




            ### debug
            if debug:

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})

                plt.suptitle(f"Extraction de spectres ... [meanADU={np.mean(data)}]")

                ax1.imshow(np.log10(data))

                ax2.axhline(0, color='k', linestyle=':')
                ax2.plot(np.sum(data, axis=0), c="r")
                ax2.set_xlabel(r"CCD axis 1")

                plt.show()





            ### Reglage AuxTel
            if hp["telescope"].lower() in ["auxtel", "auxtelqn"]:

                datap = np.zeros((hp["SIM_NX"]*2, hp["SIM_NX"]*2))

                data = data.T[::-1, ::-1]
                datap[::2, ::2] = data
                data = datap




            ### create FITS
            hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
            hdul = fits.HDUList([hdu])
            savefile = f"{testdir}/image_fits/images_{true_name}.fits"
            hdul.writeto(savefile, overwrite=True)
            printinfo(f"Creation of {savefile}")

            if spectractor_debug == "debug":
                printinfo(f"Debug mode for spectractor ...")
                parameters.DEBUG = True 
                parameters.VERBOSE = True
            if spectractor_debug == "verbose":
                printinfo(f"Verbose mode for spectractor ...")
                parameters.VERBOSE = True

            yt = np.load(f"{testdir}/spectrum/spectrum_{true_name}.npy")
            spectractor_ok = False
            loop = 0

            t0 = time()

            while not spectractor_ok:

                loop += 1

                try:
                    # EXTRACTION
                    spectrum = extractor.Spectractor(savefile, f"{testdir}/spectrum_fits", guess=[64*rebin, 512*rebin], target_label=vp["TARGET"][n], disperser_label=hp["DISPERSER"], config=config)
                    
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
                    spectractor_ok = True

                except Exception as e:

                    # Spectractor failed
                    printinfo(traceback.format_exc(), color=c.lk)
                    printinfo(f"Exception : {e}", color=c.r)
                    printinfo(f"WARNING : loop {loop}/{maxloop} extractor failed ...", color=c.r)
                    if loop == maxloop:
                        spectractor_ok = True
                        yp = np.zeros_like(xt) * np.nan
                        yperr = np.zeros_like(xt) * np.nan
                        printinfo(f"Make nan yt ....", color=c.g)
                        dictOfExceptionSpectractor[images[n]] = traceback.format_exc()

            times[n] = time() - t0
            nbsok[n] = spectractor_ok

            # save spectrum
            np.save(f"{testdir}/{pred}/spectrum_{true_name}.npy", yp)
            np.save(f"{testdir}/{pred}/spectrumerr_{true_name}.npy", yperr)

            # show result
            if debug:
                plt.title(f"Extraction of {testdir}/{pred}/spectrum_{true_name}.npy with Spectractor")
                plt.plot(xt, yt , c='g', label='Spectrum to predict')
                plt.errorbar(xt, yp, yerr=yperr, c='r', label="Spectractor extraction")
                plt.legend()
                plt.show()



    ### On affiche le temps si c'était pas un makeonly
    if makeonly is None:
        printinfo(f"THE TIMES : {np.mean(times):.6f} += {np.std(times):.6f} s")
        printinfo(f"Total : {np.sum(times):.6f} s")
        printinfo(f"Number of good extraction : {np.sum(nbsok)} / {hp['nsimu']}")
        printinfo(f"Save {len(dictOfExceptionSpectractor)} exception(s)...")

        suffixe = "" if partition is None else f"_{partition[0]}_{len(partition)}"

        with open(f"{testdir}/spectractor_exceptions/spectractor_exceptions{suffixe}.json", "w") as f:
            json.dump(dictOfExceptionSpectractor, f, indent=4)




def viewRSpectractor(test_folder, n, testdir="./results/output_simu"):


    images = np.sort([f"{testdir}/{test_folder}/spectrum/{fimage}" for fimage in os.listdir(f"{testdir}/{test_folder}/spectrum") if ".npy" in fimage])
    name = images[n].split("_")[-1][:-4]
    print(images[n])

    xt = np.arange(300, 1100)

    yt = np.load(f"./results/output_simu/{test_folder}/spectrum/spectrum_{name}.npy")
    yp = np.load(f"./results/output_simu/{test_folder}/pred_Spectractor_x_x_0e+00/spectrum_{name}.npy")
    yp_err = np.load(f"./results/output_simu/{test_folder}/pred_Spectractor_x_x_0e+00/spectrumerr_{name}.npy")
    img = np.load(f"./results/output_simu/{test_folder}/image/image_{name}.npy")


    plt.title(f"{test_folder} n°{name}")
    plt.imshow(np.log10(img+1), cmap="gray")
    plt.xlabel("CCD [axis 0]")
    plt.ylabel("CCD [axis 1]")
    plt.show()


    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})

    plt.suptitle("Extraction de spectres ...")

    ax1.plot(xt, yt, c='k', label='Spectrum to predict')
    ax1.errorbar(xt, yp, yerr=yp_err, c='r', label='Spectractor', alpha=0.5)
    ax1.set_ylabel(f"{test_folder} n°{name}")

    ax2.axhline(0, color='k', linestyle=':')
    ax2.errorbar(xt, (yt-yp)/yp_err, yerr=1, color='r')
    ax2.set_xlabel(r"$\lambda$ (nm)")
    ax2.set_ylabel(f"Residus / err")

    ax1.legend()
    plt.show()



def compareSpec(test_folder, n, folderML, testdir="./results/output_simu"):

    images = np.sort([f"{testdir}/{test_folder}/spectrum/{fimage}" for fimage in os.listdir(f"{testdir}/{test_folder}/spectrum") if ".npy" in fimage])
    name = images[n].split("_")[-1][:-4]
    print(images[n])

    xt = np.arange(300, 1100)

    yt = np.load(f"./results/output_simu/{test_folder}/spectrum/spectrum_{name}.npy")
    yp = np.load(f"./results/output_simu/{test_folder}/pred_Spectractor_x_x_0e+00/spectrum_{name}.npy")
    yp_err = np.load(f"./results/output_simu/{test_folder}/pred_Spectractor_x_x_0e+00/spectrumerr_{name}.npy")
    ys = np.load(f"./results/output_simu/{test_folder}/pred_{folderML}/spectrum_{name}.npy")

    yp_err[yp_err < 1] = 1 # avoid 0 ...

    img = np.load(f"./results/output_simu/{test_folder}/image/image_{name}.npy")


    plt.title(f"{test_folder} n°{name}")
    plt.imshow(np.log10(img+1), cmap="gray")
    plt.xlabel("CCD [axis 0]")
    plt.ylabel("CCD [axis 1]")
    plt.show()


    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1, 1]})

    plt.suptitle("Extraction de spectres ...")

    ax1.plot(xt, yt, c='k', label='Spectrum to predict')
    ax1.errorbar(xt, yp, yerr=yp_err, c='r', label='Spectractor', alpha=0.5)
    ax1.errorbar(xt, ys, yerr=yp_err, c='b', label=folderML, alpha=0.5)
    ax1.set_ylabel(f"{test_folder} n°{name}")
    ax1.legend()

    ax2.axhline(0, color='k', linestyle='-')
    ax2.axhline(1, color='k', linestyle=':')
    ax2.axhline(-1, color='k', linestyle=':')
    ax2.errorbar(xt, (yt-yp)/yp_err, yerr=1, color='r')
    ax2.errorbar(xt, (yt-ys)/yp_err, yerr=1, color='b')
    ax2.set_ylim(-np.max(np.abs(yt-yp)/yp_err), np.max(np.abs(yt-yp)/yp_err))
    ax2.set_xlabel(r"$\lambda$ (nm)")
    ax2.set_ylabel(f"Residus / err")

    med = max(np.percentile((yt-ys)/yp_err, 99), 2)

    ax3.axhline(0, color='k', linestyle='-')
    ax3.axhline(1, color='k', linestyle=':')
    ax3.axhline(-1, color='k', linestyle=':')
    ax3.plot(xt, (yt-ys)/yp_err, color='b')
    ax3.set_ylim(-med, med)
    ax3.set_xlabel(r"$\lambda$ (nm)")
    ax3.set_ylabel(f"Residus / err")

    plt.show()





if __name__ == "__main__":

    makeonly = None
    folderML = None
    test_folder = sys.argv[1]
    nrange = None

    debug = False
    spectractor_debug = None

    for arg in sys.argv[1:]:

        if arg[:9] == "makeonly=" : makeonly = int(arg[9:])
        elif arg[:9] == "folderML=" : folderML = arg[9:]
        elif arg[:6] == "range=" : nrange = arg[6:]

        elif arg == "debugspec" : spectractor_debug = "debug"
        elif arg == "verbspec" : spectractor_debug = "verbose"
        elif arg == "debug" : debug = True



    # build partition
    if nrange is None:
        partition = None
    else:
        nbegin, nsimu = nrange.split("_")
        partition = np.arange(int(nbegin), int(nbegin) + int(nsimu))



    if "view" in sys.argv:
        if makeonly is None:
            raise Exception(f"Need makeonly value for view")
        viewRSpectractor(test_folder, makeonly)
    elif "compare" in sys.argv:
        if makeonly is None or folderML is None:
            raise Exception(f"Need makeonly and folderML (like SCaM_chi2_train16kauxtel_1e-04) for compare")
        compareSpec(test_folder, makeonly, folderML)
    else:
        apply_spectractor(test_folder, makeonly=makeonly, specver=spectractor_version, debug=debug, spectractor_debug=spectractor_debug, partition=partition)

















