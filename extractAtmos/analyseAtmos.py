import numpy as np
import matplotlib.pyplot as plt
import sys, json, os, shutil
import coloralf as c
# import alftool as alf
from scipy import interpolate
from time import time
from copy import deepcopy

sys.path.append('./models/')
from get_argv import get_argv






def getTrueValues(hp, vp, label):

    if label == "vaod" : return vp["ATM_AEROSOLS"]
    elif label == "ozone" : return vp["ATM_OZONE"]
    elif label == "pwv" : return vp["ATM_PWV"]
    elif label == "d_ccd" : return hp["DISTANCE2CCD"]
    else : raise Exception(f"In [extractAtmo.py/getTrueValues] : label {label} unknow")



def analyseExtraction(Args, path="./results/output_simu", atmoParamFolder="atmos_params_fit", pathSave="./results/analyse/extratAtmos", colors=["r", "g", "b", "y", "m"]):

    targets = ["vaod", "ozone", "pwv", "d_ccd"]
    nums_str = np.sort([fspectrum.split("_")[1][:-4] for fspectrum in os.listdir(f"{path}/{Args.test}/spectrum")])

    if Args.test in os.listdir(f"{pathSave}"):
        shutil.rmtree(f"{pathSave}/{Args.test}")
    os.mkdir(f"{pathSave}/{Args.test}")

    for t in targets:
        os.mkdir(f"{pathSave}/{Args.test}/{t}")

    
    saveFolders = [pf for pf in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}") if not "." in pf]
    saveFolders_str = list()

    full_data = dict() # {savef : {t:[list(), list()] for t in targets} for savef in saveFolders}

    for savef in saveFolders:

        if savef.startswith("pred_"):
            saveFolders_str.append("_".join(savef.split("_")[1:3]))
        else:
            saveFolders_str.append(savef)

        rdata = {t:[np.zeros(len(nums_str)), np.zeros(len(nums_str))] for t in targets}

        for i, n in enumerate(nums_str):

            if f"atmos_params_{n}_spectrum.json" in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}/{savef}"):

                with open(f"{path}/{Args.test}/{atmoParamFolder}/{savef}/atmos_params_{n}_spectrum.json", "r") as f:

                    data = json.load(f)

                for t in targets:
                    rdata[t][0][i] = data[t][0]
                    rdata[t][1][i] = data[t][1]

            else:

                print(f"Info [extractAtmos.py] in analyse, skip atmos_params_{n}_spectrum.json")
                for t in targets:
                    rdata[t][0][i] = np.nan
                    rdata[t][1][i] = np.nan

        full_data[savef] = deepcopy(rdata)



    ### Importation hparams & variable params
    with open(f"{path}/{Args.test}/hparams.json", "r") as fjson:
        hp = json.load(fjson)
    vp = np.load(f"{path}/{Args.test}/vparams.npz")

    save_txt = "Save extract atmo performances :\n"
    print("\nSave extract atmo performances :")

    scores = {savef:dict() for savef in saveFolders}

    for i, t in enumerate(targets):

        save_txt += f"\n{t}\n"
        print(f"\n{t}")

        true_vals = getTrueValues(hp, vp, t)
        if t in ["ozone", "vaod", "pwv"]:
            true_sort = np.argsort(true_vals)
            x = true_vals[true_sort]
            y = true_vals[true_sort]
        else:
            true_sort = np.arange(len(nums_str))
            x = np.arange(len(nums_str))
            y = true_vals


        for mode in ["plot", "subplot", "full"]:

            plt.figure(figsize=(16, 9))

            for i, savef in enumerate(saveFolders):

                # color
                if savef == "true":
                    color = "k"
                elif i < len(colors):
                    color = colors[i]
                else:
                    color = None

                res = full_data[savef][t][0][true_sort]-y
                
                score = np.nanmean(np.abs(res))
                std = np.nanstd(np.abs(res))
                score_mean = np.nanmean(res)
                score_std = np.nanstd(res)

                if mode == "plot":
                    save_txt += f"{savef} : {score:.3f} +- {std:.3f} --- {score_mean:.3f} +- {score_std:.3f}\n"
                    print(f"{savef} : {score:.3f} +- {std:.3f} --- {score_mean:.3f} +- {score_std:.3f}")
                    scores[savef][t] = [score, std]

                if mode != "full":
                    if mode == "subplot" : plt.subplot(2, 2, i+1)
                    plt.errorbar(x, res, yerr=full_data[savef][t][1][true_sort], color=color, ls="", marker=".", label=f"{savef} : {score:.3f}")
                    plt.plot()
                    plt.xlabel(t)
                    plt.ylabel("Residus")
                    plt.axhline(0, color="k", ls=":", label="True value")
                    plt.title(f"{savef} : residus abs = {score:.3f}$\pm${std:.3f} [mean={score_mean:.3f}$\pm${score_std:.3f}]")
                    plt.ylim(np.nanmin(res), np.nanmax(res))
                    if mode == "plot": 
                        plt.savefig(f"{pathSave}/{Args.test}/{t}/{t}_{savef}.png")
                        plt.close()
                else:
                    plt.plot(x, res, color=color, ls="", marker=".")

            if mode == "full":
                plt.xlabel(t)
                plt.ylabel("Residus")
                plt.axhline(0, color="k", ls=":", label="True value")
                plt.legend()
            
                plt.tight_layout()
                plt.savefig(f"{pathSave}/{Args.test}/full_{t}.png")
                plt.close()

            elif mode == "subplot":
                plt.tight_layout()
                plt.savefig(f"{pathSave}/{Args.test}/subplot_{t}.png")
                plt.close()

    with open(f"{pathSave}/{Args.test}/save_extraction_score.txt", "w") as f:
        f.write(save_txt)

    borne_PWV = hp["vparams"]["ATM_PWV"]
    borne_VAOD = hp["vparams"]["ATM_AEROSOLS"]
    borne_OZONE = hp["vparams"]["ATM_OZONE"]

    for savef, vals in scores.items():

        o, v, p = vals["ozone"], vals["vaod"], vals["pwv"]
        scores[savef]["total"] = [
            (o[0] / (borne_OZONE[1] - borne_OZONE[0]) + v[0] / (borne_VAOD[1] - borne_VAOD[0]) + p[0] / (borne_PWV[1] - borne_PWV[0])) * 100., # en %
            np.sqrt((o[1] / (borne_OZONE[1] - borne_OZONE[0]))**2 + (v[1] / (borne_VAOD[1] - borne_VAOD[0]))**2 + (p[1] / (borne_PWV[1] - borne_PWV[0]))**2) * 100., # en %
        ]


    for inPC in [False, True]:

        plt.figure(figsize=(16, 9))

        for i, (t, borne) in enumerate(zip(["pwv", "vaod", "ozone", "total"], [borne_PWV, borne_VAOD, borne_OZONE, None])):

            x = np.arange(len(saveFolders))
            divide = (borne[1] - borne[0])/100. if inPC and borne is not None else 1.0
            y = [scores[savef][t][0] / divide for savef in saveFolders]
            yerr = [scores[savef][t][1] / divide for savef in saveFolders]

            plt.subplot(2, 2, i+1)
            plt.errorbar(x, y, yerr=yerr, color=colors[i], ls="", marker=".")
            plt.xticks(x, saveFolders)
            if inPC or t == "total":
                plt.ylabel(f"{t} (%)")
            else:
                plt.ylabel(f"{t}")

        plt.tight_layout()
        if inPC:
            plt.savefig(f"{pathSave}/{Args.test}/resume_all_INPC.png")
        else:
            plt.savefig(f"{pathSave}/{Args.test}/resume_all.png")






if __name__ == "__main__":

    """
    For extract atmos with spectractor minimisation !
    From pred spectrum or :
        * true : using true spectrum from simulation
        * spectractorfile : using spectrum.fits product of spectractor extraction (different from pred_Spectractor_x_x_0e+00 -> 2 interpolation)
    """

    # arguments needed
    path = "./results/output_simu"
    pathSave = "./results/analyse/extractAtmos"
    atmoParamFolder = "atmos_params_fit"
    Args = get_argv(sys.argv[1:], prog="analyse_atmo")

    # Build arborescence if needed
    if "results" not in os.listdir():
        os.mkdir(f"./results")
    if "analyse" not in os.listdir(f"./results"):
        os.mkdir(f"./results/analyse")
    if "extractAtmos" not in os.listdir(f"./results/analyse"):
        os.mkdir(pathSave)

    analyseExtraction(Args, path, atmoParamFolder, pathSave)













