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

sys.path.append('./analyses')
from recup_score import generate_html_table




def getTrueValues(hp, vp, label):

    if label == "vaod" : return vp["ATM_AEROSOLS"]
    elif label == "ozone" : return vp["ATM_OZONE"]
    elif label == "pwv" : return vp["ATM_PWV"]
    elif label == "d_ccd" : return hp["DISTANCE2CCD"]
    elif label == "angstrom_exp" : return hp["ATM_ANGSTROM_EXPONENT"]
    else : raise Exception(f"In [extractAtmo.py/getTrueValues] : label {label} unknow")



def analyseExtraction(Args, path="./results/output_simu", atmoParamFolder="atmos_params_fit", pathSave="./results/analyse/extratAtmos", colors=["r", "g", "b", "y", "m"]):

    targets = ["vaod", "ozone", "pwv"]
    nums_str = np.sort([fspectrum.split("_")[1][:-4] for fspectrum in os.listdir(f"{path}/{Args.test}/spectrum")])

    if Args.test in os.listdir(f"{pathSave}"):
        shutil.rmtree(f"{pathSave}/{Args.test}")
    os.mkdir(f"{pathSave}/{Args.test}")
    os.mkdir(f"{pathSave}/{Args.test}/html")

    for t in targets:
        os.mkdir(f"{pathSave}/{Args.test}/{t}")

    
    saveFolders = [pf for pf in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}") if not "." in pf]
    saveFolders_str = list()

    full_data = dict() # {savef : {t:[list(), list()] for t in targets} for savef in saveFolders}

    for savef in saveFolders:

        if savef.startswith("pred_"):
            saveFolders_str.append("_".join(savef.split("_")[1:3] + [savef.split("_")[4]]))
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
        print(f"\n{c.m}For {t}{c.d}")

        true_vals = getTrueValues(hp, vp, t)
        if t in ["ozone", "vaod", "pwv"]:
            true_sort = np.argsort(true_vals)
            x = true_vals[true_sort]
            y = true_vals[true_sort]
        else:
            true_sort = np.arange(len(nums_str))
            x = np.arange(len(nums_str))
            y = true_vals


        for mode in ["plot"]: #, "subplot", "full"]:

            for i, savef in enumerate(saveFolders):

                plt.figure(figsize=(16, 9))

                res = full_data[savef][t][0][true_sort]-y
                
                RMS = np.sqrt(np.nanmean(res**2))
                MEAN = np.abs(np.nanmean(res))
                STD = np.nanstd(res)

                title = f"{savef} : MEAN={MEAN:.3f} RMS={RMS:.3f} STD={STD:.3f}"
                save_txt += f"{title}\n"
                print(title)
                scores[savef][t] = [MEAN, RMS]

                plt.errorbar(x, res, yerr=full_data[savef][t][1][true_sort], color="b", ls="", marker=".")
                plt.plot()
                plt.xlabel(t)
                plt.ylabel("Residus")
                plt.axhline(0, color="k", ls=":", label="True value")
                plt.title(title)
                # plt.ylim(np.nanmin(res), np.nanmax(res))
                plt.savefig(f"{pathSave}/{Args.test}/{t}/{t}_{savef}.png")
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


    # Resume plots
    print(f"\n{c.m}Make resume plots{c.d}")

    for inPC in [False, True]:

        for i, (t, borne) in enumerate(zip(["pwv", "vaod", "ozone", "total"], [borne_PWV, borne_VAOD, borne_OZONE, None])):

            plt.figure(figsize=(16, 9))

            x = np.arange(len(saveFolders))
            divide = (borne[1] - borne[0])/100. if inPC and borne is not None else 1.0
            y = np.array([scores[savef][t][0] / divide for savef in saveFolders])
            yerr = [scores[savef][t][1] / divide for savef in saveFolders]

            argsort_y = np.argsort(np.abs(y))

            plt.axhline(0, color="k", ls=":")
            plt.errorbar(x, y[argsort_y], yerr=yerr, color=colors[i], ls="", marker=".")
            plt.xticks(x, np.array(saveFolders_str)[argsort_y], rotation=45)
            if inPC or t == "total":
                plt.ylabel(f"Residus {t} (%)")
            else:
                plt.ylabel(f"Residus {t}")

            plt.tight_layout()
            if inPC:
                plt.savefig(f"{pathSave}/{Args.test}/resume_{t}_INPC_.png")
            else:
                plt.savefig(f"{pathSave}/{Args.test}/resume_{t}.png")




    # make HTML
    print(f"\n{c.m}Make HTML results{c.d}")

    y = np.zeros((len(saveFolders), len(targets)+3)) + np.inf
    e = np.zeros((len(saveFolders), len(targets)+3)) + np.inf
    x = np.zeros((len(saveFolders), len(targets)+3)).astype(str)
    x[:, :] = '---'

    
    for m, model in enumerate(saveFolders):

        print(f"    model {model}")

        tot_mean = list()
        tot_std = list()

        for t, target in enumerate(targets):

            print(f"        test {target}")

            mean, std = scores[model][target]
            y[m, t] = mean
            e[m, t] = std
            x[m, t] = f" {mean:.3f} &plusmn {std:.3f} "

            tot_mean.append(mean)
            tot_std.append(std)

        y[m, -3] = scores[model]['total'][0]
        e[m, -3] = scores[model]['total'][1]
        x[m, -3] = f"{scores[model]['total'][0]:.2f} &plusmn {scores[model]['total'][1]:.2f}"




    y[:, -3][np.isnan(y[:, -3])] = np.inf
    nb_m = len(y[:, -3])
    order = np.zeros(nb_m)

    for m, cl in enumerate(np.argsort(y[:, -3])):

        order[cl] = m

    order_norma = order / (nb_m-1) * 100

    y[:, -1] = order_norma + 100
    x[:, -1] = [f"{o:.2f} %" for o in order_norma]

    y[:, -2] = order_norma + 100
    x[:, -2] = [f"{1+o}" for o in order]


    for sorting, sorting_str in [(False, ""), (True, "_sorting")]:

        with open(f"{pathSave}/{Args.test}/html/extract_atmo{sorting_str}.html", "w", encoding="utf-8") as f:

            html_codes = [f'<meta charset="UTF-8">', f"<h1>Extraction Atmosphere</h1>"]
            html_codes.append(generate_html_table(targets+["Total", "Classement (N)", "Classement (%)"], saveFolders_str, x, y, sorting=sorting))

            f.write('\n'.join(html_codes))





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













