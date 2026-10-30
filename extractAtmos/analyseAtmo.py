import numpy as np
import matplotlib.pyplot as plt
import sys, json, os, shutil
import coloralf as c
from scipy import interpolate
from time import time
from copy import deepcopy

sys.path.append('./models/')
from get_argv import get_argv

sys.path.append('./analyses')
from recup_score import generate_html_table




models_colors = {
    "True"            : ["#000000", "#000000"],
    "SpectractorFile" : ["#000000", "#000000"],
    "Spectractor" : ["#888888", "#000000"],

    "SCaM"    : ["#ff0000", "#ff6666"],
    "SCaMv2"  : ["#ff007f", "#ff66b2"],
    "SotSu"   : ["#0000ff", "#6666ff"],
    "SotSuv2" : ["#00cccc", "#33ffff"],

    "SCaMD10" : ["#00ff00", "#22ff22"],
    "SCaMD20" : ["#00bb00", "#22bb22"],
    "SCaMD40" : ["#007700", "#227722"],
}




def calcul_RMS(r):

    return np.sqrt(np.nanmean(r**2))

def calcul_dRMS(r, dr):

    N = np.sum(~np.isnan(r))
    return 1 / (N * calcul_RMS(r)) * np.sqrt(np.nansum(r**2 * dr**2))

def read_atmos(file):

    with open(file, "r") as f:
        data = f.read().split("\n")

    num = np.zeros(len(data)).astype(str)
    val = np.zeros(len(data)).astype(float)

    for i, line in enumerate(data):
        num[i], val[i] = line.replace(" ", "").split(":")

    argsort = np.argsort(num)

    return num[argsort], val[argsort]

def read_score(file):

    with open(file, "r") as f:
        data = f.read().split("\n")

    num = np.zeros(len(data)).astype(str)
    val = np.zeros(len(data)).astype(float)

    for i, line in enumerate(data):
        num_i, val[i] = line.replace(" ", "").split(":")
        num[i] = num_i.split("_")[1].split(".")[0]

    argsort = np.argsort(num)

    return num[argsort], val[argsort]




def getTrueValues(hp, vp, label):

    if label == "vaod" : return vp["ATM_AEROSOLS"]
    elif label == "ozone" : return vp["ATM_OZONE"]
    elif label == "pwv" : return vp["ATM_PWV"]
    elif label == "d_ccd" : return hp["DISTANCE2CCD"]
    elif label == "angstrom_exp" : return hp["ATM_ANGSTROM_EXPONENT"]
    else : raise Exception(f"In [extractAtmo.py/getTrueValues] : label {label} unknow")




def search_missing_extraction(sysArgv, path="./results/output_simu", atmoParamFolder="atmos_params_fit"):

    Args = get_argv(sysArgv[1:], prog="analyse_atmo")

    saveFolders = [pf for pf in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}") if not "." in pf]
    saveFolders_str = list()
    allFolders = [pf for pf in os.listdir(f"{path}/{Args.test}") if not "." in pf and pf.startswith("pred_")]
    allFolders += [f"pred_True_x_x_0e+00", f"pred_SpectractorFile_x_x_0e+00"]

    for savef in saveFolders:
        if savef in allFolders:
            allFolders.remove(savef)

    ### show all miss folders
    batchs = list()
    #print(f"{c.r}Folders without atmos extraction :{c.d}")

    for pf in allFolders:

        _, model_, loss_, trainseed_, lr_ = pf.split("_")
        batch = f"extract_atmo model={model_} loss={loss_} train={trainseed_} test={Args.test} lr={lr_} load=None"
        batchs.append(batch)
        #print(f"    * {c.ti}{pf}{c.d} [{batch}]")

    return batchs








def analyseExtraction(Args, path="./results/output_simu", atmoParamFolder="atmos_params_fit", pathSave="./results/analyse/extratAtmos", colors=["r", "g", "b", "y", "m"]):

    targets = ["vaod", "ozone", "pwv"]
    hp_labels = {
        "vaod" : "ATM_AEROSOLS",
        "ozone" : "ATM_OZONE",
        "pwv" : "ATM_PWV",
    }

    nums_str = np.sort([fspectrum.split("_")[1][:-4] for fspectrum in os.listdir(f"{path}/{Args.test}/spectrum")])

    if Args.test in os.listdir(f"{pathSave}"):
        shutil.rmtree(f"{pathSave}/{Args.test}")
    os.mkdir(f"{pathSave}/{Args.test}")
    os.mkdir(f"{pathSave}/{Args.test}/html")
    os.mkdir(f"{pathSave}/{Args.test}/graph")
    os.mkdir(f"{pathSave}/{Args.test}/remember")
    os.mkdir(f"{pathSave}/{Args.test}/cross_analyse")

    os.mkdir(f"{pathSave}/{Args.test}/cross_analyse/classic linear")
    os.mkdir(f"{pathSave}/{Args.test}/cross_analyse/classic log")
    os.mkdir(f"{pathSave}/{Args.test}/cross_analyse/norma linear")
    os.mkdir(f"{pathSave}/{Args.test}/cross_analyse/norma log")

    for t in targets:
        os.mkdir(f"{pathSave}/{Args.test}/{t}")

    
    saveFolders = [pf for pf in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}") if not "." in pf]
    saveFolders_str = list()
    allFolders = [pf for pf in os.listdir(f"{path}/{Args.test}") if not "." in pf and pf.startswith("pred_")]

    full_data = dict() # {savef : {t:[list(), list()] for t in targets} for savef in saveFolders}

    for savef in saveFolders:

        if savef in allFolders:
            allFolders.remove(savef)

        if savef.startswith("pred_"):
            saveFolders_str.append("_".join(savef.split("_")[1:3] + [savef.split("_")[4]]))
        else:
            saveFolders_str.append(savef)

        rdata = {t:[np.zeros(len(nums_str)), np.zeros(len(nums_str))] for t in targets}
        os.mkdir(f"{pathSave}/{Args.test}/remember/{savef}")

        for i, n in enumerate(nums_str):

            if f"atmos_params_{n}_spectrum.json" in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}/{savef}"):

                with open(f"{path}/{Args.test}/{atmoParamFolder}/{savef}/atmos_params_{n}_spectrum.json", "r") as f:

                    data = json.load(f)

                for t in targets:
                    rdata[t][0][i] = data[t][0]
                    rdata[t][1][i] = data[t][1]

            else:

                print(f"Info [extractAtmos.py] in analyse, for {savef}, skip atmos_params_{n}_spectrum.json")
                for t in targets:
                    rdata[t][0][i] = np.nan
                    rdata[t][1][i] = np.nan

        full_data[savef] = deepcopy(rdata)


    ### show all miss folders
    print(f"{c.r}Folders without atmos extraction :{c.d}")
    for pf in allFolders:
        print(f"    * {c.ti}{pf}{c.d}")
        _, model_, loss_, trainseed_, lr_ = pf.split("_")




    ### Importation hparams & variable params
    with open(f"{path}/{Args.test}/hparams.json", "r") as fjson:
        hp = json.load(fjson)
    vp = np.load(f"{path}/{Args.test}/vparams.npz")

    save_txt = {t:list() for t in targets}
    save_all = {savef:dict() for savef in saveFolders}
    print("\nSave extract atmo performances :")

    scores = {savef:dict() for savef in saveFolders}

    for i, t in enumerate(targets):

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

        for i, savef in enumerate(saveFolders):

            plt.figure(figsize=(16, 9))

            res = full_data[savef][t][0][true_sort]-y
            dres = full_data[savef][t][1][true_sort]
            nums_str_sort = nums_str[true_sort]
            
            RMS = calcul_RMS(res)
            #STD = np.nanstd(np.abs(res))
            STD = calcul_dRMS(res, dres)
            MEAN = np.abs(np.nanmean(res))

            title = f"{savef} : {RMS}~{STD}"
            save_all[savef][t] =  [RMS, STD]
            save_txt[t].append(title)
            print(title)
            scores[savef][t] = [RMS, STD]

            plt.errorbar(x, res, yerr=dres, color="b", ls="", marker=".")
            plt.plot()
            plt.xlabel(t)
            plt.ylabel("Residus")
            plt.axhline(0, color="k", ls=":", label="True value")
            plt.title(title)
            plt.savefig(f"{pathSave}/{Args.test}/{t}/{t}_{savef}.png")
            plt.close()

            # remember each res
            with open(f"{pathSave}/{Args.test}/remember/{savef}/{t}.txt", "w") as f:

                fstr = list()

                for r, n in zip(res, nums_str_sort):
                    fstr.append(f"{n} : {r}")

                f.write('\n'.join(fstr))




    save_txt["total"] = list()

    for i, (savef, vals) in enumerate(scores.items()):

        bornes = np.array([hp["vparams"][hp_labels[t]][1] - hp["vparams"][hp_labels[t]][0] for t in targets])
        y  = np.array([vals[t][0] for t in targets]) / bornes * 100
        dy = np.array([vals[t][1] for t in targets]) / bornes * 100

        scores[savef]["total"] = [calcul_RMS(y), calcul_dRMS(y, dy)]
        save_txt["total"].append(f"{savef} : {calcul_RMS(y)}~{calcul_dRMS(y, dy)}")
        save_all[savef]["total"] = [calcul_RMS(y), calcul_dRMS(y, dy)]

        if saveFolders[i] != savef:
            raise Exception(f"Order is break") 




    for k, v in save_txt.items():
        with open(f"{pathSave}/{Args.test}/save_{k}.txt", "w") as f:
            f.write("\n".join(v))




    # Resume plots
    print(f"\n{c.m}Make resume plots{c.d}")

    for inPC in [False, True]:

        for i, t in enumerate(["pwv", "vaod", "ozone", "total"]):

            diff_borne = 1 if t not in targets or not inPC else (hp["vparams"][hp_labels[t]][1] - hp["vparams"][hp_labels[t]][0]) / 100

            plt.figure(figsize=(16, 9))

            x = np.arange(len(saveFolders))
            y = np.array([scores[savef][t][0] / diff_borne for savef in saveFolders])
            yerr = [scores[savef][t][1] / diff_borne for savef in saveFolders]

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
                plt.savefig(f"{pathSave}/{Args.test}/resume_{t}_INPC.png")
            else:
                plt.savefig(f"{pathSave}/{Args.test}/resume_{t}.png")




        # make HTML
        print(f"\n{c.m}Make HTML results INPC={inPC}{c.d}")

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
                diff_borne = 1 if target not in targets or not inPC else (hp["vparams"][hp_labels[target]][1] - hp["vparams"][hp_labels[target]][0]) / 100

                mean, std = scores[model][target]
                y[m, t] = mean / diff_borne
                e[m, t] = std / diff_borne
                x[m, t] = f" {mean/diff_borne:.3f} &plusmn {std/diff_borne:.3f} "

                tot_mean.append(mean/diff_borne)
                tot_std.append(std/diff_borne)

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

            inPC_suffixe = "" if not inPC else "_inPC"
            targets_str = targets if not inPC else [f"{target} %" for target in targets]

            with open(f"{pathSave}/{Args.test}/html/extract_atmo{sorting_str}{inPC_suffixe}.html", "w", encoding="utf-8") as f:

                html_codes = [f'<meta charset="UTF-8">', f"<h1>Extraction Atmosphere</h1>"]
                html_codes.append(generate_html_table(targets_str + ["Total", "Num", ""], saveFolders, x, y, e, sorting=sorting, path_graph=f"{pathSave}/{Args.test}/graph", subfolderOneTestPlot=""))

                f.write('\n'.join(html_codes))




    ### cross analyse
    score = "chi2"

    for savef in saveFolders:

        model = savef.split("_")[1]

        if model not in ["True", "SpectractorFile"] and os.path.exists(f"./results/analyse/{score}/{savef}/{Args.test}/remember_score_classic.txt"):

            # path to save cross figure
            pathSave_cross = f"./results/analyse/extractAtmos/{Args.test}/remember/{savef}"

            # find atmos extraction : line format : `<n> : XXX.XXX`
            path_remember_atmos = f"./results/analyse/extractAtmos/{Args.test}/remember/{savef}"
            atmos = ["vaod", "ozone", "pwv"]

            # find score : line format -> `spectrum_<n>.npy : XXX.XXX`
            path_remember_score = f"./results/analyse/{score}/{savef}/{Args.test}"
            num_score, data_score = read_score(f"{path_remember_score}/remember_score_classic.txt")
            num_score_norma, data_score_norma = read_score(f"{path_remember_score}/remember_score_norma.txt")



            for type_score in ["classic", "norma"]:

                num_score, data_score = read_score(f"{path_remember_score}/remember_score_{type_score}.txt")

                for scale in ["linear", "log"]:

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    fig.suptitle(f"Cross analyse for {savef.replace(' ', '')}", fontweight="bold")

                    for atmo, ax in zip(atmos, axes):

                        num_atmo, data_atmo = read_atmos(f"{path_remember_atmos}/{atmo}.txt")

                        if (num_score != num_atmo).all():
                            raise Exception(f"Num extract from atmo & score is not the same ...")

                        ax.plot(data_score, np.abs(data_atmo), ls="", marker="+", c="r")
                        ax.set_xscale(scale)
                        ax.set_yscale(scale)
                        ax.set_title(f"For {atmo}")
                        ax.set_xlabel(f"Score {score}")
                        ax.set_ylabel(f"Residus (abs) for {atmo}")

                    plt.tight_layout()
                    plt.savefig(f"{pathSave}/{Args.test}/cross_analyse/{type_score} {scale}/cross_analyse {savef}.png")
                    plt.close()


    # recup score chi2 in resume.txt
    type_score = "chi2"

    for savef in saveFolders:

        if "True" not in savef and "SpectractorFile" not in savef:

            resume_file = f"./results/analyse/{type_score}/{savef}/{Args.test}/resume.txt"

            with open(resume_file, "r") as f:

                classic, norma = f.read().split("\n")

                score, std = classic.split("=")[1].split("~")
                save_all[savef]["score"] = [float(score), float(std)]

                score, std = norma.split("=")[1].split("~")
                save_all[savef]["score_norma"] = [float(score), float(std)]


    # make global cross analyse figure

    for t in targets+["total"]:

        fig, axes = plt.subplots(1, 1, figsize=(12, 12))

        for savef, vals in save_all.items():

            if "True" not in savef and "SpectractorFile" not in savef:

                total, total_std = vals[t]
                score, score_std = vals["score"]

                model = savef.split("_")[1]
                color = models_colors[model][0]
                _, model_, loss_, trainseed_, lr_, *_ = pf.split("_")
                savef_str = f"{model_} {loss_} {lr_}"

                axes.errorbar(score, total, xerr=score_std, yerr=total_std, color=color)
                axes.scatter(score, total, color=color)
                axes.annotate(
                    savef_str,
                    xy=(score, total),                      # position du point
                    xytext=(4, 4),                    # décalage en points (pas en unités de données)
                    textcoords="offset points",
                    rotation=45,                      # texte penché à 45°
                    #rotation_mode="anchor",           # la rotation se fait autour du point d'ancrage
                    ha="left", va="bottom",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3",     # coins arrondis
                        facecolor=color,                  # couleur de fond depuis la liste colors
                        edgecolor="k",
                        alpha=0.8,
                    ),
                )

        plt.tight_layout()
        plt.savefig(f"{pathSave}/{Args.test}/cross_analyse/cross_analyse_global @ {t}.png")
        plt.close()





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













