import os, sys, shutil
import numpy as np
from tqdm import tqdm
import coloralf as c
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace
import seaborn as sns
import pandas as pd




models_colors = {
    "SCaM"    : ["#ff0000", "#ff6666"],
    "SCaMv2"  : ["#ff007f", "#ff66b2"],
    "SotSu"   : ["#0000ff", "#6666ff"],
    "SotSuv2" : ["#00cccc", "#33ffff"],

    "SCaMD10" : ["#00ff00", "#22ff22"],
    "SCaMD20" : ["#00bb00", "#22bb22"],
    "SCaMD40" : ["#007700", "#227722"],
}

lrs_markers = {
    "1e-04" : "o",
    "5e-05" : "s",
    "1e-05" : "v",
    "5e-06" : "d",
    "1e-06" : "*",
}




def is_valid_model(ARGV, model):

    _, model_str, loss_str, train_str, lr_str, *_ = model.split("_")
    seed_str= train_str.split("-")[-1]

    return not (model_str in ARGV.xmodel or seed_str in ARGV.xseed or "." in model_str)

def is_valid_test(ARGV, test):

    return not ("." in test or test in ARGV.xtest)

def recup_mt(ARGV):

    """
        Recupere les noms de modèles et tests
    """

    models = [f"pred_Spectractor_x_x_0e+00"]
    tests = list()

    for score in ARGV.score_type:

        models += [m for m in os.listdir(f"./results/analyse/{score}") if "." not in m]

        for model in models:
            tests += [t.split("-")[0] for t in os.listdir(f"./results/analyse/{score}/{model}") if "." not in t] # on retire la seed du label du test

    # selection models
    models_retenu = list()
    for m in set(models):

        if is_valid_model(ARGV, m):
            models_retenu.append(m)
        else:
            print(f"{c.ly}INFO : Model not recup : {m}{c.d}")

    # selection tests
    tests_retenu = list()
    for t in set(tests):

        if is_valid_test(ARGV, t):
            tests_retenu.append(t)
        else:
            print(f"{c.ly}INFO : Test not recup : {t}{c.d}")

    return models_retenu, tests_retenu




def del_seed(text):

    ele = text.split(" ")
    ele[2] = ele[2].split("-")[0]

    return " ".join(ele)




def general_plot(x, y, ymin, ymax, title="<title>", di=0.1, savefig_name=None):

    """
        Make a general plots with all model and Spectractor

        Parameters:
            * x [list of str] : list with models name (with seed)
            * y [numpy.array of float] : scores of models
            * ymin [numpy.array of float] : min scores of models
            * ymax [numpy.array of float] : max scores of models
            * title [str] : title of the plot
            * savefig_name [str] : path to save. If `None`, with `plt.show()` and no saving 
    """

    # x2r : r are models name WITHOUT seeds
    x2r = dict()

    # r2s : s are list of 3 list : y, ymin, ymax for each r
    r2s = dict()
    r = list()

    # Spectractor data
    spectractor_scores = None

    for xi, yi, ymini, ymaxi in zip(x, y, ymin, ymax):

        if "Spectractor" in xi:
            spectractor_scores = [yi, ymini, ymaxi]

        else:
            ri = del_seed(xi)

            if ri not in r2s.keys():
                r2s[ri] = [list(), list(), list()]
                r.append(ri)

            x2r[xi] = ri
            r2s[ri][0].append(yi)
            r2s[ri][1].append(ymini)
            r2s[ri][2].append(ymaxi)

    r = np.array(r)
    s = np.array([np.mean(r2s[ri][0]) for ri in r])
    nseed = len(r2s[r[0]][0])
    di0 = di/2*nseed

    args = np.argsort(s)
    r = r[args]
    s = s[args]

    plt.figure(figsize=(16, 12))

    for i, (ri, si) in enumerate(zip(r, s)):

        modeli = ri.split(" ")[0]

        plt.scatter(i-di0, si, color=models_colors[modeli][0])

        if i == 0:
            title += f" [best {ri} with {si:.3f}]"

        for j, (yj, yminj, ymaxj) in enumerate(zip(*r2s[ri])):

            plt.scatter(i-di0+di*(j+1), yj, color=models_colors[modeli][1], marker="+")
            plt.plot([i-di0+di*(j+1)]*2, [yminj, ymaxj], color=models_colors[modeli][1])

    if spectractor_scores is not None and not np.isnan(spectractor_scores[0]) and spectractor_scores[0] != np.inf:

        xs = np.arange(len(r))
        x1 = np.ones(len(r))
        if spectractor_scores[2] - spectractor_scores[1] > 1e-6:
            # plt.fill_between(xs, x1*spectractor_scores[1], x1*spectractor_scores[2], color="k")
            plt.axhspan(spectractor_scores[1], spectractor_scores[2], color="k", alpha=0.2)
        plt.axhline(spectractor_scores[0], color="k", label=f"Spectractor with {spectractor_scores[0]:.3f}")
        plt.legend()

    plt.xticks(np.arange(len(r)), r, rotation=90)
    plt.title(title)
    plt.tight_layout()
    if savefig_name is not None:
        plt.savefig(savefig_name)
        plt.close()
    else:
        plt.show()




def oneTest_plot(col, x, y, ystd, title="<title>", savefig_name=None, di=0.1):
    
    # x2r : r are models name WITHOUT seeds
    x2r = dict()

    # r2s : s are list of 3 list : y, std for each r
    r2s = dict()
    r = list()

    # Spectractor data
    spectractor_scores = None

    for xi, yi, ystdi in zip(x, y, ystd):

        if "Spectractor" in xi:
            spectractor_scores = [yi, ystdi]

        else:
            ri = del_seed(xi)

            if ri not in r2s.keys():
                r2s[ri] = [list(), list()]
                r.append(ri)

            x2r[xi] = ri
            r2s[ri][0].append(yi)
            r2s[ri][1].append(ystdi)

    r = np.array(r)
    s = np.array([np.mean(r2s[ri][0]) for ri in r])
    nseed = len(r2s[r[0]][0])
    di0 = di/2*nseed

    args = np.argsort(s)
    r = r[args]
    s = s[args]

    # remember model for legend
    models_use = list()

    plt.figure(figsize=(16, 12))

    # plots model scores
    for i, (ri, si) in enumerate(zip(r, s)):

        modeli = ri.split(" ")[0]
        if modeli not in models_use : models_use.append(modeli)

        plt.scatter(i-di0, si, color=models_colors[modeli][0])

        if i == 0:
            title += f" [best {ri} with {si:.3f}]"

        for j, (yj, ystdj) in enumerate(zip(*r2s[ri])):

            if not np.isnan(yj) and yj != np.inf:

                plt.scatter(i-di0+di*(j+1), yj, color=models_colors[modeli][1], marker="+")
                plt.errorbar([i-di0+di*(j+1)], yj, yerr=ystdj, color=models_colors[modeli][1])

    # plots Spectractor score if avaiable
    if spectractor_scores is not None and not np.isnan(spectractor_scores[0]) and spectractor_scores[0] != np.inf:

        xs = np.arange(len(r))
        x1 = np.ones(len(r))
        plt.axhspan(max(0, spectractor_scores[0]-spectractor_scores[1]), spectractor_scores[0]+spectractor_scores[1], color="k", alpha=0.2)
        plt.axhline(spectractor_scores[0], color="k", label=f"Spectractor with {spectractor_scores[0]:.3f}")

    # legend for models
    for mu in models_use:
        plt.scatter(None, None, color=models_colors[mu][0], label=mu)

    plt.xticks(np.arange(len(r)), r, rotation=90)
    plt.title(title)
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    if savefig_name is not None:
        plt.savefig(savefig_name)
        plt.close()
    else:
        plt.show()




def generate_html_table(colonnes, lignes, text, y, e, sorting=False, marker='.', score="---", typeScore="---", path_graph=f"results/analyse/graph", markers=None, colors=None, absSorting=False):

    """
        Créer le fichier HTML pour visualiser la les scores produit par les analyses.
    """


    # Pour trier par le score
    if sorting:

        if not absSorting:
            index = np.argsort(y[:, -3])
        else:
            index = np.argsort(np.abs(y[:, -3]))

        y = y[index]
        e = e[index]
        text = text[index]
        lignes = [lignes[i][5:].replace("_", " ") for i in index]
        # lignes4graph = [ligne for ligne in lignes if not ("cal" in ligne and not "wc" in ligne)]

        mean_scores = y[:, -3]
        std_scores = e[:, -3]

        ynan = np.copy(y)
        ynan[~np.isfinite(ynan)] = np.nan

        min_scores = np.nanmin(ynan[:, :-3], axis=1)
        max_scores = np.nanmax(ynan[:, :-3], axis=1)

        general_plot(lignes, mean_scores, min_scores, max_scores, title=f"Score {score} ({typeScore})", savefig_name=f"{path_graph}/{score}_{typeScore}.png")

        for i, col in enumerate(colonnes[:-3]):
            oneTest_plot(col, lignes, y[:, i], e[:, i]/2, title=f"Score {score} ({typeScore})", savefig_name=f"{path_graph}/{score}_{typeScore}/{col}.png")


    # Definition du CSS (qui sera directement integrer dans le HTML, pas de fichier à coté tant pis)
    tds = {
        "def" : "",
            
        "far_min" : 'background-color: #CCFFCC;',
        "near_min" : 'background-color: #66FF66;',
        "min" : 'background-color: #00CC00; font-weight: bold;',

        "far_max" : 'background-color: #FFCCCC;',
        "near_max" : 'background-color: #FF6666;',
        "max" : 'background-color: #CC0000; font-weight: bold;',

        "nan"   : 'background-color: #888888;',
    }


    # Vérifie rapidement que les lignes et colonnes coincide
    if text.shape != (len(lignes), len(colonnes)):
        raise ValueError("Les dimensions de y ne correspondent pas aux longueurs des listes ligne et colonne.")


    # on démarre l'HTML, puis on commence l'entête
    html = '<head>\n  <meta charset="UTF-8"/>\n</head>\n<table border="1" style="border-collapse: collapse; text-align: center;">\n'
    html += '  <tr>\n    <th></th>'  # Coin supérieur gauche vide
    for col in colonnes[:-2]:

        if col != "Total":
            html += f'\n    <th style="width: 150px;"> {col.replace("test", "")} </th>'
        else:
            html += f'\n    <th style="width: 150px; border-left: 4px double black;"> {col} </th>'

    for col in colonnes[-2:]:

        html += f'\n    <th style="width: 50px;"> {col.replace("test", "")} </th>'
    
    html += '\n  </tr>\n'


    # calcule les extremums 
    buffer_y = np.copy(y)
    buffer_y[y == np.inf] = np.nan

    argmin, argmax = np.zeros(buffer_y.shape[1]) + np.nan, np.zeros(buffer_y.shape[1]) + np.nan
    valmin, valmax = np.zeros(buffer_y.shape[1]) + np.nan, np.zeros(buffer_y.shape[1]) + np.nan

    for k in range(buffer_y.shape[1]):

        if not np.all(np.isnan(buffer_y[:, k])):

            argmin[k], argmax[k] = np.nanargmin(buffer_y[:, k]), np.nanargmax(buffer_y[:, k])
            valmin[k], valmax[k] = np.nanmin(buffer_y[:, k]),    np.nanmax(buffer_y[:, k])


    # Lignes de données
    for i, ligne in enumerate(lignes):

        html += f'  <tr>\n    <th> {ligne} </th>'

        for j, col in enumerate(colonnes):

            style2add = list()

            # si la colonne est 'Total', alors on fait un double trait a gauche
            if col == "Total":
                style2add.append("border-left: 4px double black;")

            if np.isnan(buffer_y[i, j]) : style2add.append(tds["nan"])
            elif i == argmin[j] : style2add.append(tds["min"])
            elif i == argmax[j] : style2add.append(tds["max"])
            elif y[i, j] < valmin[j] * 1.2 : style2add.append(tds["near_min"])
            elif y[i, j] < valmin[j] * 1.5 : style2add.append(tds["far_min"])
            elif y[i, j] < valmin[j] / 1.5 : style2add.append(tds["far_max"])
            elif y[i, j] > valmax[j] / 1.2 : style2add.append(tds["near_max"])
            else : pass


            html += f'\n    <td style="{" ".join(style2add)}"> {text[i, j]} </td>'

        html += '\n  </tr>\n'
    
    html += '</table>'

    return html




def make_score(score_type, models, tests, seed4spectractor):

    """
        Fonction principale, repère les scores, génère l'HTML ...
    """
    

    # iteration sur les type de score (L1, chi2, ...)
    for score in score_type:

        print(f"\n{c.g}Recup data for score : {c.tu}{score}{c.d}")

        # Sorting lists
        models.sort()

        l = np.zeros(len(models)) + np.inf
        lstr = np.zeros(len(models)).astype(str)
        lstr[:] = '---'
        mstr = np.zeros(len(models)).astype(str)
        mstr[:] = '---'
        lrstr = np.zeros(len(models)).astype(str)
        lrstr[:] = '---'
        all_loss_str = list()

        y = np.zeros((2, len(models), len(tests)+3)) + np.inf
        e = np.zeros((2, len(models), len(tests)+3)) + np.inf
        x = np.zeros((2, len(models), len(tests)+3)).astype(str)
        x[:, :] = '---'

        
        # iteration sur les models 
        for m, model in enumerate(models):

            _, model_str, loss_str, train_str, lr_str, *_ = model.split("_")
            if loss_str not in all_loss_str and loss_str != "x" : all_loss_str.append(loss_str)


            # remember loss & seed recup
            if "Spectractor" in model:

                seed_detected = seed4spectractor

                l[m] = np.nan

            else:

                seed_detected = train_str.split("-")[-1]

                loss_npy = f"results/models_output/{model_str}_{loss_str}/loss/{train_str}_{lr_str}.npy"
                loss_best = np.min(np.load(loss_npy))
                l[m] = loss_best
                lstr[m] = loss_str
                mstr[m] = model_str
                lrstr[m] = lr_str


            print(f"    {c.lm}* model {model} {c.m}[seed:{seed_detected}]{c.d}")
            tot_mean = [list(), list()]
            tot_std = [list(), list()]


            # iteration sur les tests
            for t, test_without_seeds in enumerate(tests):

                test = test_without_seeds + "-" + seed_detected

                if model in os.listdir(f"{path_analyse}/{score}") and test in os.listdir(f"{path_analyse}/{score}/{model}"):

                    print(f"        - extract test {test_without_seeds}{c.lk}-{seed_detected}{c.d}")

                    with open(f"{path_analyse}/{score}/{model}/{test}/resume.txt", "r") as f:
                        data = f.read().split("\n")[:-1]

                    for i, line in enumerate(data):

                        label, score_i = line.split("=")
                        mean, std = score_i.split("~")
                        mean = float(mean)
                        std = float(std)

                        if score in ["L1"]:
                            mean *= 100
                            std *= 100

                        y[i, m, t] = mean
                        e[i, m, t] = std
                        x[i, m, t] = f"{mean:.2f} ± {std:.2f}"
                        if score == "L1"     : x[i, m, t] = f"{mean:.1f} ± {std:.1f}"
                        elif score == "chi2" : x[i, m, t] = f"{mean:.2f} ± {std:.2f}"
                        else : raise Exception(f"Score {score} unknow")

                        tot_mean[i].append(mean)
                        tot_std[i].append(std)

                elif test.split("-")[-1] == seed_detected:


                    if model in os.listdir(f"results/output_simu/{test}") and len(os.listdir(f"results/output_simu/{test}/{model}")) > 1:

                        x[0, m, t] = "Not analyse"
                        x[1, m, t] = "Not analyse"
                        print(f"{c.lk}        - Not analyse find : {path_analyse}/{score}/{model}/{test}/resume.txt{c.d}")

                    else:

                        x[0, m, t] = "Not apply"
                        x[1, m, t] = "Not apply"
                        print(f"{c.lk}        - Not apply find : results/output_simu/{test}/{model}{c.d}")



                else:

                    # la seed n'est pas la même donc c'est ok
                    pass


            for i in range(2):

                mom = np.mean(tot_mean[i])
                soa = np.sum(np.array(tot_std[i])**2)**0.5
                y[i, m, -3] = mom
                e[i, m, -3] = soa
                if score == "L1"     : x[i, m, -3] = f"{mom:.1f} ± {soa:.1f}"
                elif score == "chi2" : x[i, m, -3] = f"{mom:.2f} ± {soa:.2f}"
                else : raise Exception(f"Score {score} unknow")


        # classic and norma
        for i in range(2):

            y[i, :, -3][np.isnan(y[i, :, -3])] = np.inf
            nb_m = len(y[i, :, -3])
            order = np.zeros(nb_m)

            for m, cl in enumerate(np.argsort(y[i, :, -3])):

                order[cl] = m

            order_norma = order / (nb_m-1) * 100

            y[i, :, -1] = order_norma + 100
            x[i, :, -1] = [f"{o:.0f} %" for o in order_norma]

            y[i, :, -2] = order_norma + 100
            x[i, :, -2] = [f"{1+o:.0f}" for o in order]


        for sorting, sorting_str in [(False, ""), (True, "_sorting")]:

            with open(f"{path_resume}/html/{score}{sorting_str}.html", "w", encoding="utf-8") as f:

                html_codes = [f"<h1>Score {score}</h1>"]

                for i, typeScore in enumerate(["classic", "norma"]):

                    html_codes.append(f"\n\n<h2>{typeScore}</h2>")
                    html_codes.append(generate_html_table(tests+["Total", "Num", ""], models, x[i], y[i], e[i], sorting=sorting, score=score, typeScore=typeScore, path_graph=f"{path_resume}/graph"))#title=f"Score {score} ({typeScore})", savefig_name=f"{path_resume}/graph/{score}_{typeScore}"))

                    # loss_plots
                    for t, test_without_seeds in enumerate(tests):

                        for li in all_loss_str:

                            os.makedirs(f"{path_resume}/graph/{score}_loss/{li}", exist_ok=True)

                            mask_loss = (lstr == li)
                            colors = [models_colors[mi][0] for mi in mstr[mask_loss]]
                            markers = [lrs_markers[lri][0] for lri in lrstr[mask_loss]]

                            plt.figure(figsize=(16, 12))

                            for xi, yi, mi, ci in zip(l[mask_loss], y[i, :, t][mask_loss], markers, colors):
                                plt.scatter(xi, yi, ls="", marker=mi, c=ci)

                            # legend for models
                            for mu in np.unique(mstr[mask_loss]):
                                plt.scatter(None, None, color=models_colors[mu][0], marker=".", label=mu)

                            # legend for models
                            for lru in np.unique(lrstr[mask_loss]):
                                plt.scatter(None, None, color="k", marker=lrs_markers[lru], label=f"LR={lru}")

                            plt.xlabel(f"Best loss")
                            plt.ylabel(f"Score {score} [{typeScore}]")
                            plt.yscale("log")
                            plt.title(f"Loss {li} for score {score} [{typeScore}] in {test_without_seeds}")
                            plt.tight_layout()
                            plt.legend()
                            plt.savefig(f"{path_resume}/graph/{score}_loss/{li}/{typeScore} {test_without_seeds}.png")
                            plt.close()


                f.write('\n'.join(html_codes))








if __name__ == "__main__":
    """

    ARGV needed:
        * specseed : seed use for Spectractor scores

    ARGV option:
        * score_type : list of score type to compute (seperate with `,`)(def : L1,chi2)
        * xtest : list of test to not recup (seperate with `,`)
        * xseed : list of seed to not recup (seperate with `,`)
        * xmodel : list of model to not recup (seperate with `,`) 

    """


    # read argv
    ARGS = SimpleNamespace(_argx=list(), xtest=list(), xmodel=list(), xseed=list(), score_type=["L1", "chi2"])
    arg2split = ["xtest", "xseed", "xmodel"]
    argv = sys.argv[1:]

    for arg in argv:

        if "=" in arg:

            k, v = arg.split("=")

            if k in arg2split : ARGS.__setattr__(k, v.split(","))
            else : ARGS.__setattr__(k, v) 

        else:

            ARGS._argx.append(arg)


    # models & tests 
    models, tests = recup_mt(ARGS)

    if len(models) == 0 or len(models) == 1 and "Spectractor" in models[0]:
        print(f"{c.r}ERROR : no models find{c.d}")
        sys.exit()

    if len(models) == 0:
        print(f"{c.r}ERROR : no test find{c.d}")
        sys.exit()

    print(f"{c.y}INFO : finding models : ", ", ".join(models), c.d)
    print(f"{c.y}INFO : finding tests folders : ", ", ".join(tests), c.d)


    # seed for spectractor
    if "specseed" not in dir(ARGS):
        raise Exception(f"Need specseed argv, for the choosen seed for Spectractor scores")
    else:
        seed4spectractor = ARGS.specseed
        print(f"Seed choose for Spectractor : {seed4spectractor}")

    path_analyse = f"./results/analyse"
    path_resume = f"{path_analyse}/all_resume"

    if 'all_resume' in os.listdir(path_analyse) : shutil.rmtree(path_resume)
    os.makedirs(path_resume, exist_ok=True)
    os.makedirs(f"{path_resume}/graph", exist_ok=True)
    os.makedirs(f"{path_resume}/html", exist_ok=True)
    for st in ARGS.score_type:
        os.makedirs(f"{path_resume}/graph/{st}_classic", exist_ok=True)
        os.makedirs(f"{path_resume}/graph/{st}_norma", exist_ok=True)
        os.makedirs(f"{path_resume}/graph/{st}_loss", exist_ok=True)


    

    make_score(ARGS.score_type, models, tests, seed4spectractor)



