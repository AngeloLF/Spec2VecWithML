import os, sys, shutil
import numpy as np
from tqdm import tqdm
import coloralf as c
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace
import seaborn as sns
import pandas as pd






def recup_mt(scores, mode="dispo"):

    models = list()
    tests = list()

    for score in scores:

        models = [m for m in os.listdir(f"./results/analyse/{score}") if not "." in m]

        for model in models:
            tests += [t for t in os.listdir(f"./results/analyse/{score}/{model}") if not "." in t]

    return list(set(models)), list(set(tests))








def generate_html_table(colonnes, lignes, text, y, sorting=False, marker='.', savefig_name=None, markers=None, colors=None, score=None):


    if sorting:

        index = np.argsort(y[:, -3])

        y = y[index]
        text = text[index]
        lignes = [lignes[i] for i in index]
        lignes4graph = [ligne for ligne in lignes if not ("cal" in ligne and not "wc" in ligne)]

        # y[y == np.inf] = np.nan

        """
        for color_palette, palette in colors.items():

            valSpectractor = None

            for zoom, zoom_str in [(lignes4graph, ""), (lignes4graph[:32], "zoom_")]:

                plt.figure(figsize=(19, 10))

                for i, name in enumerate(zoom):

                    xg = np.ones(len(colonnes)-2) * i
                    yg = y[i][:-2]

                    color = 'k'
                    for pal, col in palette.items():
                        if   color_palette != "learningRate" and pal in name      : color = col
                        elif color_palette == "learningRate" and pal == name[-5:] : color = col

                    plt.plot(xg, yg, color=color)

                    plt.scatter([i], yg[-1], color=color, marker="s")

                    if "Spectractor_x" in name and valSpectractor is None : valSpectractor = y[i][-3]
                
                for pal, col in palette.items():
                    plt.scatter([], [], color=col, marker='s', label=pal.replace("_", ""))

                if valSpectractor is not None : plt.axhline(valSpectractor, color='k', linestyle=':')
                plt.xticks(np.arange(len(zoom)), zoom, rotation=90)
                plt.tight_layout()
                plt.yscale("log")
                plt.legend()
                # plt.show()
                try:
                    plt.savefig(f"{savefig_name}_{zoom_str}{score}_{color_palette}.png")
                except:
                    plt.close()"""



    tds = {
        "def" : "td",
            
        "far_min" : 'td style="background-color: #CCFFCC;"',
        "near_min" : 'td style="background-color: #66FF66;"',
        "min" : 'td style="background-color: #00CC00; font-weight: bold;"',

        "far_max" : 'td style="background-color: #FFCCCC;"',
        "near_max" : 'td style="background-color: #FF6666;"',
        "max" : 'td style="background-color: #CC0000; font-weight: bold;"',

        "nan" : 'td style="background-color: #000000;"',
    }

    if text.shape != (len(lignes), len(colonnes)):
        raise ValueError("Les dimensions de y ne correspondent pas aux longueurs des listes ligne et colonne.")

    html = '<table border="1" style="border-collapse: collapse; text-align: center;">\n'
    
    # En-tête
    html += '  <tr><th></th>'  # Coin supérieur gauche vide
    for col in colonnes:
        html += f'<th>{col}</th>'
    html += '</tr>\n'


    buffer_y = np.copy(y)
    buffer_y[y == np.inf] = np.nan

    argmin, argmax = np.zeros(buffer_y.shape[1]) + np.nan, np.zeros(buffer_y.shape[1]) + np.nan
    valmin, valmax = np.zeros(buffer_y.shape[1]) + np.nan, np.zeros(buffer_y.shape[1]) + np.nan

    for k in range(buffer_y.shape[1]):

        # print(buffer_y[:, k])

        if not np.all(np.isnan(buffer_y[:, k])):

            argmin[k], argmax[k] = np.nanargmin(buffer_y[:, k]), np.nanargmax(buffer_y[:, k])
            valmin[k], valmax[k] = np.nanmin(buffer_y[:, k]),    np.nanmax(buffer_y[:, k])

    # print(len(argmin), buffer_y.shape)

    # Lignes de données
    for i, ligne in enumerate(lignes):
        html += f'  <tr><th>{ligne}</th>'
        for j in range(len(colonnes)):

            if   i == argmin[j] : td = tds["min"]
            elif i == argmax[j] : td = tds["max"]
            elif y[i, j] < valmin[j] * 1.2 : td = tds["near_min"]
            elif y[i, j] < valmin[j] * 1.5 : td = tds["far_min"]
            elif y[i, j] < valmin[j] / 1.5 : td = tds["far_max"] 
            elif y[i, j] > valmax[j] / 1.2 : td = tds["near_max"] 
            else : td = tds["def"]

            if np.isnan(buffer_y[i, j]) : td = tds["nan"]

            html += f'<{td}>{text[i, j]}</td>'
        html += '</tr>\n'
    
    html += '</table>'
    return html






def make_score(score_type, models, tests):
 
    for score in score_type:

        print(f"Make score {score}")

        # Sorting lists
        models.sort()


        y = np.zeros((2, len(models), len(tests)+3)) + np.inf
        e = np.zeros((2, len(models), len(tests)+3)) + np.inf
        x = np.zeros((2, len(models), len(tests)+3)).astype(str)
        x[:, :] = '---'

        
        for m, model in enumerate(models):

            print(f"    model {model}")

            tot_mean = [list(), list()]
            tot_std = [list(), list()]

            for t, test in enumerate(tests):

                print(f"        test {test}")

                if model in os.listdir(f"{path_analyse}/{score}") and test in os.listdir(f"{path_analyse}/{score}/{model}"):

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
                        x[i, m, t] = f"{mean:.2f} ~ {std:.2f}"
                        if score == "L1"     : x[i, m, t] = f"{mean:.2f} ~ {std:.2f}"
                        elif score == "chi2" : x[i, m, t] = f"{mean:.4f} ~ {std:.4f}"
                        else : raise Exception(f"Score {score} unknow")

                        tot_mean[i].append(mean)
                        tot_std[i].append(std)

                else:

                    print(f"{c.lk}        -> Not find : {path_analyse}/{score}/{model}/{test}/resume.txt{c.d}")


            for i in range(2):

                mom = np.mean(tot_mean[i])
                soa = np.sum(np.array(tot_std[i])**2)**0.5
                y[i, m, -3] = mom
                e[i, m, -3] = soa
                if score == "L1"     : x[i, m, -3] = f"{mom:.2f} ~ {soa:.2f}"
                elif score == "chi2" : x[i, m, -3] = f"{mom:.6f} ~ {soa:.6f}"
                else : raise Exception(f"Score {score} unknow")


        for i in range(2):

            y[i, :, -3][np.isnan(y[i, :, -3])] = np.inf
            nb_m = len(y[i, :, -3])
            order = np.zeros(nb_m)

            for m, cl in enumerate(np.argsort(y[i, :, -3])):

                order[cl] = m

            order_norma = order / (nb_m-1) * 100

            y[i, :, -1] = order_norma + 100
            x[i, :, -1] = [f"{o:.2f} %" for o in order_norma]

            y[i, :, -2] = order_norma + 100
            x[i, :, -2] = [f"{1+o}" for o in order]


        for sorting, sorting_str in [(False, ""), (True, "_sorting")]:

            with open(f"{path_resume}/html/{score}{sorting_str}.html", "w") as f:

                html_codes = [f"<h1>Score {score}</h1>"]

                for i, typeScore in enumerate(["classic", "norma"]):

                    html_codes.append(f"<h2>{typeScore}</h2>")
                    html_codes.append(generate_html_table(tests+["Total", "Classement (N)", "Classement (%)"], models, x[i], y[i], sorting=sorting, savefig_name=f"{path_resume}/graph/zzz", score=score))

                f.write('\n'.join(html_codes))






if __name__ == "__main__":

    score_type = ["L1", "chi2"]
    path_analyse = f"./results/analyse"
    path_resume = f"{path_analyse}/all_resume"

    if 'all_resume' in os.listdir(path_analyse) : shutil.rmtree(path_resume)
    os.makedirs(path_resume, exist_ok=True)
    os.makedirs(f"{path_resume}/graph", exist_ok=True)
    os.makedirs(f"{path_resume}/html", exist_ok=True)

    models, tests = recup_mt(score_type)
    print(models)
    print(tests)

    make_score(score_type, models, tests)



