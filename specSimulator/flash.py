import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time, sys, platform
import coloralf as c

if platform.system() == "Darwin":
    print(f"{c.r}WARNING : macOS detected. Package mouse doesnt work for mac : flash.py need an update{c.d}")
    sys.exit()

import keyboard
import mouse

from simulator import SpecSimulator
import utils_spec.psf_func as pf
import hparameters as hp




def give_simulator():

    psf_function = {
        # For l lambdas in nm :
        # f : def func of (XX, YY, amplitude, x, y, f_argv[0](l, *argv[0]), ..., f_argv[n](l, *argv[n]))
        'f' : pf.moffat2d_jit,

        # function for argument
        'f_arg' : [pf.simpleLinear, pf.simpleLinear],
        
        # argument for argument function
        'arg' : [[3.0], [3.0]],

        # argument order 0
        'order0' : {'amplitude':22900.0, 'arg':[3.0, 2.0]},

        # timbre size function
        'timbre' : pf.moffat2d_timbre,
    }

    return SpecSimulator(psf_function, {}, input_argv=sys.argv[1:]+["f=flash"], with_noise=True, output_dir="output_simu", output_fold=f"simulation")


def show_var_params(sim):


    print(f"{c.g}\nParameters :{c.d}")

    for hpi in list(hp.VARIABLE_PARAMS.keys()) + ["TARGET"]:

        print(f"{c.y}{hpi}{c.d} : {c.ly}{c.ti}{sim.__getattribute__(hpi)}{c.d}")



def make_simu(sim, ax, ax_spec, ax_img, ax_labels):

    sim.makeSim(0)

    spectrum = np.load(f"./results/output_simu/flash/spectrum/spectrum_0.npy") / ccd_gain

    if not sim.colorSimu: 
        image = np.load(f"./results/output_simu/flash/image/image_0.npy")
        ax_img.set_data(np.log10(image+1))
    else: 
        image = np.load(f"./results/output_simu/flash/imageRGB/imageRGB_0.npy") / len(sim.lambdas)
        image /= np.max(image)
        ax_img.set_data(image)

    ax_spec.set_ydata(spectrum)
    

    ax[0].set_ylim(0, np.max(spectrum)*1.01)
    target_name, disperser_name, amplitude = sim.__getattribute__('TARGET'), sim.__getattribute__('disperser_name'), sim.__getattribute__('A') 
    ax[0].set_title(f"Target : {target_name} | Disperser : {disperser_name} | Amplitude x{amplitude:.2f} | Rota. Angle : {sim.__getattribute__('ROTATION_ANGLE'):.2f}° | $\gamma$ : {sim.psf_function['arg'][0][0]:.1f}")

        
    titleImage = ["order 0"]
    if sim.As[1] > 0.0 : titleImage.append(f"Order 1")
    if sim.As[2] > 0.0 : titleImage.append("Order 2")
    if np.abs(sim.__getattribute__('ROTATION_ANGLE')) > 0.01 : titleImage.append(f"Rotation")
    if sim.with_adr : titleImage.append(f"ADR")
    if sim.psf_function['arg'][0][0] > 1.0 : titleImage.append(f"PSF")
    if sim.with_background : titleImage.append(f"Back.")
    if sim.with_noise : titleImage.append(f"Noise")

    ax[1].set_title(f"With {', '.join(titleImage)}")


    for label, axl in ax_labels.items():
        axl.set_label(f"{label[4:]} = {sim.__getattribute__(label):.2f}")

    ax[0].legend()


def on_press(event):
    global lastKey, ckeys
    # print(f"Touche pressée (callback) type event: {event.event_type}")
    lastKey = event.name

    if event.event_type == keyboard.KEY_DOWN and event.name not in ckeys:
        # print(f"{c.lk}KEY DOWN : Adding {event.name} ... {c.d}") 
        ckeys.add(event.name)
    if event.event_type == keyboard.KEY_UP: 
        # print(f"{c.lk}KEY UP : Removing {event.name} .. {c.d}") 
        ckeys.remove(event.name)


def timeFormat(t):

    hh, t = t // 3600, t % 3600
    mm, t = t // 60, t % 60
    ss, t = int(t), t - int(t)
    ms = int(np.round(t*1000))

    return f"{hh:02.0f}:{mm:02.0f}:{ss:02}.{ms:03}"



def new_simu(key2var, optionParam, ope, sim, ax, ax_spec, ax_img, ax_labels):

    any_change = False

    for key, (var_name, _, (vmin, vmax, step)) in key2var.items():

        if key in ckeys and key not in ["t", "d"]:

            if key != "g" : current_val = sim.__getattribute__(var_name)
            else : current_val = sim.psf_function['arg'][0][0]
            
            new_val = current_val + step if ope == "+" else current_val - step

            if new_val < vmin : new_val = vmin
            if new_val > vmax : new_val = vmax

            print(f"{c.y}Change {var_name} : {current_val:.2f} -> {new_val:2f}{c.d}")

            if key != "g" : sim.__setattr__(var_name, new_val)
            else : sim.psf_function['arg'][0][0] = new_val
            any_change = True



        elif key in ckeys and key == "t":

            new_val = key2var["t"][2][0] + step if ope == "+" else key2var["t"][2][0] - step
            new_val = new_val % len(key2var["t"][2][1]) 

            key2var["t"][2][0] = new_val

            sim.__setattr__("TARGET", key2var["t"][2][1][new_val])
            print(f"{c.y}Change {var_name} : {key2var['t'][2][1][new_val]} [{new_val}]{c.d}")
            any_change = True

        elif key in ckeys and key == "d":

            new_val = key2var["d"][2][0] + step if ope == "+" else key2var["d"][2][0] - step
            new_val = new_val % len(key2var["d"][2][1])
            key2var["d"][2][0] = new_val

            sim.set_new_disperser(key2var['d'][2][1][key2var['d'][2][0]])
            print(f"{c.y}Change {var_name} : {key2var['d'][2][1][key2var['d'][2][0]]} [{new_val}]{c.d}")
            any_change = True


    for key, (func, attr) in optionParam.items():

        if key in ckeys:
            sim.__getattribute__(func)()
            print(f"{c.y}Change {attr} to {sim.__getattribute__(attr)}{c.d}")
            any_change = True


    if "+" in ckeys : ope = "+"
    if "-" in ckeys : ope = "-"

    if "s" in ckeys : show_var_params(sim)
    if "h" in ckeys : show_help(key2var, optionParam)

    if "*" in ckeys : any_change = True

    if any_change : make_simu(sim, ax, ax_spec, ax_img, ax_labels)

    return ope



def show_help(key2var, optionParam):

    print(f"{c.g}\nLes changement dans la simu : {c.d}")

    for key, (var_name, ekey, (vmin, vmax, step)) in key2var.items():

        print(f"Clé {c.ly}{key}{c.d} : {c.y}{var_name[:ekey]}{c.d}{c.ly}{c.tu}{var_name[ekey]}{c.d}{c.y}{var_name[ekey+1:]}{c.d}")

    for key, (func, attr) in optionParam.items():

        print(f"Clé {c.ly}{key}{c.d} : func {c.y}{func}{c.d}")



if __name__ == "__main__":


    # var = ["A", "ROTATION_ANGLE", "ATM_AEROSOLS", "ATM_OZONE", "ATM_PWV", "ATM_AIRMASS", "TARGET"]

    target_set = "set0"
    ccd_gain = 3. # e / ADu

    for argv in sys.argv[1:]:

        if argv[:3] == "set" : target_set = argv



    targets = hp.TARGETS_NAME[target_set]


    key2var = {
        "a" : ["A",              0, [0.5, 1.5,  0.02]],
        "r" : ["ROTATION_ANGLE", 0, [-4.0, 4.0, 0.20]],
        "e" : ["ATM_AEROSOLS",   5, [0.0, 1.0,  0.02]],
        "o" : ["ATM_OZONE",      4, [0.0, 400,  10.0]],
        "p" : ["ATM_PWV",        4, [0.0, 15.0, 0.20]],
        "i" : ["ATM_AIRMASS",    5, [1.0, 2.5,  0.10]],
        "t" : ["TARGET",         0, [0,   targets, 1]],
        "g" : ["GAMMA",          0, [1.0, 10.0, 0.50]],
        "d" : ["Disperser",      0, [0, ["HoloAmAg", "HoloPhP"], 1]],
    }

    optionParam = {

        "1" : ["change_A1", "As"],
        "2" : ["change_A2", "As"],
        "3" : ["change_adr", "with_adr"],
        "4" : ["change_back", "with_background"],
        "5" : ["change_noise", "with_noise"],
        "6" : ["change_RGB", "colorSimu"],
        
    }

    bbox_ope = {
        "+" : dict(boxstyle="round,pad=0.2", fc="green", ec="black", lw=1),
        "-" : dict(boxstyle="round,pad=0.2", fc="red", ec="black", lw=1),
    }

    ope = "+"

    sim = give_simulator()
    sim.variable_params = dict()
    sim.__setattr__('TARGET', targets[key2var["t"][2][0]])
    print(f"Set disperser : {key2var['d'][2][1][0]}")
    sim.set_new_disperser(key2var["d"][2][1][0])
    sim.makeSim(0)



    X, lastKey, ckeys = None, None, set()
    x, y = mouse.get_position()
    dx, dy = 0.1, 0.1

    xl = np.linspace(300, 1100, 800)

    # Activer le mode interactif de Matplotlib
    plt.ion()

    # Créer une figure et un axe
    fig, ax = plt.subplots(2, 1)

    anno = ax[0].annotate(f"--- fps", (650, -10), color='k', rotation=0, bbox=dict(boxstyle="round,pad=0.2", fc="gray", ec="black", lw=1))
    opeo = ax[0].annotate(f"{ope}", (750, -10), color='k', rotation=0, bbox=bbox_ope[ope])

    ax_labels = {
        "ATM_AEROSOLS" : ax[0].scatter([], [], marker='*', label=f"AEROSOLS = {sim.__getattribute__(f'ATM_AEROSOLS'):.2f}"),
        "ATM_OZONE" : ax[0].scatter([], [], marker='*', label=f"OZONE = {sim.__getattribute__(f'ATM_OZONE'):.2f}"),
        "ATM_PWV" : ax[0].scatter([], [], marker='*', label=f"PWV = {sim.__getattribute__(f'ATM_PWV'):.2f}"),
        "ATM_AIRMASS" : ax[0].scatter([], [], marker='*', label=f"AIRMASS = {sim.__getattribute__(f'ATM_AIRMASS'):.2f}"),
    }

    spectrum = np.load(f"./results/output_simu/flash/spectrum/spectrum_0.npy")
    image = np.load(f"./results/output_simu/flash/image/image_0.npy")
    
    ax_spec, = ax[0].plot(xl, spectrum / ccd_gain, color='k')
    ax_img = ax[1].imshow(np.log10(image+1), cmap='gray')

    ax[0].set_xlabel(f"$\lambda$ (nm)")
    ax[0].set_ylabel(f"$ADU$")
    ax[1].set_xlabel(f"CCD axis 0")
    ax[1].set_ylabel(f"CCD axis 1")

    ax[0].set_xlim(300, 1100)
    ax[0].set_ylim(0, np.max(spectrum)*1.01)
    target_name, disperser_name, amplitude = sim.__getattribute__('TARGET'), sim.__getattribute__('disperser_name'), sim.__getattribute__('A') 
    ax[0].set_title(f"None")
    ax[1].set_title(f"None")
    ax[0].legend()


    for key in mpl.rcParams.keys():
        if key.startswith('keymap.'):
            mpl.rcParams[key] = ''


    arg2val = {"argv":list()}
    for arg in sys.argv[1:]:

        if '=' in arg:
            k, v = arg.split('=')
            arg2val[k] = v
        else:
            arg2val["argv"].append(arg)


    fps_fix = 30 if not "fps" in arg2val.keys() else int(arg2val["fps"])
    spf_fix = 1 / fps_fix
    print(f"{c.lg}Set fps to {fps_fix}{c.d}")

    # keyboard.on_press(on_press)
    # keyboard.on_release(on_press)
    keyboard.hook(on_press)

    # Boucle de mise à jour
    i = 0
    t0 = time.time()
    tInit = time.time()

    make_simu(sim, ax, ax_spec, ax_img, ax_labels)

    while True:

        spf = time.time()-t0
        if spf < spf_fix : time.sleep(spf_fix - spf)
        fps = 1 / (time.time()-t0)
        anno.set_text(f"{np.round(fps):.0f}")
        # Redessiner la figure

        # make_key_action(sim, ax, ax_spec, ax_img)
        ope = new_simu(key2var, optionParam, ope, sim, ax, ax_spec, ax_img, ax_labels)
        opeo.set_text(ope)
        opeo.set_bbox(bbox_ope[ope])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        t0 = time.time()

        # Incrémenter le compteur
        i += 1

        if lastKey is not None:
            X = lastKey
            lastKey = None
            
            # print(f"{timeFormat(time.time()-tInit)} >>> {c.r}Press of `{X}` {c.d} -> {ckeys}")


        # xi, yi = mouse.get_position()
        # if x != xi and y != yi:
        #     x, y = xi, yi
        #     print(f"{timeFormat(time.time()-tInit)} >>> {c.b}Move to {x}, {y}{c.d}")
            # set_pnt_mouse(pnt, x/100, y/100)

        # Ajouter une petite pause pour contrôler la vitesse de l'animation
        # time.sleep(0.01)

        # Ajouter une condition de sortie (facultatif)
        if X == "esc":
            break

    # Garder la fenêtre ouverte à la fin (si la boucle se termine)
    plt.close()
    plt.ioff()
    plt.show()