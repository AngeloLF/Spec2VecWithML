import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys
from tqdm import tqdm

sys.path.append('./models/')
from get_argv import get_argv



if __name__ == "__main__":

    Args = get_argv(sys.argv[1:], prog="analyse_training")

    if f"{Args.model}_{Args.loss}" in os.listdir(f"./results/models_output"):

        if "training_evolution" in os.listdir(f"./results/models_output/{Args.model}_{Args.loss}"):
            
            if f"{Args.fulltrain_str}_{Args.lr_str}" in os.listdir(f"./results/models_output/{Args.model}_{Args.loss}/training_evolution"):

                nb_simu = len(os.listdir(f"./results/output_simu/{Args.train}/spectrum"))
                n_str = len(str(nb_simu))

                true_train_spectrum = np.load(f"./results/output_simu/{Args.train}/spectrum/spectrum_{0:0>{n_str}}.npy")
                true_valid_spectrum = np.load(f"./results/output_simu/{Args.valid}/spectrum/spectrum_{0:0>{n_str}}.npy")
                folder_evolution = f"./results/models_output/{Args.model}_{Args.loss}/training_evolution/{Args.fulltrain_str}_{Args.lr_str}"
                nb_epochs = int(len(os.listdir(folder_evolution))/2)

                print(f"Find {nb_epochs} spectrum for evolution of {Args.fullname}...")

            else:
                raise Exception(f"The training of {Args.model}_{Args.loss} is not make with {Args.fulltrain_str}_{Args.lr_str} (because not in ./results/models_output/{Args.model}_{Args.loss}/training_evolution/")

        else:
            raise Exception(f"training_evolution not in ./results/models_output/{Args.model}_{Args.loss} [weird Exception, because the creation of {Args.model}_{Args.loss} cause directyl the creation of training_evolution ...]")

    else:
        raise Exception(f"No training of {Args.model}_{Args.loss} in ./results/models_output")



    ##
    time = 10.0
    if "zoom" in sys.argv:
        nb_frame = int(nb_epochs / 10)
        time = 5.0
        suffixe = "_zoom"
    else:
        nb_frame = nb_epochs
        time = 10.0
        suffixe = ""
    fps = nb_frame / time
    x = np.arange(300, 1100)

    fig, ax = plt.subplots(2, 1)

    train_true, = ax[0].plot(x, true_spectrum, color='g', label="Train set")
    train_pred, = ax[0].plot(x, np.load(f"{folder_evolution}/train_0.npy"), c='r', label="Prediction")
    ax[0].legend()
    ax[0].set_xlabel(r"$lambdas$ (nm)")
    ax[0].set_ylabel(f"Intensity (e-)")
    ax[0].set_title(f"Evolution of {Args.model}_{Args.loss} training with {Args.fulltrain_str}_{Args.lr_str}")

    valid_true, = ax[1].plot(x, true_spectrum, color='g', label="Valid set")
    valid_pred, = ax[1].plot(x, np.load(f"{folder_evolution}/valid_0.npy"), c='r', label="Prediction")
    ax[1].legend()
    ax[1].set_xlabel(r"$lambdas$ (nm)")
    ax[1].set_ylabel(f"Intensity (e-)")
    ax[1].set_title(f"Epoch n°1")

    pbar = tqdm(total=nb_frame)

    def update(frame):
        pbar.update(1)

        # update image
        train_pred.set_ydata(np.load(f"{folder_evolution}/train_{frame}.npy"))
        valid_pred.set_ydata(np.load(f"{folder_evolution}/valid_{frame}.npy"))
        ax[1].set_title(f"Epoch n°{frame+1}")
        
        return train_pred,

    ani = animation.FuncAnimation(fig, update, frames=nb_frame, blit=False, repeat=True)

    plt.tight_layout()
    ani.save(f"./results/models_output/{Args.model}_{Args.loss}/divers_png/{Args.fulltrain_str}_{Args.lr_str}{suffixe}.gif", fps=fps, dpi=300)
    plt.close()
    pbar.close()




