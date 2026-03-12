import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys, json
from tqdm import tqdm

sys.path.append('./models/')
from get_argv import get_argv



if __name__ == "__main__":

    Args = get_argv(sys.argv[1:], prog="analyse_training")

    if f"{Args.model}_{Args.loss}" in os.listdir(f"./results/models_output"):

        if "training_evolution" in os.listdir(f"./results/models_output/{Args.model}_{Args.loss}"):
            
            if f"{Args.fulltrain_str}_{Args.lr_str}" in os.listdir(f"./results/models_output/{Args.model}_{Args.loss}/training_evolution"):

                nb_train_simu = len(os.listdir(f"./results/output_simu/{Args.train}/spectrum"))
                n_train_str = len(str(nb_train_simu))
                true_train_spectrum = np.load(f"./results/output_simu/{Args.train}/spectrum/spectrum_{0:0>{n_train_str}}.npy")
                true_train_image = np.load(f"./results/output_simu/{Args.train}/image/image_{0:0>{n_train_str}}.npy")

                nb_valid_simu = len(os.listdir(f"./results/output_simu/{Args.valid}/spectrum"))
                n_valid_str = len(str(nb_valid_simu))
                true_valid_spectrum = np.load(f"./results/output_simu/{Args.valid}/spectrum/spectrum_{0:0>{n_valid_str}}.npy")
                true_valid_image = np.load(f"./results/output_simu/{Args.valid}/image/image_{0:0>{n_valid_str}}.npy")

                folder_evolution = f"./results/models_output/{Args.model}_{Args.loss}/training_evolution/{Args.fulltrain_str}_{Args.lr_str}"
                folder_evolution_image = f"./results/models_output/{Args.model}_{Args.loss}/training_evolution_images/{Args.fulltrain_str}_{Args.lr_str}"
                nb_epochs = int(len(os.listdir(folder_evolution))/2)

                print(f"Find {nb_epochs} spectrum for evolution of {Args.fullname}...")

            else:
                raise Exception(f"The training of {Args.model}_{Args.loss} is not make with {Args.fulltrain_str}_{Args.lr_str} (because not in ./results/models_output/{Args.model}_{Args.loss}/training_evolution/")

        else:
            raise Exception(f"training_evolution not in ./results/models_output/{Args.model}_{Args.loss} [weird Exception, because the creation of {Args.model}_{Args.loss} cause directyl the creation of training_evolution ...]")

    else:
        raise Exception(f"No training of {Args.model}_{Args.loss} in ./results/models_output")




    ### Evolution of spectrum in train and valid
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

    train_true, = ax[0].plot(x, true_train_spectrum, color='g', label="Train set")
    train_pred, = ax[0].plot(x, np.load(f"{folder_evolution}/train_0.npy"), c='r', label="Prediction")
    ax[0].legend()
    ax[0].set_xlabel(r"$lambdas$ (nm)")
    ax[0].set_ylabel(f"Intensity (e-)")
    ax[0].set_title(f"Evolution of {Args.model}_{Args.loss} training with {Args.fulltrain_str}_{Args.lr_str}")

    valid_true, = ax[1].plot(x, true_valid_spectrum, color='g', label="Valid set")
    valid_pred, = ax[1].plot(x, np.load(f"{folder_evolution}/valid_0.npy"), c='r', label="Prediction")
    ax[1].legend()
    ax[1].set_xlabel(r"$lambdas$ (nm)")
    ax[1].set_ylabel(f"Intensity (e-)")
    ax[1].set_title(f"Epoch n°1")

    pbar = tqdm(total=nb_frame, desc="Spectrum evolution")

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




    ### Evolution in images in train and valid (if needed)
    if "training_evolution_images" in os.listdir(f"./results/models_output/{Args.model}_{Args.loss}"):

        with open(f"./results/output_simu/{Args.train}/hparams.json", "r") as f:
            hp = json.load(f)
            gain = hp["CCD_GAIN"]
            sigma_READ = hp["cparams"]["CCD_READ_OUT_NOISE"]

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

        fig, ax = plt.subplots(4, 1)

        pred_train_image = np.load(f"{folder_evolution_image}/train_0.npy")
        train_residus = true_train_image - pred_train_image
        train_chi2eq = train_residus**2 / (sigma_READ**2 + true_train_image / gain) * np.sign(train_residus)

        ax_train_true = ax[0].imshow(np.log10(true_train_image+1), cmap='gray')
        ax_train_pred = ax[1].imshow(np.log10(pred_train_image+1), cmap='gray')
        vmax = max(np.abs(np.min(train_residus)), np.max(train_residus))
        ax_train_resi = ax[2].imshow(train_residus, cmap='bwr', vmin=-vmax/2, vmax=vmax/2)
        vmax = max(np.abs(np.min(train_chi2eq)), np.max(train_chi2eq))
        ax_train_chi2 = ax[3].imshow(train_chi2eq, cmap='bwr', vmin=-vmax/2, vmax=vmax/2)

        ax[0].set_ylabel("True")
        ax[1].set_ylabel("Pred")
        ax[2].set_ylabel("Residus")
        ax[3].set_ylabel(f"$\\chi_2$")
        ax[0].set_title(f"Evolution of {Args.model}_{Args.loss} training with {Args.fulltrain_str}_{Args.lr_str}")

        pbar = tqdm(total=nb_frame, desc="Images evolution")

        def update(frame):
            pbar.update(1)

            pred_train_image = np.load(f"{folder_evolution_image}/train_{frame}.npy")
            train_residus = true_train_image - pred_train_image
            train_chi2eq = train_residus**2 / (sigma_READ**2 + true_train_image / gain) * np.sign(train_residus)

            # update image
            ax_train_pred.set_array(np.log10(pred_train_image+1))
            ax_train_resi.set_array(train_residus)
            ax_train_chi2.set_array(train_chi2eq)
            
            return ax_train_pred,

        ani = animation.FuncAnimation(fig, update, frames=nb_frame, blit=False, repeat=True)

        plt.tight_layout()
        ani.save(f"./results/models_output/{Args.model}_{Args.loss}/divers_png/{Args.fulltrain_str}_{Args.lr_str}{suffixe}_images.gif", fps=fps, dpi=300)
        plt.close()
        pbar.close()

        
        





