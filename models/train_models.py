import os, sys, shutil, importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import coloralf as c
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('./models/')
from get_argv import get_argv, get_device
import params
from losses import give_Loss_Function, Chi2Loss





def load_from_pretrained(model_name, loss_name, folder_pretrain, prelr, device, is_pret=False):

    model, Custom_dataloader = load_model(model_name, device)

    MODEL_W = f"{params.out_path}/{params.out_dir}/{model_name}_{loss_name}/{params.out_states}/{folder_pretrain}_{prelr}_best.pth"
    print(f"{c.ly}INFO : Loading {model_name} with w. : {c.tu}{MODEL_W}{c.ru} ... {c.d}")

    state = torch.load(MODEL_W, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    print(f"{c.ly}Loading ok{c.d}")

    return model, Custom_dataloader



def load_model(model_name, device, path2architecture='./models/'):

    if model_name is None:

        print(f"{c.r}WARNING : model name is not define (model=<model_name>){c.d}")
        raise ValueError("Model name error")

    else:
        sys.path.append(path2architecture)

        module_name = f"{model_name}"

        print(f"{c.y}INFO : Import module {module_name} ...{c.d}")
        module = importlib.import_module(f"architecture.{module_name}")

        print(f"{c.y}INFO : Import model {model_name}_Model et le dataloader {model_name}_Dataset ...{c.d}")
        Custom_model = getattr(module, f"{model_name}_Model")
        Custom_dataloader = getattr(module, f"{model_name}_Dataset")

        model = Custom_model().to(device)      

        return model, Custom_dataloader



def load_model_from_Args(args):

    device = get_device(args)

    if args.from_pre:
        model, Custom_dataloader = load_from_pretrained(args.model, args.loss, args.pre_train, args.pre_lr_str, device)
    else: 
        model, Custom_dataloader = load_model(args.model, device)

    return model, Custom_dataloader, device



def load_optim(args):

    if optim_name == "Adam" : return optim.Adam()



def default_training(Args, device, train_loader, valid_loader, loss_function):

    ### Define losses

    # Train loss
    train_list_loss = np.zeros(Args.epochs)
    valid_list_loss = np.zeros(Args.epochs)

    # MSE loss
    mse_loss = nn.MSELoss()
    train_list_loss_mse = np.zeros(Args.epochs)
    valid_list_loss_mse = np.zeros(Args.epochs)

    # Chi2 loss
    chi2_loss = give_Loss_Function("chi2", Args.model)
    train_list_loss_chi2 = np.zeros(Args.epochs)
    valid_list_loss_chi2 = np.zeros(Args.epochs)

    # Save lr
    lrates = np.zeros(Args.epochs)

    best_val_loss = np.inf
    best_state = None



    ### Boucle d'entraînement

    for epoch in range(Args.epochs):


        ### Training 

        train_loss = 0.0
        train_loss_mse = 0.0
        train_loss_chi2 = 0.0

        for images, spectra in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Args.epochs} (Train)"):

            # Make the training
            model.train()
            images = images.to(device)
            spectra = spectra.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, spectra)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            # Evaluate with mse and chi2 loss
            model.eval()
            train_loss_mse += mse_loss(outputs, spectra) * images.size(0)
            train_loss_chi2 += chi2_loss(outputs, spectra) * images.size(0)



        train_loss = train_loss / len(train_loader)
        train_list_loss[epoch] = train_loss
        train_list_loss_mse[epoch] = train_loss_mse / len(train_loader)
        train_list_loss_chi2[epoch] = train_loss_chi2 / len(train_loader)

        # Predict of first train
        model.eval()
        pred_train0 = model(Args.train0_img).cpu().detach().numpy()[0]
        np.save(f"{Args.output.evolution_here}/train_{epoch}.npy", pred_train0)


        ### Validation

        model.eval()
        valid_loss = 0.0
        valid_loss_mse = 0.0
        valid_loss_chi2 = 0.0
        
        with torch.no_grad():

            for images, spectra in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{Args.epochs} (Validation)"):

                # classique validation
                images = images.to(device)
                spectra = spectra.to(device)
                outputs = model(images)
                loss = loss_function(outputs, spectra)
                valid_loss += loss.item() * images.size(0)

                # Evaluate with mse and chi2 loss
                model.eval()
                valid_loss_mse += mse_loss(outputs, spectra) * images.size(0)
                valid_loss_chi2 += chi2_loss(outputs, spectra) * images.size(0)

        valid_loss = valid_loss / len(valid_loader)
        # Predict of first train
        model.eval()
        pred_valid0 = model(Args.valid0_img).cpu().detach().numpy()[0]
        np.save(f"{Args.output.evolution_here}/valid_{epoch}.npy", pred_valid0)

        # Show epoch
        lrates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"Epoch [{epoch+1}/{Args.epochs}], loss train = {c.g}{train_loss:.6f}{c.d}, val loss = {c.r}{valid_loss:.6f}{c.d} | LR={c.y}{lrates[epoch]:.2e}{c.d}")
        with open(f"{Args.output.epoch_here}/INFO - epoch {epoch+1} - {Args.epochs} - {train_loss:.6f} , {valid_loss:.6f}", "wb") as f : pass

        # save state at each epoch to be able to reload and continue the optimization
        if valid_loss < best_val_loss:

            best_val_loss = valid_loss
            best_state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val_loss}

        # save loss
        valid_list_loss[epoch] = valid_loss
        valid_list_loss_mse[epoch] = valid_loss_mse / len(valid_dataset)
        valid_list_loss_chi2[epoch] = valid_loss_chi2 / len(valid_dataset)



    # dict of statistiques
    run_stats = {
        "train" : train_list_loss, "train_mse" : train_list_loss_mse,
        "valid" : valid_list_loss, "valid_mse" : valid_list_loss_mse,
        "lrates" : lrates
    }

    return best_state, run_stats











if __name__ == "__main__":



    ### capture params
    Args = get_argv(sys.argv[1:], prog="training")



    ### Define some params
    name = f"{Args.model}_{Args.loss}" # Ex. : SCaM_chi2
    batch_size = params.batch_size_def if Args.model not in params.batch_size_models.keys() else params.batch_size_models[Args.model]
    model, Custom_dataloader, device = load_model_from_Args(Args)
    loss_function = give_Loss_Function(Args.loss, Args.model, f"{params.path}/{Args.train}", device)
    if "particular_training" in dir(model):
        print(f"{c.ly}{c.tb}INFO : Find particular training function{c.d}")
        training_function = model.particular_training
    else:
        print(f"{c.ly}INFO : default training function{c.d}")
        training_function = default_training



    ### Define optimizer
    optim_name = params.optim_def if Args.model not in params.optim_models.keys() else params.optim_models[Args.model]
    if   optim_name == "Adam"  : optimizer = optim.Adam(model.parameters(), lr=Args.lr)
    elif optim_name == "AdamW" : optimizer = optim.AdamW(model.parameters(), lr=Args.lr)
    else : raise Exception(f"{c.r}The optimizer `optim_name` unknow. Please select Adam or AdamW.")



    ### Definition of paths
    path = params.path                                             # Ex. : ./results/output_simu
    Args.train_name = f"{Args.from_prefixe}{Args.train}_{Args.lr_str}"  # Ex. : (pre_)(trainCalib_1e-4_)train16k_1e-04
    Args.full_out_path = f"{params.out_path}/{params.out_dir}/{name}"   # Ex. : ./results/models_output/SCaM_chi2

    Args.train_inp_dir = f"{path}/{Args.train}/{model.folder_input}"    # Ex. : ./results/output_simu/train16k/image
    Args.train_out_dir = f"{path}/{Args.train}/{model.folder_output}"   # Ex. : ./results/output_simu/train16k/spectrum
    Args.valid_inp_dir = f"{path}/{Args.valid}/{model.folder_input}"    # Ex. : ./results/output_simu/valid2k/image
    Args.valid_out_dir = f"{path}/{Args.valid}/{model.folder_output}"   # Ex. : ./results/output_simu/valid2k/spectrum

    # Define path of losses
    Args.output = SimpleNamespace()
    Args.output.loss       = f"{Args.full_out_path}/{params.out_loss}"       # Ex. : ./results/models_output/SCaM_chi2/loss
    Args.output.loss_mse   = f"{Args.full_out_path}/{params.out_loss_mse}"   # Ex. : ./results/models_output/SCaM_chi2/loss_mse
    Args.output.loss_png   = f"{Args.full_out_path}/{params.out_loss_png}"   # Ex. : ./results/models_output/SCaM_chi2/loss_png -> save loss on png
    Args.output.state      = f"{Args.full_out_path}/{params.out_states}"     # Ex. : ./results/models_output/SCaM_chi2/states
    Args.output.divers     = f"{Args.full_out_path}/{params.out_divers}"     # Ex. : ./results/models_output/SCaM_chi2/divers
    Args.output.divers_png = f"{Args.full_out_path}/{params.out_divers_png}" # Ex. : ./results/models_output/SCaM_chi2/divers_png -> save divers on png

    # find tel
    if "ctio" in Args.train : tel = "ctio"
    elif "auxtel" in Args.train : tel = "auxtel"
    else : tel = None

    # Create folder in case ...
    os.makedirs(f"{params.out_path}/{params.out_dir}", exist_ok=True) # Ex. : ./results/models_output
    os.makedirs(Args.full_out_path, exist_ok=True)                         # Ex. : ./results/models_output/SCaM_chi2
    for f in dir(Args.output):
        if not f.startswith("__") and not f.endswith("__"): 
            os.makedirs(getattr(Args.output, f), exist_ok=True)

    # Folder for push epoch
    Args.output.epoch = f"{Args.full_out_path}/{params.out_epoch}"      # Ex. : ./results/models_output/SCaM_chi2/epoch
    os.makedirs(Args.output.epoch, exist_ok=True)
    Args.output.epoch_here = f"{Args.output.epoch}/{Args.train_name}"   # Ex. : ./results/models_output/SCaM_chi2/epoch/train16k_1e-04
    if Args.train_name in os.listdir(Args.output.epoch) : shutil.rmtree(Args.output.epoch_here)
    os.mkdir(Args.output.epoch_here)

    # Folder for traning evolution
    Args.output.evolution = f"{Args.full_out_path}/{params.out_evolution}"      # Ex. : ./results/models_output/SCaM_chi2/training_evolution
    os.makedirs(Args.output.evolution, exist_ok=True)
    Args.output.evolution_here = f"{Args.output.evolution}/{Args.train_name}"   # Ex. : ./results/models_output/SCaM_chi2/training_evolution/train16k_1e-04
    if Args.train_name in os.listdir(Args.output.evolution) : shutil.rmtree(Args.output.evolution_here)
    os.mkdir(Args.output.evolution_here)







    ### Data set loading

    # Créer le Dataset complet
    train_dataset = Custom_dataloader(Args.train_inp_dir, Args.train_out_dir)
    valid_dataset = Custom_dataloader(Args.valid_inp_dir, Args.valid_out_dir)

    # Créer les DataLoaders pour l'entraînement et la validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Keep first img / spectrum file for train & valid
    Args.train0_img = train_dataset[0][0].unsqueeze(0).to(device)
    Args.valid0_img = valid_dataset[0][0].unsqueeze(0).to(device)

    # Some print
    print(f"{c.ly}INFO : Size of the loaded train dataset : {c.d}{c.y}{len(train_dataset)}{c.d}")
    print(f"{c.ly}INFO : Size of the loaded valid dataset : {c.d}{c.y}{len(valid_dataset)}{c.d}")
    print(f"{c.ly}INFO : Telescope detected               : {c.d}{c.y}{tel}{c.d}")
    print(f"{c.ly}INFO : Utilisation de l'appareil        : {c.d}{c.y}{device}{c.d}")
    print(f"{c.ly}INFO : Optimizer                        : {c.d}{c.y}{optim_name}{c.d}")
    print(f"{c.ly}INFO : Model architecture               : {c.d}{c.y}{name}{c.d}")
    print(f"{c.ly}INFO : Name                             : {c.d}{c.y}{Args.train_name}{c.d}")
    print(f"{c.ly}INFO : Train                            : {c.d}{c.y}{Args.train}{c.d}")
    print(f"{c.ly}INFO : Valid                            : {c.d}{c.y}{Args.valid}{c.d}")
    print(f"{c.ly}INFO : Epoch                            : {c.d}{c.y}{Args.epochs}{c.d}")
    print(f"{c.ly}INFO : Lrate                            : {c.d}{c.y}{Args.lr}{c.d}")
    print(f"{c.ly}INFO : batch size                       : {c.d}{c.y}{batch_size}{c.d}")
    print(f"{c.ly}INFO : Number of parameters             : {c.d}{c.y}{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f} millions{c.d}")






    ### Training :
    best_state, run_stats = training_function(Args, device, train_loader, valid_loader, loss_function)

    




    ### save everything
    print("Savings ...")


    # Saving loss
    print(f"{c.lm}INFO : Save losses (& plots) ... {c.d}")
    np.save(f"{Args.output.loss}/{Args.train_name}.npy", np.array((run_stats["train"], run_stats["valid"])))
    np.save(f"{Args.output.loss_mse}/{Args.train_name}.npy", np.array((run_stats["train_mse"], run_stats["valid_mse"])))

    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(1, Args.epochs+1), run_stats["train"], c="k", label="Train loss")
    plt.plot(np.arange(1, Args.epochs+1), run_stats["valid"], c="g", label="Valid loss")
    plt.axvline(best_state["epoch"], c="g", ls=":", label=f"Best state at valid loss = {best_state['best_val_loss']:.3e}")
    plt.yscale("log")
    plt.title(f"Loss for {name} train with {Args.train_name}")
    plt.legend()
    plt.xlabel(f"Epochs")
    plt.ylabel(f"Loss {Args.loss}")
    plt.savefig(f"{Args.output.loss_png}/{Args.train_name}.png")
    plt.close()


    # Saving lr
    print(f"{c.lm}INFO : Save learnings rates (& plot) ... {c.d}")
    if "lrates" in run_stats.keys():
        np.save(f"{Args.output.divers}/lr_{Args.train_name}.npy", run_stats["lrates"])
        plt.figure(figsize=(16, 9))

        # Les losses
        plt.plot(np.arange(1, Args.epochs+1), run_stats["train"], c="k", ls=":", label="Train loss")
        plt.plot(np.arange(1, Args.epochs+1), run_stats["valid"], c="k", label="Valid loss")
        plt.axvline(best_state["epoch"], c="g", ls="-", label=f"Best state at valid loss = {best_state['best_val_loss']:.3e}")
        plt.ylabel(f"Loss {Args.loss}", color="black")
        plt.tick_params(axis="y", labelcolor="black")
        plt.legend()
        plt.yscale("log")

        # Lr
        plt.twinx()
        plt.plot(np.arange(1, Args.epochs+1), run_stats["lrates"], c="r")
        plt.ylabel("Learning rates", color="red")
        plt.tick_params(axis="y", labelcolor="red", color="r")
        plt.yscale("log")

        # final config
        plt.xlabel("Epochs")
        plt.title("Evolution of learning rates")
        plt.savefig(f"{Args.output.divers_png}/{Args.train_name}_lr.png")
        plt.close()


    print(f"{c.lm}INFO : Save best state ... {c.d}")
    if Args.save:
        torch.save(best_state, f"{Args.output.state}/{Args.train_name}_best.pth")














