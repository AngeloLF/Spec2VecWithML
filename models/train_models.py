import os, sys, shutil, importlib
import numpy as np
from tqdm import tqdm
import coloralf as c

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('./Spec2vecModels/')
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



def load_model(model_name, device, path2architecture='./Spec2vecModels/'):

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




if __name__ == "__main__":


    ### capture params
    Args = get_argv(sys.argv[1:], prog="training")

    ### Define some params
    name = f"{Args.model}_{Args.loss}" # Ex. : SCaM_chi2
    batch_size = params.batch_size_def if Args.model not in params.batch_size_models.keys() else params.batch_size_models[Args.model]
    model, Custom_dataloader, device = load_model_from_Args(Args)
    loss_function = give_Loss_Function(Args.loss, Args.model, f"{params.path}/{Args.train}", device)

    ### Define optimizer
    optim_name = params.optim_def if Args.model not in params.optim_models.keys() else params.optim_models[Args.model]
    if   optim_name == "Adam"  : optimizer = optim.Adam(model.parameters(), lr=Args.lr)
    elif optim_name == "AdamW" : optimizer = optim.AdamW(model.parameters(), lr=Args.lr)
    else : raise Exception(f"{c.r}The optimizer `optim_name` unknow. Please select Adam or AdamW.")




    ### Definition of paths
    path = params.path                                             # Ex. : ./results/output_simu
    train_name = f"{Args.from_prefixe}{Args.train}_{Args.lr_str}"       # Ex. : (pre_)(trainCalib_1e-4_)train16k_1e-04
    full_out_path = f"{params.out_path}/{params.out_dir}/{name}"   # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2

    train_inp_dir = f"{path}/{Args.train}/{model.folder_input}"    # Ex. : ./results/output_simu/train16k/image
    train_out_dir = f"{path}/{Args.train}/{model.folder_output}"   # Ex. : ./results/output_simu/train16k/spectrum
    valid_inp_dir = f"{path}/{Args.valid}/{model.folder_input}"    # Ex. : ./results/output_simu/valid2k/image
    valid_out_dir = f"{path}/{Args.valid}/{model.folder_output}"   # Ex. : ./results/output_simu/valid2k/spectrum

    # Define path of losses
    output_loss       = f"{full_out_path}/{params.out_loss}"       # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/loss
    output_loss_mse   = f"{full_out_path}/{params.out_loss_mse}"   # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/loss_mse
    output_loss_chi2  = f"{full_out_path}/{params.out_loss_chi2}"  # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/loss_chi2
    output_state      = f"{full_out_path}/{params.out_states}"     # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/states
    output_divers     = f"{full_out_path}/{params.out_divers}"     # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/divers

    # find tel
    if "ctio" in Args.train : tel = "ctio"
    elif "auxtel" in Args.train : tel = "auxtel"
    else : tel = None

    # Create folder in case ...
    os.makedirs(f"{params.out_path}/{params.out_dir}", exist_ok=True) # Ex. : ./results/Spec2vecModels_Results
    os.makedirs(full_out_path, exist_ok=True)                         # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2
    for f in [output_loss, output_loss_mse, output_loss_chi2, output_state, output_divers] : os.makedirs(f, exist_ok=True)

    # Folder for push epoch
    output_epoch      = f"{full_out_path}/{params.out_epoch}"      # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/epoch
    os.makedirs(output_epoch, exist_ok=True)
    output_epoch_here = f"{output_epoch}/{train_name}"             # Ex. : ./results/Spec2vecModels_Results/SCaM_chi2/epoch/train16k_1e-04
    if train_name in os.listdir(output_epoch) : shutil.rmtree(output_epoch_here)
    os.mkdir(output_epoch_here)





    ### Data set loading

    # Créer le Dataset complet
    train_dataset = Custom_dataloader(train_inp_dir, train_out_dir)
    valid_dataset = Custom_dataloader(valid_inp_dir, valid_out_dir)
        

    # Créer les DataLoaders pour l'entraînement et la validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)





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





    ### Some print
    print(f"{c.ly}INFO : Size of the loaded train dataset : {c.d}{c.y}{len(train_dataset)}{c.d}")
    print(f"{c.ly}INFO : Size of the loaded train dataset : {c.d}{c.y}{len(valid_dataset)}{c.d}")
    print(f"{c.ly}INFO : Telescope detected               : {c.d}{c.y}{tel}{c.d}")
    print(f"{c.ly}INFO : Utilisation de l'appareil        : {c.d}{c.y}{device}{c.d}")
    print(f"{c.ly}INFO : Optimizer                        : {c.d}{c.y}{optim_name}{c.d}")
    print(f"{c.ly}INFO : Model architecture               : {c.d}{c.y}{name}{c.d}")
    print(f"{c.ly}INFO : Name                             : {c.d}{c.y}{train_name}{c.d}")
    print(f"{c.ly}INFO : Train                            : {c.d}{c.y}{Args.train}{c.d}")
    print(f"{c.ly}INFO : Valid                            : {c.d}{c.y}{Args.valid}{c.d}")
    print(f"{c.ly}INFO : Epoch                            : {c.d}{c.y}{Args.epochs}{c.d}")
    print(f"{c.ly}INFO : Lrate                            : {c.d}{c.y}{Args.lr}{c.d}")
    print(f"{c.ly}INFO : batch size                       : {c.d}{c.y}{batch_size}{c.d}")
    print(f"{c.ly}INFO : Number of parameters             : {c.d}{c.y}{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f} millions{c.d}")





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

        train_loss = train_loss / len(train_dataset)
        train_list_loss[epoch] = train_loss
        train_list_loss_mse[epoch] = train_loss_mse / len(train_dataset)
        train_list_loss_chi2[epoch] = train_loss_chi2 / len(train_dataset)



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

        valid_loss = valid_loss / len(valid_dataset)

        # Show epoch
        lrates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"Epoch [{epoch+1}/{Args.epochs}], loss train = {c.g}{train_loss:.6f}{c.d}, val loss = {c.r}{valid_loss:.6f}{c.d} | LR={c.y}{lrates[epoch]:.2e}{c.d}")
        with open(f"{output_epoch_here}/INFO - epoch {epoch+1} - {Args.epochs} - {train_loss:.6f} , {valid_loss:.6f}", "wb") as f : pass

        # save state at each epoch to be able to reload and continue the optimization
        if valid_loss < best_val_loss:

            best_val_loss = valid_loss
            best_state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}

        # save loss
        valid_list_loss[epoch] = valid_loss
        valid_list_loss_mse[epoch] = valid_loss_mse / len(valid_dataset)
        valid_list_loss_chi2[epoch] = valid_loss_chi2 / len(valid_dataset)


    ### save everything
    print("Saving loss history and states of best models")

    np.save(f"{output_loss}/{train_name}.npy", np.array((train_list_loss, valid_list_loss)))
    np.save(f"{output_loss_mse}/{train_name}.npy", np.array((train_list_loss_mse, valid_list_loss_mse)))
    np.save(f"{output_loss_chi2}/{train_name}.npy", np.array((train_list_loss_chi2, valid_list_loss_chi2)))
    np.save(f"{output_divers}/lr_{train_name}.npy", lrates)

    if Args.save : torch.save(best_state, f"{output_state}/{train_name}_best.pth")

