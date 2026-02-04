import torch
import torch.nn.functional as nnfunc

import json, pickle, sys, os, shutil, importlib
from tqdm import tqdm
import coloralf as c
import numpy as np
from time import time

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv, get_device
from train_models import load_from_pretrained
sys.path.append('./')



if __name__ == "__main__":

    
    ### capture params
    Args = get_argv(sys.argv[1:], prog="apply")
    device = get_device(Args)
    full_train_str = f"{Args.from_prefixe}{Args.train}"
    model, Custom_dataloader = load_from_pretrained(Args.model, Args.loss, full_train_str, Args.lr_str, device=device)
    pred_fold_name = f"pred_{Args.model}_{Args.loss}_{full_train_str}_{Args.lr_str}"

    path_results = './results'
    path_test = path_results
    path_train = f"{path_results}/output_simu"

    if "output" in Args.test : folders = [f"{path_test}/{Args.test}/{sf}" for sf in os.listdir(f"{path_test}/{Args.test}") if "test" in sf]
    else :                     folders = [f"{path_test}/output_simu/{Args.test}"]

    if Args.loss in ["MSEn"]:

        with open(f"{path_train}/{Args.train}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        mu    = np.array([oms[0], pms[0], ams[0]])
        sigma = np.array([oms[1], pms[1], ams[1]])

    # find tel
    if "ctio" in Args.train : tel = "ctio"
    elif "auxtel" in Args.train : tel = "auxtel"
    else : tel = None


    
    ### Apply 

    nb_folds = len(folders)

    for i, fold in enumerate(folders):

        print(f"{c.lg}Make {fold} [{i+1}/{nb_folds}] ...{c.d}")

        if pred_fold_name in os.listdir(fold) : shutil.rmtree(f"{fold}/{pred_fold_name}")
        os.mkdir(f"{fold}/{pred_fold_name}")
        print(f"{c.ly}INFO : creation of {fold}/{pred_fold_name}{c.d}")

        test_dataset = Custom_dataloader(f"{fold}/{model.folder_input}", f"{fold}/{model.folder_output}")

        with torch.no_grad():

            pbar = tqdm(total=len(test_dataset))
            t0s = list()

            for (img, true_tensor), spec_name in zip(test_dataset, test_dataset.spectrum_files):

                true_spec_name = spec_name.replace("\\", "/").split("/")[-1]
                test_image = img.unsqueeze(0).to(device)

                t0 = time()
                pred = model(test_image).cpu().numpy()[0]

                if Args.loss in ["MSEn"] : pred = pred * sigma + mu

                t0s.append(time()-t0)

                np.save(f"{fold}/{pred_fold_name}/{true_spec_name}", pred)

                if "extraApply" in dir(model):
                    getattr(model, "extraApply")(pred, f"{fold}/{pred_fold_name}", true_spec_name)
                
                pbar.update(1)


            print(f"Full time : {np.sum(t0s)} --- {np.mean(t0s)}")

            pbar.close()



