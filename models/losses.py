import sys, json, pickle
import torch
import torch.nn as nn
import numpy as np

sys.path.append('./Spec2vecModels/')
import params

import coloralf as c



def give_Loss_Function(loss_name, model_name, train_path=None, device=None):

    if   loss_name == "chi2":
        n_bins = params.n_bins_def if model_name not in params.n_bins_models.keys() else params.n_bins_models[model_name]
        return Chi2Loss(params.Csigma_chi2, n_bins)

    elif loss_name == "MSE": 
        return nn.MSELoss()

    elif loss_name == "L1N":
        return L1NLoss()

    elif loss_name == "MSEf3":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss MSEf3 (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss MSEf3 (here is None)")
        with open(f"{train_path}/hist_params.json", 'r') as f:
            hp = json.load(f)
        ozone, pwv, aerosols = hp["ATM_OZONE"], hp["ATM_PWV"], hp["ATM_AEROSOLS"] 
        
        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {ozone}")
        print(f"ATM_PWV      : {pwv}")
        print(f"ATM_AEROSOLS : {aerosols}")

        return MSEf3Loss(ozone, pwv, aerosols, device)

    elif loss_name == "MSEmf3":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss MSEmf3 (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss MSEmf3 (here is None)")
        with open(f"{train_path}/hist_params.json", 'r') as f:
            hp = json.load(f)
        ozone, pwv, aerosols = hp["ATM_OZONE"], hp["ATM_PWV"], hp["ATM_AEROSOLS"] 
        
        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {ozone}")
        print(f"ATM_PWV      : {pwv}")
        print(f"ATM_AEROSOLS : {aerosols}")

        return MSEmf3Loss(ozone, pwv, aerosols, device)

    elif loss_name == "MSEn":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss MSEn (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss MSEn (here is None)")
        with open(f"{train_path}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {oms}")
        print(f"ATM_PWV      : {pms}")
        print(f"ATM_AEROSOLS : {ams}")

        return MSEnLoss(oms, pms, ams, device)

    elif loss_name == "MSEs":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss MSEs (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss MSEs (here is None)")
        with open(f"{train_path}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {oms}")
        print(f"ATM_PWV      : {pms}")
        print(f"ATM_AEROSOLS : {ams}")

        return MSEsLoss(oms, pms, ams, device)

    elif loss_name == "Ozone":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss Ozone (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss Ozone (here is None)")
        with open(f"{train_path}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {oms}")
        print(f"ATM_PWV      : {pms}")
        print(f"ATM_AEROSOLS : {ams}")

        return OzoneLoss(oms, pms, ams, device, model_name)

    elif loss_name == "OzoneS":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss Ozone (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss Ozone (here is None)")
        with open(f"{train_path}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {oms}")
        print(f"ATM_PWV      : {pms}")
        print(f"ATM_AEROSOLS : {ams}")

        return OzoneSLoss(oms, pms, ams, device, model_name)

    elif loss_name == "OzoneFree":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss Ozone (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss Ozone (here is None)")
        with open(f"{train_path}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {oms}")
        print(f"ATM_PWV      : {pms}")
        print(f"ATM_AEROSOLS : {ams}")

        return OzoneFreeLoss(oms, pms, ams, device, model_name)

    elif loss_name == "OzoneSFree":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss Ozone (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss Ozone (here is None)")
        with open(f"{train_path}/variable_params.pck", 'rb') as f:
            vp = pickle.load(f)
        ozone, pwv, aerosols = vp["ATM_OZONE"], vp["ATM_PWV"], vp["ATM_AEROSOLS"]
        oms = (np.mean(ozone), np.std(ozone) + 1e-8)
        pms = (np.mean(pwv), np.std(pwv) + 1e-8)
        ams = (np.mean(aerosols), np.std(aerosols) + 1e-8)

        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {oms}")
        print(f"ATM_PWV      : {pms}")
        print(f"ATM_AEROSOLS : {ams}")

        return OzoneSFreeLoss(oms, pms, ams, device, model_name)


    else: 
        print(f"{c.r}WARNING : loss name {loss_name} unknow{c.d}")
        raise ValueError("Loss name unknow")



class Chi2Loss(nn.Module):

    def __init__(self, C, norma):
        super(Chi2Loss, self).__init__()
        self.C2 = C ** 2
        self.norma = norma

    def forward(self, y_pred, y_true):
        nume = torch.pow(y_true - y_pred, 2)
        deno = y_true + self.C2
        loss = torch.sum(nume / deno) / self.norma
        return loss



class L1NLoss(nn.Module):

    def __init__(self):
        super(L1NLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sum(torch.abs(y_true-y_pred)) / torch.sum(y_true)



class MSEf3Loss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device):
        super(MSEf3Loss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols
        self.opaMIN = np.array([ozone[0], pwv[0], aerosols[0]])
        self.opaMAX = np.array([ozone[1], pwv[1], aerosols[1]])
        
        self.opaDEN = self.opaMAX - self.opaMIN
        self.opaDEN = torch.tensor(self.opaDEN, dtype=torch.float32).to(device)


    def forward(self, y_pred, y_true):

        diff = (y_pred - y_true)
        l1 = diff / self.opaDEN
        loss = torch.sum(torch.pow(l1, 2))

        # print(f"forward MSEf3Loss : {y_pred} \n {y_true} \n {diff} \n {l1} : {l1s} -2> {loss}")

        return loss



class MSEmf3Loss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device):
        super(MSEmf3Loss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols
        self.opaMIN = np.array([ozone[0], pwv[0], aerosols[0]])
        self.opaMAX = np.array([ozone[1], pwv[1], aerosols[1]])

        self.ranges = torch.tensor(self.opaMAX-self.opaMIN).to(device)
        self.divide = 1.0 / torch.pow(self.ranges, 2)
        

    def forward(self, y_pred, y_true):

        diff = torch.pow(y_pred - y_true, 2)
        loss = (self.divide * diff).mean()

        # print(f"forward MSEf3Loss : {y_pred} \n {y_true} \n {diff} \n {l1} : {l1s} -2> {loss}")

        return loss



class MSEnLoss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device):
        super(MSEnLoss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols
        self.mu = torch.tensor(np.array([ozone[0], pwv[0], aerosols[0]]).astype(np.float32)).to(device)
        self.sigma = torch.tensor(np.array([ozone[1], pwv[1], aerosols[1]]).astype(np.float32)).to(device)


    def forward(self, y_pred, y_true):

        y_true_norma = (y_true - self.mu) / self.sigma
        loss = torch.nn.functional.mse_loss(y_pred, y_true_norma)

        return loss




class MSEsLoss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device):
        super(MSEsLoss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols
        self.mu = torch.tensor(np.array([ozone[0], pwv[0], aerosols[0]]).astype(np.float32)).to(device)
        self.sigma = torch.tensor(np.array([ozone[1], pwv[1], aerosols[1]]).astype(np.float32)).to(device)


    def forward(self, y_pred, y_true):

        y_pred_norma = (y_pred - self.mu) / self.sigma
        y_true_norma = (y_true - self.mu) / self.sigma
        loss = torch.nn.functional.mse_loss(y_pred_norma, y_true_norma)

        return loss











class OzoneSLoss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device, model_name):
        super(OzoneSLoss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols

        if "FOBIQ" in model_name:

            self.mu, self.sigma = ozone
            self.w = 1.0

        else:

            self.mu = torch.tensor(np.array([ozone[0], pwv[0], aerosols[0]]).astype(np.float32)).to(device)
            self.sigma = torch.tensor(np.array([ozone[1], pwv[1], aerosols[1]]).astype(np.float32)).to(device)
            self.w = torch.tensor(np.array([1., 0., 0.]).astype(np.float32)).to(device)


    def forward(self, y_pred, y_true):

        y_pred_norma = (y_pred - self.mu) / self.sigma
        y_true_norma = (y_true - self.mu) / self.sigma * self.w
        loss = torch.nn.functional.mse_loss(y_pred_norma, y_true_norma)

        return loss



class OzoneSFreeLoss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device, model_name):
        super(OzoneSFreeLoss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols
        self.mu = torch.tensor(np.array([ozone[0], pwv[0], aerosols[0]]).astype(np.float32)).to(device)
        self.sigma = torch.tensor(np.array([ozone[1], pwv[1], aerosols[1]]).astype(np.float32)).to(device)
        self.w = torch.tensor(np.array([1., 0., 0.]).astype(np.float32)).to(device)


    def forward(self, y_pred, y_true):

        y_pred_norma = (y_pred - self.mu) / self.sigma * self.w
        y_true_norma = (y_true - self.mu) / self.sigma * self.w
        loss = torch.nn.functional.mse_loss(y_pred_norma, y_true_norma)

        return loss



class OzoneLoss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device, model_name):
        super(OzoneLoss, self).__init__()
        self.w = torch.tensor(np.array([1., 0., 0.]).astype(np.float32)).to(device)


    def forward(self, y_pred, y_true):

        diff = (y_pred - y_true * self.w)
        loss = torch.sum(torch.pow(diff, 2))

        return loss



class OzoneFreeLoss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device, model_name):
        super(OzoneFreeLoss, self).__init__()
        self.w = torch.tensor(np.array([1., 0., 0.]).astype(np.float32)).to(device)


    def forward(self, y_pred, y_true):

        diff = (y_pred - y_true) * self.w
        loss = torch.sum(torch.pow(diff, 2))

        return loss