import sys
import coloralf as c
from hparams import Hparams
from specsimulator import SpecSimulator


### define var params

var_params = {
    "train" : {
        "ATM_AEROSOLS" : [0.1, 0.8],
        "ATM_OZONE" : [250.0, 350.0],
        "ATM_PWV" : [2.0, 10.0],
        "ATM_AIRMASS" : [1.2, 2.0],
        "A" : [0.8, 1.2],
    },
    "test" : {
        "ATM_AEROSOLS" : [0.0, 0.9],
        "ATM_OZONE" : [220.0, 380.0],
        "ATM_PWV" : [0.0, 12.0],
        "ATM_AIRMASS" : [1.0, 2.2],
        "A" : [0.6, 1.4],
    }
}




### extraction argv of sys

argv = {"__free__":list()}
for arg in sys.argv[1:]:
    if "=" in arg:
        k, v = arg.split("=")
        argv[k] = v
    else:
        argv["__free__"].append(arg)




### var params for telescope

if "tel" in argv.keys():

    if argv["tel"] == "ctio":
        var_params["train"]["ROTATION_ANGLE"] = [-2.0, 2.0]
        var_params["test"]["ROTATION_ANGLE"] = [-3.0, 3.0]

    elif argv["tel"] == "auxtel":
        var_params["train"]["ROTATION_ANGLE"] = [-0.1, 0.1]
        var_params["test"]["ROTATION_ANGLE"] = [-0.1, 0.1]

else: # it's ctio by default
    var_params["train"]["ROTATION_ANGLE"] = [-2.0, 2.0]
    var_params["test"]["ROTATION_ANGLE"] = [-3.0, 3.0]




### var params for psf

if "psf" in argv.keys():

    if argv["psf"] == "moffat2d":
        var_params["train"]["arg.0.0"] = [3.0, 8.0]
        var_params["test"]["arg.0.0"] = [2.0, 10.0]

    elif argv["psf"] == "gaussian2d":
        var_params["train"]["arg.0.0"] = [3.0, 8.0]
        var_params["test"]["arg.0.0"] = [2.0, 10.0]

else: # it's moffat by default
    var_params["train"]["arg.0.0"] = [3.0, 8.0]
    var_params["test"]["arg.0.0"] = [2.0, 10.0]

# Variation of the argument "alpha" for moffat2d
if "moffat2d_alpha_variation" in argv["__free__"]:
    var_params["train"]["arg.1.0"] = [1.5, 2.5]
    var_params["test"]["arg.1.0"] = [1.5, 2.5]

# 




### noisy or not

noisy = False if "noisyless" in sys.argv else True 




### go simu

if "test" not in sys.argv:
    hp = Hparams(var_params=var_params["train"], with_noise=noisy)
else:
    hp = Hparams(var_params=var_params["test"], with_noise=noisy)

sim = SpecSimulator(hp)
sim.run()



