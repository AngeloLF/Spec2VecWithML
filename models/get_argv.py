from types import SimpleNamespace
import coloralf as c
import torch
import numpy as np



def get_argv(argv, prog=None, correction=False, show=False):


    Args = SimpleNamespace()
    Args.save = True
    Args.show = False
    Args.aug_smooth = False
    Args.from_pre, Args.from_prefixe = False, ""
    Args.pre_train, Args.pre_lr, Args.pre_lr_str = None, None, None

    Args.device = "cpu"
    if prog in ["training"] : Args.device = "gpu"



    for arg in argv:

        if arg[:6] == "model=" : Args.model = arg[6:]
        if arg[:5] == "loss="  : Args.loss  = arg[5:]
        if arg[:6] == "train=" : Args.train = arg[6:]
        if arg[:6] == "valid=" : Args.valid = arg[6:]
        if arg[:5] == "test="  : Args.test  = arg[5:]

        if arg[:2] == "e="     : Args.epochs = int(arg[2:])
        if arg[:6] == "epoch=" : Args.epochs = int(arg[6:])
        if arg[:3] == "lr="    : Args.lr     = float(arg[3:])
        if arg[:3] == "lr="    : Args.lr_str = f"{float(arg[3:]):.0e}"
        if arg[:6] == "score=" : Args.score  = arg[6:]

        if arg[:6] == "image=" : Args.image = arg[6:]
        if arg[:2] == "n="     : Args.n     = int(arg[2:])
        if arg[:5] == "mode="  : Args.mode  = arg[5:]

        if arg[:5] == "spec="  : Args.spec  = arg[5:]
        
        if arg == "cpu"        : Args.device = "cpu"
        if arg == "gpu"        : Args.device = "gpu"

        if arg == "show"       : Args.show = True
        if arg == "nosave"     : Args.save = False
        if arg == "aug_smooth" or arg == "augs" : Args.aug_smooth = True

        if arg[:5] == "load="  : # <folder_pretrain>_<prelr>
            load_pret = arg[5:] 
            if load_pret != "None":
                pre_train, pre_lr = load_pret.split("_")
                Args.from_pre = True
                Args.pre_train = pre_train if pre_train[:5] == "train" else f"train{pre_train}"
                Args.pre_lr = float(pre_lr)
                Args.pre_lr_str = f"{float(pre_lr):.0e}"
                Args.from_prefixe = f"{Args.pre_train}_{Args.pre_lr_str}_"


    ### Exception

    # extract_atmo -> model "spectrum" & "spectractorfile" (not pred Spectractor) dont have loss ...
    if prog == "extract_atmo" and "model" in dir(Args) and Args.model in ["true", "spectractorfile"]:

        Args.loss = None
        Args.train = None
        Args.lr = None
        Args.lr_str = ""



    
    if prog is not None:

        Errors = list()


        if "model" not in dir(Args) and prog in ["training", "apply", "analyse", "gradcam", "residus", "extract_atmo", "analyse_training"]:

            print(f"{c.r}WARNING : model architecture is not define (model=<model_architecture>){c.d}")
            Errors.append("Model architecture")



        if "loss" not in dir(Args) and prog in ["training", "apply", "analyse", "gradcam", "residus", "extract_atmo", "analyse_training"]:

            print(f"{c.r}WARNING : loss function is not define (loss=<loss_function>){c.d}")
            Errors.append("Loss function")



        if "train" not in dir(Args) and prog in ["training", "apply", "analyse", "gradcam", "residus", "extract_atmo", "analyse_training"]:

            print(f"{c.r}WARNING : train folder is not define (train=<train_folder>){c.d}")
            Errors.append("Train folder")

        elif "train" in dir(Args) and correction and Args.train[:5] != "train" : Args.train = f"train{Args.train}"



        if "valid" not in dir(Args) and prog in ["training"]:

            print(f"{c.r}WARNING : valid folder is not define (valid=<valid_folder>){c.d}")
            Errors.append("Valid folder")# raise Exception("Valid folder is not define")

        elif "valid" in dir(Args) and correction and Args.valid[:5] != "valid" : Args.valid = f"valid{Args.valid}"



        if "test" not in dir(Args) and prog in ["apply", "analyse", "gradcam", "residus", "extract_atmo", "analyse_atmo"]:

            print(f"{c.r}WARNING : test folder is not define (test=<test_folder>){c.d}")
            Errors.append("Test folder") # raise Exception("test folder is not define")

        elif "test" in dir(Args) and correction and Args.test[:4] != "test" : Args.test = f"test{Args.test}"



        if "epochs" not in dir(Args) and prog in ["training"]:

            print(f"{c.r}WARNING : Epoch number is not define (e=<epochs_number> or epoch=<epochs_number){c.d}")
            Errors.append("Epoch number") # raise Exception("Epoch number is not define")



        if "lr" not in dir(Args) and prog in ["training", "apply", "analyse", "gradcam", "residus", "extract_atmo", "analyse_training"]:

            print(f"{c.r}WARNING : learning rate is not define (lr=<learning_rate>){c.d}")
            Errors.append("Learning rate") # raise Exception("Learning rate is not define")



        if "score" not in dir(Args) and prog in ["analyse"]:

            print(f"{c.r}WARNING : score type is not define (score=<score_type>){c.d}")
            Errors.append("Score type")



        if "image" not in dir(Args) and prog in ["gradcam"]:

            print(f"{c.r}WARNING : image is not define (image=<image>){c.d}")
            Errors.append("Image")



        if "spec" not in dir(Args) and prog in ["residus"]:

            print(f"{c.r}WARNING : spectrum is not define (spec=<spectrum>){c.d}")
            Errors.append("Spectrum")



        if "n" not in dir(Args) and prog in ["gradcam"]:

            print(f"{c.r}WARNING : n is not define (n=<n>){c.d}")
            Errors.append("n")



        if "mode" not in dir(Args) and prog in ["gradcam"]:

            print(f"{c.r}WARNING : mode is not define (mode=<mode>){c.d}")
            Errors.append("Mode")



        if len(Errors) > 0:
            all_errors = ", ".join(Errors)
            raise Exception(f"{all_errors} not defines")



    if prog in ["training", "apply", "analyse", "gradcam", "residus", "extract_atmo", "analyse_training"]:

        Args.model_loss = f"{Args.model}_{Args.loss}"
        Args.fulltrain_str = f"{Args.from_prefixe}{Args.train}"
        Args.fullname = f"{Args.model}_{Args.loss}_{Args.fulltrain_str}_{Args.lr_str}"

        Args.wl = np.arange(300, 1100) if Args.model not in ["ViTESoS"] else np.arange(0, 1024)
        Args.folder_output = "spectrum" if Args.model not in ["ViTESoS"] else "spectrumPX"


    if show : show_Args(Args)

    return Args




def show_Args(args):

    atts = [a for a in dir(args) if a[:2] != "__"]

    print(f"\n{c.y}Args attributs :{c.d}")

    for att in atts:

        print(f"{c.ly}{att}{c.d} : {args.__getattribute__(att)}")

    print(f"")



def get_device(args):

    # For GPU, with NVIDIA
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{c.ly}INFO : torch device turn to GPU/CUDA (NVIDIA)")

    # For GPU, with Apple M
    elif args.device == "gpu" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"{c.ly}INFO : torch device turn to GPU/MPS (Apple)")

    # If no GPU avaible or not wanted 
    else:
        if args.device == "cuda" : print(f"{c.r}WARNING : GPU is not available for torch ... device turn to CPU ... ")
        device = torch.device("cpu")
    
    print(f"{c.ly}INFO : Utilisation du device {c.tu}{device}{c.d}{c.d}")
    return device


if __name__ == "__main__":

    import sys

    args = get_argv(sys.argv[2:], sys.argv[1], show=True)