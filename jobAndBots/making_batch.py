from types import SimpleNamespace
from itertools import product as szip
import os, sys, shutil, importlib
import params
import coloralf as c
import numpy as np

sys.path.append('./')

sys.path.append('./models/')
from get_argv import get_argv




def generate_batch(batch_name, codes, device, mult=False, mail=True, log=True, ext="slurm", mem=None, local=False, gpu_device="v100", nbj=None):
    """
    Function to create batch

    Param:
        * batch name [str] : name of the batch
        * codes [str or list of str] : codes to compute
        * device [str] : cpu or gpu
        TOMAKE
    """


    ### ENTETE
    if not local:
        slurm = ["#!/bin/bash"]
        slurm.append(f"#SBATCH --job-name={batch_name}               # Nom du job")
    else:
        slurm = ["#!/bin/bash"]
        # slurm = ["Write-Host \"PS1 : sbatch\""]


    ### Number of task & time max
    ntasks = len(codes) if mult else 1
    true_nbj = nbj if nbj is not None else params.nbj[device]


    ### DEVICE
    if device == "cpu" and not local:

        memcpu = "4G" if mem is None else f"{mem}G" 

        slurm.append(f"\n# Description Partition")
        slurm.append(f"#SBATCH --partition={params.partition_cpu}")
        slurm.append(f"#SBATCH --account={params.account}")

        slurm.append(f"\n# Description de la taches")
        slurm.append(f"#SBATCH --cpus-per-task=1        # Nombre de CPUs par tâche")
        slurm.append(f"#SBATCH --time={true_nbj}-00:00:00        # Limite de temps")
        slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
        slurm.append(f"#SBATCH --mem={memcpu}        # Mémoire demandée")   

    elif device == "gpu" and not local:

        memgpu = "16G" if mem is None else f"{mem}G"

        slurm.append(f"\n# Description Partition")
        slurm.append(f"#SBATCH --partition={params.partition_gpu}_{gpu_device}")
        slurm.append(f"#SBATCH --gpus=1")
        slurm.append(f"#SBATCH --account={params.account}")
        
        slurm.append(f"\n# Description de la taches")
        slurm.append(f"#SBATCH --cpus-per-task=5")
        slurm.append(f"#SBATCH --time={true_nbj}-00:00:00        # Limite de temps")
        slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
        slurm.append(f"#SBATCH --mem={memgpu}        # Mémoire demandée")

    elif not local:

        print(f"WARNING : le device {device} n'existe pas ! Cela doit être : cpu, gpu")


    ### MAIL
    if mail and not local:
        slurm.append(f"\n#SBATCH --mail-user={params.mail}")
        slurm.append(f"#SBATCH --mail-type=BEGIN,END,FAIL")


    ### OUTPUT
    if log and not local:
        slurm.append(f"\n#SBATCH --output=slurm_log/slurm_out/{batch_name}_%j.txt        # Fichier de sortie")
        slurm.append(f"#SBATCH --error=slurm_log/slurm_err/{batch_name}_%j.txt        # Fichier d'erreur")


    ### V-ENV
    if params.venv is not None and not local:
        slurm.append(f'\nVENV_PATH="{params.venv}"')
        slurm.append('source ${' + 'VENV_PATH' + '}/bin/activate')


    ### code
    slurm.append(f"\n# Codes to cumpute :")

    if isinstance(codes, (str)):
        codes = [codes]

    if not mult:
        executing_code = "\n".join([f"{params.python} {code}" for code in codes])
        slurm.append(executing_code)
        print(executing_code)
    else:

        for i, code in enumerate(codes):
            with open(f"{params.path}/sjob_{i}.sh", "w") as f:
                f.write(f"{params.python} {code}")

        slurm.append(" &\n".join([f"{params.srun} --exclusive sjob_{i}.sh" for i in range(len(codes))]))        
        slurm.append(f"wait")


    ### WRITING SLURM FILE
    true_ext = f".{ext}" if len(ext) > 0 else ""
    with open(f"{params.path}/{batch_name}{true_ext}", "w") as f:
        f.write("\n".join(slurm))






def read_SYSargv(batch_codes, arg2split):
    """
    Function to read the sys.argv for create batch
    """

    # create var
    ARGS = SimpleNamespace(a2s=dict())
    if len(sys.argv) > 1:
        batch = sys.argv[1]
        argv = sys.argv[2:]
    else:
        batch = "help"
        argv = list()


    # if batch not in batch_codes
    if batch not in batch_codes.keys(): 
        if batch in ["help", "h", "-h", "--h"]: 
            print(f"Possible batch :")
            for bck, bcv in batch_codes.items():
                color_str = c.g if not bck.startswith("search@") else c.m
                print(f"    * {color_str}{bck}{c.d} [{', '.join(bcv[1])}]")
            sys.exit()
        else:
            raise Exception(f"BATCH {batch} unknow [help for a list of possible batch]")


    # view argv
    for arg in argv:

        if "=" in arg:

            k, v = arg.split("=")

            if k in arg2split : ARGS.a2s[k] = v.split(",")
            else : ARGS.__setattr__(k, v) 

        else:

            ARGS.__setattr__(arg, None)


    # Show errors if needed args missing
    errors = list()
    for attr in batch_codes[batch][1]:
        if attr not in ARGS.a2s: 
            errors.append(f"{c.r}{attr}{c.d}")

    if len(errors) > 0: 
        print(f"Making batch stop : argv {', '.join(errors)} missing")
        sys.exit()


    # set default args
    ARGS.discobot = True if "discobot" in dir(ARGS) else False
    ARGS.mult = False if "nomult" in dir(ARGS) or "local" in dir(ARGS) else True
    ARGS.local = True if "local" in dir(ARGS) else False
    ARGS.make_all = True if "makeall" in sys.argv else False
    ARGS.device = batch_codes[batch][2] if "device" not in dir(ARGS) else "cpu"

    if "load" not in dir(ARGS) : ARGS.load = ["None"]
    if "mem" not in dir(ARGS) : ARGS.mem = None
    if "gd" not in dir(ARGS) : ARGS.gd = "v100"
    if "nbj" not in dir(ARGS) : ARGS.nbj = None
    if "validsize" not in dir(ARGS) : ARGS.validsize = "2k" if not ARGS.local else "128"
    if "testsize" not in dir(ARGS) : ARGS.testsize = "1k" if not ARGS.local else "8"
    if "load" not in ARGS.a2s.keys() : ARGS.a2s["load"] = ["None"]


    return batch, ARGS








###########
########### MAKE PARTITION FOR LONGUE CODE
###########


def give_partition(n, ncpu):

    """
    Function to optimize n job with ncpu number of cpu
    
    Exemple : n = 13 and , cpu = 5, return : [3, 3, 3, 2, 2]
    
    Param :
        * n [int] : number of job
        * ncpu [int] : number of cpu
    """

    q = n // ncpu
    r = n % ncpu

    partition = np.ones(ncpu).astype(int) * q
    partition[:r] += 1

    return partition


def search_Ntest_from_code(code):

    test = None

    for ele in code.split(" "):
        if ele.startswith("test="):
            test = ele.split("=")[1]

    if test is None:
        raise Exception(f"No test found in code {code}.")

    return len(os.listdir(f"./results/output_simu/{test}/image"))


def codes_with_ncpu(ncpu, batch_names, codes):

    new_batch_names = list()
    new_codes = list()

    for batch_name, code in zip(batch_names, codes):

        n = search_Ntest_from_code(code)
        partition = give_partition(n, int(ncpu))
        begin_with = np.concatenate((np.array([0]), np.cumsum(partition)[:-1])) # [3, 3, 2, 2] to [0, 3, 6, 8]

        for p, b in zip(partition, begin_with):
            new_codes.append(f"{code} range={b}_{p}")
            new_batch_names.append(f"{batch_name}_{b}_{p}")

    return new_batch_names, new_codes








##########
########## Verify if there are missings results to make the wanted batch
##########


def verify_simu(args):

    return list()


def verify_train_folder(args):

    missings = list()

    for train, tel, seed in szip(args.a2s["train"], args.a2s["tel"], args.a2s["seed"]):

        train_folder = f"train{train}{tel}-{seed}"
        valid_folder = f"valid{args.validsize}{tel}-{seed}"

        if train_folder not in os.listdir(f"./results/output_simu"):
            missings.append(f"Train : ./results/output_simu/{train_folder}")

        if valid_folder not in os.listdir(f"./results/output_simu"):
            missings.append(f"Valid : ./results/output_simu/{valid_folder}")

    return missings


def verify_test_folder(args):

    missings = list()

    for test, tel, seed in szip(args.a2s["test"], args.a2s["tel"], args.a2s["seed"]):

        test_folder = f"test{args.testsize}{test}{tel}-{seed}"

        if test_folder not in os.listdir(f"./results/output_simu"):
            missings.append(f"Test : ./results/output_simu/{test_folder}")

    return missings


def verify_model(args):

    missings = list()

    for model in args.a2s["model"]:

        if f"{model}.py" not in os.listdir(f"./models/architecture"):
            missings.append(f"Model : ./models/architecture/{model}.py")

    return missings


def verify_model_training(args):

    missings = list()

    for model, loss, train, tel, seed, lr in szip(args.a2s["model"], args.a2s["loss"], args.a2s["train"], args.a2s["tel"], args.a2s["seed"], args.a2s["lr"]):

        model_folder = f"./results/models_output/{model}_{loss}/states"
        model_training = f"train{train}{tel}-{seed}_{lr}_best.pth"

        if model_training not in os.listdir(model_folder):
            missings.append(f"Training : {model_folder}/{model_training}")

    return missings


def verify_apply(args):

    missings = list()

    for model, loss, train, tel, seed, lr, test in szip(args.a2s["model"], args.a2s["loss"], args.a2s["train"], args.a2s["tel"], args.a2s["seed"], args.a2s["lr"], args.a2s["test"]):

        test_folder = f"./results/output_simu/test{args.testsize}{test}{tel}-{seed}"
        pred_folder = f"pred_{model}_{loss}_train{train}{tel}-{seed}_{lr}"

        if not os.path.exists(test_folder):
            missings.append(f"Test not exist : {test_folder}")
        elif pred_folder not in os.listdir(test_folder):
            missings.append(f"Pred not exist : {test_folder}/{pred_folder}")
        elif len(os.listdir(f"{test_folder}/image")) > len(os.listdir(f"{test_folder}/{pred_folder}")):
            nbMiss = len(os.listdir(f"{test_folder}/image")) - len(os.listdir(f"{test_folder}/{pred_folder}"))
            missings.append(f"Pred miss {nbMiss} : {test_folder}/{pred_folder}")

    return missings


def verify_extract_atmo(args):

    missings = list()

    for tel, seed, test in szip(args.a2s["tel"], args.a2s["seed"], args.a2s["test"]):

        test_folder = f"./results/output_simu/test{args.testsize}{test}{tel}-{seed}"

        if not os.path.exists(test_folder):
            missings.append(f"Test not exist : {test_folder}")
        elif "atmos_params_fit" not in os.listdir(test_folder):
            missings.append(f"Extract atmos not exist : {test_folder}/atmos_params_fit")
        elif len(os.listdir(f"{test_folder}/atmos_params_fit")) == 0:
            missings.append(f"Extract atmos empty : {test_folder}/atmos_params_fit")

    return missings


def verify_batch(batch_name, args):

    missings = list()

    if batch_name == "simu":

        missings += verify_simu(args)

    elif batch_name == "training":

        missings += verify_train_folder(args)
        missings += verify_model(args)

    elif batch_name == "apply":

        missings += verify_model_training(args)
        missings += verify_test_folder(args)

    elif batch_name in ["apply_spectractor", "search@analyse"]:

        missings += verify_test_folder(args)

    elif batch_name == "analyse":

        missings += verify_apply(args)

    elif batch_name == "extract_atmo":

        missings += verify_apply(args)

    elif batch_name in ["analyse_atmo", "search@extract_atmo"]:

        missings += verify_extract_atmo(args)

    else:

        print(f"{c.r}WARNING : Batch_name {batch_name} not verify{c.d}")


    # raise missings
    if len(missings) > 0:

        print(f"{c.r}Some missings results to make batch ({len(missings)}) :{c.d}")

        for missing in missings:
            print(f"    * {c.ti}{missing}{c.d}")

        print(f"{c.r}##########{c.d}\n")
        print(f"Making batch stop : some missings results ({len(missings)})")
        sys.exit()

    else:

        pass








##########
########## SEARCHING BATCH
##########


def search_analyse(sysArgv, make_all=False):

    Args = get_argv(sysArgv[1:], prog="analyse_atmo")
    batchs = list()
    remake = list()

    for pf in os.listdir(f"./results/output_simu/{Args.test}"):

        if pf.startswith("pred_"):

            # verify if we need to make this pf
            if os.path.exists(f"./results/analyse/{Args.score}/{pf}/{Args.test}"):
                remake.append(True)
                if not make_all : run = False
                else : run = True
            else:
                remake.append(False)
                run = True

            # make if needed
            if run:
                _, model_, loss_, trainseed_, lr_, *_ = pf.split("_")
                batchs.append(f"analyse model={model_} loss={loss_} train={trainseed_} test={Args.test} lr={lr_} score={Args.score} load=None")

    return batchs, remake    


def search_extract_atmo(sysArgv, make_all=False):

    Args = get_argv(sysArgv[1:], prog="analyse_atmo")
    batchs = list()
    remake = list()

    save_folders = [pf for pf in os.listdir(f"./results/output_simu/{Args.test}/atmos_params_fit") if not "." in pf]
    pred_folders = [pf for pf in os.listdir(f"./results/output_simu/{Args.test}") if not "." in pf and pf.startswith("pred_")] + [f"pred_True_x_x_0e+00", f"pred_SpectractorFile_x_x_0e+00"]

    for pf in pred_folders:

        # verify if we need to make this pf
        if pf in save_folders:
            remake.append(True)
            if not make_all : run = False
            else : run = True
        else:
            remake.append(False)
            run = True

        # make if needed
        if run:
            _, model_, loss_, trainseed_, lr_ = pf.split("_")
            batchs.append(f"extract_atmo model={model_} loss={loss_} train={trainseed_} test={Args.test} lr={lr_} load=None")
        

    return batchs, remake





##########
########## MAIN
##########

if __name__ in "__main__":

    # Define possible batch, with needed argv : batch -> [<code python>, [<needed argv], <device>]
    batch_codes = {
        "simu"                 : ["specSimulator/main_simu.py",   ["nsimu", "tel", "test", "seed"],                                 "cpu"],
        "training"             : ["models/train_models.py",       ["model", "loss", "train", "lr", "tel", "e", "seed"],             "gpu"],
        "apply"                : ["applies/apply_model.py",       ["model", "loss", "train", "lr", "tel", "test", "seed"],          "cpu"],
        "apply_spectractor"    : ["applies/apply_spectractor.py", ["test", "tel", "seed"],                                          "cpu"],
        "analyse"              : ["analyses/analyse_test.py",     ["model", "loss", "train", "lr", "tel", "test", "seed", "score"], "cpu"],
        "extract_atmo"         : ["extractAtmos/extractAtmo.py",  ["model", "loss", "train", "lr", "tel", "test", "seed"],          "cpu"],
        "analyse_atmo"         : ["extractAtmos/analyseAtmo.py",  ["tel", "test", "seed"],                                          "cpu"],
        
        "search@analyse"       : [search_analyse,                 ["tel", "test", "seed", "score"],                                 "cpu"],
        "search@extract_atmo"  : [search_extract_atmo,            ["test", "tel", "seed"],                                          "cpu"],
    }

    # batch can accept multiple cpu usage
    batch_can_ncpu = ["apply_spectractor", "extract_atmo"]


    # Define test name possible
    test_fold = {
        "x" : "",
        "ext" : "EXT",
        "ot" : "OT",
        "na" : "NA",
        "gauss" : "GAUSSIAN", "gaussian" : "GAUSSIAN",
        "gaussna" : "GAUSSIANNA", "gaussianna" : "GAUSSIANNA",
        "stardice" : "STARDICE",
    }


    # argv to split into list if containt ","
    arg2split = ["model", "loss", "train", "test", "lr", "load", "nsimu", "score", "tel", "seed", "e"]


    # search batch et args from sys.argv
    batch, args = read_SYSargv(batch_codes, arg2split)


    # verify test fold
    if "test" in args.a2s.keys():

        test = list()
        test_error = list()

        for t in args.a2s["test"]:

            if t in test_fold.keys():
                test.append(test_fold[t])
            else:
                test_error.append(f"{c.r}{t}{c.d}")

        if len(test_error) > 0:
            raise Exception(f"{', '.join(test_error)} not exist (key available : {', '.join(list(test_fold.keys()))})")
        else:
            args.a2s["test"] = test


    # change lr for good format (1e-4 -> 1e-04)
    if "lr" in args.a2s.keys():

        args.a2s["lr"] = [f"{float(lr):.0e}" for lr in args.a2s["lr"]]


    # verify batch validity
    verify_batch(batch, args)




    ### create all batch --- args.a2s[""], 

    batch_code = batch_codes[batch][0]
    batch_names = list()
    codes = list()

    if batch == "simu":

        raise Exception(f"Simu batch TOMAKE")

    # create batch for training
    elif batch == "training":

        for model, loss, train, lr, tel, e, seed, load in szip(args.a2s["model"], args.a2s["loss"], args.a2s["train"], args.a2s["lr"], args.a2s["tel"], args.a2s["e"], args.a2s["seed"], args.a2s["load"]):

            train_folder = f"train{train}{tel}-{seed}"
            valid_folder = f"valid{args.validsize}{tel}-{seed}"
            codes.append(f"{batch_code} model={model} loss={loss} train={train_folder} valid={valid_folder} epoch={e} lr={lr} load={load}")
            batch_names.append(f"{batch}_{model}_{loss}_{train_folder}_{lr}_{e}_{load}")

    # create batch for apply
    elif batch == "apply":

        for model, loss, train, lr, tel, test, seed, load in szip(args.a2s["model"], args.a2s["loss"], args.a2s["train"], args.a2s["lr"], args.a2s["tel"], args.a2s["test"], args.a2s["seed"], args.a2s["load"]):

            train_folder = f"train{train}{tel}-{seed}"
            test_folder = f"test{args.testsize}{tel}-{seed}"
            codes.append(f"{batch_code} model={model} loss={loss} train={train_folder} test={test_folder} lr={lr} load={load} {device}")
            batch_names.append(f"{batch}_{model}_{loss}_{train_folder}_{test_folder}_{lr}_{load}")

    # create batch for apply_spectractor
    elif batch in "apply_spectractor":

        for tel, test, seed in szip(args.a2s["tel"], args.a2s["test"], args.a2s["seed"]):

            test_folder = f"test{args.testsize}{tel}-{seed}"
            codes.append(f"{batch_code} {test_folder}")
            batch_names.append(f"{batch}_{test_folder}")

    # create batch for analyse
    elif batch == "analyse":

        for model, loss, train, lr, tel, test, seed, score, load in szip(args.a2s["model"], args.a2s["loss"], args.a2s["train"], args.a2s["lr"], args.a2s["tel"], args.a2s["test"], args.a2s["seed"], args.a2s["score"], args.a2s["load"]):

            train_folder = f"train{train}{tel}-{seed}"
            test_folder = f"test{args.testsize}{tel}-{seed}"
            codes.append(f"{batch_code} model={model} train={train_folder} test={test_folder} loss={loss} lr={lr} score={score} load={load}")
            batch_names.append(f"{batch}_{model}_{loss}_{train_folder}_{test_folder}_{lr}_{score}_{load}")

    # create batch for extract atmo
    elif batch == "extract_atmo":

        for model, loss, train, lr, tel, test, seed, load in szip(args.a2s["model"], args.a2s["loss"], args.a2s["train"], args.a2s["lr"], args.a2s["tel"], args.a2s["test"], args.a2s["seed"], args.a2s["load"]):

            train_folder = f"train{train}{tel}-{seed}"
            test_folder = f"test{args.testsize}{tel}-{seed}"
            codes.append(f"{batch_code} model={model} train={train_folder} test={test_folder} loss={loss} lr={lr} load={load}")
            batch_names.append(f"{batch}_{model}_{loss}_{train_folder}_{test_folder}_{lr}_{load}")

    # create batch for analyse atmo
    elif batch in ["analyse_atmo", "search@extract_atmo"]:

        for tel, test, seed in szip(args.a2s["tel"], args.a2s["test"], args.a2s["seed"]):

            test_folder = f"test{args.testsize}{tel}-{seed}"
            codes.append(f"{batch_code} test={test_folder}")
            batch_names.append(f"{batch}_{test_folder}")

    # create batch for search@analyse
    elif batch in "search@analyse":

        for tel, test, seed, score in szip(args.a2s["tel"], args.a2s["test"], args.a2s["seed"], args.a2s["score"]):

            test_folder = f"test{args.testsize}{tel}-{seed}"
            codes.append(f"{batch_code} test={test_folder} score={score}")
            batch_names.append(f"{batch}_{test_folder}")

    else:

        raise Exception(f"{batch} batch TOMAKE")


    # if we want multiple cpu : ncpu=<number of cpu>
    if "ncpu" in dir(args):

        if batch in batch_can_ncpu:
            print(f"Make ncpu codes [ncpu={args.ncpu}] ...")
            batch_names, codes = codes_with_ncpu(args.ncpu, batch_names, codes)

        else:
            print(f"{c.r}WARNING : No multiple cpu (ncpu) for {batch}, only for {', '.join(batch_can_ncpu)}.{c.d}")


    # Number final of codes
    print(f"Number of codes : {c.lm}{len(codes)}{c.d}")




    ### use searching batch
    if "@" in batch:

        function = batch_codes[batch][0]
        search_batchs = list()
        remake_batchs = list()

        for batch_name, code in zip(batch_names, codes):

            print(f"{c.y}INFO : CODE sysArgv : `{code}`{c.d}")
            new_search, new_remake = function(code.split(" "), args.make_all)
            search_batchs += new_search
            remake_batchs += new_remake

        # create new batchs from the searching
        codes = list()
        batch_names = list()

        for i, (sb, rb) in enumerate(zip(search_batchs, remake_batchs)):
            batch = sb.split(" ")[0]
            argsi = " ".join(sb.split(" ")[1:])
            code = f"{batch_codes[batch][0]} {argsi}"
            batch_name = f"{batch}_{i+1}sur{len(search_batchs)}"

            remake_str = f" {c.b}{c.tb}{c.ti}{c.tu}REMAKE{c.d}" if rb and args.make_all else ""

            print(f"New batch [{c.tu}{batch_name}{c.d}] : {c.ti}{c.lm}{code}{c.d}{remake_str}")
            codes.append(code)
            batch_names.append(batch_name)

        print(f"Batch find : {c.lm}{len(codes)}{c.d} [already make : {np.sum(remake_batchs)}]")


        # if we want multiple cpu : ncpu=<number of cpu>
        if "ncpu" in dir(args):

            if batch in batch_can_ncpu:
                print(f"Make ncpu codes [ncpu={args.ncpu}] ...")
                batch_names, codes = codes_with_ncpu(args.ncpu, batch_names, codes)
                print(f"New number of codes : {c.lm}{len(codes)}{c.d}")

            else:
                raise Exception(f"No multiple cpu ({c.r}ncpu{c.d}) for {batch}, only for {', '.join(batch_can_ncpu)}.")


        # make the choice of create batch file or note
        choice = input(f"Tap `yes` or `y` to create batch file from search : ")

        if choice not in ["yes", "y"]:
            print(f"Close making batch...")
            sys.exit()




    ### Construction of SLURM file:
    print(f"Construction of batches ...")

    if not args.mult:

        extsup = "slurm" if not args.local else "local"
        generate_batch(batch, codes, device=args.device, mem=args.mem, ext=extsup, local=args.local, gpu_device=args.gd, nbj=args.nbj)

    else:

        extsup = "slurm" if not args.local else ""
        ext = "sh" if not args.local else ""

        with open(f"{batch}.{extsup}", "w") as f:
            for name, code in zip(batch_names, codes):
                generate_batch(name, code, device=args.device, ext=ext, mem=args.mem, local=args.local, gpu_device=args.gd, nbj=args.nbj)
                f.write(f"sbatch {params.path}/{name}.sh\n")
                                    











