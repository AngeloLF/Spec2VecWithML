from types import SimpleNamespace
import os, sys, shutil
import params
import coloralf as c
import numpy as np


def generate_batch(batch_name, codes, device, mult=False, mail=True, log=True, ext="slurm", mem=None, local=False, gpu_device="v100", nbj=None):



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


    ARGS = SimpleNamespace()
    batch = sys.argv[1]
    argv = sys.argv[2:]


    if batch not in batch_codes.keys(): 
        if batch == "help": 
            print(arg2split)
            sys.exit()
        else: 
            raise Exception(f"BATCH {batch} unknow")


    for arg in argv:

        if "=" in arg:

            k, v = arg.split("=")

            if k in arg2split : ARGS.__setattr__(k, v.split(","))
            else : ARGS.__setattr__(k, v) 

        else:

            ARGS.__setattr__(arg, None)


    errors = list()
    for attr in batch_codes[batch][1]:
        if attr not in dir(ARGS) : errors.append(attr)

    if len(errors) > 0 : raise Exception(f"{', '.join(errors)} missing")


    return batch, ARGS



def addJob2args(ARGS, model, loss, train, lr, load, test, score=None):

    ARGS.list_model.append(model)
    ARGS.list_loss.append(loss)
    ARGS.list_train.append(train)
    ARGS.list_lr.append(lr)
    ARGS.list_load.append(load)
    ARGS.list_test.append(test)
    ARGS.nb += 1
    if score is not None : ARGS.list_score.append(score)
    
    return ARGS



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





def findJob(args, states_path="./results/Spec2vecModels_Results"):

    if "score" not in dir(args):

        if "FOPA" in args.modelwl[0] or "FOBIQ" in args.modelwl[0]:
            print(f"{c.ly}INFO : in findJob, score is not indicated (but FOPA/FOBIQ detected) -> def sfopa")
            args.score = ["sfopa"]
        else:
            print(f"{c.ly}INFO : in findJob, score is not indicated -> def L1,chi2")
            args.score = ["L1", "chi2"]

    if "test" not in dir(args):

        if "FOPA" in args.modelwl[0] or "FOBIQ" in args.modelwl[0]:
            print(f"{c.ly}INFO : in findJob, test is not indicated (but FOPA/FOBIQ detected) -> def testAtmo1k")
            args.test = ["testAtmo1k"]
        else:
            print(f"{c.ly}INFO : in findJob, test is not indicated -> def test1k,test1kOT,test1kExt")
            args.test = ["test1k", "test1kOT", "test1kExt"]

    ARGS_APPLY = SimpleNamespace()
    ARGS_ANALYSE = SimpleNamespace()


    for ARGS in [ARGS_APPLY, ARGS_ANALYSE]:

        ARGS.list_model = list()
        ARGS.list_loss = list()
        ARGS.list_train = list()
        ARGS.list_lr = list()
        ARGS.list_load = list()
        ARGS.list_test = list()
        ARGS.list_score = list()
        ARGS.nb = 0


    allModelWL = list(set(args.modelwl))


    for modelwl in allModelWL:
        
        model, loss = modelwl.split("_")
        print(f"\n{c.y}Analyseof model {model} [{loss}]{c.d}")

        for state in os.listdir(f"{states_path}/{modelwl}/states"):

            if "_best.pth" in state:

                subs = state[:-9]
                    
                if subs.count("_") == 1: 
                    train, lr = subs.split("_")
                    load = "None"
                    pred_folder = f"pred_{model}_{loss}_{train}_{lr}"
                elif subs.count("_") == 3:
                    loadtrain, loadlr, train, lr = subs.split("_")
                    load = f"{loadtrain}_{loadlr}"
                    pred_folder = f"pred_{model}_{loss}_{load}_{train}_{lr}"
                else:
                    raise Exception(f"WARNING [making_sbatch.py] : State {state} counts `_` not in [1, 3]")

                print(f"{c.ly}Analyseof : detect {subs} -> {train} > {lr} > {load}{c.d}")

                for otest in args.test:

                    test = otest if "no0" not in train else f"{otest}no0"
                    if test == "testAtmo1k" and "2Atmo" in train : test = "test2Atmo1k"

                    if pred_folder in os.listdir(f"./results/output_simu/{test}"):
                        print(f"{c.lg}- Test {test} apply{c.d}")

                        for score in args.score:

                            if pred_folder in os.listdir(f"./results/analyse/{score}") and test in os.listdir(f"./results/analyse/{score}/{pred_folder}") and ("resume.txt" in os.listdir(f"./results/analyse/{score}/{pred_folder}/{test}") or "resume.json" in os.listdir(f"./results/analyse/{score}/{pred_folder}/{test}"))  :
                                print(f"{c.lg}  |-- score {score} analyse{c.d}")
                            else:
                                print(f"{c.lr}  |-- score {score} not analyse{c.d}")
                                ARGS_ANALYSE = addJob2args(ARGS_ANALYSE, model, loss, train, lr, load, test, score)


                    else:
                        print(f"{c.r}- Test {test} not apply{c.d}")
                        ARGS_APPLY = addJob2args(ARGS_APPLY, model, loss, train, lr, load, test)



    choice = None

    while choice not in ["", "analyse", "apply", "analyseFOPA"]:

        print(f"Number of job `apply`   : {ARGS_APPLY.nb}")
        print(f"Number of job `analyse` : {ARGS_ANALYSE.nb}")
        choice = input(f"Make anything ? (an/analyse or ap/apply) : ")

        if choice in ["an", "analyse"] : choice = "analyse"
        elif choice in ["ap", "apply"] : choice = "apply" 

    if "FOPA" in args.modelwl[0] or "FOBIQ" in args.modelwl[0] and choice == "analyse" : choice = "analyseFOPA"

    args.ARGS_APPLY = ARGS_APPLY
    args.ARGS_ANALYSE = ARGS_ANALYSE
    args.findjob_choice = choice

    return args








if __name__ in "__main__":


    batch_codes = {
        "flash"                : ["None",                         ["jobname", "code"]],
        "simu"                 : ["specSimulator/main_simu.py",   ["nsimu", "tel", "type", "seed"]],
        "training"             : ["models/train_models.py",       ["model", "loss", "train", "lr", "tel", "e"]],
        "apply"                : ["applies/apply_model.py",       ["model", "loss", "train", "lr", "tel", "test"]],
        "apply_spectractor"    : ["applies/apply_spectractor.py", ["test", "tel"]],
        "analyse"              : ["analyses/analyse_test.py",     ["model", "loss", "train", "lr", "tel", "test", "score"]],
        "extract_atmo"         : ["extractAtmos/extractAtmo.py",  ["model", "loss", "train", "lr", "tel", "test"]],
        "analyseFOPA"          : ["analyses/analyse_FOPA.py",     ["model", "loss", "train", "lr", "tel", "test", "score"]],
        "findjob"              : ["None",                         ["modelwl"]] # Model with loss like `SCaM_chi2`
    }

    arg2split = ["type", "model", "modelwl", "loss", "train", "test", "lr", "load", "nsimu", "score", "tel", "seed"]

    batch, args = read_SYSargv(batch_codes, arg2split)
    if batch == "findjob" : args = findJob(args)

    args.discobot = True if "discobot" in dir(args) else False
    args.mult = False if "nomult" in dir(args) else True
    args.local = True if "local" in dir(args) else False
    if "load" not in dir(args) : args.load = ["None"]
    if "mem" not in dir(args) : args.mem = None
    if "gd" not in dir(args) : args.gd = "v100"
    if "nbj" not in dir(args) : args.nbj = None

    batch_names = list()
    codes = list()
    make_jobs = True



    if batch == "flash":

        device = "cpu"

        codes.append(args.code)
        batch_names.append(args.jobname)



    elif batch == "simu":

        device = "cpu"

        for n_i, type_i, tel_i, seed_i in zip(args.nsimu, args.type, args.tel, args.seed):

            simupi = ""
            set_i = "set0"
            psf_i = "moffat2d"

            if int(n_i) < 1000:
                str_n = n_i
            else:
                str_n = f"{int(int(n_i)/1000)}k"


            if type_i == "train":
                filename = f"train{str_n}{tel_i}"

            elif type_i == "valid":
                filename = f"valid{str_n}{tel_i}"

            elif type_i == "test":
                filename = f"test{str_n}{tel_i}"

            elif type_i.lower() == "testext":
                filename = f"test{str_n}EXT{tel_i}"
                simupi = "test"

            elif type_i.lower() == "testot":
                filename = f"test{str_n}OT{tel_i}"
                set_i = "set1"

            elif type_i.lower() == "testgauss":
                filename = f"test{str_n}GAUSSIAN{tel_i}"
                psf_i = "gaussian2d"

            codes.append(f"{batch_codes['simu'][0]} nsimu={n_i} f={filename} set={set_i} tel={tel_i} seed={seed_i} psf={psf_i} {simupi}")
            batch_names.append(f"{batch}_{filename}")



    elif batch == "findjob" and args.findjob_choice == "apply":

        device = "cpu"

        for model, loss, train, lr, load, test in zip(args.ARGS_APPLY.list_model, args.ARGS_APPLY.list_loss, args.ARGS_APPLY.list_train, args.ARGS_APPLY.list_lr, args.ARGS_APPLY.list_load, args.ARGS_APPLY.list_test):

            codes.append(f"{batch_codes['apply'][0]} model={model} loss={loss} train={train} test={test} lr={lr} load={load}")
            batch_names.append(f"apply_{model}_{loss}_{train}_{test}_{lr}_{load}")



    elif batch == "findjob" and args.findjob_choice == "analyse":

        device = "cpu"

        for model, loss, train, lr, load, test, score in zip(args.ARGS_ANALYSE.list_model, args.ARGS_ANALYSE.list_loss, args.ARGS_ANALYSE.list_train, args.ARGS_ANALYSE.list_lr, args.ARGS_ANALYSE.list_load, args.ARGS_ANALYSE.list_test, args.ARGS_ANALYSE.list_score):
        
            codes.append(f"{batch_codes['analyse'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
            batch_names.append(f"analyse_{model}_{loss}_{train}_{test}_{lr}_{score}_{load}")



    elif batch == "findjob" and args.findjob_choice == "analyseFOPA":

        device = "cpu"

        for model, loss, train, lr, load, test, score in zip(args.ARGS_ANALYSE.list_model, args.ARGS_ANALYSE.list_loss, args.ARGS_ANALYSE.list_train, args.ARGS_ANALYSE.list_lr, args.ARGS_ANALYSE.list_load, args.ARGS_ANALYSE.list_test, args.ARGS_ANALYSE.list_score):
        
            codes.append(f"{batch_codes['analyseFOPA'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
            batch_names.append(f"analyse_{model}_{loss}_{train}_{test}_{lr}_{score}_{load}")



    elif batch == "findjob":

        make_jobs = False



    elif batch == "apply_spectractor":

        device = "cpu" if "gpu" not in sys.argv else "gpu"

        for tel in args.tel:

            for str_test in args.test:
                str_nbtest = "1k" if not args.local else "" 
                test = f"test{str_nbtest}{tel}" if str_test == "x" else f"test{str_nbtest}{str_test.upper()}{tel}" 
                # Check test
                if test not in os.listdir(f"./results/output_simu") and "passall" not in sys.argv:
                    raise Exception(f"Test folder {test} not in ./results/output_simu")
                elif "passall" not in sys.argv:
                    ntest = len(os.listdir(f"./results/output_simu/{test}/image"))
                    print(f"Info : {ntest} images in {test}")
                else:
                    ntest = None

                # if we want multiple cpu
                if "ncpu" in dir(args):
                    print(f"{c.y}INFO [new_making_batch.py] : reset Spectractor folders for {test}{c.d}")
                    for fold in ["image_fits", "spectrum_fits", "pred_Spectractor_x_x_0e+00", "spectractor_exceptions"]:
                        if fold in os.listdir(f"./results/output_simu/{test}") : shutil.rmtree(f"./results/output_simu/{test}/{fold}")
                        os.mkdir(f"./results/output_simu/{test}/{fold}")
                    partition = give_partition(ntest, int(args.ncpu))
                    begin_with = np.concatenate((np.array([0]), np.cumsum(partition)[:-1])) # [3, 3, 2, 2] to [0, 3, 6, 8]
                    for p, b in zip(partition, begin_with):
                        codes.append(f"{batch_codes['apply_spectractor'][0]} {test} range={b}_{p}")
                        batch_names.append(f"{batch}_{test}_{b}_{p}")
                else:
                    codes.append(f"{batch_codes['apply_spectractor'][0]} {test}")
                    batch_names.append(f"{batch}_{test}")


    else:

        model_pass = ["Spectractor", "true", "spectractorfile"]

        for model in args.model:
            # Check model
            if f"{model}.py" not in os.listdir(f"./Spec2vecModels/architecture") and model not in model_pass:
                raise Exception(f"The architecture model {model} unknow")


            for loss in args.loss:


                for tel in args.tel:


                    for str_train in args.train:
                        # Check train & valid
                        train = f"train{str_train}{tel}" if model not in model_pass else str_train
                        if train not in os.listdir(f"./results/output_simu") and model not in model_pass and "passall" not in sys.argv:
                            raise Exception(f"Train folder {train} not in ./results/output_simu")
                        valid = f"valid2k{tel}"
                        if valid not in os.listdir(f"./results/output_simu") and model not in model_pass and "passall" not in sys.argv:
                            raise Exception(f"Valid folder {valid} not in ./results/output_simu")


                        for lr in args.lr:


                            for load in args.load:


                                if batch == "training":

                                    device = "gpu"
                                    codes.append(f"{batch_codes['training'][0]} model={model} loss={loss} train={train} valid={valid} epoch={args.e} lr={lr} load={load}")
                                    batch_names.append(f"{batch}_{model}_{loss}_{train}_{lr}_{load}")


                                else:

                                    for str_test in args.test:

                                        if str_test == "x":
                                            test = f"test1k{tel}"
                                        elif str_test in ["ext", "ot", "gaussian"]:
                                            test = f"test1k{str_test.upper()}{tel}"
                                        else:
                                            test = f"test{str_test.upper()}{tel}"

                                        # Check test
                                        if test not in os.listdir(f"./results/output_simu") and "passall" not in sys.argv:
                                            raise Exception(f"Test folder {test} not in ./results/output_simu")
                                        elif "passall" not in sys.argv:
                                            ntest = len(os.listdir(f"./results/output_simu/{test}/image"))
                                            print(f"Info : {ntest} images in {test}")
                                        else:
                                            try:
                                                ntest = len(os.listdir(f"./results/output_simu/{test}/image"))
                                                print(f"Info : {ntest} images in {test}")
                                            except:
                                                ntest = None

                                        if batch == "apply":

                                            device = "cpu" if "gpu" not in sys.argv else "gpu"
                                            codes.append(f"{batch_codes['apply'][0]} model={model} loss={loss} train={train} test={test} lr={lr} load={load} {device}")
                                            batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{load}")


                                        elif batch == "analyse":

                                            device = "cpu"
                                            for score in args.score:
                                                codes.append(f"{batch_codes['analyse'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
                                                batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{score}_{load}")

                                        elif batch == "extract_atmo":

                                            device = "cpu"

                                            # if we want multiple cpu
                                            if "ncpu" in dir(args):
                                                partition = give_partition(ntest, int(args.ncpu))
                                                begin_with = np.concatenate((np.array([0]), np.cumsum(partition)[:-1])) # [3, 3, 2, 2] to [0, 3, 6, 8]
                                                for p, b in zip(partition, begin_with):
                                                    codes.append(f"{batch_codes['extract_atmo'][0]} extract_atmo model={model} train={train} test={test} loss={loss} lr={lr} load={load} range={b}_{p}")
                                                    batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{load}_{b}_{p}")
                                            else:
                                                codes.append(f"{batch_codes['extract_atmo'][0]} extract_atmo model={model} train={train} test={test} loss={loss} lr={lr} load={load}")
                                                batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{load}")


                                        elif batch == "analyseFOPA":

                                            device = "cpu"
                                            for score in args.score:
                                                codes.append(f"{batch_codes['analyseFOPA'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
                                                batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{score}_{load}")



    # Construction of SLURM file:

    if make_jobs:

        if not args.mult:

            extsup = "slurm" if not args.local else ""
            generate_batch(batch, codes, device, mem=args.mem, ext=extsup, local=args.local, gpu_device=args.gd, nbj=args.nbj)

        else:

            extsup = "slurm" if not args.local else ""
            ext = "sh" if not args.local else ""

            with open(f"{batch}.{extsup}", "w") as f:
                for name, code in zip(batch_names, codes):
                    generate_batch(name, code, device, ext=ext, mem=args.mem, local=args.local, gpu_device=args.gd, nbj=args.nbj)
                    f.write(f"sbatch {params.path}/{name}.sh\n")
                                    




