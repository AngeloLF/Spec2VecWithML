import os, sys, time
import coloralf as c




def extraction_code(code):

	args = code.split(" ")

	params = dict()

	for arg in args:

		if "=" in arg:
			k, v = arg.split("=")
			params[k] = v
		else:
			params[arg] = None

	return params





def extraction(sh, debug):

	with open(sh, "r") as f:

		if debug : print(f"\nSh : {c.ti}{sh}{c.d}")

		lines = f.read().split("\n")

		for line in lines:

			if "python" in line:

				if debug : print(f"{c.ti}Get python ...{c.d}{c.lk}(for line `{c.lg}{line}{c.lk}`{c.d}")
				params = extraction_code(line)

				if "train_models" in line: 
					if debug : print(f"{c.ti}train_models ...{c.d}")
					inspect_training(params)
				elif "alfsimu" in line: 
					if debug : print(f"{c.ti}alfsimu ...{c.d}")
					inspect_simu(params)
				elif "apply_model" in line:
					if debug : print(f"{c.ti}apply_model ...{c.d}")
					inspect_apply(params)
				else:
					if debug : print(f"{c.ti}Unknow ...{c.d}")




def inspect_training(params):

	LR = f"{float(params['lr']):.0e}"

	if "load" in params.keys() and params["load"] != "None":

		pre_train, pre_lr = params['load'].split('_')
		pre_LR = f"{float(pre_lr):.0e}"
		load_name = f"{pre_train}_{pre_LR}_"

	else:

		load_name = ""

	model_name = f"{params['model']}_{params['loss']}"
	train_name = f"{params['train']}_{LR}"
	epoch = int(params['epoch'])

	if model_name in os.listdir(f"./results/Spec2vecModels_Results") and "epoch" in os.listdir(f"./results/Spec2vecModels_Results/{model_name}") and f"{load_name}{train_name}" in os.listdir(f"./results/Spec2vecModels_Results/{model_name}/epoch/"):

		nb_make = len(os.listdir(f"./results/Spec2vecModels_Results/{model_name}/epoch/{load_name}{train_name}"))
		lmax = len(str(epoch))

		color = get_color(nb_make, epoch)

		print(f"Training {model_name} with {load_name}{train_name} : {color}{nb_make:{lmax}}/{epoch}{c.d} [{nb_make/epoch*100:6.2f} %]")

	else:

		print(f"Training {model_name} with {load_name}{train_name} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}")



def inspect_simu(params):

	path = f"./results/output_simu/{params['f']}"
	

	for param in params:

		if len(param) > 0:

			if param[0] == "x" : x = int(param[1:])
			if param[:3] == "set" : s = param


	if params['f'] in os.listdir(f"./results/output_simu/"):

		nb_make = len(os.listdir(f"{path}/image"))
		color = get_color(nb_make, x)
		lmax = len(str(x))

		print(f"Simulator {params['f']} > {s} : {color}{nb_make:{lmax}}/{x}{c.d} [{nb_make/x*100:6.2f} %]")

	else:

		print(f"Simulator {params['f']} > {s} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}")



def inspect_apply(params):

	model, loss, train, lr = params['model'], params['loss'], params['train'], f"{float(params['lr']):.0e}"

	if "load" in params.keys() and params["load"] != "None":

		pre_train, pre_lr = params['load'].split('_')
		pre_LR = f"{float(pre_lr):.0e}"
		load_name = f"{pre_train}_{pre_LR}_"

	else:

		load_name = ""

	path = f"./results/output_simu/{params['test']}"
	pred_folder = f"pred_{model}_{loss}_{load_name}{train}_{lr}"

	to_make = len(os.listdir(f"{path}/spectrum"))


	if pred_folder in os.listdir(path):

		nb_make = len(os.listdir(f"{path}/{pred_folder}"))
		color = get_color(nb_make, to_make)
		lmax = len(str(to_make))

		print(f"Apply {model}_{loss} > {load_name}{train}_{lr} on {params['test']} : {color}{nb_make:{lmax}}/{to_make}{c.d} [{nb_make/to_make*100:6.2f} %]")

	else:

		print(f"Apply {model}_{loss} > {load_name}{train}_{lr} on {params['test']} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}")



def get_color(nb_make, nb_total):

	if   nb_make == nb_total : color = c.lg
	elif nb_make > nb_total * 0.8 : color = c.ly
	elif nb_make > 0 : color = c.lr
	else : color = c.lk

	return color



if __name__ == "__main__":

	print(f"TIME : {c.m}{c.ti}{time.ctime()}{c.d}")

	if len(sys.argv) > 1:
		debug = True if sys.argv[1] == 'debug' else False
	else:
		debug = False

	shs = [file for file in os.listdir() if file[-3:] == ".sh"] + [file for file in os.listdir() if file[-6:] == ".slurm"]

	for sh in shs:
		extraction(sh, debug)










