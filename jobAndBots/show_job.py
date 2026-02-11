import os, sys, time
import coloralf as c
import numpy as np




def printdebug(msg, debug, color=c.bm):

	if debug:
		print(f"{color}DEBUG : {msg}{c.d}")




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

		printdebug(f"Sh : {sh}", debug)

		lines = f.read().split("\n")

		for line in lines:

			if "python" in line:

				printdebug(f"Get python ... (for {line})", debug)
				params = extraction_code(line)
				params["LINE"] = line

				if "main_simu" in line: 
					printdebug(f"enter in main_simu ...", debug)
					return inspect_simu(params)
				elif "train_models" in line: 
					printdebug(f"enter in train_models ...", debug)
					return inspect_training(params)
				elif "apply_model" in line:
					printdebug(f"enter in apply_model ...", debug)
					return inspect_apply(params)
				elif "apply_spectractor" in line:
					printdebug(f"enter in apply_model ...", debug)
					return inspect_apply_spectractor(params, debug)
				else:
					printdebug(f"Unknow python ...", debug)




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

	if "Spec2vecModels_Results" not in os.listdir("./results"):

		rl = f"Training {model_name} with {load_name}{train_name} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}"
		ra = -1

	if model_name in os.listdir(f"./results/Spec2vecModels_Results") and "epoch" in os.listdir(f"./results/Spec2vecModels_Results/{model_name}") and f"{load_name}{train_name}" in os.listdir(f"./results/Spec2vecModels_Results/{model_name}/epoch/"):

		nb_make = len(os.listdir(f"./results/Spec2vecModels_Results/{model_name}/epoch/{load_name}{train_name}"))
		lmax = len(str(epoch))

		color = get_color(nb_make, epoch)

		labelPC = "" if nb_make == epoch else f" [{nb_make/epoch*100:6.2f} %]"

		rl = f"Training {model_name} with {load_name}{train_name} : {color}{nb_make:{lmax}}/{epoch}{c.d}{labelPC}"
		ra = nb_make/epoch*100


	else:

		rl = f"Training {model_name} with {load_name}{train_name} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}"
		ra = -1

	return rl, ra, "1-training"



def inspect_simu(params):

	path = f"./results/output_simu/{params['f']}"
	

	# for param in params:

	# 	if len(param) > 0:

	# 		print(param)

	# 		if param[:5] == "nsimu" : x = int(param[6:])
	# 		if param[:3] == "set" : s = param[4:]

	x = int(params["nsimu"])
	s = params["set"]


	if params['f'] in os.listdir(f"./results/output_simu/"):

		nb_make = len(os.listdir(f"{path}/image"))
		color = get_color(nb_make, x)
		lmax = len(str(x))

		labelPC = "" if nb_make == x else f" [{nb_make/x*100:6.2f} %]"

		rl = f"Simulator {params['f']} > {s} : {color}{nb_make:{lmax}}/{x}{c.d}{labelPC}"
		ra = nb_make/x*100


	else:

		rl = f"Simulator {params['f']} > {s} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}"
		ra = -1

	return rl, ra, "0-simu"



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

		labelPC = "" if nb_make == to_make else f" [{nb_make/to_make*100:6.2f} %]"

		rl = f"Apply {model}_{loss} > {load_name}{train}_{lr} on {params['test']} : {color}{nb_make:{lmax}}/{to_make}{c.d}{labelPC}"
		ra = nb_make/to_make*100

	else:

		rl = f"Apply {model}_{loss} > {load_name}{train}_{lr} on {params['test']} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}"
		ra = -1

	return rl, ra, "2-apply"




def inspect_apply_spectractor(params, debug):

	pred_folder = "pred_Spectractor_x_x_0e+00"

	test = params["LINE"].split(" ")[2]
	a, z = params["range"].split("_")

	to_make = len(os.listdir(f"./results/output_simu/{test}/spectrum"))
	nb_len = len(str(to_make))
	nb_make = 0

	if pred_folder in os.listdir(f"./results/output_simu/{test}"):

		for i in range(a, z):

			if f"spectrum_{i:0>{nb_len}}" in os.listdir(f"./results/output_simu/{test}/{pred_folder}"):
				nb_make += 1

		color = get_color(nb_make, to_make)
		lmax = len(str(to_make))

		labelPC = "" if nb_make == to_make else f" [{nb_make/to_make*100:6.2f} %]"

		rl = f"Apply spectractor > {test} : {color}{nb_make:{lmax}}/{to_make}{c.d}{labelPC}"
		ra = nb_make/to_make*100

	else:

		rl = f"Apply spectractor > {test} : {c.r}{c.tu}{c.ti}Not exist ...{c.d}"
		ra = -1

	return rl, ra, "2-apply_spectractor"














def get_color(nb_make, nb_total):

	if   nb_make == nb_total : color = c.lg
	elif nb_make > nb_total * 0.8 : color = c.ly
	elif nb_make > 0 : color = c.lr
	else : color = c.lk

	return color


def countingColor(pcs, c100=0):

	# Not exist
	x = np.sum(pcs < 0)
	print(f"Total {c.r}{c.tu}{c.ti}Not exist{c.d} : {x}")

	x0 = np.sum(pcs == 0.0)
	print(           f"Total {c.lk}0 %{c.d}       : {x0}")

	xMide = np.sum((pcs > 0.0) & (pcs <= 80.0))
	print(           f"Total {c.lr}< 80 %{c.d}    : {xMide}")

	x80 = np.sum((pcs > 80.0) & (pcs < 100.0))
	print(           f"Total {c.ly}> 80 %{c.d}    : {x80}")

	x100 = np.sum(pcs == 100.0) + c100
	print(           f"Total {c.lg}Finish{c.d}    : {x100}")




if __name__ == "__main__":

	print(f"TIME : {c.m}{c.ti}{time.ctime()}{c.d}")

	if len(sys.argv) > 1:
		debug = True if sys.argv[1] == 'debug' else False
	else:
		debug = False

	shs = [file for file in os.listdir() if file[-3:] == ".sh"] + [file for file in os.listdir() if file[-6:] == ".slurm"]

	labels = list()
	advanc = list()
	typesj = list()

	c100 = 0

	for sh in shs :
		extract = extraction(sh, debug)
		
		if extract is not None:

			if "nf" in sys.argv and extract[1] >= 100.0:

				c100 += 1

			else:

				labels.append(extract[0])
				advanc.append(extract[1])
				typesj.append(extract[2])

	if "type" in sys.argv and "n" in sys.argv: argsort = np.lexsort((labels, typesj))
	elif "type" in sys.argv : argsort = np.lexsort((advanc, typesj))
	elif "n" in sys.argv : argsort = np.lexsort((advanc, labels))
	else : argsort = np.argsort(advanc)

	

	for asort in argsort:

		print(labels[asort])

	print(f"")
	countingColor(np.array(advanc), c100=c100)