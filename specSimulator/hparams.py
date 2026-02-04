import os, sys, json
import numpy as np
import coloralf as c
import utils_spec.psf_func as pf





def HparamsFromJson(jsonfile):


    with open(jsonfile, "r") as f:
        hp = json.load(f)

    # If a dataset was created with a old version of Hparams
    if not "seed" in hp.keys():
        hp["seed"] = None

    return Hparams(hp["telescope"], hp["target_set"], hp["psf_function"], hp["vparams"], hp["nsimu"], hp["with_noise"], hp['seed'],
        [hp["LAMBDA_MIN"], hp["LAMBDA_MAX"]], hp["LAMBDA_STEP"], hp["SPECTRACTOR_ATMOSPHERE_SIM"], hp["FLAM_TO_ADURATE"],
        disperser_dir=hp["DISPERSER_DIR"], throughput_dir=hp["THROUGHPUT_DIR"], 
        output_path=".", output_dir=hp["output_path"], output_simu_dir=hp["output_dir"], output_simu_fold=hp["output_fold"])





class Hparams():

    """
    """



    # Targets for spectrums simulations
    __TARGET_SETS = {

        # 20 Calspec to use
        "set0" : ['HD009051', 'HD031128', 'HD101452', 'HD106252', 'HD111980', 
                'HD115169', 'HD142331', 'HD14943',  'HD158485', 'HD159222', 
                'HD160617', 'HD163466', 'HD165459', 'HD167060', 'HD185975', 
                'HD200654', 'HD205905', 'HD209458',  'HD37962',  'HD38949'],
                
        # 11 Calspec en +
        "set1" : ['HD074000', 'HD116405', 'HD128998', 'HD172728', 'HD180609', 
                  'HD2811',   'HD37725',  'HD55677',  'HD60753',  'HD93521'], #HD200775

        # THE Calspec
        "set2" : ['HD111980'],

        # set0 + calib
        "set0calib" : ['HD009051', 'HD031128', 'HD101452', 'HD106252', 'HD111980', 
                'HD115169', 'HD142331', 'HD14943',  'HD158485', 'HD159222', 
                'HD160617', 'HD163466', 'HD165459', 'HD167060', 'HD185975', 
                'HD200654', 'HD205905', 'HD209458',  'HD37962',  'HD38949',
                'calib'],

        "setCalib" : ["calib"],

        # set0 + calPX
        "set0calPX" : ['HD009051', 'HD031128', 'HD101452', 'HD106252', 'HD111980', 
                'HD115169', 'HD142331', 'HD14943',  'HD158485', 'HD159222', 
                'HD160617', 'HD163466', 'HD165459', 'HD167060', 'HD185975', 
                'HD200654', 'HD205905', 'HD209458',  'HD37962',  'HD38949',
                'calPX'],

        "setCalPX" : ["calPX"],

        # setAll
        "setAll" : ['HD009051', 'HD031128', 'HD101452', 'HD106252', 'HD111980', 
                'HD115169', 'HD142331', 'HD14943',  'HD158485', 'HD159222', 
                'HD160617', 'HD163466', 'HD165459', 'HD167060', 'HD185975', 
                'HD200654', 'HD205905', 'HD209458', 'HD37962',  'HD38949',
                'HD074000', 'HD116405', 'HD128998', 'HD172728', 'HD180609', 
                'HD2811',   'HD37725',  'HD55677',  'HD60753',  'HD93521']
    }



    # Liste of telescope parameters
    __TELESCOPES_KEYS = ["SIM_NX", "SIM_NY", "R0",

            # Instrument characteristics
            "THROUGHPUT",
            "OBS_NAME",
            "OBS_ALTITUDE",  # altitude in k meters from astropy package
            "OBS_LATITUDE",  # latitude
            "OBS_SURFACE",  # Effective surface of the telescope in cm**2 accounting for obscuration
            "OBS_EPOCH",
            "OBS_OBJECT_TYPE",  # To choose between STAR, HG-AR, MONOCHROMATOR
            "OBS_FULL_INSTRUMENT_TRANSMISSON", # Full instrument transmission file
            "OBS_TRANSMISSION_SYSTEMATICS",
            "OBS_CAMERA_ROTATION",  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
            "OBS_CAMERA_DEC_FLIP_SIGN",  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_CAMERA_RA_FLIP_SIGN",  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_PRESSURE",

            # CCD characteristics
            "CCD_PIXEL2MM",  # pixel size in mm
            "CCD_PIXEL2ARCSEC",  # pixel size in arcsec
            "CCD_MAXADU",  # approximate maximum ADU output of the CCD
            "CCD_GAIN", # electronic gain : elec/ADU
            "CCD_REBIN", # rebinning of the image in pixel
            "DISTANCE2CCD", # distance between hologram and CCD in mm
            "DISTANCE2CCD_ERR",

            "DISPERSER"]

    __TELESCOPES = {

        # alf
        "alf" : {

            # Simu parameters
            "SIM_NX" : 1024,
            "SIM_NY" : 128,
            "R0" : [64, 64],

            # Instrument characteristics
            "THROUGHPUT" : "CTIOThroughput/ctio_throughput_300517_v1.txt",
            "TEL_NAME" : "ctio",
            "OBS_NAME" : 'CTIO',
            "OBS_ALTITUDE" : 2.200,  # CTIO altitude in k meters from astropy package (Cerro Pachon)
            "OBS_LATITUDE" : '-30 10 07.90',  # CTIO latitude
            "OBS_SURFACE" : 6361,  # Effective surface of the telescope in cm**2 accounting for obscuration
            "OBS_EPOCH" : "J2000.0",
            "OBS_OBJECT_TYPE" : 'STAR',  # To choose between STAR, HG-AR, MONOCHROMATOR
            "OBS_FULL_INSTRUMENT_TRANSMISSON" : 'ctio_throughput_300517_v1.txt', # Full instrument transmission file
            "OBS_TRANSMISSION_SYSTEMATICS" : 0.005,
            "OBS_CAMERA_ROTATION" : 180,  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
            "OBS_CAMERA_DEC_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_CAMERA_RA_FLIP_SIGN" : -1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_PRESSURE" : 784.0, # 784.0

            # CCD characteristics
            "CCD_PIXEL2MM" : 24e-3,  # pixel size in mm
            "CCD_PIXEL2ARCSEC" : 0.401,  # pixel size in arcsec
            "CCD_MAXADU" : 60000,  # approximate maximum ADU output of the CCD
            "CCD_GAIN" : 3.,  # electronic gain : elec/ADU
            "CCD_REBIN" : 1,  # rebinning of the image in pixel
            "DISTANCE2CCD" : 55.45,  # distance between hologram and CCD in mm
            "DISTANCE2CCD_ERR" : 0.2,  # uncertainty on distance between hologram and CCD in mm

            # Disperser
            "DISPERSER" : "HoloAmAg",
        },

        # CTIO
        "ctio" : {

            # Simu parameters
            "SIM_NX" : 1024,
            "SIM_NY" : 128,
            "R0" : [64, 64],

            # Instrument characteristics
            "THROUGHPUT" : "CTIOThroughput/ctio_throughput_300517_v1.txt",
            "OBS_NAME" : 'CTIO',
            "OBS_ALTITUDE" : 2.200,  # CTIO altitude in k meters from astropy package (Cerro Pachon)
            "OBS_LATITUDE" : '-30 10 07.90',  # CTIO latitude
            "OBS_SURFACE" : 6361,  # Effective surface of the telescope in cm**2 accounting for obscuration
            "OBS_EPOCH" : "J2000.0",
            "OBS_OBJECT_TYPE" : 'STAR',  # To choose between STAR, HG-AR, MONOCHROMATOR
            "OBS_FULL_INSTRUMENT_TRANSMISSON" : 'ctio_throughput_300517_v1.txt', # Full instrument transmission file
            "OBS_TRANSMISSION_SYSTEMATICS" : 0.005,
            "OBS_CAMERA_ROTATION" : 180,  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
            "OBS_CAMERA_DEC_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_CAMERA_RA_FLIP_SIGN" : -1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_PRESSURE" : 784.0, # 784.0

            # CCD characteristics
            "CCD_PIXEL2MM" : 24e-3,  # pixel size in mm
            "CCD_PIXEL2ARCSEC" : 0.401,  # pixel size in arcsec
            "CCD_MAXADU" : 60000,  # approximate maximum ADU output of the CCD
            "CCD_GAIN" : 3.,  # electronic gain : elec/ADU
            "CCD_REBIN" : 1,  # rebinning of the image in pixel
            "DISTANCE2CCD" : 55.45,  # distance between hologram and CCD in mm
            "DISTANCE2CCD_ERR" : 0.19,  # uncertainty on distance between hologram and CCD in mm

            # Disperser
            "DISPERSER" : "HoloAmAg",
        },

        # StarDice
        "stardice" : {

            # Simu parameters
            "SIM_NX" : 1024,
            "SIM_NY" : 128,
            "R0" : [64, 64],

            # Instrument characteristics
            "THROUGHPUT" : "StarDiceThroughput/StarDice_EMPTY_response_75um_pinhole.txt",
            "OBS_NAME" : 'OHP',
            "OBS_ALTITUDE" : 0.650,
            "OBS_LATITUDE" : '+43 55 57.449',
            "OBS_SURFACE" : 1161.6, 
            "OBS_EPOCH" : "J2000.0",
            "OBS_OBJECT_TYPE" : 'STAR',  # To choose between STAR, HG-AR, MONOCHROMATOR
            "OBS_FULL_INSTRUMENT_TRANSMISSON" : 'StarDice_EMPTY_response_75um_pinhole.txt',
            "OBS_TRANSMISSION_SYSTEMATICS" : 0.005,
            "OBS_CAMERA_ROTATION" : 180,  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
            "OBS_CAMERA_DEC_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_CAMERA_RA_FLIP_SIGN" : -1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_PRESSURE" : 937.2,

            # CCD characteristics
            "CCD_PIXEL2MM" : 13e-3,  # pixel size in mm
            "CCD_PIXEL2ARCSEC" : 1.674,  # pixel size in arcsec
            "CCD_MAXADU" : 60000,  # approximate maximum ADU output of the CCD
            "CCD_GAIN" : 1.2,  # electronic gain : elec/ADU
            "CCD_REBIN" : 1,  # rebinning of the image in pixel
            "DISTANCE2CCD" : 33.3,  # distance between hologram and CCD in mm
            "DISTANCE2CCD_ERR" : 0.1,  # uncertainty on distance between hologram and CCD in mm

            # Disperser
            "DISPERSER" : "star_analyzer_200",
        },

        # auxtel
        "auxtel" : {

            # Simu parameters
            "SIM_NX" : 4096,
            "SIM_NY" : 512,
            "R0" : [256, 256],

            # Instrument characteristics
            "THROUGHPUT" : "AuxTelThroughput/multispectra_holo4_003_HD142331_AuxTel_throughput.txt",
            "OBS_NAME" : 'LSST',
            "OBS_ALTITUDE" : 2.66299616375123,
            "OBS_LATITUDE" : '-30 14 40.7',
            "OBS_SURFACE" : 9636.0, 
            "OBS_EPOCH" : "J2000.0",
            "OBS_OBJECT_TYPE" : 'STAR',  # To choose between STAR, HG-AR, MONOCHROMATOR
            "OBS_FULL_INSTRUMENT_TRANSMISSON" : 'multispectra_holo4_003_HD142331_AuxTel_throughput.txt',
            "OBS_TRANSMISSION_SYSTEMATICS" : 0.005,
            "OBS_CAMERA_ROTATION" : 0,  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
            "OBS_CAMERA_DEC_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_CAMERA_RA_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_PRESSURE" : 731.5,

            # CCD characteristics
            "CCD_PIXEL2MM" : 10e-3,  # pixel size in mm
            "CCD_PIXEL2ARCSEC" : 0.0952,  # pixel size in arcsec
            "CCD_MAXADU" : 60000,  # approximate maximum ADU output of the CCD
            "CCD_GAIN" : 1.3,  # electronic gain : elec/ADU
            "CCD_REBIN" : 2,  # rebinning of the image in pixel
            "DISTANCE2CCD" : 187.1,  # distance between hologram and CCD in mm
            "DISTANCE2CCD_ERR" : 0.05,  # uncertainty on distance between hologram and CCD in mm

            # Disperser
            "DISPERSER" : "holo4_003",
        },

        "auxtelqn" : {

            # Simu parameters
            "SIM_NX" : 4096,
            "SIM_NY" : 512,
            "R0" : [256, 256],

            # Instrument characteristics
            "THROUGHPUT" : "AuxTelThroughput/multispectra_holo4_003_HD142331_AuxTel_throughput.txt",
            "OBS_NAME" : 'LSST',
            "OBS_ALTITUDE" : 2.66299616375123,
            "OBS_LATITUDE" : '-30 14 40.7',
            "OBS_SURFACE" : 9636.0, 
            "OBS_EPOCH" : "J2000.0",
            "OBS_OBJECT_TYPE" : 'STAR',  # To choose between STAR, HG-AR, MONOCHROMATOR
            "OBS_FULL_INSTRUMENT_TRANSMISSON" : 'multispectra_holo4_003_HD142331_AuxTel_throughput.txt',
            "OBS_TRANSMISSION_SYSTEMATICS" : 0.005,
            "OBS_CAMERA_ROTATION" : 0,  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
            "OBS_CAMERA_DEC_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_CAMERA_RA_FLIP_SIGN" : 1,  # Camera (x,y) flip signs with respect to (north-up, east-left) system
            "OBS_PRESSURE" : 731.5,

            # CCD characteristics
            "CCD_PIXEL2MM" : 10e-3,  # pixel size in mm
            "CCD_PIXEL2ARCSEC" : 0.0952,  # pixel size in arcsec
            "CCD_MAXADU" : 60000,  # approximate maximum ADU output of the CCD
            "CCD_GAIN" : 1.3,  # electronic gain : elec/ADU
            "CCD_REBIN" : 2,  # rebinning of the image in pixel
            "DISTANCE2CCD" : 187.1,  # distance between hologram and CCD in mm
            "DISTANCE2CCD_ERR" : 0.05,  # uncertainty on distance between hologram and CCD in mm

            # Disperser
            "DISPERSER" : "blue300lpmm_qn1",
        }
    }



    # Parameters whitch can be variables in simulation
    __PARAMS = {

        # Facteur Amplitude of order:
        "A0" : 1.0, # order 0
        "A1" : 1.0, # order 1
        "A2" : 1.0, # order 2
        "A3" : 0.0, # order 3
        "A" : 1.0, # all order
        
        # Angle of dispersion
        "ROTATION_ANGLE" : 0.0,

        # For atmoshere
        "ATM_AEROSOLS" : 0.05, # 0.00 et 1.00
        "ATM_OZONE" : 300.0, # 250 et 350
        "ATM_PWV" : 5.0, # 0.0 et 15.0
        "ATM_AIRMASS" : 1.2, # 1.0 et 2.5
        "ATM_ANGSTROM_EXPONENT" : 1.2,
        "ATM_TEMPERATURE" : 8.6,
        "ATM_HUMIDITY" : 25,

        # For ADR
        "ADR_DEC" : -18.6,
        "ADR_HOUR_ANGLE" : 28.2,

        # Divers
        "EXPOSURE" : 120.0,
        "BACKGROUND_LEVEL" : 0.5,
        "CCD_READ_OUT_NOISE" : 12.0, # e-
    } 



    __PSF_FUNCTIONS = {

        "moffat2d" : {
            'f' : pf.moffat2d_jit, # f : def func of (XX, YY, amplitude, x, y, f_argv[0](l, *argv[0]), ..., f_argv[n](l, *argv[n])), for l lambdas in nm
            'f_arg' : [pf.simpleLinear, pf.simpleLinear], # function for argument
            'arg' : [[3.0], [3.0]], # argument for argument function
            'order0' : {'amplitude':22900.0, 'arg':[3.0, 2.0]}, # argument order 0
            'timbre' : pf.moffat2d_timbre, # timbre size function
        },

        "gaussian2d" : {
            'f' : pf.gaussian2d_jit, # f : def func of (XX, YY, amplitude, x, y, f_argv[0](l, *argv[0]), ..., f_argv[n](l, *argv[n])), for l lambdas in nm
            'f_arg' : [pf.simpleLinear], # function for argument
            'arg' : [[3.0]], # argument for argument function
            'order0' : {'amplitude':22900.0, 'arg':[3.0]}, # argument order 0
            'timbre' : pf.gaussian2d_timbre, # timbre size function
        }
    }



    def __init__(self, telescope=None, target_set=None, psf=None, var_params=dict(), nsimu=None, with_noise=True, seed=None,
                 lambdas=[300, 1100], lambdas_step=1, atmo_model="getobsatmo", flam2adurate=1_067_400_516_204.6393,
                 disperser_dir="./SpecSimulator/datafile/dispersers",
                 throughput_dir="./SpecSimulator/datafile/throughput",
                 output_path=".", output_dir = "results", output_simu_dir="output_simu", output_simu_fold="simulation"):

        """
        """

        # capture argv
        self.init_argv()

        # LAMBDA PARAMS
        self.LAMBDA_MIN = lambdas[0]
        self.LAMBDA_MAX = lambdas[1]
        self.LAMBDA_STEP = lambdas_step
        self.LAMBDAS = np.arange(self.LAMBDA_MIN, self.LAMBDA_MAX, self.LAMBDA_STEP)
        self.N = len(self.LAMBDAS)
        self.with_noise = with_noise

        # SOME CONFIG
        self.FLAM_TO_ADURATE = flam2adurate
        self.SPECTRACTOR_ATMOSPHERE_SIM = atmo_model
        self.DISPERSER_DIR = disperser_dir
        self.THROUGHPUT_DIR = throughput_dir
        self.CCD_ARCSEC2RADIANS = 1 / 180 * np.pi / 3600
        self.GRATING_ORDER_2OVER1 = 0.1  # default value for order 2 over order 1 transmission ratio

        # SOME PATHS and DIRS
        self.output_path = output_path + "/" + output_dir
        try:
            if output_dir not in os.listdir(output_path):
                os.mkdir(self.output_path)
        except Exception as e:
            print(f"INFO warning : {e}...")

        self.output_dir = output_simu_dir
        self.output_fold = output_simu_fold if "f" not in self.argv else self.argv["f"]

        # INIT ALL PARAMETERS
        self.init_target_set(target_set)
        self.init_telescope(telescope)
        self.init_psf_function(psf)
        self.init_var_params(var_params)
        self.init_nb_simu(nsimu)
        self.init_seed(seed)

        # UPDATE IF REBIN != 1
        if self.CCD_REBIN != 1:

            print(f"\n{c.g}INFO : CCD_REBIN != 1, update of some parameters ...{c.d}")
            self.oldSIM_NX = self.SIM_NX
            self.oldSIM_NY = self.SIM_NY
            self.oldR0 = self.R0.copy()

            self.SIM_NX = int(self.SIM_NX // self.CCD_REBIN)
            self.SIM_NY = int(self.SIM_NY // self.CCD_REBIN)
            self.R0[0] = int(self.R0[0] // self.CCD_REBIN)
            self.R0[1] = int(self.R0[1] // self.CCD_REBIN)
            self.CCD_PIXEL2MM *= self.CCD_REBIN
            self.CCD_PIXEL2ARCSEC *= self.CCD_REBIN

            print(f"SIM_NX : {c.lk}{self.oldSIM_NX}{c.d} -> {self.SIM_NX}")
            print(f"SIM_NY : {c.lk}{self.oldSIM_NY}{c.d} -> {self.SIM_NY}")
            print(f"R0     : {c.lk}{self.oldR0}{c.d} -> {self.R0}")



    def init_argv(self):

        self.argv = {"__free__":list()}

        for argv in sys.argv[1:]:

            if "=" in argv:

                k, v = argv.split("=")
                self.argv[k] = v

            else:

                self.argv["__free__"].append(argv)



    def init_target_set(self, target_set):

        # Def of target set
        if target_set is not None and "set" in self.argv.keys():
            print(f"\n{c.r}WARNING [Hparams.py] : {c.ti}target_set{c.ri} is define twice ; in argument of Hparams and in sys.argv -> sys.argv has priority{c.r}")
            target_set = self.argv["set"]

        elif target_set is None and "set" in self.argv.keys():
            target_set = self.argv["set"]

        elif target_set is None:
            print(f"\n{c.y}INFO [Hparams.py] : {c.ti}target_set{c.ri} is not set, default {c.ti}set0{c.ri} (witch containt 20 calspecs){c.d}")
            target_set = "set0"

        print(f"{c.g}INFO : {c.ti}target_set{c.ri} set to {c.ti}{target_set}{c.ri}{c.d}")
        self.target_set = target_set

        self.target_name = self.__TARGET_SETS[self.target_set.lower()]



    def init_telescope(self, telescope):

        # Def of telescope
        if telescope is not None and "tel" in self.argv.keys():
            print(f"\n{c.r}WARNING [Hparams.py] : {c.ti}telescope{c.ri} is define twice ; in argument of Hparams and in sys.argv -> sys.argv has priority{c.r}")
            telescope = self.argv["tel"]

        elif telescope is None and "tel" in self.argv.keys():
            telescope = self.argv["tel"]

        elif telescope is None:
            print(f"\n{c.y}INFO [Hparams.py] : {c.ti}telescope{c.ri} is not set, default {c.ti}ctio{c.ri}{c.d}")
            telescope = "ctio"

        
        print(f"{c.g}INFO : {c.ti}telescope{c.ri} set to {c.ti}{telescope}{c.ri}{c.d}")
        self.telescope = telescope

        # set all parameters from __TELESCOPES to hparams
        for k, v in self.__TELESCOPES[self.telescope.lower()].items():
            setattr(self, k, v)

        # checks if the required parameters are present
        exitProg = False
        for k in self.__TELESCOPES_KEYS:
            if k not in dir(self):
                print(f"{c.r}WARNING : {c.ti}{k}{c.ri} missing in telescope parameters ({c.ti}{self.telescope}{c.ri}){c.d}")
                exitProg = True
        if exitProg:
            raise Exception(f"{c.r}Error, complete missing parameter(s)...{c.d}")



    def init_psf_function(self, psf):

        # Def of telescope
        if psf is not None and "psf" in self.argv.keys():
            print(f"\n{c.r}WARNING [Hparams.py] : {c.ti}psf{c.ri} is define twice ; in argument of Hparams and in sys.argv -> sys.argv has priority{c.r}")
            psf = self.argv["psf"]

        elif psf is None and "psf" in self.argv.keys():
            psf = self.argv["psf"]

        elif psf is None:
            print(f"\n{c.y}INFO [Hparams.py] : {c.ti}psf{c.ri} is not set, default {c.ti}moffat2d{c.ri}{c.d}")
            psf = "moffat2d"

        print(f"{c.g}INFO : {c.ti}psf function{c.ri} set to {c.ti}{psf}{c.ri}{c.d}")
        self.psf_function = psf

        self.psf = self.__PSF_FUNCTIONS[self.psf_function.lower()]




    def init_nb_simu(self, nsimu):

        # Def of telescope
        if nsimu is not None and "nsimu" in self.argv.keys():
            print(f"\n{c.r}WARNING [Hparams.py] : {c.ti}nsimu{c.ri} is define twice ; in argument of Hparams and in sys.argv -> sys.argv has priority{c.r}")
            nsimu = int(self.argv["nsimu"])

        elif nsimu is None and "nsimu" in self.argv.keys():
            nsimu = int(self.argv["nsimu"])

        elif nsimu is None:
            print(f"\n{c.y}INFO [Hparams.py] : {c.ti}nsimu{c.ri} is not set, default {c.ti}10{c.ri}{c.d}")
            nsimu = 10

        print(f"{c.g}INFO : {c.ti}nsimu{c.ri} set to {c.ti}{nsimu}{c.ri}{c.d}")
        self.nsimu = nsimu




    def init_seed(self, seed):

        # Def of telescope
        if "seed" in self.argv.keys() and self.argv["seed"] == "None":
            seed = None

        elif seed is not None and "seed" in self.argv.keys():
            print(f"\n{c.r}WARNING [Hparams.py] : {c.ti}seed{c.ri} is define twice ; in argument of Hparams and in sys.argv -> sys.argv has priority{c.r}")
            seed = int(self.argv["seed"])

        elif seed is None and "seed" in self.argv.keys():
            seed = int(self.argv["seed"])

        elif seed is None:
            print(f"\n{c.y}INFO [Hparams.py] : {c.ti}seed{c.ri} is not set, default {c.ti}None{c.ri}{c.d}")
            seed = None

        print(f"{c.g}INFO : {c.ti}seed{c.ri} set to {c.ti}{seed}{c.ri}{c.d}")
        self.seed = seed




    def init_var_params(self, var_params):

        self.cparams = dict() # constante
        self.vparams = dict() # variable
        self.aparams = dict() # args for psf function 

        for k, v in self.__PARAMS.items():

            if k in var_params.keys():

                print(f"Set var param {c.lm}{k}{c.d} to range {c.lm}{var_params[k]}{c.d}")
                self.vparams[k] = var_params[k]

            else:

                print(f"Set constante param {c.m}{k}{c.d} to {c.m}{v}{c.d}")
                self.cparams[k] = v


        for k, v in var_params.items():

            if k not in self.vparams.keys():

                if k.startswith("arg."):

                    num_arg, num_coef = k.split('.')[1:]
                    self.aparams[k] = [int(num_arg), int(num_coef)]
                    self.vparams[k] = v

                else:
                    print(f"{c.r}INFO : le parametre variable {k} n'est pas utilisé car n'est pas dans les __PARAMS de Hparams{c.d}")


    def save(self, path=None, file="hparams", debug=False):

        if path is None:
            path = f"{self.output_path}/{self.output_dir}/{self.output_fold}"

        dico = dict()

        for p in dir(self):

            if p not in ['LAMBDAS', 'psf'] and not "__" in p and not callable(getattr(self, p)):

                dico[p] = getattr(self, p)
                if debug : print(f"\n{c.m}{p}{c.d} : {getattr(self, p)}")


        with open(f"{path}/{file}.json", 'w') as f:
            json.dump(dico, f, indent=4)



if __name__ == "__main__":

    vp = {"A":[0.5, 2.0], "ROTATION_ANGLE":[-1.0, 1.0], "arg.0.0":[2.0, 8.0], "ccbb":[0.0, 999.0]}

    hp = Hparams(var_params=vp)
    hp.save(path=".", debug=True)









