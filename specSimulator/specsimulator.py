import os, sys, shutil, json, platform
import numpy as np
import matplotlib.pyplot as plt
import coloralf as c
import astropy.units as u
from scipy.interpolate import interp1d
import scipy.stats as stats
from time import time, ctime
from getCalspec import getCalspec
from tqdm import tqdm

from utils_spec.ctTime import ctTime
from utils_spec.adr import adr_calib
from utils_spec.load_disperser import MyDisperser




class SpecSimulator():

    """
    ---
    """

    def __init__(self, hparameters, with_adr=True, with_atmosphere=True, with_background=True, with_flat=True, with_convertADU=True,
                    show_times=True, show_specs=True, savingFolders=True, overwrite=True, verbose=2, spectroData=False):

        """
        verbose :
            * 0 pour aucun print, a par la pbar et le nom du folder
            * 1 pour les print principaux
            * 2 pour tout les prints
        """


        time_init = time()

        # Save hparams
        self.hp = hparameters


        # PSF function and output_dir for simulation results.
        self.psf_function = self.hp.psf
        if self.hp.output_dir not in os.listdir(self.hp.output_path) : os.mkdir(f"{self.hp.output_path}/{self.hp.output_dir}")
        self.output_dir = f"{self.hp.output_path}/{self.hp.output_dir}"

        # Parameters define by sys.argv
        self.nb_simu = self.hp.nsimu
        self.len_simu = len(str(self.nb_simu-1))

        self.show_times = show_times
        self.show_specs = show_specs
        self.overwrite = overwrite
        self.output_fold = self.hp.output_fold
        self.verbose = verbose
        self.colorSimu = False if "color" not in self.hp.argv["__free__"] else True
        self.no0 = False if "no0" not in self.hp.argv["__free__"] else True # no order 0
        self.spectroData = spectroData


        # Initialisation
        if self.verbose >= 0: 
            print(f"{c.y}\nInitialisation of SpecSimulator at {c.d}{c.ly}{c.ti}{ctime()}{c.d}{c.d}")

        # Init seed
        if self.hp.seed is not None:
            print(f"{c.y}Set seed for numpy to {self.hp.seed}{c.d}")
            np.random.seed(self.hp.seed)
        else:
            print(f"{c.y}Spectractor not seeded{c.d}")

        # Define variables parameters for the simulation
        self.init_var_params()
        if self.no0 : self.A0 = 0.0

        # Define output directory
        self.savingFolders = savingFolders
        if self.savingFolders:
            num = 0
            self.save_fold = self.output_fold
            if not self.overwrite:
                while self.save_fold in os.listdir(self.output_dir):
                    num += 1
                    self.save_fold = self.output_fold + "_" + str(num)
            elif self.save_fold in os.listdir(self.output_dir):
                if self.verbose > 0: 
                    print(f"{c.y}Overwriting... delete of {self.output_dir}/{self.save_fold}{c.d}")
                shutil.rmtree(f"{self.output_dir}/{self.save_fold}")
            if self.verbose >= 0: 
                print(f"{c.y}Create folder {self.output_dir}/{self.save_fold}{c.d}")
            os.mkdir(f"{self.output_dir}/{self.save_fold}")
            os.mkdir(f"{self.output_dir}/{self.save_fold}/spectrum")
            if self.spectroData: 
                os.mkdir(f"{self.output_dir}/{self.save_fold}/spectro")
                os.mkdir(f"{self.output_dir}/{self.save_fold}/spectrumPX")
            os.mkdir(f"{self.output_dir}/{self.save_fold}/image")
            os.mkdir(f"{self.output_dir}/{self.save_fold}/imageRGB")
            os.mkdir(f"{self.output_dir}/{self.save_fold}/divers")
            os.mkdir(f"{self.output_dir}/{self.save_fold}/opa")
            if self.hp.telescope in ["auxtel", "auxtelqn"]:
                os.mkdir(f"{self.output_dir}/{self.save_fold}/imageOrigin")
        
        # Order 0 coord.
        self.R0 = self.hp.R0

        # Simulation size
        self.Nx = self.hp.SIM_NX
        self.Ny = self.hp.SIM_NY
        self.xpixels = np.arange(0, self.Nx)
        self.yy, self.xx = np.mgrid[:self.Ny, :self.Nx].astype(np.int32)
        self.pixels = np.asarray([self.xx, self.yy])

        # Class ctTime, for detailled execution time
        self.ctt = ctTime("SpecSimulation", verbose=self.show_times, nbLoop=self.nb_simu)

        # Definition of lambdas
        self.N = self.hp.N
        self.lambdas = self.hp.LAMBDAS
        self.lambda_adr_ref = 550 #550

        # Loading Disperser, Amplitude and transmission ratio
        self.disperser_name = self.hp.DISPERSER 
        if self.verbose >= 0 : print(f"Loading disperser {c.ti}{self.disperser_name}{c.d}")
        self.As = [0.0, self.A1, self.A2, self.A3]
        self.disperser = MyDisperser(self.hp, self.As, self.lambdas, self.R0)
        self.tr = [None] + self.giveTr()
        self.order2make = {order:[tr, A] for order, (tr, A) in enumerate(zip(self.tr, self.As)) if tr is not None and A != 0.0}

        # Simulation parameters
        self.with_atmosphere = with_atmosphere
        self.with_adr = with_adr
        self.with_background = with_background
        self.with_flat = with_flat
        self.with_convertADU = with_convertADU
        self.with_noise = self.hp.with_noise
        if self.verbose >= 0 : print(f"With noise : {c.ti}{self.with_noise}{c.d}")

        # Loading telescope transmission
        self.telescope_transmission = self.loading_tel_transmission()

        # Loading target spectrum
        self.target_spectrum = self.loading_target_spectrum(self.hp.target_name)
        
        # Loading flat
        self.flat = self.loading_flat()

        # Loading Atmosphere
        self.load_atm_transmission()

        # Total time for this initialisation
        total_time = time() - time_init
        if self.verbose > 0:
            print(f"{c.y}Initialisation of SpecSimulator : {total_time:.2f} s. {c.d}")



    def set_new_disperser(self, disperser_name):

        self.disperser_name = disperser_name
        self.disperser = MyDisperser(disperser_name, self.As, self.lambdas, self.R0)
        self.refresh_order2make()



    def refresh_order2make(self):

        self.tr = [None] + self.giveTr()
        self.order2make = {order:[tr, A] for order, (tr, A) in enumerate(zip(self.tr, self.As)) if tr is not None and A != 0.0}



    def run(self):

        times = list()
        pbar = tqdm(total=self.nb_simu)
        for i in range(self.nb_simu):
            t0 = time()
            self.ctt.newLoop()
            image, spectrum = self.makeSim(num_simu=i)
            pbar.update(1)
            times.append(time()-t0)
        pbar.close()

        self.hp.save() # self.json_save(self.historic_params, 'hist_params')
        np.savez(f"{self.output_dir}/{self.save_fold}/vparams.npz", **self.variable_params)

        if self.show_times:
            self.ctt.result()
            nb_train = 50000
            time_per_train = nb_train * (np.sum(times) / self.nb_simu) / 60
            if self.verbose > 0: 
                print(f"{c.lm}Result of ctTime with {self.nb_simu} loop : {np.mean(times)*1e3:.1f} ~ {np.std(times)*1e3:.1f} ms{c.d}") 
            if self.verbose > 1:
                print(f"Time for {nb_train} pict. : {time_per_train:.1f} min with {image.shape[0] * image.shape[1] * 8 / 1024**3 * nb_train:.2f} Go")


        # view some specs in divers/
        if self.show_specs:

            # few single image
            nbSingleImage = min(4, self.nb_simu)
            files = os.listdir(f"{self.output_dir}/{self.save_fold}/image")[:nbSingleImage]
            
            for i, file in enumerate(files):

                plt.figure(figsize=(16, 12))
                img = np.load(f"{self.output_dir}/{self.save_fold}/image/{file}")
                plt.imshow(np.log10(img+1), cmap='gray', origin='lower')
                plt.title(self.variable_params['TARGET'][i])
                plt.xlabel(f"Axis 0")
                plt.ylabel(f"Axis 1")
                plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/image_single_{i}.png")
                plt.close()


            if self.nb_simu >= 10:

                # image
                plt.figure(figsize=(16, 12))
                files = os.listdir(f"{self.output_dir}/{self.save_fold}/image")[:10]

                for i, file in enumerate(files):
                    img = np.load(f"{self.output_dir}/{self.save_fold}/image/{file}")

                    plt.subplot(5, 2, i+1)
                    plt.imshow(np.log10(img+1), cmap='gray', origin='lower')
                    plt.title(self.variable_params['TARGET'][i])
                    plt.axis('off')

                plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/images.png")
                plt.close()

                # image RGB
                if self.colorSimu:
                    plt.figure(figsize=(16, 12))
                    files = os.listdir(f"{self.output_dir}/{self.save_fold}/imageRGB")[:10]

                    for i, file in enumerate(files):
                        img = np.load(f"{self.output_dir}/{self.save_fold}/imageRGB/{file}") / len(self.lambdas)
                        img /= np.max(img)

                        plt.subplot(5, 2, i+1)
                        plt.imshow(img, origin='lower')
                        plt.title(self.variable_params['TARGET'][i])
                        plt.axis('off')

                    plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/imagesRGB.png")
                    plt.close()

                # spectro
                if self.spectroData:
                    plt.figure(figsize=(16, 12))
                    files = os.listdir(f"{self.output_dir}/{self.save_fold}/spectro")[:10]

                    for i, file in enumerate(files):
                        img = np.load(f"{self.output_dir}/{self.save_fold}/spectro/{file}")

                        plt.subplot(5, 2, i+1)
                        plt.imshow(img+1, cmap='gray', origin='lower')
                        plt.title(self.variable_params['TARGET'][i])
                        plt.axis('off')

                    plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/spectro.png")
                    plt.close()

                # spectrum
                plt.figure(figsize=(24, 12))
                files = os.listdir(f"{self.output_dir}/{self.save_fold}/spectrum")[:10]
                for i, file in enumerate(files):
                    title = [self.variable_params['TARGET'][i]] + [f"{var}={self.variable_params[var][i]:.2f}" for var in self.variable_params.keys() if var!='TARGET' and var[:4]=="ATM_"]
                    spec = np.load(f"{self.output_dir}/{self.save_fold}/spectrum/{file}")
                    plt.plot(self.lambdas, spec, label=', '.join(title))
                plt.legend()
                plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/spectrum.png")
                plt.close()


                # spectrumPX
                if self.spectroData:
                    plt.figure(figsize=(24, 12))
                    files = os.listdir(f"{self.output_dir}/{self.save_fold}/spectrumPX")[:10]
                    for i, file in enumerate(files):
                        title = [self.variable_params['TARGET'][i]] + [f"{var}={self.variable_params[var][i]:.2f}" for var in self.variable_params.keys() if var!='TARGET' and var[:4]=="ATM_"]
                        spec = np.load(f"{self.output_dir}/{self.save_fold}/spectrumPX/{file}")
                        plt.plot(self.xpixels, spec, label=', '.join(title))
                    plt.legend()
                    plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/spectrumPX.png")
                    plt.close()


    def makeSim(self, num_simu, updateParams=True, giveSpectrum=None, with_noise=True, for_analyse=False):

        ### set variable params
        self.ctt.o(f"set var params", rank="Full")
        if updateParams:
            for param in self.variable_params.keys():

                # set var params
                if param[:4] != "arg.": 
                    self.__setattr__(param, self.variable_params[param][num_simu])

                # set var arg psf
                else:
                    num_arg, num_coef = self.hp.aparams[param]
                    self.psf_function['arg'][num_arg][num_coef] = self.variable_params[param][num_simu]
        self.ctt.c(f"set var params")

        # set timbre
        self.ctt.o(f"arg timbre", rank="Full")
        arg_timbre = [int(np.round(np.max(f_arg(self.lambdas, *arg)))) for f_arg, arg in zip(self.psf_function['f_arg'], self.psf_function['arg'])]
        timbre_size = self.psf_function['timbre'](*arg_timbre)
        self.ctt.c(f"arg timbre")

        # SIMULATE
        self.ctt.o(f"Blank simulate", rank="Full")
        self.ctt.o(f"Init simulate", rank="BlankS")
        spectrogram_data = np.zeros((self.Ny, self.Nx), dtype="float32")
        if self.spectroData: spectro_deconv = np.zeros((self.Ny, self.Nx), dtype="float32")
        spectrogram_data_RGB = np.zeros((self.Ny, self.Nx, 3), dtype="float32") if self.colorSimu else None
        self.ctt.c(f"Init simulate")
        
        self.ctt.o(f"Construction spectrum", rank="BlankS")
        spectrum = self.simulate_spectrum() * self.A if giveSpectrum is None else giveSpectrum / self.hp.CCD_GAIN / self.EXPOSURE
        self.ctt.c(f"Construction spectrum")

        allXc = np.array([])
        allYc = np.array([])

        for order, (tr, A) in self.order2make.items():
            
            # Dispersion law
            self.ctt.o(f"Compute dispersion & params", rank="BlankS")
            if self.with_adr : adr_x, adr_y = self.loading_adr()
            else : adr_x, adr_y = 0.0, 0.0

            Amp = A * tr(self.lambdas) * spectrum
            X_p = self.disperser.dist_along_disp_axis[order]                                             + adr_x + self.R0[0]
            X_c = self.disperser.dist_along_disp_axis[order] * np.cos(self.ROTATION_ANGLE * np.pi / 180) + adr_x + self.R0[0]
            Y_c = self.disperser.dist_along_disp_axis[order] * np.sin(self.ROTATION_ANGLE * np.pi / 180) + adr_y + self.R0[1]
            allXc = np.append(allXc, X_c)
            allYc = np.append(allYc, Y_c)
            self.ctt.c(f"Compute dispersion & params")

            # Building PSF
            self.ctt.o(f"Building PSF cube", rank="BlankS")
            sdo, sdo_RGB = self.build_psf_cube(X_c, Y_c, Amp, timbre_size)
            spectrogram_data += sdo
            if self.colorSimu : spectrogram_data_RGB += sdo_RGB 

            if order == 1:

                func_amp = interp1d(X_p, Amp, kind='linear', bounds_error=False, fill_value=0.)
                yamp = func_amp(self.xpixels)     
                if self.spectroData: spectro_deconv[int(self.R0[1])] = yamp

            self.ctt.c(f"Building PSF cube")
        self.ctt.c(f"Blank simulate")


        # IMAGE RECOMBINAISON
        self.ctt.o(f"Image Computation", rank="Full")
        self.ctt.o(f"orders", rank="imageC")
        psf_order_0 = self.psf_function['f'](self.xx, self.yy, self.psf_function['order0']['amplitude']*self.A0*self.A, *self.R0, *self.psf_function['order0']['arg']).astype(np.float32)
        data_image = spectrogram_data + psf_order_0
        if self.colorSimu : 
            data_image_RGB = spectrogram_data_RGB
            norma = np.max(psf_order_0) if np.max(spectrogram_data_RGB) == 0 else np.max(spectrogram_data_RGB)
            data_image_RGB[:, :, 0] += psf_order_0 / np.max(psf_order_0) * norma
            data_image_RGB[:, :, 1] += psf_order_0 / np.max(psf_order_0) * norma
            data_image_RGB[:, :, 2] += psf_order_0 / np.max(psf_order_0) * norma
        self.ctt.c(f"orders")

        if self.with_background:
            self.ctt.o(f"back", rank="imageC")
            data_image += self.BACKGROUND_LEVEL
            if self.colorSimu : 
                data_image_RGB[:, :, 0] += self.BACKGROUND_LEVEL
                data_image_RGB[:, :, 1] += self.BACKGROUND_LEVEL
                data_image_RGB[:, :, 2] += self.BACKGROUND_LEVEL
            self.ctt.c(f"back")

        if self.with_flat:
            self.ctt.o(f"flat", rank="imageC")
            data_image *= self.flat
            if self.colorSimu : 
                data_image_RGB[:, :, 0] *= self.flat
                data_image_RGB[:, :, 1] *= self.flat
                data_image_RGB[:, :, 2] *= self.flat
            self.ctt.c(f"flat")

        if self.with_convertADU:
            self.ctt.o(f"convertADU", rank="imageC")
            data_image *= self.EXPOSURE
            if self.colorSimu : data_image_RGB *= self.EXPOSURE
            self.ctt.c(f"convertADU")

        if self.with_noise and with_noise:
            self.ctt.o(f"noise", rank="imageC")
            data_image = self.add_poisson_and_read_out_noise(data_image)
            if self.colorSimu : 
                data_image_RGB[:, :, 0] = self.add_poisson_and_read_out_noise(data_image_RGB[:, :, 0])
                data_image_RGB[:, :, 1] = self.add_poisson_and_read_out_noise(data_image_RGB[:, :, 1])
                data_image_RGB[:, :, 2] = self.add_poisson_and_read_out_noise(data_image_RGB[:, :, 2])
            self.ctt.c(f"noise")
        self.ctt.c(f"Image Computation")


        if for_analyse:
            return data_image, allXc, allYc

        self.ctt.o(f"Save npy", rank="Full")

        if self.hp.telescope in ["auxtel", "auxtelqn"]:
            # data_image = data_image.T[::-1, ::-1]
            # for IA models, we need a small (like 128x1024) images.
            if self.savingFolders : np.save(f"{self.output_dir}/{self.save_fold}/imageOrigin/image_{num_simu:0{self.len_simu}}.npy", data_image)
            data_image = data_image[::2, ::2] + data_image[1::2, ::2] + data_image[::2, 1::2] + data_image[1::2, 1::2]
            allXc /= 2
            allYc /= 2

        if self.savingFolders:
        
            np.save(f"{self.output_dir}/{self.save_fold}/image/image_{num_simu:0{self.len_simu}}.npy", data_image)
            if self.colorSimu : np.save(f"{self.output_dir}/{self.save_fold}/imageRGB/imageRGB_{num_simu:0{self.len_simu}}.npy", data_image_RGB)
            spectrum_to_save = (spectrum * self.hp.CCD_GAIN * self.EXPOSURE).astype(np.float32)
            np.save(f"{self.output_dir}/{self.save_fold}/spectrum/spectrum_{num_simu:0{self.len_simu}}.npy", spectrum_to_save.astype(np.float32))
            np.save(f"{self.output_dir}/{self.save_fold}/opa/opa_{num_simu:0{self.len_simu}}.npy", np.array([self.ATM_OZONE, self.ATM_PWV, self.ATM_AEROSOLS]).astype(np.float32))

            # Save spectro data if wanted
            if self.spectroData:
                spectro_deconv_to_save = (spectro_deconv * self.hp.CCD_GAIN * self.EXPOSURE).astype(np.float32)
                np.save(f"{self.output_dir}/{self.save_fold}/spectro/spectro_{num_simu:0{self.len_simu}}.npy", spectro_deconv_to_save.astype(np.float32))
                np.save(f"{self.output_dir}/{self.save_fold}/spectrumPX/spectrumPX_{num_simu:0{self.len_simu}}.npy", np.sum(spectro_deconv_to_save, axis=0).astype(np.float32))
            
        self.ctt.c(f"Save npy")

        if giveSpectrum is None : return data_image, spectrum
        else : return data_image, spectrum, allXc, allYc



    def simulate_spectrum(self):

        self.ctt.o(f"load_atm", rank="sim spec")
        if self.with_atmosphere : self.atm = self.give_atm_transmission()
        self.ctt.c(f"load_atm")

        self.ctt.o(f"multiplier", rank="sim spec")
        spectrum = self.targets_spectrum[self.TARGET](self.lambdas)
        if self.TARGET not in ["calib", "calPX"] :
            spectrum *= self.disperser.transmission(self.lambdas)
            spectrum *= self.telescope_transmission(self.lambdas)
            if self.with_atmosphere : spectrum *= self.atm(self.lambdas)
            spectrum *= self.hp.FLAM_TO_ADURATE * self.lambdas * np.gradient(self.lambdas)
        self.ctt.c(f"multiplier")

        return spectrum



    def build_psf_cube(self, X_c, Y_c, amplitude, timbre_size, dtype="float32"):

        self.ctt.o(f"init cube", rank='bpc')

        argmin = max(0,  int(np.argmin(np.abs(X_c))))
        argmax = min(self.Nx, np.argmin(np.abs(X_c-self.Nx)) + 1)
        psf_cube = np.zeros((self.hp.SIM_NY, self.hp.SIM_NX), dtype=dtype)  
        timbreX = np.zeros((int(timbre_size*2), int(timbre_size*2)), dtype=dtype)
        timbreY = np.zeros((int(timbre_size*2), int(timbre_size*2)), dtype=dtype)
        psf_cube_RGB = np.zeros((self.hp.SIM_NY, self.hp.SIM_NX, 3), dtype=dtype) if self.colorSimu else None
        self.ctt.c(f"init cube")

        for x in range(argmin, argmax):

            self.ctt.o(f"find min/max", rank='bpc')
            xmin = max(0, int(X_c[x]                  - timbre_size))
            xmax = min(self.hp.SIM_NX, int(X_c[x] + timbre_size))
            ymin = max(0, int(Y_c[x]                  - timbre_size))
            ymax = min(self.hp.SIM_NY, int(Y_c[x] + timbre_size))
            self.ctt.c(f"find min/max")

            self.ctt.o(f"Xpix, Ypix", rank='bpc')
            Xpix, Ypix = self.pixels[:, ymin:ymax, xmin:xmax]
            timbreX[:ymax-ymin, :xmax-xmin] = Xpix
            timbreY[:ymax-ymin, :xmax-xmin] = Ypix
            argf = [f_arg(self.lambdas[x], *arg) for f_arg, arg in zip(self.psf_function['f_arg'], self.psf_function['arg'])]
            self.ctt.c(f"Xpix, Ypix")

            self.ctt.o(f"psf_func", rank='bpc')
            psf2add = self.psf_function['f'](timbreX, timbreY, amplitude[x], X_c[x], Y_c[x], *argf)[:ymax-ymin, :xmax-xmin]
            psf_cube[ymin:ymax, xmin:xmax] += psf2add
            if self.colorSimu:
                R, G, B, A = self.wavelength_to_rgb(self.lambdas[x])
                psf_cube_RGB[ymin:ymax, xmin:xmax, 0] += R * psf2add
                psf_cube_RGB[ymin:ymax, xmin:xmax, 1] += G * psf2add
                psf_cube_RGB[ymin:ymax, xmin:xmax, 2] += B * psf2add
            self.ctt.c(f"psf_func")

        return psf_cube, psf_cube_RGB



    ##### 
    #####    Fonction d'initialisation
    #####



    def init_var_params(self):

        # Pour ce souvenir des paramètres utlisé
        self.historic_params = {'nb_simu': self.nb_simu, 'target_set':self.hp.target_name}

        # On calcule les vecteurs pour les paramètres variables	
        self.variable_params = {'TARGET' : np.random.choice(self.hp.target_name, self.nb_simu)}

        # On parcoure toute les parametres de hparameters qui peuvent etre variable
        for param, value in self.hp.vparams.items():

            if self.verbose > 1: print(f"Set var param {c.lm}{param}{c.d} to range {c.lm}{value}{c.d}")
            self.historic_params[param] = value
            self.variable_params[param] = np.random.uniform(*value, self.nb_simu)

        # Les variables renseigné dans le dict d'entrée mais non variables
        for param, value in self.hp.cparams.items():

            if self.verbose > 1: print(f"Set fix param {c.m}{param}{c.d} to {c.m}{value}{c.d}")
            self.historic_params[param] = value
            self.__setattr__(param, value)

        # # Les variables d'args de la psf function, on s'occupe ici des variation des paramètres de moffat
        # for param, value in self.hp.vparams.items():

        #     if self.verbose > 1: print(f"Set args params {c.m}{param}{c.d} to {c.m}{value}{c.d}")
        #     self.historic_params[param] = value
        #     self.variable_params[param] = np.random.uniform(*value, self.nb_simu)



    def loading_tel_transmission(self):
        """
        Méthode pour load la transmission du telescope 
        """

        filename = f"{self.hp.THROUGHPUT_DIR}/{self.hp.THROUGHPUT}"
        data = np.loadtxt(filename).T
        lambdas = data[0]
        sorted_indices = lambdas.argsort()
        lambdas = lambdas[sorted_indices]
        y = data[1][sorted_indices]
        indexes = np.logical_and(lambdas > np.min(self.lambdas), lambdas < np.max(self.lambdas))

        to = interp1d(lambdas[indexes], y[indexes], kind='linear', bounds_error=False, fill_value=0.)
        TF = lambda x: 1
        transmission = lambda x: to(x) * TF(x)

        return transmission



    def loading_target_spectrum(self, targets):
        """
        Méthode pour load les spectre des target dans la liste targets
        """

        self.targets_spectrum = dict()

        if self.verbose > 1 : 
            sys.stdout.write(f"Loading targets spectrum : ")
            sys.stdout.flush()

        for target in targets:

            if getCalspec.is_calspec(target):

                calspec = getCalspec.Calspec(target)
                spec_dict = calspec.get_spectrum_numpy()
                spec_dict["WAVELENGTH"] = spec_dict["WAVELENGTH"].to(u.nm)

                for key in ["FLUX", "STATERROR", "SYSERROR"]:
                    spec_dict[key] = spec_dict[key].to(u.erg / u.second / u.cm**2 / u.nm)

                wavelengths = spec_dict["WAVELENGTH"].value
                spectra = spec_dict["FLUX"].value
                sed = interp1d(wavelengths, spectra, kind='linear', bounds_error=False, fill_value=0.)

            elif target == "calib":

                sed = self.make_calib_spectrum

            elif target == "calPX":

                sed = self.make_calPX_spectrum

            else:

                print(f"{c.r}WARNING : label {target} for loading spectrum is not avaible ...{c.d}")
                wavelengths, spectra = None, None
                sed = interp1d(wavelengths, spectra, kind='linear', bounds_error=False, fill_value=0.)

            self.targets_spectrum[target] = sed
            if self.verbose > 1 :
                sys.stdout.write(f"{c.g}{target}{c.d}, ")
                sys.stdout.flush()

        if self.verbose > 1:
            print(f" ... ok")



    def make_calib_spectrum(self, x, npeak=[1, 1], amp=[1e4, 1e5], sig=[10, 50]):

        num_peak = np.random.randint(npeak[0], npeak[1]+1)
        amps = np.random.uniform(*amp, num_peak)
        sigs = np.random.uniform(*sig, num_peak)
        lbds = np.random.uniform(x[0], x[-1], num_peak)

        spectrum = np.zeros_like(x).astype(float)
        for x0, a, s in zip(lbds, amps, sigs):
            spectrum += stats.norm.pdf(x, loc=x0, scale=s) * a

        return spectrum



    def make_calPX_spectrum(self, x, npeak=[1, 2], amp=[1e3, 1e4]):

        num_peak = np.random.randint(npeak[0], npeak[1]+1)
        amps = np.random.uniform(*amp, num_peak)
        # sigs = np.random.uniform(*sig, num_peak)
        lbds = np.random.randint(0, len(x), num_peak)

        spectrum = np.zeros_like(x).astype(float)
        for x0, a in zip(lbds, amps):
            spectrum[x0] = a

        return spectrum


    def giveTr(self, order=1):

        # load the disperser relative transmissions
        tr_ratio = interp1d(self.lambdas, np.ones_like(self.lambdas), bounds_error=False, fill_value=1.)
        if abs(order) == 1:
            tr_ratio_next_order = self.disperser.ratio_order_2over1
            tr_ratio_next_next_order = self.disperser.ratio_order_3over1
        elif abs(order) == 2:
            tr_ratio_next_order = self.disperser.ratio_order_3over2
            tr_ratio_next_next_order = None
        elif abs(order) == 3:
            tr_ratio_next_order = None
            tr_ratio_next_next_order = None
        else:
            raise ValueError(f"{abs(self.order)=}: must be 1, 2 or 3. "
                             f"Higher diffraction orders not implemented yet in full forward model.")
        
        return [tr_ratio, tr_ratio_next_order, tr_ratio_next_next_order]



    def loading_flat(self, list_gains=[[1]], randomness_level=1e-2, dtype="float32"):
        """
        Méthode pour faire le flat
        """

        gains = np.atleast_2d(list_gains).astype(float)
        if np.mean(gains) != 1. : true_gains /= np.mean(gains)

        flat = np.ones((self.Ny, self.Nx), dtype=dtype)

        hflats = np.array_split(flat, gains.shape[0])

        for h in range(gains.shape[0]):
            vflats = np.array_split(hflats[h].T, gains.shape[1])
            for v in range(gains.shape[1]):
                vflats[v] *= gains[h, v]
            hflats[h] = np.concatenate(vflats).T

        flat = np.concatenate(hflats)
        if randomness_level != 0:
            flat += np.random.uniform(-randomness_level, randomness_level, size=flat.shape)

        return flat



    #####
    #####    Function *utils*
    #####



    def loading_adr(self, dispersion_axis_angle=0, lambdas=None):
        """
        Méthode pour load l'ADR
        """

        if self.with_adr:

            if lambdas is None : lambdas = self.lambdas

            ADR_PARAMS = [self.ADR_DEC, self.ADR_HOUR_ANGLE, self.ATM_TEMPERATURE, self.hp.OBS_PRESSURE, self.ATM_HUMIDITY, self.ATM_AIRMASS]
            adr_ra, adr_dec = adr_calib(self.hp, lambdas, ADR_PARAMS, self.hp.OBS_LATITUDE, lambda_ref=self.lambda_adr_ref)

            # flip_and_rotate_radec_vector_to_xy_vector of 
            flip = np.array([[self.hp.OBS_CAMERA_RA_FLIP_SIGN, 0], [0, self.hp.OBS_CAMERA_DEC_FLIP_SIGN]], dtype=float)
            a = - self.hp.OBS_CAMERA_ROTATION * np.pi / 180 # minus sign as rotation matrix is apply on the right on the adr vector
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=float)

            transformation = flip @ rotation
            adr_x, adr_y = (np.asarray([adr_ra, adr_dec]).T @ transformation).T

            if "debug" in sys.argv:
                print("flip, a, rotation, transformation")
                print(flip, "\n", a, "\n", rotation, "\n", transformation)
                print(f"Here ADR XY with order with size : {np.shape(adr_x)}")
                plt.plot(adr_ra, c="r", label="adrx")
                plt.plot(adr_dec, c="g", label="adry")
                plt.show()


            # flip_and_rotate_adr_to_image_xy_coordinates
            if not np.isclose(dispersion_axis_angle, 0, atol=0.001):
                # minus sign as rotation matrix is apply on the right on the adr vector
                a = - dispersion_axis_angle * np.pi / 180
                rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=float)
                adr_x, adr_y = (np.asarray([adr_x, adr_y]).T @ rotation).T
            # self.adr_x, self.adr_y = flip_and_rotate_adr_to_image_xy_coordinates(self.adr_ra, self.adr_dec, dispersion_axis_angle=0)
        else:
            adr_x = np.zeros_like(self.disperser.dist_along_disp_axis)
            adr_y = np.zeros_like(self.disperser.dist_along_disp_axis)
        
        return adr_x, adr_y


    def npy_save(self, dico, savefile):

        np.save(f"{self.output_dir}/{self.save_fold}/{savefile}.npy", dico)



    def npy_load(self, dico, savefile):

        return np.load(f"{self.output_dir}/{self.save_fold}/{savefile}.npy")



    def json_save(self, dico, savefile):

        with open(f"{self.output_dir}/{self.save_fold}/{savefile}.json", 'w') as f:
            json.dump(dico, f, indent=4)



    def json_load(self, savefile):

        with open(f"{self.output_dir}/{self.save_fold}/{savefile}.json", 'r') as f:
            dico = json.load(f)

        return dico



    def add_poisson_and_read_out_noise(self, data):  # pragma: no cover

        d = np.copy(data).astype(np.float32)
        # convert to electron counts
        d *= self.hp.CCD_GAIN

        # Poisson noise
        dmin = np.min(d)
        if dmin < 0 : d += np.abs(dmin) * 1.1
        noisy = np.random.poisson(d).astype(np.float32)
        # Add read-out noise is available
        if self.hp.cparams["CCD_READ_OUT_NOISE"] is not None:
            noisy += np.random.normal(scale=self.hp.cparams["CCD_READ_OUT_NOISE"]*np.ones_like(noisy)).astype(np.float32)
        # reconvert to ADU
        data = noisy / self.hp.CCD_GAIN
        # removes zeros
        min_noz = np.min(data[data > 0])
        data[data <= 0] = min_noz
        return data



    def load_atm_transmission(self):

        if self.hp.SPECTRACTOR_ATMOSPHERE_SIM.lower() == "getobsatmo":

            import getObsAtmo

            if not getObsAtmo.is_obssite(self.hp.OBS_NAME):
                raise ValueError(f"getObsAtmo does not have observatory site {self.hp.OBS_NAME}.")

            self.ctt.o(f"emulator", rank='load atm')
            self.emulator = getObsAtmo.ObsAtmo(obs_str=self.hp.OBS_NAME, pressure=self.hp.OBS_PRESSURE)
            self.emulator.lambda0 = 500.
            self.ctt.c("emulator")

            self.ctt.o(f"get all", rank='load atm')
            return self.give_atm_transmission
            self.ctt.c(f"get all")

        else:

            raise ValueError(f"Unknown value for {self.hp.SPECTRACTOR_ATMOSPHERE_SIM=}.")


        return transmission



    def give_atm_transmission(self):

        atm = self.emulator.GetAllTransparencies(self.lambdas, am=self.ATM_AIRMASS, pwv=self.ATM_PWV, oz=self.ATM_OZONE, tau=self.ATM_AEROSOLS, beta=self.ATM_ANGSTROM_EXPONENT, flagAerosols=True)
        return interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))



    def wavelength_to_rgb(self, wavelength, gamma=0.8):
        """ taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
        This converts a given wavelength of light to an
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).

        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
        Additionally alpha value set to 0.5 outside range
        """
        wavelength = float(wavelength)
        if 380 <= wavelength <= 750:
            A = 1.
        else:
            A = 1.0
        if wavelength < 380:
            wavelength = 380.
        if wavelength > 750:
            wavelength = 750.
        if 380 <= wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif 440 <= wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif 490 <= wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif 510 <= wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif 580 <= wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif 645 <= wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        return R, G, B, A



    #####
    #####    Function for change params
    #####



    def change_adr(self):

        self.with_adr = not self.with_adr



    def change_back(self):

        self.with_background = not self.with_background



    def change_noise(self):

        self.with_noise = not self.with_noise



    def change_RGB(self):

        self.colorSimu = not self.colorSimu



    def change_A1(self):

        self.As[1] = 1 - self.As[1]
        self.refresh_order2make()



    def change_A2(self):

        self.As[2] = 1 - self.As[2]
        self.refresh_order2make()



    def change_A3(self):

        self.As[3] = 1 - self.As[3]
        self.refresh_order2make()