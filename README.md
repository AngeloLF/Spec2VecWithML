# Spec2VecWithML

Machine Learning Project, mixing old repositories in one projects: 
  * `specSimulator` : capable to create simulate pairs of Spectrum / Spectrogram for multiple observatory and multiple parameters
    * *Old repository* : https://github.com/AngeloLF/SpecSimulator
  * `models` : containt some ML models (CNNs, ViTs, AE ...) to predict a spectrum from a spectrogram
    * *Old repository* : https://github.com/AngeloLF/Spec2vecModels
  * `applies` : for apply ML models (and Spectractor)
    * *Old repository* : https://github.com/AngeloLF/Spec2vecAnalyse
  * `analyses` : for analyse ML models (and Spectractor)
    * *Old repository* : https://github.com/AngeloLF/Spec2vecAnalyse
  * `extractAtmos` : for extract atmosphere parameters from predicted spectrum by ML models (and Spectractor)
    * *Old repository* : https://github.com/AngeloLF/ExtractAtmos
  * `jobsAndBots` : for create batch with ccin2p3 and create bot (Telegram and Discord)
    * *Old repository* : https://github.com/AngeloLF/TeleBot_Perso

This repo containt also a folder **Spectractor**. It's a modified version of original algo here :
 * https://github.com/LSSTDESC/Spectractor




## Using Making jobs at CCIN2P3

Full pipeline, for create and train few models and compare to Spectractor.
I make an exemple to train 4 models with the architecture SCaM on AuxTel simulations with 2 loss function ("chi2" and "MSE") and 2 learning rate (1e-4 and 1e-5)


### Simulation of datasets :

For create *train*, *valid* and some *test*.
Need arguments:
  * `nsimu` : the size of each dataset
  * `type` : the type of each dataset, like :
    * *train* : train dataset 
    * *valid* : valid dataset
    * *test* : test dataset
    * *testot* : test "other targets", with unknow target in train
    * *testext* : test "extension", with a wide range of parameters
    * *testgauss* : test with gaussian 2D PSF
    * *testna* : test with 2 moffats PSF, with one not aligned
    * *testgaussna* : test with 2 gaussians 2D PSF, with one not aligned
  * `tel` : telescope. Can be give individualy or once for all. Tel available :
    * *ctio* : Cerro Tololo Inter-American Observatory, Chili
    * *stardice* : A 40 cm telescope in OHP, France
    * *auxtel* : The auxialiary telescope of Rubin Observatoire, Chili
  * `seed` : seed to pick parameters. Can be give individualy or once for all.

```bash
python jobAndBots/making_batch.py simu nsimu=16384,2048,1024,1024,1024 type=train,valid,test,testext,testot seed=413 tel=auxtel
```


### Training models

For training models :
  * *model* : list of model architecture (like "SCaM", "SotSu")
  * *loss* : loss funcron (like "chi2", "MSE")
  * *train* : name of trains, but only the number of simulations (like "16k" for "train16kauxtel")
  * *lr* : list of learning rates (like "1e-4", "1e-5")
  * *tel* : list of telescope (like "auxtel", "ctio", "stardice")

```bash
python jobAndBots/making_batch.py training model=SCaM loss=chi2,MSE train=16k lr=1e-4,1e-5 tel=auxtel e=500
```


### Apply models

For apply a model, and also apply Spectractor :
  * *model* : list of model architecture
  * *loss* : loss funcron
  * *train* : name of trains, but only the number of simulations
  * *lr* : list of learning rates
  * *tel* : list of telescope
  * *test* : list of test to apply, like :
    * *x* : for classic test
    * *ext* : for testEXT
    * *ot* : for testOT
    * *gaussian* for testGAUSSIAN
    * *gaussianna* : for testGAUSSIANNA
    * *stardice* for testSTARDICE

For apply Spectractor, *test* and *tel* needed. *ncpu* can also be given, to cut the apply.

```bash
python jobAndBots/making_batch.py apply model=SCaM loss=chi2,MSE train=16k lr=1e-4,1e-5 tel=auxtel test=x,ext,ot
python jobAndBots/making_batch.py apply_spectractor test=x tel=auxtel ncpu=100
python jobAndBots/making_batch.py apply_spectractor test=ext tel=auxtel ncpu=100
python jobAndBots/making_batch.py apply_spectractor test=ot tel=auxtel ncpu=100
```


### Analyse spectrum from apply

For analyse results :
  * *model* : list of model architecture
  * *loss* : loss funcron
  * *train* : name of trains, but only the number of simulations
  * *lr* : list of learning rates
  * *tel* : list of telescope
  * *test* : list of test to apply
  * *score* : list of score to calculate (L1 and chi2)


```bash
python jobAndBots/making_batch.py analyse model=SCaM loss=chi2,MSE lr=1e-4,1e-5,5e-5 train=16k test=x,ext,ot tel=auxtel score=L1,chi2
python jobAndBots/making_batch.py analyse model=Spectractor loss=x lr=0e+0 train=x test=x,ext,ot tel=auxtel score=L1,chi2
```













