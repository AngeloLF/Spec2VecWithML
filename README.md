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

Full pipeline, for create and train few models and compare to Spectractor :

### Simulation of datasets :

For create *train*, *valid* and some *test*.
Need arguments:
  * `nsimu` : the size of each dataset
  * `type` : the type of each dataset, like:
    * *train* : train dataset 
    * *valid* : valid dataset
    * *test* : test dataset
    * *testot* : test "other targets", with unknow target in train
    * *testext* : test "extension", with a wide range of parameters
    * *testgauss* : test with gaussian 2D PSF
    * *testna* : test with 2 moffats PSF, with one not aligned
    * *testgaussna* : test with 2 gaussians 2D PSF, with one not aligned
  * 

```bash
python jobAndBots/making_batch.py simu nsimu=16384,2048,1024,1024,1024 type=train,valid,test,testext,testot seed=413 tel=auxtel
```