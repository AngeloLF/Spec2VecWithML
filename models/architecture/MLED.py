import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import coloralf as c
import sys




### ENCODER 
class MLED_Encoder(nn.Module):
    def __init__(self, input_shape=(1, 128, 1024), latent_dim=512):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 7), stride=(2, 2), padding=(1, 3))
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 7), stride=(2, 2), padding=(1, 3))
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2))
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculer la taille apres convolutions genre (128, 1024) -> (64, 512) -> (32, 256) -> (16, 128) -> (8, 64)
        self.feature_size = 256 * 8 * 64
        
        # Bottleneck
        self.fc_encode = nn.Linear(self.feature_size, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        latent = self.fc_encode(x)
        
        return latent




### DECODER 
class MLED_Decoder(nn.Module):
    def __init__(self, latent_dim=512, output_shape=(1, 128, 1024)):
        super().__init__()
        
        self.feature_size = 256 * 8 * 64
        self.fc_decode = nn.Linear(latent_dim, self.feature_size)
        
        # Transposed convolutions (miroir de l'encoder)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), output_padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), output_padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 7), stride=(2, 2), padding=(1, 3), output_padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=(3, 7), stride=(2, 2), padding=(1, 3), output_padding=(1, 1))
        
    def forward(self, latent):
        x = self.fc_decode(latent)
        x = x.view(x.size(0), 256, 8, 64)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)
        
        return x




### Partie pour remplacer le decoder par une sortie spectrale (a ajouter avec un encoder deja entrainer)
# Pour l'entrainement, on peut freeze la partie encoder si besoin pour les ~20 premières epochs)
class MLED_Regspectrum(nn.Module):
    def __init__(self, latent_dim=512, spectrum_length=800):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(1024, spectrum_length)
        
    def forward(self, latent):
        x = F.relu(self.bn1(self.fc1(latent)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        spectrum = self.fc3(x)
        return spectrum




### ENCODER-DECODER:
class MLED_EncoderDecoder(nn.Module):
    def __init__(self, input_shape=(1, 128, 1024), latent_dim=512):
        super().__init__()
        self.encoder = MLED_Encoder(input_shape, latent_dim)
        self.decoder = MLED_Decoder(latent_dim, input_shape)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent




# Classe pour le Dataset personnalisé
class MLED_Dataset(Dataset):
    def __init__(self, image_dir, spectrum_dir):
        self.image_dir = Path(image_dir)
        self.spectrum_dir = Path(spectrum_dir)
        
        self.image_files = sorted(list(self.image_dir.glob("image_*.npy")))
        self.num_samples = len(self.image_files)
        
        self.indices = []
        for f in self.image_files:
            # Extraire le nombre du nom de fichier (ex: "image_001.npy" -> "001")
            name = f.stem  # genre "image_001"
            idx_str = name.split('_')[1]  # puis "001"
            self.indices.append(idx_str)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        idx_str = self.indices[idx]
        
        img = np.load(self.image_dir / f"image_{idx_str}.npy")
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        spec = np.load(self.spectrum_dir / f"spectrum_{idx_str}.npy")
        spec = torch.from_numpy(spec).float()
        return img, spec




### ENCODER-SPECTRUM: -> the final model
class MLED_Model(nn.Module):

    folder_input = "image"
    folder_output = "spectrum"

    def __init__(self, encoder=None, latent_dim=512, spectrum_length=800):
        super().__init__()
        if encoder is None:
            self.encoder = MLED_Encoder(latent_dim=latent_dim)
        else:
            self.encoder = encoder
        self.regressor = MLED_Regspectrum(latent_dim=latent_dim, spectrum_length=spectrum_length)
        
    def forward(self, x):
        latent = self.encoder(x)
        spectrum = self.regressor(latent)
        return spectrum


    # particular training function
    def particular_training(self, Args, device, train_loader, valid_loader, loss_function="MSE"):

        # some variables ...
        latent_dim = 512
        spectrum_length = 800
        save_EncodeDecoder = f"{Args.output.state}/{Args.from_prefixe}{Args.train}_{Args.lr_str}_EncoderDecoder_best.pth"
        best_val_loss_AE = np.inf
        best_state_AE = None
        lrates = np.zeros(Args.epochs)

        # Train loss
        train_list_loss_AE = np.zeros(Args.epochs)
        valid_list_loss_AE = np.zeros(Args.epochs)

        # MSE loss
        mse_loss = nn.MSELoss()
        train_list_loss_mse_AE = np.zeros(Args.epochs)
        valid_list_loss_mse_AE = np.zeros(Args.epochs)




        ### TRAIN partie encoder-decoder
        print(f"{c.lm}\nINFO : Beginning of training encoder (1/2){c.d}")
        autoencoder = MLED_EncoderDecoder(latent_dim=latent_dim)
        autoencoder = autoencoder.to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=Args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        

        for epoch in range(Args.epochs):


            ### Training
            autoencoder.train()
            train_loss = 0.0
            train_loss_mse = 0.0

            for images, spectra in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Args.epochs} (Train)"):
                
                images = images.to(device)
                
                # Forward
                reconstruction, latent = autoencoder(images)
                loss = F.mse_loss(reconstruction, images)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Evaluate with mse
                autoencoder.eval()
                train_loss_mse += mse_loss(reconstruction, images)
            
            train_loss = train_loss / len(train_loader)
            train_list_loss_AE[epoch] = train_loss
            train_list_loss_mse_AE[epoch] = train_loss_mse / len(train_loader)
            scheduler.step(train_loss)
        

            ### Validation
            autoencoder.eval()
            valid_loss = 0.0
            valid_loss_mse = 0.0
            
            with torch.no_grad():

                for images, spectra in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{Args.epochs} (Validation)"):

                    images = images.to(device)
                    reconstruction, latent = autoencoder(images)

                    loss = F.mse_loss(reconstruction, images)
                    valid_loss += loss.item()

                    # Evaluate with mse
                    autoencoder.eval()
                    valid_loss_mse += mse_loss(reconstruction, images)

            valid_loss = valid_loss / len(valid_loader)

            # Show epoch
            lrates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']
            print(f"Epoch [{epoch+1}/{Args.epochs}], loss train = {c.g}{train_loss:.6f}{c.d}, val loss = {c.r}{valid_loss:.6f}{c.d} | LR={c.y}{lrates[epoch]:.2e}{c.d}")
            with open(f"{Args.output.epoch_here}/INFO - epoch {epoch+1} - {Args.epochs} - {train_loss:.6f} , {valid_loss:.6f}", "wb") as f : pass

            # save state at each epoch to be able to reload and continue the optimization
            if valid_loss < best_val_loss_AE:

                best_val_loss_AE = valid_loss
                best_state_AE = {"epoch": epoch + 1, "model_state_dict": autoencoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val_loss_AE}

            # save loss
            valid_list_loss_AE[epoch] = valid_loss
            valid_list_loss_mse_AE[epoch] = valid_loss_mse / len(valid_loader)

        if Args.save:
            torch.save(best_state_AE, save_EncodeDecoder)
        print(f"{c.lm}INFO : Save of the AutoEncoder MLED at {save_EncodeDecoder}{c.d}")











        ### TRAIN partie encoder-spectrum
        print(f"{c.y}Beginning to train encoder-spectrum (2/2){c.d}")
        best_val_loss_AE = np.inf
        best_state_AE = None
        lrates = np.zeros(Args.epochs)

        # Train loss
        train_list_loss = np.zeros(Args.epochs)
        valid_list_loss = np.zeros(Args.epochs)

        # MSE loss
        mse_loss = nn.MSELoss()
        train_list_loss_mse = np.zeros(Args.epochs)
        valid_list_loss_mse = np.zeros(Args.epochs)

        # model, opti & scheduler
        spectrum_model = MLED_Model(autoencoder.encoder, spectrum_length=spectrum_length)
        spectrum_model = spectrum_model.to(device)
        optimizer = torch.optim.Adam(spectrum_model.parameters(), lr=Args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # On peut freeze l'encoder au début pour forcer à changer d'abord les poids de la partie spectrum
        # for param in model.encoder.parameters():
        #     param.requires_grad = False


        for epoch in range(Args.epochs):

            # Après quelques epochs, on peut dégeler l'encoder pour du fine-tuning
            # if epoch == 20:
            #     for param in spectrum_model.encoder.parameters():
            #         param.requires_grad = True


            ### Training
            spectrum_model.train()
            train_loss = 0.0
            train_loss_mse = 0.0

            for images, spectra in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Args.epochs} (Train)"):
                
                images = images.to(device)
                spectra = spectra.to(device)
                
                # Forward
                pred_spectra = spectrum_model(images)
                loss = F.mse_loss(pred_spectra, spectra)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Evaluate with mse
                spectrum_model.eval()
                train_loss_mse += mse_loss(pred_spectra, spectra)
            
            train_loss = train_loss / len(train_loader)
            train_list_loss[epoch] = train_loss
            train_list_loss_mse[epoch] = train_loss_mse / len(train_loader)
            scheduler.step(train_loss)
        

            ### Validation
            spectrum_model.eval()
            valid_loss = 0.0
            valid_loss_mse = 0.0
            
            with torch.no_grad():

                for images, spectra in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{Args.epochs} (Validation)"):

                    images = images.to(device)
                    spectra = spectra.to(device)
                    pred_spectra = spectrum_model(images)
                    loss = F.mse_loss(pred_spectra, spectra)
                    valid_loss += loss.item()

                    # Evaluate with mse
                    spectrum_model.eval()
                    valid_loss_mse += mse_loss(pred_spectra, spectra)

            valid_loss = valid_loss / len(valid_loader)

            # Show epoch
            lrates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']
            print(f"Epoch [{epoch+1}/{Args.epochs}], loss train = {c.g}{train_loss:.6f}{c.d}, val loss = {c.r}{valid_loss:.6f}{c.d} | LR={c.y}{lrates[epoch]:.2e}{c.d}")
            with open(f"{Args.output.epoch_here}/INFO - epoch {epoch+1} - {Args.epochs} - {train_loss:.6f} , {valid_loss:.6f}", "wb") as f : pass

            # save state at each epoch to be able to reload and continue the optimization
            if valid_loss < best_val_loss_AE:

                best_val_loss = valid_loss
                best_state = {"epoch": epoch + 1, "model_state_dict": autoencoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val_loss}

            # save loss
            valid_list_loss[epoch] = valid_loss
            valid_list_loss_mse[epoch] = valid_loss_mse / len(valid_loader)
        print(f"{c.g}End of final MLED train{c.d}")

        # dict of statistiques
        run_stats = {
            "train" : train_list_loss, "train_mse" : train_list_loss_mse,
            "valid" : valid_list_loss, "valid_mse" : valid_list_loss_mse,
            "lrates" : lrates
        }

        return best_state, run_stats






### partie train ENCODER-SPECTRUM (partie 2)
def train_spectrum_regressor(model, train_loader, num_epochs=100, lr=1e-4, device='cuda'):
    model = model.to(device)
    
    # On freeze l'encoder au début (optionnel)
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Après quelques epochs, on peut dégeler l'encoder pour du fine-tuning
        # if epoch == 20:
        #     for param in model.encoder.parameters():
        #         param.requires_grad = True
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, spectra in pbar:
            images = images.to(device)
            spectra = spectra.to(device)
            
            # Forward
            pred_spectra = model(images)
            loss = F.mse_loss(pred_spectra, spectra)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        history[epoch] = avg_loss
        scheduler.step(avg_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
    
    return history






















### script test
if __name__ == "__main__":
    
    # Configuration
    if "train" in sys.argv:
        path_folder = "./results/output_simu/train16auxtel"
    elif "test" in sys.argv:
        path_folder = "./results/output_simu/test10auxtel"
    else:
        raise Exception(f"Need train or test in sys.argv")

    image_dir = f"{path_folder}/image"
    spectrum_dir = f"{path_folder}/spectrum"
    batch_size = 16
    latent_dim = 512
    spectrum_length = 800

    nepochs1 = 5
    nepochs2 = 5

    lr1 = 1e-3
    lr2 = 1e-4

    save_autoencoder = "SpecMLED/autoencoder.pth"
    save_spectrum = "SpecMLED/spectrum.pth"


    # TRAIN 
    if "train" in sys.argv:

        MLED_train()

    # TEST
    if "test" in sys.argv:

        # device
        device = get_device("cpu")
        print(f"Using device: {device}")

        # Dataset et DataLoader
        dataset = SpectrogramDataset(image_dir, spectrum_dir)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Dataset size: {len(dataset)}")

        # load auto encoder
        state_encoder = torch.load(save_autoencoder, map_location=device)
        autoencoder = SpectrogramAutoEncoder(latent_dim=latent_dim)
        autoencoder.load_state_dict(state_encoder)
        autoencoder.to(device)
        print(f"{c.ly}Loading auto encoder ok{c.d}")    

        # load spec encoder
        state = torch.load(save_spectrum, map_location=device)
        encoder = SpectrogramEncoder(latent_dim=latent_dim)
        spectrum_model = SpectrogramToSpectrum(encoder, spectrum_length=spectrum_length)
        spectrum_model.load_state_dict(state)
        spectrum_model.to(device)
        print(f"{c.ly}Loading spec encoder ok{c.d}")    

        spectrum_model.eval()
        with torch.no_grad():
            # Prendre un exemple
            img, true_spec = dataset[0]
            img = img.unsqueeze(0).to(device)
            
            pred_spec = spectrum_model(img).cpu().numpy()[0]
            true_spec = true_spec.numpy()

            pred_img = autoencoder(img)[0][0][0]
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            # Spectrogramme
            plt.subplot(311)
            plt.imshow(dataset[0][0].squeeze(), aspect='auto', cmap='viridis')
            plt.title("Spectrogram input")

            # Spectrogramme autoenc
            plt.subplot(312)
            plt.imshow(pred_img, aspect='auto', cmap='viridis')
            plt.title("Spectrogram autoencoder")
            
            # Comparaison spectres
            plt.subplot(313)
            wavelengths = np.linspace(300, 1100, spectrum_length)
            plt.plot(wavelengths, true_spec, label="True spectrum", c="g")
            plt.plot(wavelengths, pred_spec, label='Pred spectrum', c="r")
            plt.title("MLED model ")
            plt.xlabel(r"$\lambda$ (nm)")
            plt.ylabel(r"Intensité ($e^{-})$")
            plt.legend()
            
            # plt.tight_layout()
            # plt.savefig("results.png", dpi=150)
            # print("Résultats sauvegardés dans 'results.png'")
            plt.show()













