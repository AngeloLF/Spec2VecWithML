import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm




class FOPASS_Model(nn.Module):
    """
    FOPA : Finding Ozone, PWV and Aerosols in Slitless Spectrophotography 
    """

    folder_input = "image"
    folder_output = "opa"

    def __init__(self, dropout=0.1):

        super(FOPASS_Model, self).__init__()

        # Image d'entrée : 1x128x1024

        def conv_block(in_ch, out_ch, pool=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(5, 7), padding=(2, 3), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=(5, 7), padding=(2, 3), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            conv_block(1, 16),     # -> 16 x 64 x 512
            conv_block(16, 32),    # -> 32 x 32 x 256
            conv_block(32, 64),    # -> 64 x 16 x 128
            conv_block(64, 128),   # -> 128 x 8 x 64
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 64))   # -> 128 x 1 x 64
        self.max_pool = nn.AdaptiveMaxPool2d((1, 64))   # -> 128 x 1 x 64

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 128 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x : (B, 1, 128, 1024)
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = self.regressor(x)
        return x





# Classe pour le Dataset personnalisé
class FOPASS_Dataset(Dataset):
    def __init__(self, image_dir, spectrum_dir):
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".npy")])
        self.spectrum_files = sorted([os.path.join(spectrum_dir, f) for f in os.listdir(spectrum_dir) if f.endswith(".npy")])

        # prof si le nombre est différent ....
        assert len(self.image_files) == len(self.spectrum_files), "Le nombre de fichiers images / spectrums ne correspond pas"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]).astype(np.float32)
        spectrum = np.load(self.spectrum_files[idx]).astype(np.float32)
        # add une dimension de canal pour le CNN (1 canal car image en niveaux de gris implicite)
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image), torch.tensor(spectrum)