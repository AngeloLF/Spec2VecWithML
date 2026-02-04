import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os



class PositionalEncoding(nn.Module):
    """
    Implémentation de l'encodage positionnel sinusoïdal.
    Ajoute des informations de position aux embeddings d'entrée du Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # Créer une matrice de zéros pour stocker les encodages positionnels
        pe = torch.zeros(max_len, d_model)
        # Calculer les positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculer le terme de division pour les fréquences sinusoïdales
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # Appliquer les fonctions sinusoïdales et cosinusoïdales
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Ajouter une dimension de lot (batch dimension) et enregistrer comme un buffer
        # Un buffer n'est pas un paramètre du modèle et n'est pas mis à jour par l'optimiseur
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique l'encodage positionnel à l'entrée.
        Args:
            x (torch.Tensor): Le tenseur d'entrée, de forme (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Le tenseur avec l'encodage positionnel ajouté.
        """
        # Ajouter l'encodage positionnel à l'entrée.
        # On prend seulement la partie de 'pe' qui correspond à la longueur de la séquence actuelle.
        x = x + self.pe[:, :x.size(1)]
        return x

class CaTS_Model(nn.Module):
    """
    Modèle SCaM : CNN simple suivi d'un Transformer.
    La partie MLP est remplacée par un Transformer Encoder.
    """

    folder_input = "image"
    folder_output = "spectrum"
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 4, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialise le modèle SCaM avec CNN et Transformer.
        Args:
            d_model (int): Dimension des embeddings d'entrée du Transformer.
            nhead (int): Nombre de têtes d'attention dans le Transformer.
            num_encoder_layers (int): Nombre de couches d'encodeur Transformer.
            dim_feedforward (int): Dimension de la couche feedforward dans le Transformer.
            dropout (float): Taux de dropout pour le Transformer.
        """
        super(CaTS_Model, self).__init__()

        # Partie CNN (inchangée par rapport à votre modèle original)
        # Image d'entrée : 1x128x1024
        
        # 1ère convolution CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))  # -> 16x128x1024
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))      # -> 16x64x512

        # 2ème convolution CNN
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)) # -> 32x64x512
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))      # -> 32x32x256

        # 3ème convolution CNN
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)) # -> 64x32x256
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))      # -> 64x16x128

        # Partie Transformer
        # La sortie du CNN est de forme (batch_size, 64, 16, 128).
        # Pour le Transformer, nous devons la transformer en une séquence de tokens.
        # Nous allons aplatir les dimensions spatiales (16x128) pour chaque canal (64 canaux),
        # puis projeter chaque "token" (qui est maintenant de 16*128 = 2048 dimensions)
        # à la dimension d_model attendue par le Transformer.
        # Cela crée une séquence de 64 tokens, chacun de dimension d_model.
        
        # Couche linéaire pour projeter les features aplaties (16*128) à d_model
        self.proj_to_d_model = nn.Linear(16 * 128, d_model) 
        
        # Encodage positionnel pour ajouter des informations de position aux tokens
        # La longueur maximale de la séquence est de 64 (nombre de canaux après CNN).
        self.positional_encoding = PositionalEncoding(d_model, max_len=64) 

        # Définition d'une seule couche d'encodeur Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, 
                                                   batch_first=True) # batch_first=True signifie (batch, seq, feature)
        
        # Empilement de plusieurs couches d'encodeur Transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Couche de sortie finale (remplace l'ancien MLP final)
        # La sortie du Transformer est de forme (batch_size, 64, d_model).
        # Nous l'aplatissons pour la passer à une couche linéaire qui produit les 800 features de sortie.
        self.fc_out = nn.Linear(64 * d_model, 800) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle.
        Args:
            x (torch.Tensor): Le tenseur d'entrée de l'image, de forme (batch_size, 1, 128, 1024).
        Returns:
            torch.Tensor: Le tenseur de sortie, de forme (batch_size, 800).
        """
        # Passe avant du CNN
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x))) # Forme de sortie après CNN : (B, 64, 16, 128)

        # Préparation pour le Transformer
        batch_size, num_channels, height, width = x.shape
        # Aplatir les dimensions spatiales (16x128) pour chaque canal
        # Nouvelle forme : (batch_size, 64, 16 * 128) -> (batch_size, 64, 2048)
        x = x.view(batch_size, num_channels, height * width) 

        # Projeter chaque token (de 2048 dimensions) à d_model
        # Nouvelle forme : (batch_size, 64, d_model)
        x = self.proj_to_d_model(x)

        # Ajouter l'encodage positionnel à la séquence de tokens
        x = self.positional_encoding(x)

        # Passe avant de l'encodeur Transformer
        # La forme reste (batch_size, 64, d_model)
        x = self.transformer_encoder(x) 

        # Aplatir la sortie du Transformer pour la couche linéaire finale
        # Nouvelle forme : (batch_size, 64 * d_model)
        x = x.view(batch_size, -1) 

        # Couche linéaire finale
        # Nouvelle forme : (batch_size, 800)
        x = self.fc_out(x) 

        return x






# Classe pour le Dataset personnalisé
class CaTS_Dataset(Dataset):
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