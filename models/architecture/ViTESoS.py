import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import torch.nn as nn
from einops import rearrange
# import math
import coloralf as c





class PatchEmbedding(nn.Module):

    """
    Convertit une image en une séquence d'embeddings de patches.
    """

    def __init__(self, in_channels=1, patch_size=(16, 16), emb_size=768, img_size=(128, 1024)):

        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        # La conv divise l'image en patches, puis on les projette dans l'espace d'embedding.
        # Flatten à partir de la 2ème dimension pour obtenir (B, Emb_Size, Num_Patches)
        self.projection = nn.Sequential(nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), nn.Flatten(2))

        # Calcul du nombre de patches en hauteur et en largeur
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (B, C, H, W)
        x = self.projection(x) # -> (B, Emb_Size, Num_Patches)
        x = x.transpose(1, 2)  # -> (B, Num_Patches, Emb_Size)
        return x





class MultiHeadSelfAttention(nn.Module):

    """
    Implémentation de l'attention multi-tête auto-attention.
    """

    def __init__(self, emb_size=768, num_heads=8, dropout=0.0):

        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # Chaque tête gère une partie de la dimension d'embedding
        self.head_dim = emb_size // num_heads
        if self.head_dim * num_heads != emb_size : raise ValueError(f"emb_size ({emb_size}) doit être divisible par num_heads ({num_heads})")

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size) # Couche de sortie après concaténation des têtes
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim) # Pour scaler les scores d'attention


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Calcul des Q, K, V pour toutes les têtes -> (B, S, H, D_head)
        keys = self.keys(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = self.queries(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.values(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transposition pour que les têtes soient la première dimension pour le calcul -> (B, H, S, D_head)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Score d'attention --- (B, H, S, D_head) @ (B, H, D_head, S) -> (B, H, S, S)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Appliquer l'attention aux valeurs et concaténer les sorties des têtes --- (B, H, S, S) @ (B, H, S, D_head) -> (B, H, S, D_head)
        out = torch.matmul(attention, values)
        # Réarranger pour regrouper les dimensions des têtes et des séquences --- -> (B, S, H, D_head) -> (B, S, Emb_Size)
        out = rearrange(out, 'b h s d -> b s (h d)')

        out = self.fc_out(out)

        return out





class FeedForward(nn.Module):

    """
    MLP dans bloc Transformer.
    """

    def __init__(self, emb_size, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_size, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, emb_size), nn.Dropout(dropout))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)






class TransformerBlock(nn.Module):

    """
    Un bloc Transformer composé d'attention multi-tête et d'un MLP.
    """

    def __init__(self, emb_size: int, num_heads: int, n_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size) # Normalisation de couche avant l'attention
        self.attn = MultiHeadSelfAttention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size) # Normalisation de couche avant le MLP
        self.ff = FeedForward(emb_size, n_hidden, dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Connexions résiduelles et normalisation
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x





class VisionTransformerEncoder(nn.Module):

    """
    Encodeur du Vision Transformer.
    """

    def __init__(self, img_size=(128, 1024), in_channels=1, patch_size=(16, 16), emb_size=768, num_heads=8, n_hidden=3072, num_layers=12, dropout=0.0):

        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        
        # Positionnal embedding pour donner un sens de l'ordre aux patches
        # Ajout d'un terme pour le CLS token si utilisé, mais pas nécessaire pour le débruitage
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(emb_size, num_heads, n_hidden, dropout) for _ in range(num_layers)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embedding(x) # -> (B, Num_Patches, Emb_Size)

        # Ajout de l'embedding de position
        x = x + self.positional_embedding
        x = self.dropout(x)

        # Passage à travers les blocs Transformer
        for block in self.transformer_blocks:
            x = block(x)
        return x # Sortie de l'encodeur: (B, Num_Patches, Emb_Size)





class ViTDecoder(nn.Module):

    """
    Décodeur pour reconstruire l'image à partir des embeddings de l'encodeur.
    Utilise des couches de convolution transposée pour remonter en résolution.
    """

    def __init__(self, emb_size=768, patch_size=(16, 16), img_size=(128, 1024), out_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]

        self.deconv = nn.Sequential(
            # De (emb_size, 8, 64) après reshape
            nn.ConvTranspose2d(emb_size, 512, kernel_size=2, stride=2), # -> (512, 16, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # -> (256, 32, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # -> (128, 64, 512)
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2), # -> (out_channels, 128, 1024)
            # Pas de fonction d'activation finale comme Sigmoid/Tanh car j'ai pas normaliser bref je pense que c'est ok 
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (B, Num_Patches, Emb_Size)
        batch_size, num_patches, emb_size = x.shape

        # Remake embeddings dans la shape "image" (B, Emb_Size, H_patches, W_patches)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.num_patches_h, w=self.num_patches_w)

        # découv
        x = self.deconv(x) # -> (B, Out_Channels, Img_H, Img_W)

        return x





class ViTESoS_Model(nn.Module):

    """
    Modèle de Visual Transformers Spectro -> Spectrum
    ViTESoS : Visual Transformers for Exposed Spectrum from Spectro
    """

    folder_input = "image"
    folder_output = "spectro"

    # emb_size   = 768
    # num_heads  = 8
    # n_hidden   = 3072
    # num_layers = 6

    def __init__(self, img_size=(128, 1024), in_channels=1, out_channels=1, patch_size=(16, 16), emb_size=384, num_heads=6, n_hidden=1536, num_layers=4, dropout=0.1):


        if "vitesostestparams" in sys.argv:
            print(f"{c.ly}Using ViTESoS test params ...{c.d}")
            emb_size=256
            num_heads=4
            n_hidden=512
            num_layers=2 


        super().__init__()

        self.encoder = VisionTransformerEncoder(img_size, in_channels, patch_size, emb_size, num_heads, n_hidden, num_layers, dropout)
        self.decoder = ViTDecoder(emb_size, patch_size, img_size, out_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_features = self.encoder(x)              # Encoder: (B, C, H, W) -> (B, Num_Patches, Emb_Size)
        denoised_image = self.decoder(encoded_features) # Decoder: (B, Num_Patches, Emb_Size) -> (B, Out_C, H, W)
        return denoised_image



    def extraApply(self, pred, pathsave, spectro_name):

        suffixe = spectro_name.split(self.folder_output)[-1]
        spectrumPX = np.sum(pred[0], axis=0)

        np.save(f"{pathsave}/spectrumPX{suffixe}", spectrumPX)






class ViTESoS_Dataset(Dataset):

    def __init__(self, image_dir, spectro_dir):
        self.image_dir = image_dir
        self.spectro_dir = spectro_dir
        self.input_files = sorted(os.listdir(image_dir))
        self.spectrum_files = sorted(os.listdir(spectro_dir))

        if len(self.input_files) != len(self.spectrum_files):
            raise ValueError("Le nombre de fichiers d'entrée et de sortie doit être le même.")

        for i in range(len(self.input_files)):
            if self.input_files[i].split("_")[-1] != self.spectrum_files[i].split("_")[-1]:
                print(f"Attention: Les noms de fichiers ne correspondent pas pour l'index {i}: {self.input_files[i]} vs {self.spectrum_files[i]}")


    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):

        input_filename = self.input_files[idx]
        output_filename = self.spectrum_files[idx]

        input_filepath = os.path.join(self.image_dir, input_filename)
        output_filepath = os.path.join(self.spectro_dir, output_filename)

        input_image_np = np.load(input_filepath)
        output_image_np = np.load(output_filepath)

        input_tensor = torch.from_numpy(input_image_np).float().unsqueeze(0)
        output_tensor = torch.from_numpy(output_image_np).float().unsqueeze(0)

        return input_tensor, output_tensor




if __name__ == "__main__":

    

    sys.path.append("./")
    datafold = sys.argv[1]

    folder_input = "image"
    folder_output = "spectro"

    BATCH_SIZE = 4

    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {DEVICE}")

    DATA_DIR = f"./results/output_simu/{datafold}"
    NOISY_DIR = os.path.join(DATA_DIR, folder_input)
    CLEAN_DIR = os.path.join(DATA_DIR, folder_output)

    dataset = VitESoS_Dataset(image_dir=f"./results/output_simu/{datafold}/{folder_input}", spectro_dir=f"./results/output_simu/{datafold}/{folder_output}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Size of loaded dataset : {len(dataset)}")

    model = ViTESoS_Model().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Nb params : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print("\nDébut de l'entraînement...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        from tqdm import tqdm
        for batch_idx, (noisy_images, clean_images) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            noisy_images = noisy_images.to(DEVICE)
            clean_images = clean_images.to(DEVICE)

            # Zéro gradiente
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy_images)

            # Calcul de la perte
            loss = criterion(outputs, clean_images)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    print("\nEntraînement finiiii.")

    