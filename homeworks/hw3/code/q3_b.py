import torch.nn as nn
import deepul.pytorch_util as ptu
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from tqdm import trange, tqdm_notebook
from deepul.hw3_helper import *
import deepul.pytorch_util as ptu
import warnings
warnings.filterwarnings('ignore')

from train_utils import *
from torch.utils.data import DataLoader, TensorDataset

from deepul.hw3_utils.lpips import LPIPS
from vqvae import VectorQuantizedVAE
from models import *


from torchvision.models import vit_b_16, ViT_B_16_Weights

import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

#referebced from https://github.com/thuanz123/enhancing-transformers/blob/main/enhancing/modules/stage1/layers.py
from vit_layers import ViTEncoder, ViTDecoder
class ViTEncoder2(nn.Module):
    def __init__(self, image_size=32, patch_size=4, code_dim=256, num_layers=4, num_heads=8):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.code_dim = code_dim
        
        self.grid_size =  int( self.num_patches**0.5)
        

        self.patch_embedding = nn.Linear(patch_size*patch_size*3, code_dim)

        encoder_layers = TransformerEncoderLayer(d_model=code_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        # Assuming x is [batch_size, channels, height, width]
        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), x.size(2) * x.size(3), -1)  # [batch_size, num_patches, patch_size*patch_size*3]
        x = self.patch_embedding(x)  # [batch_size, num_patches, code_dim]

        x = self.transformer_encoder(x)  # [batch_size, num_patches, code_dim]
        x = x.permute(0, 2, 1).contiguous().view(x.size(0), -1, self.grid_size, self.grid_size)
        return x

class ViTDecoder2(nn.Module):
    def __init__(self, code_dim=256, num_patches=64, patch_size=4, num_layers=4, num_heads=8):
        super(ViTDecoder, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.code_dim = code_dim

        # Assuming the encoded feature dimension matches the transformer's input dimension (code_dim)
        decoder_layer = TransformerDecoderLayer(d_model=code_dim, nhead=num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Projection from transformer output to patch pixels, then reshape to image
        self.patch_projection = nn.Linear(code_dim, patch_size * patch_size * 3)

    def forward(self, encoded_patches):
        
        encoded_patches = encoded_patches.permute(0,2,3,1).contiguous().view(-1, self.num_patches, self.code_dim)
        # encoded_patches shape: [batch_size, num_patches, code_dim]
        # Apply transformer decoder
        decoded_patches = self.transformer_decoder(encoded_patches, encoded_patches)
        # Project back to pixel space and reshape
        batch_size = decoded_patches.shape[0]
        decoded_patches = self.patch_projection(decoded_patches)
        decoded_patches = decoded_patches.view(batch_size, self.num_patches, self.patch_size, self.patch_size, 3)

        # Reassemble patches into full images (This step requires careful implementation)
        # The reassembly can be a custom function similar to an inverse of the patchify operation
        reconstructed_images = self.rearrange_patches_to_image(decoded_patches, image_size=32, patch_size=4)

        return reconstructed_images.permute(0,3,1,2).contiguous()

    def rearrange_patches_to_image(self, patches, image_size, patch_size):
        """
        Rearrange patches back into full images.
        patches: [batch_size, num_patches, patch_size, patch_size, 3]
        """
        batch_size, num_patches, _, _, _ = patches.shape
        grid_size = int(image_size / patch_size)
        
        # Reshape to grid form
        patches = patches.view(batch_size, grid_size, grid_size, patch_size, patch_size, 3)
        
        # Permute to [batch_size, grid_size, patch_size, grid_size, patch_size, 3]
        patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Flatten the grid dimensions
        images = patches.view(batch_size, image_size, image_size, 3)
        
        return images

class Discriminator(nn.Module):
    def __init__(self, n_filters=128):
        super(Discriminator, self).__init__()
        # self.patchify = Patchify(patch_size=patch_size)
        self.block1 = ResnetBlockDown(3, n_filters)
        self.block2 = ResnetBlockDown(n_filters, n_filters)
        self.block3 = ResBlock(n_filters, n_filters)
        self.block4 = ResBlock(n_filters, n_filters)
        self.relu = nn.ReLU()
        self.global_sum_pooling = nn.AdaptiveAvgPool2d(1)  # Emulates global sum pooling
        self.final_linear = nn.Linear(n_filters, 1)

    def forward(self, x):
        #split to 8*8 patches
        # x = self.patchify(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.relu(x) #shape (batch_size, n_filters, 8, 8)
        x = self.global_sum_pooling(x) #shape (batch_size, n_filters, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        output = self.final_linear(x)
        
        return output  #scale the output of discirminator


def train_gan_q3(vqvae, discriminator, g_optimizer, d_optimizer, dataloader, val_loader, device, epochs=100, debug_mode=False, g_scheduler=None, d_scheduler=None):
    generator_losses = []
    discriminator_losses = []
    perceptual_losses = []
    l2_recon_losses = []
    l2_recon_val_losses = []

    LPIPS_lossfunc = LPIPS().to(device)  # Make sure LPIPS is moved to the correct device
    L1_loss = nn.L1Loss()

    debug_batches = 1
    epochs = 1 if debug_mode else epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        g_epoch_loss, d_epoch_loss, epoch_perceptual_loss, epoch_l2_recon_loss = 0, 0, 0, 0
        batch_count = 0

        for real_data in dataloader:
            if debug_mode and batch_count >= debug_batches:
                break
            real_data = real_data[0].to(device)
            batch_size = real_data.shape[0]

            # Assuming vqvae.forward() returns the reconstructed image and VQ loss difference
            x_tilde, diff = vqvae.forward(real_data)

            # Corrected use of discriminator for fake data
            disc_real = discriminator(real_data)
            disc_fake = discriminator(x_tilde.detach())

            # Compute losses
            gan_loss = -torch.mean(disc_fake) + torch.mean(disc_real)
            recon_loss = F.mse_loss(x_tilde, real_data)
            
            recon_L1_loss = torch.abs(x_tilde - real_data).mean()
            perceptual_loss = LPIPS_lossfunc(x_tilde, real_data).mean()

            # Update generator
            g_loss = diff + 0.5 * perceptual_loss + 0.1 * gan_loss + recon_loss + 0.1 * recon_L1_loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Update discriminator
            disc_real = discriminator(real_data)
            disc_fake = discriminator(x_tilde.detach())
            d_loss = -torch.mean(disc_real) + torch.mean(disc_fake)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Track losses
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
            perceptual_losses.append(perceptual_loss.item())
            l2_recon_losses.append(recon_loss.item())

            batch_count += 1

        # Compute L2 reconstruction loss on validation data
        val_l2_recon_loss = compute_l2_recon_loss(vqvae, val_loader, device)
        l2_recon_val_losses.append(val_l2_recon_loss)

        # Optional debug message

        print(f'Debug Mode: Epoch [{epoch+1}/{epochs}], Generator Loss: {np.mean(generator_losses)}, Discriminator Loss: {np.mean(discriminator_losses)}, Perceptual Loss: {np.mean(perceptual_losses)}, L2 Recon Loss: {np.mean(l2_recon_losses)}')

    return np.array(discriminator_losses), np.array(perceptual_losses), np.array(l2_recon_losses), np.array(l2_recon_val_losses)


def compute_l2_recon_loss(vqvae, val_loader, device):
    l2_recon_loss = []
    with torch.no_grad():
        for real_data in val_loader:
            real_data = real_data[0].to(device)
            x_tilde, _ = vqvae.forward(real_data)
            l2_loss = F.mse_loss(x_tilde, real_data)
            l2_recon_loss.append(l2_loss.item())
    return np.mean(l2_recon_loss)

def q3b(train_data, val_data, reconstruct_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]
    val_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]
    reconstruct_data: An (100, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]. To be used for reconstruction

    Returns
    - a (# of training iterations,) numpy array of the discriminator train losses evaluated every minibatch
    - None or a (# of training iterations,) numpy array of the perceptual train losses evaluated every minibatch
    - a (# of training iterations,) numpy array of the l2 reconstruction evaluated every minibatch
    - a (# of epochs + 1,) numpy array of l2 reconstruction loss evaluated once at initialization and after each epoch on the val_data
    - a (100, 32, 32, 3) numpy array of reconstructions from your model in [0, 1] on the reconstruct_data.  
    """

    """ YOUR CODE HERE """

    hyperparams = {'lr': 1e-4, 'num_epochs': 50}
    
    vqvae = VectorQuantizedVAE(code_size=1024, code_dim=256)
    # add ViT modules for encoder decoder
    vqvae.encoder = ViTEncoder(image_size=32, patch_size=4,  heads = 8, depth = 4, dim= 256, mlp_dim = 256)
    vqvae.decoder = ViTDecoder(image_size=32, patch_size=4, dim=256, depth = 4, heads = 8,mlp_dim = 256)    
    discriminator = Discriminator()
   
    train_tensor = torch.tensor(train_data, dtype = torch.float32)
    valid_tensor = torch.tensor(val_data, dtype = torch.float32)
    recon_tensor = torch.tensor(reconstruct_data, dtype = torch.float32)
    
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_tensor), batch_size=256, shuffle=False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    vqvae = vqvae.to(device)
    discriminator = discriminator.to(device)

    #optimizer
    #Training optimizer
    total_steps = hyperparams["num_epochs"] * (len(train_loader))

    lambda_lr = lambda step: 1 - step / total_steps

    d_optimizer = optim.Adam(discriminator.parameters(), lr = hyperparams["lr"], betas=(0.5, 0.9))
    g_optimizer = optim.Adam(vqvae.parameters(), lr = hyperparams["lr"], betas=(0.5, 0.9))
    
    # d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_lr)
    # g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_lr)


    
    discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test = train_gan_q3(
        dataloader=train_loader,
        val_loader = valid_loader,
        vqvae=vqvae,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        # g_scheduler=g_scheduler,
        # d_scheduler=d_scheduler,
        # checkpoint_path=f"homeworks/hw3/results/q1a",
        epochs = hyperparams["num_epochs"],
        device=device,
        debug_mode=False,
    )
    
    
    vqvae.eval()
    recon_tensor = recon_tensor.to(device)
    reconstrcutions, _ = vqvae.forward(recon_tensor)
    reconstrcutions = reconstrcutions.permute(0, 2, 3, 1).detach().cpu().numpy()  # Change to (N,H,W,C) for numpy

    
    return discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test, reconstrcutions
    

if __name__ == "__main__":
   q3_save_results(q3b, "b") # with pips