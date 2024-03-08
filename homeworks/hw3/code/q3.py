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


class Patchify(nn.Module):
    """
    Splits the input images into 8x8 patches.
    Assumes input of shape (N, C, H, W).
    """
    def __init__(self, patch_size=8):
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, channels, self.patch_size, self.patch_size)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_filters=128, patch_size = 8):
        super(Discriminator, self).__init__()
        self.patchify = Patchify(patch_size=patch_size)
        self.block1 = ResnetBlockDown(3, n_filters)
        self.block2 = ResnetBlockDown(n_filters, n_filters)
        self.block3 = ResBlock(n_filters, n_filters)
        self.block4 = ResBlock(n_filters, n_filters)
        self.relu = nn.ReLU()
        self.global_sum_pooling = nn.AdaptiveAvgPool2d(1)  # Emulates global sum pooling
        self.final_linear = nn.Linear(n_filters, 1)

    def forward(self, x):
        #split to 8*8 patches
        x = self.patchify(x)
        
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
            perceptual_loss = LPIPS_lossfunc(x_tilde, real_data).mean()

            # Update generator
            g_loss = diff + 0.5 * perceptual_loss + 0.1 * gan_loss + recon_loss
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

def q3a(train_data, val_data, reconstruct_data):
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
    hyperparams = {'lr': 1e-4, 'num_epochs': 20}
    
    vqvae = VectorQuantizedVAE(code_size=1024, code_dim=256)
    discriminator = Discriminator()
   
    train_tensor = torch.tensor(train_data, dtype = torch.float32)
    valid_tensor = torch.tensor(val_data, dtype = torch.float32)
    recon_tensor = torch.tensor(reconstruct_data, dtype = torch.float32)
    
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_tensor), batch_size=128, shuffle=False)
    

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
    
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_lr)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_lr)


    
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
        debug_mode=True,
    )
    
    
    vqvae.eval()
    recon_tensor = recon_tensor.to(device)
    reconstrcutions, _ = vqvae.forward(recon_tensor)
    reconstrcutions = reconstrcutions.permute(0, 2, 3, 1).detach().cpu().numpy()  # Change to (N,H,W,C) for numpy
    reconstrcutions = (reconstrcutions * 0.5) + 0.5  # Scal

    #Fréchet inception distance (bonus, 5pts)
    
    return discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test, reconstrcutions
    

if __name__ == "__main__":
   q3_save_results(q3a, "a") # with pips