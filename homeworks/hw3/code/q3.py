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

from models import *

def generator_loss(discriminator_output, dummy):
    return -discriminator_output.mean()
def discriminator_loss_fn(discriminator_output_real, discriminator_output_fake):
    return -discriminator_output_real.mean() + discriminator_output_fake.mean()
def train_gan_q2(generator, discriminator, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, dataloader, device, epochs=100, debug_mode=False, wgan_gp = False, n_critic = 5, g_scheduler = None, d_scheduler = None):
    """
    Train a GAN consisting of a generator and discriminator with an option for a quick debug iteration.

    Args:
    - generator: The generator model.
    - discriminator: The discriminator model.
    - g_optimizer: Optimizer for the generator.
    - d_optimizer: Optimizer for the discriminator.
    - g_loss_fn: Loss function for the generator.
    - d_loss_fn: Loss function for the discriminator.
    - dataloader: DataLoader for the real data.
    - device: The device to train on ('cuda' or 'cpu').
    - epochs: Number of epochs to train for.
    - debug_mode: If True, runs a single epoch with a limited number of batches for debugging.

    Returns:
    - A tuple of (generator_losses, discriminator_losses) capturing the loss history.
    """

    generator_losses = []
    discriminator_losses = []
    gp_losses = []

    debug_epochs = 1 if debug_mode else epochs
    debug_batches = 10

    for epoch in tqdm(range(debug_epochs), desc="Epochs"):
        g_epoch_loss = 0
        d_epoch_loss = 0
        batch_count = 0

        for real_data  in dataloader:
            if debug_mode and batch_count >= debug_batches:
                break
            real_data = real_data[0].to(device)
    
            batch_size = real_data.shape[0]
            
            # -----------------
            #  Train Generator
            # -----------------
            g_optimizer.zero_grad()
            # z = torch.randn((batch_size,*generator.shape),device=device, dtype = torch.float32)  # generator.input_size needs to match your generator's input size
            fake_data = generator.sample(batch_size, device = device)
            output = discriminator(fake_data)
            g_loss = g_loss_fn(output, torch.ones_like(output, device=device))
            g_loss.backward()
            g_optimizer.step()
            if g_scheduler is not None:
                g_scheduler.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for i in range(n_critic):
                d_optimizer.zero_grad()
                real_output = discriminator(real_data)
                
                fake_output = discriminator(fake_data.detach())
                
                
                d_loss = d_loss_fn(real_output, fake_output)
                gp_loss = gradient_penalty(real_data, fake_data, discriminator)
                d_loss += gp_loss
               
                d_loss.backward()
                d_optimizer.step()
                if d_scheduler is not None:
                    d_scheduler.step()
                    
                discriminator_losses.append(d_loss.item())
                gp_losses.append(gp_loss.item())

            
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

            batch_count += 1

            generator_losses.append(g_epoch_loss)
          
        if debug_mode:
            print(f'Debug Mode: Epoch [{epoch+1}/{debug_epochs}], Generator Loss: {g_epoch_loss / batch_count}, Discriminator Loss: {d_epoch_loss / batch_count}')
    return generator_losses, discriminator_losses, gp_losses
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

    
    hyperparams = {
        "num_epochs":10
        
    }
    n_critic = 5
    generator = Generator_SNGAN(n_filters=128)
    discriminator = Discriminator(n_filters=128)

    train_tensor = torch.tensor(train_data, dtype = torch.float32)
    # train_tensor = train_tensor #nomralized to -1 1
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    #optimizer
    #Training optimizer
    total_steps = hyperparams["num_epochs"] * (len(train_loader)) * n_critic

    lambda_lr = lambda step: 1 - step / total_steps

    #2𝑒−4 ,  𝛽1=0 ,  𝛽2=0.9 ,  𝜆=10 ,
    d_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas=(0, 0.9))
    g_optimizer = optim.Adam(generator.parameters(), lr = 2e-4, betas=(0, 0.9))
    
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_lr)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_lr)

    g_loss_fn = generator_loss
    d_loss_fn = discriminator_loss_fn
    
    generator_losses, discriminator_loss, gradient_penalty = train_gan_q2(
        dataloader=train_loader,
        generator=generator,
        discriminator=discriminator,
        g_loss_fn=g_loss_fn,
        d_loss_fn=d_loss_fn,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
        # checkpoint_path=f"homeworks/hw3/results/q1a",
        epochs = hyperparams["num_epochs"],
        device=device,
        debug_mode=False,
        wgan_gp = True,
        n_critic = n_critic
    )
    
    samples = generator.sample(1000, device = device).permute(0, 2, 3, 1).cpu().detach().numpy()
    #unormallize
    samples = samples 
    #plot the losses curve
    # Creating a figure for subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

    # Plotting Generator Loss on the first subplot
    axes[0].plot(generator_losses, label="Generator Loss", color='blue')
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("training iter")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plotting Gradient Penalty on the second subplot
    axes[1].plot(gradient_penalty, label="Gradient Penalty", color='green')
    axes[1].set_title("Gradient Penalty")
    axes[1].set_xlabel("training iter")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("WGAN-GP_Losses_and_Gradient_Penalty.png")
    
    #Fréchet inception distance (bonus, 5pts)
    
    return discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test, reconstructions
    

if __name__ == "__main__":
   q3_save_results(q3a, "a") # with pips